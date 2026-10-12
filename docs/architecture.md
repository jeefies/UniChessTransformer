# UniChess Transformer 训练框架架构

> 本文只描述**当前实现**。逐代实验数据与结论见 [`experiments.md`](experiments.md)，
> 运维要点与踩坑见 [`loop.md`](loop.md)，仓库纪律见 [`../AGENTS.md`](../AGENTS.md)。

## 1. 总览

训练框架分三层，依赖方向单向：

```
┌─ 调度层 ────────────────────────────────────────────────┐
│  Kit/pipelines/loop.py   换代循环（selfplay→train→arena） │
│  tmp/watchdog.sh         看门狗（loop 死了就重启）        │
└──────────────────────┬──────────────────────────────────┘
                       │ 每阶段拉起子进程
┌─ 管线层（Kit/）──────┴──────────────────────────────────┐
│  selfplay  Gumbel 自对弈 → .sp.bin 分片 + games.jsonl    │
│  train     planes19 混合数据 → ChessLoss → best_model.pt │
│  match     批量对弈 + 五项式统计 + SPRT 判决              │
│  search    GumbelCpp（循环用）/ PUCTCpp（生产用）         │
└──────────────────────┬──────────────────────────────────┘
                       │ EngineSpec + PlayerFactory
┌─ 引擎层（Transformer/）┴────────────────────────────────┐
│  model.py       stratified_20m（三专家）+ 四个头          │
│  evaluator.py   批量前向（evaluate_planes）              │
│  kit.py         工厂：make_gumbel_player_factory 等       │
│  engine.py      Server 六方法插件                        │
└─────────────────────────────────────────────────────────┘
```

**数据流**：champion 权重 → Gumbel 自对弈 4096 局 → 分片落盘 → 混合 ResNet 监督数据
→ 训练出候选 → arena 宽判决 vs champion → H1 则候选成为新 champion，H0 则守住。
换代的唯一依据是 arena 的 SPRT 结论。

import 根是 `~/UniChess`（六个仓库的共同父目录，仓库根目录即包）。

---

## 2. 模型

### 2.1 规格

| 预设 | 层数 | d_model | 头 | d_ff | 参数量 | 用途 |
|---|---|---|---|---|---|---|
| `transformer_20m` | 11 | 384 | 12 | 1024 | 20,343,688 | 单主干基线 |
| **`stratified_20m`** | **3×11** | **384** | **12** | **1024** | **61,031,064** | **在用规格** |

`stratified_20m` = 3 个 `transformer_20m` 专家 + 共享路由。state_dict 同时含
`opening./middlegame./endgame.` 与 `experts.0/1/2.` 两套键（同一模块的别名，
`self.experts = nn.ModuleList([opening, middlegame, endgame])`），加载器两种命名都认，
另容忍 `torch.compile` 的 `_orig_mod.` 前缀。

### 2.2 输入与主干

- **19 平面** `(19, 8, 8)` float32，按行棋方归一（黑方镜像）：0-5 己方六种子力、
  6-11 对方六种子力、12-15 双方王车易位权、16 过路兵目标格、17 半回合钟（/100）、
  18 重复次数（/2）
- **ConvStem**：3×3 卷积 stride 1 pad 1，19→d_model，展平得 64 个格子 token
- **token 与位置**：前置一个 `[CLS]` token（共 65）；每格加解耦的 rank/file 嵌入；
  注意力里加可学习的平方间相对位置偏置 `B_rel ∈ R^{H×65×65}`（编码文件、等级、
  对角与马步关系），直接喂 `F.scaled_dot_product_attention` 的 attn_mask
- **策略头**：双线性平方到平方——64 格出 origin query 与 destination key
  （$D_p=64$），$M_{from,to}=QK^T/\sqrt{D_p}+B_{move}$，摊平到 4096 着法空间；
  升变走独立头（ rank-8 表示 → 4 种升变子）
- **WDL 值头**：`[CLS]` 过 MLP 出 3 类 logits（胜/和/负），搜索用的标量评估
  $Q=P(\text{win})-P(\text{loss})\in[-1,1]$
- **MLH 头**：`[CLS]` 过两层 MLP 预测剩余步数，Smooth-L1、权重 0.05；每个专家都有。
  加载器只允许缺 `mlh_head.*` 一组键，其他缺键一律报错

### 2.3 三专家相位路由（固定规则，非学习门控）

| 相位 | 专家 | 条件 |
|---|---|---|
| 0 | opening | 子力数 ≥ 24 **或** ply ≤ 20 |
| 1 | middlegame | 12 < 子力数 < 24 |
| 2 | endgame | 子力数 ≤ 12 |

批量推理时按路由相位分组，每个专家看到连续批次。kit 的 C++ 搜索只交预编码平面时，
从平面 0-11 的子力数判相位。

### 2.4 精度

训练 `bf16`（与监督配方一致），引擎推理 `fp16`（与生产预设一致）——**两绝不混用同一批前向**。
注意 bf16/fp32 舍入会让极小的 weight decay 完全失效（衰减项不足一个 ULP）。

---

## 3. 自对弈（数据生成）

### 3.1 搜索：Gumbel 顺序减半

循环的自对弈与 arena 都走 `GumbelCpp`（C++ 实现，与 Python 参考整树逐位一致）：

| 参数 | 值 | 说明 |
|---|---|---|
| `simulations` | 1600 | 根节点总模拟预算 |
| `m0` | 16 | 根候选数上界 |
| `g` | 1.0 | 根 Gumbel 噪声尺度 |
| `c_visit` / `c_scale` | 50.0 / 0.1 | σ 的访问偏置与价值缩放 |
| `claim_draw` | True | 搜索内把可申和（三次重复/五十步）当终局 |
| `parallel` | True | 轮内各候选并发模拟 |
| `expand_width` | 1 | 每次前向展开的叶子数（1 = 每拍一个着法） |

- **顺序减半**：16 个候选逐轮砍半，预算均分、余数前置，总和恰为 1600；唯一候选时
  剩余预算全投
- **σ 混合**：$\sigma(\hat q)=(c_{visit}+\max_b N(b))\cdot c_{scale}\cdot\hat q$，
  $\hat q$ 用**本节点自身** completed Q 集合的量程归一
- **训练目标 = π′**（improved policy）：全部合法着上的
  $\text{softmax}(\ell+\sigma(\text{completedQ}))$，经 `MoveDecision.info["visits"]`
  以 `(uci, prob)` 对交给 sink。**不能拿根的 N 当目标**——Gumbel 下只有 m0 个候选
  有访问，比 PUCT 的访问分布更稀疏
- **着法选择**：顺序减半的最终幸存者；`temperature>0` 时改为从 π′ 采样
  （自对弈 `temperature=1.0`，arena/生产 `0` = 确定性）
- **重复规避**：`avoid_repetition` 按根 N 降序取第一个不造成重复的着法（同一 Player
  执双方时必须，否则镜像进三次重复）

### 3.2 对局结构

- 开局库 `Kit/data/openings_sp.txt`：2000 条合成线，第 g 局用第 `g % 2000` 条，
  裁到 `book_plies=6`，**不去重**（裁切后重复的线各自保留序号）
- book ply 由 Player 照 book 走（不搜索、不进训练目标）
- `StandardReferee`，`max_plies=400` 按整盘计（含 book）；超长记 `truncated`
  （按和棋计分但单独统计）
- 每局随机数流 = `SeedSequence(全局 seed, spawn_key=(局号,))`，只取决于局号

### 3.3 并行与落盘

- **4 个 worker 进程 × 每进程 32 个在途协程** = 128 局并发；`OMP_NUM_THREADS=4`
- **跨局攒批**：batcher 把所有在途局的叶子请求按 `model_key` 拼成一次前向
  （实测每批约 27 个请求 / ~101 个叶子）
- **MPS 守护**：4 个 worker 的 kernel 共卡并发（`nvidia-cuda-mps-control -d`，
  loop 启动 env 带 `CUDA_MPS_PIPE_DIRECTORY/CUDA_MPS_LOG_DIRECTORY`）
- **sink**：`SelfPlayShardSink` 逐局追加写 `.sp.bin` 分片（每 worker 一个，
  160 字节/条）与 `.games.jsonl`（每局一行元数据），**每条记录落盘后 fsync**
- **续跑**：分片按全局局号区间切分（`first_game = g × games`），重启后
  `done_games()` 从 jsonl 重建已完成局号集合并跳过——**被杀掉也不丢已完成的局**
- 实测聚合速率 ~8 局/分钟（外部租户不争用时）

---

## 4. 训练

### 4.1 数据混合

`{"kind": "mix", "batch_size": 256}`，两个来源按权重采样：

| 来源 | 权重 | 内容 | q_ratio |
|---|---|---|---|
| ResNet 监督分片 | 0.3 | `ResNet/data/shards_evals/evals_*.bin`（Stockfish 蒸馏，slice [0,4]） | 0.0 |
| 自对弈分片 | 0.7 | 最近 `window=2` 代（含本代）的全部分片（`{selfplay_files}` 占位符） | 0.0 |

- **价值目标只有终局 z**：自对弈记录里存了根搜索 Q 和 WDL 估计，但 `q_ratio=0`
  表示不混入 MCTS 价值，WDL 头只学终局结果
- 监督记录 96 字节/条，自对弈记录 160 字节/条（棋盘字段 70B + top-16 着法索引与占比
  + z + flags + q + wdl + game/ply）

### 4.2 损失（`t_chess`）

```
policy_weight 1.0   策略 KL（softmax 目标，kl_divergence 口径）
promo_weight  0.1   升变交叉熵（有效子集）
wdl_weight    1.0   WDL 软交叉熵
anchor_weight 2.0   锚定正则 KL(π_ref ‖ π_θ)，ref = base_ckpt（当前 champion）
mlh_weight    0.05  剩余步数 Smooth-L1
```

锚定项把微调钉在 champion 策略附近（`anchor_kl`，ref 当常量，两路都先转 fp32）。
策略与锚定都先转 fp32——bf16 下 `log` 会把小概率压成 0。

### 4.3 优化与导出

| 项 | 值 |
|---|---|
| 优化器 | AdamW（betas [0.9, 0.999]） |
| 基座 lr / wd | 1e-4 / 1e-4，onecycle，`pct_start=0.25` |
| **锁定变体**（gen 3 起生效） | **lr 5e-4 / wd 1e-4**，onecycle |
| steps × accum × batch | 400 × 8 × 256（有效批 2048，约 82 万样本/代） |
| 精度 | bf16 |
| 导出 | `best_model.pt`（按训练损失选）+ `final.pt` |

早期代枚举过 5-10 个 lr×wd 变体（筛选赛选优），gen 3 起锁定
`1c_5e-4_wd1e-4`，之后每代直接按锁定配置单路径训练。注意：数据窗口约 126 万自对弈位置，
而每代只看 82 万样本——**模型每代对自对弈数据走不到一个 epoch**，加局数前应先加 steps。

---

## 5. 判决（arena）

### 5.1 对局配置

| 参数 | 值 |
|---|---|
| A / B | 本代候选 `gen_XXXX/train/best_model.pt` vs 当前 champion |
| `simulations` | 2400（Gumbel，与生产预设同强度） |
| `workers` × `concurrency` | 4 × 64 = **256 局在途** |
| `pairs` | 350（700 局上限） |
| `max_plies` / 开局 | 400 / 同一开局库，种子按代派生（`seed_base + which·7919 + g·104729`） |
| SPRT | `elo0=0, elo1=25, alpha=0.05, beta=0.1, min_pairs=100` |

### 5.2 统计与判决语义

- **五项式（pentanomial）统计**：按配对开局汇总双方各胜/和/负，输出 Elo 与 CI95、LOS、
  重复率 `duplicate_rate`
- **SPRT 序贯检验**：LLR 越过 `[ln(β/α), ln((1-β)/α)] = [-2.251, 2.890]` 即判决；
  `min_pairs=100` 防止 200 局前早停
- **判决在越界那一刻冻结**：之后补写的在途局（停止信号传播后最多 256 局在途要下完）
  与从结果文件续跑的重放都**不许推翻它**。终值 Elo / 中间过程读数会漂
  （实测漂移可达 26 Elo），**只有冻结的 verdict 可作数**
- **晋级门**：`gate.kind="sprt"`，H1 才把 candidate 写进 `loop_state.json` 的 champion；
  H0 或打满未判都守住

### 5.3 吞吐（pro 实测）

| 配置 | 局/秒 | 说明 |
|---|---|---|
| 4×8（旧） | 0.025-0.031 | 每批 ~15 叶子，GPU 固定开销主导 |
| **4×64（当前）** | **0.058**（2400 sims） | 每批 ~100 叶子，2.3× |

规律：**吞吐 ≈ 总在途局数 ÷ 单局时长**，与批次大小基本无关（前向是固定开销主导，
CUDA graph 加速前向对端到端无益）；同总数下 worker 少 conc 高占优（4×32 比 8×16 快 1.49×，
4×64 比 8×32 快 1.50×）。**显存与并发无关**（树在 CPU 侧，4 worker 恒定 ~6.5GB）。

### 5.4 和棋归因（draw_attribution）

汇总时自动分析和棋对局中双方的最大领先子力与终局子力，拆成：
`a_blundered_draws`（A 优势断送和）、`b_blundered_draws`、`equal_draws`。
这是当前唯一能量化“优势转化能力”的指标。

---

## 6. 换代循环

### 6.1 状态机

```
gen g:  selfplay(4096 局) → train(锁定变体) → arena(宽判决) → [H1] champion=candidate
                                                                  [H0] champion 不变
        g+1 …
```

`enumerate_generations=3`：只有前 3 代枚举 lr×wd 变体（打筛选赛），之后按
`loop_state.json` 的 `train_variant` 单路径训练。当前 `generations` 额度由配置给
（pro 上跑单代时设 17；长期计划 32）。

### 6.2 状态与续跑

- `loop_state.json`：`{generation, phase, champion, history[], train_variant}`，
  每阶段结束即写
- **各阶段都可安全续跑**：selfplay 靠 sink 的 `done_games()` 跳已完成局；
  train 靠 `latest.pt` 的 step 与配置哈希判断续训；arena 靠结果文件的配置哈希续跑
  （哈希不同直接报错，不吞）
- 每代种子由 `seed_base` 确定性派生（train/selfplay/arena.match 三处），
  中断续跑同一代算出同一套种子
- 启动时取 `out/loop.lock` 的 flock（进程退出自动释放，残留锁文件不挡重启）
- **配置只在启动时读一次**：改 loop 配置必须重启；改 Kit 代码不用（子进程下次拉起即生效）

### 6.3 看门狗

`~/UniChess/tmp/watchdog.sh`（setsid 脱离）：每 2 分钟检查 loop 是否在跑，不在就重启
（顺带拉起 MPS），代数达到配置的 `generations` 就退出。防共享机上被无声击杀。

---

## 7. 生产服务

### 7.1 接入方式

Server 通过软链 `~/UniChess/Server/models/T -> ~/UniChess/Transformer` 发现本仓，
`engine.py` 实现六方法 `GameEngine` 契约（终局判定、白方视角评估、悔棋重放），
工厂是 `Transformer.kit:make_player_factory`（走 **PUCT** 搜索，与循环的 Gumbel 路径不同）。
批量对弈与观战直接建 `EngineSpec` 走 registry（有 SPI 版本检查）。
`Transformer.evaluator.TransformerEngine` 本身只是批量前向：载权重、编码平面、
返回 softmax 后的 `(policy, promo, wdl)`。

### 7.2 预设（`config.json`）

| 预设 | ckpt | sims | batch | 温度 | top_k | 角色 |
|---|---|---|---|---|---|---|
| `max_mcts` | **gen_0014** | 2400 | 64 | 0（确定性） | 0 | 生产默认路线 |
| `max_t` | **gen_0014** | 2400 | 64 | 1.0 | 3 | 同模型同搜索，采样多样性 + 质量护栏 |

- 预设里的 `mcts_sims`/`mcts_batch` 映射到 kit 的 `simulations`/`batch_size`
- 温度语义：$p_i \propto N_i^{1/t}$（对根**访问数**幂变形，非原始策略）；
  `t<=0` 贪婪、`t=1` 正比访问、`t<1` 锐化、`t>1` 被 kit 钳回 1.0；
  `root_top_k` 把采样限制在访问数前 K 的着法上
- UI 取 `list_presets()` 排序第一个作为默认（`max_mcts` < `max_t`，
  **任何排序在 max_mcts 之前的新预设都会 silently 变成生产路线**）
- **生产权重只能人工切换**：改预设 ckpt + 重启 `unichess-server`，loop 不许碰

---

## 8. 反和棋包（已实现，未接入循环）

Kit 的 `GumbelConfig` 已支持以下旋钮（**默认全部关闭，开启后与基线不对拍**）：

| 旋钮 | 语义 |
|---|---|
| `c_scale_schedule` | 动态 c_scale：按盘面子力与步数双轮驱动，开局 ~0.015 → 残局 ~0.15（基线是常数 0.1） |
| `stalemate_penalty` | 优势方（≥1 兵）逼和对方的 Q 惩罚，1.0 = 判负 |
| `insufficient_penalty` | 优势方（≥2 兵）兑子成和的 Q 惩罚，1.0 = 判负 |
| `twofold_penalty` | 优势方走入二次重复的 Q 惩罚，1.0 = 判负 |
| `contempt` | 优势方对三次重复的惩罚 |

**当前状态**：这些只实现在 Python 参考搜索里；循环走的是 `GumbelCpp`，
`gumbel_cpp.py` / `puct_native.cpp` 均未实现，`make_gumbel_player_factory` 也没有
参数通道——**往 loop 配置里写这些键会直接 TypeError**。要用起来需先做 C++ 移植 +
工厂透传 + parity 测试。动机：三代宽判决实测候选方“优势断送”次数稳定为对手的 5-11 倍
（下风守和强、领先握不死），同配方堆数据治不了。

---

## 9. 运行环境

### 9.1 设备路由

| 设备 | 用途 |
|---|---|
| **pro**（`ssh -p 26657 fwj@connect.westd.seetacloud.com`，RTX PRO 6000 96GB，cgroup 22 核） | 换代循环、GPU 训练/对弈 |
| **70Ti**（`jeefy@ulab.jeefy.top`，域名优先） | 生产引擎 + 备份（权重训完 scp 过去） |
| Windows 本机 | 代码编辑、文档、纯 Python 单测 |

### 9.2 pro 的关键约束

- cgroup 只有 **22 核**（`nproc` 报的 208 是宿主）→ 训练/自对弈**必须
  `OMP_NUM_THREADS=4`**（漏了 worker 烧 ~500% CPU 且初始化拖几分钟）
- `/tmp` 不可写 → `TMPDIR=~/UniChess/tmp`；临时脚本一律放 `~/UniChess/tmp`
- 工作树是路径补丁过的“脏”树 → **禁止 git pull**，代码更新只能 scp 具体文件
- **起 loop 前必须 `cd /root/autodl-tmp/fwj/UniChess`**（否则 `-m Kit` 找不到包）
- **与外部租户共卡**：`nvidia-smi` 看不到别的容器的进程，但会抢算力。
  实测拖慢可达 1.9×——判断争用只能靠“同配置重跑”对冲，不能信 nvidia-smi
- 批量击杀确实会发生（连同 MPS 守护和外部任务一起，无 traceback、无 OOM 记录），
  所以有看门狗

### 9.3 配置与机器绑定

`configs/loop_p4_v2.json` **不是可移植配置**：① 路径全是 pro 绝对路径；
② `arena.match` 的 workers/concurrency 是按 pro 的硬件（96GB 显存、22 核、
上述争用模式）实测标定的——换机器必须重扫，不可照抄。
SPRT 参数是统计量，与机器无关。

---

## 10. 不变式（改动前先看对应测试）

1. **C++/Python 搜索逐位一致**：`GumbelCpp` 对 Python 参考、`PUCTCpp` 对 `PUCT`
   都要求整树一致（`test_gumbel_cpp` / `test_puct_cpp`）。改一侧必须同步另一侧，
   C++ 编译失败必须报错、不得静默回退
2. **换代的唯一依据**是 arena 冻结的 SPRT verdict；`score_a` 只用于筛选变体
3. **生产权重只能人工切换**，loop 不许碰 `config.json` 预设
4. **模拟撞终局节点也计入预算**；**树复用前核对 EPD**；**装死判定统一 `claim_draw=True`**
5. **续跑拒绝配置哈希不同的结果文件**；中间行损坏报错，不吞尾半行
6. **配置哈希只用 `EngineSpec.identity()`**（不含 runtime）；会改结构的参数必须进 `kwargs`
7. 出错整批停止，不留“运行中”假象；worker 静默退出算错误
8. 确定性双方逐局重复必须在汇总里暴露（`duplicate_rate`）
9. 训练 `bf16` / 引擎 `fp16` 绝不混用同一批前向
