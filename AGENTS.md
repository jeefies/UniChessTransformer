# AGENTS.md — UniChess Transformer

> 面向 AI 编码 agent。最后更新：2026-09-25（扁平化重构后）。

## 仓库形态

**仓库根目录即包**：`import Transformer`，import 根是 `~/UniChess`（五个仓库的共同父目录）。
仓库里只有这些能改的文件：

| 文件 | 职责 |
|---|---|
| `model.py` | 结构 + 三专家路由。`state_dict` 的 `experts.*` / `opening.*` 前缀键名与重构前**逐字节兼容** |
| `evaluator.py` | 特征前端（`evaluate_planes`，按 64 一批） |
| `kit.py` | kit 接入：`make_player_factory` / `make_evaluators` / `make_task` / `make_adapter` / `TTrainAdapter` |
| `engine.py` | Server 六方法插件（`KIT_FACTORY="Transformer.kit:make_player_factory"`） |
| `configs/{t20m,stratified_opening,stratified_middlegame,stratified_endgame,p3_mlh}.json` | 监督训练配方（复刻旧脚本口径） |
| `configs/loop_p4.json` | 换代循环初版（200 局 / 40 对，无枚举搜索） |
| `configs/loop_p4_v2.json` | 激进版（1024 局 / 256 对 / lr×步数枚举搜索） |
| `tests/test_r3.py` | 单测（17 项） |
| `docs/architecture.md` | 架构与模块说明 |
| `__init__.py` | 包声明 |

**已删除**（git 历史可查，勿 recreate）：`unichess_t/` 包（含自带的 C++ MCTS 与 Python MCTS）、
`train/` 五个脚本、`tools/`、`eval/`、`logs/`、`benchmark_transformer.py`、`uci.py`。

## 常用命令

```bash
cd ~/UniChess
python -m Kit train Transformer/configs/t20m.json       # curriculum 配方同理
python -m Kit loop  Transformer/configs/loop_p4_v2.json # 自对弈换代循环（见下）
python -m unittest Transformer.tests.test_r3            # 17 项
python -m Kit match <config.json> --out runs/<name>/results.jsonl
```

## 搜索不在本仓库

所有对局都走 kit 的 `PUCTCpp` / `PUCT`（`Kit/search/`），后者与 Python 实现整树逐位一致；
Syzygy 桌库在 `Kit/rules/tablebase.py`，开局库在 `Kit/rules/openings.py`。

## 换代循环（自对弈 RL）

```bash
cd ~/UniChess
python -m Kit loop Transformer/configs/loop_p4_v2.json   # 当前在跑的（激进版）
python -m Kit loop Transformer/configs/loop_p4.json      # 初版（200 局 / 40 对 / 无枚举）
```

- **两个配方，训练口径相同（P4），差别在规模**：`loop_p4.json` 每代自对弈 200 局 800 sims
  → 70% 监督（前 4 片）+ 30% 自对弈混合训练（lr 5e-6、accum 4、KL、1000 步、bf16）
  → 候选对冠军 40 对 2400 sims，SPRT 判 H1 才换代。
  `loop_p4_v2.json` 是**激进版**：自对弈 **4096 局**、arena **256 对** 且 `elo1=60`，
  并且每代自对弈完成后做一次 **lr × wd 二维枚举搜索**（见 `Kit/pipelines/loop.py` 的
  `train.variants`）：8 个变体各自训练到 `gen_XXXX/train_<label>/`，再与冠军各打一场
  192 对筛选赛，取 `score_a` 最高者进最终 arena。
  **只枚举前 3 代**（`enumerate_generations: 3`）：第 4 代起按上一代选中变体的配置直接训练，
  不再打筛选赛，每代省约 7.4h。前三代的 `search.json` 就是"最优 lr×wd 是否稳定"的答案；
   若稳定就照现在这样，若漂移再改成每 3 代枚举一次。
- 初版已跑满 8 代，结论见 `docs/experiments.md` §15：gen 0 换代成功，之后七代全"判不出"，
  原因不是候选差而是 80 局分辨不出 +57~70 Elo；v2 就是按那些教训改的。
  当前冠军 = `runs/loop_p4/gen_0000/train/final.pt`（v2 的 `initial` 指向它）。
- **v2 的 gen 0 已跑完（2026-09-27），结论见 `docs/experiments.md` §17**：arena 101 局
  score_a 0.559 / Elo +41.5 / CI95 [−19.3, +104.9]，`stopped_by_sprt: true` 但记成
  `promoted: false`——**假阴性**，越界时 llr=+3.164 已判 H1，是在途对局把收尾 llr 拖回
  +1.471 所致（Kit `4925c62` 已修，gen 1 起不再复现）。**gen 0 是混合体**：十场筛选赛
  跑在旧网格（10 个恒定 lr 变体）+ bundled 开局库上，只有最终 arena 用新配方；gen 1 起
  才是完整的 8 变体 1cycle + 合成开局库。gen 1 自对弈从 09-27 21:35 起步。
  **gen 4 起的锁定配置**来自 `st["train_variant"]`，gen 0 设的是 `{lr 2e-5, steps 1200}`
  （旧恒定 lr 网格的胜者）；gen 1/2 会用新网格重选并覆盖，读实验时注意这个混合效应。
- **换代的唯一依据** 是 arena 的 SPRT 结论：判决 H1 才更新 `loop_state.json` 的 champion。
  生产权重（`config.json` 的 `max_mcts` / `max_t` 预设指向
  `runs/stratified_p4_selfplay_corrected/best_model.pt`）**只能由人工切换**：
  改 `config.json` 一次提交 + 重启 `unichess-server`，loop 不许碰它。
- **并发参数是实测枚举出来的（2026-09-26，5070 Ti），别再瞎调**：
  - 自对弈 `concurrency 32` / `batch_size 64`：batch_size 64/128/256 都是 8.8 s/局，
    concurrency 32/64 都是 8.5 s/局——GPU 已饱和，加并发或加批都不涨。
  - 训练 `batch_size 512` / `num_workers 4`：workers 4→8 只快 1%，而 batch 1024 直接 OOM
    （61M 三专家 + accum 4 的有效 batch 2048 已吃掉 12.5G）。1.76 步/s。
  - **arena / 筛选赛 `workers 2`**：比 workers=1 快约 10%（33→29.6 s/局），3/4 反而回落到
    30.2/30.8。concurrency 8/12/16 无差别。
    **`workers>1` 曾踩过一个假阴性坑**：停止信号发出后还有几局在途对局落盘，最终 llr 可能
    从越界值退回界内，把 `verdict` 重算成 `None`（gen 0 就因此丢了一次本该成功的换代，
    越界时 llr=+3.164、收尾 +1.471）。**Kit `4925c62` 已修**：判决改为在越界那一刻冻结
    （`_Run.stopped_verdict`），续跑同样冻结，`summary` 不再覆盖它。改 match 相关逻辑前
    先看 `Kit/tests/test_match.py::test_sprt_verdict_is_frozen_at_boundary`。
    **改 Kit 不必重启 loop**（`selfplay/train/arena/screen` 都是子进程，`loop.py:_run`），
    但**改 loop 配置必须重启**（进程只在启动时读一次配置）——gen 0 的筛选赛跑在旧配置上
    就是因为这个（详见 `docs/experiments.md` §17.4）。
  - 每代耗时构成（**2026-09-27 实测**，5070 Ti）：自对弈 4096 局 **9.6h**（8.4 s/局）+
    变体训练 8 个 × 1200 步 **1.5h** + 筛选赛 8 场（192 对 × 800 sims）**7.3h**（46–51 min/场）
    + 最终 arena **0.6h**（22.1 s/局，SPRT 通常 ~101 局即停；打满 512 局是 3.1h）
    ≈ **19h/代**；第 4 代起锁定配置，只剩自对弈 + 1 次训练 + arena ≈ **10.4h/代**。
    十代跑满约 109h（≈4.6 天），gen 1 arena ≈ +16.8h、gen 3 ≈ +46h。
    筛选赛是大头，但它只决定"哪个变体进 arena"，判决靠完整 arena，所以不为它省规格；
    真要压时间，砍变体个数比砍筛选赛局数划算。
  - **自对弈局数与训练步数是配套的**：每步要抽 `steps × accum 4 × batch 512 × 0.3` 条自对弈记录，
    而每局只产出约 133 条。4096 局 = 54.5 万条，让 1200 步（抽 73.7 万）落在 1.4x；
    1024 局时是 5.4x，就是初版连续不过门槛的病因。
  - **学习率计划与搜索维度（2026-09-27 改，依据是论文实测）**：
    - 计划用 **1cycle**（`{"kind": "onecycle", "pct_start": 0.25}`，峰值 lr = `optimizer.lr`），
      不用恒定 lr。两条理由：(a) 恒定 lr 要么噪声大（lr 大、终点落在噪声球里）要么不动
      （lr 小），1cycle 用"先升后降、末段比初值低几个数量级"解除这个绑定；
      (b) SGDR 论文的 incumbent 规则明确只在"跑完退火至 η_min 的那一点"取推荐解。
      Trainer 会存取 `scheduler.state_dict()`，所以循环中断续跑不会把计划错位。
    - 搜索维度从**单 lr 变成 lr × wd 二维**：Smith & Topin 2018（arXiv:1708.07120）实测
      大 lr 起正则化作用，必须同步削弱其他正则——ImageNet 上为用 0.05→1.0 的 lr，
      他们把 weight decay 从 1e-4 降到 3e-6。我们此前只扫 lr、wd 钉死 1e-4，
      扫出的"最优 lr"很可能是被 wd 压住的假顶点。
    - 步数固定 **1200**：同 lr 下 1200 全程优于 600（gen 0 实测 1e-6 从 −23.6 到 −9.0、
      2e-6 从 −4.5 到 +21.7），把这一维撤掉换给 wd。
    - 网格含一个**代内对照** `ctrl_const_2e-5_wd4`（恒定 lr、上一代最优配置），
      用来区分"1cycle 的功劳"和"这一代数据/冠军变了"的影响。
    - **开局库换成 2000 条合成线路**（Kit/data/openings_sp.txt）：arena / 筛选赛是
      确定性对局（temperature=0、无 Dirichlet 噪声），开局线路数直接决定去重局数。
      bundled 的 34 条跑 384 局时 distinct_games 只有约 276——**28% 的算力花在重复
      样本上**。合成库 2000 条，384 局拿到 384 个不同开局。这些线路只为多样性服务、
      不代表真实开局水准，双方走同一条所以比较仍公平；换库后 Elo 绝对值与旧库结果
      不可直接比较，但每代换代判定本来就是独立的一次完整 arena。**不要改回 bundled。**
    - **迁移风险**：这两篇论文用的都是 SGD+momentum，且明说 Adam 这类自适应方法
      "不使用足够大的学习率、也不会出现 super-convergence"。我们是 AdamW，
      lr 2e-5 在 Adam 尺度里偏小，与论文里 0.05~3.0 的 SGD 大 lr 不是同一量纲，
      凡是基于"大 lr 正 则化"的推论都要打折。**一个都不必信，只信 arena。**
- 与旧 `tools/gumbel_selfplay_corrected.py` 的**已知口径差异**（有意为之，别当 bug 查）：
  1. 自对弈每步都加 Dirichlet 噪声（kit `run_selfplay`），旧脚本只首步加；
  2. 混合数据按 batch 内比例切分（kit `data.kind:"mix"`），旧脚本是每 step 二选一；
  3. 每局 Player 种子由 (seed, 局序号) 派生（kit `_game_seed`），旧的 C++ 路径是全局 seed
     + 树复用；4. 开局 ply 由 Player 照 `GameStart.book` 原样走、不搜索（没有访问分布，
     不进训练目标），旧脚本在开局 ply 照常搜索。
- **训练与对局的精度不同**：`train` 段 `bf16`（与五个监督配方一致），`engine` 段 `fp16`
  （与 `config.json` 生产预设一致），两者绝不混用同一批前向。
- 每一代要看的四个数：自对弈局面数与三类终止分布、各变体的筛选赛成绩与选中的 label、
  arena 分数与 SPRT 判决、墙钟。若自对弈 90%+ 三次重复，说明数据没多样性，
  先查 kit 的每局种子是否真的落到了 Player 的 RNG 上。
- **暂停纪律**（对齐 SSM AGENTS §9）：连续 3 代未换代就停下复盘，**不放宽门槛**。
  `loop_p4_v2.json` 目前没有自动暂停规则，只能人工 `kill`（`loop_state.json` 会接着续跑）。


- **Curriculum 配方只训一个专家**（由 config 指定），另外两个从 `stratified_20m` 预训练
  **冻结**导入，导出仍是完整三个专家权重。
- 预训练权重早于 `mlh_head`：按新模型加载会缺键，`load_model` 只允许缺 `mlh_head.*`，
  别的缺键一律报错。Server 侧用的是 `max_mcts` / `max_t` 预设。
- `config.json` 预设的 `max_mcts` / `max_t` 语义与排序门槛见 `docs/architecture.md`。
- 温度采样的实测数据（mean SF rank 表）仍在 `docs/experiments.md`。

## 架构概览

Transformer 20M 引擎：平面编码 → Transformer 主干（2D 空间几何先验）→ 平方到平方双线性
policy 头 + promo 专用头 + WDL 值头 + MLH 头；开局/中盘/残局三专家由**固定 phase-stratified
路由**选择（不做学习的门控）。训练期数据、损失、调度器全在 kit（`Kit/planes19/` + `Kit/train/`）；
批量对弈与观战走 kit 原生 Player（跨局攒批）。历史契约（对局复现性、eval 换算、
GSPBT 口径）由 `Kit/tests/` 的回归测试接替。
