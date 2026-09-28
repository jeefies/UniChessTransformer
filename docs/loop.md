# 换代循环（loop_p4 / loop_p4_v2）：要点与结论

> 本文件是**当前状态 + 实测结论**的活文档。逐代的原始数据、证据链、失败过程在
> [`experiments.md`](experiments.md)：§15 初版 8 代实录、§16 配方依据、§17 v2 的 gen 0/gen 1。
> `AGENTS.md` 只保留"不知道会出事"的那几条，细节都指向这里。

## 1. 现在跑的是什么

```bash
cd ~/UniChess
python -m Kit loop Transformer/configs/loop_p4_v2.json   # 在跑的（激进版）
```

| 东西 | 位置 |
|---|---|
| 状态 | `Transformer/runs/loop_p4_v2/loop_state.json`（`generation` / `phase` / `variant` / `champion` / `history`） |
| 每代记录 | `Transformer/runs/loop_p4_v2/loop.jsonl` |
| 每代产物 | `gen_XXXX/{selfplay/,train_<label>/,screen_<label>.jsonl,arena.jsonl.summary.json,search.json}` |
| 日志 | `loop.log`，各子进程 `.log` 同目录 |
| loop PID | `/tmp/unichess_t_loop_v2.pid` |

**换代的唯一依据**是 arena 的 SPRT 结论：判决 H1 才更新 `loop_state.json` 的 champion。
生产权重（`config.json` 的 `max_mcts` / `max_t` 预设，指向
`runs/stratified_p4_selfplay_corrected/best_model.pt`）**只能由人工切换**：
改 `config.json` 一次提交 + 重启 `unichess-server`，loop 不许碰它。

截至 **2026-09-28 12:00 UTC**：gen 2 自对弈 54%，新冠军 =
`runs/loop_p4_v2/gen_0001/train_1c_1e-4_wd4/final.pt`，loop PID 155800。
（这一行会过期，以 `loop_state.json` 为准。）

## 2. 两个配方的规模差异

训练口径相同（P4），差别在规模与是否枚举：

|  | `loop_p4.json`（初版） | `loop_p4_v2.json`（当前） |
|---|---|---|
| 自对弈 | 200 局 / 800 sims | **4096 局** |
| 训练 | lr 5e-6、1000 步、恒定 lr | **lr 枚举** 5 变体 × 1200 步、**1cycle** |
| 筛选赛 | 无 | 每变体 192 对（384 局）@800 sims，取 `score_a` 最高者 |
| arena | 40 对 @2400 sims，`elo1=40` | **256 对（512 局）** @2400 sims，`elo1=60` |
| 枚举范围 | — | 只前 3 代（`enumerate_generations: 3`），第 4 代起按锁定配置直接训练 |

## 3. 逐代结论

**初版（loop_p4，8 代）**：gen 0 换代成功（+163 Elo、SPRT 57 局判 H1），
之后七代全部"判不出"——不是候选差，是 80 局分辨不出 +57~70 Elo
（SE 5.5%，真实提升整段落在 CI 里；蒙特卡洛：真实 +60 Elo 判 H1 的期望局数 133、P90 240）。
gen 3/4 掉到负 Elo 是自对弈数据被反复过拟合（每条记录平均被抽 23 次）。
v2 就是按这些教训改的：详见 `experiments.md` §15。

**v2**：当前冠军 = `loop_p4_v2/gen_0001/train_1c_1e-4_wd4/final.pt`。

| 代 | 结果 | arena | 判决 |
|---|---|---|---|
| gen 0 | 未换代（**假阴性**） | 101 局，score_a 0.559 / Elo +41.5 / CI95 [−19.3, +104.9] | 越界时 llr=+3.164 已判 H1，在途对局把收尾 llr 拖回 +1.471 → 旧代码 verdict 变 None |
| gen 1 | **换代成功** | 43 局（21 对），score_a 0.6047 / Elo +73.8 / CI95 [−12.7, +170.7] | llr=+1.868、verdict=**H1**、`stopped_at_game=29` |

gen 0 那次假阴性是 `Kit/pipelines/match.py` 的快照不一致 bug（`workers>1` 才暴露），
已修为"越界即冻结判决"，gen 1 的 arena 就是这个修复的现场证明——同样出现
"verdict=H1 而收尾 llr 在界内"，但现在有 `stopped_at_game` 可以对账。
证据与修法见 `experiments.md` §17.2/§17.3。

## 4. 实测参数（都枚举过，别再瞎调）

### 4.1 并发

- 自对弈 `concurrency 32` / `batch_size 64`：batch 64/128/256 都是 8.8 s/局，
  concurrency 32/64 都是 8.5 s/局——GPU 已饱和，加并发或加批都不涨。
  （gen 1 实测 8.3–8.4 s/局。）
- 训练 `batch_size 512` / `num_workers 4`：workers 4→8 只快 1%；batch 1024 直接 OOM
  （61M 三专家 + accum 4 的有效 batch 2048 吃掉 12.5G）。1.65–1.89 步/s。
- **arena / 筛选赛 `workers 2`**：比 workers=1 快约 10%，3/4 反而回落到 30.2/30.8。
  concurrency 8/12/16 无差别。

### 4.2 每代耗时与预算（5070 Ti 实测）

| 阶段 | 实测 | 规模 |
|---|---|---|
| 自对弈 | **9.4–9.6 h**（8.3–8.4 s/局） | 4096 局 |
| 变体训练 | 约 10 min/变体 | 5 个 × 1200 步（2026-09-28 起；之前 8 个） |
| 筛选赛 | **46–49 min/场**（一场曾到 80 min） | 每变体 384 局 @800 sims |
| 最终 arena | 22.1 s/局；SPRT 早停约 0.6 h，打满 512 局 3.1 h | 512 局 @2400 sims |

- **枚举代**（前 3 代）≈ **16 h**（5 变体；8 变体时实测 19 h）；**锁定代**（第 4 代起）
  ≈ **10.4 h**。十代跑满 = 3×16 + 7×10.4 ≈ **121 h**（≈5 天）。
- 筛选赛是大头，但它只决定"哪个变体进 arena"，判决靠完整 arena，所以不为它省规格；
  真要压时间，**砍变体个数比砍筛选赛局数划算**（见 §5.2 的死区问题，白烧过 6 个）。

### 4.3 自对弈局数与训练步数是配套的

每步要抽 `steps × accum 4 × batch 512 × 0.3` 条自对弈记录，每局只产出约 133 条
（= plies − 6 个开局 ply，kit 的 sink 口径已由测试钉住）。
4096 局 = 54.5 万条，让 1200 步（抽 73.7 万）落在 1.4x；1024 局时是 5.4x
——那就是初版连续不过门槛的病因。

## 5. 配方的依据与已知风险

### 5.1 计划用 1cycle（不用恒定 lr）

- 恒定 lr 要么噪声大（lr 大、终点落在噪声球里）要么不动（lr 小），1cycle 用
  "先升后降、末段比初值低几个数量级"解除这个绑定。
- SGDR（arXiv:1608.03983）的 incumbent 规则明确只在"跑完退火至 η_min 的那一点"取推荐解；
  Smith & Topin（arXiv:1708.07120）明确多周期锯齿重启观察不到 super-convergence，
  所以只做单周期。取 `pct_start=0.25`。
- Trainer 会存取 `scheduler.state_dict()`，**循环中断续跑不会把计划错位**（已验证）。

### 5.2 lr × wd 网格：wd 这一维是数值死区（2026-09-28 实测，已决策）

原依据：Smith & Topin 实测大 lr 起正则化作用、必须同步削弱其他正则
（ImageNet 上为用 0.05→1.0 的 lr 把 weight decay 从 1e-4 降到 3e-6）；
我们此前只扫 lr、wd 钉死 1e-4，怕扫出的"最优 lr"是被 wd 压住的假顶点。

**但这一维根本没测出东西**：gen 1 的 8 个变体里，同一 lr 组的三个 wd 档
训出的权重**逐位相同**（816 个浮点张量，max|Δ|=0，2026-09-28 复验仍如此），
三场筛选赛 384/384 局逐字节相同、score_a / Elo 完全相同。已排除配置与管道问题
（screen 的 `a.checkpoint` 各指各的、落盘配置和优化器 `param_groups` 里
`weight_decay` 确是 1e-4/1e-5/3e-6），最小复现定位到机制。

**机制**：AdamW 的解耦衰减每步把 p 乘 `(1 − lr·wd)`，衰减项 `|p|·lr·wd`
约 2e-10，而 |p|≈0.02 处 fp32 一个 ULP 约 1.9e-9——**每步的衰减项都不到一个 ULP，
每步被舍入成 0，永远累积不起来**。实测阈值：wd=1e-4 完全不可见，**1e-3 起才可见**。
参数主副本是 fp32（`bf16` 只包前向，`Kit/train/trainer.py` 的 autocast），
所以舍入确实发生在参数更新这一步。

**更重要的账**（AdamW 下衰减与更新之比 = `|p|·wd`，与 lr 无关）：

| wd | 每步衰减(lr=1e-4) | 过 fp32 ULP? | 1200 步累积收缩 |
|---|---|---|---|
| 1e-4 | 2.0e-10 | 否 → 死区 | 0.00% |
| 1e-3 | 2.0e-9 | 是 | 0.01% |
| 3e-3 | 6.0e-9 | 是 | 0.04% |
| 1e-2 | 2.0e-8 | 是 | 0.12% |
| 1e-1 | 2.0e-7 | 是 | 1.19% |

参照：优化器本身 1200 步把 |p| 移动 O(lr·steps)=O(0.12)，约 10%+。
也就是说**"过了 ULP"和"有训练意义"之间还差三四个数量级**——原方案 (b) 的
1e-3/3e-3/1e-2 只会重演"位不同但功能相同"，只是换了一级台阶。

**已定网格（gen 2 起，2026-09-28 人工决定）**：8 变体砍到 5 个，每枚举代省约
2.9 h（3 次训练 + 3 场筛选赛），且 5 个 slot 全部有信息量：

| label | lr | wd | 作用 |
|---|---|---|---|
| `ctrl_const_2e-5_wd1e-4` | 2e-5 | 1e-4 | 代内对照（恒定 lr） |
| `1c_1e-4_wd1e-4` | 1e-4 | 1e-4 | **gen 1 新冠军配置，本轮要打败的基线** |
| `1c_2e-4_wd1e-4` | 2e-4 | 1e-4 | lr 向上探一档 |
| `1c_5e-4_wd1e-4` | 5e-4 | 1e-4 | lr 再上一档：趋势延续还是破 |
| `1c_1e-4_wd1e-2` | 1e-4 | **1e-2** | wd 证伪探针：按上表预期与 1e-4 档功能等价 |

lr 轴保留并向上的理由：gen 1 筛选赛里 lr 是单调的（1e-4 0.642 > 5e-5 0.596 >
3e-5 0.589），而 1e-4 恰好是当时测试范围的上边缘，"lr 影响不大"与这组实测不符；
用这 3 个 slot 探明 lr 的真实上界，比再测一档无感的 wd 划算。
`wd1e-2` 那一档是把"wd 无感"从解析结论变成实证结论，成本 55 min。
label 从 `wd4/wd5/wd6` 改成显式数值（`wd1e-4`），就是为了不再让"位相同"藏在命名里。

**注意**：`enumerate_generations: 3`，gen 2 是**最后一个枚举代**，本轮胜者的配置
会被锁定、用于 gen 3–9 共 7 代的训练。所以 gen 2 的败者不是"下代再试"，
而是真的换不回来——除非人工删 `loop_state.json` 的 `train_variant` 再续跑，
或把 `enumerate_generations` 调到 4（代价约 +9 h 总时长）。

### 5.3 步数固定 1200 / 代内对照

同 lr 下 1200 全程优于 600（gen 0：1e-6 从 −23.6 到 −9.0、2e-6 从 −4.5 到 +21.7），
步数这一维不再动。网格保留 `ctrl_const_2e-5_wd1e-4`（恒定 lr + 上一代最优配置）
作代内对照，用来区分"1cycle 的功劳"和"这一代数据/冠军变了"的影响。

### 5.4 开局库用 2000 条合成线路

`Kit/data/openings_sp.txt`（python-chess 生成，seed 20260927）。arena / 筛选赛是
**确定性对局**（temperature=0、无 Dirichlet 噪声），开局线路数直接决定去重局数：
bundled 34 条跑 384 局只有约 276 个不同对局（重复率 0.27–0.375），
合成库 2000 条把重复率压到 **0.008**（381/384）。**不要改回 bundled。**

### 5.5 迁移风险

上面两篇论文用的都是 SGD+momentum，且明说 Adam 这类自适应方法
"不使用足够大的学习率、也不会出现 super-convergence"。我们是 AdamW，
lr 1e-4~2e-5 在 Adam 尺度里偏小，与论文里 0.05~3.0 的 SGD 大 lr 不是同一量纲。
凡是由"大 lr 正则化"推出的推论都要打折，**唯一可信的判据仍然是 arena 的 512 局**。

## 6. 运维坑（每一个都真的踩过）

1. **`phase == "search"` 时不许重启 loop**：换网格（label 集合变）会让
   `_search_all` 的"变体结果不齐"校验报错；旧 label 残留在 `search.json` 里还会
   让之后每次重启都卡在同一处。现在 kit 会丢弃旧网格 label 并告警（Kit 修复），
   但**能避就避**。安全窗口是 `selfplay` / `arena`（arena 会按同配置哈希续跑）。
2. **改 Kit 代码不用重启 loop**（`selfplay/train/arena/screen` 都是子进程，
   下次拉起即生效），**改 loop 配置必须重启**（进程只在启动时读一次配置）。
   gen 0 的筛选赛跑在旧配置上（开局库没换）就是这个原因。
3. **配置哈希不一致是硬失败**：`Kit match` 对结果文件哈希不符直接 `ValueError`，
   不是自动重跑。且它**先写表头再建引擎**——建引擎崩了会留下只有表头的文件，
   堵死后续所有重试（kit 已修为"无对局记录就重写表头"）。
4. **每局记录的 `elapsed_s` 不是墙钟**（虚高），墙钟只看汇总的 `elapsed_s`。
5. **arena / 筛选赛 `workers>1`** 时 SPRT 判决由父进程冻结（见 §3）；在途对局
   可能让收尾 llr 回到界内，`verdict` 与 `llr` 看着矛盾是正常的，用 `stopped_at_game` 对账。
6. **锁定代的 schedule 必须整体替换**：变体若带 `{"schedule":{"kind":"constant"}}`，
   与基座 `onecycle+pct_start=0.25` 普通合并会得到非法组合，让那一代
   **9.6 h 自对弈白跑之后**才在训练阶段报错（kit 已修，`_apply_overrides`）。
7. **与旧 `tools/gumbel_selfplay_corrected.py` 的已知口径差异**（有意为之，别当 bug 查）：
   ① 自对弈每步都加 Dirichlet 噪声（旧脚本只首步加）；② 混合数据按 batch 内比例切分
   （旧脚本每 step 二选一）；③ 每局 Player 种子由 (seed, 局序号) 派生（旧的是全局 seed
   + 树复用）；④ 开局 ply 由 Player 照 `GameStart.book` 原样走、不搜索、不进训练目标。

## 7. 每代要看的四个数 + 暂停纪律

每代：自对弈局面数与三类终止分布 → 各变体筛选赛成绩与选中 label →
arena 分数与 SPRT 判决 → 墙钟。若自对弈 90%+ 三次重复，说明数据没多样性，
先查 kit 的每局种子是否真的落到 Player 的 RNG 上。

**暂停纪律**（对齐 SSM AGENTS §9）：连续 3 代未换代就停下复盘，**不放宽门槛**。
`loop_p4_v2.json` 没有自动暂停规则，只能人工 `kill`（`loop_state.json` 会接着续跑）。
