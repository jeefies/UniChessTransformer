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
生产预设（`config.json` 的 `max_mcts` / `max_t`，当前指向
`runs/loop_p4_v2/gen_0003/train/final.pt`）只能由人工切换：
改 `config.json` 一次提交 + 重启 `unichess-server`，loop 不许碰它。
gen 1（+73.8）与 gen 3（+65.1）两个 arena 验证的新冠军都以该文件为链路终点，
血统：`stratified_p4_selfplay_corrected/best_model.pt` → loop_p4 gen0 +163
→ v2 gen1 +73.8 → v2 gen3 +65.1。回滚 = 改回旧 ckpt 路径后重启。

截至 **2026-09-30 13:45 本地**：gen 4（Elo **−74.8**）、gen 5（Elo **−15.45**）
接连判 **H0、均未换代**，新冠军仍是 `gen_0003/train/final.pt`，**连续未换代计数 = 2**
（暂停线是 3 代，下一代会触发）。gen 4 的归因见 §3 下方：**过训练 + 导出点错**，
与 lr 无关；gen 5 用同一份锁定配置同样 H0，是第二个数据点。

**配置已于 13:41 改并重启**（§3.1）：`steps 1200 → 400`、自对弈权重 `0.3 → 0.5`，
lr 仍为锁定值 5e-4。gen 6 起按新配置训练（自对弈窗口内重启，已写 998 局完整保留，
稳态 7.50 s/局），预计 **20:12** 跑完自对弈、判决 **20:30–21:00**。

**14:28 第二批**（§3.2）：候选改取 `best_model.pt` 并开启 Kit 的
`select_best_by=train`（无 validation 时按训练 loss 最低步导出）。
第二次重启，新 loop PID **832414**，1326 局数据保留。gen 6 预期顺延到
自对弈约 **20:15** 完成、判决 **20:30–21:00**。

**21:00 loop 崩过一次、22:04 恢复**：崩溃根因是 Kit `bf17fd5` 的
`select_best_by` 判据写成 `validate is None`，而 T 的 `Planes19Task` 自带
validate 方法（没配数据源时拿不到 score），分支未触发 → best_model.pt 没导出
→ `phase_train` fail-fast。Kit `cf831fa` 已修判据并补 3 个回归测试。
gen 6 训练产物完整、恢复后 best_model.pt 与 final.pt 张量 sha256 相同
（gen6 的 loss 最低点正是 step 400），**判决未被污染**。
arena 22:04 重跑，判决预计 **22:10–22:30**。
详见 `experiments.md` §18.13。

gen 3–9 按锁定配置训练 = **lr 5e-4 + 1cycle(400 步) + wd 1e-4 + 自对弈配比 0.5**，
注意**锁的是筛选赛胜者、不是 arena 判决**（§5.2）。
（这一行会过期，以 `loop_state.json` 为准。）

> **生产权重已落后两代**：v2 线上有两个经 arena 验证的新冠军（gen 1 +73.8、gen 3 +65.1），
> 而 `config.json` 仍指向 `runs/stratified_p4_selfplay_corrected/best_model.pt`。
> 切换是人工动作（改 `config.json` + 重启 `unichess-server`），loop 不许碰。

> `loop.log` 里有 4 条指向 `gen_0000/arena.log` 的 Traceback，是 **2026-09-27 的旧代码**
> 留下的（栈里行号 346/390，现在是 443/462），不是新故障，别被它误导。

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
| gen 2 | 未换代 | **39 局**（19 对），score_a 0.5641 / Elo +44.8 / 18 胜 8 和 13 负 | 第 **20** 局越下界 −2.251 判 H0，收尾 llr 被在途局抬回 +0.284、`stopped_at_game=20` |
| gen 3 | **换代成功**（锁定代） | **81 局**，score_a 0.5926 / Elo **+65.1** / CI95 [0.2, +134.9] / 38 胜 20 和 23 负 | llr=+3.712、第 **61** 局越上界 +2.890、verdict=**H1**、`stopped_by_sprt=true` |
| gen 4 | 未换代（锁定代） | **33 局**，score_a 0.3939 / Elo **−74.8** / CI95 [−185.2, +22.0] / 8 胜 10 和 15 负 | 第 **14** 局越下界 −2.251 判 H0，llr 收尾 **−2.672**（一直在下界下方，无在途局拉回）、`stopped_by_sprt=true` |
| gen 5 | 未换代（锁定代） | **45 局**，score_a 0.4778 / Elo **−15.45** | H0，`sec` 35592.0（9.89 h）；比 gen 4 温和但方向一致——候选没有变强 |

gen 2 是 5 变体新网格首战：筛选赛 lr 单调（`5e-4` 0.622 > `2e-4` 0.578 >
`1e-4` 0.574 > `ctrl` 0.555 > `1e-4_wd1e-2` 0.553），选中 `1c_5e-4_wd1e-4` 进 arena，
但 **arena 判 H0**：候选是冠军在 gen 2 新数据上的一次 1200 步重训，
+44.8 Elo 够不上本循环 `elo1=60` 的晋级线。冠军未动，仍是 gen 1 的 lr 1e-4 模型。
**注意 gen 2 全代五个变体的筛选赛分数都比 gen 1 同代低 0.03–0.07**
（`ctrl` 0.555 vs 0.591、`1c_4` 0.574 vs 0.642），是本代数据的边际收益在下降的信号，
不是某一档 lr 的问题。证据与 wd 探针结论见 `experiments.md` §18。

**gen 3 用实证回答了上一段的疑问**：它是第一个锁定代（无筛选赛，直接按锁定的
lr 5e-4 训练），arena **+65.1 Elo 判 H1 换代成功**。所以 gen 2 的 H0 是数据/抽样
问题而不是 lr 问题，§5.2 那个"默认原样跑、不改锁定"的决定是对的。
连续未换代计数已清零（gen 2 H0 → gen 3 H1）。新冠军 = `gen_0003/train/final.pt`。

**gen 4 反转了 gen 3 的结论**：同样是锁定 lr 5e-4，arena **−74.8 Elo 判 H0**，
llr = −2.672 一直在下界下方（不像 gen 2 的收尾 llr 被在途局拉回），stopped_by_sprt=true。
所以 lr 5e-4 轨迹现在是 **1 胜 2 负**（gen 3 +65.1 H1，gen 4 −74.8 H0，gen 5 −15.45 H0）。
**连续未换代计数 = 2**（gen 4、gen 5），暂停线 3 代只差一代。

**gen 4 的归因结论与 lr 无关，别再往 lr 上靠**（2026-09-30 完整排查，`experiments.md` §18.9）：
候选在同一份数据上的 loss 比冠军**还低 0.0275**（自对弈来源低 0.0714），
fp16/fp32 比对无损（激活峰值 42.6，上限 65504），与冠军策略 KL 仅 0.013
（比 gen1→gen3 的 0.025 还小），也没有向弱引擎回归。真因是三件事叠加：

1. **step 300 就到最优，之后 900 步把它推出最优点**——`train.log` 里 loss
   从 1.4014 单调回升到 1.4269，而 `export.final` 只导 step 1200，
   **导出的正是全程最差的点**。gen3 是"终点即最低点"所以同样 1200 步无害。
2. **过拟合签名**：训练内分片 loss 降 0.046/0.093，训练外的 gen2 分片反而升 0.0099。
3. **两个源体量差 4.6 倍**（次要因素）：同样 70/30 批内比例下，自对弈语料每代
   覆盖率 71%、静态只 36%，自对弈被重复榨干的速度是静态 2 倍。
   ⚠ 我一度误判成"`_mix_batches` 取整导致自对弈占比只有 17.7%"，
   **那是错的**（把数据集大小当成了训练目标占比；实测每批 [359,153]=70.1/29.9，
   权重被如实执行）。已在 `experiments.md` §18.9「真因三」纠错。

可执行的改法（按优先级：`export` 早停语义 → 压短 onecycle 到 400–600 步 →
**不要去"修"`_mix_batches` 取整**，那不是 bug，见上文纠错），都写在 §18.9 末尾。

**gen 5 用同一份锁定配置（lr 5e-4 / 1200 步）同样判 H0**：Elo **−15.45**、
score_a 0.4778、45 局、`sec` 35592.0（9.89 h）。比 gen 4 的 −74.8 温和得多，
但方向一致——候选没有变强。**连续未换代计数来到 2**，暂停线（3 代）只差一代。

### 3.1 配置已改并重启：steps 400 + 自对弈 0.5（2026-09-30 13:41 落地）

上一节"1200 步没落地"的状态在 **2026-09-30 13:41 本地**结束。按 §18.9/§18.10 的
归因改了 `configs/loop_p4_v2.json`（提交 `511403e`，远端已 pull）：

| 字段 | 原值 | 新值 | 理由 |
|---|---|---|---|
| `train.steps` | 1200 | **400** | gen4 的 loss 在 step 300（lr 峰值处）就见底，之后 900 步单调回升 |
| `data.sources[0].weight`（ResNet 静态） | 0.7 | **0.5** | 与自对弈配平 |
| `data.sources[1].weight`（自对弈） | 0.3 | **0.5** | 每批取整后 `[256,256]` 恰好一半；静态源每代只被用掉 36%，自对弈 71% |

**未动**：`optimizer.lr` 仍是锁定值 **5e-4**（`loop_state.json` 的 `train_variant`
只覆盖 `optimizer.lr/weight_decay`，`_merge` 是递归合并，`steps` 与 `sources`
走基座模板——已实测确认）、`schedule`（onecycle / `pct_start 0.25`）、
arena SPRT 三条参数、`games 4096`、`enumerate_generations 3`。

**重启细节**（`phase=selfplay` 窗口内，安全）：

- 先 SIGTERM 自对弈子进程，loop 因子进程 `rc=-15` 抛 `RuntimeError` 自行退出
  （`loop.log` 里因此多一条 Traceback，**这是计划内的，不是新故障**）。
  不需要 SIGKILL——`Kit/pipelines/loop.py:466` 的 `FileLock` 是 OS 级 flock，
  进程退出即释放，实测 2 秒后 `loop.lock` 恢复空闲。
- 已在写的 998 局**完整保留**：`SelfPlayShardSink._repair()` 按元数据条数截断半截尾，
  `plan_selfplay` 按全局局号 `24576` 起编号，`run_selfplay` 用
  `skip_games=done_games()` 跳过已落盘局（`selfplay.py:156`），
  续跑与单次连跑产出同一批对局。
- 新 PID **811011**（父进程 init，已 setsid 脱离会话），自对弈子进程 811032，
  `/tmp/unichess_t_loop_v2.pid` 已写回。

**重启后的预热坑又踩了一次**：重启后头 90 秒只写 2 局（≈45 s/局），
差点误判成续跑有 bug。拉长到 5 分钟测，**稳态 7.50 s/局**，与重启前一致
（重启前 5.5 分钟 44 局 = 7.5 s/局）。§4.2「别再踩的测量坑」再次应验。

**gen 6 预期**：自对弈剩 3023 局 × 7.50 s = **6.3 h**，约 **20:12 本地**跑完；
训练 400 步约 3.5 min；arena 判决预计 **20:30–21:00 本地**。

### 3.2 第二批：候选改取 best_model.pt（2026-09-30 14:28 落地）

§3.1 只修了 gen 4 事故的两个根因之一（步数过长），第二个——**导出的点是全程最差的
点**——靠 Kit 侧修，commit `bf17fd5`：

- `Kit/train/config.py`：`export` 新增 `select_best_by ∈ ("train","none")`，
  **默认 `"none"`**（完全保持旧行为，任何现有配方的速度与语义都不变）。
  `export` 整体不进配置哈希，加开关不影响续训。
- `Kit/train/trainer.py`：`select_best_by="train"` 且无 validation 时，在每个日志点
  比较训练 loss（最近 `log_every` 个优化步的平均，覆盖
  `log_every×accum×batch` 个样本，循环里约 10 万，足够稳），创新低就把该步导出到
  `export.best` 并在日志记 `best`/`improved`。只有一步都没选出（极短训练 /
  续跑紧贴结尾、整个区间没落到 log 点）才退回复制 final。
- `Kit/tests/test_train.py`：新增 `TestSelectBestByTrain`（4 项）+
  `VShapedLossTask`（CE 之外叠一个以 step 为自变量的 U 形偏置，1:1 复刻 gen 4 的形状，
  否则"best=loss 最低点"和"best=final 副本"分不开）。远端全量 **346 项 OK**
  （原基线 342 + 4）。

配套改 loop 配置（commit `5c3d68c`）：

| 字段 | 原值 | 新值 |
|---|---|---|
| 顶层 `export`（`Loop.candidate_path` 用，arena/screens 同源） | `"final.pt"` | **`"best_model.pt"`** |
| `train.export.select_best_by` | （无，默认 none） | **`"train"`** |

⚠ **耦合**：loop 的 `export=best_model.pt` 依赖 Kit 的 `select_best_by`。
Kit 必须在 loop 之前升级，否则 `loop.py:338` 会因找不到候选文件 fail-fast
（不静默、不坑人）。

**第二次重启**（14:28，同样是 `phase=selfplay` 安全窗口）：新 loop PID **832414**，
自对弈子进程 832435，1326 局数据完整保留。稳态约 7 s/局（头一分钟仍是预热）。

**gen 6 更新的预期**：剩约 2746 局 × 7.5 s ≈ 5.7 h → 自对弈约 **20:15 本地**完成；
训练 400 步约 3.5 min；arena 判决预计 **20:30–21:00**。

**这一批解决了 gen 4 事故的第二个根因**，两条合起来的口径是：
轨迹不再过长（步数 400）+ 导出的点是最优点而非终点（best_model.pt）。
对 gen 6 而言，若 loss 单调下降（400 步下大概率如此），`best_model.pt` 与
`final.pt` 逐位相同，这次改动是**零风险的保险**；若中途仍有回升，则恰好兜住。

⚠ **上面那句"零风险"当晚就被推翻**——Kit `bf17fd5` 的 `select_best_by` 判据写错，
loop 在 21:00 左右崩了、gen 6 的 arena 没跑成。完整经过、根因、修复（Kit `cf831fa`）、
回归测试与 22:04 的恢复操作见 `experiments.md` §18.13。
一句话版本：判据写成 `validate is None`（方法不存在），但 T/R 的 `Planes19Task`
**自带** validate 方法、没配数据源时返回空 dict 拿不到 score，行为与没有方法一致，
分支永不触发、best_model.pt 不导出 → loop 在 `phase_train` 的候选检查处 fail-fast。

**教训**：跨仓耦合（Kit + loop）落地后必须端到端 smoke，
不能只看单测绿——本例单测全绿、全量 346 项 OK，线上照样崩。

**gen 6 的判决没被崩溃污染**（已验证）：训练产物完整；恢复时导出的
`best_model.pt` 与 `final.pt` 的 816 个张量 **sha256 完全相同**；
而 gen 6 的 loss 最低点正是 step 400（1.274855，见 §18.13 表），
所以 best=final 语义等价，arena 测的就是 gen 6 训出的模型。

**gen 6 更新的预期**：arena **22:04 本地**开跑，24 s/局，
SPRT 早停 14–61 局（≈6–25 min）→ 判决约 **22:10–22:30**。

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
| 自对弈 | **9.6 h**（gen 2 实测 8.35 s/局、gen 3 实测 8.45 s/局，两代一致） | 4096 局 |
| 变体训练 | 约 10 min/变体 | 5 个 × 1200 步（2026-09-28 起；之前 8 个） |
| 筛选赛 | **46–49 min/场**（一场曾到 80 min） | 每变体 384 局 @800 sims |
| 最终 arena | 22.1 s/局；SPRT 早停约 0.6 h，打满 512 局 3.1 h | 512 局 @2400 sims |

- **枚举代**（前 3 代）≈ **16 h**（5 变体；8 变体时实测 19 h）；**锁定代**
  （第 4 代起）gen 3 实测 **10.2 h**（自对弈 9.45 h + 训练 11 min + arena 33 min）。
  十代 ≈ **2×16 + 1×16 + 7×10.2 ≈ 132 h**（≈5.5 天）；从 gen 4 起算 6 代 ≈ 61 h，
  预计 **10-03 05:10 本地**收尾。
- gen 2 的实测分账：自对弈 9.5 h + 5×训练 0.83 h + 5 场筛选赛 4.83 h + arena 0.26 h
  （39 局，953.8 s）= 15.4 h，与上表一致。
- **2026-09-30 13:41 改配置后重启**（详见 §3.1）：`steps` 1200 → **400**、
  自对弈权重 0.3 → **0.5**（每批 [256,256] 恰好一半）。训练从 11 min 降到约 3.5 min，
  锁定代单代预算相应降到约 **9.9 h**（自对弈仍是 9.45 h 的大头）。
  gen 6 起按新配置训练。
- 筛选赛是大头，但它只决定"哪个变体进 arena"，判决靠完整 arena，所以不为它省规格；
   真要压时间，**砍变体个数比砍筛选赛局数划算**（见 §5.2 的死区问题，白烧过 6 个）。

**别再踩的测量坑**（2026-09-29 自己犯过）：估自对弈速率别用"进程启动到现在"，
那段包含引擎加载与首次编译缓存；也别只信 `selfplay.games.jsonl` 的行数跳变。
可靠做法是 `wc -l` 配 `ps` 的 ELAPSED 交叉验证，再看 `.sp.bin` 的增长
（每游戏约 20 KB / 每记录 160 字节）。按"650 局 / 101 分钟"会算出 9.3 s/局，
实际是 **8.45 s/局**——那一小时几乎全是预热，直接害 gen 3 的 ETA 虚高 1 h。

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

**gen 2 已把这个探针用掉了（2026-09-29）**：`1c_1e-4_wd1e-2` 与 `1c_1e-4_wd1e-4`
训出的权重 **792/816 个浮点张量不同**（平均相对 L2 差 **3.4e-3**、最大绝对差 1.4e-2，
落在上表"1200 步累计收缩 0.12%"同一量级，比一阶估算大约 3 倍是轨迹分歧），
但两场筛选赛 score_a **0.5534 vs 0.5742**（差 0.021 ≈ **0.8 倍标准误**）分不出高低。
**wd 从 1e-4 到 1e-2 跨两个半数量级，棋力无感——wd 这一维到此关闭**，
后面的代不该再为它花 slot。

**锁定的是筛选赛胜者，不是 arena 判决**（`Kit/pipelines/loop.py:493`）：
`train_variant` 在 search 阶段结束时就写盘，那一刻 arena 还没跑，
所以 `enumerate_generations` 的锁定**在结构上不看 arena 结果**。
gen 2 的实况正是如此：`1c_5e-4_wd1e-4` 赢下筛选赛 → 锁定
`{"optimizer": {"lr": 5e-4, "weight_decay": 1e-4}}` → arena 判 H0 未换代 →
**gen 3–9 仍按 lr 5e-4 训练**，而冠军还是 gen 1 的 lr 1e-4 模型。
即"刚被 arena 否掉的配置"被锁 7 代，"唯一赢过一次 arena 的配置"（1e-4）反倒出局。
这不是 bug，是枚举预算与判决预算分离的代价；但**换网格或动锁定前必须知道自己在动什么**。

**结局（2026-09-29 16:43）**：gen 3 就按这份锁定的 lr 5e-4 训练，
arena **+65.1 Elo 判 H1 换代成功**（见 §3）。所以 gen 2 的 H0 是数据/抽样问题，
不是 lr 问题，"默认原样跑"赌对了。下面留给以后真要改锁定时用：

| 选项 | 做法 | 代价 / 风险 |
|---|---|---|
| 原样跑（默认，gen 3 已验证） | 什么都不做 | 筛选赛本来就更看好 5e-4；H0 的含义是"够不上 elo1=60"而非"更差"（收尾 llr 还是正的 +0.28）。暂停纪律是内置刹车：连续 3 代未换代才复盘 |
| 改回 lr 1e-4 | 该代自对弈窗口内停 loop → 改 `loop_state.json` 的 `train_variant` 为 `{"optimizer": {"lr": 1e-4, "weight_decay": 1e-4}}` → 重启 | 要重启一次 loop（selfplay 窗口是安全的，已实测）；放弃本次枚举的结论 |
| 重新枚举 | 把 `enumerate_generations` 调到 4 后重启 | 总时长 +9 h（多一代 5 变体网格） |

> 改 `train_variant` 的**硬截止点是该代的训练阶段**（读取内存状态的那一行）。
> gen 3 的实际时点是 09-29 15:10 前后；以后每代按"自对弈跑完 + 0 min"估算即可。

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
8. **`enumerate_generations` 的锁定发生在 search 结束、arena 之前**（`loop.py:493`）：
   锁的是**筛选赛胜者**的配置，arena 判 H0 也不会改。想改锁定只能在下一代**训练阶段
   开始前**动 `loop_state.json` 的 `train_variant` 并重启 loop（selfplay 窗口安全）。
   gen 2 已实际发生：锁定 lr 5e-4，而冠军是 lr 1e-4。详见 §5.2。

## 7. 每代要看的四个数 + 暂停纪律

每代：自对弈局面数与三类终止分布 → 各变体筛选赛成绩与选中 label →
arena 分数与 SPRT 判决 → 墙钟。若自对弈 90%+ 三次重复，说明数据没多样性，
先查 kit 的每局种子是否真的落到 Player 的 RNG 上。

**暂停纪律**（对齐 SSM AGENTS §9）：连续 3 代未换代就停下复盘，**不放宽门槛**。
`loop_p4_v2.json` 没有自动暂停规则，只能人工 `kill`（`loop_state.json` 会接着续跑）。
