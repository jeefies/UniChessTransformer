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
  `loop_p4_v2.json` 是**激进版**：自对弈 **1024 局**、arena **256 对** 且 `elo1=60`，
  并且每代自对弈完成后做一次 **lr × 步数枚举搜索**（见 `Kit/pipelines/loop.py` 的
  `train.variants`）：10 个变体各自训练到 `gen_XXXX/train_<label>/`，再与冠军各打 64 对
  筛选赛，取 `score_a` 最高者进最终 arena。
- 初版已跑满 8 代，结论见 `docs/experiments.md` §15：gen 0 换代成功，之后七代全"判不出"，
  原因不是候选差而是 80 局分辨不出 +57~70 Elo；v2 就是按那些教训改的。
  当前冠军 = `runs/loop_p4/gen_0000/train/final.pt`（v2 的 `initial` 指向它）。
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
    30.2/30.8。concurrency 8/12/16 无差别。`workers>1` 时 SPRT 判决仍由父进程按已回传
    记录给出，只是不承诺"停止时点"。
  - 每代耗时构成：自对弈 4096 局约 9.7h + 变体训练约 1.4h + 10 场筛选赛（192 对 × 800 sims）
    约 9.3h + 最终 arena（512 局 × 2400 sims）约 4.7h ≈ **25h/代**。
    筛选赛是大头，但它只决定"哪个变体进 arena"，判决靠最终 512 局，所以不为它省规格；
    真要压时间，砍变体个数（10 → 6）比砍筛选赛局数划算。
  - **自对弈局数与训练步数是配套的**：每步要抽 `steps × accum 4 × batch 512 × 0.3` 条自对弈记录，
    而每局只产出约 133 条。4096 局 = 54.5 万条，正好让 600 步（抽 36.9 万）和 1200 步（抽 73.7 万）
    都落在 0.7x~1.4x 的健康区；1024 局时是 2.7x~5.4x，就是初版连续不过门槛的病因。
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
