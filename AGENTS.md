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
| `configs/{t20m,stratified_opening,stratified_middlegame,stratified_endgame,p3_mlh}.json` | 训练口径 |
| `tests/test_r3.py` | 单测（17 项） |
| `docs/architecture.md` | 架构与模块说明 |
| `__init__.py` | 包声明 |

**已删除**（git 历史可查，勿 recreate）：`unichess_t/` 包（含自带的 C++ MCTS 与 Python MCTS）、
`train/` 五个脚本、`tools/`、`eval/`、`logs/`、`benchmark_transformer.py`、`uci.py`。

## 常用命令

```bash
cd ~/UniChess
python -m Kit train Transformer/configs/t20m.json       # curriculum 配方同理
python -m unittest Transformer.tests.test_r3            # 17 项
python -m Kit match <config.json> --out runs/<name>/results.jsonl
```

## 搜索不在本仓库

所有对局都走 kit 的 `PUCTCpp` / `PUCT`（`Kit/search/`），后者与 Python 实现整树逐位一致；
Syzygy 桌库在 `Kit/rules/tablebase.py`，开局库在 `Kit/rules/openings.py`。

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
