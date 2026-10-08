# AGENTS.md — UniChess Transformer

> 面向 AI 编码 agent。最后更新：2026-10-08。
>
> **换代循环的一切实测数字、结论与踩过的坑都在 [`docs/loop.md`](docs/loop.md)**，
> 逐代原始数据与证据链在 [`docs/experiments.md`](docs/experiments.md)（§15/§16/§17/§18）。
> 本文件只留"不知道会出事"的部分。

## 训练主机（pro）与三路由

本仓的换代循环与 GPU 训练**自 2026-10-04 起在 pro 上跑**——这是全项目群的**唯一例外**，
其余仓仍是全局 `../AGENTS.md` 的双设备路由（Windows + 70Ti）。本仓是三路由：

| 设备 | 路径 | 本仓用途 |
|---|---|---|
| Windows 本机 | 本仓工作副本 | 代码编辑、文档、纯 Python 单测 |
| **pro**（`ssh -p 26657 fwj@connect.westd.seetacloud.com`，RTX PRO 6000 96 GB，cgroup 22 核） | `/root/autodl-tmp/fwj/UniChess`（HOME 同盘） | **换代循环、GPU 训练/评测** |
| 远端 5070 Ti（`jeefy@172.16.2.12`，SSH 免密） | `~/UniChess/Transformer` | **生产引擎（Server 的 T 插件）+ 备份**（权重只在 pro 训练完 scp 过去） |

**pro 的关键坑**（详细迁移记录与吞吐实测见 `docs/experiments.md` §18.26-27）：
- cgroup 只有 **22 核**（nproc 报 208 是宿主，别信）；torch 默认开 104 线程池会在 22 核上互踩
  → 训练/自对弈**必须 `OMP_NUM_THREADS=4`**（漏了会看到 worker 烧 ~500% CPU 且初始化拖几分钟）；
- `/tmp` 不可写 → `TMPDIR=$HOME/UniChess/tmp`（已进 .bashrc）；
  **临时脚本一律放 `~/UniChess/tmp`，家目录不放**（用户指示）；
- pro 的工作树是路径补丁过的"脏"树 → **禁止 git pull**，代码更新只能 scp 具体文件；
- **起 loop 前必须 `cd /root/autodl-tmp/fwj/UniChess`**（否则 `-m Kit` 找不到包）；
- 自对弈多进程常态：`workers=4 × concurrency=32` + **MPS 守护**
  （`setsid nvidia-cuda-mps-control -d`，pipe/日志在 `~/UniChess/tmp/mps_*`，loop 启动 env 带
  `CUDA_MPS_PIPE_DIRECTORY/CUDA_MPS_LOG_DIRECTORY`）→ 实测 ~5.3 s/局（单进程 15-17）；
- pro 与用户自己的其他任务（ICLR 实验等）**共用 GPU**，动 GPU 前先 `nvidia-smi` 看占用、
  别动对方进程；
- 换代中途重启要用"剩余局数"法续跑（顶层 `games` 改成剩余 → loop 的
  `first_game=g*games` 生成不相交局号段），**事后必须恢复 4096**——忘了的话后续代
  都只打半量局（gen 11 就中过招，见 §18.27/28）。

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
| `configs/loop_p4.json` | 换代循环初版（200 局 / 40 对 / 无枚举） |
| `configs/loop_p4_v2.json` | 激进版（4096 局 / 256 对 / 8 变体枚举，**在跑**） |
| `tests/test_r3.py` | 单测（20 项） |
| `docs/architecture.md` | 架构与模块说明 |
| `docs/loop.md` | **换代循环的要点、结论、实测参数与运维坑** |
| `docs/experiments.md` | 实验记录（含 §15/§16/§17 三代循环实录） |
| `__init__.py` | 包声明 |

**已删除**（git 历史可查，勿 recreate）：`unichess_t/` 包（含自带的 C++ MCTS 与 Python MCTS）、
`train/` 五个脚本、`tools/`、`eval/`、`logs/`、`benchmark_transformer.py`、`uci.py`。

## 常用命令

（`Kit loop` 与 GPU 相关命令在 **pro** 上跑，见上方「训练主机」节；单测无 torch 也能跑的部分在本机。）

```bash
cd ~/UniChess
python -m Kit train Transformer/configs/t20m.json       # curriculum 配方同理
python -m Kit loop  Transformer/configs/loop_p4_v2.json  # 自对弈换代循环（见 docs/loop.md）
python -m unittest Transformer.tests.test_r3            # 20 项
python -m Kit match <config.json> --out runs/<name>/results.jsonl
```

## 搜索不在本仓库

所有对局都走 kit 的 `PUCTCpp` / `PUCT`（`Kit/search/`），后者与 Python 实现整树逐位一致；
Syzygy 桌库在 `Kit/rules/tablebase.py`，开局库在 `Kit/rules/openings.py`。

## 换代循环的四条红线

1. **换代的唯一依据**是 arena 的 SPRT 结论（判决 H1 才更新 `loop_state.json` 的 champion）；
   筛选用 `score_a` 只决定"哪个变体进 arena"，不算判决。
2. **生产权重只能由人工切换**：`config.json` 的 `max_mcts` / `max_t` 预设指向
   `runs/stratified_p4_selfplay_corrected/best_model.pt`，改它 = 一次提交 + 重启
   `unichess-server`，**loop 不许碰**。
3. **改 Kit 代码不用重启 loop**（子进程下次拉起即生效），**改 loop 配置必须重启**
   （进程只在启动时读一次配置）；且 `phase == "search"` 时不许重启。
4. **连续 3 代未换代就停下人工复盘，不放宽门槛**；loop 没有自动暂停规则。

并发参数、每代耗时、lr×wd 依据与**当前 wd 一维是数值死区**这个待决问题、
运维坑清单，全部在 [`docs/loop.md`](docs/loop.md)，照它做，不要重新发明。

## 其它必须记住的

- **训练与对局的精度不同**：`train` 段 `bf16`（与五个监督配方一致），`engine` 段 `fp16`
  （与 `config.json` 生产预设一致），两者绝不混用同一批前向。
  ⚠ bf16/fp32 的舍入会让**极小的 weight decay 完全失效**（每步衰减项不足一个 ULP），
  配 wd 前先看 `docs/loop.md` §5.2 的实测阈值。
- **Curriculum 配方只训一个专家**（由 config 指定），另外两个从 `stratified_20m` 预训练
  **冻结**导入，导出仍是完整三个专家权重。
- 预训练权重早于 `mlh_head`：按新模型加载会缺键，`load_model` 只允许缺 `mlh_head.*`，
  别的缺键一律报错。Server 侧用的是 `max_mcts` / `max_t` 预设。
- `config.json` 预设的 `max_mcts` / `max_t` 语义与排序门槛见 `docs/architecture.md`；
  温度采样的实测数据（mean SF rank 表）在 `docs/experiments.md`。

## 架构概览

Transformer 20M 引擎：平面编码 → Transformer 主干（2D 空间几何先验）→ 平方到平方双线性
policy 头 + promo 专用头 + WDL 值头 + MLH 头；开局/中盘/残局三专家由**固定 phase-stratified
路由**选择（不做学习的门控）。训练期数据、损失、调度器全在 kit（`Kit/planes19/` + `Kit/train/`）；
批量对局与观战走 kit 原生 Player（跨局攒批）。历史契约（对局复现性、eval 换算、
GSPBT 口径）由 `Kit/tests/` 的回归测试接替。
