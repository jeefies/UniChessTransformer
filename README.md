# UniChessTransformer

**UniChessTransformer** (Model T) is a neural chess engine combining Transformer backbones with 2D spatial geometric priors, a bilinear square-to-square policy head, Win-Draw-Loss (WDL) value heads, a moves-left head, phase-stratified expert routing, and a C++-accelerated batched Monte Carlo Tree Search (MCTS).

The flagship model (`runs/stratified_p4_selfplay_corrected/best_model.pt`) achieved a perfect **10-0** sweep against Model R (`chess_ai` / ResNet 15x192) in a 10-game compute-aligned match at 2400 MCTS simulations, running at **0.22s–0.29s/move** — all 10 games ending in checkmate.

Further reading: [`docs/architecture.md`](docs/architecture.md) for the architecture spec, [`docs/experiments.md`](docs/experiments.md) for the full experimental record, and [`AGENTS.md`](AGENTS.md) for agent-facing commands.

---

## Key Features

- **Geometric Transformer Architecture**
  - **ConvStem** maps 19 canonical bitboard planes straight into 64 spatial square tokens.
  - **Decoupled 2D embeddings**: learned rank/file embeddings per square.
  - **Pairwise relative attention bias**: a learned $(H, 64, 64)$ square-to-square bias passed directly into `F.scaled_dot_product_attention`, so fused FlashAttention/SDPA kernels are used.
  - **Bilinear square-to-square policy head**: origin queries x destination keys + learned move bias, flattened to 4096 move logits, with a decoupled promotion head for (Q, R, B, N).
  - **WDL value head** on the `[CLS]` token predicting Win/Draw/Loss; scalar $Q = P(\text{Win}) - P(\text{Loss})$.
  - **Moves-left head (MLH)**: auxiliary Smooth-L1 head predicting moves remaining (since Stage P3).

- **Model Hierarchy & Phase-Stratified Routing**
  - **`stratified_20m`** — **61,031,064** params, 3 x 11 layers. Routes each position to a specialized ~20.3M expert:
    - **Opening**: `piece_count >= 24` or `ply <= 20`
    - **Middlegame**: `12 < piece_count < 24`
    - **Endgame**: `piece_count <= 12`
  - **`transformer_20m`** — 20,343,688 params, 11 layers, $d=384$, 12 heads, SwiGLU $d_{\text{ff}}=1024$.
  - **`transformer_50m`** — 49,757,960 params, 17 layers, $d=512$, 16 heads, $d_{\text{ff}}=1160$.
  - Lightweight presets: `transformer_tiny` (3.8M), `transformer_small` (6.7M), `transformer_medium` (18.5M), `transformer_large` (35.1M).

- **Search is UniChessKit's PUCT** (`Kit/search/`) — no second search implementation in this repo
  - `PUCTCpp` (C++ via `Kit/search/_native/puct_native.cpp`) is the production path and is bitwise
    identical to the Python `PUCT`; leaves are encoded in C++ and forwarded as planes.
  - Batched **across games**, which is what the Server and the arena share.
  - Syzygy 3-4-5 (`Kit/rules/tablebase.py`) and openings (`Kit/rules/openings.py`) live in kit.

- **Training & Curriculum Learning**
  - Distillation from Stockfish evaluations with joint policy cross-entropy + WDL loss (+ MLH since P3).
  - Phase-stratified curriculum fine-tuning over 64 binary evaluation shards
    (`/home/jeefy/UniChess/data/shards_evals`), 96-byte fixed records, `np.memmap` + vectorized
    bitboard decoding — data code is `Kit/planes19/build`.
  - Training itself is `Kit.train.Trainer` + `Kit/planes19/task.py`; this repo only supplies the
    model, the batch forward and the recipes in `configs/`.
  - Current best: `runs/stratified_p4_selfplay_corrected/best_model.pt`.

- **Standards & Server Integration**
  - UCI via kit: `python -m Kit uci Transformer/engine.py --preset max_mcts`.
  - Server integration via symlink: `/home/jeefy/UniChess/Server/models/T -> /home/jeefy/UniChess/Transformer`.
  - Presets in `config.json` drive the Server's model selection. `max_mcts` is the frontend
    default (fully deterministic); `max_t` is identical plus `temperature=1.0` +
    `root_top_k=3` for game-to-game variety with a quality guard — it samples the root visit
    distribution restricted to the 3 most-visited moves, so it can never pick a move worse
    ranked than its 3rd root move. Relative preset paths resolve against this repository.
  - Production service is the systemd user unit `unichess-server`.

---

## Architecture Overview

```
                  Canonical Board Input (19 x 8 x 8), side-to-move oriented
                                    |
                          ConvStem (3x3, stride 1)
                                    |
                 64 Square Tokens + 1 prepended [CLS] token
                                    |
             + Rank / File Embeddings + Pairwise 2D Relative Bias
                                    |
                   Transformer Encoder Layers (Pre-LN)
                   - SDPA / FlashAttention with additive bias
                   - SwiGLU feed-forward MLP
                                    |
        +-----------------------+---+---+-----------------------+
        |                       |       |                       |
  Bilinear Policy Head    Promotion     WDL Value Head      MLH Head
  (Origin Q x Dest K)     Head (QRBN)   (MLP on [CLS])    (moves left)
        |                       |           |                   |
 4096 move logits        4 promo       W / D / L probs      scalar
```

For the full specification — plane layout, attention math, head formulations, and routing
thresholds — see [`docs/architecture.md`](docs/architecture.md).

---

## Head-to-Head Benchmark Results

### P4 Self-Play Championship vs Model R (`chess_ai` / ResNet 15x192) — current best
Evaluated across 5 balanced opening pairs (Italian, Ruy Lopez, Scotch, Four Knights, Petroff):

| Matchup | Model T (P4 Self-Play Corrected) | Model R (ResNet 15x192) |
| :--- | :---: | :---: |
| **Search** | **C++ MCTS + leaf Syzygy, 2400 sims** | Python MCTS + Syzygy, 800 sims |
| **Final Score** | **10.0 / 10 (100.0%)** | 0.0 / 10 (0.0%) |
| **Record** | **10W – 0D – 0L** | 0W – 0D – 10L |
| **Terminations** | 10 checkmates | — |
| **Avg move latency** | **0.22s – 0.29s** | 0.21s – 0.28s |

### Curriculum Match vs Model R (10 games, Stage 4)

| Matchup | Model T (Stratified 20M Curriculum) | Model R |
| :--- | :---: | :---: |
| **Search** | C++ MCTS + leaf Syzygy, 2400 sims | Python MCTS + Syzygy, 800 sims |
| **Final Score** | **5.5 / 10 (55.0%)** | 4.5 / 10 (45.0%) |
| **Avg move latency** | **0.34s** (~1.9x faster) | 0.65s |

### vs Baseline Engine (`chess_ai v2.0.0`, CNN + Negamax)
- **18.5 / 20 (92.5%)** undefeated — 17 wins, 3 draws, 0 losses, +436.4 Elo.
- 12.1 ms/move with C++ MCTS (100 sims) vs 268.3 ms for the baseline.

A full stage-by-stage progression table is in [`docs/experiments.md`](docs/experiments.md) §14.

---

## Verification & Test Suite

```bash
cd ~/UniChess

# 远端（conda unichess，torch 在这里）
/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest Transformer.tests.test_r3 -v
```

17 项：模型结构与参数量、分层路由、检查点往返（含 mlh_head 缺失的兼容）、kit 适配与
Server 插件契约、5 个配方配置的逐字段钉死。需要真实权重（`runs/` 存在）的部分会自动跑。

训练配方的验收口径不是"能跑完"，而是与旧脚本**逐位对齐**：前 50 个优化步的 loss / lr /
参数哈希全部相等（见 `AGENTS.md` 的「验收」一节）。

---

## Directory Structure

```
Transformer/                # 仓库根即包（import 根是 ~/UniChess）
├── model.py               # 网络结构：ConvStem + 2D 相对偏置 + 双线性策略头 + WDL/MLH 头
│                           #   + StratifiedChessTransformer 三专家路由；state_dict 键名冻结
├── evaluator.py           # 批量前向（kit 的 Player / Trainer 只用这个）
├── kit.py                 # kit 接入：对局 PlayerFactory + 训练 TrainTask
├── configs/               # 训练配方（复刻旧脚本）
│   ├── t20m.json                    # 旧 train/train.py（transformer_20m）
│   ├── stratified_opening.json      # 旧 train/curriculum_opening.py
│   ├── stratified_middlegame.json   # 旧 train/curriculum_middlegame.py
│   ├── stratified_endgame.json      # 旧 train/curriculum_endgame.py
│   └── p3_mlh.json                  # 旧 train/train_p3_pretrain.py（MLH）
├── engine.py              # Server 的模型插件（六方法 GameEngine + KIT_FACTORY）
├── config.json            # Server 预设（max_mcts / max_t）
├── tests/test_r3.py       # 上面那 17 项
├── docs/
│   ├── architecture.md    # 架构规格
│   └── experiments.md     # 实验与对局记录
├── AGENTS.md              # 面向 agent 的命令、不变量、契约
└── README.md
```

`runs/`（权重）、`logs/`（对局记录）与 `session-*.md` 均被 git 忽略；数据在
`ResNet/data/shards_evals`，由 `Kit/planes19/build` 构建。

`runs/` (checkpoints), `logs/` (match traces, PGNs, metrics) and `session-*.md` (local AI session
exports) are git-ignored.

---

## Quick Start

### UCI Engine

```bash
cd ~/UniChess && /home/jeefy/miniconda3/envs/unichess/bin/python -m Kit uci Transformer/engine.py \
  --preset max_mcts
```

### Training

```bash
cd ~/UniChess && /home/jeefy/miniconda3/envs/unichess/bin/python -m Kit train \
  Transformer/configs/t20m.json
```

中断后用同一命令接着跑（`out` 目录里的 `latest.pt` 记着配置哈希，不符会拒绝续训）。
自对弈换代循环是 `python -m Kit loop Transformer/configs/loop_p4.json`
（P4 口径的唯一配方，说明与换代纪律见 `AGENTS.md` 的「换代循环」一节）。

### Server Service

```bash
systemctl --user status unichess-server
systemctl --user restart unichess-server

# After editing config.json, verify routing:
curl -s http://127.0.0.1:8000/api/models
```

> The Server caches both the preset table and loaded engine weights per process, so **restart
> the service after changing `config.json`** or the old model stays resident.

---

## License

This project is licensed under the MIT License.
