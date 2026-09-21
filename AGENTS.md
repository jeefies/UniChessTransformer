# AGENTS.md

## Overview
High-performance neural chess engine combining Transformer backbones with 2D spatial geometric priors, bilinear square-to-square policy heads, Win-Draw-Loss (WDL) value heads, phase-stratified routing, and high-performance batched Monte Carlo Tree Search (MCTS) with C++ acceleration.

## Environment & Python Toolchain
- **Conda Environment**: `/home/jeefy/miniconda3/envs/unichess/bin/python` (Python 3.12, PyTorch 2.x with CUDA 12.8 support). Always use this Python binary; do not assume system `python3` or `pytest`.
- **Hardware Profile**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM).
- **Core Dependencies**: `torch`, `python-chess` (`chess`), `numpy`. Note: `pytest` is not installed in the environment.

## Execution & Verification Commands
- **Run Full Unit Tests (13 tests)**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python tests/test_all.py
  ```
- **Run C++ MCTS Dedicated Test Suite**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python tests/test_cpp_mcts.py
  ```
- **Run Architectural Benchmark (Throughput & Latency)**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python benchmark_transformer.py
  ```
- **Run MCTS Hyperparameter Simulation Search**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python tools/hyperparam_search.py --rounds 5
  ```
- **Run Tactical Puzzle Benchmark**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python eval/puzzle_bench.py --ckpt runs/stratified_middlegame_curriculum/best_model.pt --sims 100
  ```
- **Launch Training**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python -u train/train.py \
    --preset stratified_20m \
    --data-dir /home/jeefy/UniChess/data/shards_evals \
    --checkpoint-dir runs/stratified_20m \
    --batch-size 1024 \
    --precision bf16 \
    --num-workers 4
  ```
- **Launch UCI Engine Interface**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python uci.py --ckpt runs/stratified_middlegame_curriculum/best_model.pt --mcts-sims 2400 --device cuda
  ```
- **Run Match Against Baseline (`chess_ai v2.0.0`)**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python eval/match_baseline.py \
    --ckpt runs/stratified_middlegame_curriculum/best_model.pt \
    --games 100 \
    --sims 100 \
    --baseline-seconds 0.25
  ```

## Architecture & Conventions

### Model Tiers
- **Current Best Model**: `runs/stratified_middlegame_curriculum/best_model.pt` (Stratified 20M trained with middlegame curriculum fine-tuning).
- `stratified_20m` (`StratifiedChessTransformer`): 3 specialized ~20M expert networks dynamically dispatched by phase:
  - Phase 0 (Opening): `piece_count >= 24` or `ply <= 20`
  - Phase 1 (Middlegame): `12 < piece_count < 24`
  - Phase 2 (Endgame): `piece_count <= 12`
- `transformer_20m`: 11 layers, $d_{\text{model}}=384$, 12 heads, SwiGLU $d_{\text{ff}}=1024$ (~20.3M parameters).
- `transformer_50m`: 17 layers, $d_{\text{model}}=512$, 16 heads, SwiGLU $d_{\text{ff}}=1160$ (~49.7M parameters). Deep tier.
- Lightweight presets: `transformer_tiny` (~3.8M), `transformer_small` (~6.7M), `transformer_medium` (~18.5M), `transformer_large` (~35.1M).

### Key Architectural Invariants
- **Input Encoding (`core/encoding.py`)**: Canonical `(19, 8, 8)` float32 representation, oriented to the side to move (mirrored if Black to move).
- **ConvStem + Tokens**: 3x3 Conv maps $(19, 8, 8) \to (d_{\text{model}}, 8, 8)$, flattened to 64 square tokens + 1 prepended `[CLS]` token (total 65 tokens). Decoupled learned 2D rank and file embeddings are added.
- **SDPA Relative Position Bias**: Pairwise $(12, 64, 64)$ square-to-square attention bias is padded with zeros for `[CLS]` to $(1, 12, 65, 65)$ and passed directly into `F.scaled_dot_product_attention(..., attn_mask=attn_bias)` to leverage fused FlashAttention/SDPA kernels. Do not mutate attention matrices in-place with slicing.
- **Bilinear Policy Head**: Projects 64 squares to queries $Q$ and keys $K$, computing move logits via $(Q K^T) / \sqrt{d_p} + \text{bias}_{\text{move}} \in \mathbb{R}^{B \times 64 \times 64}$, flattened to $(B, 4096)$. Decoupled promotion head predicts $(Q, R, B, N)$ for promotions.
- **WDL Value Head**: MLP on `[CLS]` token predicting 3 classes: `[P(Win), P(Draw), P(Loss)]`. Scalar $Q = P(\text{Win}) - P(\text{Loss}) \in [-1, 1]$.

### Dataset & Records (`model/dataset.py`)
- Shards are 96-byte structured binary records (`RECORD_DTYPE`) located at `/home/jeefy/UniChess/data/shards_evals/evals_*.bin` (symlinked to `/home/jeefy/UniChess/ResNet/data/shards_evals`).
- Memory-mapped reading (`np.memmap`) with vectorized bitboard decoding on batches.

### MCTS Search Engines
- **C++ MCTS Integration (`search/cpp/`)**:
  - High-performance C++ PyBind11 MCTS extension delivering **6,155+ sims/sec** batched tree search.
  - Native leaf-level Syzygy 3-4-5 tablebase probing: positions with $\le 5$ pieces are probed directly during tree traversal, returning exact game-theoretic values without neural network evaluation.
  - Python MCTS fallback: seamlessly falls back to Python batched MCTS (`search/mcts.py`) if C++ extension compilation/loading is unavailable.
- **Multi-Process Parallel MCTS (`search/parallel_mcts.py`)**: Lock-free worker processes communicating with central GPU batched evaluator.

### Head-to-Head Performance vs Model R
- **Match Result**: Model T won **5.5 - 4.5** against Model R (`chess_ai` / ResNet 15x192) in a 10-game championship match across 5 balanced opening pairs.
- **Configuration**: Model T configured at 2400 MCTS simulations vs Model R at 800 simulations; Model T executed at **0.34s/move** (~1.9x faster than Model R's 0.65s/move) due to C++ MCTS throughput.

### Server Integration (`~/UniChess/Server`)
- Model adapter integrated via symlink: `/home/jeefy/UniChess/Server/models/T -> /home/jeefy/UniChess/Transformer`.
- Server configuration (`config.json`): uses `runs/stratified_middlegame_curriculum/best_model.pt` at `max_mcts` tier (2400 simulations, batch size 64, fp16 on CUDA, leaf Syzygy 3-4-5 tablebase).
- Managed as systemd user service `unichess-server`:
  ```bash
  systemctl --user status unichess-server
  systemctl --user restart unichess-server
  ```
