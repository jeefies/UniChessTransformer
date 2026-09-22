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
- **Run Stage P2 Feature Tests** (tree reuse, dynamic FPU, contempt):
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python tests/test_p2_features.py
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
  /home/jeefy/miniconda3/envs/unichess/bin/python eval/puzzle_bench.py --ckpt runs/stratified_p4_selfplay_corrected/best_model.pt --sims 100
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
  /home/jeefy/miniconda3/envs/unichess/bin/python uci.py --ckpt runs/stratified_p4_selfplay_corrected/best_model.pt --mcts-sims 2400 --device cuda --book data/opening_book.bin
  ```
- **Run Self-Play Pipeline (P4, corrected recipe)**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python tools/gumbel_selfplay_corrected.py \
    --ckpt runs/stratified_p1_opening/best_model.pt \
    --num-games 100 \
    --sims 800 \
    --lr 5e-6 \
    --grad-accum 4 \
    --device cuda
  ```
  Use `tools/gumbel_selfplay.py` (original) or `tools/gumbel_selfplay_mixed.py` only for
  reproduction/ablation — the corrected script is the one that produced the 10-0 result.
- **Run Stage P4 Diagnostic Ablations**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python tools/p4_diagnostic_experiments.py
  ```
- **Run Match Against Baseline (`chess_ai v2.0.0`)**:
  ```bash
  /home/jeefy/miniconda3/envs/unichess/bin/python eval/match_baseline.py \
    --ckpt runs/stratified_p4_selfplay_corrected/best_model.pt \
    --games 100 \
    --sims 100 \
    --baseline-seconds 0.25
  ```

## Architecture & Conventions

### Model Tiers
- **Current Best Model**: `runs/stratified_p4_selfplay_corrected/best_model.pt` (Stratified 20M trained with P4 self-play + mixed data, LR=5e-6, grad_accum=4).
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
- **Moves-Left Head (MLH)**: Auxiliary MLP on the `[CLS]` token — `Linear(d_model, 64) -> SiLU -> Linear(64, 1)` — predicting moves remaining. Enabled via `return_mlh=True`; trained with Smooth-L1 (`mlh_weight=0.05`). Present in every expert since Stage P3. **Checkpoints predating P3 lack `mlh_head` keys and fail to load** against the current `StratifiedChessTransformer`.
- **Stratified checkpoint aliasing**: `StratifiedChessTransformer` registers `self.experts = nn.ModuleList([self.opening, self.middlegame, self.endgame])`, so saved state dicts contain *both* `opening/middlegame/endgame.*` and `experts.0/1/2.*` keys for the same modules. Counting raw state-dict entries therefore over-reports params ~2x; the real `stratified_20m` count is **61,031,064**.

### Dataset & Records (`model/dataset.py`)
- Shards are 96-byte structured binary records (`RECORD_DTYPE`) located at `/home/jeefy/UniChess/data/shards_evals/evals_*.bin` (symlinked to `/home/jeefy/UniChess/ResNet/data/shards_evals`).
- Memory-mapped reading (`np.memmap`) with vectorized bitboard decoding on batches.

### MCTS Search Engines
- **C++ MCTS Integration (`search/cpp/`)**:
  - High-performance C++ PyBind11 MCTS extension delivering **6,155+ sims/sec** batched tree search.
  - Native leaf-level Syzygy 3-4-5 tablebase probing: positions with $\le 5$ pieces are probed directly during tree traversal, returning exact game-theoretic values without neural network evaluation.
  - Python MCTS fallback: seamlessly falls back to Python batched MCTS (`search/mcts.py`) if C++ extension compilation/loading is unavailable.
- **Stage P2 search features** (`search/cpp/mcts.hpp`, `mcts_pybind.cpp`):
  - **Tree reuse** (`reuse`): after the opponent replies, descends the retained subtree via `reuse_root(move_uci)` instead of rebuilding, reusing all prior visit counts.
  - **Dynamic FPU** (`c_fpu`, default 0.5): first-play urgency is `parent_q - c_fpu * sqrt(1/(1 + parent_visits))` via `compute_fpu_q`, replacing the old fixed `fpu_reduction = 0.2`.
  - **Contempt** (`contempt`): biases root draws by `contempt / (1 + root_N)` during backup to discourage premature draw acceptance; `0.0` disables.
- **Multi-Process Parallel MCTS (`search/parallel_mcts.py`)**: Lock-free worker processes communicating with central GPU batched evaluator; peaks at **8,820 sims/sec** (16 workers, batch 64).

### Head-to-Head Performance vs Model R
- **Match Result**: Model T won **10-0** against Model R (`chess_ai` / ResNet 15x192) in a 10-game P4 self-play corrected championship match across 5 balanced opening pairs.
- **Configuration**: Model T configured at 2400 MCTS simulations vs Model R at 800 simulations; Model T executed at **0.22s-0.29s/move** (parity with Model R's 0.21s-0.28s/move at 2400 sims due to C++ MCTS throughput).
- **Previous Result**: Model T won **5.5 - 4.5** against Model R in the Stage 4 curriculum match (10 games, 2400 vs 800 sims).

### Self-Play Pipeline (`tools/gumbel_selfplay_corrected.py`)
- **Gumbel AlphaZero self-play RL pipeline** with C++ MCTS tree reuse.
- **Phases**: (1) Self-play data collection, (2) Training on self-play positions, (3) Save updated model.
- **P4 Corrected Recipe**: LR=5e-6, grad_accum=4, 30% mixed real/self-play data.
- `tools/gumbel_selfplay.py` (original, LR=1e-4) regressed to 5.0-5.0 vs Model R — keep it only
  for reproduction. `tools/gumbel_selfplay_mixed.py` is the intermediate mixed-data variant.
  `tools/p4_diagnostic_experiments.py` holds the Exp A-E ablations that isolated the cause.

### Tooling Inventory (`tools/`, `eval/`)
- **Match runners vs Model R**: `run_match_T_vs_R.py`, `run_match_curriculum_T_vs_R.py`,
  `run_match_aligned_curriculum_T_vs_R.py`, `run_match_p3_T_vs_R.py`, `run_match_p4_T_vs_R.py`;
  `eval/match_p1_vs_r.py` for the P1 stage.
- **Benchmarks**: `eval/puzzle_bench.py` (tactical suite), `eval/match_baseline.py` (vs
  `chess_ai v2.0.0`), `eval/arena.py` (round-robin), `benchmark_transformer.py` (throughput),
  `tools/hyperparam_search.py` (MCTS parameter grid search).

### Documentation Map
| File | Purpose |
| :--- | :--- |
| `README.md` | Project overview, features, benchmarks, quick start |
| `docs/architecture.md` | Architecture specification (single source of truth) |
| `docs/experiments.md` | Experimental records & stage-by-stage match history |
| `AGENTS.md` | This file — commands, invariants, contracts |

### Server Integration (`~/UniChess/Server`)
- Model adapter integrated via symlink: `/home/jeefy/UniChess/Server/models/T -> /home/jeefy/UniChess/Transformer`.
- Managed as systemd user service `unichess-server`:
  ```bash
  systemctl --user status unichess-server
  systemctl --user restart unichess-server
  ```
- **`config.json` is the routing contract.** The Server loads it from `models/T/config.json`
  (resolved through the symlink) via `models/__init__.py:resolve_kwargs(model_name, arg_name)`,
  and instantiates `engine.py:GameEngine(**kwargs)`.
  - **`max_mcts` is the only T preset**, and the frontend auto-selects the first preset
    (`list_presets` returns sorted keys, the UI sets no explicit default). So `max_mcts` is
    effectively the production route — keep it pointed at the strongest checkpoint.
  - **All paths in `config.json` must be absolute.** The T engine passes paths straight to
    `Path()`/`torch.load()`, so a relative path resolves against the Server's
    `WorkingDirectory` (`~/UniChess/Server`) and fails. Unlike the R engine, which rebases
    relative paths against `RESNET_ROOT` via `_resolve_path`, T has no such rebasing.
  - **Restart the service after editing `config.json`.** The Server caches the preset table
    (`_config_cache`) and engine weights (`_SHARED_ENGINES`) per process, so a live process
    keeps serving the previously loaded model until restarted.
  - Current routing (2400 sims, batch 64, fp16 on CUDA, leaf Syzygy 3-4-5):
    ```
    $ python -c "import sys; sys.path.insert(0,'.'); \
      from models import resolve_kwargs, list_presets; \
      print(resolve_kwargs('T', list_presets('T')[0])['ckpt'])"
    /home/jeefy/UniChess/Transformer/runs/stratified_p4_selfplay_corrected/best_model.pt
    ```
