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

- **C++ Accelerated MCTS (`search/cpp/`)** — production search path
  - PyBind11 extension delivering **6,155+ sims/sec** of batched tree search.
  - **Native leaf-level Syzygy probing**: nodes with $\le 5$ pieces are probed against the 3-4-5 tablebases during traversal, returning exact game-theoretic values with no neural evaluation.
  - **Tree reuse** (Stage P2): descends the retained subtree after the opponent's reply, reusing visit counts.
  - **Dynamic FPU** (Stage P2): `parent_q - c_fpu * sqrt(1/(1 + parent_visits))` replaces the fixed first-play-urgency reduction.
  - **Contempt** (Stage P2): biases root draws by `contempt / (1 + root_N)` to discourage premature draws.
  - **Python fallback**: `search/mcts.py` (batched PUCT, virtual loss, Dirichlet noise) takes over automatically if the C++ extension is unavailable.
  - **Multi-process parallel MCTS** (`search/parallel_mcts.py`): lock-free workers over a central GPU evaluator, peaking at **8,820 sims/sec** (16 workers, batch 64).

- **Training & Curriculum Learning**
  - Distillation from Stockfish evaluations with joint policy cross-entropy + WDL loss (+ MLH since P3).
  - Phase-stratified curriculum fine-tuning over 64 binary evaluation shards
    (`/home/jeefy/UniChess/data/shards_evals`), 96-byte fixed records, `np.memmap` + vectorized
    bitboard decoding.
  - **Gumbel AlphaZero self-play** with C++ MCTS tree reuse (`tools/gumbel_selfplay_corrected.py`).
  - Current best: `runs/stratified_p4_selfplay_corrected/best_model.pt`.

- **Standards & Server Integration**
  - Full UCI protocol compliance (`uci.py`).
  - Server integration via symlink: `/home/jeefy/UniChess/Server/models/T -> /home/jeefy/UniChess/Transformer`.
  - Presets in `config.json` drive the Server's model selection; the single `max_mcts` preset is
    the frontend default. **All paths in `config.json` must be absolute** — unlike the R engine,
    the T engine does not rebase relative paths against its repo root.
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
# Full unit test suite (13 test functions)
/home/jeefy/miniconda3/envs/unichess/bin/python tests/test_all.py

# Dedicated C++ MCTS suite (perft, legal-move parity, 19-plane parity, stress)
/home/jeefy/miniconda3/envs/unichess/bin/python tests/test_cpp_mcts.py

# Stage P2 features (tree reuse, dynamic FPU, contempt)
/home/jeefy/miniconda3/envs/unichess/bin/python tests/test_p2_features.py

# Throughput / latency benchmark
/home/jeefy/miniconda3/envs/unichess/bin/python benchmark_transformer.py
```

---

## Directory Structure

```
UniChessTransformer/
├── core/                    # Bitboards, 19-plane encoder, move indexing
│   ├── encoding.py          # 19-plane canonical encoding & symmetry transforms
│   └── moves.py             # Move indexing (4096 square-to-square + promotions)
├── model/                   # Neural network implementations
│   ├── transformer.py       # ConvStem, 2D relative bias attention, bilinear policy,
│   │                        #   WDL + MLH heads, StratifiedChessTransformer
│   ├── dataset.py           # Vectorized binary shard dataset & dataloaders
│   └── loss.py              # Multi-task policy + WDL + MLH loss
├── search/                  # Search algorithms
│   ├── cpp/                 # C++ MCTS with leaf Syzygy probing (PyBind11)
│   │   ├── chess_board.hpp  # Fast C++ bitboard move generator + 19-plane encoder
│   │   ├── mcts.hpp         # Batched search, tree reuse, dynamic FPU, contempt
│   │   └── mcts_pybind.cpp  # PyBind11 bindings
│   ├── mcts.py              # Python PUCT MCTS w/ virtual loss & Dirichlet (fallback)
│   └── parallel_mcts.py     # Multi-process batched GPU evaluation search
├── engine/                  # Engine wrappers
│   └── engine.py            # TransformerEngine: model + search + tablebase + book
├── engine.py                # UniChess Server GameEngine adapter (shared-weight singleton)
├── train/                   # Training pipeline & curriculum fine-tuning
│   ├── train.py             # Distributed/AMP training script
│   ├── train_p3_pretrain.py # Stage P3 pretraining with the MLH head
│   ├── curriculum_opening.py    # Stage P1 opening-expert fine-tuning
│   ├── curriculum_middlegame.py # Middlegame curriculum fine-tuning
│   └── curriculum_endgame.py    # Endgame curriculum fine-tuning
├── eval/                    # Benchmark & evaluation harnesses
│   ├── arena.py             # Automated round-robin & head-to-head runner
│   ├── match_baseline.py    # Match harness vs chess_ai baseline
│   ├── match_p1_vs_r.py     # Stage P1 match vs Model R
│   └── puzzle_bench.py      # Tactical puzzle suite evaluator
├── tools/                   # Self-play, match runners, optimization
│   ├── hyperparam_search.py          # Parallel grid search for MCTS parameters
│   ├── p4_diagnostic_experiments.py  # Stage P4 self-play ablations (Exp A-E)
│   ├── gumbel_selfplay.py            # Gumbel AlphaZero self-play (original)
│   ├── gumbel_selfplay_mixed.py      # Self-play with mixed real/self-play data
│   ├── gumbel_selfplay_corrected.py  # Self-play, corrected recipe (LR 5e-6) — the winning run
│   ├── run_match_T_vs_R.py                 # Baseline match vs Model R
│   ├── run_match_curriculum_T_vs_R.py      # Stage 4 curriculum match
│   ├── run_match_aligned_curriculum_T_vs_R.py
│   ├── run_match_p3_T_vs_R.py              # Stage P3 match
│   └── run_match_p4_T_vs_R.py              # Stage P4 match
├── tests/                   # Test suites
│   ├── test_all.py          # Full unit test suite (13 tests)
│   ├── test_cpp_mcts.py     # Dedicated C++ MCTS verification suite
│   ├── test_p2_features.py  # Stage P2 search feature tests
│   └── bench_p2.py          # Stage P2 benchmark helper
├── data/
│   └── opening_book.bin     # Polyglot opening book
├── docs/
│   ├── architecture.md      # Architecture specification
│   └── experiments.md       # Experimental records
├── uci.py                   # UCI entrypoint
├── benchmark_transformer.py # Throughput / latency benchmark
├── config.json              # Presets consumed by the UniChess Server (absolute paths)
├── AGENTS.md                # Agent-facing commands, invariants, contracts
└── README.md
```

`runs/` (checkpoints), `logs/` (match traces, PGNs, metrics) and `session-*.md` (local AI session
exports) are git-ignored.

---

## Quick Start

### UCI Engine

```bash
/home/jeefy/miniconda3/envs/unichess/bin/python uci.py \
  --ckpt runs/stratified_p4_selfplay_corrected/best_model.pt \
  --mcts-sims 2400 \
  --device cuda \
  --book data/opening_book.bin
```

### Self-Play RL (Stage P4 corrected recipe)

```bash
/home/jeefy/miniconda3/envs/unichess/bin/python tools/gumbel_selfplay_corrected.py \
  --ckpt runs/stratified_p1_opening/best_model.pt \
  --num-games 100 \
  --sims 800 \
  --lr 5e-6 \
  --grad-accum 4 \
  --device cuda
```

### Training

```bash
/home/jeefy/miniconda3/envs/unichess/bin/python -u train/train.py \
  --preset stratified_20m \
  --data-dir /home/jeefy/UniChess/data/shards_evals \
  --checkpoint-dir runs/stratified_20m \
  --batch-size 1024 \
  --precision bf16 \
  --num-workers 4
```

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
