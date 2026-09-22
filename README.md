# UniChessTransformer

**UniChessTransformer** (Model T) is a state-of-the-art neural chess engine combining Transformer backbones with 2D spatial geometric priors, bilinear square-to-square policy heads, Win-Draw-Loss (WDL) value heads, phase-stratified routing, and high-performance batched Monte Carlo Tree Search (MCTS) accelerated by C++.

UniChessTransformer's flagship model (`runs/stratified_p4_selfplay_corrected/best_model.pt`) achieved a perfect **10-0** clean sweep against Model R (`chess_ai` / ResNet 15x192) in a 10-game compute-aligned championship match at 2400 MCTS simulations, running at **0.22s-0.29s/move**.

---

## Key Features

- **Geometric Transformer Architecture**:
  - **ConvStem (19 -> d_model)**: Maps 19 canonical bitboard planes directly into 64 spatial square tokens.
  - **Decoupled 2D Embeddings & Relative Attention Bias**: Pairs learned rank/file embeddings with learned pairwise 64x64 relative position biases injected into scaled dot-product attention (SDPA / FlashAttention).
  - **Bilinear Square-to-Square Policy Head**: Projects square representations into origin queries (Q in R^{64x64}) and destination keys (K in R^{64x64}), enforcing chessboard geometric constraints while avoiding parameter explosion.
  - **WDL Value Head**: Directly predicts Win, Draw, and Loss probabilities via global [CLS] token representation.

- **Model Hierarchy & Phase-Stratified Routing**:
  - **`stratified_20m`** (~60.8M params total, 3 x 11 layers): Dynamic phase-stratified routing dispatching positions across 3 specialized ~20M expert networks:
    - **Opening Expert**: `piece_count >= 24` or `ply <= 20`
    - **Middlegame Expert**: `12 < piece_count < 24` (further refined with curriculum fine-tuning)
    - **Endgame Expert**: `piece_count <= 12` (curriculum fine-tuned with Syzygy guidance)
  - **`transformer_20m`** (~20.3M params, 11 layers, d=384, 12 heads): Fast inference single-backbone model.
  - **`transformer_50m`** (~49.7M params, 17 layers, d=512, 16 heads): Deep monolithic transformer tier.
  - Lightweight presets (`transformer_tiny`, `transformer_small`, `transformer_medium`, `transformer_large`).

- **C++ Accelerated MCTS Engine (`search/cpp/`)**:
  - High-performance PyBind11 C++ MCTS extension delivering **6,155+ sims/sec** batched tree search.
  - **Native Leaf-Level Syzygy Probing**: Traversal directly probes 3-4-5 piece Syzygy tablebases at leaf nodes, returning exact game-theoretic values without GPU inference.
  - **Pure Python Fallback**: Seamless fallback to batched Python MCTS (`search/mcts.py`) if the C++ extension is uncompiled.
  - **Multi-Process Parallel MCTS (`search/parallel_mcts.py`)**: Lock-free worker architecture delivering up to **8,820 sims/sec** (16 workers, batch size 64).

- **Training & Curriculum Learning**:
  - Distillation from Stockfish evaluations with joint policy softmax cross-entropy and WDL loss.
  - Phase-stratified curriculum training on 64 binary shards (`/home/jeefy/UniChess/data/shards_evals`).
  - Current best model: `runs/stratified_p4_selfplay_corrected/best_model.pt`.

- **Self-Play Pipeline (`tools/gumbel_selfplay.py`)**:
  - Gumbel AlphaZero self-play RL pipeline with C++ MCTS tree reuse.
  - 3 phases: (1) Self-play data collection via C++ MCTS, (2) Training on self-play positions, (3) Save updated model.
  - P4 corrected configuration: LR=5e-6, grad_accum=4, 30% mixed real/self-play data.

- **Standards & Server Integration**:
  - Full UCI protocol compliance (`uci.py`) supporting standard chess GUIs.
  - Server integration via symlink: `/home/jeefy/UniChess/Server/models/T -> /home/jeefy/UniChess/Transformer`.
  - Production service managed under systemd user service `unichess-server`.

---

## Architecture Overview

```
                      Canonical Board Input (19x8x8)
                                    |
                          ConvStem (3x3, stride 1)
                                    |
                     64 Square Tokens + [CLS] Token
                                    |
                 + Rank / File Embeddings & 2D Rel Bias
                                    |
                   Transformer Encoder Layers (Pre-LN)
                 - FlashAttention / SDPA with Pairwise Bias
                 - SwiGLU / GeLU Feed-Forward MLP
                                    |
             +---------------------------------------+
             |                                       |
   Bilinear Policy Head                        WDL Value Head
   (Origin Queries x Dest Keys)               (MLP on [CLS] Token)
             |                                       |
   4096 Move Logits + Promotion Logits     Win / Draw / Loss Probabilities
```

---

## Head-to-Head Benchmark Results

### P4 Self-Play Championship Match vs Model R (`chess_ai` / ResNet 15x192)
Evaluated across 5 balanced opening pairs (Italian, Ruy Lopez, Scotch, Four Knights, Petroff):

| Matchup | Model T (P4 Self-Play Corrected) | Model R (ResNet 15x192) |
| :--- | :---: | :---: |
| **Search Engine** | **C++ MCTS + Leaf Syzygy (2400 sims)** | Python MCTS + Syzygy (800 sims) |
| **Final Score** | **10.0 / 10 (100.0%)** | 0.0 / 10 (0.0%) |
| **Game Record** | **10 Wins, 0 Draws, 0 Losses** | 0 Wins, 0 Draws, 10 Losses |
| **Average Move Latency** | **0.22s - 0.29s / move** | 0.21s - 0.28s / move |

### Curriculum Match vs Model R (10 Games)
| Matchup | Model T (Stratified 20M Curriculum) | Model R (ResNet 15x192) |
| :--- | :---: | :---: |
| **Search Engine** | **C++ MCTS + Leaf Syzygy (2400 sims)** | Python MCTS + Syzygy (800 sims) |
| **Final Score** | **5.5 / 10 (55.0%)** | 4.5 / 10 (45.0%) |
| **Game Record** | **3 Wins, 5 Draws, 2 Losses** | 2 Wins, 5 Draws, 3 Losses |
| **Average Move Latency** | **0.34s / move** (~1.9x faster) | 0.65s / move |

### Match vs Baseline Engine (`chess_ai v2.0.0` CNN + Negamax)
- **Score**: **18.5 / 20 (92.5%)** undefeated (17 wins, 3 draws, 0 losses, +436.4 Elo).
- **Latency**: 12.1 ms/move with C++ MCTS (100 sims) vs 268.3 ms for baseline.

---

## Verification & Test Suite

The test suite covers full correctness and stability across Python and C++ components:

```bash
# Run full unit tests (13 test functions)
/home/jeefy/miniconda3/envs/unichess/bin/python tests/test_all.py

# Run dedicated C++ MCTS test suite (Perft, legal moves, 19-plane parity, stress test)
/home/jeefy/miniconda3/envs/unichess/bin/python tests/test_cpp_mcts.py
```

---

## Directory Structure

```
UniChessTransformer/
├── core/                   # Bitboard representations, 19-plane encoder, move mapping
│   ├── encoding.py         # 19-plane canonical board encoding & symmetry transforms
│   └── moves.py            # Move indexing (4096 square-to-square + promotion encoding)
├── model/                  # Neural network implementations
│   ├── transformer.py      # ConvStem, 2D relative bias attention, Bilinear policy, WDL head
│   ├── dataset.py          # Vectorized binary shard dataset & dataloaders
│   └── loss.py             # Multi-task policy distillation + WDL loss
├── search/                 # Search algorithms
│   ├── cpp/                # C++ MCTS implementation with leaf Syzygy probing (PyBind11)
│   │   ├── chess_board.hpp # Fast C++ bitboard move generator & 19-plane encoder
│   │   ├── mcts.hpp        # Batched C++ tree search engine & leaf tablebase hook
│   │   └── mcts_pybind.cpp # PyBind11 bindings for Python integration
│   ├── mcts.py             # Python PUCT MCTS with virtual loss & Dirichlet noise (fallback)
│   └── parallel_mcts.py    # Multi-process batched GPU evaluation search engine
├── engine/                 # UCI & high-level engine wrappers
│   └── engine.py           # Unified engine interface with C++ MCTS & tablebase support
├── train/                  # Training pipeline & curriculum fine-tuning
│   ├── train.py            # Distributed/AMP training script
│   ├── curriculum_middlegame.py # Middlegame curriculum fine-tuning
│   └── curriculum_endgame.py    # Endgame curriculum fine-tuning
├── eval/                   # Benchmark and evaluation scripts
│   ├── arena.py            # Automated round-robin & head-to-head match runner
│   ├── match_baseline.py   # Baseline match harness against chess_ai
│   └── puzzle_bench.py     # Lichess/curated tactical puzzle suite evaluator
├── tools/                  # Analysis & hyperparameter optimization
│   ├── hyperparam_search.py# Parallel Bayesian/Grid search for MCTS parameters
│   └── gumbel_selfplay.py  # Gumbel AlphaZero self-play RL pipeline
├── tests/                  # Test suites
│   ├── test_all.py         # Full unit test suite (13 tests)
│   └── test_cpp_mcts.py    # Dedicated C++ MCTS verification suite
├── uci.py                  # Universal Chess Interface (UCI) entrypoint
├── config.json             # Engine configuration for deployment
└── README.md
```

---

## Quick Start

### Running the UCI Engine
```bash
/home/jeefy/miniconda3/envs/unichess/bin/python uci.py \
  --ckpt runs/stratified_p4_selfplay_corrected/best_model.pt \
  --mcts-sims 2400 \
  --device cuda
```

### Running Self-Play Pipeline
```bash
/home/jeefy/miniconda3/envs/unichess/bin/python tools/gumbel_selfplay.py \
  --ckpt runs/stratified_p1_opening/best_model.pt \
  --num-games 100 \
  --sims 800 \
  --lr 5e-6 \
  --grad-accum 4 \
  --device cuda
```

### Running Server Service
```bash
# Verify systemd service status
systemctl --user status unichess-server

# Restart server
systemctl --user restart unichess-server
```

---

## License

This project is licensed under the MIT License.
