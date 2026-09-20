# UniChessTransformer

**UniChessTransformer** is a state-of-the-art neural chess engine combining Transformer backbones with 2D spatial geometric priors, bilinear square-to-square policy heads, Win-Draw-Loss (WDL) value heads, phase-stratified routing, and high-throughput multi-process batched Monte Carlo Tree Search (MCTS).

UniChessTransformer achieves **84.5% win rate (71 wins, 27 draws, 2 losses, +295 Elo)** against baseline `chess_ai v2.0.0` over a 100-game match under matched time controls.

---

## Key Features

- **Geometric Transformer Architecture**:
  - **ConvStem (19 $\to d_{\text{model}}$)**: Maps 19 canonical bitboard planes directly into 64 spatial square tokens.
  - **Decoupled 2D Embeddings & Relative Attention Bias**: Pairs learned rank/file embeddings with learned pairwise $64 \times 64$ relative position biases injected into scaled dot-product attention (SDPA / FlashAttention).
  - **Bilinear Square-to-Square Policy Head**: Projects square representations into origin queries ($Q \in \mathbb{R}^{64 \times 64}$) and destination keys ($K \in \mathbb{R}^{64 \times 64}$), enforcing chessboard geometric constraints while avoiding parameter explosion.
  - **WDL Value Head**: Directly predicts Win, Draw, and Loss probabilities via global `[CLS]` token representation.

- **Model Hierarchy & Tiers**:
  - **`transformer_20m`** (~20.3M params, 11 layers, $d=384$, 12 heads): Primary competitive engine, delivering fast inference and grandmaster-level positional evaluation.
  - **`transformer_50m`** (~49.7M params, 17 layers, $d=512$, 16 heads): Flagship deep model for deep tactical computation.
  - **`stratified_20m`** (~60.8M params, 3 $\times$ 11 layers): Dynamic phase-stratified routing (Opening, Middlegame, Endgame) dispatching positions to specialized phase experts.
  - Lightweight presets (`transformer_tiny`, `transformer_small`, `transformer_medium`) for fast CPU inference and experimentation.

- **High-Performance Parallel MCTS**:
  - Multi-process lock-free tree search with leaf batched evaluation on GPU.
  - Up to **8,820 simulations/second** (16 CPU workers, batch size 64 on RTX 5070 Ti).
  - Virtual loss, PUCT formula with exploration tuning, Dirichlet root noise, and Syzygy endgame tablebase probing.

- **Training Pipeline**:
  - Distillation from Stockfish MultiPV evaluations with custom joint loss (Softmax Cross-Entropy policy loss + WDL Cross-Entropy / MSE value loss).
  - Vectorized binary shard decoding with PyTorch bfloat16 AMP mixed precision and fused AdamW.

- **Standards & Deployment**:
  - Full UCI protocol compliance (`uci.py`) compatible with standard GUIs (Arena, Cutechess, Banksia, Lichess bots).
  - Integrated into FastAPI dual-engine web service with human-vs-AI board interface.

---

## Architecture Overview

```
                      Canonical Board Input (19x8x8)
                                    │
                         ConvStem (3x3, stride 1)
                                    │
                    64 Square Tokens + [CLS] Token
                                    │
               + Rank / File Embeddings & 2D Rel Bias
                                    │
                 Transformer Encoder Layers (Pre-LN)
               - FlashAttention / SDPA with Pairwise Bias
               - SwiGLU / GeLU Feed-Forward MLP
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                               │
  Bilinear Policy Head                                WDL Value Head
(Origin Queries x Dest Keys)                       (MLP on [CLS] Token)
            │                                               │
4096 Move Logits + Promotion Logits              Win / Draw / Loss Probabilities
```

---

## Benchmark & Match Results

### 100-Game Match vs `chess_ai v2.0.0`
Tested over 100 games across 25 diverse opening systems (1.e4, 1.d4, 1.c4, 1.Nf3):

| Engine | Score | Wins | Draws | Losses | Win Rate | Elo Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UniChessTransformer (20M)** | **84.5 / 100** | **71** | **27** | **2** | **84.5%** | **+294.6 ± 94.1** |
| `chess_ai v2.0.0` (CNN + Negamax) | 15.5 / 100 | 2 | 27 | 71 | 15.5% | Reference |

- **Decisive Finishes**: 73% of games concluded in checkmate.
- **Robustness**: Only 2 losses across 100 competitive games under matched clock conditions.

### Parallel MCTS Throughput (RTX 5070 Ti + 20 CPU Cores)

| Workers | Batch 64 | Batch 128 | Batch 256 | Scaling vs 1 Worker |
| :---: | :---: | :---: | :---: | :---: |
| **1 Worker** | 283.4 sims/s | 285.4 sims/s | 276.3 sims/s | 1.0x |
| **4 Workers** | 2,971.9 sims/s | 3,095.5 sims/s | 2,806.3 sims/s | 10.8x |
| **8 Workers** | 5,846.5 sims/s | 5,139.4 sims/s | 5,032.3 sims/s | 20.6x |
| **16 Workers** | **8,820.2 sims/s** | 6,986.7 sims/s | 6,738.4 sims/s | **31.1x** |
| **32 Workers** | 6,453.5 sims/s | 5,778.6 sims/s | 5,965.5 sims/s | 22.8x |

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
│   ├── mcts.py             # PUCT MCTS with virtual loss & Dirichlet noise
│   └── parallel_mcts.py    # Multi-process batched GPU evaluation search engine
├── engine/                 # UCI & high-level engine wrappers
│   └── engine.py           # Unified engine interface with tablebase support
├── train/                  # Training pipeline
│   └── train.py            # Distributed/AMP training script
├── eval/                   # Benchmark and evaluation scripts
│   ├── arena.py            # Automated round-robin & head-to-head match runner
│   ├── match_baseline.py   # Baseline match harness against chess_ai
│   └── puzzle_bench.py     # Lichess/curated tactical puzzle suite evaluator
├── tools/                  # Analysis & hyperparameter optimization
│   └── hyperparam_search.py# Parallel Bayesian/Grid search for MCTS parameters
├── tests/                  # Unit tests for encoding, models, and search
│   └── test_all.py         # Full test suite
├── docs/                   # Detailed specifications & experiment logs
│   ├── architecture.md     # Mathematical & structural design specification
│   └── experiments.md      # Comprehensive experimental records & metrics
├── uci.py                  # Universal Chess Interface (UCI) entrypoint
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone git@github.com:jeefies/UniChessTransformer.git
cd UniChessTransformer

# Create and activate environment
conda create -n unichess python=3.12 -y
conda activate unichess

# Install PyTorch with CUDA support and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install python-chess numpy
```

### Running the UCI Engine

```bash
python uci.py --weights path/to/model.pt --preset transformer_20m --sims 800 --device cuda
```

### Running Tests

```bash
python -m unittest discover -s tests
```

### Training

```bash
python -m train.train \
    --preset transformer_20m \
    --data-dir /path/to/binary_shards \
    --batch-size 512 \
    --lr 1e-3 \
    --amp bf16 \
    --output-dir runs/transformer_20m
```

### Running Tactical Benchmark

```bash
python -m eval.puzzle_bench --weights path/to/model.pt --sims 100
```

---

## License

This project is licensed under the MIT License.
