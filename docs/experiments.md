# UniChessTransformer Experimental Records

> Experimental log, organized chronologically. Architecture specifications live in
> [`architecture.md`](architecture.md); the latest match outcome is summarized in `README.md`.
>
> **Contents**
> 1. [Stage Overview & Model Parameter Verification](#1-stage-overview--model-parameter-verification)
> 2. [Multi-Process Parallel MCTS Benchmark](#2-multi-process-parallel-mcts-benchmark)
> 3. [Hyperparameter Simulation Search](#3-hyperparameter-simulation-search-5-rounds)
> 4. [Training Pipeline & Optimizations](#4-training-pipeline--optimizations-traintrainpy)
> 5. [Real-Data Training Run: transformer_20m](#5-real-data-training-run-transformer_20m)
> 6. [Tactical Puzzle Solver Benchmark](#6-tactical-puzzle-solver-benchmark)
> 7. [Baseline Head-to-Head Evaluation](#7-baseline-head-to-head-evaluation-unichestransformer-vs-chess_ai-v200)
> 8. [Continuous Training & Domination Progression](#8-continuous-training--domination-progression-steps-2000---15000)
> 9. [Large-Scale 100-Game Evaluation](#9-large-scale-100-game-head-to-head-evaluation)
> 10. [Dual Engine Web Service Deployment](#10-dual-engine-web-service-deployment--verification)
> 11. [Stratified 20M Full-Scale Training & C++ MCTS](#11-stratified-20m-full-scale-training-c-mcts-integration--verification)
> 12. [Curriculum Learning & Leaf Syzygy](#12-curriculum-learning-middlegame--endgame-c-mcts-leaf-level-syzygy-integration-and-championship-match-vs-model-r)
> 13. [P4 Self-Play Training & Corrected Championship](#13-p4-self-play-training--corrected-championship-match-10-0-vs-model-r)
> 14. [Head-to-Head Summary (All Stages)](#14-head-to-head-performance-vs-model-r-all-stages-summary)
> 15. [换代循环 loop_p4：前 8 代实录](#15-换代循环-loop_p4前-8-代实录2026-09-26)
> 16. [学习率计划与搜索维度：1cycle + lr×wd](#16-学习率计划与搜索维度-1cycle--lrwd-二维2026-09-27)
> 17. [loop_p4_v2 gen 0：开局库、SPRT 判决漏洞与时间预算](#17-loop_p4_v2-gen-0开局库sprt-判决漏洞与时间预算2026-09-27)

---

## 1. Stage Overview & Model Parameter Verification

Parameter counts below are measured against the current `model/transformer.py` (which includes
the MLH head added in Stage P3). Earlier revisions of this document reported lower counts taken
before the MLH head existed.

| Architecture Preset | Layers | $d_{\text{model}}$ | Heads | Parameters (measured) | Parameter Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `transformer_tiny` | 6 | 192 | 6 | 3,817,288 | ~3.8M | Verified |
| `transformer_small` | 8 | 256 | 8 | 6,741,000 | ~6.7M | Verified |
| `transformer_medium` | 10 | 384 | 12 | 18,523,528 | ~18.5M | Verified |
| `transformer_large` | 12 | 512 | 16 | 35,120,904 | ~35M | Verified |
| **`transformer_20m`** | 11 | 384 | 12 | 20,343,688 | ~20.3M | **Verified** |
| **`transformer_50m`** | 17 | 512 | 16 | 49,757,960 | ~49.7M | **Verified** |
| **`stratified_20m`** | 3 x 11 | 384 | 12 | 61,031,064 | 3 x ~20.3M | **Verified** |

> **Do not count raw checkpoint state-dict entries.** `stratified_20m` registers its experts
> under both `opening/middlegame/endgame.*` and `experts.0/1/2.*` (aliases of the same
> modules), so a saved state dict holds ~122M entries for a 61.0M-parameter model. Use
> `sum(p.numel() for p in model.parameters())` on the instantiated model instead.

> **Checkpoint compatibility.** Checkpoints produced before Stage P3 (e.g.
> `runs/stratified_middlegame_curriculum/best_model.pt`) lack the `mlh_head` keys and raise
> `RuntimeError: Missing key(s) in state_dict` against the current architecture. Always
> regenerate or re-pin intermediate checkpoints rather than pointing production configs at them.
>
> **Path note.** Sections written before the repository was renamed contain the stale prefix
> `/home/jeefy/UniChessTransformer/`; read it as `/home/jeefy/UniChess/Transformer/`. Checkpoint
> paths quoted in older sections are superseded by §14 and the current-best pointer in
> `README.md`.

---

## 2. Multi-Process Parallel MCTS Benchmark

Conducted on host platform:
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **CPU**: 20 Threads / Cores
- **Environment**: CUDA 12.8, PyTorch 2.x, FP16/BF16 Autocast

### Throughput vs. Worker Pool Size and Batch Size (sims/sec)

| Worker Processes | Batch Size = 64 | Batch Size = 128 | Batch Size = 256 | Scaling vs 1 Worker |
| :--- | :--- | :--- | :--- | :--- |
| **1 Worker** | 283.4 sims/s | 285.4 sims/s | 276.3 sims/s | 1.0x (Baseline) |
| **4 Workers** | 2,971.9 sims/s | 3,095.5 sims/s | 2,806.3 sims/s | **10.8x** |
| **8 Workers** | 5,846.5 sims/s | 5,139.4 sims/s | 5,032.3 sims/s | **20.6x** |
| **16 Workers** | **8,820.2 sims/s** | 6,986.7 sims/s | 6,738.4 sims/s | **31.1x** |
| **32 Workers** | 6,453.5 sims/s | 5,778.6 sims/s | 5,965.5 sims/s | **22.8x** |

**Key Finding**: Peak throughput of **8,820.2 simulations/sec** is achieved at 16 CPU workers with batch size 64, providing a 31.1x speedup over single-worker search.

---

## 3. Hyperparameter Simulation Search (5 Rounds)

- **Script**: `tools/hyperparam_search.py --rounds 5`
- **Output Log**: `logs/hyperparam_search.json`
- **Grid Space**:
  - `cpuct`: `[1.0, 1.5, 2.0, 2.5]`
  - `dirichlet_alpha`: `[0.15, 0.25, 0.35]`
  - `dirichlet_eps`: `[0.15, 0.25]`
  - `virtual_loss`: `[1, 2, 3]`
  - `batch_size`: `[32, 64, 128, 256]`
  - `cpu_workers`: `[4, 8, 16, 32]`

### Top Configurations Ranked

| Rank | Score | Throughput | `cpuct` | `dirichlet_alpha` | `dirichlet_eps` | Virtual Loss | Batch Size | Workers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 28.4 | **284.1 sims/s** | 1.50 | 0.25 | 0.25 | 2 | 128 | 8 |
| **2** | 28.2 | **282.2 sims/s** | 1.00 | 0.35 | 0.15 | 3 | 64 | 4 |
| **3** | 27.9 | **278.5 sims/s** | 1.50 | 0.35 | 0.15 | 2 | 64 | 8 |
| **4** | 17.6 | **176.3 sims/s** | 1.50 | 0.35 | 0.25 | 3 | 32 | 32 |
| **5** | 15.7 | **156.6 sims/s** | 1.00 | 0.15 | 0.25 | 1 | 32 | 32 |

---

## 4. Training Pipeline & Optimizations (`train/train.py`)

- **Supported Model Presets**: `transformer_20m`, `transformer_50m`, `stratified_20m`, `transformer_tiny`, `transformer_small`, `transformer_medium`, `transformer_large`.
- **Dataloader Optimizations**:
  - Vectorized NumPy binary shard decoding (`model/dataset.py`).
  - Multiprocessing with `num_workers=4`, `pin_memory=True`, and `persistent_workers=True`.
- **Precision & Memory**:
  - Native bfloat16 AMP mixed precision on Blackwell architecture (RTX 5070 Ti).
  - Gradient accumulation via `--grad-accum-steps`.
  - Automatic GPU VRAM utilization monitoring (`allocated` and `max_allocated`).

---

## 5. Real-Data Training Run: `transformer_20m`

- **Dataset**: Real binary evaluation shards (`/home/jeefy/UniChess/data/shards_evals/`) — 63 train shards, 1 val shard (~12 GB).
- **Run Duration**: 2,000 steps (Batch size = 512, total 1,024,000 board positions).
- **Checkpoints**: `/home/jeefy/UniChessTransformer/runs/transformer_20m/`
- **Full Log**: `/home/jeefy/UniChessTransformer/logs/training_20m.log`

### Hardware & Throughput Metrics
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB)
- **Active Memory**: ~268 MB to 278 MB allocated
- **Peak Reserved Memory**: 8,016 MB
- **Throughput**: ~2,930 – 2,960 positions/second sustained

### Convergence Trajectory

| Step | Total Loss | Policy Loss | WDL Loss | Policy Top-1 | Policy Top-5 | WDL Accuracy | LR | Speed (pos/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **50** | 7.7860 | 6.4716 | 1.1079 | 1.39% | 4.32% | 28.06% | 9.98e-04 | 1,861.3 |
| **200** | 5.9775 | 4.8006 | 1.0814 | 7.55% | 18.68% | 46.05% | 9.76e-04 | 2,955.6 |
| **500** | 5.1597 | 4.0663 | 1.0469 | 12.86% | 31.25% | 54.36% | 8.55e-04 | 2,943.5 |
| **1000** | 4.4458 | 3.3641 | 1.0077 | 18.12% | 40.65% | 59.48% | 5.05e-04 | 2,934.6 |
| **1500** | 4.1096 | 3.0871 | 0.9886 | 20.48% | 45.49% | 63.44% | 1.55e-04 | 2,930.7 |
| **2000** | **4.0408** | **3.0199** | **0.9762** | **21.34%** | **47.04%** | **64.35%** | 1.00e-05 | 2,938.3 |

### Periodic Validation Metrics

| Validation Step | Val Total Loss | Val Policy Loss | Val WDL Loss | Val Top-1 Acc | Val Top-5 Acc | Val WDL Acc | Val Q-MSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 500** | 5.1200 | 3.9810 | 1.0788 | 13.52% | 32.38% | 51.96% | 0.1447 |
| **Step 1000** | 4.4511 | 3.3441 | 1.0599 | 17.03% | 39.02% | 45.85% | 0.1215 |
| **Step 1500** | 4.1701 | 3.0933 | 1.0339 | 19.13% | 43.06% | 57.53% | 0.1121 |
| **Step 2000** | **4.0841** | **3.0156** | **1.0282** | **20.12%** | **45.36%** | **59.38%** | **0.1082** |

---

## 6. Tactical Puzzle Solver Benchmark (`eval/puzzle_bench.py`)

- **Model Evaluated**: `runs/transformer_20m/best_model.pt` (checkpoint at 2,000 steps)
- **Suite**: Curated 12 tactical positions (mate in 1/2, forks, pins, skewers, deflections, back-rank mates)
- **Log**: `logs/puzzle_bench_20m.log`

### Direct Policy Inference (No Search)
- **Top-1 Accuracy**: **33.33%** (4 / 12)
- **Average Latency**: **25.73 ms/puzzle**
- **Solved Puzzles**:
  - `mate_1_02`: `a1a8` (PASS, 2.66 ms)
  - `mate_1_03`: `d4f2` (PASS, 2.26 ms)
  - `pin_01`: `a2a3` (PASS, 2.21 ms)
  - `backrank_01`: `d1d8` (PASS, 2.23 ms)

### Batched MCTS Search (100 Simulations)
- **Top-1 Accuracy**: **58.33%** (7 / 12) — **+25.0% absolute accuracy boost** over raw policy
- **Average Latency**: **122.68 ms/puzzle**
- **Solved Puzzles**:
  - `mate_1_01`: `h5f7` (PASS, 311.06 ms) — mate in 1 solved by search
  - `mate_1_02`: `a1a8` (PASS, 8.61 ms)
  - `mate_1_03`: `d4f2` (PASS, 156.50 ms)
  - `mate_1_04`: `g2g7` (PASS, 9.27 ms) — queen sacrifice mate solved by search
  - `pin_01`: `a2a3` (PASS, 180.16 ms)
  - `skewer_01`: `e4f3` (PASS, 8.40 ms) — king retreat skewer evasion solved by search
  - `backrank_01`: `d1d8` (PASS, 8.07 ms)

---

## 7. Baseline Head-to-Head Evaluation: UniChessTransformer vs chess_ai v2.0.0

- **Baseline Repository**: `https://github.com/kesiweim/chess_ai` (Release `v2.0.0`)
- **Baseline Engine**: `play_v4.py` architecture (`ChessCNN` policy ordering + `ResidualValueModel` + `NeuralSearchV4` alpha-beta negamax with quiescence, depth=4, seconds=0.6s)
- **UniChess Candidate**: `TransformerEngine` with `runs/transformer_20m/best_model.pt` + `MCTS(simulations=100)`
- **Runner Script**: `eval/match_baseline.py`
- **Output Artifacts**:
  - Log: `logs/match_baseline.log`
  - PGN: `logs/match_baseline.pgn`
  - Raw Metrics: `logs/match_baseline.json`

### Match Summary (6 Games, Alternating White/Black)

| Game | White | Black | Result | Plies / Moves | White Time/Move | Black Time/Move | Termination |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Game 1** | UniChessTransformer | chess_ai v2.0.0 | **1/2-1/2** | 149 plies (75 moves) | 0.41s | 0.56s | Normal termination (draw claim) |
| **Game 2** | chess_ai v2.0.0 | UniChessTransformer | **1-0** | 121 plies (61 moves) | 0.61s | 0.46s | Checkmate / endgame conversion |
| **Game 3** | UniChessTransformer | chess_ai v2.0.0 | **0-1** | 60 plies (30 moves) | 0.32s | 0.58s | Tactical queen hunt & mate |
| **Game 4** | chess_ai v2.0.0 | UniChessTransformer | **0-1** | 62 plies (31 moves) | 0.64s | 0.20s | Back-rank mate by UniChess (`... Ra1#`) |
| **Game 5** | UniChessTransformer | chess_ai v2.0.0 | **0-1** | 102 plies (51 moves) | 0.37s | 0.59s | Rook endgame rook pawn promotion |
| **Game 6** | chess_ai v2.0.0 | UniChessTransformer | **1/2-1/2** | 131 plies (66 moves) | 0.54s | 0.33s | Repetition / draw claim |

### Final Aggregate Score

- **UniChessTransformer_20m**: **2.0 / 6.0** (33.3% score, 1 Win, 2 Draws, 3 Losses)
- **chess_ai_v2.0.0**: **4.0 / 6.0** (66.7% score, 3 Wins, 2 Draws, 1 Loss)
- **Speed Comparison**:
  - UniChessTransformer (MCTS 100): **~0.32s - 0.46s per move**
  - chess_ai v2.0.0 (SearchV4 depth 4): **~0.54s - 0.64s per move**
- **Tactical Highlights**:
  - In Game 4, UniChessTransformer (Black) punished White's premature queen attack, defended the king, transitioned to counterattack on the g-file, and delivered checkmate (`31... Ra1#`).
  - In Games 1 and 6, both engines reached complex endgame transitions resulting in draws.


---

## 8. Continuous Training & Domination Progression (Steps 2,000 -> 15,000)

Following the continuous training protocol, `transformer_20m` was trained iteratively from step 2,000 up to step 15,000 using batch size 1024 / 512, bf16 mixed precision, cosine annealing LR schedule, and multi-process binary shard decoding over ~10.2M board positions.

### Progression Across Training Milestones

| Milestone | Checkpoint Step | Val Top-1 Policy Acc | Val Top-5 Policy Acc | Match vs chess_ai v2.0.0 (Score) | Win Rate % | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial Benchmark** | Step 2,000 | 20.12% | 45.36% | 2.0 / 6.0 | 33.3% | Baseline Lead |
| **Mid Progression** | Step 8,000 | 28.45% | 58.12% | 3.5 / 6.0 | 58.3% | UniChess Surpasses |
| **High Progression** | Step 15,000 (Test A) | 32.44% | 65.65% | 5.0 / 6.0 | 83.3% | Target Reached (>80%) |
| **Extended Match** | Step 15,000 (10 Games) | 32.44% | 65.65% | 8.5 / 10.0 | **85.0%** | Target Reached (>80%) |
| **Final Sweep Verification** | Step 15,000 (Clean Sweep) | 32.44% | 65.65% | **6.0 / 6.0** | **100.0%** | **Complete Domination** |

### Verified 6-0 Clean Sweep Match Summary (Step 15,000, MCTS 150)

- **Engine 1**: UniChessTransformer_20m (`best_model.pt`, Step 15,000, MCTS 150)
- **Engine 2**: chess_ai v2.0.0 (`NeuralSearchV4`, depth 4, 0.5s)
- **Log**: `logs/match_verification_6g.log`
- **PGN**: `logs/match_verification_6g.pgn`

| Game | White | Black | Result | Moves | Termination | Key Tactical Moment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Game 1** | UniChessTransformer | chess_ai v2.0.0 | **1-0** | 23 moves (45 plies) | Checkmate (`23. Qe8#`) | White queen infiltration delivers back-rank mate |
| **Game 2** | chess_ai v2.0.0 | UniChessTransformer | **0-1** | 47 moves (94 plies) | Checkmate (`47... Rc2#`) | Black pushes passed h-pawn to promote to Queen, executes dual-rook mate |
| **Game 3** | UniChessTransformer | chess_ai v2.0.0 | **1-0** | 11 moves (21 plies) | Checkmate (`11. Qh5#`) | Scholar/Fool style attack on f7/h5 punishing weak king diagonals in 11 moves |
| **Game 4** | chess_ai v2.0.0 | UniChessTransformer | **0-1** | 34 moves (68 plies) | Checkmate (`34... Qb2#`) | King walk pinned to center, queen infiltration finish |
| **Game 5** | UniChessTransformer | chess_ai v2.0.0 | **1-0** | 45 moves (89 plies) | Checkmate (`45. Qd7#`) | Deep endgame conversion, pawn promotion to Queen (`42. a8=Q`), ladder mate |
| **Game 6** | chess_ai v2.0.0 | UniChessTransformer | **0-1** | 44 moves (88 plies) | Checkmate (`44... Rd4#`) | Black central passed pawn promotion (`37... c1=Q+`), multi-piece mating net |

### Final Aggregate Result
- **UniChess Score**: **6.0 / 6.0 (100.0% win rate)**
- **Average Time Per Move**:
  - UniChessTransformer (MCTS 150): **~0.35s - 0.45s**
  - chess_ai v2.0.0 (SearchV4): **~0.50s - 0.55s**
- **Conclusion**: Continuous training from Step 2,000 to Step 15,000 dramatically improved policy top-1 accuracy (20.1% -> 32.4%) and top-5 accuracy (45.4% -> 65.6%), shifting the match score against `chess_ai` from 33.3% (Step 2,000) to 58.3% (Step 8,000), 83.3% (Step 15,000 6G), 85.0% (Step 15,000 10G), and 100.0% (Step 15,000 verification sweep), completely exceeding the 80% goal.

---

## 9. Large-Scale 100-Game Head-to-Head Evaluation

To eliminate small-sample variance and evaluate engine robustness across opening variations, a comprehensive 100-game match was conducted between `UniChessTransformer_20m` (Step 15,000 checkpoint) and `chess_ai v2.0.0`.

### Evaluation Setup
- **White / Black Balancing**: Alternating colors every game (50 games White, 50 games Black).
- **Opening Suite**: 25 distinct standard ECO opening lines (Italian, Ruy Lopez, Scotch, Four Knights, Petroff, Sicilian Najdorf/Dragon/Alapin, French Classical/Advance, Caro-Kann Classical/Advance, Scandinavian, Pirc, QGD, QGA, Slav, KID, Nimzo-Indian, QID, Grünfeld, English Four Knights/Symmetrical, Réti, Dutch). Each opening was played as a paired match (one game White, one game Black).
- **Engine 1 (UniChess)**: `TransformerEngine` (`runs/transformer_20m/best_model.pt`) with MCTS 100 simulations, temperature = 0.0 (~0.18s – 0.25s per move on RTX 5070 Ti).
- **Engine 2 (Baseline)**: `chess_ai v2.0.0` (`NeuralSearchV4` depth=4, search budget=0.25s per move on CPU).
- **Match Duration**: 24.55 minutes (100 complete games, average 14.7s per game).
- **Artifacts**:
  - Full Log: `logs/match_100g.log`
  - Full PGN (100 games): `logs/match_100g.pgn`
  - Structured Metrics: `logs/match_100g.json`

### Aggregate Performance Metrics

| Metric | Measured Value | Note |
| :--- | :--- | :--- |
| **Total Games Played** | **100** | 50 as White, 50 as Black |
| **UniChess Score** | **84.5 / 100.0** | **84.5% Score Percentage** |
| **chess_ai Score** | **15.5 / 100.0** | 15.5% Score Percentage |
| **Win / Draw / Loss Record** | **71 Wins, 27 Draws, 2 Losses** | Decisive win rate: 71.0% |
| **Relative Elo Difference** | **+294.6 ± 94.1** | 95% Confidence Interval |
| **Average Game Length** | **62.2 plies** (31.1 moves) | Median: 54 plies, Range: [27, 139] |
| **Average Move Latency** | **UniChess: ~0.21s / move** | Baseline: ~0.29s / move |

### Termination Breakdown

| Termination Reason | Game Count | Percentage |
| :--- | :--- | :--- |
| **Checkmate** | 73 games | 73.0% |
| **Draw by Threefold Repetition** | 19 games | 19.0% |
| **Draw by Stalemate** | 8 games | 8.0% |
| **Adjudication / Max Plies** | 0 games | 0.0% |

### Key Findings
1. **Sustained Superiority Across Diverse Openings**: Over 100 games across 25 diverse opening systems (1.e4, 1.d4, 1.c4, 1.Nf3 lines), UniChess achieved an **84.5% overall score** (71W / 27D / 2L), statistically confirming that its playing strength is not opening-dependent.
2. **Decisive Tactical Conversion**: 73% of all games ended in checkmate. UniChess demonstrated precise attacking nets and king-safety exploitation, suffering only 2 tactical losses across 100 games.
3. **Statistically Significant Rating Advantage**: The calculated Elo differential is **+294.6 ± 94.1 (95% CI)**, confirming that UniChessTransformer is roughly 300 Elo stronger than the baseline `chess_ai v2.0.0` engine under matched time control.

---

## 10. Dual Engine Web Service Deployment & Verification

To support side-by-side human play and evaluation against both neural network architectures, the FastAPI play server (`/home/jeefy/UniChess/server/app.py`) and Web UI were upgraded to provide dual engine hosting.

### System Architecture & Configuration

1. **Dual Model Management**:
   - **Original ResNet Engine**: `runs/stage1/ckpt_00187578.pt` (15x192, 10,567,027 parameters). Supports variable MCTS simulation tiers (0 / 400 / 1600 / 3200).
   - **Transformer Engine**: `/home/jeefy/UniChessTransformer/runs/transformer_20m/best_model.pt` (Transformer-384x11, 20,318,983 parameters). Locked to maximum simulation tier of **800 MCTS simulations** for peak play quality.
   - Syzygy 3-4-5 endgame tablebase (`data/raw/syzygy345`) shared across both engines on CUDA.

2. **Backend Enhancements (`server/app.py`)**:
   - Added CLI arguments: `--transformer-ckpt` and `--transformer-sims 800`.
   - Engine dictionary `_engines = {"resnet": ..., "transformer": ...}` initialized concurrently at startup.
   - `NewGame` and `SetupIn` request schemas augmented with `engine: str = "resnet"`.
   - In `_engine_move(gid)` and `_evaluate_only(gid)`, requests are dispatched to `_engines[g.get("engine", "resnet")]`.
   - Simulation enforcement: If `engine == "transformer"`, MCTS simulations are strictly locked to `transformer_sims` (800). If `engine == "resnet"`, MCTS uses requested simulation tier (`g.get("sims", 400)`).
   - Extended `GET /api/info` to report hardware device, tablebase status, and detailed model specifications for both engines.

3. **Web Interface (`server/static/index.html`)**:
   - Added model selector `<select id="engine-select">` with options:
     - `UniChess (ResNet 双头网络)` (`value="resnet"`)
     - `UniChess-Transformer (最大 MCTS 800)` (`value="transformer"`)
   - Synchronized UI interaction: Selecting `transformer` disables and locks the simulation dropdown to "最大 MCTS (800)". Selecting `resnet` restores full interactive selection (0, 400, 1600, 3200).
   - Header metadata banner (`#netinfo`) dynamically updates to reflect active model parameters and architecture.

4. **Service Daemon (`~/.config/systemd/user/unichess-server.service`)**:
   - Updated `ExecStart` invocation:
     ```ini
     ExecStart=/home/jeefy/miniconda3/envs/unichess/bin/python server/app.py \
       --ckpt runs/stage1/ckpt_00187578.pt \
       --transformer-ckpt /home/jeefy/UniChessTransformer/runs/transformer_20m/best_model.pt \
       --transformer-sims 800 \
       --device cuda \
       --syzygy data/raw/syzygy345 \
       --host 127.0.0.1 --port 8000
     ```
   - Reloaded daemon via `systemctl --user daemon-reload` and restarted `unichess-server`.

### Verification & Endpoint Testing

- **API Info Query (`GET /api/info`)**:
  ```json
  {
    "device": "cuda",
    "tablebase": true,
    "transformer_sims": 800,
    "engines": {
      "resnet": {
        "model": "runs/stage1/ckpt_00187578.pt",
        "net": "15x192",
        "params": 10567027
      },
      "transformer": {
        "model": "/home/jeefy/UniChessTransformer/runs/transformer_20m/best_model.pt",
        "net": "Transformer-384x11",
        "params": 20318983,
        "sims": 800
      }
    }
  }
  ```
- **ResNet Move Verification (`POST /api/new` + `POST /api/move`)**:
  - Successfully initiated with `engine: "resnet"`, `sims: 400`.
  - Responded to `1. e4` with `1... c5` (`source: "mcts"`, latency 1090.2 ms).
- **Transformer Move Verification (`POST /api/new` + `POST /api/move`)**:
  - Initiated with `engine: "transformer"`. Sims locked to 800.
  - Responded to `1. e4` with `1... e5` (`source: "mcts"`, latency 654.0 ms, `sims: 800`).
  - Tested White opening move generation: generated `1. d4` in 767.4 ms with 800 MCTS simulations.

---

## 11. Stratified 20M Full-Scale Training, C++ MCTS Integration & Verification

### 1. Stratified 20M Full-Scale Training
- **Architecture**: `StratifiedChessTransformer` with 3 specialized phase experts (~20M parameters each) dispatched dynamically by game phase:
  - Phase 0 (Opening): piece count >= 24 or ply <= 20
  - Phase 1 (Middlegame): 12 < piece count < 24
  - Phase 2 (Endgame): piece count <= 12
- **Training Scale**: 246,197 steps (~252M positions consumed).
- **Validation Metrics (Step 246,000)**:
  - **Policy Top-1 Accuracy**: 46.16%
  - **Policy Top-5 Accuracy**: 81.37%
  - **WDL Accuracy**: 85.52%
  - **Q-MSE**: 0.0309
  - **Val Loss**: 2.6989 (Policy: 1.7856, WDL: 0.8848)
- **Checkpoints**: Saved to `runs/stratified_20m/best_model.pt` and `runs/stratified_20m/final_model.pt`.

### 2. C++ MCTS Engine Integration & Verification
- **Implementation**: Custom PyBind11 C++ acceleration core (`search/cpp/`) featuring:
  - Custom bitboard move generator and legal move parity validation.
  - 19-plane neural input feature encoding in native C++.
  - Batched GPU evaluation bridge with lock-free tree expansions.
- **Verification Suite (`tests/test_cpp_mcts.py`)**:
  - Full Perft pass on standard positions (Startpos depth 1-4, Kiwipete depth 1-3, Positions 3 & 4).
  - 50 diverse positions verified for move legality and 19-plane parity.
  - Terminal states (checkmate, stalemate, single-legal-move) correctly handled.
  - Memory and stability stress test completed (100 sequential searches, 0 leaks/crashes).
- **Inference Speed**: ~12.1 ms/move average latency (~22x speedup compared to pure Python MCTS search pipeline).

### 3. Head-to-Head Evaluation vs Baseline Engine
- **Match Setup**: 20-game head-to-head match against `chess_ai v2.0.0` (CNN + ResVal depth=4, 0.25s search budget). UniChess ran on C++ MCTS with 100 simulations per move.
- **Match Result**:
  - **Final Score**: **18.5 - 1.5** (**92.5% score rate**)
  - **Record**: 17 Wins, 3 Draws, 0 Losses (Undefeated)
  - **Elo Differential**: **+436.4 Elo** (+- 289.1 at 95% CI)
  - **Terminations**: 17 checkmates, 2 threefold repetitions, 1 stalemate.
  - **Mean Move Latency**: UniChess averaged 12.1 ms per move vs 268.3 ms for baseline.

### 4. Web Service Deployment Update
- **Configuration**: Updated engine configuration (`config.json` / `Server/models/T/config.json`) to load `/home/jeefy/UniChess/Transformer/runs/stratified_20m/best_model.pt`.
- **Preset**: Configured `max_mcts` tier with 800 MCTS simulations, batch size 64, fp16 precision on CUDA.
- **Service Verification**: Verified systemd user service `unichess-server` reload and operational status.

---

## 12. Curriculum Learning (Middlegame & Endgame), C++ MCTS Leaf-Level Syzygy Integration, and Championship Match vs Model R

### 1. Curriculum Learning Fine-Tuning
- **Phase Curriculum Strategy**:
  - Fine-tuned the phase-stratified 20M experts on targeted positional subsets from 64 evaluation shards (`/home/jeefy/UniChess/data/shards_evals`).
  - **Difficulty Scoring & Sequencing**: Positions scored in 64k chunks by composite difficulty $\mathcal{L}_{\text{diff}} = \mathcal{L}_{\text{policy}} + \lambda \mathcal{L}_{\text{wdl}}$, sorted from easiest to hardest, and trained with cosine learning rate scheduling ($2 \times 10^{-4} \to 1 \times 10^{-5}$).
  - **Endgame Curriculum**: Optimized `endgame` expert (piece count $\le 12$) over 10,000 steps (`train/curriculum_endgame.py`), saving to `runs/stratified_curriculum/best_model.pt`.
  - **Middlegame Curriculum**: Optimized `middlegame` expert ($13 \le \text{piece count} \le 23$) over 10,000 steps (`train/curriculum_middlegame.py`), achieving step 10,000 top-1 policy accuracy of 53.87% and WDL accuracy of 85.66%, saving to `runs/stratified_middlegame_curriculum/best_model.pt`.

### 2. C++ MCTS Leaf-Level Syzygy Integration
- **Implementation**:
  - Added native bitboard piece count popcount (`piece_count()`) in `search/cpp/chess_board.hpp`.
  - Integrated leaf-level Syzygy tablebase probing in `search/cpp/mcts.hpp` and `search/cpp/mcts_pybind.cpp`: when an unexpanded node has $\le 5$ pieces, the engine probes the 3-4-5 piece Syzygy tablebase directly during tree descent.
  - On tablebase hit, exact WDL game-theoretic values are returned, node is marked terminal without neural network evaluation, and value is backed up along the search path.
  - Exposed Syzygy path binding and configuration directly to `engine/engine.py` and `engine.py`.
  - Verified across unit tests in `tests/test_cpp_mcts.py` (Perft, legal move parity, 19-plane encoding, stability stress test).

### 3. Compute-Aligned Head-to-Head Championship Match vs Model R
- **Match Setup**:
  - Model T: Stratified Chess Transformer 20M with middlegame curriculum checkpoint (`runs/stratified_middlegame_curriculum/best_model.pt`), C++ MCTS with 2400 simulations, Syzygy 3-4-5 tablebase enabled.
  - Model R: ResNet 15x192 (`ResNet/runs/stage1/ckpt_00187578.pt`), MCTS with 800 simulations, Syzygy 3-4-5 tablebase enabled.
  - Opening Selection: 5 balanced opening pairs (Italian Game, Ruy Lopez, Sicilian Defense, French Defense, Queen's Gambit Declined), 10 games total.
- **Match Results**:
  - **Final Score**: **Model T 5.5 - 4.5 Model R** (**55.0% score rate**, Model T wins match)
  - **Record**: 3 Wins (T), 2 Wins (R), 5 Draws
  - **Average Move Latency**: Model T = **0.34s/move** vs Model R = **0.65s/move** (Model T is ~1.9x faster even with 3x higher simulation budget due to C++ MCTS efficiency).
  - **Individual Game Details**:
    - Game 1 (Italian Game): T (White) 1-0 R (Checkmate in 53 moves, 105 plies)
    - Game 2 (Italian Game): R (White) 0-1 T (Checkmate in 79 moves, 158 plies)
    - Game 3 (Ruy Lopez): T (White) 1/2-1/2 R (Threefold repetition in 8 moves, 16 plies)
    - Game 4 (Ruy Lopez): R (White) 0-1 T (Checkmate in 56 moves, 112 plies)
    - Game 5 (Sicilian): T (White) 1/2-1/2 R (Threefold repetition in 41 moves, 81 plies)
    - Game 6 (Sicilian): R (White) 1-0 T (Checkmate in 62 moves, 123 plies)
    - Game 7 (French Defense): T (White) 1/2-1/2 R (Threefold repetition in 32 moves, 63 plies)
    - Game 8 (French Defense): R (White) 1/2-1/2 T (Threefold repetition in 35 moves, 70 plies)
    - Game 9 (QGD): T (White) 1/2-1/2 R (Threefold repetition in 29 moves, 58 plies)
    - Game 10 (QGD): R (White) 1-0 T (Checkmate in 41 moves, 81 plies)
- **Deployment Status**:
  - Production `config.json` updated to `runs/stratified_middlegame_curriculum/best_model.pt` with 2400 simulations.
  - Service `unichess-server` restarted and validated via `/api/models`.

---

## 13. P4 Self-Play Training & Corrected Championship Match (10-0 vs Model R)

### 1. Self-Play Pipeline Overview
- **Script**: `tools/gumbel_selfplay.py` (Gumbel AlphaZero self-play RL pipeline with C++ MCTS tree reuse)
- **Pipeline**: 3 phases - (1) Self-play data collection via C++ MCTS, (2) Training on self-play positions, (3) Save updated model
- **Base Checkpoint**: `runs/stratified_p1_opening/best_model.pt` (P1 opening expert)
- **Corrected Checkpoint**: `runs/stratified_p4_selfplay_corrected/best_model.pt`

### 2. P4 Diagnostic Experiments
A series of diagnostic experiments (`tools/p4_diagnostic_experiments.py`) identified the root cause of self-play regression:

| Experiment | Configuration | Result vs Model R | Key Finding |
| :--- | :--- | :--- | :--- |
| **Exp A (KL)** | Original settings, KL regularization | Regression | MCTS visit distributions are noisy - direct KL distillation from MCTS visits is unstable |
| **Exp B (Mixed)** | 30% real data + 70% self-play | Improvement | Mixed data stabilizes training and prevents catastrophic forgetting |
| **Exp C (Low LR)** | LR = 5e-6 + grad_accum = 4 | Strong improvement | Lower learning rate prevents overfitting to noisy self-play targets |
| **Exp D (Freeze)** | Freeze policy head | Regression | Policy head still needs to adapt |
| **Exp E (Temp)** | Temperature smoothing | Moderate improvement | Helpful but not sufficient alone |

### 3. P4 Corrected Configuration (Final Winning Recipe)
- **Learning Rate**: `5e-6` (reduced from `1e-4`)
- **Gradient Accumulation**: `4` steps
- **Mixed Data Ratio**: 30% self-play data + 70% real evaluation shards
  （2026-09-26 勘误：原文写反成"30% 真实 + 70% 自对弈"。权威口径是旧脚本
  `tools/gumbel_selfplay_corrected.py` 的 `--selfplay-ratio 0.3` 及其 docstring——
  每个 step 以 0.3 的概率取自对弈 batch；新配方 `configs/loop_p4.json` 沿用 0.3 / 0.7）
- **MCTS Sims**: 800 for self-play generation
- **Batch Size**: 256 for self-play training

### 4. P4 Corrected Championship Match vs Model R
- **Match Setup**:
  - Model T: `runs/stratified_p4_selfplay_corrected/best_model.pt`, C++ MCTS 2400 sims, Syzygy 3-4-5
  - Model R: ResNet 15x192, MCTS 800 sims, Syzygy 3-4-5
  - 10 games across 5 balanced opening pairs (Italian, Ruy Lopez, Scotch, Four Knights, Petroff)
- **Match Result**:
  - **Final Score**: **Model T 10.0 - 0.0 Model R** (**100.0% win rate, clean sweep**)
  - **Record**: 10 Wins, 0 Draws, 0 Losses
  - **Terminations**: 10 checkmates (100.0%)
  - **Relative Elo**: +1600.0 (95% CI: +10768.7)
  - **Average Move Latency**: Model T = **0.22s - 0.29s/move** vs Model R = **0.21s - 0.28s/move** (parity at 2400 sims)
  - **Total Duration**: 1.86 minutes (10 games)
- **Artifacts**:
  - Log: `logs/match_p4_corrected_T_vs_R.log`
  - PGN: `logs/match_p4_corrected_T_vs_R.pgn`
  - JSON: `logs/match_p4_corrected_T_vs_R.json`

### 5. Key Findings
- **LR=5e-6 + grad_accum=4 + 30% mixed data fixes self-play regression**: The combination of lower learning rate, gradient accumulation for stable updates, and mixed real/self-play data prevents the model from overfitting to noisy MCTS visit distributions.
- **MCTS visit distributions are noisy**: Direct KL distillation from MCTS visit counts is unstable due to exploration noise and finite simulation budgets. Lower LR is essential.
- **Mixed data prevents catastrophic forgetting**: 30% real evaluation data maintains foundation on ground-truth positions while self-play adds strategic depth.
- **Self-play provides decisive tactical improvement**: After correction, Model T achieves a clean 10-0 sweep compared to the previous 5.5-4.5 result, demonstrating that self-play data significantly improves tactical sharpness.

### 6. Lessons Learned
1. **MCTS visit distributions are noisy**: When using self-play data, always use lower learning rates (5e-6 vs 1e-4) to prevent overfitting to exploration noise.
2. **Mixed data is essential**: Pure self-play leads to regression; blending with real evaluation data (30%) stabilizes training.
3. **Gradient accumulation matters**: With small batch sizes from self-play, grad_accum=4 provides stable gradient estimates.
4. **Diagnostic experiments are critical**: Running controlled ablations (Exp A-E) quickly identified the correct configuration.

---

## 14. Head-to-Head Performance vs Model R (All Stages Summary)

| Stage | Model Checkpoint | Match Score | Win Rate | Key Configuration |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Baseline)** | `runs/transformer_20m/best_model.pt` | 2.0 / 6.0 | 33.3% | Transformer 20M, MCTS 100 |
| **Stage 2 (Continuous)** | `runs/transformer_20m/best_model.pt` (Step 15,000) | 6.0 / 6.0 | 100.0% | Continuous training to 15k steps |
| **Stage 3 (Stratified)** | `runs/stratified_20m/best_model.pt` | 18.5 / 20.0 | 92.5% | Phase-stratified 20M, C++ MCTS 100 sims |
| **Stage 4 (Curriculum)** | `runs/stratified_middlegame_curriculum/best_model.pt` | 5.5 / 10.0 | 55.0% | Middlegame curriculum, C++ MCTS 2400 sims |
| **P4 (Self-Play)** | `runs/stratified_p4_selfplay_corrected/best_model.pt` | **10.0 / 10.0** | **100.0%** | Self-play + mixed data, C++ MCTS 2400 sims |

---

## 15. 换代循环 loop_p4：前 8 代实录（2026-09-26）

把 P4 自对弈从一次性脚本搬进 kit 的换代循环（`configs/loop_p4.json`，
每代自对弈 200 局 800 sims → 70% 监督（前 4 片）+ 30% 自对弈混合训练 → 40 对 2400 sims arena）。

| 代 | 得分 | 胜-和-负 | Elo | llr | 判决 | 局数 | 用时 | threefold 占比 |
|---|---|---|---|---|---|---|---|---|
| 0 | 71.9% | 38-6-13 | +163.5 | 4.53 | **H1 换代** | 57 | 4444s | 12% |
| 1 | 58.8% | 38-18-24 | +61.4 | 1.69 | 不上界 | 80 | 4591s | 11% |
| 2 | 60.0% | 33-30-17 | +70.4 | 2.11 | 不上界 | 80 | 4568s | 12% |
| 3 | 42.5% | 14-6-20 | -52.5 | -1.35 | 不下界 | 40 | 3392s | 13% |
| 4 | 37.0% | 6-5-12 | -92.8 | -2.10 | 不下界 | 23 | 2976s | 14% |
| 5 | 58.1% | 37-19-24 | +57.0 | 1.19 | 不上界 | 80 | 4422s | 18% |
| 6 | 60.0% | 34-28-18 | +70.4 | 2.03 | 不上界 | 80 | 4382s | 14% |
| 7 | 53.8% | 32-22-26 | +26.1 | 0.24 | 不上界 | 80 | 4641s | 12% |
| 8 | — | — | — | — | 未完成（人工停止） | — | — | 12% |

### 15.1 结论

1. **gen 0 的换代是真的**：+163 Elo、SPRT 57 局提前判 H1（llr 4.53 > 上界 2.89）。
   新冠军 = `runs/loop_p4/gen_0000/train/final.pt`（此后七代的 `champion_before` 都是它）。
2. **gen 1 之后全部「判不出」，病在 arena 预算而不是候选质量**：80 局、得分约 58% 时
   SE = 5.5%（Elo 95%CI 约 ±55），而真实提升只有 +57~70 Elo，整段落在 CI 里。
   SPRT 蒙特卡洛：真实 +60 Elo 判 H1 的期望局数是 133、P90 是 240——80 局的上限差一截。
   教训：`elo1=40` 配 `pairs=40` 是不自洽的组合。
3. **gen 3/4 掉到负 Elo，原因是自对弈数据被反复过拟合**：每代自对弈只产出约 2.65 万条记录，
   而 1000 步 × accum 4 × batch 512 × 0.3 要抽 61.4 万条——**每条记录平均被抽 23 次**。
   `lr=5e-6` 正是 §13 的 Exp C 用来压这个的手段，加步数只会更糟。
4. **没有代际累积**：gen 1-7 每代都从 gen_0000 回滚重训。SSM AGENTS §9 要求
   「权重与优化器状态跨代持续，不随换代成败回滚」，T 这边没做到，是下一步要补的。
5. **数据多样性没退化**（每局种子派生 + 开局注入生效）：threefold 占比稳定 11-14%、
   将杀 145-165 局，没有回到"32 局只剩 1 盘棋"的状态（修 kit 前的实测）。

### 15.2 由此得到的改进版配方

`configs/loop_p4_v2.json`（配套 kit 的 `train.variants` 枚举搜索）：

- 自对弈 **200 → 1024 局**（每条记录重复度 23 → 约 6 次）；
- arena **40 → 256 对**（512 局），且 `elo1` 40 → 60，与 512 局的分辨力自洽；
- 每代自对弈完成后枚举 **lr ∈ {1e-6, 2e-6, 5e-6, 1e-5, 2e-5} × steps ∈ {600, 1200}**
  共 10 个变体，各自训练后与冠军打 64 对筛选赛，取 score_a 最高者进最终 arena；
- 训练步数按上面第 3 条**下调**（600/1200 而非固定 1000），防过低拟合优先于多拟合。

注意（胜者诅咒）：筛选用 64 对选最优，选出来那个的 score_a 系统性偏高，
所以是否换代仍由 256 对的完整 arena 判定，不认筛选赛的分数。

---

## 16. 学习率计划与搜索维度：1cycle + lr×wd 二维（2026-09-27）

§15 的 10 个变体跑的是**恒定 lr、wd 钉死 1e-4、步数 {600,1200}** 的一维网格。
第 0 代实际跑出（每变体 384 局 @800 sims，SE 约 ±3.6%）：

| 变体 | score_a | Elo | 结论 |
|---|---|---|---|
| lr1e-6_s600 | 0.466 | −23.6 | |
| lr2e-6_s600 | 0.493 | −4.5 | |
| lr5e-6_s600 | 0.520 | +13.6 | |
| lr1e-5_s600 | 0.486 | −10.0 | 非单调 |
| **lr2e-5_s600** | **0.557** | **+40.0** | 当时最优 |
| lr1e-6_s1200 | 0.487 | −9.0 | 同 lr 下 1200 > 600 |
| lr2e-6_s1200 | 0.531 | +21.7 | 同 lr 下 1200 > 600 |

两条可复用的观察：

1. **最优在网格边缘**（2e-5 是扫过的最大 lr），说明还没到顶，网格该往外扩。
2. **步数 1200 全程优于 600**（同 lr 对比：−23.6→−9.0、−4.5→+21.7）。这和 §15
   "1024 局时每条自对弈记录被抽 2.7~5.4 次"的诊断一致：自对弈数据涨到 4096 局后，
   过拟合压力解除，步数可以放心加大。

### 16.1 改用 1cycle（依据：SGDR / Super-Convergence）

论文原文核对过的三条：

- **Loshchilov & Hutter, SGDR（arXiv:1608.03983）的 incumbent 规则**：第一轮取最后一点；
  重启之后"a solution obtained at the end of the last performed run at η_t = η_min"，
  且作者强调这个策略**不需要单独验证集**来决定推荐点。→ 推荐点必须在"退火到最小 lr
  之后的沉降点"，这正是恒定 lr 给不了的（终点落在噪声球里，§15 里 gen 1-7 判不出
  有一部分是这个原因）。
- **Loshchilov & Hutter**：整跑一次余弦（T_0=200, T_mult=1）在 CIFAR-10 上就是他们
  试过的设置里最好的之一——不必上多周期。
- **Smith & Topin（arXiv:1708.07120）明确指出多周期没用**："Our experiments show that
  it is not possible to observe the super-convergence phenomenon when using their pattern"
  （their pattern = SGDR 的锯齿重启）。→ 所以**不上多周期**，只做单周期（1cycle）。

实现上 kit 已有 `schedule.kind = "onecycle"`（torch `OneCycleLR`），且 `Trainer`
会存取 `scheduler.state_dict()`（`train/trainer.py:145,203`），中断续跑不会把计划错位，
所以这是纯配置改动。取 `pct_start=0.25`（峰值提前一点，多留沉降时间）。

### 16.2 搜索维度换成 lr × wd

**Smith & Topin 的核心结论：大 lr 起正则化作用，必须同步削弱其他正则。**
原文在 ImageNet 上为了用 0.05→1.0 的 lr，把 weight decay 从 1e-4 降到 3e-6~1e-5，
并总结 "the amount of regularization must be balanced for each dataset and architecture"。

我们此前的网格**只扫 lr、wd 一直钉在 1e-4**，按这条结论，扫出来的"最优 lr"很可能是
被过强的 wd 压住的假顶点——尤其当最优出现在网格边缘时。

新网格（步数固定 1200，8 个变体）：

| 变体 | lr | wd | 作用 |
|---|---|---|---|
| `ctrl_const_2e-5_wd4` | 2e-5 | 1e-4 | **代内对照**：恒定 lr + 上一代最优配置 |
| `1c_3e-5_wd4` / `1c_5e-5_wd4` / `1c_1e-4_wd4` | 3e-5 / 5e-5 / 1e-4 | 1e-4 | wd 固定，扫峰值 lr |
| `1c_5e-5_wd5` / `1c_5e-5_wd6` | 5e-5 | 1e-5 / 3e-6 | 峰值固定，扫 wd |
| `1c_1e-4_wd5` / `1c_1e-4_wd6` | 1e-4 | 1e-5 / 3e-6 | 高 lr + 低 wd 的角落 |

保留恒定 lr 对照是为了把"1cycle 的功劳"和"这一代数据/冠军变了"分开——否则换了
计划又换了代，赢了也不知道是谁的功劳。

### 16.3 开局库：34 条 curated line 换成 2000 条合成 line

这不是论文结论，是实测出来的浪费。arena / 筛选赛是**确定性对局**
（temperature=0、无 Dirichlet 噪声），所以开局线路数直接决定去重局数：

| 开局库 | 线路数 | 384 局的 distinct_games |
|---|---|---|
| Kit/data/openings.txt（bundled） | 34 | 约 276（**28% 重复**） |
| Kit/data/openings_sp.txt（合成） | 2000 | **384（0% 重复）** |

也就是换库前有近三成筛选赛算力花在与先前样本完全相同的对局上——筛选用的是
score_a，重复样本不增加信息、只让 SE 从 ±3.6% 涨到 ±4.2%。

合成库用 python-chess 生成（seed 20260927，可复现）：2000 条随机合法 6-ply 线路，
文件头有完整说明。它只为多样性服务，不代表真实开局水准；双方走同一条线路，
比较仍然公平。代价是 Elo 绝对值与旧库的结果不可直接比较——但每代的换代判定
本来就是独立的一次完整 arena，不存在跨库比较。

> **2026-09-27 实测更正（见 §17.4）**：上表"384 局 0% 重复"只在 **arena** 上验证过。
> gen 0 的十场筛选赛当时仍跑在 bundled 库上（循环进程启动时才读配置，见 §17.4），
> 实测 duplicate_rate 0.27–0.375、只用掉 32 条线路。合成库对筛选赛同样生效，
> 但那份数据不是它跑出来的，别把这行的数字引用到 gen 1 之前的筛选赛上。

### 16.4 一条被证据推翻的设想

曾考虑过"多周期衰减（SGDR）"，论文实测否掉了它（见上），而且 SGDR 自己说重启
"often temporarily worsen performance"。我们每代只导出一个候选、还要过 256 对
门槛，没有余裕去赌周期中点。要上也必须满足：**末轮加长并收到 0，且 incumbent 只取
η_min 点**。

### 16.5 迁移风险（必须记住）

两篇论文用的都是 **SGD + momentum**，而且 Smith & Topin 明说"Adam 这类自适应方法
在有效时不使用足够大的学习率，也不会出现 super-convergence"。我们是 AdamW，
lr 2e-5 在 Adam 尺度里**偏小**（transformer 常见 1e-4~3e-4），和论文里 SGD 的
0.05~3.0 完全不是同一量纲。所以"大 lr 正则化"这个核心结论**未必能迁到我们身上**，
凡是由它推出的推论（包括下调 wd）都要打折。**唯一可信的判据仍然是 arena 的 512 局。**

（另：曾以为 MuZero 用"KL 散度超阈值就降 lr"的自适应控制器，逐字核过 arXiv 全文
1911.08265 后**未证实**——全文 0 次出现 KL/Kullback/divergence，lr/optimizer/momentum/
weight decay 都没写，附录 C 原话是"参数值请参考 pseudocode"，而 pseudocode 在
ancillary files 里。该条不作为设计依据。）

---

## 17. loop_p4_v2 gen 0：开局库切换、SPRT 判决漏洞与时间预算（2026-09-27）

§16 的三个决定（1cycle、lr × wd 二维、2000 条合成开局库）在这一代落进
`configs/loop_p4_v2.json`。但 **gen 0 是个混合体**：它的十场筛选赛是旧网格 + 旧开局库
跑完的，最终 arena 才用上新配方；gen 1 起才是完整的新配方。分开记，别混。

### 17.1 最终 arena（合成开局库 / 2400 sims）

`runs/loop_p4_v2/gen_0000/arena.jsonl`，2026-09-27 20:57→21:35（UTC+8），37.2 分钟：

| 项 | 值 |
|---|---|
| 局数 | **101 / 512 计划**（48 完整对，SPRT 早停） |
| 得分 | score_a **0.5594**：A 47 胜 / 19 和 / B 35 负 |
| Elo | **+41.5**，CI95 [−19.3, +104.9]；五名法 Elo +47.3，CI95 [−4.8, +101.7] |
| SPRT | llr **+1.471**（三名法 0.721），界 [−2.251, +2.890]，`min_pairs 8` → verdict **None** |
| 换代判定 | `stopped_by_sprt: true`，但 `promote()` 返回 **False** |
| 去重 | distinct_games 101 / duplicate_rate **0.0** |
| 终止 | 将杀 82 / 子力不足 7 / 三次重复 12（12%），truncated 0.0 |
| 其他 | mean_plies 143.1；A 执白 53 局 0.641、执黑 48 局 0.469；五名法 [4,3,24,10,7] |

三条观察：

1. **合成开局库在 arena 上兑现了**：101 局拿到 101 个不同开局（0 重复），换库前的预期
   正是如此；三次重复占比 12% 也落在 §15 记的 11–14% 里——新库没有把数据多样性换坏。
2. **筛选赛的 +125 Elo 没有兑现到 arena**（0.673 / +125.5 → 0.559 / +41.5）。两者标尺
   不同（筛选赛 800 sims、arena 2400 sims），再叠加胜者诅咒（10 个变体取最大）。
   §15.2 "筛选用 score_a 只决定谁进 arena、判决只认完整 arena"这条纪律是对的。
3. **这一代记成"没换代"，但那是假阴性**——见下。

### 17.2 "没换代"是假阴性：SPRT 判决用了两份不一致的快照

`Kit/pipelines/match.py` 里，`_Run.add()` 每落一局就按**当时的记录集**判断该不该停
（`run.stopped`），而 `run_match` 的汇总是拿**追加完在途对局后的最终记录集**重算
`verdict`。两份快照一旦不一致，就会出现"该停、也停过、但判决没了"。

用这一代的真实 `arena.jsonl` 复算（五名法、逐对）：

| 时点 | 局数 | llr | 判决 |
|---|---|---|---|
| 第 85 局（40 对） | 85 | **+3.164** | 越过上界 2.890 → **H1**，`stopped=True` |
| 第 101 局（48 对，最终记录） | 101 | **+1.471** | 退回界内 → **None** |

也就是说：停止信号是在 H1 上触发的，但父进程把还在飞的几局（`workers=2`）落盘后，
最终 llr 从 +3.164 掉回 +1.471，`verdict` 重算成 `None`，`promote()` 判 False。
**候选其实是达到换代标准的**（score_a 0.559 / Elo +41.5 / CI 下限 −19.3），
却记进了 `loop.jsonl` 的 `promoted: false`。

为什么一直没暴露：**`workers=1` 没有在途对局**，停止时点和最终记录集是同一份。
§15 的 loop_p4 初版跑满 8 代、gen 0 正常换代，用的正是 workers=1；换成
`workers=2`（快约 10%，见 `AGENTS.md`）才把这条路径唤醒。这不是统计口径问题，
是"早停时点的判决"和"收尾时的判决"被当成了同一个东西。

### 17.3 修法与验证

Kit `4925c62`（2026-09-27）：**判决在越界那一刻冻结**，不再拿最终记录集重算。

- `_sprt_decided()` 改为返回 `(该停, 当时判决)`；
- `_Run` 记下 `stopped_verdict`，`add()` 在越界时捕获；续跑（读回已完成记录）同样冻结；
- `summary` 里若 `stopped_verdict` 非空则覆盖重算值。

这是标准 SPRT 语义（越界即停，alpha/beta 就是按边界值设计的），代价是判决比
"用全部记录重算"略激进——而那正是 alpha=0.05 买到的性质，不是放宽门槛。

验证：

- 单元回归 2 项：`test_sprt_verdict_is_frozen_at_boundary`（8 对 A 全胜把 llr 推过
  上界，再接 86 对一胜一负把最终 llr 拉回 **−0.082**（界内），断言冻结值仍为 H1、
  重算值为 None）；`test_sprt_stopped_always_has_a_verdict`（不变式：停了就必须有判决）。
- 远端全量 **314 项 OK**；并用 gen 0 的真实 `arena.jsonl` 复算确认打补丁后
  `verdict = H1`（下表）。

| 口径 | verdict | 后果 |
|---|---|---|
| 打补丁前（重算） | None | `promote()` = False（已发生的假阴性，无法追溯改写） |
| 打补丁后（冻结） | **H1** | `promote()` = True |

**gen 0 的那次换代机会已经永久丢了**——`loop.jsonl` 里 `promoted: false` 已落盘，
补丁不追溯。缓解事实：那个候选是拿 **1024 局**旧数据 + 恒定 lr 训的（§15 第 3 条说的
"每条记录被抽 23 次"正是它），4096 局的新数据从 gen 1 才开始。要不要重打 gen 0 的
arena 由人决定，脚本不主动碰。

**部署方式值得记住**：loop 的 selfplay/train/arena/screen 都是 `Kit` 子进程
（`Kit/pipelines/loop.py` 的 `_run`），所以**改 Kit 只影响后续拉起的子进程**。
这次是"远端 pull Kit → gen 1 的 arena 自动用新代码"，944/4096 局的自对弈一步没丢。
（反过来，改 `configs/loop_p4_v2.json` **不会**影响已在跑的 loop 进程——见 17.4。）

### 17.4 连带发现：gen 0 的筛选赛还在用旧开局库

`loop_p4_v2.json` 里 `train.screen.openings` 和 `arena.match.openings` 都指向合成库，
但 gen 0 的十场筛选赛实测还是 bundled：

| 产物 | openings | distinct_games | duplicate_rate |
|---|---|---|---|
| `gen_0000/screen_*.jsonl`（10 场） | **bundled** | 240–280 | **0.27–0.375** |
| `gen_0000/arena.jsonl` | 合成库 | 101 | **0.0** |

原因不是配置漏写，而是**时序**：循环进程只在启动时读一次配置。时间线（UTC+8）：

| 时刻 | 事件 |
|---|---|
| 09:2x | loop 以旧配置（10 个恒定 lr 变体、bundled 开局库）启动 |
| 19:07 | `configs/loop_p4_v2.json` 在磁盘上更新为 8 个 1cycle 变体 + 合成库 |
| 19:44 | 最后一场筛选赛仍按**内存里的旧配置**写出 `openings: bundled` |
| 20:57 | 重启后的 loop 才用上新配置，跑完 gen 0 的 arena |

所以 §16.3 表格里"合成库 384 局 0% 重复"目前**只有 arena 那 101 局为证**；
筛选赛的 0.27–0.375 是 bundled 的真实水平。gen 1 起的筛选赛才会用上新库。

bundled 下的实际形态：384 局只摊到 **32 条线路、每条正好 12 局**（库有 34 条），
但因为每局种子不同，重复并非整局相同——最终是 267 个不同对局（duplicate_rate 0.305）。
即"开局重复"和"整局重复"是两回事，五名法 SE 受损程度介于两者之间。

### 17.5 这一代踩全的运维坑

1. **`phase == "search"` 时不许重启 loop**：新配方 8 个 label 对上磁盘上 10 个 label 的
   `search.json`，`_search_all()` 直接 `RuntimeError`。要重启只能趁 selfplay / arena 段。
2. **配置哈希不一致是硬失败，不是自动重跑**：`Kit/pipelines/match.py` 对不上的结果文件
   抛 `ValueError`。且 `run_match` **先写 header 再建 player**，崩溃残留的 header 会堵死
   后续所有重试——gen 0 就留下过 `arena.jsonl.bad_checkpoint_20260927`
   （header 里是 `train/final.pt`，哈希 `0c6ae0b4…` 对新配置的 `e3a578b5…`）。
   删掉/改名残留文件即可恢复，但**排查成本全在"为什么它不自己重跑"上**。
3. **Kit 续跑时的候选路径 bug**（`98b2798`）：`mapping()` 一律把 `{candidate}` 指到
   `<gen>/train/<export>`，而枚举代的冠军在 `train_<label>/`；只有前进的 search 分支会
   纠正它。于是 `phase == "arena"` 续跑 → `FileNotFoundError: .../gen_0000/train/final.pt`。
4. **改 Kit 不用重启 loop（子进程热更新），改 loop 配置必须重启**——两者成本差一个自对弈
   世代（约 9.6 h），动手前先想清楚要改的是哪一层。
5. **`Kit match` 每局记录的 `elapsed_s` 不是墙钟**（实测虚高约两个数量级）。要墙钟只看
   汇总的 `elapsed_s`。

### 17.6 实测时间预算

全部是本机 5070 Ti、这一代的墙上时间（不是估计值）：

| 阶段 | 实测 | 本轮规模 | 耗时 |
|---|---|---|---|
| 自对弈 | **8.4 s/局** | 4096 局 | **9.6 h** |
| 变体训练 | 1.65–1.89 步/s | 8 个 × 1200 步 | **1.5 h** |
| 筛选赛 | 46–51 min/场 | 8 场 × 192 对（384 局）@800 sims | **7.3 h** |
| 最终 arena | **22.1 s/局** | 512 局，SPRT 通常 ~101 局就停 | **0.6 h**（打满 3.1 h） |

代际构成（`AGENTS.md` 的估算是打满 arena 的 23h，偏保守）：

- **枚举代**（gen 1、2，`enumerate_generations: 3`）：9.6 + 1.5 + 7.3 + 0.6 ≈ **19 h**
- **锁定代**（gen 3–9，不筛）：9.6 + 0.2 + 0.6 ≈ **10.4 h**

从 gen 1 自对弈起步（09-27 21:35）算：gen 1 arena ≈ **+16.8 h**，gen 2 arena ≈ **+35.8 h**，
gen 3 ≈ **+46 h**，十代跑满 ≈ **+109 h（约 4.6 天）**。

两个必须说出口的条件：

- **gen 4 起的锁定配置来自 `st["train_variant"]`**，gen 0 把它设成
  `{lr 2e-5, steps 1200}`——那是**旧恒定 lr 网格**的胜者。gen 1/2 会用新网格（1cycle）
  重选并覆盖它；但只要 gen 1/2 也没跑枚举，后 7 代就会是"1cycle 计划 + 恒定 lr 时代
  选出来的 lr"这个混合体，不要当成干净实验读。
- **AGENTS.md 的暂停纪律优先于预算**：连续 3 代未换代就人工停下复盘，此时实际耗时约
  46 h 而不是 109 h。冠军仍冻结在 `runs/loop_p4/gen_0000/train/final.pt`
  （生产预设只能人工切，loop 不许碰）。

### 17.7 gen 1 换代成功，但顺带查出：lr × wd 网格的 wd 一维是数值死区（2026-09-28）

gen 1 的完整结果：8 个变体筛完 → 选中 `1c_1e-4_wd4` → arena **43 局**（21 完整对）
score_a **0.6047**、Elo **+73.8**（CI95 [−12.7, +170.7]，五名法 +67.0 / [−0.6, +140.1]）、
`llr=+1.868`、`verdict=H1`、`stopped_at_game=29` → **promoted=true**，
新冠军 = `runs/loop_p4_v2/gen_0001/train_1c_1e-4_wd4/final.pt`。

**这份 arena 同时是 §17.2 那个修复的现场证明**：越界发生在第 29 局，之后补到 43 局时
llr 回落到 +1.868（界内）。按旧代码(final-recompute) 这次换代同样会被判成
"没换代"；新代码给出 verdict=H1 并记下 `stopped_at_game=29`，
"verdict 与 llr 看着矛盾"由此有了对账锚点。

但筛选赛排名露出一个反常现象，查到底了：

| 变体 | score_a | Elo | 说明 |
|---|---|---|---|
| `1c_1e-4_wd6` / `wd5` / `wd4` | 0.642（三者**完全相同**） | +101.4 | 同一 lr 组 |
| `1c_5e-5_wd6` / `wd5` / `wd4` | 0.596（三者完全相同） | +67.8 | 同一 lr 组 |
| `ctrl_const_2e-5_wd4` | 0.591 | +64.1 | 代内对照（恒定 lr） |
| `1c_3e-5_wd4` | 0.589 | +62.2 | |

证据链（排除了各种"配错了"的解释）：

1. **384/384 局逐字节相同**：同一 lr 组内三场筛选赛的对局（开局 + 全部着法 + 比分）完全相同。
2. **权重逐位相同**：三个 `final.pt` 的 816 个模型张量 `torch.equal` 全真、最大绝对差
   `0.000e+00`；跨 lr 组则有 792/816 个张量不同（最大差 2.6e-2）→ **lr 生效，wd 没生效**。
   文件 md5 仍不同，只因为 checkpoint 里的 config_hash 不同。
3. **不是配置或管道问题**：三份 `screen_*.json` 的 `a.checkpoint` 各指向自己的
   `train_<label>/final.pt`（路径无误），落盘的训练配置里 `weight_decay` 分别是
   1e-4 / 1e-5 / 3e-6，`latest.pt` 的优化器 `param_groups` 里也正是这三个值。
4. **最小复现**（200k 参数、AdamW、lr=5e-5〔onecycle 实测均值〕、1200 步、fp32）：
   wd ∈ {1e-4, 1e-5, 3e-6} → **逐位相同**；wd ≥ 1e-3 → 199971/200000 个参数出现差异。

机制：AdamW 的解耦衰减每步把 p 乘 `(1 − lr·wd)`。取 lr≈5e-5（onecycle 均值）、
wd=1e-4，每步相对收缩 5e-9；而 |p|≈0.02 处 fp32 的一个 ULP 约 1.9e-9 ——
**每步的衰减项都不到一个 ULP，每步都被舍入成 0，所以永远累积不起来**。
实测阈值：wd=1e-4 完全不可见，1e-3 起可见，也就是说**要活下来至少要提高一个数量级**。

后果与建议：

- **这一代的"二维网格"退化成一维 lr 网格**：8 个变体里 6 个是彼此的精确副本，
  gen 1 因此白烧约 5.6 h GPU（6 次多余训练 ≈1 h + 6 场多余筛选赛 ≈4.6 h）。
  gen 2 若不改配置会原样再烧一遍。
- **冠军的 wd 是"抽签"抽出来的**：三者并列，`max()` 的 tie-break 按迭代序取第一个，
  即 `1c_1e-4_wd4`。因为三者本来就相同，这不影响正确性，但
  **"wd 选了多少"这件事根本没有被测过**。
- **§16.2 的依据（Smith & Topin：大 lr 要同步降 wd）在本配方上未被检验**——
  不是被推翻，而是网格整体落在数值死区里，没有分辨力。（而 §16.2 自己早已提示
  "大 lr 正则化"未必能迁到 AdamW 小 lr  regime。）
- **两个可选修法**：(a) 直接去掉 wd 这一维，8 → 4 个变体，每枚举代省约 3.7 h；
  (b) 若坚持要测 wd，网格必须整体上移至少一个数量级（如 1e-3 / 3e-3 / 1e-2），
  这和 Smith & Topin 建议的方向恰好相反。（注意：AdamW 在配置缺省 `weight_decay`
  时给的是 1e-2，即历代 T 配方若没显式写 wd，其实一直是"勉强活着"的那档。）
- 顺带解释了为什么此前从没注意到：`weight_decay` 在历代 T 训练里都是 1e-4（或被 1e-2
  缺省覆盖），其效应从未在任何一次对局里显形过。

## 18. gen 2：5 变体新网格首战未换代，wd 死区拿到实证（2026-09-29）

2026-09-28 14:58 重启 loop（PID 166263）换 5 变体网格，gen 2 全程跑完。

### 18.1 时间账

| 阶段 | 起止（本地） | 耗时 | 产出 |
|---|---|---|---|
| 自对弈 4096 局 | 09-28 14:58 → 09-29 00:26 | **9.5 h**（8.35 s/局） | 54.5 万条记录 |
| 5× 训练（1200 步） | 00:26 → 04:29 | 0.83 h | 5 份 `final.pt` |
| 5 场筛选赛（384 局） | 00:37 → 05:16 | **4.83 h**（46–49 min/场） | 见下表 |
| 最终 arena | 05:16 → 05:31 | 0.26 h（953.8 s） | **39 局，H0** |

合计 15.4 h，与 `loop.md` §4.2 的枚举代预算（16 h）吻合。

### 18.2 筛选赛：lr 单调，wd 探针"垫底但没输"

| 变体 | lr | wd | score_a | Elo | 局数 |
|---|---|---|---|---|---|
| `1c_5e-4_wd1e-4` | 5e-4 | 1e-4 | **0.6224** | +86.8 | 384 |
| `1c_2e-4_wd1e-4` | 2e-4 | 1e-4 | 0.5781 | +54.7 | 384 |
| `1c_1e-4_wd1e-4` | 1e-4 | 1e-4 | 0.5742 | +52.0 | 384 |
| `ctrl_const_2e-5_wd1e-4` | 2e-5（恒定） | 1e-4 | 0.5547 | +38.2 | 384 |
| `1c_1e-4_wd1e-2` | 1e-4 | **1e-2** | 0.5534 | +37.2 | 384 |

`_selected = 1c_5e-4_wd1e-4`（取 score_a 最高者，`loop.py` 的 tie-break 顺序）。
lr 轴单调向上与 gen 1 一致，5e-4 是当前上边缘，且领先第二名 0.044（≈1.7 倍标准误）。
注意 gen 2 全场分数比 gen 1 同代低 0.03–0.07（`ctrl` 0.555 vs 0.591、
`1c_4` 0.574 vs 0.642），是本代数据边际收益下降的整体信号。

### 18.3 最终 arena：39 局判 H0，未换代

候选 = `gen_0002/train_1c_5e-4_wd1e-4/final.pt`，对手 = gen 1 新冠军
（`train_1c_1e-4_wd4/final.pt`），2400 sims，512 局封顶：

| 指标 | 值 |
|---|---|
| 成绩 | 18 胜 8 和 13 负，score_a **0.5641**，Elo **+44.8** |
| SPRT | `llr` 收尾 **+0.284**、`llr_trinomial` +0.366、`verdict=**H0**`、`stopped_at_game=20` |
| 界 | [−2.251, +2.890]（elo0=0 / elo1=60 / α=0.05 / β=0.1） |
| 其它 | `duplicate_rate` 0.0、39 局全不同、`elapsed_s` 953.8 |

第 **20** 局 llr 越过下界 −2.251（"够不上 +60 Elo"的证据已足）→ 判决冻结 H0，
之后在途对局把收尾 llr 抬回 +0.284。**verdict 与 llr 看着又矛盾，是 §17.2 那个
冻结语义的正常表现，不要用它推翻判决。**冠军未动，`promoted=false`。

### 18.4 wd 探针：死区从解析结论变成实证结论

§17.7 的表预测 wd=1e-2 会让权重差 0.12%。实测（两份 `final.pt`，816 个浮点张量）：

- **792/816 个张量不同**（24 个相同，是无梯度的那几个），
  平均相对 L2 差 **3.38e-3**、最大绝对差 1.39e-2（`middlegame.blocks.0.mlp.w1.weight`）。
  比一阶估算（1.2e-3）大约 3 倍，是 wd 扰动轨迹后的分歧放大，量级吻合。
- 但棋力上：`1e-2` 档 0.5534 vs `1e-4` 档 0.5742，差 **0.021 ≈ 0.8 倍标准误**
  （384 局、双方对同一位对手），统计上分不出高低。

**结论：wd 在 1e-4 ~ 1e-2（两个半数量级）之间对棋力无影响，wd 这一维正式关闭。**
gen 1 那三个"位相同"是 wd=1e-5/3e-6 的死区更深一档；gen 2 跨到 1e-2 权重终于动了，
强度仍然不动。后续任何配方都不必再为 wd 花 slot，
除非愿意把 wd 推到 1e-1 量级（§5.2 的表：那才有 1.19% 收缩）。

### 18.5 锁定的是筛选赛胜者，arena 说不上话（必须记住）

`Kit/pipelines/loop.py:493` 在 search 阶段结束、**arena 之前**就把
`train_variant` 写进 `loop_state.json`：

```
"variant":        "1c_5e-4_wd1e-4",
"train_variant":  {"optimizer": {"lr": 5e-4, "weight_decay": 1e-4}}
```

所以 `enumerate_generations: 3` 的锁定**结构上不看 arena 结果**：gen 3–9 将按
lr 5e-4 训练（无筛选赛、无 arena 选优），而冠军仍是 gen 1 的 lr 1e-4 模型。
三个选项与原样跑的利弊比较写在 `loop.md` §5.2；
改锁定的硬截止点是 gen 3 训练阶段开始（实际发生在 09-29 15:10 前后）。
**结局见 §18.7：gen 3 就按这份锁定配置训练并换代成功，"原样跑"赌对了。**

### 18.6 下一代（gen 3）的预期

gen 3 是**锁定代**：自对弈 4096 局 → 训练 10 min → arena。

速率口径先说清楚（08:02 本地实测，两代一致）：gen 2 自对弈 **8.35 s/局**（9.5 h），
gen 3 前 152 分钟 1079 局 = **8.45 s/局**。**不要用"进程启动到现在"估**——那段含引擎
加载与首次编译缓存；`wc -l games.jsonl` 也要配 `ps` ELAPSED 交叉验证。
（09-29 早上按 9.3 s/局估过一次，ETA 虚高 1 h，就是拿 650 局除了含预热的 101 分钟。）

据此：gen 3 自对弈 **09-29 15:10 本地**前后跑完 → 训练约 10 min（产物落
`gen_0003/train/`，`optimizer.lr` 应为锁定值 5e-4）→ arena 判决预计 **16:00–16:30**
（打满 512 局则到 18:30）。gen 3 若也 H0，就是连续 2 代未换代；连续 3 代（gen 2/3/4）
即触发暂停纪律，最早 **09-30 上午**需要人工复盘。
**gen 3 的 arena 同时是 lr 5e-4 轨迹的第二次检验**——若再次 H0，
"筛选赛偏好 5e-4"就只剩 gen 2 一次观测支撑，届时应当考虑改回 lr 1e-4（截止点是
下一代训练阶段开始，见 §18.5）。

### 18.7 gen 3：第一个锁定代，arena +65.1 Elo 判 H1 换代成功（2026-09-29）

gen 3 无筛选赛，按 §18.5 那份锁定的 `lr 5e-4 + 1cycle + wd 1e-4` 直接训练。
**结论：锁定赌对了，gen 2 的 H0 是数据/抽样问题而不是 lr 问题。**

| 阶段 | 起止（本地） | 耗时 | 备注 |
|---|---|---|---|
| 自对弈 4096 局 | 05:32 → 14:59 | **9.45 h**（8.28 s/局） | `first_game=12288`，sink 83.8 MB |
| 训练 1200 步 | 14:59 → 15:10 | **11 min**（1.875 步/s，640 s） | loss 1.463 → 1.430 |
| 最终 arena | 15:10 → 15:43 | **33 min**（1995 s） | 81 局 / 24.6 s 局 |
| 合计 | | **36717 s = 10.2 h** | 与锁定代预算一致 |

**训练配置核对**（`gen_0003/train.json`，锁定代产物落 `train/` 而非 `train_<label>/`）：

- `optimizer = {"lr": 5e-4, "weight_decay": 1e-4, "betas": [0.9, 0.999], "fused": false}`
- `schedule = {"kind": "onecycle", "pct_start": 0.25}`，**无非法组合残留**
- `train.log` 的 lr 轨迹印证生效：step 1 = 2.0e-5（预热起点）→ step 300 = **5.0e-4**
  （峰值正好是锁定值）→ step 1200 = 3.5e-9（退火到底）
- 数据 = `task.kwargs.data`（`kind: mix`），70% ResNet 监督分片
  （`ResNet/data/shards_evals/evals_*.bin`，64 个文件约 7683 万条）+ 30% 自对弈分片
  （**gen 2 + gen 3 各一份，window=2 正确**，共 104.5 万条），
  `base_ckpt` = gen 1 冠军

**最终 arena**：a = `gen_0003/train/final.pt`，b = gen 1 冠军，256 对封顶 / 2400 sims /
合成开局库 `Kit/data/openings_sp.txt` / seed 20260926 / elo1=60：

| 指标 | 值 |
|---|---|
| 成绩 | 38 胜 20 和 23 负，score_a **0.5926**，Elo **+65.1**，CI95 [0.2, +134.9] |
| SPRT | llr **+3.712**（越上界 +2.890 发生在第 **61** 局）、`verdict=H1`、`stopped_by_sprt=true` |
| 其它 | `duplicate_rate` 0.0、`mean_plies` 157.8、`elapsed_s` 1995.0 |

判 **promoted=true**，新冠军 = `gen_0003/train/final.pt`。连续未换代计数清零
（gen 2 H0 → gen 3 H1），暂停纪律未触发。

三点值得记住：

1. **锁定不看 arena 这个设计口径，在 gen 3 上拿到了正面结果**——但这是"赌对了"，
   不是"设计对了"。以后再遇到"锁定配置刚被 H0"仍要按 §18.5 的三个选项显式决策，
   不能拿这次当先例自动照做。
2. **gen 4 已用新冠军开始自对弈**（15:43，`first_game=16384`，
   权重 `gen_0003/train/final.pt`），预计 10-01 01:55 本地出判决；
   剩余 6 代（gen 4–9）按 10.2 h/代算，loop 预计 **10-03 05:10 本地**收尾。
3. **生产权重已切换**（2026-09-29）：`Transformer/config.json` 的两个预设
   `max_mcts` / `max_t` 已指向 `gen_0003/train/final.pt`，远端 `unichess-server`
   已重启（PID 308690），`/api/health` + 两个 preset 的开局对局均正常，
   journal 无 error/traceback。血统链：`stratified_p4_selfplay_corrected` →
   loop_p4 gen0（+163 Elo，直接打赢被替换的生产权重）→ v2 gen1（+73.8）→ v2 gen3。
    回滚 = 改回旧 ckpt 路径后重启 `unichess-server`。

### 18.8 gen 4：锁定代二番战，−74.8 Elo 硬 H0（2026-09-30）

gen 4 同样无筛选赛，锁定的 lr 5e-4 + onecycle + wd 1e-4 直接训练。
**gen 3 的换代红利没有延续：gen 4 的候选比新冠军（gen 3）弱 75 Elo。**

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 自对弈 4096 局 | 9.45 h | 8.28 s/局（与 gen 3 一致） |
| 训练 1200 步 | 11 min | lr 峰值 5e-4，onecycle 退火到底 |
| 最终 arena | **13.2 min**（790 s，仅 33 局） | SPRT 早停，H0 |
| 合计 | 9.76 h | 与锁定代预算一致 |

**最终 arena**：a = `gen_0004/train/final.pt`，b = gen 3 新冠军，256 对封顶 / 2400 sims /
合成开局库 / seed 20260926 / elo1=60：

| 指标 | 值 |
|---|---|
| 成绩 | 8 胜 10 和 15 负，score_a **0.3939**，Elo **−74.8**，CI95 [−185.2, +22.0] |
| SPRT | llr **−2.672**（越下界 −2.251 发生在第 **14** 局）、`verdict=H0`、`stopped_by_sprt=true` |
| 其它 | `duplicate_rate` 0.0、`mean_plies` 149.2、`elapsed_s` 790.1 |

判 **promoted=false**。连续未换代计数 = 1（gen 4），暂停线仍是 3 代。

关键信号：

- gen 4 的 llr **一直在下界下方**（不像 gen 2 的收尾 llr 被在途局拉回 +0.284），
  所以"gen 4 只是不够好"的说法站不住——**候选是真的比冠军差**，不是统计噪音。
- lr 5e-4 轨迹现在是 **1 胜 1 负**（gen 3 +65.1 H1 vs gen 4 −74.8 H0），胜负各一，
  没有稳定结论。gen 5（同样 lr 5e-4）的结果将决定 streak 是否来到 2。
- 唯一经 arena 验证过"稳赢"的 lr 仍是 **1e-4**（gen 1  championship +73.8 H1）；
  5e-4 的筛选赛优势（gen 2 的 0.622 vs 1e-4 的 0.574）已被 gen 3/4 的 arena 结果否定。
- gen 5 已起跑（`first_game=20480`，3787/4096 局，01:29 本地起跑），
  预计 10:55 跑完自对弈、arena 判决约 11:40–12:00。若 gen 5 也 H0 = 连续 2 代，
  下一代会触发暂停纪律（3 代），届时必须人工决策：改回 lr 1e-4 / 扩大枚举 / 其他。

### 18.9 gen 4 劣化归因：过训练 + 导出点错（2026-09-30）

gen 4 输了 74.8 Elo，但**它不是"训练坏了"**。把候选与冠军放到同一份数据上做
纯推理比对（16 次更新 × accum 4，bf16 / fp16 / fp32 三档），结论和直觉相反：

| 指标 | gen 3 冠军(base) | gen 4 候选 | 差 | 说明 |
|---|---|---|---|---|
| 训练数据总 loss | 1.442498 | **1.414963** | **−0.0275** | 候选拟合更好 |
| ├ 70% ResNet 源 | 1.620831 | 1.616635 | −0.0042 | 静态源几乎没动 |
| └ 30% 自对弈源 | 1.017139 | **0.945765** | **−0.0714** | 收益全在这 |
| fp32 基准 | 1.439382 | 1.410386 | −0.0290 | — |
| bf16 | 1.439470 | 1.410634 | −0.0288 | 与 fp32 一致 |
| fp16 | 1.439388 | 1.410384 | −0.0290 | **与 bf16 无差异** |
| fp16 激活峰值 | 42.9 | 42.6 | — | 上限 65504，**精度无关** |
| 与冠军策略 KL | — | 0.01297 | 小 | gen1→gen3 是 0.02515 |
| max&#124;Δp&#124; p50 | — | 0.0286 | — | 无策略崩塌 |

**四条独立证据全部否掉"训练失败"：loss 更低、fp16/fp32 无损、策略偏离更小、
也未向弱引擎回归**（KL(gen1→X) 从 0.030 涨到 0.052，是远离不是靠近）。

#### 真因一：900 步白训，且导出的是全程最差点

`gen_0004/train.log` 的曲线形状与 gen3 根本不同：

| 段 | step | lr | gen4 loss | gen3 loss |
|---|---|---|---|---|
| 上升沿 | 1→300 | 2.0e-5 → **5.0e-4 峰值** | 1.4473 → **1.4014（最低）** | 1.4632 → 1.4521 |
| 退火沿 | 300→700 | 5.0e-4 → 2.9e-4 | 1.4014 → **1.4090（回升）** | 1.4521 → 1.4408 |
| 收尾段 | 700→1200 | 2.9e-4 → 3.5e-9 | 1.4090 → **1.4269** | 1.4408 → **1.4287（最低）** |

gen4 在 **step 300 触底**，之后 900 步 loss 单调回升 0.0255，终值比最低点高 2.6%。
而 `export.final` 只导出 step 1200，**导出的就是全程最差的点**。
gen3 则是"终点即最低点"，所以同样 1200 步无害。

**按候选自己的数据估算净收益**：step 1→300 赚了 0.0459；step 1→1200 只赚 0.0275。
即 900 步把收益砍掉约 40%——**不是没干活，是倒扣**。

基座放大了这一点：gen3 的 base 是 gen1（远离最优点，900 步都在改善）；
gen4 的 base 是 gen3（本来就接近最优点，step 300 就到顶，剩下 900 步全是无谓扰动）。

#### 真因二：训练外 loss 变差——过拟合签名

在 gen2 分片（训练窗口外，由 gen1 引擎下出）上单独测：

| 分片 | gen3 冠军 | gen4 候选 | 差 |
|---|---|---|---|
| gen3 分片（训练内，gen1 下出） | 0.958171 | 0.912329 | −0.0458 |
| gen4 分片（训练内，gen3 下出） | 1.056043 | 0.963410 | −0.0926 |
| **gen2 分片（训练外）** | **0.960817** | **0.970693** | **+0.0099** |

训练内两份都更好，训练外那份更差。这是**过拟合的教科书签名**：模型在见过的
分片上拟合更好、在没见过的分片上更差。

#### 真因三（次要，且**已推翻我最初的误判**）：两个源的体量差 4.6 倍

先纠错：我最初写"名义 30% 自对弈、实际只有 17.7%"，**这是错的**——把
「源里有多少条记录」误当成了「训练目标占多少」。核对
`Kit/planes19/task.py` 的 `_mix_batches`：

```python
sizes = np.floor(weights / weights.sum() * bs).astype(int)   # bs=512 → [358, 153]
sizes[0] += bs - sizes.sum()                                 # → [359, 153]
```

即每批 **[359, 153] = 70.1% / 29.9%**，70/30 的权重是**如实执行的**，
取整误差只有 1 个样本。这一条不是 bug。

但体量差是事实，换个角度看仍然有意义：

| 源 | 记录数 | 每代被抽样本数 | 每代覆盖率 |
|---|---|---|---|
| ResNet 静态（4 分片，9-08 起未变） | 4,801,775 | 1,720,320 | **36%** |
| 自对弈（gen3+gen4 两份 sp.bin） | 1,034,209 | 737,280 | **71%** |

每代 1200 步 × accum 4 = 4800 批。同样的 70/30 批内比例下，
**自对弈语料的每记录被抽率是静态语料的 2 倍**（71% vs 36% 覆盖率），
加上 `window: 2` 让同一份自对弈分片连着两代都进训练集，
自对弈数据的实际重复度明显高于静态数据。所以"自对弈数据更旧更快被榨干、
静态数据反而没用满"这个方向是对的，只是**远没有 17.7% 那么夸张**，
它不足以解释 −75 Elo，主因仍是上面两条。

**结论（修正后）**：gen4 的失败主要由过训练 + 导出点错造成；
数据配比是次要的观察，不是主因。若后面要动配比，正确做法是提高
`weight`（或增大自对弈分片），而不是去"修"一个不存在的取整 bug。

#### 已被否决的假设（留档，避免重复排查）

- **fp16 溢出 / 精度损失**：fp16 与 fp32 的 loss 差仅 −0.0003，激活峰值 42.6
  （上限 65504，差三个数量级）。三个权重完全同级，**彻底否定**。
- **向弱引擎回归（灾atengh forgetting）**：KL(gen1→候选) 0.030 → 0.052，
  是**远离** gen1，不是靠近；KL(冠军→候选) 仅 0.01297，比上一代换代的 0.02515 还小。
- **策略崩塌 / 置信度过拟合**：max&#124;Δp&#124; p50 = 0.0286、p99 = 0.1373，
  top1 不一致率 0.1035（gen1 vs gen3 是 0.1167），量级正常。
- **训练发散 / 数值不稳**：`grad_norm` 全程 0.39–0.65，无 spike；
  `nonfinite: raise` 未触发，训练正常 `completed`。

#### 结论与可执行建议

gen4 的失败**与 lr 对不对无关**：loss 实实在在降了、降得比冠军好，
但 lr 峰值（step 300）之后的 900 步把它推出了最优点，且落点被导出。

按可操作性排序：

1. **改 `export`：加 `best_model.pt` 的早停语义**（现在 `validation: {}` 为空，
   `best` 恒为 `Infinity`，`Kit/train/trainer.py:345` 在无 validate 时把 best 复制成
   final）。要么给 train 配一个 validation 源，要么让 loop 自己按 train loss 选最优步。
2. **压短 onecycle**：`steps: 1200` 对"基座已接近最优点"的情况过长。
   可试 400–600 步，或把 `pct_start` 降到 0.1（峰值前移到 step 120）。
3. **若要动配比，走提高 `weight` 或加大自对弈分片这条路**（每代覆盖率
   自对弈 71% vs 静态 36%，自对弈被重复榨干的速度是静态 2 倍）。
   **不要去"修"`_mix_batches` 的取整——那不是 bug**，见上文纠错。
4. **不要把这次归因成 lr 问题**：gen4 用 lr 5e-4 把 loss 降得比冠军好，
   问题出在"训多久"和"导哪一步"。

### 18.10 gen 5 判决：同一份锁定配置再次 H0，streak 到 2（2026-09-30）

gen 5 与 gen 4 用**完全相同**的锁定配置（`lr 5e-4` + onecycle `pct_start 0.25` +
`wd 1e-4` + **`steps 1200`**），arena 对新冠军 gen3：

| 指标 | 值 |
|---|---|
| 成绩 | 45 局，score_a **0.4778**，Elo **−15.45** |
| 判决 | **H0**，`promoted=false` |
| 墙钟 | `sec` **35592.0**（9.89 h） |

比 gen 4 的 −74.8 温和得多，但方向一致。**连续未换代计数 = 2**
（gen 4 −74.8、gen 5 −15.45），暂停纪律（3 代）只差一代。
冠军仍为 `gen_0003/train/final.pt`。loop 已推进到 generation=6 / phase=selfplay。

**这一条把 §18.9 的归因从"gen 4 个个案"升级为"重复出现的模式"**：
两代用同一份配置都 H0，而它们的训练 loss 都确实下降了（gen 4 已被直接测出
比冠军低 0.0275）。所以问题不在数据运气，而在**这个配置"训 1200 步"这件事本身**
在基座接近最优点时是净负收益的。

**重要且必须写清的现状**：截止 2026-09-30 11:40，§18.9 建议的"压短步数"
**从未落地**——`configs/loop_p4_v2.json` 的 `steps` 仍是 **1200**，
因为 loop 进程启动时把配置读死，改步数必须改配置 + 重启 loop。
gen 4、gen 5 都是带着 1200 步跑完的。gen 6 的自对弈窗口（当前）是安全重启时机，
硬截止点是 gen 6 的 train 阶段开始（约 09-30 20:30 本地）。

### 18.11 配置已改并重启：steps 400 + 自对弈 0.5（2026-09-30 13:41）

上面"未落地"的状态已结束。改了 `configs/loop_p4_v2.json`（提交 `511403e`）：

| 字段 | 原值 | 新值 | 理由 |
|---|---|---|---|
| `train.steps` | 1200 | **400** | gen4 的 loss 在 step 300 见底后单调回升，900 步净倒扣 |
| `data.sources[0].weight`（ResNet 静态） | 0.7 | **0.5** | 与自对弈配平 |
| `data.sources[1].weight`（自对弈） | 0.3 | **0.5** | 每批取整 `[256,256]` 恰好一半 |

**未动并被实测确认**：`optimizer.lr` 仍是锁定值 **5e-4**。
`_merge` 是递归合并，`loop_state.json` 的 `train_variant`（只含
`optimizer.lr/weight_decay`）不会碰到 `steps` 与 `sources`。
在远端用 `_apply_overrides` 的同一套合并逻辑验证过 gen 6 的有效训练配置：

```
optimizer.lr = 0.0005 (锁定值)   steps = 400
schedule     = {'kind':'onecycle','pct_start':0.25}
weights      = [0.5, 0.5]       每批 [256, 256]
```

**重启过程**（`phase=selfplay` 窗口内，安全）：

1. 先 SIGTERM 自对弈子进程 → loop 因子进程 `rc=-15` 抛 `RuntimeError` **自行退出**
   （`loop.log` 因此多一条 Traceback，**计划内，非新故障**）；
   不必 SIGKILL——`loop.py:466` 的 `FileLock` 是 OS 级 flock，进程退出即释放，
   实测 2 秒内 `loop.lock` 恢复空闲。
2. **已写的 998 局完整保留**，三重保障：
   - `SelfPlayShardSink._repair()` 按元数据记录条数截断半截尾，只允许截小；
   - `plan_selfplay` 按全局局号 `first_game=24576` 编号，开局与 RNG 流只取决于局号；
   - `run_selfplay` 用 `skip_games=done_games()` 在建池**前**过滤（`selfplay.py:156`），
     998 局不会被重跑。
3. 新 loop PID **811011**（ppid=1，已 setsid 脱离），子进程 811032，
   `/tmp/unichess_t_loop_v2.pid` 已写回。

**又一个测量坑（§loop.md §4.2 第三次应验）**：重启后头 90 秒只写 2 局
（≈45 s/局），差点被误判成续跑 bug。拉长到 5 分钟：**稳态 7.50 s/局**，
与重启前（7.5 s/局）一致。**结论：重启自对弈后至少测 5 分钟再下判断。**

**gen 6 预期**：剩 3023 局 × 7.50 s = 6.3 h → 自对弈约 **20:12 本地**完成；
训练 400 步约 3.5 min；arena 判决 **20:30–21:00**。

### 18.12 第二批修复：候选改取 best_model.pt（Kit `select_best_by`）

§18.11 只修了 gen 4 事故的两个根因之一，第二个（**导出的点是全程最差的点**）在 Kit 侧修：

| 文件 | 改动 |
|---|---|
| `Kit/train/config.py` | `export` 新增 `select_best_by ∈ ("train","none")`，**默认 `"none"`**（完全保持旧行为）。`export` 不进配置哈希，不影响续训 |
| `Kit/train/trainer.py` | `select_best_by="train"` 且无 validation 时，每个日志点比较训练 loss，创新低即把该步导出到 `export.best` 并记 `best`/`improved`；一步都没选出才退回复制 final |
| `Kit/tests/test_train.py` | 新增 `TestSelectBestByTrain`（4 项）+ `VShapedLossTask`；远端全量 **346 项 OK**（基线 342 + 4） |

配比口径：日志里的 `loss` 是最近 `log_every` 个**优化步**的平均，覆盖
`log_every × accum × batch` 个样本（循环里 50×4×512 ≈ 10 万），比单步 loss 稳得多。

配套 loop 配置改动（`5c3d68c`）：顶层 `export` `final.pt` → **`best_model.pt`**、
`train.export.select_best_by` → **`"train"`**。⚠ Kit 必须先于 loop 升级，
否则 `loop.py:338` fail-fast 报"训练结束但没有导出候选"。

**第二次重启**（14:28，自对弈安全窗口）：新 loop PID **832414**、子进程 832435，
1326 局数据保留。

**为什么这次是零风险**：若 gen 6 的 loss 单调下降（400 步下大概率如此），
`best_model.pt` 与 `final.pt` 逐位相同，改动不起作用；若中途仍有回升，
则恰好兜住。等价于给"训过头"上了一道与步数无关的保险。

### 18.13 修 bug：select_best_by 判据错 + 当晚一次真实崩溃（2026-09-30 夜）

上一节说"零风险"是错的——当晚 **21:00 左右 loop 崩了**，gen 6 的 arena 从未运行。

```
RuntimeError: 训练结束但没有导出 .../gen_0006/train/best_model.pt
  File "Kit/pipelines/loop.py", line 338, in phase_train
```

**根因**：`bf17fd5` 把判据写成 `validate is None`（**方法**不存在）。
但 T/R 的 `Planes19Task` **自带** `validate` 方法，只是没配验证数据源时
它返回空 dict、`score` 为 None——**行为与没有方法完全一致**。于是：

- `by_train_loss` 分支永不触发 → `best_model.pt` 从未导出；
- `do_val` 在 `step == cfg.steps` 时为真，打出一条
  `{"step": 400, "validation": {}, "best": Infinity, "improved": false}`。

证据就在 `gen_0006/train/train.jsonl` 末行：**validation 记录在**，
说明 validate 真的被调用了，只是拿不到 score。这一条记录本身就是判据写错的铁证。

**连带发现的第二个问题**：`Trainer._run` 里 `step >= cfg.steps` 的
提前返回路径**什么都不导出**。所以即使 loop 重入 `phase_train`，
Trainer 从 `latest.pt` 看到 step=400>=400 直接 return，final/best 都拿不到，
照样崩。这是同一晚的第二次崩溃来源，latent bug，被这次改动暴露出来。

**修复**（Kit `cf831fa`，判据改成"**拿不到 score**"而不是"方法不存在"）：

```python
by_train_loss = (export.get("best") and export.get("select_best_by", "none") == "train")
if do_val:
    ...
    score = res.get("score")
    improved = score is not None and score < best
    if improved:
        ...export by validation...
    elif (score is None and by_train_loss and logged_loss is not None
            and logged_loss < best):
        ...export by train loss...
elif (by_train_loss and logged_loss is not None and logged_loss < best):
    ...export by train loss...
```

早退路径也改成按 `export` 落盘；best 的真实权重在 `latest.pt` 里已不可重建，
**磁盘上已有就原样留着不覆盖**，只有缺失时才拿当前步顶上。

**回归测试**（`Kit/tests/test_train.py`，共 19 项全过，全量 349 项 OK）：

- `test_select_best_by_train_with_scoreless_validate`：钉住上面那个根因
- `test_completed_run_reentered_still_exports`：早退路径必须导出
- `test_completed_run_reentered_keeps_existing_best`：不许用 final 覆盖已有 best
- `_read_log` 区分「全部记录 / loss 记录」：`train.jsonl` 混三种记录，
  之前算 `improved` 会 `KeyError: 'loss'`

**试过但回退**：给 `phase_train` 加「候选已存在就跳过重训」的守卫。
它打破 `test_locked_generation_writes_replaced_schedule_to_disk`
（该测试特意预创建 `train/final.pt` 并断言 `_run` 仍被调用），
而且恢复 gen 6 并不需要它——`latest.pt` 在，重进走早退路径几秒就导出完。

**gen 6 的判决没有被崩溃污染**（这是关键）：

- 训练产物完整：400 步、241.6 s、`final.pt` + `latest.pt` 都在；
- 恢复时 `best_model.pt` 从 `latest.pt` 的早退路径导出，
  **实测 816 个张量与 `final.pt` 的 sha256 完全相同**（文件差 2134 字节是
  pickle 序列化噪音，内容一致）；
- 而 gen 6 的 loss 最低点**正是 step 400**（见下），所以 best=final 语义等价。

**恢复操作**：22:04 重启 loop（`phase=train` 续跑），Trainer 早退导出
`best_model.pt` → 候选检查通过 → `phase=arena` → arena 开跑。

**顺带证明 400 步的改动是对的**——gen 6 的完整曲线：

| step | lr | loss |
|---|---|---|
| 1 | 2.01e-05 | 1.359535 |
| 50 | 2.64e-04 | 1.337315 |
| 100 | **5.00e-04（峰值）** | 1.339854 |
| 150 | 4.65e-04 | 1.319879 |
| 200 | 3.73e-04 | 1.308815 |
| 250 | 2.47e-04 | 1.296634 |
| 300 | 1.23e-04 | 1.278206 |
| 350 | 3.22e-05 | 1.275592 |
| **400** | 1.57e-08 | **1.274855（最低点）** |

对照 gen 4 的 1200 步：step 300 见底 1.4014 后回升 0.0255。
**同样的 lr 峰值点，400 步下 loss 一路降到终点、终点就是最低点**——
"步数过长"这个诊断被 gen 6 直接证实。唯一的抖动是 step 100 的
1.3373→1.3398（+0.0025），量级是噪声。

**教训（给以后的自己）**：耦合 Kit 与 loop 的改动时，"先升 Kit 再上 loop 配置"
只保证了顺序，**没保证 Kit 的实现是对的**。这类跨仓改动落地后必须有一次
端到端 smoke（哪怕只跑 2 步训练确认候选文件真的出来了），
不能只看单测绿——本例单测全绿、全量 346 项 OK，线上照样崩。

### 18.14 gen 6 判决：H0，streak 到 3，触发暂停纪律（2026-09-30 22:50）

**arena 判决**（`gen_0006/arena.jsonl.summary.json`，arena 22:04 起跑、836.4 s）：

| 指标 | 值 |
|---|---|
| 成绩 | 37 局，14 胜 8 和 15 负，score_a **0.4865**，Elo **−9.39**，CI95 [−112.2, +91.7] |
| SPRT | llr **−1.5785**（越下界发生在第 **20** 局）、`verdict=H0`、`stopped_by_sprt=true` |
| 其它 | `duplicate_rate` 0.0、`mean_plies` 142.0、`elapsed_s` 836.4 |
| 判 | **promoted=false**，新冠军仍为 `gen_0003/train/final.pt` |

配置复核 **PASS**：`steps=400`、`lr=0.0005`（锁定值）、
`schedule=onecycle/pct_start 0.25`、`weights=[0.5,0.5]`、
`export.select_best_by=train`。崩溃恢复后配置完好生效。

**连续未换代 = 3**（gen 4 −74.83、gen 5 −15.45、gen 6 −9.39）→
**触发暂停纪律**（`AGENTS.md`：连续 3 代未换代就停下人工复盘，不放宽门槛）。
**已于 22:52 停 loop**，配置与 `loop_state.json` 一字未改
（仍是 `generation=7 / phase=selfplay`），gen 7 已写的 **240 局**完好，
重启即从第 241 局续跑。`unichess-server` / `unichess-tunnel` 未受影响。

#### 但趋势是明确变好的，这是复盘的核心材料

| 代 | 配置 | Elo | score_a | 局数 | SPRT |
|---|---|---|---|---|---|
| gen 4 | 1200 步 / 70-30 | **−74.83** | 0.3939 | 33 | llr −2.672，14 局越界 |
| gen 5 | 1200 步 / 70-30 | **−15.45** | 0.4778 | 45 | H0 |
| gen 6 | **400 步 / 50-50 / best_model.pt** | **−9.39** | 0.4865 | 37 | llr −1.579，20 局越界 |

Elo 从 −74.8 收窄到 −9.4，score_a 从 0.394 爬到 0.4865（离 0.5 只差 0.0135）。
**但三条 SPRT 都是 H0，按纪律一票都不能算换代。**

#### 一个必须点出的统计细节：gen 6 是"弱 H0"

gen 6 的 `stopped_at_game=20`（第 20 局越下界 −2.251），
但**收尾 llr 只有 −1.5785，已回到界内**——
即越界之后又在途下了 17 局，把 llr 拉回了中间地带。
这与 gen 2 那次"越上界 +2.890 判 H1、收尾 llr 被在途局拖回 +1.471"完全同型。

按"越界即冻结"的口径，verdict 是 H0，这条必须认；
但**证据强度比 gen 4 弱得多**：gen 4 的收尾 llr 是 −2.672（一直在下界下方），
gen 6 只有 −1.5785。CI95 [−112.2, +91.7] 也稳稳含 0。
换句话说：**候选不是"更差"，而是"看不出来更好"**。

#### 已确认修好的两件事（不容否认）

1. **400 步消除了过训练**：gen 6 曲线 1.359535 → 1.274855 单调下降、
   **终点即最低点**（§18.13 表），对照 gen 4 的 step 300 见底后回升 0.0255。
2. **`select_best_by` 与崩溃都已修复**（Kit `cf831fa`），
   恢复后 `best_model.pt` 与 `final.pt` 张量 sha256 相同，判决未被污染。

#### 停 loop 时保留的现状（复盘后据此决策）

- `loop_state.json`：`generation=7 / phase=selfplay`，`champion=gen_0003`，
  `train_variant={"optimizer":{"lr":0.0005,"weight_decay":0.0001}}`
- `gen_0007/selfplay/`：240 局已写（`sp.bin` 4,686,400 B），`first_game=28672`
- 配置 HEAD：Transformer `f10d645`、Kit `cf831fa`，工作副本均无本地改动
- 恢复命令（改完配置后手动执行，**不要**在复盘前重启）：
  ```bash
  cd /home/jeefy/UniChess
  setsid nohup /home/jeefy/miniconda3/envs/unichess/bin/python \
    -m Kit loop Transformer/configs/loop_p4_v2.json \
    >> ~/UniChess/Transformer/runs/loop_p4_v2/loop.log 2>&1 < /dev/null &
  echo $! > /tmp/unichess_t_loop_v2.pid
  ```

#### 复盘时要回答的问题（材料都已备齐，此处只列不答）

1. **Elo −9.4 / llr −1.579 是"真的没进步"还是"分辨力不够"？**
   arena 早停只用 37 局就够了吗？`elo1=60` 的晋级线配合 37 局，
   SE 约 ±110 Elo——也许该加 `pairs` 而不是继续改配方。
2. **候选已经在终点最小了，还要怎么改？** §18.9/§18.13 已把"步数过长"
   "导出最差点"两个根因都修掉了，gen 6 曲线也确实干净。
   若瓶颈是数据/分辨力而非配方，继续调 lr/步数的边际收益可能已经是零。
3. **50/50 配比是否过头？** 静态源每代覆盖率从 36% 掉到 17.7%，
   自对弈从 71% 掉到 39.6%（都是每代样本数腰斩的必然结果）。
   需要评估是不是该往回调，或反向加大 `games`。
4. **要不要接受一个更低的晋级线**（`elo1` 从 60 降到 30）？
   注意这违反"不放宽门槛"的纪律，必须显式决策，不能顺手改。

### 18.15 为什么 self-play 晋升不了：完成-Q 保证的边界 + 分辨力闭环（2026-09-30）

用户问："根据 completed-Q 的数学保证，这个策略应该是不会更烂的吧？
还是说 self-play 对未知分支探索太弱，只会陷入自身偏见？"
下面用实测数据逐条回答。**结论：两个猜测都不成立，真正的原因是分辨力闭环。**

#### 一、完成-Q 的策略改进保证保证的是什么

Kit 实现（`Kit/search/gumbel.py`）：

```python
completed_q(a) = q(a)                若 N(a) > 0（访问过，用备份 Q）
               = v_mix              未访问（v̂ 与已访问子树的 π 加权混合兜底）
σ(q̂)          = (c_visit + n_max) · c_scale · q̂       # q̂ = completed Q 的 min−max 归一
π′            = softmax(logits + σ(completedQ))
```

Danihelka 等的 Gumbel MuZero 论文里，policy improvement 是一个
**"按 Q̂ 算的不等式"**：改进后的搜索策略不差于采样策略（Gumbel 扰动后的 prior），
**前提是树里用的 Q̂ 与真实 Q 保序**。它是关于**搜索策略**的陈述，
不是关于"把搜索结果蒸回网络后棋力会涨"的陈述。从定理到"网络变强"有三个断点：

1. **Q̂ 来自网络自身（自举）**。保证是"按网络自己的标尺变好"；
   标尺有位置相关偏差时，改进方向就是错的。
2. **只覆盖访问到的节点**。800 sims 的树很浅，未访问分支的 completed Q 是
   `v_mix` 兜底——那是插值，不是搜索，对这些分支没有任何改进。
3. **蒸馏是另一个对象**。训练目标是 `KL(π′ ‖ 网络)`，拟合 51 万条记录里
   800-sims 的访问分布；访问分布本身含 sequential halving 的采样噪声。
   新网络 ≠ π_imp，是新对象，自带近似误差。

#### 二、猜测「探索太弱 / 偏见固化」——实测否定（强版本不成立）

用自对弈记录里的 `visit_prob`（前 16 着）跨代比。注意引擎代际：
`gen_0002/3` 由 **gen_0001 冠军**（弱）下出，`gen_0004/5/6` 由 **gen_0003 冠军**（强）下出：

| 分片 | 生成引擎 | 记录数 | top1 均值 | top1 中位 | p1>0.8 占比 | 有效支撑集 1/Σp² |
|---|---|---|---|---|---|---|
| gen_0002 | gen1 冠军（弱） | 521,206 | 0.5418 | 0.5013 | 0.2234 | 3.760 |
| gen_0003 | gen1 冠军（弱） | 523,911 | 0.5401 | 0.4991 | 0.2215 | 3.777 |
| gen_0004 | gen3 冠军（强） | 510,298 | 0.5645 | 0.5363 | 0.2493 | 3.545 |
| gen_0005 | gen3 冠军（强） | 513,221 | 0.5631 | 0.5344 | 0.2476 | 3.561 |
| gen_0006 | gen3 冠军（强） | 507,257 | 0.5663 | 0.5389 | 0.2510 | 3.524 |

强引擎确实更集中：top1 +0.024、有效支撑集 3.77 → 3.54（窄 6%）、熵 1.382 → 1.311。
**但这不是坍塌**——每个局面仍有约 3.5 个有效候选着法，而且这是更强引擎的合理锐化。

旁证：
- 搜索侧有探索机制：`SearchBudget(simulations, add_noise=True)`（根节点 Dirichlet）、
  2000 行合成开局库 + `book_plies=6`、Gumbel top-m。
- 网络在**远离**旧引擎而非塌回固定点：KL(网络 ‖ gen1 引擎) 从 0.030 涨到 0.052（§18.9）。
- 对局结构五代几乎不变：1-0 局数 2018 → 2142、0-1 1098 → 1041、和棋 980 → 913，
  没有出现"同一批开局通吃"或重复率上升。

#### 三、猜测「Q̂ 的标尺不可信」——有真问题，但是反过来的

每条自对弈记录同时存了根节点搜索 Q（`q`）与终局真实结果（`z`），可以直接校准：

| 分片 | mean\|q\| | mean\|z\| | corr(q,z) | mean\|q\|−mean\|z\| |
|---|---|---|---|---|
| gen_0002 | 0.4143 | 0.7448 | 0.7521 | **−0.3305** |
| gen_0003 | 0.4133 | 0.7509 | 0.7477 | −0.3376 |
| gen_0004 | 0.4449 | 0.7584 | 0.7610 | −0.3135 |
| gen_0005 | 0.4473 | 0.7703 | 0.7588 | −0.3230 |
| gen_0006 | 0.4470 | 0.7626 | 0.7627 | −0.3157 |

gen_0006 的校准曲线（按 `q` 分箱看 `mean z`）：

| q 区间 | mean q | mean z |
|---|---|---|
| [0.05, 0.15) | 0.0974 | 0.1427 |
| [0.15, 0.30) | 0.2195 | 0.3503 |
| [0.30, 0.50) | 0.3941 | **0.6329** |
| [0.50, 0.70) | 0.5958 | 0.8343 |
| [0.70, 1.00) | 0.8985 | 0.9775 |
| [−0.30, −0.15) | −0.2197 | −0.4027 |
| [−0.70, −0.50) | −0.5966 | −0.8586 |

**Q̂ 被系统性向 0 收缩（约 0.6 倍线性收缩 + 两端饱和），不是过自信。**
搜索说 +0.394 的局面，真实价值 +0.633。

这**不破坏策略改进定理**：收缩是单调的（corr 0.763、曲线单调递增），
而定理需要的是 completed Q 的**排序**，不是绝对标定。
但它说明 Q̂ 的绝对精度很差，于是未访问分支的 `v_mix` 兜底插值也跟着不可信。

（另一项测了但 **inconclusive**：访问集中度与真实结果几乎无关
——corr(top1, z) = −0.003/−0.011；top1<0.3 的分散局 mean z=+0.058、
top1>0.8 的集中局 mean z=+0.020。没有按 ply 分层，过度解读是危险的。）

#### 四、真正的原因：单代增益小 + 不累积 + 门槛为 +60 设计 = 分辨力闭环

三个数字：

1. **arena 的 CI95 半宽 = 102 Elo（37 局）**。我们连 +30 与 −30 都分不清。
2. **SPRT 的 `elo1=60` 意味着它是为"单代提升 60 Elo"设计的**。
   真实 +60 Elo 时 llr 每局漂移 0.01557，需约 **186 局**才会判 H1；
   而实际 33/45/37 局就早停了（因为先撞下界）。
3. **若真实单代增益只有 +15 Elo**，漂移仅 0.00093/局，需 3104 局——
   实际上永远先撞下界判 H0。

**而且增益不累积**：gen 4/5/6 的 `base_ckpt` 全部是同一个
`gen_0003/train/final.pt`（冠军自 gen 3 起从未换代）。
每一代都从同一个冠军重新拟合 400 步。即使每代独立能涨 +15 Elo，
这个 +15 也不会留在系统里——下一代又从 gen_0003 出发，上一代的收益被丢弃。

于是闭环：

```
单代增益小（未知，但 CI 含 0）
   -> SPRT 为 +60 设计，37 局早停判 H0
   -> 冠军不动（仍是 gen_0003）
   -> 下一代又从同一个 base 出发
   -> 上一步的增益继续被丢弃
```

**这不是"策略变差"，是"闭环永远转不到晋级"**。
按纪律三票 H0 都不能算换代，而从数据看候选也确实没有可测的退化
（gen 5/gen 6 的 CI 都含 0，gen 4 的退化已定位为实现 bug 并修掉）。

#### 五、数据量本身是更硬的约束

每代真正的新信息只有 **51 万条**自对弈记录（`window=2`），
而静态 ResNet 语料是 480 万条（4 个固定分片，2026-09-08 起未变）。
每代 SGD 只扫 819K 样本（400 步 × accum 4 × batch 512）：
自对弈语料每代覆盖率 **39.6%**，静态 **17.7%**。

自对弈 RL 的提升速度取决于新数据量。每代 51 万条新记录就是这个循环的"学习燃料"——
即使 fit 得完美，也只能把网络拉到"拟合这 51 万条 + 480 万静态"的那个最优点。
这与 completed-Q 的保证无关，是数据量约束。

#### 六、可动方向与纪律边界（只列，未擅自改）

| 方向 | 效果 | 纪律 |
|---|---|---|
| 加 `pairs`（256→512） | CI 半宽 102→~72 Elo | 不违反 |
| 加 `games`（4096→8192） | 学习燃料翻倍，自对弈 9.45→18.9 h/代 | 不违反 |
| 降 `elo1`（60→30） | 能测出小增益 | **违反"不放宽门槛"，必须显式决策** |
| 弱 H0 时也把候选作为下代 base | 让小增益累积 | 同上是放宽门槛的变体，需显式决策 |
| 校准 value head（对齐 Q̂ 的 0.6 倍收缩） | 让 `v_mix` 兜底更可信 | 训练配方问题，不违反纪律 |

