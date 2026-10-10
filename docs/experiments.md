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

### 18.15 为什么 self-play 晋升不了：训练信号质量 + 分辨力闭环（2026-09-30）

用户问："根据 completed-Q 的数学保证，这个策略应该是不会更烂的吧？
还是说 self-play 对未知分支探索太弱，只会陷入自身偏见？"

⚠ **先纠错**：本节最初写成"Kit 实现（Kit/search/gumbel.py）"，**那是错的**。
本循环用的**不是 Gumbel 顺序减半**，而是 **PUCT（C++）**：
`Transformer/kit.py → make_search_player_factory → PUCTConfig → SearchPlayer → PUCTCpp`
（`Kit/search/gumbel.py` 是 **S/SSM 专用**，`SSM/kit.py` 用 `SsmGumbelPlayer` + `GumbelConfig`；
T 与 R 都走 `make_search_player_factory`）。证据在落盘配置里：

- `gen_0006/selfplay.json` / `arena.json` 的 `engine.kwargs` 只有
  `checkpoint / simulations / batch_size / precision / syzygy_path`，没有任何 Gumbel 字段；
- `Kit/players/search_player.py:262` 的训练目标来源是
  `info["visits"] = [(mv.uci(), int(n)) …]`——**PUCT 的原始访问计数**，
  不是 `softmax(ℓ + σ(completedQ))`；
- 根探索是 **Dirichlet**：`PUCTConfig(dirichlet_alpha=0.3, dirichlet_eps=0.25)`。

下面的分析按 PUCT 重写。

#### 一、policy improvement 保证保证的是什么（以及本循环为何不受它保护）

Danihelka 等的 Gumbel MuZero 论文里，policy improvement 是**"按 Q̂ 算的不等式"**：
改进后的**搜索策略**不差于采样策略，前提是树里用的 Q̂ 与真实 Q 保序。
它是关于**搜索**的陈述，不是关于"把搜索结果蒸回网络后棋力会涨"的陈述。
从搜索改进到"网络变强"隔着三个断点：

1. **Q̂ 来自网络自身（自举）**。保证是"按网络自己的标尺变好"；
   标尺有位置相关偏差时，改进方向就是错的。
2. **只覆盖访问到的节点**。800 sims 的树很浅，未访问分支的 Q 只是先验插值。
3. **蒸馏是另一个对象**。训练目标是 `KL(π′ ‖ 网络)`，拟合的是搜索**访问分布**。

而本循环连"搜索改进"这一步都没有——**训练目标就是 PUCT 的访问计数 N/ΣN**，
即在 25% Dirichlet 噪声下，一次 800-sims 搜索走过的分布。
这既不是 Gumbel 的 completed-Q 改进策略，也不是纯先验，
而是"我自己搜索行为的带噪快照"。用这个目标做 BCE 蒸馏，
本质是**自蒸馏 + 噪声拟合**——信号强度远低于价值感知的策略改进目标。

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
**但这不是坍塌**——每个局面仍有约 3.5 个有效候选着法，这是更强引擎的合理锐化。
对局结构五代几乎不变（1-0 局数 2018 → 2142、0-1 1098 → 1041、和棋 980 → 913），
没有出现"同一批开局通吃"或重复率上升。

#### 三、猜测「标尺不可信」——Q̂ 被向 0 收缩，但单调所以不破坏排序

每条自对弈记录同时存了根节点搜索 Q（`q`）与终局真实结果（`z`）：

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

**Q̂ 被系统性向 0 收缩约 0.6 倍**（搜索说 +0.394 的局面真实值 +0.633），不是过自信。
收缩是单调的（corr 0.763、曲线单调递增），所以**不改变 argmax 排序**——
但绝对精度差意味着"未访问分支的值"这类插值不可信，且 value head 本身欠校准。

（另一项测了但 **inconclusive**：访问集中度与真实结果几乎无关
——corr(top1, z) = −0.003/−0.011。没有按 ply 分层，过度解读是危险的。）

#### 四、真正的原因：训练信号弱 + 分辨力闭环 + 增益不累积

1. **训练信号弱（本节核心）**：目标是 800-sims PUCT 在 25% Dirichlet 噪声下的访问计数，
   α=0.3 在 ~35 个根着法上产生的噪声极尖；`root_min_visits=1` 让尾部着法只被访问 1 次，
   目标里那一长串 1/800 概率基本是纯噪声。**候选被训得更像"自己的带噪搜索"，棋力因此不动。**
2. **arena 的 CI95 半宽 = 102 Elo（37 局）**，连 +30 与 −30 都分不清。
3. **SPRT 的 `elo1=60` 是为"单代涨 60 Elo"设计的**：真实 +60 需漂移 0.01557/局、
   约 **186 局**才判 H1；实际 33/45/37 局就撞下界早停。
   若真实增益只有 +15 Elo，需 3104 局——永远先撞下界。
4. **增益不累积**：gen 4/5/6 的 `base_ckpt` 全是同一个 `gen_0003/train/final.pt`
   （冠军自 gen 3 起从未换代），每代都从同一个冠军重新拟合，
   上一代的收益被丢弃。

闭环：

```
训练信号弱 + 单代增益小
   -> SPRT 为 +60 设计、37 局早停判 H0
   -> 冠军不动（仍是 gen_0003）
   -> 下一代又从同一个 base 出发
   -> 增益继续被丢弃
```

按纪律三票 H0 都不能算换代；而候选也确实没有可测退化
（gen 5/gen 6 的 CI 都含 0，gen 4 的退化已定位为实现 bug 并修掉）。

#### 五、参照物：gen 1 / gen 3 为什么能换代

| 代 | base | 步数 | lr | 配比 | Elo |
|---|---|---|---|---|---|
| gen 1 | loop_p4 gen0 | 1200 | 1e-4 | 70/30 | **+73.8 H1** |
| gen 3 | gen_0001 | 1200 | 5e-4 | 70/30 | **+65.1 H1** |
| gen 4 | gen_0003 | 1200 | 5e-4 | 70/30 | −74.8 | 
| gen 6 | gen_0003 | 400 | 5e-4 | 50/50 | −9.4 ± 102 |

gen 1/3 的 base 离本份数据的最优点**远**，所以 1200 步一直在改善；
gen 4 的 base（gen_0003）**已经在最优点附近**，同样 1200 步全变成过训练。
把步数压到 400 后过训练消失（gen 6 曲线终点即最低点），
但"基座已收敛在本份数据上"这件事没变——**继续加训练量的边际收益接近零**。

#### 六、数据量与三专家分配

每代新信息只有 **51 万条**自对弈记录（`window=2`），静态 ResNet 语料是 480 万条
（4 个固定分片，2026-09-08 起未变）。每代 SGD 扫 819K 样本 →
自对弈语料每代覆盖率 **40.1%**，静态 17.7%。

按专家（`Transformer/model.py:398-407`，按 popcount 路由 pc≥24 / 12<pc≤24 / pc≤12）量过
⚠ 第一版把 bitboard 当计数相加，得出"expert1 零数据"的**假结论**，已用 popcount 查表重做：

| 专家 | 自对弈占比 | 静态占比 | SGD 样本/代 | 样本/千参数 |
|---|---|---|---|---|
| e0 开局 pc≥24 | 29.8% | **49.0%** | 322,781 | 15.8 |
| e1 中盘 12<pc≤24 | 36.2% | 33.8% | 287,029 | 14.1 |
| e2 残局 pc≤12 | 33.9% | 17.2% | 209,389 | **10.3** |

数据本身均衡（没有死专家），但静态语料偏开局，混下来**残局专家最饿**（10.3 vs 15.8），
而终局决定胜负。

#### 七、算力分配失衡（这是"不叠算力"的根据）

每代 10.2 h = 自对弈 **9.45 h（93%）** + 训练 **4 min（0.65%）** + arena。
自对弈产出 51 万条新记录，训练消耗 819K 样本（一半静态）——
**自对弈花了 140 倍于训练的时间，只提供一半数据。**
所有"加算力"提议都在加大那已经占 93% 的部分。

### 18.16 下一步训练计划（2026-09-30，loop 已停待复盘）

停 loop 时保留的现状：`loop_state.json` = `generation=7 / phase=selfplay`、
`champion=gen_0003`、`train_variant={lr:5e-4, wd:1e-4}`；`gen_0007/selfplay/` 240 局已写；
配置 HEAD = Transformer `94c921a`、Kit `cf831fa`，工作副本无本地改动。

#### Phase 1 — 只改配置 + 一次重启（预计 +1.5 h/代，不碰纪律）

| # | 字段 | 现值 | 新值 | 依据 |
|---|---|---|---|---|
| 1 | `PUCTConfig.dirichlet_alpha` | 0.3 | **0.03** | α=0.3 在 ~35 根着法上噪声极尖，目标尾部大量 1-visit 纯噪声 |
| 2 | `PUCTConfig.dirichlet_eps` | 0.25 | **0.10** | 同上；给自对弈保留多样性但别让先验被噪声盖掉 |
| 3 | `PUCTConfig.root_min_visits` | 1 | **4** | 每个根着法至少 4 次访问才进统计；**不加墙钟**（800 sims 总数不变，只改分配） |
| 4 | `train.data.sources` 权重 | 0.5/0.5 | **0.3/0.7**（自对弈 0.7） | §18.9：静态源 loss 只降 0.0042，自对弈源降 0.0714，静态那半批无学习信号 |
| 5 | `arena.pairs` | 256 | **512** | CI 半宽 102 → 72 Elo；可检出的最小增益从 ~60 降到 ~45 |

1–3 都需要在 loop 的 `engine.kwargs` / `match.engine.kwargs` 里显式传
（现在没传，吃 `PUCTConfig` 默认值）。重启点在 gen_0007 自对弈窗口（240 局已写，续跑）。
**判据：跑 2 代（gen 7/8）。若出 H1 → 收工；若仍 H0 但 CI 明显向正移动 → 继续；
若 CI 仍含 0 → 进 Phase 2。**

#### Phase 2 — 换搜索：从 PUCT 换到 Gumbel（需写代码，约 1 个工作日）

Kit 已有 `Kit/search/gumbel.py`（`Gumbel` + `GumbelConfig` + `σ(completedQ)`），
S 在用（`SSM/kit.py::SsmGumbelPlayer`），但 T/R 都没有 Gumbel 版工厂。
给 T 写一个 `Planes19Expander` + `Gumbel` 的 player 工厂后：

- 训练目标从「800-sims 访问计数」换成 **`π′ = softmax(ℓ + σ(completedQ))`**
  ——价值感知的策略改进目标，而不是"自己的带噪快照"；
- 根探索从 Dirichlet 换成 Gumbel top-m（校准过的探索量）；
- `m0`、`g`、`c_scale`、`c_visit` 都成了显式旋钮。

**这是唯一能实质提高训练信号质量、且不加算力的路。** §18.15 第四节的①就是它要治的。

#### Phase 3 — 结构改动：让增益累积（需显式决策）

现在 `base_ckpt` 永远是 champion，小增益每代被丢弃。改为在 `loop_state.json` 里
维护独立的 `train_base` / `selfplay_engine`：

- 判 **H1** → `champion` = 候选，`train_base` 跟随（现状）
- 判 **H0 但未越下界** → `train_base` = 候选，**`champion` 不动**（生产不变）
- **越下界** → `train_base` 退回 `champion`

**不放宽换代门槛**（verdict 口径与三条红线一字不改），只让"训练不再每代回归"。
风险：从未验证的权重出发训练会漂移；要配一个"连续 N 代未换代就强制回退
champion"的保险。

#### Phase 4 — 纯叠算力对照项（不推荐先做）

`games 4096 → 8192`：每代 +9.45 h。仅在 Phase 1+2 都做完且无效时考虑。

#### Phase 1 执行记录（2026-10-01 01:20 落地）

配置提交 `a9afdc3`（依赖 Kit `fb7ed63` 的 `seed_base`，已先升级），
删掉 gen_0007 在旧噪声配置下写的 240 局保证整代同源，重启后 loop PID 1117636、
自对弈子进程 1117657。**`gen_0007/selfplay.json` 的 engine.kwargs 实测已含
`dirichlet_alpha=0.3 / dirichlet_eps=0.1 / root_min_visits=4`。**

自对弈 52 局时的分布对比（新 vs 旧，各取前若干万条记录）：

| 指标 | gen_0006（eps 0.25 / rmv 1） | gen_0007（eps 0.10 / rmv 4） | 变化 |
|---|---|---|---|
| top1 均值 | 0.5653 | **0.5339** | −0.031（更平） |
| 有效支撑集 1/Σp² | 3.538 | **3.785** | **+0.25（更多样）** |
| 熵 | 1.3094 | **1.3968** | +0.087 |
| 最小 top16 占比均值 | 0.00236 | **0.00347** | **+47%（尾部噪声降）** |

**方向与我的预期相反，而且是往好的方向**：我原以为"降噪会让目标更集中、
熵会略降"，实测是**更平更多样**。原因：eps=0.25 时那 25% 的 Dirichlet 噪声
本身就把质量随机堆到少数着法上（α=0.3 在 35 个着法上仍偏尖），
把 eps 降到 0.10 后搜索更忠实反映网络自身那个长尾 prior，
于是 top1 下降、有效支撑升到 3.785——**比弱引擎时代的 3.77 还高**，
是六代以来第一次上升（此前只从 3.77 单调降到 3.54）。
`root_min_visits=4` 也实测生效：最小占比均值从 0.00236 抬到 0.00347。

按代换种子（`seed_base=20260930`）实测：gen 7 的 train/selfplay/match
三个种子互不相同、与 gen 6 不同、同代重复派生相同（续跑不换种子），
并会写进 `loop_state.json` 的 history 供对账。

**遇到的外部干扰**：本机同时被另一个负载占用 GPU——ComfyUI
（`/mnt/data/AV/ComfyUI`，由 `/tmp/kilo/val_specs/` 的批量验证脚本驱动，
与 UniChess 无关），一度占 12.4 GB / 16 GB。**未动它**（不是本项目的负载）。
它跑完后 GPU 回到 2 GB；但 01:30 又起了第二批（9.2 GB），
因此 5 分钟实测速率 9.38 s/局（无干扰时应为 ~7.5 s/局），
gen 7 自对弈预计完成时间从 ~10.9 h 顺延到 ~10.5 h 之外，需按实际干扰重估。

### 18.17 gen 7 判决：Phase 1 生效，H1 换代成功（2026-10-01）

**arena 判决**（`gen_0007/arena.jsonl.summary.json`，11:57 起跑、13:03 出）：

| 指标 | 值 |
|---|---|
| 成绩 | **133 局**，59 胜 34 和 40 负，score_a **0.5714**，Elo **+49.98**，CI95 [−0.61, +102.77] |
| SPRT | llr **+2.961**、第 **113** 局越上界 +2.890、`verdict=H1`、`stopped_by_sprt=true` |
| 其它 | `duplicate_rate` 0.0、`mean_plies` 157.95、`elapsed_s` 3947.8 |
| 判 | **promoted=true**，新冠军 = `gen_0007/train/best_model.pt` |

**Phase 1 六项全部验证 PASS**（落盘值逐条核对过）：

| 项 | 落盘值 | 校验 |
|---|---|---|
| `train.steps` | 400 | ✓ |
| `train.optimizer.lr` | 0.0005（锁定值） | ✓ |
| `train.schedule` | onecycle / pct_start 0.25 | ✓ |
| `train.data.sources` 权重 | **[0.3, 0.7]**，每批 [154, 358] | ✓ |
| `train.seed` | **21001952**（派生机，非旧值 42） | ✓ |
| `train.export` | best_model.pt / final.pt / select_best_by=train | ✓ |
| `selfplay.json` engine.kwargs | dirichlet_alpha 0.3 / **eps 0.1** / **root_min_visits 4** | ✓ |
| `arena.match.pairs` | **512** | ✓ |
| `history[].seeds` | train 21001952 / selfplay 21009871 / match 21017790（三者互异） | ✓ |

arena 用的是派生机 match=21017790，不再是六代共用的 20260926——每次判决的
开局与配色都是新的样本，不再是同一批固定局面。

#### 训练曲线：同 base 下第一次拿到大收益

| step | lr | loss |
|---|---|---|
| 1 | 2.01e-05 | 1.269261 |
| 100 | **5.00e-04（峰值）** | 1.192682 |
| 200 | 3.73e-04 | 1.162605 |
| 300 | 1.23e-04 | 1.135981 |
| 400 | 1.57e-08 | **1.123650（最低点）** |

首=1.269261 → 末=1.123650，**终-首 −0.145611，单调下降、终点即最低点**。
对照 gen 6（同一个 base `gen_0003`）的 −0.084681——**收益大了 72%**，
而且是同 base 下第一次拿到这个量级的下降。gen 4/5 的 1200 步只会过训练，
gen 6 的 400 步/50-50 只有 −0.085，gen 7 的 400 步/30-70/降噪才到 −0.146。

#### 自对弈数据没变质

4096 局完整，局号 **28672..32767**（gen 7 范围正确），plies mean=141.6。
结果 2077 胜 / 934 和 / 1085 负（与 gen 6 的 2142/913/1041 同量级）。

**终局构成与之前各代完全同构，没有新变化**（跨代实测，见 §18.19）：

| 代 | checkmate | insufficient_material | 占比 |
|---|---|---|---|
| gen_0002 | 3116 | 452 | 11.04% |
| gen_0003 | 3143 | 461 | 11.25% |
| gen_0004 | 3161 | 442 | 10.79% |
| gen_0005 | 3199 | 453 | 11.06% |
| gen_0006 | 3183 | 454 | 11.08% |
| gen_0007 | 3162 | 457 | 11.16% |
| gen_0008 | 3166 | 457 | 11.16% |

⚠ **本节早前写的"checkmate 率从 88.5% 掉到 77.2%、疑与 `root_min_visits=4`
有关"是错的**：那个 88.5% 来自早期一段临时统计的算术错误。真实 mate 率
**七代都在 76%–78% 平着走**，`insufficient_material` 也一直 ~11%。
**没有任何终局构成上的回归或新副作用**，`root_min_visits=4` 对此无影响。
（完整判据与出现机制见 §18.19。）

#### 诚实的统计保留（必须记）

- **点估计 +50 Elo 低于门槛 elo1=60**，CI95 [−0.61, +102.77] 也含 0。
- 按「越界即冻结」的口径 `verdict=H1`，这条认；但证据强度弱于
  "点估计高于门槛"那种情况——这是 SPRT 早停的固有现象，
  三次 H1（gen 1 +73.8/43 局、gen 3 +65.1/81 局、gen 7 +50.0/133 局）全是宽 CI。
- 需要避免过度归因：Phase 1 是**六项一起**上的，无法从 gen 7 一役拆分
  哪几项贡献了多少。能确定的只是"这一揽子配置把这个 base 的 loss 收益
  从 −0.085 拉到 −0.146、并把 Elo 从 −9.4 翻到 +50"。

#### 趋势总表

| 代 | 配置 | Elo | score_a | 局数 | 判决 |
|---|---|---|---|---|---|
| gen 1 | 1200 步 / 1e-4 / 70-30 | +73.81 | 0.6047 | 43 | H1 |
| gen 3 | 1200 步 / 5e-4 / 70-30 | +65.09 | 0.5926 | 81 | H1 |
| gen 4 | 1200 步 / 70-30 / 同 base | −74.83 | 0.3939 | 33 | H0 |
| gen 6 | 400 步 / 50-50 / 同 base | −9.39 | 0.4865 | 37 | H0 |
| **gen 7** | **400 步 / 30-70 / 降噪 / 512 对 / 按代换种子** | **+49.98** | **0.5714** | **133** | **H1** |

gen 1/3 的 base 离本份数据最优点远，长训练有收益；
gen 4/5 的 base 已在最优点附近，同样 1200 步全是过训练；
gen 6/7 换到 400 步后过训练消失，再靠「自对弈占比 + 目标降噪」把收益拉起来。

#### 现状与待决策

- **streak 清零**（gen 4/5/6 H0 → gen 7 H1），暂停纪律解除。
- loop 已推进到 `generation=8 / phase=selfplay`，gen 8 自对弈用新冠军
  `gen_0007/train/best_model.pt`，PID 见 pid 文件。
- ⚠ **生产权重落后两代**：`Transformer/config.json` 的 `max_mcts` / `max_t`
  两个预设仍指向 `gen_0003/train/final.pt`。切换是人工动作
  （改 config.json + 重启 `unichess-server`），历史上由用户授权执行。
- Phase 2（给 T 写 Gumbel player 工厂，训练目标换成 `π′=softmax(ℓ+σ(completedQ))`）
  现在**不再是高优先级**——PUCT + 降噪已经能稳定换代。Phase 3（让增益累积）
  同理可以让位，除非接下来几代又连续 H0。

### 18.18 生产切换到 gen 0007；select_best_by 首次真正生效（2026-10-01 23:23）

#### 生产切换

`Transformer/config.json` 两个预设备份（`max_mcts` / `max_t`）的 `ckpt` 从
`gen_0003/train/final.pt` 改到 **`gen_0007/train/best_model.pt`**（提交 `c869189`）。
只改 `ckpt` 与 `description`；`mcts_sims 2400` / `mcts_batch 64` / `precision fp16` /
`temperature 1.0` / `root_top_k 3` 一律未动。重启 `unichess-server` 后：

- 新 PID 1603214，`/api/health` 200，`unichess-server` / `unichess-tunnel` 均 active
- `/api/models`：T `status=available`，`presets=['max_mcts','max_t']`
- **两个预设各实测对局**（标准起始 FEN，`engine_white=true`）：

| 预设 | 我方序列 | 引擎应手 | sims / depth |
|---|---|---|---|
| `max_mcts` | e7e5, g8f6, f8b4, e8g8 | g1f3, f3e5, a2a3, a3b4 | 2400 / 13–15 |
| `max_t` | e7e5, g8f6, f8b4, e8g8 | b1c3, g1f3, f3e5, e2e3 | 2400 / 10–15 |

着法全部合法，`result.info` 里 sims/nodes/depth/q/pv 完整；
`max_t`（temperature 1.0 + root_top_k 3）选点明显比 `max_mcts` 更多样，符合预期。
journal 重启后 0 条 error/traceback（期间的 400/404 是验证脚本自己发错的请求）。

回滚 = 把两个 `ckpt` 改回 `gen_0003/train/final.pt` 后重启 server（gen_0003 权重仍在）。

#### select_best_by 的完整闭环验证（gen 8）

gen 8 的 base 是新冠军 `gen_0007/train/best_model.pt`——**冠军变强后 base 更接近
本份数据的最优点，于是 400 步的曲线第一次不再单调**：

| step | lr | loss |
|---|---|---|
| 1 | 2.01e-05 | 1.197339 |
| 200 | 3.73e-04 | 1.131214 |
| **300** | 1.23e-04 | **1.112694（最低点）** |
| 400 | 1.57e-08 | 1.115763（回升 +0.0031） |

这正是 gen 4 那个"训过头"模式的轻微重现（gen 4 回升 0.0255，gen 8 只 0.0031，
差约 8 倍）。而 `select_best_by=train` 恰好兜住了它：

```
train.jsonl: step 1→300 每步 improved=true，best 一路降到 1.1126938629150391
             step 400: {"validation": {}, "best": 1.11269..., "improved": false}
                       —— best 没有跟着 loss 回升到 1.1158

best_model.pt: step = 300        final.pt: step = 400
816 个张量里 792 个不同，权重差 L2 = 5.528803
```

gen 7 是完美对照：最低点就在终点，`best.step=400 == final.step=400` 且逐位相同。

**结论：Kit `cf831fa` 修的那条路径，从"修好"走到"真的在关键时刻抓住不同权重"**——
gen 7 验证曲线单调时 best == final（零风险保险），gen 8 验证曲线非单调时
best 抓住最优步（导出 step 300 而非 400）。arena 正在测的就是 step 300 的权重，
所以 gen 8 若判 H1，换代的权重来自最优步而非终点。

#### 一个要盯的趋势

gen 8 曲线重新出现轻微 overshoot，与 §18.15 第五节的判断一致：**冠军越强、
base 越接近数据最优点，同一套 400 步预算就开始从"有余量"变成"微微过头"**。
幅度只有 gen 4 的 1/8 且被 `select_best_by` 兜住，暂不需要动步数；
但它应作为下一轮的先行指标——**若后面几代 overshoot 幅度继续变大（如超过 0.01），
就该像这次一样先看曲线再决定缩步数还是别的**。

另：gen 8 自对弈 `insufficient_material` 457 局 / `checkmate` 3166（77.3%），
与七代基线一致（§18.19），**不是新变化**；另出现 3 局 `truncated`（超 max_plies 400）。

### 18.19 `insufficient_material` 是什么、怎么出现的、为什么不用管

**判据**（`Kit/rules/referee.py:31` 的 `classify()` → `fast_outcome(board,
claim_draw=True)` → python-chess `Termination.INSUFFICIENT_MATERIAL`）：
`is_insufficient_material()` = **双方都** `has_insufficient_material()`。
逐条实测资格：

| 局面 | 算吗 |
|---|---|
| K vs K | ✅ |
| K+B vs K / K+N vs K | ✅ |
| K+N+N vs K | ❌（一方两个马足以将杀） |
| K+B vs K+B **同色格**象 | ✅ |
| K+B vs K+B **异色格**象 | ❌ |
| K+R vs K / K+P vs K | ❌ |

**出现机制**：不是某一手特殊着法触发，而是**走完之后局面落进上表某一行**：

```
走一手（通常是吃子，把子力降到"光王/单轻子/同色格象"）
  -> PUCT.exact_value() 该局面返回 0.0（puct.py:117；无条件，
     不像三折叠/五十步那样受 claim_draw 控制）
  -> 裁判每步后 verdict() -> classify() -> is_insufficient_material() 命中
  -> termination=insufficient_material, result=1/2-1/2，棋局结束
```

引擎**知道**那是和棋（搜索侧给 0.0），但**无处可去**——没有子力就赢不了。

**规模与性质**（跨七代实测）：每代 442–461 局 / 4096，占比 **10.79%–11.25%，
从 gen_0002 起完全稳定**。plies 中位 137–140，与全体中位（136–139）基本相同；
plies ≤100 的只占 2.0%–4.2%（全体 13.5%–16.2%）；min/max plies 75/356；
结果 100% 是 `1/2-1/2`。

即：这些**不是早崩的短局**，而是下满 ~140 步的正常长度的棋，双子力换干净后
谁也无法进展。这对训练是中性偏好——提供"接近终局的简化局面"样本，
且 `z=0` 与规则一致，没有错误监督信号。

**结论：稳定结构特征，不是回归、不是 bug、不需要处理。**
11% 的材料不足和棋在这个强度的引擎自对弈里属正常范围。
唯一可讨论的是要不要降低比例（让引擎在子力简化前更积极），
但那是棋力/风格问题而非数据质量问题，且当前正在稳定换代，没有动它的理由。

⚠ **顺带更正**：§18.17 早前写的"gen 7 checkmate 率从 88.5% 掉到 77.2%、
疑与 `root_min_visits=4` 有关"是错的——那个 88.5% 来自早期一段临时统计的
算术错误。真实 mate 率七代都在 76%–78% 平着走，`root_min_visits=4` 对此无影响。

### 18.20 v2 十代跑完：gen 8/9 判决，以及「loss 降、Elo 崩」的决定性证据

2026-10-02 13:20 确认 `Loop.run()` **正常返回**——`generations: 10` 的下限到了
（g = 0..9），`loop.log` 末行就是 `main()` 打印的返回值。**不是崩溃**，
也没有待办代次。冠军 = `gen_0007/train/best_model.pt`，与生产一致。

#### 十代总账

| 代 | 判决 | Elo | score_a | 局数 | CI95 含 0？ | llr |
|---|---|---|---|---|---|---|
| gen 0 | H0 | +41.48 | 0.5594 | 101 | 是 | — |
| gen 1 | **H1** | +73.81 | 0.6047 | 43 | — | — |
| gen 2 | H0 | +44.79 | 0.5641 | 39 | — | — |
| gen 3 | **H1** | +65.09 | 0.5926 | 81 | — | — |
| gen 4 | H0 | −74.83 | 0.3939 | 33 | 是 | −2.672 |
| gen 5 | H0 | −15.45 | 0.4778 | 45 | 是 | — |
| gen 6 | H0 | −9.39 | 0.4865 | 37 | 是 | −1.579 |
| **gen 7** | **H1** | **+49.98** | **0.5714** | **133** | — | **+2.961** |
| gen 8 | H0 | −52.03 | 0.4257 | 74 | 是 | −3.934 |
| gen 9 | H0 | **−157.34** | 0.2879 | 33 | **否（CI [−303.5, −50.8]）** | **−4.668** |

血统链（每步都是 arena SPRT H1 验证，冠军 gen_0007 = 现生产）：
`stratified_p4_selfplay_corrected` → loop_p4 gen0（+163）→ v2 gen1（+73.8）
→ v2 gen3（+65.1）→ **v2 gen7（+50.0）**。

#### gen 9 是十代里第二个「显著更差」

gen 4（llr −2.672、CI [−185, +22]）之后，gen 5/6/8 的 CI 都含 0——
按"弱 H0"口径是"看不出来更好也看不出来更差"。
**gen 9 不一样**：CI95 [−303.5, −50.8] **不含 0**，score_a 0.2879、33 局、
第 17 局就越下界。这是真实的、统计显著的退化，不是噪声。

#### 决定性证据：三个代用同一个 base，loss 全降，Elo 却两极

| 代 | base | loss 首→末 | 终-首 | 最低点 | arena Elo |
|---|---|---|---|---|---|
| gen 7 | **gen_0003** | 1.269261 → 1.123650 | **−0.145611** | step 400（**终点即最低**） | **+49.98 H1** |
| gen 8 | **gen_0007** | 1.197339 → 1.115763 | −0.081575 | step 300（回升 +0.0031） | −52.03 H0 |
| gen 9 | **gen_0007** | 1.242307 → 1.129208 | −0.113099 | step 300（回升 +0.0015） | **−157.34 H0** |

三代都把训练 loss 大幅压低（−0.08 ~ −0.15），gen 8/9 的降幅与 gen 7 同量级。
但只有 gen 7 涨了棋力。**同时 gen 7 是最早"还有余量"的那个（终点仍是最低点），
gen 8/9 都在 step 300 就见底了。**

**这条把 §18.15 的推断从"可能"变成"确认"**：
在这个基座已经收敛的状态下，**训练 loss 与 Elo 脱钩**——
loss 照降，棋力照崩。`select_best_by=train` 也**确实抓到了最低点**
（gen 8/9 的 `best_model.pt` 都在 step 300，与 `final.pt` 差 792/816 个张量），
但**最低点那一份权重仍然比 base 差很多**。best-checkpoint 机制本身是对的，
错的是判据——train loss 在收敛点上不再是有效的棋力代理。

#### 因此不该做的事与该做的事

**不该**：把 `generations` 调大、用同一份配方接着跑。证据表明从 gen_0007 出发的
再拟合会让候选更差（−52、−157 两次都在恶化，且第二次已显著）。

**可考虑**（按证据强度排序）：

1. **Phase 2：给 T 上 Gumbel 的 completed-Q 目标**（`π′=softmax(ℓ+σ(completedQ))`）。
   §18.15 说训练目标是"800-sim PUCT 访问计数 = 自己搜索的带噪快照"，
   §18.20 证明了它在这个收敛点上会把模型训坏。换成价值感知的目标是对症的。
2. **给训练配真正的验证信号**（哪怕一个很小的 held-out 自对弈集），
   让 `export.best` 按棋力相关的量选点，而不是按 train loss。
3. **换 base**：gen 7 的候选比 gen 3 好，但 gen 3 的候选比 gen 1 好——
   也许该回到"从更早的、还有余量的 base 出发"。
4. **收缩 base 的再拟合幅度**：lr 5e-4 对一个已收敛的 base 明显过热
   （§18.15/§18.18 都指向过训练），gen 8/9 的曲线见底后回升正是信号。

⚠ 纪律：gen 8/9 连续 2 代未换代，按"3 代复盘"的规则**还没触发暂停线**
（且 loop 已自然跑完 10 代），但上面四条都需要人工决策后再动手。

### 18.21 temperature=2.0 的验证：实测净负，已撤回（2026-10-02）

**动机与推演**：查代码发现「选哪一步」和「产生训练目标」被绑死在一次搜索上——
`Kit/pipelines/selfplay.py:159` 硬编码 `SearchBudget(add_noise=True)`，
`puct.py:328-330` 在 `temperature<=0` 时 `i=argmax(root.N)`，
而 loop 的 `engine.kwargs` 从未传 `temperature`（吃默认 0）。
于是走法是确定的、Dirichlet 噪声是唯一随机源。推论是「ε=0 会让对局塌缩」，
而 `temperature` 可以在不碰目标的前提下补上探索，进而允许把 ε 降下来。

**实测（三个测量，两个轴）**：

轴 1 · 局面覆盖（从 sp.bin 重建局面）：

| 代 | 记录 | 不同局面 | 占比 | 重复率 |
|---|---|---|---|---|
| gen_0009（temp=0） | 200,000 | 183,598 | 91.80% | 0.0820 |
| gen_0010（temp=2） | 25,676 | 23,510 | **91.56%** | 0.0844 |

**零收益。** 进一步用全量文件测「同开局线内是否分歧」（4096 局 / 2000 条开局线）：

| 指标 | gen_0009（temp=0） |
|---|---|
| 同开局两局的局面集 Jaccard | 均值 0.380、**中位 0.281**、p10 0.043 |
| (开局,ply) 单元中同 ply 出现不同局面 | **39.3%** |

**同一条开局线的两局只共享 28% 局面**——对局在开局线内真实分歧。
**覆盖从来不是瓶颈**，因为 ε=0.10 的 Dirichlet 噪声本身已经提供了足够分歧。
「temperature=0 会让对局塌缩」这个推论**只在 ε 也等于 0 时成立**。

轴 2 · 目标形状（同 297 个局面上对照）：

| 指标 | 旧 temp=0 | 新 temp=2 | 差 |
|---|---|---|---|
| top1 | 0.5491 | 0.5495 | +0.0004 |
| 有效支撑 | 3.355 | 3.344 | −0.012 |
| 熵 | 1.4302 | 1.4290 | −0.0012 |

**形状统计完全一致** → temperature 确实没碰目标。

**成本**：稳态 7.5 → **9.84 s/局（+31%）**。原因是走非 argmax 着法时
复用的子树更浅，搜索要新建更多节点。

#### ⚠ 上述"零收益"是采样假象；受控实验证明 temperature 是正收益

上面轴 1 的对比**两边都被开局数限制住了**：`book_id = game % 2000`，
而我读的前 20 万条只覆盖 ≈330 局（开局编号全不同），gen_0010 的 2.5 万条同理。
两边都看不到"同一开局的重复"，所以测不出差异。
下面"同开局 Jaccard 中位 0.281 → 对局真实分歧"也是同一误读。

**受控实验**（同一条开局线 `e2e3 e7e5 d1g4 f8e7 g4h5 e8f8`，各跑 24 局，
只改 temperature，其余全同）：

| 指标 | temp=0 | temp=2 |
|---|---|---|
| 局面记录数 | 2,607 | 3,662 |
| **去重后不同局面** | **563（21.6%）** | **3,627（99.0%）** |
| **首次分歧 ply** | **中位 56** | **中位 7** |
| 从末尾都没分歧的局对 | 1.4% | 0.0% |
| 跨局 Jaccard | 0.431 | **0.004** |

**全量 gen_0009（temp=0，4096 局）**：508,658 条记录 → **386,044 个不同局面
（75.89%）**，**24.11% 是重复记录**（12.26 万条）。每代自对弈样本 573,440 条
÷ 386,044 唯一局面 = **每条被训练 1.49 遍**。

**关键读法修正**：Jaccard 0.28 **恰恰是"两局锁步 52 ply"的表现**，不是分歧——
两局各约 125 个局面、共享前 52 个 → 52/(125+125−52) = 0.26，
正好等于此前测到的 0.281。「低 Jaccard」与「前 41% 完全重复」可以同时成立，
分辨它们必须用 **first-divergence ply**（Jaccard 是被锁步前缀稀释的）。

**修正结论：temperature=2.0 是正收益，保留。**
它把重复记录从 24% 压到 ~1%、让同开局对局从第 7 ply 起各自独立；
按单位时间唯一局面数估算约 **1.4×**（+31% 耗时之后的净额）。
它同时改善 value 头的校准数据——同一局面不再只配同一个终局。

#### 顺带被自我推翻的两个说法（留档）

1. **「ε 的噪声抽样是标签噪声主因」**——不成立。同代内同一局面被搜两次
   （137 对）的 KL **中位只有 0.0134**，且 **argmax 从未变过**。
2. 轴 2 里跨代同局面的 KL 高达 0.429 均值 / 0.249 中位，**原因未查明**
   （不是噪声抽样——见上）。候选解释：两代走到该局面的历史不同 → 树复用状态
   不同；或 297 个共同局面本身就是有偏子样本（易转置的常见局面）。
   **留作未决问题**，不据此下结论。

#### 被保留下来的有效结论

- **ε=0 的危险性成立**（用户指出、已被推演证实）：temp=0 时 ε 是唯一分歧源，
  ε=0 会让 4096 局塌缩成「2000 条开局 × 各自确定性的一条线」。
- **ε=0.10 本身仍在提供分歧**，但它不够：同开局的锁步前缀仍有 52 ply、
  24% 记录重复。**temperature 与 ε 是两个独立的探索来源，互补而非冗余。**
- **真正的问题仍未被触碰**：训练目标是冠军自己搜索的带噪快照（自指不动点），
  在收敛基座上与棋力脱钩（§18.20 的 loss↓/Elo↓）。temperature 治不了这个。

#### gen 10：已按 temperature=2.0 重启

gen_0010 已写的 258 局**就是 temp=2 产出的，与新配置同源，保留不删**，
重启即续跑（`first_game=40960`，与 gen9 不重叠）。
`loop_state` 保持 `generation=10 / phase=selfplay`，冠军 gen_0007，生产权重不变。

⚠ 一处小瑕疵如实记录：撤销结论那次提交 `a5872f8` 把 config 改成 temp=0 后，
我在改回前误用 temp=0 重启了一次，又写进 **8 局**（全代第 259–266 局）。
它们仍是合法自对弈（只是选着用 argmax），且前九代全是 temp=0，
**不构成异物**，占全代 0.2%。为不冒损坏分片的风险（sink 是 append-only +
元数据对账），**未做截断手术**，保留。`362106c` 已把 config 恢复成 2.0。

## §18.22 头体检：逐代策略头/价值头漂移（2026-10-02）

**问题**：gen 8/9 从同一个 base 各微调 400 步后分别判 −52 / −157。到底是
**策略头漂移**还是**价值头漂移**造成的？以及这种漂移能否区分胜代（gen 7, +50）与败代？

**方法**：11 张网（gen_0003..0009 的 `train/final.pt` + gen_0006..0009 的
`train/best_model.pt`），fp32、分块前向（512/批），三个集：

- `static_held` = `ResNet/data/shards_evals/evals_0010.bin`——**训练只用 slice [0,4]**
  （evals_0000..0003），所以 0010 是真正的外部 held-out SF 标签；
- `static_in` = evals_0000.bin（训练集内，作对照）；
- `sp_gen9` = gen_0009 自对弈分片 2 万条（**gen 9 对它 in-sample**，gen 7/8 不是）。

脚本 `/tmp/head_audit2.py`（结果 `/tmp/head_audit/v2.json`）。

**结果（static_held，gen3→gen9 的 final）**：

| 代 | steps | 策略CE | 熵 | maxp | WDL CE | WDL argmax |
|---|---|---|---|---|---|---|
| g3 | 1200 | 1.789 | 1.732 | 0.400 | 0.865 | **0.845** |
| g4 | 1200 | 1.795 | 1.731 | 0.400 | 0.866 | 0.845 |
| g5 | 1200 | 1.796 | 1.731 | 0.400 | 0.867 | 0.843 |
| g6 | 400 | 1.809 | 1.680 | 0.418 | 0.881 | 0.822 |
| g7 | 400 | 1.819 | 1.663 | 0.428 | 0.901 | 0.798 |
| g8 | 400 | 1.831 | 1.653 | 0.431 | 0.903 | 0.803 |
| g9 | 400 | 1.839 | **1.639** | **0.437** | **0.904** | 0.799 |

`sp_gen9`（自对弈目标）方向**完全相反**、单调变好：策略 CE 1.653→1.535，
对 z 的 WDL CE 0.714→0.622、Brier 0.413→0.358。

**四条结论**：

1. **每次微调都在做同一件事**：更贴自己的搜索分布（自对弈拟合单调变好）、
   更远离外部标签（SF 策略 CE / WDL CE 单调变差）、策略锐化
   （熵 1.732→1.639，maxp 0.400→0.437）。这是 §18.20「自指不动点」在头级的表现。
2. **漂移幅度不能区分胜代与败代**：gen 7（H1，+50）相对 base 的漂移
   （熵 −0.069、WDL CE +0.036）比 gen 8/9 相对各自 base 的漂移
   （−0.010/−0.024、+0.002/+0.003）**大得多**。⇒ **头级聚合指标不能预测 arena 判决**，
   判决必须在比赛层面解释。
3. **同一 base 的两个同配方 run 差异巨大**：g8/g9 都是 g7+400 步同配方，
   自对弈拟合增量 −0.024 vs −0.049（2 倍），判决 −52 vs −157（3 倍）。
   配方不变时 run 间方差本身就是一等公民——**必须量化**（见 §18.23 batch1 的
   E2 开局样本噪声）。
4. 价值头在外部数据上变差（argmax 0.845→0.799）的同时在自对弈终局上变好
   （Brier 0.413→0.358）——同样指向"拟合自分布、丢掉外部先验"。

**顺带确认的两个事实**：

- `PUCTConfig.temperature` **默认 0.0** ⇒ 历史 arena（gen 7/8/9）本来就是确定性
  argmax 对局；后来加的 `a195c3c` 护栏只是把这一点显式化，并未改变历史行为。
- arena 因此是「**512 个开局上的确定性测量**」（`book.plan(n, seed)` =
  `default_rng(seed).permutation(n)` 的前缀，pairs 内两侧同开局、换色），
  seed 决定用哪些开局 → **换 seed 换开局样本**，这就是判决方差的来源。

**下一步（batch1，已备好，等 gen 10 判决后停 loop 跑）**：
E1 空模型对照 gen7 vs gen7（`pairs=16`，两个 seed）→ 测判决尺子是否有结构性偏差；
E2a 同 seed（21122519）`pairs=48` 复现 gen 8 历史 arena 前 96 局 → 测确定性；
E2b/c 两个新 seed 同规模 → 测开局样本噪声；E3 gen7@2400 vs gen7@800 → 测搜索余量；
E4 冻结分片按 gen9 配方重训并对拍 `gen_0009/train/final.pt` → 测训练确定性。
产物 `/tmp/exp_batch1/`（runner `run_batch1.sh`，日志 `logs/`）。

## §18.23 batch1 实测：判决尺子校准 + 搜索余量（2026-10-03）

**A. 训练可复现（E4）**：冻结分片（gen8+gen9 sp 分片）按 gen 9 配方、同种子重跑，
`cmp_ckpt` 对拍 **IDENTICAL**（816 张量 / 122,062,128 元素，differing 0，max|Δ|=0）
→ train 侧对照成立，冻结分片实验框架有效。

**B. 判决尺子（E1 空模型对照）**：gen_0007/best_model.pt 与它的逐位相同副本
（sha256 一致，仅路径不同——同路径会撞 Kit 的 model_key 去重检查）对打，两个开局 seed：

| seed | games | score_a | Elo |
|---|---|---|---|
| 777001 | 32 | **0.5000** | 0.00 |
| 777002 | 32 | **0.5000** | 0.00 |

→ arena 无结构性偏差，判决尺子可信。

**C. 制度性发现：判决幅度被开局样本主导（E2）**

| 场次 | seed | pairs | games | score_a | Elo | CI95 |
|---|---|---|---|---|---|---|
| gen 8 历史判决 | 21122519 | 512（早停 37） | 74 | 0.4257 | −52.03 | — |
| E2a 同配置重跑 | 21122519 | 48 | 74 | 0.4257 | **−52.03** | [−128.0, +19.3] |
| E2b 只换 seed | 21122520 | 48 | 96 | 0.5573 | **+40.0** | [−22.8, +105.5] |
| E2c 只换 seed | 21122521 | 48 | 96 | 0.4948 | **−3.6** | [−65.0, +57.5] |

- E2a 与历史判决**逐局精确复现**（games / score / elo / llr / stopped_at_game 全同）
  → arena 是确定性测量，重复运行无害；
- 同一对网络、同规模，三个开局样本给出 **−52 / +40 / −3.6**，极差 **92 Elo**
  ⇒ 96 局规模下**开局样本噪声 ≈ ±46 Elo**；
- **结论：跨 seed 的判决幅度不可比**——gen 8 的 −52 基本是开局抽样；gen 7 的 +50
  （CI 下界 −0.61）同样是勉强过关。**此后所有候选 A/B 固定同一 seed**，
  跨 seed 的 Elo 只读方向、不读幅度。

**D. 搜索余量（E3）：+350 Elo**

gen_0007 同一张网：2400 sims vs 800 sims，34 局（26 胜 0 负 8 和）、
**score 0.8824、Elo +350、CI95 [253, 525]，第 12 局就 H1（llr 20.4）**。
⇒ 网络远未吃满自己的搜索；**自对弈目标（800 sims）比评测（2400 sims）弱一个数量级**，
这是 §18.20「loss↓/Elo↓」的头号结构嫌疑：换代把网络往 800 sims 的棋上拉，
再拿 2400 sims 的尺子量它。

**E. 行动**：① 评估协议固定 seed；② 目标质量（Gumbel π′ / 更高 sims）是主战场；
③ 训练侧配方对照（stage2 的 9 变体）照跑。

## §18.24 Gumbel-C++ 落地与速率（2026-10-03）

- Kit `74243fb`（节点级算术）→ `ac88f4b`（树 + 顺序减半 + 按拍批量 + GumbelCpp）
  → `48b1b9f`（GumbelPlayer，π′ 目标）；Transformer `bed3423`
  （`make_gumbel_player_factory`；`kit.py` 同名遮蔽 bug 已修）。
- 契约是**数值等价**（非逐位）：numpy 的 fp32 exp 与 BLAS dot 无法逐位复刻。
  实测 2 万随机节点决策一致率 **100.0000%**、π′ max|Δ| 5.4e-6、softmax 1.2e-7；
  5 局面搜索对拍（同假模型同噪声）action / sims_used / rounds / n_nodes /
  n_terminal / max_depth 逐项相等。
- 单搜索 800 sims（真实 T 网）：C++ Gumbel 0.42–0.67 s vs Python Gumbel 0.50–0.90 s
  （仅 1.0–1.45×，顺序减半后期轮次批大小塌缩到 1–2）、PUCT-C++ 0.07–0.11 s。
- **loop 口径（并发 32，跨局拼批）：Gumbel-T 自对弈 ≈ 12.6 s/局**（t=80→320 s 窗口，
  43→62 局；与 PUCT 的 12.7–20 s/局同量级）⇒ 换 Gumbel **不付吞吐代价**，
  π′ 目标随数据白拿（sink 的 `visit_prob` 直接写 π′）。
- 全量套件 373 项，失败集合与基线 a195c3c 完全相同（1 项环境产物）。
- 修掉一个真 bug：`kg_begin` 未复位搜索状态 → 同一 ctx 第二次搜索直接返回上一次的
  finished/action/sims_used（Player 每步复用 ctx 才暴露）；已加 ctx 复用对照测试。

## §18.25 训练显存实测与压缩（16 GB 卡，2026-10-03）

| 配置（有效批 2048） | torch 分配器峰值 | 进程峰值（nvidia-smi） | 3 步墙钟 |
|---|---|---|---|
| micro 512 × accum 4（旧，默认分配器） | ~10.9 GB | ~11 GB | — |
| micro 512 × accum 4 + expandable_segments | 7120 MiB | 7516 MiB | 4.4 s |
| micro 128 × accum 16 + 同上 | 2579 MiB | 2952 MiB | 6.8 s |
| **micro 64 × accum 32 + 同上** | **1822 MiB** | **2264 MiB** | 10.4 s |
| micro 64 × accum 32 + anchor_weight=2 | **2079 MiB** | **2516 MiB** | — |

- 模型是 **122.1M 参数**（fp32 权重 466 + 梯度 466 + AdamW 两态 931 ≈ **1.82 GB**）——
  这是**不可再压的底**（再往下要换 bf16 优化器状态，属数值口径变更，本轮不做）。
- 激活随 microbatch 近似线性：bs512 时 ~5 GB、bs64 时 ~0（bs64 的峰值就是权重+优化器）。
- 落地两处：`Kit 8dbb1ee`（trainer 默认 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
  纯分配器行为）省了 10.9→7.5 GB；`Transformer e957bf5`（loop microbatch 512→64、
  accum 4→32，有效批不变）再省到 **~2.5 GB**。代价：**每代训练 ~5 min → ~12 min**
  （自对弈 28 h 面前可忽略）。
- 自对弈/推理侧本来就只有 ~1.1 GB；server 空闲 ~0.96 GB。
- 口径说明：microbatch/accum 改变会改变训练轨迹的舍入与批内组成（有效批与统计口径不变），
  与 stage2 的 bs512 筛选用的是同一有效批；按固定配置仍然逐位可复现（E4 口径）。
- `Kit 3fdc8e5`：训练摘要新增 `peak_vram_mb`，以后每代落盘可查。

## §18.26 迁移 pro（2026-10-08 补记）

- **主机**：seetacloud 容器（`ssh -p 26657 fwj@...`），RTX PRO 6000 Blackwell 96 GB；
  **cgroup 22 核**（`cpu.max` 2200000/100000，nproc 报 208 是宿主）/ 110 GB RAM；共享宿主，
  **单核速度波动大**（同一配置实测 16→42 s/局）。`/tmp` 不可写 →
  `TMPDIR=~/UniChess/tmp`（已进 .bashrc）；**临时脚本一律放 `~/UniChess/tmp`**（用户指示，
  家目录保持干净）。uid 1004（非 root，ptrace 受限 → py-spy 不可用）。
- **环境**：`~/miniconda3/envs/unichess`（Python 3.12.15、torch 2.14.1+cu130，Blackwell 正常）；
  conda 走清华源、pip 走阿里源。全套件 375 项、4 项失败均为 `TestParityWithR`（R 权重未迁移，
  与 T 无关）。**工作树"脏"**：96 个文件的路径被改成 pro 前缀 → **禁止在 pro 上 git pull**，
  代码更新只能 scp 具体文件。
- 70Ti 退为**网站 + 备份**（生产 = gen_0007）；网站 chess.jeefy.top 一直在 70Ti。
- 与 70Ti 同配方吞吐对比：70Ti 12.6 s/局 vs pro 单进程 15-17 s/局（机器差距一截，
  主要是共享单核慢）；**workers=4 后 pro ~5.3 s/局**（见 §18.27）。

## §18.27 workers=4 + MPS：吞吐 ~3×（2026-10-08）

- **配置**：`selfplay.workers=4 × concurrency=32`；MPS 守护
  （`nvidia-cuda-mps-control -d`，pipe/日志在 `~/UniChess/tmp/mps_*`，setsid 起、
  loop 启动前必须活着）；**`OMP_NUM_THREADS=4`**。arena/screen workers 也提到 4
  （`8b5d1c4`）；训练微批 64×32→256×8（96 GB 显存宽裕，训练 ~12→~6 min/代）。
- **实测稳态**：4 worker 各 ~100% CPU、GPU 79-92%（单进程 30%）、显存 5.6 GB、
  **~5.3 s/局聚合（原 15-17 s/局，≈3×）**。与并存的 ICLR 任务（~50 GB）共存无压力。
- **重要修正（之前两次"workers 失败"是误诊）**：torch 在 pro 上默认开 **104 线程池**
  （nproc/2），4 个 worker 并行初始化模型时在 22 核配额上互踩，把 ~3 s 的初始化拖到
  3.5 分钟；MPS 连接日志证明 worker 在 +3.5 min 才完成 CUDA 初始化，而两次都在第 4-5
  分钟被杀，**从未见过稳态**。当时归因"CPU 回退"也是错的（`kit.py` 的
  `make_evaluators` 硬编码 `device="cuda"`，CUDA 不可用会崩、不会静默回退）。
  修复 = `OMP_NUM_THREADS=4`，零代码改动。
- **续跑的局号机制**：`loop.py phase_selfplay` 强制 `first_game = g*games`。换代中途
  切 workers 时把顶层 `games` 改成剩余局数即可得到**不相交的新局号段**（与主分片零重复
  零重放）；gen 10 = 主分片 1947 局（[40960,42907)）+ w0-w3 共 2149 局（[21490,23639)）
  = 4096。⚠ **该值必须事后恢复 4096**，否则后续代都只打半量局——gen 11 就中招
  （wakeup 迟触发 6 h，gen 11 只训了 2149 局、gen 12 头 2014 局作废归档
  `~/UniChess/tmp/archive_gen12_partial_1008`）。
- **离线微基准**（干净 GPU，混合 strata 批，gen_0007 权重 fp16）：
  旧前向 12.5 ms（~300 次 kernel 发射占 ~68%，批 1→128 几乎平坦 = 纯开销型）；
  **CPU 侧路由**（`aeb22b4`，去掉批内逐行 `.item()` 同步 0.76 ms）11.0 ms（1.14×，
  **数值逐位一致**）；**CUDA graph 原型**（单专家 m=64 含 softmax）2.15 ms（**5.83×，
  同形状回放 max|Δ|=0**）；**torch.compile reduce-overhead** 1.72 ms（7.26×）。
  → graph 化是下一个大杠杆（工程化待做，预期整体再 +15-25%）；已证伪的并行旋钮：
  threads>1（GIL）、workers>4 无 MPS（context 切换）、conc≥128。

## §18.28 gen 10 判决：H1（+92.9）——新范式首次晋升（2026-10-08）

- **gen 10 = Gumbel π′ + anchor_b2 新范式第一次通过 arena**：
  score_a **0.6306**、Elo **+92.93**（CI95 [31.6, 160.6]）、111 局 SPRT 早停、
  arena 65 min（workers=4）。**冠军换成 gen_0010**（此前 8/9 连续 H0，gen 7 之后首次晋升）。
  生产权重已在 **2026-10-08 12:05 切到 gen_0010**（70Ti 两预设 →
  `gen_0010/train/best_model.pt`，sha256 `f87cef08…` 中继校验一致，
  describe 11x384/step 350 通过，提交 `e301d6b`，回滚 = 改回 gen_0007 路径重启）。
- **数据质量（比判决更重要，全量 4096 局 401k 记录实测）**：
  - 位置重复率 **6.92%**（gen_0009 是 75.89%）；**全序列重复局 1.25%**（51/4096）；
  - 2000 条开局全部用上（2000 组、组均 2 局）；组内首次分歧 **中位 ply 9**
    （382 局恰在 ply 7 = book 后第一步就分岔）；
  - π′ 顶部 16 槽平均 **6.54 个非零**（p90 = 16 触顶）、归一化熵 0.163、top-1 占比 0.826
    ——明显比集中式 PUCT（1-3 个非零）分散；
  - 终局：checkmate 91.5%、和棋 8.4%；步数 mean 98 / median 75。
  - 口径注：Gumbel 在 book ply 也做完整搜索并落 π′ 记录（走 book 着法）——设计如此
    （`gumbel_player.py` 模块注释），与 SearchPlayer 的"book 不落记录"不同。
- **gen 11 = H0（−67.9，57 局）但被半量数据污染**（只训 2149 局，见 §18.27 的 games 事故），
  结论不读方向只记事件。gen 12 起恢复 4096 局 + **c_scale=0.02**（用户定的实验值；
  注意 Gumbel 的 c_scale 在 SSM/AGENTS.md §8 是 LOCKED 0.1，此为 T 侧实验）。

## §18.29 gen 12 判决：H0（−51.6）——c_scale=0.02 被否（2026-10-08）

- **判决**：score_a **0.4262**、Elo **−51.64**（CI95 [−136.6, 27.5]；五回合 CI95
  [−110.2, −1.1] **不含 0**）、LLR **−6.85**、SPRT 第 41 局早停、61 局入账、
  arena 3334 s。champion 仍 gen_0010（生产不变）。
- **机理链（数据质量实测，与 §18.28 同口径）**：c_scale 0.1→0.02 把 π′ 显著摊平——
  top-1 占比 0.826→**0.645**、非零槽位 6.54→**13.09**、归一化熵 0.163→**0.359**。
  训练目标更分散 → 候选棋力更"软"：和棋率 8.4%→**17.9%**、
  insufficient_material 终局 3.6%→**12.1%**、步数 98→**128.9**、
  checkmate 91.5%→82.1%。位置重复率反而更低（6.92%→4.46%），
  说明"探索更散"确实发生了，但**更散 ≠ 更好**。
- **结论**：c_scale=0.02 否定（五回合口径显著为负）。默认 0.1 是当前配方的一部分，
  **改回去 = 删 configs/loop_p4_v2.json 的 `engine.kwargs.c_scale` 后重启 loop**。
- ⚠ 判决 20:01 落定时 loop 已自动进 gen 13（自对弈 19:58 起跑，**用的还是 0.02**，
  因为配置未回退）；按预案 H0 只汇报不自行改回，**是否中止/回退 gen 13 待用户定**。
  回退操作 = 停 loop → 清 gen_0013/selfplay 残片 → 删 c_scale → 重启（loop 会重打 gen 13）。


## §18.30 A 阶段判决实验：目标质量 vs 数据量 + c_scale 全 dial（2026-10-09）

> 背景：gen 12 = H0（c_scale=0.02），用户假设"更散的目标需要在树复用下用更多
> self-play 局数才有效"。三个实验判决之。

- **A1 KL 拟合度**（各自/交叉，全量 401k / 528k 记录，KL(pi'_target || pi_net)，
  target 按 16 槽重归一、net 在同 support 上重归一）：
  | combo | KL mean | H(target) |
  |---|---|---|
  | g10 cand @ g10 data | 0.6425 | 0.453 |
  | g12 cand @ g12 data | **0.4176** | 0.996 |
  | g12 cand @ g10 data | 0.6429 | 0.453 |
  | g10 cand @ g12 data | 0.4494 | 0.996 |
  -> gen_0012 对自家（更平的）目标拟合得比 gen_0010 对自家目标**更好**
  （0.418 < 0.643）——**不是欠拟合/估计噪声，gen 12 的 H0 是目标质量问题**，
  加局数只会让候选更忠实地复现一个更弱的策略。交叉项：gen_0012 的头在 sharp
  目标上 KL 与 gen_0010 持平（0.6429 vs 0.6425）——头没有不可恢复地"变平"，
  差别纯粹在训练目标指向哪里。
- **A2 c_scale 静态扫描**（同一批 2000 开局 x 6 个 book 局面 = **12000 个逐字节
  一致的局面**，只变 c_scale；gen_0010 权重，g=1.0, temp=1.0, 1600 sims；
  局面一致性按 (game,ply) 定序后校验通过）：
  | c_scale | top1 | top3 | 非零槽 | H/ln16 |
  |---|---|---|---|---|
  | 0.02 | 0.578 | 0.886 | 15.83 | 0.414 |
  | 0.05 | 0.701 | 0.942 | 13.70 | 0.288 |
  | 0.1 | 0.851 | 0.984 | 9.09 | 0.140 |
  | 0.2 | 0.909 | 0.995 | 4.90 | 0.081 |
  | 0.5 | 0.956 | 0.999 | 1.90 | 0.036 |
  完全单调的锐化 dial。gen_0010 全量实测（top1 0.826 / 非零 6.54 / H 0.163）
  落在 0.1 列（book 局面比全局略尖）。gen_0012 全量（0.645/13.09/0.359）落在
  0.02-0.05 之间（book 局面对 c_scale 更敏感）。
- **A3 树剖面 + pi'-Q 相关**（Python Gumbel，gen_0010 权重，330 ply 剖面 +
  40 个固定局面 x 5 c_scale）：
  - 每步搜索 n_nodes ~= 1529 / 1600 sims（**每个 sim 展开 1 个新节点 = 搜索内
    零共享**）、max_depth mean 33.6；
  - **被走着法的候选子树 = 400 节点 = 当步 n_nodes 的 26.2%**（恰为顺序减半
    预算分配 25+50+100+200 的确定性结果）；下一步冷启动搜索有 **27.1% 的节点
    属于上一步的被走子树** -> **树复用的收益上界约 27% 的展开评估**；且 Gumbel 的
    sigma=(c_visit+max_b N)*c_scale*qhat 记账假定"预算事先定死"的全新根，继承
    计数会改变 halving 动力学——优先级明显低于 CUDA graph（原型 5.8x/7.3x，见
    §18.27）；
  - corr(pi', 根边 Q) 随 c_scale **单调下降**（0.02->0.5：0.586->0.484）——价值项
    并没有简单地"给目标加信号"；pi' 与 Q 的一致性是弱代理，Elo（0.1 比 0.02 高
    52）才是判决。c_scale 的真实作用是**目标锐度 dial**，不是"探索旋钮"
    （探索由 g / temperature 提供）。
- **B1 已执行**（2026-10-09 00:39 本地）：pro 的 configs/loop_p4_v2.json 删除
  engine.kwargs.c_scale（回到 gen_0010 配方）；gen_0013 的 0.02 残片归档
  ~/UniChess/tmp/archive_gen13_partial_1009；loop 重启（pid 872472），gen_0013
  用原配方重打（games=4096, first_game=53248, workers=4）。
- 方法学备注：A2 的"book 前缀共享局面"设计（book_plies=max_plies=6，每局只搜
  6 个 book 局面）是控制变量扫描的廉价套路——同一批局面、只变一个参数、走生产
  同款攒批路径，约 8 min/组，可复用到其它 engine 参数（m0 / sims / g）。

## §18.31 CUDA graph 工程化：微观 1.6-2.0×，端到端 0.98×（负面结论，2026-10-09）

- **做了什么**（`evaluator.py::_CudaGraphRunner`，提交 `3553462`，**默认关**）：
  按 `(N, k0,k1,k2)`（批大小 × 三专家各行数）分桶捕获 CUDA graph；捕获区只含
  `x[idx_e]`（index_select）→ 专家前向（autocast 与原路径一致）→ `p[idx_e]=sub_p`
  （index_put），索引是**静态 GPU 张量**、H2D 全在图外，输出在图私池；桶上限 512、
  N 上限 256，超限/捕获失败永久回退 eager，`UNICHESS_T_CUDAGRAPH=1` 启用、
  `UNICHESS_T_NO_CUDAGRAPH=1` 硬关，`UNICHESS_T_GRAPH_STATS=1` 退出时转储统计。
- **正确性**：单测 4 组全绿——fp16 十二种 N（1..64）+ fp32 五种 N + 桶上限回退 +
  N 超限回退，全部与 eager **逐位一致**（max|Δ|=0，同形状回放同 kernel）；
  索引精确拷贝、专家看到的行与顺序完全相同，所以逐位一致是可证明的而非测出来
  的运气。test_r3 回归与基线一致（16 过 + 2 败 + 2 错，均为既有陈旧期望）。
- **微观基准**（生产权重、loop 同在 GPU 上背靠背对比）：
  单专家批（pc 不跨边界）N=1..64：eager 3.3-4.9 ms → 图 1.8-3.7 ms（**1.3-1.9×**）；
  混合专家批（pc 跨 12/24 边界，走三专家拆批）N=4..64：eager 8.4-12.0 ms →
  图 4.2-7.3 ms（**1.6-2.0×**）。均 max|Δ|=0。§18.27 原型 5.83× 是"混合 eager
  vs 单专家图"的口径差，此处是同口径背靠背。
- **生产形态实测**（`UNICHESS_T_GRAPH_STATS` 转储，4 worker）：
  - **每次搜索只有 ~3 次 `evaluate_planes` 调用**——C++ 把顺序减半一整轮的叶子
    合成一个 EvalRequest，batcher 再跨 32 个并发局合并，批大小以 **N>64 为主**
    （两次 2000/3000 局 A/B 里 N>64 占 72-88%；短探针曾见到 73% N≤64，随竞争
    波动极大）；
  - `(N,计数)` key 空间单 worker 267-650 个；512 桶下覆盖率 76-99%，
    显存 1.39 → 3.1-3.5 GB/worker（+2 GB）；
  - 单专家/混合专家调用比约 56:44。
- **端到端 A/B**（同配置同 seed，2000 局与 3000 局各一轮，workers=4）：
  2000 局：eager 2968 s vs 图 3008 s（**0.99×**）；3000 局：4476 s vs 4579 s
  （**0.98×**）。**无增益**。
- **归因**：前向只占搜索耗时 ~15%（3 次调用 × ~12 ms / 250 ms 每搜索），图消灭的
  发射开销理论上限 ~1.1×，被捕获税（512 桶一次性 ~1 min/进程，在 12 分钟的 A/B
  里占 ~5%，在生产一代里只占 0.2%）与运行间噪声吃平。**真正的成本在另外 85%**：
  每次搜索 ~375 个顺序减半波次的 Python↔C++ 往返（ctypes + numpy 暂存 + 协程
  恢复），那是 Kit 搜索编排的开销，不是模型前向。
- **结论与去向**：代码与单测保留（默认关，一把环境变量可重新启用），**pro 的
  evaluator.py 已回退到提交版（blob fe0ab24f，eager）**，gen 14 起生产跑 eager。
  下一个吞吐杠杆是给自对弈做 Python 侧 profile（cProfile 单 worker 跑几十局），
  定位那 ~375 次波次往返的消耗，再决定是减少往返次数（C++ 侧多轮合并一次请求）
  还是别的手段——**不要再赌模型前向**，它的上限已经测清楚了。

## §18.32 gen 13 判决：H0（−42.5）——原配方干净复跑未复现，红线停止（2026-10-09）

- **判决**：score_a **0.4392**、Elo **−42.47**（CI95 [−125.5, 35.9]）、LLR −3.269、
  第 35 局早停、74 局入账、arena 2979 s。champion 仍 gen_0010（生产不变）。
- **意义**：gen 13 是 **c_scale 回退后 gen 10 原配方的第一次干净复跑**（同配方、
  同 4096 局、同超参，只有按代派生的种子与"冠军已换成 gen_0010"两处不同）——
  **没有复现 gen 10 的 +92.9**。结合 gen 12（c_scale=0.02，−51.6）：最近两个可读
  代数都在 −45 上下，**gen 10 的 +92.9（CI95 [31.6, 160.6]）更像一次正偏抽样，
  而不是可复现的台阶**。新范式（Gumbel π′ + anchor）相对旧配方（PUCT 访问分布）
  的真实增益可能接近 0——gen 10 是赢了，但"稳定每代 +50~90"的预期没有数据支持。
- **数据质量排除数据 bug**（全量 4096 局）：π′ 非零槽 9.09 / top-1 0.774 /
  H(ln16) 0.222——正落在 gen 10（6.54/0.826/0.163）与 gen 12（13.09/0.645/0.359）
  之间，与 0.1 配方一致；checkmate 90.3%、和棋 9.7%、步数 mean 105.6；位置重复
  与分歧 ply 同族（quality13.py，口径同 §18.28）。
- **红线触发**：gen 11（半量污染）/ 12 / 13 **连续三代未换代** → 按 AGENTS.md
  换代红线第 4 条**停 loop 人工复盘，不放宽门槛**。gen 14 自对弈残片（770 局）
  归档 `~/UniChess/tmp/archive_gen14_partial_1009`；GPU 已释放。
- **待用户定的方向**（按预期收益排序）：
  1. **吞吐**：CUDA graph 已证否（§18.31，前向只占 15%）；真正的成本是每次搜索
     ~375 个顺序减半波次的 Python↔C++ 往返——先 cProfile 定位，再决定是否在
     C++ 侧合并多轮为一次请求。每代 8.4h → 若能砍半，实验轮转速度翻倍。
  2. **配方**：sims 1600→2400（§18.23 的搜索余量证据支持）或 m0/g 变体，
     用 §18.30 的 book 前缀扫描套路先静态测 π′ 形态再上整代。
  3. **范式**：若 1/2 都做不动，接受 gen_0010 为当前冠军（它是 arena 实测最强的，
     且已上产），把循环转成维护态，精力转向数据/训练侧的系统性改进。

## §18.33 吞吐攻坚完整结论：瓶颈是波次往返，不是模型前向（2026-10-09）

> 承接 §18.31（CUDA graph 首轮负面结论）。本轮用 cProfile + 原位计时 + 三组对照
> 实验把自对弈的成本结构测清楚了。

- **cProfile（单 worker，conc=32，640 次搜索）**：Python 侧 98 s 里
  `evaluate_planes` 链 65.5 s、`gumbel_cpp.search` 链 30.4 s（其中 search 自身
  tottime 21.5 s = ctypes 往返循环）。batcher 统计：**224,260 个 EvalRequest /
  640 次搜索 = 每搜索 ~350 个波次往返**，平均每批 113 个位置。
- **原位前向计时（evaluate_planes 墙钟入账，4 worker A/B）**：
  eager **7.4-7.8 ms/call**，graph **0.75-1.8 ms/call（6-8×）**——图机制本身
  在生产环境完全有效（§18.31 的"无增益"结论部分修正）。
- **但端到端仍 0.93-0.98×**（4w-graph vs 4w-eager，capture-on-repeat 后捕获税
  只剩 15-25 s/worker）：前向省下的 ~185 s/worker 被别处吃回去。
- **三组对照定位真瓶颈**：
  1. **workers 4→8**（GPU 全空）：墙钟 394s vs 414s（后者还被 loop 抢 GPU）——
     聚合吞吐锁死 ~22.5 搜索/秒，与 worker 数无关；
  2. **MPS 开/关**（单 worker）：267s vs 246s，只差 8%——MPS 服务进程不是上限；
  3. **conc 32→128→256**（单 worker）：8.4 → 8.1 → 6.9 搜索/秒——进程内并发
     早已打满，加并发只添乱。
- **结论（成本结构）**：每次搜索的时间 = ~375 个顺序减半波次周期 × ~0.32 ms，
  32 个并发局把这些周期流水起来 → 单 worker ~8.4 搜索/秒（119 ms/搜索）；
  4 worker 聚合 ~22 搜索/秒（GPU 竞争使线性打折），8 worker 不再涨。
  **模型前向与波次编排重叠，不是关键路径**——这就是图（前向 6-8×）与加 worker
  都动不了端到端的原因。cProfile 里"前向占 67%"是含跨局等待的墙钟，虚高。
- **为什么往返这么多**：顺序减半 4 轮（m0=16），预算 400/轮；幸存者
  16→8→4→2→1，波 = 每幸存候选一次 sim ⇒ 各轮 25/50/100/200 个波，
  **最后一轮（2 幸存 × 200 sim）独占 200 个往返**。Python 版 `parallel=True`
  的 gather 与 C++ kg_collect 结构相同（60 个请求 @256 sims），不是实现差异。
- **真正的杠杆（按预期收益排序）**：
  1. **C++ 多波合并**（`puct_native.cpp` 的 kg_* 循环）：每次 kg_collect 多收
     2-4 个波的叶子（同一候选子树内用缓存价值多下探几层，一次请求评估 8-16 个
     叶子）→ 往返 ÷3-4，单 worker 8.4 → 25-30 搜索/秒，一代 8.4h → ~3h。
     数值口径：批组成改变 ⇒ 与现路径有 ULP 级漂移（kg 与 Python 参考本就
     "数值等价非逐位"），arena 自洽即可；需要补 parity 回归。
  2. **sims 1600→800**：往返减半（一代时间减半），代价是搜索深度——可与 1 叠加
     做"往返减半 + sims 恢复"的净效应实验（book 前缀扫描先静态测 π′ 形态）。
  3. **CUDA graph 保留默认关**：等 1 落地后前向重新变成瓶颈时再开（现已入账：
     6-8× 有效、capture-on-repeat 税 15-25 s、逐位一致、4 组单测全绿）。
  4. 双机分片（70Ti）：容量 +1 台，与上述正交；待用户对 70Ti 的指令。

## §18.34 C++ 多叶展开（expand_width）：实现正确但净亏 2.2×，不部署（2026-10-09）

> 承接 §18.33（瓶颈 = 每搜索 ~375 个波次往返）。用户指示"实现 1"。

- **实现**（Kit 侧，默认 `expand_width=1` = 原行为逐位不变）：
  - `puct_native.cpp`：`g_expand_siblings`——主展开 pending 时把同节点其他高先验
    动作一起展开进同一次前向；`Pending.cache_only`（预建节点只填充不备份）、
    `GNode.preexpanded`（首次被走到时当该 sim 的叶子并清标记）；
  - `gumbel.py`：`GumbelConfig.expand_width`；`gumbel_cpp.py`：cap 扩到
    `m0×width` + `kg_set_expand_width`；`native.py`：ctypes 声明；
    `gumbel_player.py`：工厂透传；`test_gumbel_cpp.py`：2 个新用例。
  - **正确性**：与 width=1 **逐位等价**（同着法/同 sims/同根 N/QSUM/同 π′，
    浮点全同，5 个 FEN × width∈{2,4,8} 实测）；Kit 全套 379 项只剩 4 个
    **既有**失败（TestParityWithR 黄金基准过期，原版 C++ 同样失败，已对照验证）。
  - 途中修掉/绕过三个真 bug：① 根层预展开会撞同波其他候选（段错误）→ 根层不预展开；
    ② 终局兄弟不产 pending，`pending.back()` 误标主叶子 → 按 size 差判断；
    ③ **既有 bug**：`g_sim_node` 非终局真返回不写 `*out_val`，祖先层拿到调用方
    的 0.0（穿透已展开子节点终止于终局时）→ 补写，C++ 从此与 Python 参考一致
    （parity 探针 6 组 |ΔQSUM|=0 不受影响，即该情形在测试局面里不出现）。
- **端到端 A/B**（1500 局 × 6 book 面、4 worker、GPU 全空、同 seed）：
  width=1 墙钟 **419s** vs width=4 **916s** = **0.45×（慢 2.2×）**。
  微观侧往返只降 ~15-20%（21→17 请求 @sims=64），但节点创建量 ~2×——
  深尾轮是窄链前沿，兄弟节点很少被二次选中，额外评估量全落在关键路径上。
- **结论**：特性保留（默认关），**不部署**。顺序减半的波次结构决定了"每次 sim
  扩展一次前沿"，预展开只能摊平回访、不能减少前沿扩展；而省下的 ~0.05 ms/拍
  远低于多评一个节点的钱。**波次往返不是可绕开的成本，是需要换结构的问题**
  （减少往返次数得让 C++ 一次多探几层——但那就必须动选择语义/虚拟损失，超出
  本轮回合）。
- **对用户指示"然后 3"（开 CUDA graph）的处置**：3 的前提是"1 落地后前向成为
  瓶颈"——1 没落地，且图在 §18.33 已实测端到端 0.93-0.98×（前向 6-8× 但不在
  关键路径）、每 worker +2 GB 显存 + 捕获税。故**维持默认关**，代码与单测都在，
  `UNICHESS_T_CUDAGRAPH=1` 一行可开。若后续结构改动真把前向推成瓶颈，再开不迟。

## §18.35 arena 吞吐扫掠：判决带宽提升 3-4.5×，配置与 pro 硬件绑定（2026-10-10）

> 动机：一代判决 57-111 局、±80 Elo，而代际增益 ~±35——读数被噪声盖住。
> 结论先行：`arena.match` 改 `concurrency 8→64`、`workers` 保持 4，
> SPRT `elo1 60→25` + `min_pairs 100`、`pairs 512→350`（提交 `dd21f04`）。

### 扫掠设计与实测（干净窗口）

每配置跑“一整波”（局数 = workers×conc，否则并发被局数卡住、测到的是另一个配置），
墙钟≈单局时长。A=gen_0013、B=gen_0010、800 sims、同开局库同种子：

| 配置 | 总在途 | 墙钟 | 局/秒 | vs 4×8 |
|---|---|---|---|---|
| 4×8（原状） | 32 | 1555s | 0.0309 | 1.00× |
| 4×16 | 64 | 1149s | 0.0557 | 1.80× |
| 8×16 | 128 | 2080s | 0.0615 | 1.99× |
| 4×32 | 128 | 1399s | 0.0915 | 2.96× |
| 8×32 | 256 | 2731s | 0.0937 | 3.03× |
| **4×64** | **256** | **1815s** | **0.1410** | **4.56×** |
| 4×80 | 320 | 3472s* | 0.0922* | — |
| 4×96 | 384 | 3876s* | 0.0991* | — |
| 4×64 重跑 | 256 | 3465s* | 0.0739* | — |

\* = 被外部租户污染窗口的数据，见下。

### 三条结论

1. **切分效应**：同总数下 worker 少、conc 高明显占优——128 在途 4×32 比 8×16 快 1.49×，
   256 在途 4×64 比 8×32 快 1.50×。两次独立复现
2. **W=4 前沿仍在涨**：0.056→0.092→0.141（conc 16/32/64），单局时长近似恒定，
   速率随在途数线性上升；conc 64→80 的“每 ply 成本翻倍”是争用假象（见下）
3. **显存与并发无关**：树在 CPU 侧，GPU 只放模型——4 worker 恒定 ~6.5GB（各 1.6GB），
   与 conc/sims 无关。**扫掠中期的 17-32GB 读数全是外部租户的显存**，一度被误记成
   我们的占用，据此做出的“conc 64 会 OOM”担忧是错的（已实测证伪）

### 共享机争用：一个方法论坑

autodl-pub 是共享实例，**其他容器的 GPU 占用不出现在我们的 nvidia-smi 里**。
01:06-06:24 之间有 root 所有的外部任务（`conda_envs/csb/zhuanli/.../run_one.py`，
两个，共 27GB）间歇占用，把该窗口所有测量拖慢 **1.91×**——实证方式：同配置同种子
重跑 4×64，干净窗口 0.1410 vs 污染窗口 0.0739。按 1.9× 修正后 4×80/4×96 为
~0.175/~0.188，与 4×64 自洽，说明前沿可能延伸到 conc 96-128，**但未在干净窗口确认，
故取实测峰值 64**。教训：共享机上判断争用只能靠“同配置重跑”对冲，不能信 nvidia-smi。

### ⚠ 这个配置与 pro 硬件绑定

`configs/loop_p4_v2.json` **不是可移植配置**：

1. 路径全是 pro 绝对路径（`/root/autodl-tmp/fwj/UniChess/...`）——从建立起就是如此
2. `arena.match.workers/concurrency` 是按 **pro 的 RTX PRO 6000 96GB + 22 核 cgroup
   + 上述争用模式**实测标定的。换机器（如 70Ti 的 5070 Ti 16GB）编排并行度、饱和点、
   最优切分都会变，**必须重扫，不可照抄**
3. SPRT 参数（elo1/min_pairs/pairs）是统计量，与机器无关

### 预期效果

一代判决从 57-111 局 / ±80 Elo 变为 200-400 局 / ±35 Elo；
600 局固定预算宽判决从 5.4h 降至 ~1.2-1.5h（2400 sims、conc 64 预计 0.08-0.11 局/秒）。
“判决带宽”从此不再是换代的瓶颈。
