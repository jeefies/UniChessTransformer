# UniChessTransformer Architecture Specification

## 1. Overview

UniChessTransformer (Model T) is a neural chess engine built on Transformer backbones with 2D
spatial geometric priors. It combines a bilinear square-to-square policy head, a Win-Draw-Loss
(WDL) value head, an optional Moves-Left (MLH) head, phase-stratified expert routing, and a
C++-accelerated batched Monte Carlo Tree Search (MCTS) with native Syzygy tablebase probing.

This document is the single source of truth for architecture. For match results and training
history see [`experiments.md`](experiments.md); for build/run commands see the root
[`README.md`](../README.md) and [`AGENTS.md`](../AGENTS.md).

---

## 2. Model Hierarchy & Tiers

Parameter counts below are measured, not estimated (verified against the current
`model/transformer.py`).

| Preset | Layers | $d_{\text{model}}$ | Heads | $d_{\text{ff}}$ | Params (measured) | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `transformer_tiny` | 6 | 192 | 6 | 768 | 3,817,288 | Ultra-fast CPU tests & rapid ablation |
| `transformer_small` | 8 | 256 | 8 | 682 | 6,741,000 | Fast prototyping |
| `transformer_medium` | 10 | 384 | 12 | 1024 | 18,523,528 | Mid-range balanced baseline |
| `transformer_large` | 12 | 512 | 16 | 1152 | 35,120,904 | Large single-backbone tier |
| **`transformer_20m`** | **11** | **384** | **12** | **1024** | **20,343,688** | **SOTA compact competitive model** |
| **`transformer_50m`** | **17** | **512** | **16** | **1160** | **49,757,960** | **Flagship deep transformer** |
| **`stratified_20m`** | **3 x 11** | **384** | **12** | **1024** | **61,031,064** | **Phase-stratified ensemble (current best)** |

`stratified_20m` = 3 x `transformer_20m` experts plus shared routing. Its parameter count is
approximately 3 x 20.3M, confirming that the three experts hold the bulk of the weights.

> **Checkpoint note:** a saved `stratified_20m` state dict contains *both* `opening.*` /
> `middlegame.*` / `endgame.*` keys **and** `experts.0/1/2.*` keys. These are aliases of the
> same modules (`self.experts = nn.ModuleList([self.opening, self.middlegame, self.endgame])`,
> `model/transformer.py:316`), so counting raw state-dict entries yields ~122M — roughly double
> the real 61.0M parameter count. The loader (`_load_state_dict`) tolerates either naming.

---

## 3. Structural Components

### 3.1 ConvStem & Input Representation

Input boards are canonical 19-plane float32 tensors of shape `(19, 8, 8)`, oriented to the side
to move (mirrored when Black is to move):

| Planes | Content |
| :--- | :--- |
| 0–5 | Friendly pieces (P, N, B, R, Q, K) |
| 6–11 | Enemy pieces (P, N, B, R, Q, K) |
| 12–15 | Kingside / queenside castling rights (friendly / enemy) |
| 16 | En passant target square |
| 17 | Halfmove clock, normalized by 100 |
| 18 | Repetition count, normalized by 2 |

The **ConvStem** is a $3\times3$ convolution with stride 1 and padding 1:

$$\mathbf{X}_0 = \text{Conv2D}_{19 \to d_{\text{model}}}(\mathbf{X})$$

Flattening the spatial dimensions yields 64 square tokens in $\mathbb{R}^{64 \times d_{\text{model}}}$.

### 3.2 Tokens & 2D Geometric Attention

1. **`[CLS]` token**: one learned token is prepended to the 64 square tokens (65 total). It
   carries the global board summary consumed by the value heads.
2. **Rank & file embeddings**: decoupled learned embeddings are added per square:
   $$\mathbf{E}_{r,c} = \mathbf{e}_{\text{rank}}(r) + \mathbf{e}_{\text{file}}(c)$$
3. **Pairwise relative position bias**: a learned $\mathbf{B}_{\text{rel}} \in \mathbb{R}^{H \times 64 \times 64}$
   square-to-square bias encodes board topology (file, rank, diagonal and knight relations)
   directly into attention:
   $$\text{Attn}(Q,K,V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + \mathbf{B}_{\text{rel}}\right)V$$

`B_rel` is zero-padded for `[CLS]` to shape $(1, H, 65, 65)$ and passed straight into
`F.scaled_dot_product_attention(..., attn_mask=attn_bias)`, so the fused
FlashAttention/SDPA kernels are used. **Never mutate attention matrices in place with slicing.**

### 3.3 Bilinear Square-to-Square Policy Head

Rather than a flat projection to 4096 move classes, the policy head projects the 64 square
representations into origin queries $Q \in \mathbb{R}^{B \times 64 \times D_p}$ and destination
keys $K \in \mathbb{R}^{B \times 64 \times D_p}$ (with $D_p = 64$):

$$\mathbf{M}_{\text{from},\text{to}} = \frac{QK^T}{\sqrt{D_p}} + \mathbf{B}_{\text{move}}$$

This keeps the parameter count low, bakes board geometry into the move factorization, and
flattens to the 4096-move policy space. A decoupled promotion head maps rank-8 representations
to promotion piece types (Q, R, B, N).

### 3.4 WDL Value Head

The `[CLS]` token is passed through an MLP:

$$\mathbf{v}_{\text{wdl}} = \text{Linear}_{d \to 3}\big(\text{GELU}(\text{Linear}_{d \to d}(\text{LayerNorm}(\mathbf{h}_{\text{CLS}})))\big)$$

Output logits correspond to Win, Draw and Loss probabilities with $\sum P(\text{WDL}) = 1$.
The scalar evaluation used by search is $Q = P(\text{Win}) - P(\text{Loss}) \in [-1, 1]$.

### 3.5 Moves-Left Head (MLH, added in Stage P3)

An optional auxiliary head predicts the number of moves remaining until game end, from a
two-layer MLP on the `[CLS]` token (`model/transformer.py:220`):

$$\hat{m} = \text{Linear}_{64 \to 1}\big(\text{SiLU}(\text{Linear}_{d \to 64}(\mathbf{h}_{\text{CLS}}))\big)$$

- Activated via `return_mlh=True` in the forward pass.
- Trained with Smooth-L1 loss against pseudo moves-left targets derived from the evaluation
  shards (`model/loss.py`, `mlh_weight = 0.05`).
- Present in every expert since Stage P3. **Checkpoints predating P3 lack `mlh_head` keys and
  cannot be loaded by the current architecture** — see the compatibility note in
  [`experiments.md`](experiments.md) §1.

---

## 4. Stratified Chess Transformer & Phase Routing

`StratifiedChessTransformer` routes each position to one specialized expert based on piece count
and ply:

| Phase | Expert | Condition |
| :--- | :--- | :--- |
| 0 | **Opening** | $\text{piece\_count} \ge 24$ **or** $\text{ply} \le 20$ |
| 1 | **Middlegame** | $12 < \text{piece\_count} < 24$ |
| 2 | **Endgame** | $\text{piece\_count} \le 12$ |

Each expert is a full `transformer_20m` network. During batched inference, inputs are grouped by
routed phase so each expert sees a contiguous batch, keeping GPU utilization high.

Since Stage P3 each expert also carries its own `mlh_head`. When loading a checkpoint whose
experts were trained separately, `_load_state_dict` broadcasts `experts[0].mlh_head` to the
other experts if MLH weights are absent, so mixed-age checkpoints stay loadable.

---

## 5. Search Engines

### 5.1 C++ MCTS (`search/cpp/`) — production path

A PyBind11 extension (`mcts_pybind.cpp` binding `mcts.hpp` + `chess_board.hpp`) providing
batched PUCT tree search at **6,155+ sims/sec** on the reference GPU.

- **Native bitboard move generator** with perft-verified legal move parity.
- **19-plane feature encoding in C++**, so leaves are encoded without a Python round-trip.
- **Batched GPU evaluation bridge**: leaf requests are queued and dispatched to the Python
  network in saturated batches (default 64), with lock-free tree expansion.
- **Leaf-level Syzygy 3-4-5 probing**: when an unexpanded node has $\le 5$ pieces, the traversal
  probes the tablebase directly, returns the exact game-theoretic WDL, marks the node terminal,
  and backs the value up the search path — no neural evaluation involved.

#### Stage P2 search features

| Feature | Parameter | Behaviour |
| :--- | :--- | :--- |
| **Tree reuse** | `reuse` (bool) | After the opponent replies, the search descends the retained subtree via `reuse_root(move_uci)` instead of rebuilding from scratch, reusing all prior visit counts. |
| **Dynamic FPU** | `c_fpu` (default 0.5) | First-play urgency for unvisited children is computed as `parent_q - c_fpu * sqrt(1 / (1 + parent_visits))` (`compute_fpu_q`, `mcts.hpp:114`), replacing the old fixed `fpu_reduction = 0.2`. |
| **Contempt** | `contempt` (float) | Draws seen from the root are biased by `contempt / (1 + root_N)` during backup, discouraging premature draw acceptance. Zero disables it. |

### 5.2 Python MCTS (`search/mcts.py`) — fallback

Batched PUCT search with virtual loss and Dirichlet noise. Used automatically when the C++
extension is not compiled or fails to load, and by the Python-side API. Also exposes
`MCTS.advance_root(root, move)` for tree reuse on the Python path.

### 5.3 Multi-Process Parallel MCTS (`search/parallel_mcts.py`)

Lock-free CPU worker processes feeding a central GPU batched evaluator over
`multiprocessing.Queue` / `Pipe` channels. Peaks at **8,820 sims/sec** with 16 workers and batch
size 64 (31.1x over a single worker). See [`experiments.md`](experiments.md) §2 for the full
scaling sweep.

```
                    +-----------------------------+
                    |      Central GPU Evaluator   |
                    |  (RTX 5070 Ti, batches 64-512)|
                    +--------------+--------------+
                                   ^
                       Requests    |    Results
                       (Queue)     |    (Pipes)
                                   v
     +-----------------------------+-----------------------------+
     |                             |                             |
+----v-----+                 +----v-----+                 +----v-----+
| Worker 1 |       ...       | Worker N |       ...       | Worker 16|
| (sims)   |                 | (sims)   |                 | (sims)   |
+----------+                 +----------+                 +----------+
```

---

## 6. Training Heads Summary

| Head | Output | Loss | Introduced |
| :--- | :--- | :--- | :--- |
| Policy (bilinear) | $(B, 4096)$ + promotion logits | Softmax cross-entropy vs Stockfish policy | Initial |
| Promotion | $(B, 4)$ | Cross-entropy on promotion moves | Initial |
| WDL value | $(B, 3)$ | Cross-entropy over Win/Draw/Loss | Initial |
| MLH (moves-left) | $(B,)$ | Smooth-L1, weight 0.05 | Stage P3 |

---

## 7. Engine & Server Contract

`engine/engine.py` (`TransformerEngine`) wraps model + search + tablebase + opening book and is
the shared inference entry point. The root `engine.py` adapts it to the UniChess Server
`GameEngine` contract, caching one engine per
`(ckpt, device, precision, syzygy_path, use_cpp_mcts, book_path)` so concurrent sessions share
weights.

Move selection priority in `engine.py:engine_move()`:

1. Syzygy tablebase (exact)
2. Polyglot opening book (`data/opening_book.bin`)
3. MCTS search (C++ with Syzygy, else Python fallback)
4. Raw network policy

### 7.1 Temperature

`TransformerEngine` carries a `temperature` (default `0.0`) that selects how the root move is
drawn from the finished search:

$$p_i = \frac{N_i^{(1/t)}}{\sum_j N_j^{(1/t)}}$$

over root **visit counts** (not the raw policy). `t <= 0` (C++: `t <= 0.01`) means greedy argmax
over `N`. Sampling is seeded from OS entropy on the Server path, so `t > 0` genuinely varies.

The Server-facing `GameEngine` exposes the same knob as a named constructor argument and
threads it into all three search call sites (C++ `search`, `MCTSConfig`, `best_move`).
Presets that omit the key inherit the `0.0` code default and therefore remain fully
deterministic.

### 7.2 Presets

The Server discovers this repository through the symlink
`~/UniChess/Server/models/T -> ~/UniChess/Transformer` and reads presets from
`config.json`. Because the T engine resolves paths directly (no repo-root rebasing), **all paths
in `config.json` must be absolute**.

| Preset | Sims | Batch | Temp | Role |
| :--- | :--- | :--- | :--- | :--- |
| `max_mcts` | 2400 | 64 | (0.0 default) | Production route — fully deterministic |
| `max_t` | 2400 | 64 | 0.0 (exposed) | Same model/search; knob surfaced for tuning |

The UI sets no explicit preset default and selects the first option, which is the first of
`list_presets()` (sorted). Because `"max_mcts" < "max_t"`, `max_mcts` stays the default — **any
new preset sorting before `max_mcts` would silently become the production route.**
