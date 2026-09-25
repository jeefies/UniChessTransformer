# UniChessTransformer Architecture Specification

## 1. Overview

UniChessTransformer (Model T) is a neural chess engine built on Transformer backbones with 2D
spatial geometric priors. It combines a bilinear square-to-square policy head, a Win-Draw-Loss
(WDL) value head, an optional Moves-Left (MLH) head, phase-stratified expert routing, and a
batched PUCT tree search with Syzygy tablebase probing, provided by UniChessKit.

This document is the single source of truth for architecture. For match results and training
history see [`experiments.md`](experiments.md); for build/run commands see the root
[`README.md`](../README.md) and [`AGENTS.md`](../AGENTS.md).

---

## 2. Model Hierarchy & Tiers

Parameter counts below are measured, not estimated (verified against the current
`model.py`; `python -m unittest Transformer.tests.test_r3` re-checks them).

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
> `model.py`), so counting raw state-dict entries yields ~122M — roughly double the real
> 61.0M parameter count. The loader (`Transformer.model.load_model`) tolerates either naming,
> plus the `_orig_mod.` prefix that `torch.compile` checkpoints carry.

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
two-layer MLP on the `[CLS]` token (`model.py`):

$$\hat{m} = \text{Linear}_{64 \to 1}\big(\text{SiLU}(\text{Linear}_{d \to 64}(\mathbf{h}_{\text{CLS}}))\big)$$

- Activated via `return_mlh=True` in the forward pass.
- Trained with Smooth-L1 loss against pseudo moves-left targets derived from the evaluation
  shards (`Kit/planes19/losses.py`, `mlh_weight = 0.05`).
- Present in every expert since Stage P3. **Checkpoints predating P3 lack `mlh_head` keys**; the
  loader tolerates exactly that one missing group and rejects anything else (a silently random
  head is worse than a loud failure) — see the compatibility note in
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

Since Stage P3 each expert also carries its own `mlh_head`. Batched input is grouped by routed
phase so each expert sees a contiguous batch. When no board is supplied (kit's C++ PUCT hands over
pre-encoded planes only) the expert is picked from the piece count in planes 0-11, which is what
`StratifiedChessTransformer.forward` does when `boards` / `route_indices` are absent.

The three curriculum recipes train **one expert at a time**: `Transformer.kit.make_task` freezes
the other two experts' parameters (`Planes19Task.param_groups`) and still exports the full
three-expert state dict, so the Server keeps loading the result the same way.

---

## 5. Search (UniChessKit)

T no longer ships a search implementation. Every path — batch arenas, Server play, self-play —
goes through UniChessKit's PUCT:

| Implementation | Where | Notes |
| :--- | :--- | :--- |
| `PUCTCpp` | `Kit/search/puct_cpp.py` + `Kit/search/_native/puct_native.cpp` | production; writes leaf encodings in C++, forwards planes to `Transformer.evaluator.TransformerEngine.evaluate_planes` |
| `PUCT` | `Kit/search/puct.py` | reference; `PUCTCpp` is bitwise identical to it |

Both batch leaves **across games**, which is why the in-repo C++ MCTS (which owned its own search
loop) was removed on the rebuild: it could not share a batch with another game, so the Server and
the arena ended up with a second, slower search. Syzygy 3-4-5 probing is kit's `rules/tablebase.py`;
openings are `Kit/rules/openings.py`.

`search_impl` selects the implementation (`"auto"` = C++ when it compiled, `"python"` forces the
reference one) and is a *runtime* option, not part of the config hash. The C++ kernel is compiled
on first use and cached under `~/.cache/unichess_kit/native/`; a failed compile raises instead of
falling back, so a missing compiler can never silently halve throughput.

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

`Transformer.evaluator.TransformerEngine` is the **batch forward only**: load a checkpoint, encode
planes, return softmaxed `(policy, promo, wdl)`. The root `engine.py` exposes it to the UniChess
Server through `Kit.serving.make_game_engine`, which implements the six-method `GameEngine`
contract (termination, white-perspective eval, undo replay) for every engine that can hand kit a
Player. Batch arenas and spectating use `KIT_FACTORY = "Transformer.kit:make_player_factory"`
directly, so they run the native kit Player with cross-game batching.

The loader in `Transformer.model.load_model` picks the architecture from the checkpoint itself
(`preset == "stratified_20m"` or expert-prefixed keys → `StratifiedChessTransformer`, otherwise
`ChessTransformer` with the saved `cfg`) and is strict except for `mlh_head.*`.

Move order is decided by kit's search; `config.json` presets map onto its parameters
(`mcts_sims` → `simulations`, `mcts_batch` → `batch_size`).

### 7.1 Temperature and root narrowing

`TransformerEngine` carries a `temperature` (default `0.0`) that selects how the root move is
drawn from the finished search:

$$p_i = \frac{N_i^{(1/t)}}{\sum_j N_j^{(1/t)}}$$

over root **visit counts** (not the raw policy). `t <= 0` means greedy argmax over `N`; `t == 1`
is proportional to visits; `t < 1` sharpens; `t > 1` flattens *past* the visit distribution and
is **clamped to 1.0** by kit's `PUCTConfig` with a one-time warning. Sampling on the Server path
is seeded from OS entropy, so `t > 0` genuinely varies between games.

| Setting | Mean SF rank | Median | Outside SF top-5 |
| :--- | :--- | :--- | :--- |
| R (800 sims, greedy) | 1.60 | 1.0 | 0 % |
| T (2400 sims), old `t=1.5`, no narrowing | 5.60 | 2.0 | 30 % |
| T (2400 sims), **`t=1.0` + `root_top_k=3`** | **2.95** | 1.0 | 20 % |

*Measured on 40 moves/config against Stockfish 19 with `Threads: 1` at 30k nodes — `Threads > 1`
is **not** reproducible at a fixed node budget.* The old path was strictly harmful
over-flattening, not a capability deficit: greedy T is at parity with R. `root_top_k` narrows
sampling to the K most visited root moves, so variety can never promote a badly ranked move.

Presets that omit `temperature` inherit the `0.0` default and therefore remain fully
deterministic. `root_top_k` defaults to `0` (no cap); the shipped `max_t` pair gives 5/5 distinct
games vs M6 over 24 plies, diverging from ply 1.

### 7.2 Presets

The Server discovers this repository through the symlink
`~/UniChess/Server/models/T -> ~/UniChess/Transformer` and reads presets from `config.json`.
Relative paths in the presets are resolved against this repository (`Transformer.kit._resolve`),
unlike the old T plugin which resolved against the Server's working directory.

| Preset | Sims | Batch | Temp | Top-K | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `max_mcts` | 2400 | 64 | (0.0 default) | (0 default) | Production route — fully deterministic |
| `max_t` | 2400 | 64 | 1.0 | 3 | Same model/search; sampled variety with a quality guard |

The UI sets no explicit preset default and selects the first option, which is the first of
`list_presets()` (sorted). Because `"max_mcts" < "max_t"`, `max_mcts` stays the default — **any
new preset sorting before `max_mcts` would silently become the production route.**
