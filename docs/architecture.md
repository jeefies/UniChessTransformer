# UniChessTransformer Architecture Specification

## 1. Overview
UniChessTransformer is a state-of-the-art neural chess engine architecture leveraging Transformer backbones with spatial 2D geometric priors, bilinear policy factorization, Win-Draw-Loss (WDL) value heads, phase-stratified routing, and high-throughput multi-process parallel MCTS.

---

## 2. Model Hierarchy & Tiers

| Preset | Layers | $d_{\text{model}}$ | Heads | MLP $d_{\text{ff}}$ / Hidden | Params | Primary Role / Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `transformer_tiny` | 6 | 192 | 6 | 768 | ~3.8M | Ultra-fast CPU testing & rapid ablation |
| `transformer_small` | 8 | 256 | 8 | 682 | ~6.7M | Fast prototyping |
| `transformer_medium` | 10 | 384 | 12 | 1024 | ~18.5M | Mid-range balanced baseline |
| **`transformer_20m`** | **11** | **384** | **12** | **1024 / 1536** | **~20.3M** | **SOTA compact competitive model** |
| **`transformer_50m`** | **17** | **512** | **16** | **1160 / 2048** | **~49.7M** | **Flagship deep transformer for maximum tactical depth** |
| **`stratified_20m`** | **3 x 11** | **384** | **12** | **1024 / 1536** | **~60.8M** | **Dynamic phase-stratified ensemble (Opening / Mid / End)** |

---

## 3. Structural Components

### 3.1 ConvStem & Input Representation
Input boards are represented in canonical 19-plane float32 tensors of shape `(19, 8, 8)` oriented to the side to move:
- Planes 0–5: Friendly pieces (P, N, B, R, Q, K)
- Planes 6–11: Enemy pieces (P, N, B, R, Q, K)
- Planes 12–15: Kingside and queenside castling rights (Friendly / Enemy)
- Plane 16: En passant target square
- Plane 17: Halfmove clock normalized by $100$
- Plane 18: Repetition count normalized by $2$

The **ConvStem** performs an initial $3\times 3$ convolution with stride 1 and padding 1:
$$\mathbf{X}_0 = \text{Conv2D}_{19 \to d_{\text{model}}}(\mathbf{X})$$
Flattening spatial dimensions yields 64 square tokens in $\mathbb{R}^{64 \times d_{\text{model}}}$.

### 3.2 2D Geometric Attention & Positional Embeddings
1. **Rank & File Embeddings**: Square tokens are augmented with decoupled learned rank ($\mathbb{R}^{8 \times d}$) and file ($\mathbb{R}^{8 \times d}$) embeddings:
   $$\mathbf{E}_{r, c} = \mathbf{e}_{\text{rank}}(r) + \mathbf{e}_{\text{file}}(c)$$
2. **Relative 2D Position Attention Bias**:
   Multi-head self-attention adds a learned pairwise square bias matrix $\mathbf{B}_{\text{rel}} \in \mathbb{R}^{H \times 64 \times 64}$:
   $$\text{Attn}(Q, K) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \mathbf{B}_{\text{rel}}\right) V$$
   This encodes chessboard topology (diagonal, orthogonal, and knight relations) directly into the attention mechanism.

### 3.3 Bilinear Policy Head
Instead of a flattened fully connected projection to 4096 move classes, the **Bilinear Policy Head** projects the 64 square representations into origin queries $Q \in \mathbb{R}^{B \times 64 \times D_p}$ and destination keys $K \in \mathbb{R}^{B \times 64 \times D_p}$ ($D_p = 64$):
$$\mathbf{M}_{\text{from}, \text{to}} = \frac{Q K^T}{\sqrt{D_p}} + \mathbf{B}_{\text{move}}$$
This reduces parameter count, enforces geometric move relationships across the board, and flattens into the 4096-move policy space. A separate promotion head maps rank-8 representations to promotion piece types (Q, R, B, N).

### 3.4 WDL Value Head
The `[CLS]` token is extracted and processed through:
$$\mathbf{v}_{\text{wdl}} = \text{Linear}_{d \to 3}(\text{GELU}(\text{Linear}_{d \to d}(\text{LayerNorm}(\mathbf{h}_{\text{CLS}}))))$$
Output logits correspond to Win, Draw, and Loss probabilities satisfying $\sum P(\text{WDL}) = 1.0$.

---

## 4. Stratified Chess Transformer
`StratifiedChessTransformer` routes positions dynamically to specialized phase experts based on piece count and move ply:
- **Phase 0 (Opening)**: $\text{piece\_count} \ge 24$ or $\text{ply} \le 20$
- **Phase 1 (Middlegame)**: $12 < \text{piece\_count} < 24$
- **Phase 2 (Endgame)**: $\text{piece\_count} \le 12$

Each phase expert is a dedicated `transformer_20m` network. Forward evaluation dynamically batches queries by routed phase expert to maintain optimal inference speed.

---

## 5. Multi-Process Parallel MCTS Architecture

```
                      +-----------------------------+
                      |   Central GPU Evaluator     |
                      | (RTX 5070 Ti, Batches 64-512)|
                      +--------------+--------------+
                                     ^
                         Requests    |    Results
                         (Queue)     |    (Pipes)
                                     v
       +-----------------------------+-----------------------------+
       |                             |                             |
+------v------+               +------v------+               +------v------+
| CPU Worker 1|               | CPU Worker 2|     ...       |CPU Worker 32|
| (Tree Sim)  |               | (Tree Sim)  |               | (Tree Sim)  |
+-------------+               +-------------+               +-------------+
```

### Architecture Features:
1. **Worker Pool**: Up to 32 parallel CPU worker processes running lock-free tree simulations.
2. **Central GPU Evaluator**: Aggregates evaluation requests across all workers, dispatching saturated tensor batches (64–512) directly to Tensor Cores.
3. **Inter-Process Communication**: High-throughput `multiprocessing.Queue` for collection and dedicated zero-copy `multiprocessing.Pipe` channels for per-worker result return.
4. **Root & Leaf Parallelization**: Root visits and edge statistics are aggregated across workers for stable and low-latency decision making.
