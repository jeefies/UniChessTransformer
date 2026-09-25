"""T 的模型结构：ConvStem + 2D 相对位置偏置 + 双线性策略头 + WDL 价值头 + MLH 头，
以及按阶段路由的三专家集合 ``StratifiedChessTransformer``。

由 ``unichess_t/model/transformer.py`` 平移而来，**state_dict 键名一个字节都没改**
（旧检查点照常加载；``StratifiedChessTransformer`` 同时注册 ``opening/…`` 与
``experts.N/…`` 两组键，参数量按去重后的 61,031,064 计）。

要改结构先想清楚三件事：``cfg`` 会进检查点、``cfg_conflicts`` 靠它判结构是否变化、
Server 的 ``models/T`` 与训练配方都指着这里的预设名。训练循环与损失在 Kit 里。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from pathlib import Path
from typing import Literal

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    in_channels: int = 19
    d_model: int = 256
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int | None = None
    hidden_dim: int | None = None
    mlp_ratio: float = 8.0 / 3.0
    d_p: int = 64
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_ff is None and self.hidden_dim is not None:
            self.d_ff = self.hidden_dim
        elif self.hidden_dim is None and self.d_ff is not None:
            self.hidden_dim = self.d_ff

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TransformerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: w3(SiLU(w1(x)) * w2(x))."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RelativeAttention(nn.Module):
    """Multi-Head Self-Attention with learned 2D relative position bias on square-to-square attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Learned 2D relative position bias for square-to-square attention (num_heads, 64, 64)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 64, 64))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Pre-pad relative position bias with 0 for the CLS token (token 0)
        # Shape: (1, num_heads, 65, 65) -> broadcasts across batch size B
        attn_bias = F.pad(self.rel_pos_bias, (1, 0, 1, 0)).unsqueeze(0)
        dropout_p = self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.training else 0.0

        # PyTorch SDPA uses optimized FlashAttention / Memory-Efficient Attention kernels
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm / RMSNorm Transformer Block with SwiGLU feed-forward."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        mlp_ratio: float = 8.0 / 3.0,
        norm_type: Literal["layernorm", "rmsnorm"] = "layernorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        norm_fn = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.norm1 = norm_fn(d_model)
        self.attn = RelativeAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = norm_fn(d_model)

        if d_ff is None:
            d_ff = int(round(d_model * mlp_ratio))
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BilinearPolicyHead(nn.Module):
    """Bilinear Policy Head projecting square tokens to Q, K in R^{B x 64 x D_p}.

    Move logits M = (Q K^T) / sqrt(D_p) + bias_move in R^{B x 64 x 64} -> flatten to (B, 4096).
    """

    def __init__(self, d_model: int, d_p: int = 64):
        super().__init__()
        self.d_p = d_p
        self.scale = d_p ** -0.5
        self.wq = nn.Linear(d_model, d_p)
        self.wk = nn.Linear(d_model, d_p)
        self.bias_move = nn.Parameter(torch.zeros(64, 64))
        nn.init.trunc_normal_(self.bias_move, std=0.02)

    def forward(self, square_tokens: torch.Tensor) -> torch.Tensor:
        # square_tokens: (B, 64, d_model)
        Q = self.wq(square_tokens)  # (B, 64, d_p)
        K = self.wk(square_tokens)  # (B, 64, d_p)
        M = torch.bmm(Q, K.transpose(1, 2)) * self.scale + self.bias_move  # (B, 64, 64)
        return M.flatten(1)  # (B, 4096)


class ChessTransformer(nn.Module):
    """SOTA Chess Transformer with ConvStem, learned 2D relative position bias,

    bilinear policy head, promotion head, and WDL value head.
    """

    def __init__(self, cfg: TransformerConfig | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = TransformerConfig(**kwargs)
        elif kwargs:
            # Override any kwargs
            cfg_dict = cfg.to_dict()
            cfg_dict.update(kwargs)
            cfg = TransformerConfig.from_dict(cfg_dict)

        self.cfg = cfg
        d_model = cfg.d_model
        norm_fn = RMSNorm if cfg.norm_type == "rmsnorm" else nn.LayerNorm

        # Stem: ConvStem (3x3 conv 19 -> d_model)
        self.stem = nn.Conv2d(cfg.in_channels, d_model, kernel_size=3, padding=1)

        # Learned 2D rank & file positional embeddings
        self.rank_embed = nn.Parameter(torch.zeros(8, d_model))
        self.file_embed = nn.Parameter(torch.zeros(8, d_model))
        nn.init.trunc_normal_(self.rank_embed, std=0.02)
        nn.init.trunc_normal_(self.file_embed, std=0.02)

        # [CLS] token (65 tokens total: 64 squares + 1 CLS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer Backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=cfg.num_heads,
                d_ff=cfg.d_ff,
                mlp_ratio=cfg.mlp_ratio,
                norm_type=cfg.norm_type,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = norm_fn(d_model)

        # Bilinear Policy Head
        self.policy_head = BilinearPolicyHead(d_model, d_p=cfg.d_p)

        # Promotion Head: Linear head mapping from 8th rank square representations (squares 56..63)
        # + CLS token to 4 logits (Q, R, B, N)
        self.promo_head = nn.Sequential(
            norm_fn(9 * d_model),
            nn.Linear(9 * d_model, 4),
        )

        # Value Head: CLS token -> LayerNorm -> MLP -> 3 logits (WDL)
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

        # Moves-Left Head (MLH): CLS token -> Linear(d_model, 64) -> SiLU -> Linear(64, 1)
        self.mlh_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_mlh: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Input:
            x: Tensor of shape (B, 19, 8, 8) or (B, 64, 19)
            return_mlh: If True, also return mlh tensor of shape (B,)
        Returns:
            If return_mlh is False:
                (policy_logits, promo_logits, value_wdl)
            If return_mlh is True:
                (policy_logits, promo_logits, value_wdl, mlh)
        """
        if x.dim() == 3 and x.shape[1] == 64 and x.shape[2] == self.cfg.in_channels:
            # Convert (B, 64, 19) -> (B, 19, 8, 8)
            B = x.shape[0]
            x = x.transpose(1, 2).contiguous().view(B, self.cfg.in_channels, 8, 8)
        elif x.dim() == 4:
            B = x.shape[0]
        else:
            raise ValueError(f"Expected input shape (B, 19, 8, 8) or (B, 64, 19), got {x.shape}")

        # ConvStem
        feat = self.stem(x)  # (B, d_model, 8, 8)
        sq_tokens = feat.flatten(2).transpose(1, 2)  # (B, 64, d_model)

        # 2D rank and file positional embeddings: square i = rank * 8 + file
        pos_2d = (self.rank_embed.unsqueeze(1) + self.file_embed.unsqueeze(0)).view(1, 64, self.cfg.d_model)
        sq_tokens = sq_tokens + pos_2d

        # Prepend CLS token (sequence length becomes 65)
        cls = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls, sq_tokens], dim=1)  # (B, 65, d_model)

        for block in self.blocks:
            x_seq = block(x_seq)

        x_seq = self.final_norm(x_seq)

        cls_out = x_seq[:, 0]          # (B, d_model)
        sq_out = x_seq[:, 1:]          # (B, 64, d_model)

        # Heads
        policy_logits = self.policy_head(sq_out)   # (B, 4096)

        # 8th rank squares: rank 7 (squares 56..63)
        rank8_sq = sq_out[:, 56:64, :].reshape(B, 8 * self.cfg.d_model)
        promo_in = torch.cat([cls_out, rank8_sq], dim=1)  # (B, 9 * d_model)
        promo_logits = self.promo_head(promo_in)          # (B, 4)

        value_wdl = self.value_head(cls_out)              # (B, 3)

        if return_mlh:
            mlh = self.mlh_head(cls_out).squeeze(-1)       # (B,)
            return policy_logits, promo_logits, value_wdl, mlh

        return policy_logits, promo_logits, value_wdl


class StratifiedChessTransformer(nn.Module):
    """Phase-stratified ensemble routing chess positions to 3 phase expert networks:

    - Phase 0: Opening (piece_count >= 24 or ply <= 20)
    - Phase 1: Middlegame (12 < piece_count < 24)
    - Phase 2: Endgame (piece_count <= 12)
    Each expert is a ~20M model (transformer_20m).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.opening = transformer_20m(**kwargs)
        self.middlegame = transformer_20m(**kwargs)
        self.endgame = transformer_20m(**kwargs)
        self.experts = nn.ModuleList([self.opening, self.middlegame, self.endgame])
        self.cfg = self.opening.cfg

    def load_base_checkpoint(self, base_ckpt_path: str | Path) -> StratifiedChessTransformer:
        """Loads a single 20M checkpoint and broadcasts parameters into opening, middlegame, and endgame experts."""
        path = Path(base_ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Base checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        clean_state_dict = {
            k.removeprefix("_orig_mod."): v
            for k, v in state_dict.items()
        }

        has_expert_prefix = any(k.startswith(("opening.", "middlegame.", "endgame.", "experts.")) for k in clean_state_dict)
        if has_expert_prefix:
            self.load_state_dict(clean_state_dict, strict=False)
        else:
            for expert in self.experts:
                expert.load_state_dict(clean_state_dict, strict=False)
            # If base checkpoint did not include mlh_head, synchronize mlh_head across all experts
            has_mlh = any("mlh_head" in k for k in clean_state_dict)
            if not has_mlh:
                for expert in self.experts[1:]:
                    expert.mlh_head.load_state_dict(self.experts[0].mlh_head.state_dict())

        return self

    @staticmethod
    def route_index(board: chess.Board) -> int:
        piece_count = len(board.piece_map())
        ply = board.ply()
        if piece_count >= 24 or ply <= 20:
            return 0
        elif 12 < piece_count < 24:
            return 1
        else:
            return 2

    def forward(
        self,
        x: torch.Tensor,
        boards: list[chess.Board] | chess.Board | None = None,
        route_indices: list[int] | torch.Tensor | None = None,
        return_mlh: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic phase routing.

        Supports single-board or batched evaluation.
        """
        if x.dim() == 3 and x.shape[1] == 64 and x.shape[2] == self.cfg.in_channels:
            B = x.shape[0]
            x = x.transpose(1, 2).contiguous().view(B, self.cfg.in_channels, 8, 8)
        elif x.dim() == 4:
            B = x.shape[0]
        else:
            raise ValueError(f"Expected input shape (B, 19, 8, 8) or (B, 64, 19), got {x.shape}")

        if boards is not None:
            if isinstance(boards, chess.Board):
                routes = [self.route_index(boards)] * B
            else:
                routes = [self.route_index(b) for b in boards]
        elif route_indices is not None:
            if isinstance(route_indices, torch.Tensor):
                routes = route_indices.tolist()
            else:
                routes = list(route_indices)
        else:
            # Extract piece count from piece planes (planes 0..11)
            piece_counts = x[:, :12].sum(dim=(1, 2, 3)).round().long()
            routes = []
            for pc in piece_counts:
                c = int(pc.item())
                if c >= 24:
                    routes.append(0)
                elif c > 12:
                    routes.append(1)
                else:
                    routes.append(2)

        # Single expert fast path
        unique_routes = set(routes)
        if len(unique_routes) == 1:
            expert_idx = routes[0]
            return self.experts[expert_idx](x, return_mlh=return_mlh)

        p_out = None
        pr_out = None
        v_out = None
        mlh_out = None

        for phase in range(3):
            idxs = [i for i, r in enumerate(routes) if r == phase]
            if not idxs:
                continue
            sub_x = x[idxs]
            if return_mlh:
                sub_p, sub_pr, sub_v, sub_m = self.experts[phase](sub_x, return_mlh=True)
            else:
                sub_p, sub_pr, sub_v = self.experts[phase](sub_x, return_mlh=False)

            if p_out is None:
                p_out = torch.empty(B, 4096, device=x.device, dtype=sub_p.dtype)
                pr_out = torch.empty(B, 4, device=x.device, dtype=sub_pr.dtype)
                v_out = torch.empty(B, 3, device=x.device, dtype=sub_v.dtype)
                if return_mlh:
                    mlh_out = torch.empty(B, device=x.device, dtype=sub_m.dtype)
            p_out[idxs] = sub_p
            pr_out[idxs] = sub_pr
            v_out[idxs] = sub_v
            if return_mlh:
                mlh_out[idxs] = sub_m

        if return_mlh:
            return p_out, pr_out, v_out, mlh_out
        return p_out, pr_out, v_out


# Presets
# Presets：**配置是唯一真相**，工厂只是它的快捷方式。新代码（kit 适配器、训练配置）
# 一律用 ``PRESET_CONFIGS[name]`` 拿 TransformerConfig，不要直接调工厂——工厂返回的是模型，
# 而训练路径要先有 cfg 才能判结构冲突、写导出权重。
_PRESET_SPECS: dict[str, dict] = {
    "transformer_tiny":   dict(d_model=192, num_layers=6,  num_heads=6,  d_ff=768),
    "transformer_small":  dict(d_model=256, num_layers=8,  num_heads=8,  d_ff=682),
    "transformer_medium": dict(d_model=384, num_layers=10, num_heads=12, d_ff=1024),
    "transformer_large":  dict(d_model=512, num_layers=12, num_heads=16, d_ff=1152),
    # 11 层 / 384 / 12 头，SwiGLU 的 d_ff=1024 对应旧式 2 矩阵 MLP 的 hidden_dim=1536
    # （参数 footprint 相同，约 20.2M）
    "transformer_20m":    dict(d_model=384, num_layers=11, num_heads=12, d_ff=1024, hidden_dim=1536),
    "transformer_50m":    dict(d_model=512, num_layers=17, num_heads=16, d_ff=1160, hidden_dim=2048),
}

PRESET_CONFIGS: dict[str, TransformerConfig] = {
    name: TransformerConfig(**spec) for name, spec in _PRESET_SPECS.items()
}
#: 分层集合里每个专家的结构就是 20M
STRATIFIED_CFG = PRESET_CONFIGS["transformer_20m"]


def _preset_cfg(name: str) -> TransformerConfig:
    if name not in PRESET_CONFIGS:
        raise KeyError(f"没有模型预设 {name!r}（可选：{', '.join(PRESET_CONFIGS)}）")
    return PRESET_CONFIGS[name]


def transformer_tiny(**kwargs) -> ChessTransformer:
    """6 layers, d_model=192, 6 heads (~3.8M params)."""
    cfg = replace(_preset_cfg("transformer_tiny"), **kwargs)
    return ChessTransformer(cfg)


def transformer_small(**kwargs) -> ChessTransformer:
    """8 layers, d_model=256, 8 heads (~6.7M params) - Default recommended."""
    cfg = replace(_preset_cfg("transformer_small"), **kwargs)
    return ChessTransformer(cfg)


def transformer_medium(**kwargs) -> ChessTransformer:
    """10 layers, d_model=384, 12 heads (~18.4M params)."""
    cfg = replace(_preset_cfg("transformer_medium"), **kwargs)
    return ChessTransformer(cfg)


def transformer_large(**kwargs) -> ChessTransformer:
    """12 layers, d_model=512, 16 heads (~35M params)."""
    cfg = replace(_preset_cfg("transformer_large"), **kwargs)
    return ChessTransformer(cfg)


def transformer_20m(**kwargs) -> ChessTransformer:
    """11 layers, d_model=384, num_heads=12, hidden_dim=1536 (~20.2M params)."""
    cfg = replace(_preset_cfg("transformer_20m"), **kwargs)
    return ChessTransformer(cfg)


def transformer_50m(**kwargs) -> ChessTransformer:
    """17 layers, d_model=512, num_heads=16, hidden_dim=2048 (~49.7M params)."""
    cfg = replace(_preset_cfg("transformer_50m"), **kwargs)
    return ChessTransformer(cfg)


def stratified_20m(**kwargs) -> StratifiedChessTransformer:
    """Stratified ensemble of 3 x 20M phase experts (~60.8M total params, ~20.2M active per position)."""
    return StratifiedChessTransformer(**kwargs)



def cfg_conflicts(saved: dict, cfg: TransformerConfig) -> list[str]:
    """比对 checkpoint 里存的 cfg 与当前 cfg，返回所有不一致处（空列表 = 可以加载）。

    与 R 的 ``cfg_conflicts`` 同规则：共有字段逐个相等；checkpoint 缺的字段当前值必须等于
    默认值（默认值 = 旧结构）；checkpoint 有、当前没有的字段一律算冲突（往回退版本）。
    不能直接 ``saved == cfg.__dict__``——TransformerConfig 加过字段，老检查点没有那些 key，
    字典相等会让每个旧 run 都加载失败，而结构其实没变。
    """
    defaults = {f.name: f.default for f in fields(TransformerConfig)}
    cur = asdict(cfg)
    out: list[str] = []
    for k in sorted(set(saved) | set(cur)):
        if k not in cur:
            out.append(f"{k}: checkpoint 有（={saved[k]!r}），当前 TransformerConfig 已无此字段")
        elif k not in saved:
            if cur[k] != defaults.get(k):
                out.append(f"{k}: checkpoint 无此字段（等同默认值 "
                           f"{defaults.get(k)!r}），当前为 {cur[k]!r}")
        elif saved[k] != cur[k]:
            out.append(f"{k}: checkpoint={saved[k]!r}，当前={cur[k]!r}")
    return out


def count_params(model: nn.Module) -> dict[str, int]:
    """按顶层模块统计参数量（``StratifiedChessTransformer`` 只算去重后的真实数量）。"""
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    groups["TOTAL"] = sum(p.numel() for p in model.parameters())
    return groups


def _clean_state_dict(sd: dict) -> dict:
    """去掉 ``_orig_mod.`` 前缀（torch.compile 保存的检查点）。"""
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def load_model(ckpt_path, device="cpu") -> tuple[nn.Module, TransformerConfig, int, dict]:
    """按 checkpoint 内容选结构并加载权重 → (eval 模式模型, cfg, step, 原始 checkpoint)。

    * ``preset == "stratified_20m"`` 或 state_dict 里出现 ``experts.`` / ``opening.`` 前缀
      → ``StratifiedChessTransformer``（三专家）；
    * 否则 ``ChessTransformer``，cfg 用 checkpoint 里的字段重建。

    **只许缺 ``mlh_head.*``**：P3 之前的检查点没有 moves-left 头（推理不用它），其余键名或
    形状对不上一律报错——静默跳过会让推理悄悄用随机权重，那比直接报错难查得多。
    """
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"权重不存在：{p}")
    ckpt = torch.load(p, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{p} 不是 T 的 checkpoint（顶层不是 dict）")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if not isinstance(sd, dict):
        raise ValueError(f"{p} 里没有 model state_dict")
    sd = _clean_state_dict(sd)
    stratified = (ckpt.get("preset") == "stratified_20m"
                  or any(k.startswith(("experts.", "opening.", "middlegame.", "endgame."))
                         for k in sd))
    if stratified:
        model, cfg = stratified_20m(), STRATIFIED_CFG
    else:
        cfg = TransformerConfig.from_dict(ckpt.get("cfg", {}) or {})
        model = ChessTransformer(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "mlh_head." not in k]
    if real_missing or unexpected:
        raise ValueError(f"{p} 与当前结构不符：缺 {real_missing[:4]}，多 {unexpected[:4]}")
    return model.eval(), cfg, int(ckpt.get("step", 0)), ckpt


PRESETS = {
    "transformer_tiny": transformer_tiny,
    "transformer_small": transformer_small,
    "transformer_medium": transformer_medium,
    "transformer_large": transformer_large,
    "transformer_20m": transformer_20m,
    "transformer_50m": transformer_50m,
    "stratified_20m": stratified_20m,
}


def preset_config(name: str) -> TransformerConfig:
    """预设名 → ``TransformerConfig``（训练适配器与配置校验都用这个）。"""
    return _preset_cfg(name)


def create_transformer(preset_name: str = "transformer_small", init_from: str | Path | None = None, **kwargs) -> nn.Module:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    model = PRESETS[preset_name](**kwargs)
    if init_from is not None:
        if isinstance(model, StratifiedChessTransformer):
            model.load_base_checkpoint(init_from)
        elif hasattr(model, "load_state_dict"):
            ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
            sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            clean_sd = _clean_state_dict(sd)
            model.load_state_dict(clean_sd, strict=False)
    return model
