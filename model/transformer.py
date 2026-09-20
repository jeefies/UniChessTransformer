"""SOTA Chess Transformer architecture with 2D relative position bias and bilinear policy head."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Input:
            x: Tensor of shape (B, 19, 8, 8) or (B, 64, 19)
        Returns:
            policy_logits: (B, 4096)
            promo_logits: (B, 4)
            value_wdl: (B, 3)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            return self.experts[expert_idx](x)

        p_out = torch.empty(B, 4096, device=x.device, dtype=x.dtype)
        pr_out = torch.empty(B, 4, device=x.device, dtype=x.dtype)
        v_out = torch.empty(B, 3, device=x.device, dtype=x.dtype)

        for phase in range(3):
            idxs = [i for i, r in enumerate(routes) if r == phase]
            if not idxs:
                continue
            sub_x = x[idxs]
            sub_p, sub_pr, sub_v = self.experts[phase](sub_x)
            p_out[idxs] = sub_p
            pr_out[idxs] = sub_pr
            v_out[idxs] = sub_v

        return p_out, pr_out, v_out


# Presets
def transformer_tiny(**kwargs) -> ChessTransformer:
    """6 layers, d_model=192, 6 heads (~3.8M params)."""
    cfg = TransformerConfig(
        d_model=192,
        num_layers=6,
        num_heads=6,
        d_ff=768,
        d_p=64,
        **kwargs,
    )
    return ChessTransformer(cfg)


def transformer_small(**kwargs) -> ChessTransformer:
    """8 layers, d_model=256, 8 heads (~6.7M params) - Default recommended."""
    cfg = TransformerConfig(
        d_model=256,
        num_layers=8,
        num_heads=8,
        d_ff=682,
        d_p=64,
        **kwargs,
    )
    return ChessTransformer(cfg)


def transformer_medium(**kwargs) -> ChessTransformer:
    """10 layers, d_model=384, 12 heads (~18.4M params)."""
    cfg = TransformerConfig(
        d_model=384,
        num_layers=10,
        num_heads=12,
        d_ff=1024,
        d_p=64,
        **kwargs,
    )
    return ChessTransformer(cfg)


def transformer_large(**kwargs) -> ChessTransformer:
    """12 layers, d_model=512, 16 heads (~35M params)."""
    cfg = TransformerConfig(
        d_model=512,
        num_layers=12,
        num_heads=16,
        d_ff=1152,
        d_p=64,
        **kwargs,
    )
    return ChessTransformer(cfg)


def transformer_20m(**kwargs) -> ChessTransformer:
    """11 layers, d_model=384, num_heads=12, hidden_dim=1536 (~20.2M params)."""
    hidden_dim = kwargs.pop("hidden_dim", 1536)
    # In SwiGLU, 3 matrices of d_ff=1024 match the 2-matrix standard MLP parameter footprint of hidden_dim=1536 (~20.2M params)
    d_ff = kwargs.pop("d_ff", 1024 if hidden_dim == 1536 else hidden_dim)
    cfg = TransformerConfig(
        d_model=384,
        num_layers=11,
        num_heads=12,
        d_ff=d_ff,
        hidden_dim=hidden_dim,
        d_p=64,
        **kwargs,
    )
    return ChessTransformer(cfg)


def transformer_50m(**kwargs) -> ChessTransformer:
    """17 layers, d_model=512, num_heads=16, hidden_dim=2048 (~49.7M params)."""
    hidden_dim = kwargs.pop("hidden_dim", 2048)
    d_ff = kwargs.pop("d_ff", 1160 if hidden_dim == 2048 else hidden_dim)
    cfg = TransformerConfig(
        d_model=512,
        num_layers=17,
        num_heads=16,
        d_ff=d_ff,
        hidden_dim=hidden_dim,
        d_p=64,
        **kwargs,
    )
    return ChessTransformer(cfg)


def stratified_20m(**kwargs) -> StratifiedChessTransformer:
    """Stratified ensemble of 3 x 20M phase experts (~60.8M total params, ~20.2M active per position)."""
    return StratifiedChessTransformer(**kwargs)


PRESETS = {
    "transformer_tiny": transformer_tiny,
    "transformer_small": transformer_small,
    "transformer_medium": transformer_medium,
    "transformer_large": transformer_large,
    "transformer_20m": transformer_20m,
    "transformer_50m": transformer_50m,
    "stratified_20m": stratified_20m,
}


def create_transformer(preset_name: str = "transformer_small", **kwargs) -> nn.Module:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset_name](**kwargs)
