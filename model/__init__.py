"""UniChessTransformer model, dataset, and loss modules."""
from .transformer import (
    TransformerConfig,
    ChessTransformer,
    transformer_tiny,
    transformer_small,
    transformer_medium,
    transformer_large,
    create_transformer,
    PRESETS,
)
from .dataset import (
    RECORD_DTYPE,
    BatchShardDataset,
    ShardDataset,
    decode_batch,
    decode_targets,
    get_shards,
    make_loader,
)
from .loss import ChessLoss, LossOutput, compute_metrics

__all__ = [
    "TransformerConfig",
    "ChessTransformer",
    "transformer_tiny",
    "transformer_small",
    "transformer_medium",
    "transformer_large",
    "create_transformer",
    "PRESETS",
    "RECORD_DTYPE",
    "BatchShardDataset",
    "ShardDataset",
    "decode_batch",
    "decode_targets",
    "get_shards",
    "make_loader",
    "ChessLoss",
    "LossOutput",
    "compute_metrics",
]
