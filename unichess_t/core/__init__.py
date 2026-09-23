"""UniChessTransformer core encoding and move indexing modules."""
from .encoding import NUM_PLANES, BOARD_SIZE, INPUT_SHAPE, orient, orient_square, orient_move, unorient_move, encode
from .moves import (
    POLICY_SIZE,
    PROMO_SIZE,
    PROMO_PIECES,
    PROMO_TO_IDX,
    IDX_TO_PROMO,
    move_to_index,
    move_to_promo_index,
    index_to_move,
    legal_move_indices,
    legal_mask,
    uci_to_index,
    index_to_uci,
)

__all__ = [
    "NUM_PLANES",
    "BOARD_SIZE",
    "INPUT_SHAPE",
    "orient",
    "orient_square",
    "orient_move",
    "unorient_move",
    "encode",
    "POLICY_SIZE",
    "PROMO_SIZE",
    "PROMO_PIECES",
    "PROMO_TO_IDX",
    "IDX_TO_PROMO",
    "move_to_index",
    "move_to_promo_index",
    "index_to_move",
    "legal_move_indices",
    "legal_mask",
    "uci_to_index",
    "index_to_uci",
]
