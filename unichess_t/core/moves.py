"""Move indexing, promotion indexing, UCI conversion, and legal move masks."""
from __future__ import annotations

import chess
import numpy as np

POLICY_SIZE = 64 * 64  # 4096
PROMO_SIZE = 4

# Queen, Rook, Bishop, Knight order
PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMO_TO_IDX = {p: i for i, p in enumerate(PROMO_PIECES)}
IDX_TO_PROMO = {i: p for i, p in enumerate(PROMO_PIECES)}


def move_to_index(move: chess.Move) -> int:
    """Move -> policy index in [0, 4096). Ignores promotion piece."""
    return move.from_square * 64 + move.to_square


def move_to_promo_index(move: chess.Move) -> int | None:
    """Promotion move -> [0, 4) promotion index, or None if not a promotion."""
    if move.promotion is None:
        return None
    return PROMO_TO_IDX[move.promotion]


def index_to_move(index: int, promo_index: int | None = None) -> chess.Move:
    """Policy index in [0, 4096) + optional promo index -> chess.Move."""
    from_square, to_square = divmod(index, 64)
    promotion = None if promo_index is None else PROMO_PIECES[promo_index]
    return chess.Move(from_square, to_square, promotion=promotion)


def legal_move_indices(board: chess.Board) -> list[int]:
    """Return sorted unique policy indices for legal moves in current oriented board."""
    return sorted({move_to_index(m) for m in board.legal_moves})


def legal_mask(board: chess.Board) -> np.ndarray:
    """Return length 4096 bool mask with True for legal moves."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for m in board.legal_moves:
        mask[move_to_index(m)] = True
    return mask


def uci_to_index(uci_str: str) -> tuple[int, int | None]:
    """Parse UCI string -> (policy_index, promo_index or None)."""
    mv = chess.Move.from_uci(uci_str)
    return move_to_index(mv), move_to_promo_index(mv)


def index_to_uci(index: int, promo_index: int | None = None) -> str:
    """Policy index + promo index -> UCI string."""
    return index_to_move(index, promo_index).uci()
