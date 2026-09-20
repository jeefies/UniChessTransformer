"""19-plane canonical board encoding (100% interoperable with UniChess).

Board encoding convention: Always encode from the perspective of the side to move.
If black is to move, mirror the board vertically and swap colors (chess.Board.mirror).
Planes layout (19 total):
    0-5:   Own pieces (P, N, B, R, Q, K)
    6-11:  Opponent pieces (P, N, B, R, Q, K)
    12:    Own kingside castling rights (filled 0/1)
    13:    Own queenside castling rights
    14:    Opponent kingside castling rights
    15:    Opponent queenside castling rights
    16:    En passant target square (single 1.0)
    17:    Halfmove clock / 100
    18:    Repetition count / 2
"""
from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 19
BOARD_SIZE = 8
INPUT_SHAPE = (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

_PIECE_ORDER = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
_PIECE_PLANE = {p: i for i, p in enumerate(_PIECE_ORDER)}


def orient(board: chess.Board) -> chess.Board:
    """Return board oriented so the side to move plays as White."""
    return board if board.turn == chess.WHITE else board.mirror()


def orient_square(square: int, turn: bool) -> int:
    """Map square to oriented perspective."""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def orient_move(move: chess.Move, turn: bool) -> chess.Move:
    """Map move to oriented perspective."""
    if turn == chess.WHITE:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def unorient_move(move: chess.Move, turn: bool) -> chess.Move:
    """Inverse of orient_move."""
    return orient_move(move, turn)


def encode(board: chess.Board, repetitions: int | None = None) -> np.ndarray:
    """Encode board into float32 (19, 8, 8) tensor."""
    if repetitions is None:
        repetitions = 2 if board.is_repetition(3) else (1 if board.is_repetition(2) else 0)

    b = orient(board)
    planes = np.zeros(INPUT_SHAPE, dtype=np.float32)

    # 0-11: Pieces
    for square, piece in b.piece_map().items():
        plane = _PIECE_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane += 6
        row, col = divmod(square, 8)
        planes[plane, row, col] = 1.0

    # 12-15: Castling rights
    if b.has_kingside_castling_rights(chess.WHITE):
        planes[12] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):
        planes[14] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK):
        planes[15] = 1.0

    # 16: En passant target square
    if b.ep_square is not None:
        row, col = divmod(b.ep_square, 8)
        planes[16, row, col] = 1.0

    # 17: Halfmove clock
    planes[17] = min(b.halfmove_clock, 100) / 100.0

    # 18: Repetition count
    planes[18] = min(repetitions, 2) / 2.0

    return planes
