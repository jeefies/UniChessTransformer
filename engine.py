"""UniChess Server GameEngine implementation for Transformer.

Exposes `class GameEngine` satisfying Server/models/__init__.py contract:
- setup(fen=None)
- human_move(uci)
- engine_move()
- state()
- undo()
- cleanup()

Shares the underlying TransformerEngine weights across sessions using a thread-safe singleton cache
keyed by (ckpt, device, precision, syzygy_path).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import chess
import torch

TRANSFORMER_ROOT = Path(__file__).resolve().parent
if str(TRANSFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_ROOT))

from engine.engine import TransformerEngine


_SHARED_ENGINES: dict[tuple[str, str, str, str | None], TransformerEngine] = {}
_SHARED_LOCK = threading.Lock()


def get_shared_engine(
    ckpt: str,
    device: str = "cuda",
    precision: str = "fp16",
    syzygy_path: str | None = None,
) -> TransformerEngine:
    key = (str(Path(ckpt).resolve()) if ckpt else "", str(device), str(precision), str(syzygy_path) if syzygy_path else None)
    with _SHARED_LOCK:
        if key not in _SHARED_ENGINES:
            engine = TransformerEngine(
                ckpt,
                device=device,
                precision=precision,
                syzygy_path=syzygy_path,
                mcts_sims=0,  # MCTS is invoked dynamically or per-move
            )
            _SHARED_ENGINES[key] = engine
        return _SHARED_ENGINES[key]


class GameEngine:
    """GameEngine adapter for UniChess Server."""

    def __init__(
        self,
        ckpt: str = str(TRANSFORMER_ROOT / "runs" / "transformer_20m" / "best_model.pt"),
        mcts_sims: int = 800,
        mcts_batch: int = 64,
        device: str = "cuda",
        precision: str = "fp16",
        syzygy_path: str | None = None,
        **kwargs: Any,
    ):
        self.ckpt = ckpt
        self.mcts_sims = int(mcts_sims)
        self.mcts_batch = int(mcts_batch)
        self.device = device
        self.precision = precision
        self.syzygy_path = syzygy_path
        self.extra_kwargs = kwargs

        self.engine = get_shared_engine(
            ckpt=self.ckpt,
            device=self.device,
            precision=self.precision,
            syzygy_path=self.syzygy_path,
        )

        self.board = chess.Board()
        # Per-session search tree root if MCTS tree reuse is desired
        self.root = None

    def setup(self, fen: str | None = None) -> dict:
        """Initializes or resets board state."""
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.root = None
        return self.state()

    def human_move(self, uci: str) -> dict:
        """Pushes human move to self.board without triggering search."""
        move = chess.Move.from_uci(uci)
        self.board.push(move)
        # Advance MCTS root if available
        if self.root is not None:
            try:
                from search.mcts import MCTS
                self.root = MCTS.advance_root(self.root, move)
            except Exception:
                self.root = None
        return self.state()

    def engine_move(self) -> dict:
        """Selects engine move, advances board, and returns status."""
        if self.board.is_game_over():
            return {
                "engine_move": None,
                "fen": self.board.fen(),
                "done": True,
            }

        # Priority 1: Tablebase
        mv = self.engine._from_tablebase(self.board)

        # Priority 2: Opening book
        if mv is None:
            mv = self.engine._from_book(self.board)

        # Priority 3: MCTS (if sims > 0)
        if mv is None and self.mcts_sims > 0:
            from search.mcts import MCTS, MCTSConfig
            mcts_cfg = MCTSConfig(
                simulations=self.mcts_sims,
                batch_size=self.mcts_batch,
                temperature=0.0,
            )
            mcts = MCTS(
                self.engine.evaluate_batch,
                cfg=mcts_cfg,
                tablebase=self.engine.tablebase,
                rng=self.engine.np_rng,
            )
            mv, self.root = mcts.best_move(
                self.board,
                simulations=self.mcts_sims,
                temperature=0.0,
                root=self.root,
            )

        # Priority 4: Network direct
        if mv is None:
            mv = self.engine._from_network(self.board)

        self.board.push(mv)

        # Advance MCTS root for next moves
        if self.root is not None:
            try:
                from search.mcts import MCTS
                self.root = MCTS.advance_root(self.root, mv)
            except Exception:
                self.root = None

        return {
            "engine_move": mv.uci(),
            "fen": self.board.fen(),
            "done": self.board.is_game_over(),
        }

    def state(self) -> dict:
        """Returns board state snapshot."""
        return {
            "fen": self.board.fen(),
            "done": self.board.is_game_over(),
        }

    def undo(self) -> dict:
        """Pops moves: if move_stack >= 2 pop 2, else pop 1 if len >= 1."""
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
        elif len(self.board.move_stack) >= 1:
            self.board.pop()
        self.root = None
        return self.state()

    def cleanup(self) -> None:
        """Releases per-session search tree/cache, clears PyTorch CUDA memory."""
        self.root = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
