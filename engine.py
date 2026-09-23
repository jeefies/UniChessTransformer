"""UniChess Server GameEngine implementation for Transformer.

Exposes `class GameEngine` satisfying Server/models/__init__.py contract:
- setup(fen=None)
- human_move(uci)
- engine_move()
- state()
- undo()
- cleanup()

Shares the underlying TransformerEngine weights across sessions using a thread-safe singleton cache
keyed by (ckpt, device, precision, syzygy_path, use_cpp_mcts, book_path).
"""
from __future__ import annotations

import logging
from pathlib import Path
import random
import sys
import threading
from typing import Any

import chess
import numpy as np
import torch

TRANSFORMER_ROOT = Path(__file__).resolve().parent
if str(TRANSFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_ROOT))

from unichess_t.engine.engine import TransformerEngine

logger = logging.getLogger(__name__)

_SHARED_ENGINES: dict[tuple[str, str, str, str | None, bool, str | None], TransformerEngine] = {}
_SHARED_LOCK = threading.Lock()


def get_shared_engine(
    ckpt: str,
    device: str = "cuda",
    precision: str = "fp16",
    syzygy_path: str | None = None,
    use_cpp_mcts: bool = True,
    book_path: str | None = None,
) -> TransformerEngine:
    key = (
        str(Path(ckpt).resolve()) if ckpt else "",
        str(device),
        str(precision),
        str(syzygy_path) if syzygy_path else None,
        bool(use_cpp_mcts),
        str(Path(book_path).resolve()) if book_path else None,
    )
    with _SHARED_LOCK:
        if key not in _SHARED_ENGINES:
            engine = TransformerEngine(
                ckpt,
                device=device,
                precision=precision,
                syzygy_path=syzygy_path,
                mcts_sims=0,  # MCTS is invoked dynamically or per-move
                use_cpp_mcts=use_cpp_mcts,
                book_path=book_path,
            )
            _SHARED_ENGINES[key] = engine
        return _SHARED_ENGINES[key]


class GameEngine:
    """GameEngine adapter for UniChess Server.

    Root move choice is greedy (argmax over root visits) by default.

    ``temperature`` activates sampling on the root *visit* distribution:
    0 = greedy argmax, 1 = sample proportional to visits.  Values above 1 are
    clamped to 1 because flattening past the visit distribution was measured to
    degrade move quality badly (mean Stockfish rank over 56 moves:
    t=0 -> 1.71, t=1.0 -> 3.36, t=1.5 -> 5.55, i.e. below the baseline's 4.29).
    ``root_top_k`` bounds that damage by restricting sampling to the K most
    visited root moves, so variety can never promote a badly ranked move.
    """

    #: Warn at most once per process about a temperature above the safe range.
    _TEMP_WARNED = False

    def __init__(
        self,
        ckpt: str = str(TRANSFORMER_ROOT / "runs" / "transformer_20m" / "best_model.pt"),
        mcts_sims: int = 800,
        mcts_batch: int = 64,
        use_cpp_mcts: bool = True,
        device: str = "cuda",
        precision: str = "fp16",
        syzygy_path: str | None = None,
        book_path: str | None = None,
        temperature: float = 0.0,
        root_top_k: int = 0,
        **kwargs: Any,
    ):
        self.ckpt = ckpt
        self.mcts_sims = int(mcts_sims)
        self.mcts_batch = int(mcts_batch)
        self.use_cpp_mcts = bool(use_cpp_mcts)
        self.device = device
        self.precision = precision
        self.syzygy_path = syzygy_path
        self.book_path = book_path
        # <= 0 keeps the deterministic argmax behaviour; > 1 is clamped.
        self.temperature = self._clamp_temperature(temperature)
        # 0 / negative = no cap; 1 = only the most visited move.
        self.root_top_k = max(0, int(root_top_k))
        self.extra_kwargs = kwargs
        # OS-entropy RNG so sampling differs per process/restart.
        self.rng = random.Random(None)

        self.engine = get_shared_engine(
            ckpt=self.ckpt,
            device=self.device,
            precision=self.precision,
            syzygy_path=self.syzygy_path,
            use_cpp_mcts=self.use_cpp_mcts,
            book_path=self.book_path,
        )

        # 每个 GameEngine 独占一个 MCTSCpp：共享引擎按 ckpt 等键跨预设复用，
        # 而 C++ search() 会写入实例的 cfg.temperature，多会话共用会互相覆盖。
        self.cpp_mcts = None
        if self.engine.cpp_mcts is not None:
            from unichess_t.search.cpp import load_cpp_mcts
            self.cpp_mcts = load_cpp_mcts()()

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
                from unichess_t.search.mcts import MCTS
                self.root = MCTS.advance_root(self.root, move)
            except Exception:
                self.root = None
        return self.state()

    @classmethod
    def _clamp_temperature(cls, temperature: float) -> float:
        """Clamp to [0, 1]; t>1 flattens past the visit distribution and is measured harmful."""
        t = float(temperature)
        if t <= 0.0:
            return 0.0
        if t > 1.0:
            if not cls._TEMP_WARNED:
                cls._TEMP_WARNED = True
                logger.warning(
                    "temperature %s > 1.0 is clamped to 1.0: values above 1 flatten "
                    "the root visit distribution past proportional-to-visits sampling "
                    "and were measured to cost real move quality (mean SF rank 3.36 at "
                    "t=1.0 vs 5.55 at t=1.5). Use root_top_k to shape variety instead.",
                    t,
                )
            return 1.0
        return t

    def _select_root_move(
        self,
        visits: dict[str, float] | list[tuple[str, float]],
        fallback_uci: str | None,
    ) -> chess.Move | None:
        """Pick a root move from visit counts, honouring temperature and root_top_k.

        Greedy (argmax over visits) when temperature <= 0.  Otherwise samples
        weights** (1/temperature) restricted to the root_top_k most visited
        moves, so the move picked can never be worse ranked than the K-th root
        move.  Sampling is only ever allowed to *move away* from the most
        visited move when root_top_k > 1.
        """
        if isinstance(visits, dict):
            items = list(visits.items())
        else:
            items = list(visits)
        items = [(uci, float(n)) for uci, n in items]
        items = [it for it in items if it[1] > 0]
        if not items:
            return chess.Move.from_uci(fallback_uci) if fallback_uci else None

        # Most-visited move first; ties broken by UCI so the result is stable.
        items.sort(key=lambda it: (-it[1], it[0]))
        greedy_uci = items[0][0]

        if self.temperature <= 0.0:
            return chess.Move.from_uci(greedy_uci)
        if self.root_top_k == 1:
            return chess.Move.from_uci(greedy_uci)

        if self.root_top_k > 0:
            k = min(self.root_top_k, len(items))
        else:
            k = len(items)
        if k <= 1:
            return chess.Move.from_uci(greedy_uci)

        candidates = items[:k]
        weights = np.array([n for _, n in candidates], dtype=np.float64)
        # t<1 sharpens toward the greedy move, t==1 is proportional to visits.
        weights = weights ** (1.0 / self.temperature)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return chess.Move.from_uci(greedy_uci)

        probs = weights / total
        pick = self.rng.choices(range(k), weights=probs.tolist(), k=1)[0]
        return chess.Move.from_uci(candidates[pick][0])

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
            if self.cpp_mcts is not None:
                try:
                    # Search greedy and sample root visits here, so top-K
                    # narrowing bounds the cost of every temperature setting.
                    move_str, metrics = self.cpp_mcts.search(
                        self.board.fen(),
                        self.engine.evaluate_tensor,
                        simulations=self.mcts_sims,
                        batch_size=self.mcts_batch,
                        temperature=0.0,
                        syzygy_path=self.syzygy_path,
                    )
                    if move_str:
                        visits = (metrics or {}).get("policy")
                        selected = self._select_root_move(visits, move_str)
                        if selected is not None and selected in self.board.legal_moves:
                            mv = selected
                except Exception as e:
                    logger.warning(f"C++ MCTS failed in GameEngine ({e}), falling back to Python MCTS")

            if mv is None:
                from unichess_t.search.mcts import MCTS, MCTSConfig
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
                greedy_mv, self.root = mcts.best_move(
                    self.board,
                    simulations=self.mcts_sims,
                    temperature=0.0,
                    root=self.root,
                )
                if greedy_mv is not None:
                    visits = []
                    if self.root is not None and self.root.expanded:
                        visits = [(m.uci(), int(n)) for m, n in zip(self.root.moves, self.root.N)]
                    selected = self._select_root_move(visits, greedy_mv.uci())
                    if selected is not None and selected in self.board.legal_moves:
                        mv = selected

        # Priority 4: Network direct
        if mv is None:
            mv = self.engine._from_network(self.board)

        self.board.push(mv)

        # Advance MCTS root for next moves
        if self.root is not None:
            try:
                from unichess_t.search.mcts import MCTS
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
