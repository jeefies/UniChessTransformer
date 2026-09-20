"""Transformer inference engine wrapper with batched evaluation, fp16/bf16 acceleration, and search integration."""
from __future__ import annotations

import logging
from pathlib import Path
import random
import sys
from typing import Literal

import chess
import numpy as np
import torch
import torch.nn as nn

from core.encoding import encode, orient_move
from core.moves import move_to_index, move_to_promo_index
from model.transformer import ChessTransformer, TransformerConfig
from search.mcts import MCTS, MCTSConfig

logger = logging.getLogger(__name__)


class TransformerEngine:
    """High-performance Chess Transformer engine supporting direct inference and batched MCTS."""

    def __init__(
        self,
        model_or_ckpt: str | Path | ChessTransformer,
        *,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        precision: Literal["fp16", "bf16", "fp32"] = "fp16",
        syzygy_path: str | Path | None = None,
        book_path: str | Path | None = None,
        book_plies: int = 10,
        temperature: float = 0.0,
        mcts_sims: int = 0,
        mcts_batch: int = 128,
        use_cpp_mcts: bool = True,
        seed: int | None = None,
    ):
        self.device = torch.device(device)
        self.precision = precision
        self.temperature = temperature
        self.book_plies = book_plies
        self.mcts_batch = mcts_batch
        self.use_cpp_mcts = use_cpp_mcts
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Model loading
        if isinstance(model_or_ckpt, (nn.Module, ChessTransformer)):
            self.model = model_or_ckpt.to(self.device).eval()
            self.cfg = getattr(self.model, "cfg", None)
        else:
            ckpt_path = Path(model_or_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            preset = ckpt.get("preset", None)
            cfg_dict = ckpt.get("cfg", {})
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            if preset == "stratified_20m" or any(k.startswith("experts.") for k in state_dict.keys()):
                from model.transformer import stratified_20m
                self.model = stratified_20m().to(self.device).eval()
                self.cfg = getattr(self.model, "cfg", None)
            else:
                self.cfg = TransformerConfig.from_dict(cfg_dict) if cfg_dict else TransformerConfig()
                self.model = ChessTransformer(self.cfg).to(self.device).eval()
            self.model.load_state_dict(state_dict)

        # Autocast dtype
        if self.device.type == "cuda":
            if precision == "fp16":
                self.dtype = torch.float16
                self.autocast = True
            elif precision == "bf16":
                self.dtype = torch.bfloat16
                self.autocast = True
            else:
                self.dtype = torch.float32
                self.autocast = False
        else:
            self.dtype = torch.float32
            self.autocast = False

        # Syzygy tablebase
        self.tablebase = None
        self._tb_missing: set[str] = set()
        if syzygy_path and Path(syzygy_path).is_dir():
            try:
                import chess.syzygy
                self.tablebase = chess.syzygy.open_tablebase(str(syzygy_path))
            except Exception as e:
                print(f"info string Failed to load Syzygy tablebase: {e}", file=sys.stderr)

        # Polyglot opening book
        self.book = None
        if book_path and Path(book_path).exists():
            try:
                import chess.polyglot
                self.book = chess.polyglot.open_reader(str(book_path))
            except Exception as e:
                print(f"info string Failed to load opening book: {e}", file=sys.stderr)

        # MCTS search
        self.mcts_sims = mcts_sims
        self.mcts = None

        # C++ MCTS initialization
        self.cpp_mcts = None
        if self.use_cpp_mcts:
            try:
                from search.cpp import load_cpp_mcts
                MCTSCpp = load_cpp_mcts()
                self.cpp_mcts = MCTSCpp()
                if seed is not None:
                    self.cpp_mcts.set_seed(seed)
            except Exception as e:
                logger.warning(f"Failed to load C++ MCTS, falling back to Python MCTS: {e}")
                self.cpp_mcts = None

        if mcts_sims > 0 and self.cpp_mcts is None:
            mcts_cfg = MCTSConfig(
                simulations=mcts_sims,
                batch_size=mcts_batch,
                temperature=temperature,
            )
            self.mcts = MCTS(self.evaluate_batch, cfg=mcts_cfg, tablebase=self.tablebase, rng=self.np_rng)

    @torch.no_grad()
    def evaluate_batch(
        self, boards: list[chess.Board]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched evaluation returning probabilities in oriented perspective.

        Returns: (policy[N, 4096], promo[N, 4], wdl[N, 3])
        """
        xs = np.stack([encode(b) for b in boards])
        x = torch.from_numpy(xs).to(self.device)

        if self.autocast:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                p_logits, pr_logits, w_logits = self.model(x)
        else:
            p_logits, pr_logits, w_logits = self.model(x)

        p_probs = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
        pr_probs = torch.softmax(pr_logits.float(), dim=-1).cpu().numpy()
        w_probs = torch.softmax(w_logits.float(), dim=-1).cpu().numpy()
        return p_probs, pr_probs, w_probs

    @torch.no_grad()
    def evaluate_tensor(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluator callback for C++ MCTS receiving torch.Tensor (B, 19, 8, 8).

        Returns: (policy_probs[B, 4096], promo_probs[B, 4], wdl[B, 3])
        """
        x = x.to(self.device)
        if self.autocast:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                p_logits, pr_logits, w_logits = self.model(x)
        else:
            p_logits, pr_logits, w_logits = self.model(x)

        p_probs = torch.softmax(p_logits.float(), dim=-1)
        pr_probs = torch.softmax(pr_logits.float(), dim=-1)
        w_probs = torch.softmax(w_logits.float(), dim=-1)
        return p_probs, pr_probs, w_probs

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single board evaluation returning (policy[4096], promo[4], wdl[3]) in oriented perspective."""
        p_probs, pr_probs, w_probs = self.evaluate_batch([board])
        return p_probs[0], pr_probs[0], w_probs[0]

    def _from_book(self, board: chess.Board) -> chess.Move | None:
        if self.book is None or board.fullmove_number * 2 > self.book_plies:
            return None
        try:
            entries = list(self.book.find_all(board))
        except Exception:
            return None
        if not entries:
            return None
        weights = [max(e.weight, 1) for e in entries]
        return self.rng.choices([e.move for e in entries], weights=weights)[0]

    def _from_tablebase(self, board: chess.Board) -> chess.Move | None:
        """Syzygy tablebase play when piece count <= 5."""
        if self.tablebase is None or chess.popcount(board.occupied) > 5:
            return None
        if not board.is_valid():
            return None

        best_move, best_key = None, None
        for mv in board.legal_moves:
            board.push(mv)
            try:
                if board.is_checkmate():
                    wdl, dtz = -2, 0
                elif board.is_stalemate() or board.is_insufficient_material():
                    wdl, dtz = 0, 0
                else:
                    wdl = self.tablebase.probe_wdl(board)
                    dtz = abs(self.tablebase.probe_dtz(board))
            except Exception as e:
                err_key = f"{type(e).__name__}:{e}"
                if err_key not in self._tb_missing:
                    self._tb_missing.add(err_key)
                    print(f"info string Syzygy probe error: {e}", file=sys.stderr)
                return None
            finally:
                board.pop()

            zeroing = 0 if board.is_zeroing(mv) else 1
            key = (wdl, zeroing, dtz)
            if best_key is None or key < best_key:
                best_move, best_key = mv, key

        return best_move

    def _from_network(self, board: chess.Board) -> chess.Move:
        """Select move directly from policy and promo heads."""
        policy, promo, _ = self.evaluate(board)
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves in position")

        scores = []
        turn = board.turn
        for mv in legal:
            om = orient_move(mv, turn)
            s = float(policy[move_to_index(om)])
            pi = move_to_promo_index(om)
            if pi is not None:
                s *= float(promo[pi])
            scores.append(s)

        scores_arr = np.asarray(scores, dtype=np.float64)
        if self.temperature <= 0:
            return legal[int(np.argmax(scores_arr))]

        p = scores_arr ** (1.0 / self.temperature)
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            return legal[int(np.argmax(scores_arr))]
        return legal[int(self.rng.choices(range(len(legal)), weights=(p / total))[0])]

    def _from_mcts(self, board: chess.Board) -> chess.Move | None:
        if self.mcts_sims <= 0:
            return None

        if self.cpp_mcts is not None:
            try:
                move_str, _ = self.cpp_mcts.search(
                    board.fen(),
                    self.evaluate_tensor,
                    simulations=self.mcts_sims,
                    batch_size=self.mcts_batch,
                    temperature=self.temperature,
                )
                if move_str:
                    mv = chess.Move.from_uci(move_str)
                    if mv in board.legal_moves:
                        return mv
            except Exception as e:
                logger.warning(f"C++ MCTS search failed ({e}), falling back to Python MCTS")

        # Python MCTS fallback
        if self.mcts is None:
            mcts_cfg = MCTSConfig(
                simulations=self.mcts_sims,
                batch_size=self.mcts_batch,
                temperature=self.temperature,
            )
            self.mcts = MCTS(self.evaluate_batch, cfg=mcts_cfg, tablebase=self.tablebase, rng=self.np_rng)
        mv, _ = self.mcts.best_move(
            board,
            simulations=self.mcts_sims,
            temperature=self.temperature,
        )
        return mv

    def play(self, board: chess.Board) -> chess.Move:
        """Select best move. Order: Syzygy -> Book -> MCTS -> Network direct."""
        for source in (self._from_tablebase, self._from_book, self._from_mcts):
            mv = source(board)
            if mv is not None and mv in board.legal_moves:
                return mv
        return self._from_network(board)
