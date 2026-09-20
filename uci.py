"""Full Universal Chess Interface (UCI) protocol driver for cutechess-cli, GUIs, and engines."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

import chess
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.engine import TransformerEngine
from model.transformer import create_transformer

ENGINE_NAME = "UniChessTransformer"
ENGINE_AUTHOR = "jeefy"


class UCILoop:
    """UCI protocol handler implementing standard chess engine communication."""

    def __init__(self, args):
        self.args = args
        self.engine: TransformerEngine | None = None
        self.board = chess.Board()
        self.stop_requested = threading.Event()
        self.search_thread: threading.Thread | None = None

        # Configurable options
        self.temperature = getattr(args, "temperature", 0.0)
        self.mcts_sims = getattr(args, "mcts_sims", 0)
        self.mcts_batch = getattr(args, "mcts_batch", 128)
        self.syzygy_path = getattr(args, "syzygy", None)
        self.book_path = getattr(args, "book", None)
        self.precision = getattr(args, "precision", "fp16")
        self.device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_engine(self) -> TransformerEngine:
        if self.engine is None:
            if self.args.ckpt and Path(self.args.ckpt).exists():
                self.engine = TransformerEngine(
                    self.args.ckpt,
                    device=self.device,
                    precision=self.precision,
                    syzygy_path=self.syzygy_path,
                    book_path=self.book_path,
                    temperature=self.temperature,
                    mcts_sims=self.mcts_sims,
                    mcts_batch=self.mcts_batch,
                )
            else:
                # Default to model preset if no checkpoint specified
                preset = getattr(self.args, "preset", "transformer_small")
                net = create_transformer(preset)
                self.engine = TransformerEngine(
                    net,
                    device=self.device,
                    precision=self.precision,
                    syzygy_path=self.syzygy_path,
                    book_path=self.book_path,
                    temperature=self.temperature,
                    mcts_sims=self.mcts_sims,
                    mcts_batch=self.mcts_batch,
                )
        return self.engine

    def handle_position(self, parts: list[str]) -> None:
        """Parse position command: position [startpos | fen <fen>] [moves <m1> <m2> ...]"""
        if len(parts) < 2:
            return

        if parts[1] == "startpos":
            self.board = chess.Board()
            moves_idx = parts.index("moves") if "moves" in parts else None
            moves = parts[moves_idx + 1:] if moves_idx is not None else []
        elif parts[1] == "fen":
            moves_idx = parts.index("moves") if "moves" in parts else len(parts)
            fen_str = " ".join(parts[2:moves_idx])
            try:
                self.board = chess.Board(fen_str)
            except ValueError:
                self.board = chess.Board()
            moves = parts[moves_idx + 1:] if moves_idx < len(parts) else []
        else:
            return

        for m in moves:
            try:
                self.board.push_uci(m)
            except ValueError:
                print(f"info string Ignoring illegal move: {m}", flush=True)

    def _search_and_respond(self, board_copy: chess.Board, go_parts: list[str]) -> None:
        eng = self._ensure_engine()
        if board_copy.is_game_over(claim_draw=True):
            print("bestmove 0000", flush=True)
            return

        t0 = time.perf_counter()
        _, _, wdl = eng.evaluate(board_copy)

        # Centipawn approximation from WDL probabilities: 100 * (P(Win) - P(Loss)) * 3
        q_val = float(wdl[0] - wdl[2])
        score_cp = int(round(300.0 * q_val))
        depth = self.mcts_sims if self.mcts_sims > 0 else 1

        w_int = int(wdl[0] * 1000)
        d_int = int(wdl[1] * 1000)
        l_int = int(wdl[2] * 1000)
        print(
            f"info depth {depth} score cp {score_cp} wdl {w_int} {d_int} {l_int}",
            flush=True,
        )

        best_move = eng.play(board_copy)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(f"info time {elapsed_ms}", flush=True)
        print(f"bestmove {best_move.uci()}", flush=True)

    def handle_go(self, parts: list[str]) -> None:
        """Start move computation."""
        board_copy = self.board.copy(stack=True)
        self.stop_requested.clear()
        self._search_and_respond(board_copy, parts)

    def handle_setoption(self, parts: list[str]) -> None:
        """Handle option setting."""
        try:
            name_idx = parts.index("name") + 1
            value_idx = parts.index("value") + 1
            opt_name = parts[name_idx].lower()
            opt_val = " ".join(parts[value_idx:])

            if opt_name == "temperature":
                self.temperature = float(opt_val)
                if self.engine is not None:
                    self.engine.temperature = self.temperature
            elif opt_name == "mctssimulations":
                self.mcts_sims = int(opt_val)
                if self.engine is not None:
                    self.engine.mcts_sims = self.mcts_sims
            elif opt_name == "syzygypath":
                self.syzygy_path = opt_val
            elif opt_name == "bookpath":
                self.book_path = opt_val
        except (ValueError, IndexError):
            pass

    def run(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "uci":
                print(f"id name {ENGINE_NAME}")
                print(f"id author {ENGINE_AUTHOR}")
                print("option name Temperature type string default 0.0")
                print("option name MCTSSimulations type spin default 0 min 0 max 100000")
                print("option name SyzygyPath type string default <empty>")
                print("option name BookPath type string default <empty>")
                print("uciok", flush=True)
            elif cmd == "isready":
                self._ensure_engine()
                print("readyok", flush=True)
            elif cmd == "ucinewgame":
                self.board = chess.Board()
            elif cmd == "position":
                self.handle_position(parts)
            elif cmd == "go":
                self.handle_go(parts)
            elif cmd == "stop":
                self.stop_requested.set()
            elif cmd == "setoption":
                self.handle_setoption(parts)
            elif cmd == "quit":
                break


def main():
    parser = argparse.ArgumentParser(description="UniChessTransformer UCI Engine Interface")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--preset", type=str, default="transformer_small", help="Model preset if no ckpt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--syzygy", type=str, default=None, help="Syzygy tablebase directory")
    parser.add_argument("--book", type=str, default=None, help="Polyglot opening book")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mcts-sims", type=int, default=0)
    parser.add_argument("--mcts-batch", type=int, default=128)
    args = parser.parse_args()

    loop = UCILoop(args)
    loop.run()


if __name__ == "__main__":
    main()
