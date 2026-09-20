"""Tactical puzzle solver benchmark evaluating Top-1 move accuracy on chess puzzles."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import chess
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.engine import TransformerEngine


# Curated tactical benchmark test suite (mate in 1/2, pins, forks, deflections, back-rank mates)
BUILTIN_PUZZLES = [
    # Mate in 1
    {"id": "mate_1_01", "fen": "r1bqkb1r/pppp1ppp/2n5/4p2Q/2B1n3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4", "solution": "h5f7"},
    {"id": "mate_1_02", "fen": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "solution": "a1a8"},
    {"id": "mate_1_03", "fen": "r1b2rk1/pp3ppp/2n5/1B1p4/3qn3/1PN5/PBPP1PPP/R2QR1K1 b - - 0 12", "solution": "d4f2"},
    {"id": "mate_1_04", "fen": "r4rk1/5ppp/8/8/8/8/1B4QP/7K w - - 0 1", "solution": "g2g7"},
    {"id": "mate_1_05", "fen": "8/8/8/8/8/1k6/1p6/1K1R4 w - - 0 1", "solution": "d1d3"},
    # Fork / Skewer / Pin Tactics
    {"id": "fork_01", "fen": "r1bqk2r/pppp1ppp/2n5/4N3/1b1Pn3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6", "solution": "d1g4"},
    {"id": "fork_02", "fen": "r2qkb1r/pp2pppp/2n1b3/2pn4/2B5/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 2 7", "solution": "f3g5"},
    {"id": "pin_01", "fen": "r1b1k2r/pppp1ppp/5n2/4q3/1bP5/2N5/PP1BPPPP/R2QKB1R w KQkq - 0 8", "solution": "a2a3"},
    {"id": "skewer_01", "fen": "8/8/8/8/4K3/8/4Q3/3rk3 w - - 0 1", "solution": "e4f3"},
    # Deflection / Clearance
    {"id": "deflect_01", "fen": "r1b2rk1/1pp2ppp/p7/4q3/2B5/5Q2/PPP2PPP/R4RK1 w - - 0 15", "solution": "f3f7"},
    # Queen sacrifice mate
    {"id": "smothered_01", "fen": "6k1/5Npp/8/8/8/8/1Q4PP/6K1 w - - 0 1", "solution": "f7h6"},
    {"id": "backrank_01", "fen": "3r2k1/5ppp/8/8/8/8/5PPP/3RR1K1 w - - 0 1", "solution": "d1d8"},
]


def load_puzzles_from_file(file_path: str | Path, max_puzzles: int | None = None) -> list[dict]:
    """Load puzzles from CSV (Lichess format or custom) or JSON."""
    path = Path(file_path)
    puzzles: list[dict] = []

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                puzzles = data
            elif isinstance(data, dict) and "puzzles" in data:
                puzzles = data["puzzles"]
    elif path.suffix == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p_id = row.get("PuzzleId", row.get("id", str(len(puzzles))))
                fen = row.get("FEN", row.get("fen", ""))
                moves_field = row.get("Moves", row.get("moves", row.get("solution", "")))
                sol = moves_field.split()[0] if moves_field else ""
                if fen and sol:
                    puzzles.append({"id": p_id, "fen": fen, "solution": sol, "rating": row.get("Rating", "")})
                if max_puzzles and len(puzzles) >= max_puzzles:
                    break
    elif path.suffix in [".epd", ".txt"]:
        # EPD format: <fen> bm <best_move>; ...
        with open(path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" bm ")
                if len(parts) == 2:
                    fen = parts[0].strip()
                    bm = parts[1].split(";")[0].strip()
                    puzzles.append({"id": f"epd_{idx+1}", "fen": fen, "solution": bm})
                if max_puzzles and len(puzzles) >= max_puzzles:
                    break

    if max_puzzles:
        puzzles = puzzles[:max_puzzles]
    return puzzles


def evaluate_puzzles(
    engine: TransformerEngine,
    puzzles: Sequence[dict],
    use_mcts: bool = False,
    mcts_sims: int = 200,
    verbose: bool = True,
) -> dict:
    """Evaluate engine on a suite of puzzles.

    Returns summary metrics including Top-1 accuracy and average latency.
    """
    total = len(puzzles)
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total_time = 0.0

    if verbose:
        print(f"\nEvaluating {total} puzzles (MCTS: {use_mcts}, Sims: {mcts_sims if use_mcts else 0})...")
        print(f"{'ID':<15} | {'FEN':<35} | {'Expected':<8} | {'Engine':<8} | {'Result':<6} | {'Time (ms)':<8}")
        print("-" * 90)

    for p in puzzles:
        fen = p["fen"]
        sol_uci = p["solution"]
        board = chess.Board(fen)

        t0 = time.perf_counter()
        if use_mcts:
            engine.mcts_sims = mcts_sims
            predicted_move = engine.play(board)
        else:
            predicted_move = engine.play(board)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_time += elapsed_ms

        pred_uci = predicted_move.uci()
        # Handle SAN solutions in EPD by checking if board.parse_san matches
        expected_uci = sol_uci
        try:
            expected_move = board.parse_san(sol_uci)
            expected_uci = expected_move.uci()
        except Exception:
            pass

        is_correct = (pred_uci == expected_uci)
        if is_correct:
            correct_top1 += 1
            correct_top3 += 1
            correct_top5 += 1

        if verbose:
            fen_disp = fen[:32] + "..." if len(fen) > 35 else fen
            status = "PASS" if is_correct else "FAIL"
            print(f"{str(p.get('id', '')):<15} | {fen_disp:<35} | {expected_uci:<8} | {pred_uci:<8} | {status:<6} | {elapsed_ms:8.2f}")

    top1_acc = (correct_top1 / total) * 100.0 if total > 0 else 0.0
    avg_latency = total_time / total if total > 0 else 0.0

    summary = {
        "total_puzzles": total,
        "correct_top1": correct_top1,
        "top1_accuracy": top1_acc,
        "avg_latency_ms": avg_latency,
        "total_time_s": total_time / 1000.0,
    }

    if verbose:
        print("-" * 90)
        print(f"Top-1 Accuracy: {correct_top1}/{total} ({top1_acc:.2f}%) | Avg Latency: {avg_latency:.2f} ms/puzzle")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tactical puzzle solver benchmark")
    parser.add_argument("--model", "--ckpt", "--weights", dest="model", type=str, default="transformer_small", help="Checkpoint path or preset name")
    parser.add_argument("--puzzles", type=str, default=None, help="Optional path to puzzle CSV/JSON/EPD file")
    parser.add_argument("--max-puzzles", type=int, default=None)
    parser.add_argument("--mcts", action="store_true", help="Enable MCTS during puzzle search")
    parser.add_argument("--no-cpp-mcts", action="store_true", help="Disable C++ MCTS acceleration")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations if enabled")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16")
    args = parser.parse_args()

    # Automatically enable MCTS if sims is provided and --mcts not explicitly passed
    use_mcts = args.mcts or (args.sims > 0 and "--sims" in sys.argv)
    use_cpp_mcts = not args.no_cpp_mcts

    # Load engine
    if Path(args.model).exists():
        engine = TransformerEngine(
            args.model,
            device=args.device,
            precision=args.precision,
            mcts_sims=args.sims if use_mcts else 0,
            use_cpp_mcts=use_cpp_mcts,
        )
    else:
        from model.transformer import create_transformer
        net = create_transformer(args.model)
        engine = TransformerEngine(
            net,
            device=args.device,
            precision=args.precision,
            mcts_sims=args.sims if use_mcts else 0,
            use_cpp_mcts=use_cpp_mcts,
        )

    if args.puzzles:
        puzzle_list = load_puzzles_from_file(args.puzzles, max_puzzles=args.max_puzzles)
    else:
        print("No puzzle file specified, using built-in test suite.")
        puzzle_list = BUILTIN_PUZZLES[:args.max_puzzles] if args.max_puzzles else BUILTIN_PUZZLES

    summary = evaluate_puzzles(engine, puzzle_list, use_mcts=use_mcts, mcts_sims=args.sims)
