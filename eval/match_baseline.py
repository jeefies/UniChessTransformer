"""Match runner between UniChessTransformer and baseline chess_ai (play_v4.py) with diverse openings and statistical evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import List, Tuple

import chess
import chess.pgn
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = PROJECT_ROOT / "baseline" / "chess_ai"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from unichess_t.engine.engine import TransformerEngine
from model_cnn import ChessCNN
from search_engine import policy_order
from neural_search_v4 import NeuralSearchV4
from neural_inference_v3 import TracedEvaluator
from model_residual_value import ResidualValueModel

# Diverse standard chess openings (ECO curated lines, 4-6 plies)
OPENING_SUITE: List[Tuple[str, List[str]]] = [
    ("Italian Game (Giuoco Piano)", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]),
    ("Ruy Lopez (Morphy Defense)", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]),
    ("Scotch Game", ["e4", "e5", "Nf3", "Nc6", "d4", "exd4"]),
    ("Four Knights Game", ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6"]),
    ("Petroff Defense", ["e4", "e5", "Nf3", "Nf6", "Nxe5", "d6"]),
    ("Sicilian Defense (Open / Najdorf setup)", ["e4", "c5", "Nf3", "d6", "d4", "cxd4"]),
    ("Sicilian Defense (Classical / Dragon setup)", ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4"]),
    ("Sicilian Defense (Alapin Variation)", ["e4", "c5", "c3", "d5", "exd5", "Qxd5"]),
    ("French Defense (Classical)", ["e4", "e6", "d4", "d5", "Nc3", "Nf6"]),
    ("French Defense (Advance Variation)", ["e4", "e6", "d4", "d5", "e5", "c5"]),
    ("Caro-Kann Defense (Classical)", ["e4", "c6", "d4", "d5", "Nc3", "dxe4"]),
    ("Caro-Kann Defense (Advance Variation)", ["e4", "c6", "d4", "d5", "e5", "Bf5"]),
    ("Scandinavian Defense", ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5"]),
    ("Pirc Defense", ["e4", "d6", "d4", "Nf6", "Nc3", "g6"]),
    ("Queen's Gambit Declined", ["d4", "d5", "c4", "e6", "Nc3", "Nf6"]),
    ("Queen's Gambit Accepted", ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6"]),
    ("Slav Defense", ["d4", "d5", "c4", "c6", "Nf3", "Nf6"]),
    ("King's Indian Defense", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7"]),
    ("Nimzo-Indian Defense", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"]),
    ("Queen's Indian Defense", ["d4", "Nf6", "c4", "e6", "Nf3", "b6"]),
    ("Grünfeld Defense", ["d4", "Nf6", "c4", "g6", "Nc3", "d5"]),
    ("English Opening (Four Knights)", ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6"]),
    ("English Opening (Symmetrical)", ["c4", "c5", "Nc3", "Nc6", "g3", "g6"]),
    ("Réti Opening", ["Nf3", "d5", "c4", "c6", "g3", "Nf6"]),
    ("Dutch Defense", ["d4", "f5", "c4", "Nf6", "g3", "e6"]),
]


class BaselinePlayer:
    """Wrapper around chess_ai v2.0.0 (play_v4 engine)."""

    def __init__(self, seconds: float = 0.25, depth: int = 4, device: str = "cpu"):
        self.device = torch.device(device)
        self.seconds = seconds
        self.depth = depth

        self.policy = ChessCNN().to(self.device)
        self.policy.load_state_dict(
            torch.load(BASELINE_ROOT / "chess_model_balanced.pt", map_location=self.device, weights_only=True)
        )
        self.policy.eval()

        candidates = sorted((BASELINE_ROOT / "value_full_runs").glob("*/value_full_epoch2.pt"))
        if not candidates:
            raise FileNotFoundError("Baseline value model not found in value_full_runs")
        self.value_model_path = candidates[-1]

        # Note: TracedEvaluator / FastEvaluator requires CPU model for torch jit optimization
        self.value_model = ResidualValueModel().to("cpu")
        self.value_model.load_state_dict(
            torch.load(self.value_model_path, map_location="cpu", weights_only=True)
        )
        self.value_model.eval()

        self.evaluator = TracedEvaluator(self.value_model)
        self.evaluator.validate([chess.Board()])
        self.engine = NeuralSearchV4(self.evaluator, self.seconds, self.depth, claim_draw=True)

    @torch.no_grad()
    def play(self, board: chess.Board) -> tuple[chess.Move, float, dict]:
        t0 = time.perf_counter()
        preferred = policy_order(board, self.policy, self.device)
        suggested, is_mate, stats = self.engine.choose(board, preferred)
        dt = time.perf_counter() - t0

        if suggested is None or suggested not in board.legal_moves:
            suggested = preferred[0] if preferred else next(iter(board.legal_moves))
        return suggested, dt, {"is_mate": is_mate, "stats": stats}


class UniChessPlayer:
    """Wrapper around UniChessTransformer engine."""

    def __init__(self, ckpt_path: str | Path, mcts_sims: int = 100, device: str = "cuda"):
        self.engine = TransformerEngine(
            ckpt_path,
            device=device,
            mcts_sims=mcts_sims,
            temperature=0.0,
        )
        self.mcts_sims = mcts_sims

    def play(self, board: chess.Board) -> tuple[chess.Move, float, dict]:
        t0 = time.perf_counter()
        mv = self.engine.play(board)
        dt = time.perf_counter() - t0
        _, _, wdl = self.engine.evaluate(board)
        ev = float(wdl[0] - wdl[2])
        return mv, dt, {"wdl": wdl.tolist(), "ev": ev}


def calculate_elo(score: float, total_games: int) -> tuple[float, float]:
    """Compute relative Elo difference and 95% confidence interval."""
    if total_games == 0:
        return 0.0, 0.0
    p = score / total_games
    # Bound p away from strict 0 and 1 to prevent infinity
    p_bounded = min(max(p, 1e-4), 1.0 - 1e-4)
    delta_elo = -400.0 * math.log10(1.0 / p_bounded - 1.0)
    # Standard error of the sample proportion
    se_p = math.sqrt(p_bounded * (1.0 - p_bounded) / total_games)
    # Derivative d(Elo)/dp = 400 / (ln(10) * p * (1 - p))
    d_elo_dp = 400.0 / (math.log(10) * p_bounded * (1.0 - p_bounded))
    error_95 = 1.96 * se_p * d_elo_dp
    return delta_elo, error_95


def play_single_game(
    white_player,
    black_player,
    white_name: str,
    black_name: str,
    game_idx: int,
    opening_name: str = "Standard Start",
    opening_moves: list[str] | None = None,
    max_plies: int = 160,
) -> tuple[chess.pgn.Game, dict]:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"UniChess vs chess_ai Match"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_idx)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Opening"] = opening_name

    node = game
    white_times = []
    black_times = []
    move_details = []

    # Apply opening moves if present
    ply = 0
    if opening_moves:
        for san in opening_moves:
            ply += 1
            mv = board.parse_san(san)
            board.push(mv)
            node = node.add_variation(mv)
            node.comment = "opening book"

    termination_reason = "Normal"

    while not board.is_game_over() and ply < max_plies:
        ply += 1
        is_white = board.turn == chess.WHITE
        current_player = white_player if is_white else black_player

        mv, move_time, info = current_player.play(board)
        san = board.san(mv)

        if is_white:
            white_times.append(move_time)
        else:
            black_times.append(move_time)

        move_details.append({
            "ply": ply,
            "turn": "W" if is_white else "B",
            "move": mv.uci(),
            "san": san,
            "time": round(move_time, 3),
            "info": info,
        })

        board.push(mv)
        node = node.add_variation(mv)
        node.comment = f"time: {move_time:.2f}s"

        if board.is_checkmate():
            termination_reason = "checkmate"
            break
        if board.is_stalemate():
            termination_reason = "stalemate"
            break
        if board.is_insufficient_material():
            termination_reason = "insufficient_material"
            break
        if board.can_claim_threefold_repetition():
            termination_reason = "threefold_repetition"
            break
        if board.can_claim_fifty_moves():
            termination_reason = "fifty_move_rule"
            break

    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
        termination = f"Checkmate ({termination_reason})"
    elif board.is_stalemate():
        result = "1/2-1/2"
        termination = f"Draw by stalemate ({termination_reason})"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        termination = f"Draw by insufficient material ({termination_reason})"
    elif board.can_claim_threefold_repetition():
        result = "1/2-1/2"
        termination = "Draw by threefold repetition"
    elif board.can_claim_fifty_moves():
        result = "1/2-1/2"
        termination = "Draw by 50-move rule"
    elif ply >= max_plies:
        result = "1/2-1/2"
        termination = "Draw by adjudication (max plies reached)"
    else:
        result = board.result(claim_draw=True)
        termination = "Normal termination"

    game.headers["Result"] = result
    game.headers["Termination"] = termination

    stats = {
        "game_idx": game_idx,
        "opening": opening_name,
        "opening_moves": opening_moves if opening_moves else [],
        "white": white_name,
        "black": black_name,
        "result": result,
        "termination": termination,
        "plies": ply,
        "moves": (ply + 1) // 2,
        "white_avg_time": sum(white_times) / len(white_times) if white_times else 0,
        "black_avg_time": sum(black_times) / len(black_times) if black_times else 0,
        "move_details": move_details,
    }
    return game, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/transformer_20m/best_model.pt", help="Path to UniChess checkpoint")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--unichess-sims", "--sims", type=int, default=100, help="UniChess MCTS sims")
    parser.add_argument("--baseline-seconds", type=float, default=0.25, help="Baseline search budget in seconds")
    parser.add_argument("--baseline-depth", type=int, default=4, help="Baseline search depth")
    parser.add_argument("--baseline-device", type=str, default="cpu", help="Device for baseline model")
    parser.add_argument("--output-log", type=str, default="logs/match_100g.log")
    parser.add_argument("--output-pgn", type=str, default="logs/match_100g.pgn")
    parser.add_argument("--output-json", type=str, default="logs/match_100g.json")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.output_log, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("MatchBaseline")

    logger.info("Initializing engines for 100-game match...")
    unichess = UniChessPlayer(args.ckpt, mcts_sims=args.unichess_sims)
    baseline = BaselinePlayer(seconds=args.baseline_seconds, depth=args.baseline_depth, device=args.baseline_device)

    unichess_name = f"UniChessTransformer_20m (MCTS {args.unichess_sims})"
    baseline_name = f"chess_ai_v2.0.0 (CNN+ResVal depth={args.baseline_depth} {args.baseline_seconds}s)"

    logger.info(f"Engine 1: {unichess_name}")
    logger.info(f"Engine 2: {baseline_name}")
    logger.info(f"Total games scheduled: {args.games} across {len(OPENING_SUITE)} diverse opening lines")

    all_games = []
    all_stats = []
    u_score = 0.0
    b_score = 0.0

    u_wins = 0
    u_draws = 0
    u_losses = 0

    plies_list = []
    termination_counts = {}

    pgn_file = open(args.output_pgn, "w", encoding="utf-8")
    start_time_all = time.time()

    for g in range(1, args.games + 1):
        # Assign opening from the opening suite in pairs (each opening played as White & Black)
        opening_pair_idx = ((g - 1) // 2) % len(OPENING_SUITE)
        opening_name, opening_moves = OPENING_SUITE[opening_pair_idx]

        if g % 2 == 1:
            w_player, b_player = unichess, baseline
            w_name, b_name = unichess_name, baseline_name
            u_is_white = True
        else:
            w_player, b_player = baseline, unichess
            w_name, b_name = baseline_name, unichess_name
            u_is_white = False

        logger.info(f"\n--- Game {g}/{args.games}: [{opening_name}] W: {w_name} vs B: {b_name} ---")
        game, stats = play_single_game(
            w_player, b_player, w_name, b_name, g,
            opening_name=opening_name, opening_moves=opening_moves
        )
        all_games.append(game)
        all_stats.append(stats)

        # Write PGN
        pgn_file.write(str(game) + "\n\n")
        pgn_file.flush()

        res = stats["result"]
        term = stats["termination"]
        termination_counts[term] = termination_counts.get(term, 0) + 1
        plies_list.append(stats["plies"])

        if res == "1-0":
            if u_is_white:
                u_score += 1.0
                u_wins += 1
            else:
                b_score += 1.0
                u_losses += 1
        elif res == "0-1":
            if u_is_white:
                b_score += 1.0
                u_losses += 1
            else:
                u_score += 1.0
                u_wins += 1
        else:
            u_score += 0.5
            b_score += 0.5
            u_draws += 1

        delta_elo, elo_err = calculate_elo(u_score, g)
        elapsed = time.time() - start_time_all
        avg_game_time = elapsed / g
        remaining_time = avg_game_time * (args.games - g)

        logger.info(
            f"Game {g} finished: Result={res} ({term}), Moves={stats['moves']}, Plies={stats['plies']}, "
            f"W_time={stats['white_avg_time']:.2f}s, B_time={stats['black_avg_time']:.2f}s"
        )
        logger.info(
            f"Score: UniChess {u_score:.1f} - {b_score:.1f} chess_ai "
            f"({u_wins}W / {u_draws}D / {u_losses}L, WinRate={u_score/g*100:.1f}%, Elo={delta_elo:+.0f}±{elo_err:.0f}) "
            f"ETA: {remaining_time/60:.1f}m"
        )

    pgn_file.close()
    total_time = time.time() - start_time_all

    final_elo, final_elo_err = calculate_elo(u_score, args.games)
    avg_plies = sum(plies_list) / len(plies_list) if plies_list else 0
    median_plies = sorted(plies_list)[len(plies_list) // 2] if plies_list else 0

    logger.info("\n================ 100-GAME MATCH COMPLETED ================")
    logger.info(f"Total Duration: {total_time/60:.2f} minutes")
    logger.info(f"Final Score: UniChess {u_score:.1f} - {b_score:.1f} chess_ai ({args.games} games)")
    logger.info(f"Record: {u_wins} Wins, {u_draws} Draws, {u_losses} Losses")
    logger.info(f"Score Percentage: {u_score / args.games * 100:.2f}%")
    logger.info(f"Relative Elo: {final_elo:+.1f} ± {final_elo_err:.1f} (95% CI)")
    logger.info(f"Plies: Mean={avg_plies:.1f}, Median={median_plies}, Min={min(plies_list)}, Max={max(plies_list)}")
    logger.info("Terminations:")
    for term, count in sorted(termination_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  - {term}: {count} ({count / args.games * 100:.1f}%)")

    summary_json = {
        "event": "100-Game Evaluation Match",
        "unichess": unichess_name,
        "baseline": baseline_name,
        "total_games": args.games,
        "total_time_seconds": round(total_time, 2),
        "score": {
            "unichess_points": u_score,
            "baseline_points": b_score,
            "unichess_wins": u_wins,
            "unichess_draws": u_draws,
            "unichess_losses": u_losses,
            "score_pct": round(u_score / args.games * 100, 2),
            "delta_elo": round(final_elo, 1),
            "elo_error_95": round(final_elo_err, 1),
        },
        "plies_distribution": {
            "mean": round(avg_plies, 1),
            "median": median_plies,
            "min": min(plies_list),
            "max": max(plies_list),
        },
        "terminations": termination_counts,
        "games": all_stats,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    logger.info(f"Results recorded in {args.output_log}, {args.output_pgn}, and {args.output_json}")


if __name__ == "__main__":
    main()
