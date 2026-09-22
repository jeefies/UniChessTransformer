#!/usr/bin/env python3
"""Head-to-head match between P4 self-play improved Model T and Model R (ResNet).

Uses UniChess Server engine loader:
  - T: Stratified Chess Transformer 20M (P4 selfplay checkpoint) (mcts_sims = 2400)
  - R: ResNet 15x192 (mcts_sims = 800)
"""
import argparse
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, '/home/jeefy/UniChess/Server')
from models import create_engine, _load_engine_class, resolve_kwargs

DEFAULT_T_CKPT = "/home/jeefy/UniChess/Transformer/runs/stratified_p4_selfplay/best_model.pt"
LOG_FILE = Path("/home/jeefy/UniChess/Transformer/logs/match_p4_T_vs_R.log")
PGN_FILE = Path("/home/jeefy/UniChess/Transformer/logs/match_p4_T_vs_R.pgn")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("MatchP4")

OPENINGS = [
    {
        "name": "Italian Game (Giuoco Piano)",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],
        "eco": "C50"
    },
    {
        "name": "Ruy Lopez (Morphy Defense)",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
        "eco": "C70"
    },
    {
        "name": "Sicilian Defense (Open Sicilian)",
        "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
        "eco": "B90"
    },
    {
        "name": "French Defense (Classical)",
        "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"],
        "eco": "C11"
    },
    {
        "name": "Queen's Gambit Declined (Orthodox)",
        "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"],
        "eco": "D60"
    }
]

def play_game(game_idx: int, opening: dict, white_model: str, black_model: str, eng_white, eng_black, t_sims: int, r_sims: int):
    board = chess.Board()
    for mv_uci in opening["moves"]:
        board.push_uci(mv_uci)
    
    start_fen = board.fen()
    eng_white.setup(start_fen)
    eng_black.setup(start_fen)
    
    w_sims = t_sims if white_model == 'T' else r_sims
    b_sims = t_sims if black_model == 'T' else r_sims

    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "UniChess P4 Selfplay T vs R Match"
    pgn_game.headers["Site"] = "Localhost (RTX 5070 Ti)"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_idx)
    pgn_game.headers["White"] = f"Model {white_model} (mcts {w_sims})"
    pgn_game.headers["Black"] = f"Model {black_model} (mcts {b_sims})"
    pgn_game.headers["FEN"] = start_fen
    pgn_game.headers["SetUp"] = "1"
    pgn_game.headers["Opening"] = opening["name"]
    pgn_game.headers["ECO"] = opening["eco"]

    node = pgn_game

    white_times = []
    black_times = []
    plies = 0
    max_plies = 300  # 150 full moves

    logger.info(f"=== Starting Game {game_idx}: White={white_model} (sims {w_sims}) vs Black={black_model} (sims {b_sims}) | Opening={opening['name']} ===")

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        turn = board.turn # True for White, False for Black
        current_eng = eng_white if turn == chess.WHITE else eng_black
        other_eng = eng_black if turn == chess.WHITE else eng_white
        current_name = white_model if turn == chess.WHITE else black_model

        t0 = time.time()
        res = current_eng.engine_move()
        dt = time.time() - t0

        move_uci = res.get("engine_move")
        if not move_uci:
            logger.warning(f"Engine {current_name} returned no move! Result: {res}")
            break

        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            logger.error(f"Engine {current_name} made illegal move: {move_uci}")
            break

        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)
        node.comment = f"[{current_name} {dt:.2f}s]"

        other_eng.human_move(move_uci)

        if turn == chess.WHITE:
            white_times.append(dt)
        else:
            black_times.append(dt)
        plies += 1

        if plies % 10 == 0 or board.is_game_over(claim_draw=True):
            logger.info(f"Game {game_idx} ply {plies}: {san} (took {dt:.2f}s) | Turn: {'W' if board.turn else 'B'}")

    # Determine game outcome
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner == chess.WHITE:
            result_str = "1-0"
            winner = white_model
            reason = outcome.termination.name
        elif outcome.winner == chess.BLACK:
            result_str = "0-1"
            winner = black_model
            reason = outcome.termination.name
        else:
            result_str = "1/2-1/2"
            winner = "Draw"
            reason = outcome.termination.name
    else:
        result_str = "1/2-1/2"
        winner = "Draw"
        reason = "Move limit reached (150 moves)"

    pgn_game.headers["Result"] = result_str
    pgn_game.headers["Termination"] = reason

    avg_w_time = sum(white_times) / len(white_times) if white_times else 0.0
    avg_b_time = sum(black_times) / len(black_times) if black_times else 0.0

    game_summary = {
        "round": game_idx,
        "opening": opening["name"],
        "white": white_model,
        "black": black_model,
        "result": result_str,
        "winner": winner,
        "reason": reason,
        "plies": plies,
        "moves": (plies + 1) // 2,
        "avg_w_time": avg_w_time,
        "avg_b_time": avg_b_time,
        "pgn": str(pgn_game)
    }

    logger.info(f"=== Game {game_idx} Finished: {result_str} ({winner}) via {reason} in {game_summary['moves']} moves ===")
    logger.info(f"Avg time per move: {white_model}(W)={avg_w_time:.2f}s, {black_model}(B)={avg_b_time:.2f}s")
    
    return game_summary

def main():
    parser = argparse.ArgumentParser(description="Model T (P4 Selfplay) vs Model R 10-game match")
    parser.add_argument("--t-ckpt", type=str, default=DEFAULT_T_CKPT)
    parser.add_argument("--t-sims", type=int, default=2400)
    parser.add_argument("--r-sims", type=int, default=800)
    parser.add_argument("--log-file", type=str, default=str(LOG_FILE))
    parser.add_argument("--pgn-file", type=str, default=str(PGN_FILE))
    args = parser.parse_args()

    log_path = Path(args.log_file)
    pgn_path = Path(args.pgn_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger.info(f"Initializing engines T (ckpt={args.t_ckpt}, sims={args.t_sims}) and R (sims={args.r_sims})...")
    
    t_kwargs = resolve_kwargs('T', 'max_mcts')
    t_kwargs['ckpt'] = args.t_ckpt
    t_kwargs['mcts_sims'] = args.t_sims
    engine_t_cls = _load_engine_class('T')
    eng_t = engine_t_cls(**t_kwargs)

    r_kwargs = resolve_kwargs('R', 'max_mcts')
    r_kwargs['mcts_sims'] = args.r_sims
    engine_r_cls = _load_engine_class('R')
    eng_r = engine_r_cls(**r_kwargs)

    logger.info(f"Engine T loaded: ckpt={eng_t.ckpt}, mcts_sims={eng_t.mcts_sims}")
    logger.info(f"Engine R loaded: ckpt={eng_r.ckpt}, mcts_sims={eng_r.mcts_sims}")

    # Total 10 games: 5 openings x 2 (White/Black reversed)
    match_pairs = []
    game_num = 1
    for op in OPENINGS:
        match_pairs.append((game_num, op, 'T', 'R'))
        game_num += 1
        match_pairs.append((game_num, op, 'R', 'T'))
        game_num += 1

    games_results = []
    
    with open(pgn_path, "w", encoding="utf-8") as f_pgn:
        pass # Truncate PGN file

    for g_idx, op, w_name, b_name in match_pairs:
        eng_w = eng_t if w_name == 'T' else eng_r
        eng_b = eng_t if b_name == 'T' else eng_r
        
        res = play_game(g_idx, op, w_name, b_name, eng_w, eng_b, args.t_sims, args.r_sims)
        games_results.append(res)
        
        with open(pgn_path, "a", encoding="utf-8") as f_pgn:
            f_pgn.write(res["pgn"] + "\n\n")

    # Aggregate stats
    t_wins = sum(1 for g in games_results if g["winner"] == 'T')
    r_wins = sum(1 for g in games_results if g["winner"] == 'R')
    draws = sum(1 for g in games_results if g["winner"] == 'Draw')
    total_games = len(games_results)

    t_score = t_wins + 0.5 * draws
    r_score = r_wins + 0.5 * draws

    t_times = []
    r_times = []
    for g in games_results:
        if g["white"] == 'T':
            t_times.append(g["avg_w_time"])
            r_times.append(g["avg_b_time"])
        else:
            r_times.append(g["avg_w_time"])
            t_times.append(g["avg_b_time"])

    avg_t_time = sum(t_times) / len(t_times) if t_times else 0.0
    avg_r_time = sum(r_times) / len(r_times) if r_times else 0.0

    logger.info("=" * 60)
    logger.info("MATCH SUMMARY: Model T (P4 Selfplay) vs Model R (10 Games)")
    logger.info("=" * 60)
    logger.info(f"Final Score: T {t_score} - {r_score} R  (T Win Rate: {t_score/total_games*100:.1f}%)")
    logger.info(f"T Wins: {t_wins}, R Wins: {r_wins}, Draws: {draws}")
    logger.info(f"Average Thinking Time: T={avg_t_time:.2f}s/move, R={avg_r_time:.2f}s/move")
    logger.info("=" * 60)
    for g in games_results:
        logger.info(f"Game {g['round']:2d} | W: {g['white']} vs B: {g['black']} | {g['result']:7s} | Winner: {g['winner']:5s} | Reason: {g['reason']:25s} | Plies: {g['plies']:3d} | Opening: {g['opening']}")

if __name__ == "__main__":
    main()
