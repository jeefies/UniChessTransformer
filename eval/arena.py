"""Match runner between two chess engines with paired openings, Elo calculation, and SPRT."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import sys
import time

import chess
import chess.engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def score_to_elo(score: float) -> float:
    """Win score ratio -> Elo difference."""
    score = min(max(score, 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


ELO_INF = float("inf")


def elo_with_error(w: int, d: int, l: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Calculate (Elo diff, lower_bound, upper_bound) from W/D/L record."""
    n = w + d + l
    if n == 0:
        return 0.0, -ELO_INF, ELO_INF
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    se = math.sqrt(var / n)
    z = 1.959963985 if abs(confidence - 0.95) < 1e-6 else 2.575829
    lo_s, hi_s = s - z * se, s + z * se
    elo = ELO_INF if s >= 1.0 else (-ELO_INF if s <= 0.0 else score_to_elo(s))
    lo = -ELO_INF if lo_s <= 0.0 else score_to_elo(lo_s)
    hi = ELO_INF if hi_s >= 1.0 else score_to_elo(hi_s)
    return elo, lo, hi


def sprt_llr(w: int, d: int, l: int, elo0: float, elo1: float) -> float:
    """Log-likelihood ratio for Sequential Probability Ratio Test."""
    n = w + d + l
    if n == 0:
        return 0.0
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    if var <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (s1 - s0) * (2 * s - s0 - s1) / (2 * var)


def sprt_verdict(llr: float, alpha: float = 0.05, beta: float = 0.05) -> str:
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    if llr >= upper:
        return "Accept H1 (Engine A is stronger)"
    if llr <= lower:
        return "Accept H0 (No significant improvement)"
    return "Continue (Inconclusive)"


@dataclass
class EngineSpec:
    cmd: list[str]
    name: str
    options: dict = field(default_factory=dict)
    nodes: int | None = None
    movetime: float | None = 0.1

    def limit(self) -> chess.engine.Limit:
        if self.nodes:
            return chess.engine.Limit(nodes=self.nodes)
        return chess.engine.Limit(time=self.movetime or 0.1)


def play_game(
    engine_white: chess.engine.SimpleEngine,
    spec_white: EngineSpec,
    engine_black: chess.engine.SimpleEngine,
    spec_black: EngineSpec,
    start_fen: str | None = None,
    max_plies: int = 200,
) -> tuple[str, list[str]]:
    """Play a single game between two SimpleEngine instances.

    Returns: (result: '1-0', '0-1', '1/2-1/2', notes)
    """
    board = chess.Board(start_fen) if start_fen else chess.Board()
    notes: list[str] = []
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        is_white = (board.turn == chess.WHITE)
        eng, spec = (engine_white, spec_white) if is_white else (engine_black, spec_black)

        try:
            res = eng.play(board, spec.limit())
        except Exception as e:
            notes.append(f"{spec.name} crashed with {type(e).__name__}: {e}")
            return ("0-1" if is_white else "1-0"), notes

        if res is None or res.move is None or res.move not in board.legal_moves:
            notes.append(f"{spec.name} played illegal move in {board.fen()}")
            return ("0-1" if is_white else "1-0"), notes

        board.push(res.move)
        plies += 1

    if plies >= max_plies and not board.is_game_over():
        return "1/2-1/2", notes

    r = board.result(claim_draw=True)
    if r == "*":
        r = "1/2-1/2"
    return r, notes


def play_match(
    spec_a: EngineSpec,
    spec_b: EngineSpec,
    openings: list[str],
    max_plies: int = 200,
    elo0: float = 0.0,
    elo1: float = 35.0,
) -> dict:
    """Run paired matches across given openings and report stats."""
    w = d = l = 0
    all_notes = []

    print(f"Starting match: {spec_a.name} vs {spec_b.name} ({len(openings)*2} games total)")

    eng_a = chess.engine.SimpleEngine.popen_uci(spec_a.cmd)
    eng_b = chess.engine.SimpleEngine.popen_uci(spec_b.cmd)

    try:
        if spec_a.options:
            eng_a.configure(spec_a.options)
        if spec_b.options:
            eng_b.configure(spec_b.options)

        for pair_idx, fen in enumerate(openings):
            # Game 1: A as White, B as Black
            res1, notes1 = play_game(eng_a, spec_a, eng_b, spec_b, start_fen=fen, max_plies=max_plies)
            all_notes.extend(notes1)
            if res1 == "1-0":
                w += 1
            elif res1 == "0-1":
                l += 1
            else:
                d += 1

            # Game 2: B as White, A as Black
            res2, notes2 = play_game(eng_b, spec_b, eng_a, spec_a, start_fen=fen, max_plies=max_plies)
            all_notes.extend(notes2)
            if res2 == "0-1":
                w += 1
            elif res2 == "1-0":
                l += 1
            else:
                d += 1

            games_done = (pair_idx + 1) * 2
            elo, elo_lo, elo_hi = elo_with_error(w, d, l)
            llr = sprt_llr(w, d, l, elo0, elo1)
            verdict = sprt_verdict(llr)

            if (pair_idx + 1) % 5 == 0 or (pair_idx + 1) == len(openings):
                print(
                    f"Games {games_done:3d}/{len(openings)*2:3d} | W: {w:3d} D: {d:3d} L: {l:3d} | "
                    f"Score: {(w + 0.5*d)/games_done*100:5.1f}% | Elo: {elo:+6.1f} [{elo_lo:+6.1f}, {elo_hi:+6.1f}] | "
                    f"LLR: {llr:+5.2f} ({verdict})"
                )
    finally:
        eng_a.quit()
        eng_b.quit()

    games_total = w + d + l
    final_elo, elo_lo, elo_hi = elo_with_error(w, d, l)
    return {
        "engine_a": spec_a.name,
        "engine_b": spec_b.name,
        "games": games_total,
        "wins": w,
        "draws": d,
        "losses": l,
        "score": (w + 0.5 * d) / max(1, games_total),
        "elo": final_elo,
        "elo_lower": elo_lo,
        "elo_upper": elo_hi,
        "notes": all_notes,
    }


def default_openings() -> list[str]:
    """Diverse set of standard balanced opening positions."""
    return [
        chess.STARTING_FEN,
        # e4 e5 Nf3 Nc6
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        # French: e4 e6 d4 d5
        "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
        # Sicilian: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3
        "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6",
        # Caro-Kann: e4 c6 d4 d5
        "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
        # d4 d5 c4 e6
        "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
        # Indian: d4 Nf6 c4 e6
        "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
        # King's Indian: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6
        "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP2BPPP/R1BQK1NR w KQ - 1 6",
        # English: c4 e5
        "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
        # Reti: Nf3 d5 g3
        "rnbqkbnr/ppp1pppp/8/3p4/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq - 0 2",
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run match between two UCI engines")
    parser.add_argument("--cmd-a", nargs="+", required=True, help="Command for Engine A")
    parser.add_argument("--cmd-b", nargs="+", required=True, help="Command for Engine B")
    parser.add_argument("--name-a", type=str, default="Engine-A")
    parser.add_argument("--name-b", type=str, default="Engine-B")
    parser.add_argument("--movetime", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=10, help="Number of opening pairs")
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    spec_a = EngineSpec(cmd=args.cmd_a, name=args.name_a, movetime=args.movetime)
    spec_b = EngineSpec(cmd=args.cmd_b, name=args.name_b, movetime=args.movetime)

    openings = default_openings()
    while len(openings) < args.rounds:
        openings.extend(default_openings())
    openings = openings[:args.rounds]

    summary = play_match(spec_a, spec_b, openings, max_plies=args.max_plies)
    print("\n--- Match Completed ---")
    print(
        f"Result: {summary['wins']}W / {summary['draws']}D / {summary['losses']}L | "
        f"Score: {summary['score']*100:.1f}% | Elo diff: {summary['elo']:+.1f}"
    )
