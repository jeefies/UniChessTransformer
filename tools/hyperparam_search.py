"""MCTS Hyperparameter Simulation Search.

Explores combinations of:
- cpuct: [1.0, 1.5, 2.0, 2.5]
- dirichlet_alpha: [0.15, 0.25, 0.35]
- dirichlet_eps: [0.15, 0.25]
- virtual_loss: [1, 2, 3]
- batch_size: [32, 64, 128, 256]
- cpu_workers: [4, 8, 16, 32]

Evaluates self-play puzzle/position speed and move quality.
Logs results to logs/hyperparam_search.json and prints the top parameter configurations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import numpy as np
import torch

from eval.puzzle_bench import BUILTIN_PUZZLES
from unichess_t.search.mcts import MCTSConfig
from unichess_t.search.parallel_mcts import ParallelMCTS

# Candidate values specified in prompt
GRID = {
    "cpuct": [1.0, 1.5, 2.0, 2.5],
    "dirichlet_alpha": [0.15, 0.25, 0.35],
    "dirichlet_eps": [0.15, 0.25],
    "virtual_loss": [1, 2, 3],
    "batch_size": [32, 64, 128, 256],
    "cpu_workers": [4, 8, 16, 32],
}


def evaluate_config(
    params: dict,
    puzzles: list[dict],
    model_preset: str = "transformer_tiny",
    simulations: int = 150,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Evaluate a single hyperparameter configuration on puzzle positions."""
    cfg = MCTSConfig(
        c_puct=float(params["cpuct"]),
        c_puct_init=float(params["cpuct"]),
        dirichlet_alpha=float(params["dirichlet_alpha"]),
        dirichlet_eps=float(params["dirichlet_eps"]),
        virtual_loss=float(params["virtual_loss"]),
        batch_size=int(params["batch_size"]),
    )

    num_workers = int(params["cpu_workers"])
    pmcts = ParallelMCTS(
        num_workers=num_workers,
        batch_size=int(params["batch_size"]),
        model_preset=model_preset,
        device=device,
        cfg=cfg,
    )

    correct = 0
    total_time = 0.0
    total_sims = 0

    try:
        for pz in puzzles:
            board = chess.Board(pz["fen"])
            target_uci = pz["solution"][0]

            t0 = time.perf_counter()
            best_move, visits = pmcts.search(board, simulations=simulations, cfg=cfg)
            dt = time.perf_counter() - t0

            total_time += dt
            sims_done = sum(visits.values())
            total_sims += max(sims_done, simulations)

            if best_move.uci() == target_uci:
                correct += 1
    finally:
        pmcts.close()

    accuracy = correct / len(puzzles) if puzzles else 0.0
    sims_per_sec = total_sims / total_time if total_time > 0 else 0.0

    # Composite score balancing quality (accuracy) and throughput
    score = accuracy * 1000.0 + sims_per_sec / 10.0

    return {
        "params": params,
        "accuracy": accuracy,
        "correct": correct,
        "total_puzzles": len(puzzles),
        "total_sims": total_sims,
        "total_time_s": total_time,
        "sims_per_sec": sims_per_sec,
        "score": score,
    }


def run_hyperparam_search(
    num_samples: int = 24,
    puzzles_count: int = 4,
    simulations: int = 150,
    model_preset: str = "transformer_tiny",
    output_file: str = "logs/hyperparam_search.json",
    seed: int = 42,
):
    """Execute hyperparameter simulation search across configuration grid."""
    random.seed(seed)
    np.random.seed(seed)

    keys = list(GRID.keys())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*[GRID[k] for k in keys])]
    print(f"Total hyperparameter grid size: {len(all_combos)} combinations")

    if num_samples is not None and num_samples < len(all_combos):
        selected_combos = random.sample(all_combos, num_samples)
    else:
        selected_combos = all_combos

    test_puzzles = BUILTIN_PUZZLES[:puzzles_count]
    print(f"Testing {len(selected_combos)} configurations across {len(test_puzzles)} test positions...")
    print("=" * 80)
    print(f"{'#':<4}{'cpuct':<8}{'alpha':<8}{'eps':<8}{'VL':<6}{'BS':<6}{'Workers':<9}{'Sims/s':<10}{'Acc %':<8}{'Score':<8}")
    print("-" * 80)

    results = []
    out_path = PROJECT_ROOT / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(selected_combos, 1):
        try:
            res = evaluate_config(
                combo,
                test_puzzles,
                model_preset=model_preset,
                simulations=simulations,
            )
            results.append(res)

            p = combo
            print(
                f"{i:<4}{p['cpuct']:<8.2f}{p['dirichlet_alpha']:<8.2f}{p['dirichlet_eps']:<8.2f}"
                f"{p['virtual_loss']:<6}{p['batch_size']:<6}{p['cpu_workers']:<9}"
                f"{res['sims_per_sec']:<10.1f}{res['accuracy']*100:<8.1f}{res['score']:<8.1f}"
            )
        except Exception as e:
            print(f"{i:<4} Failed for config {combo}: {e}", file=sys.stderr)

    # Sort results by composite score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Save to JSON
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 80)
    print(f"Results saved to {out_path}")

    # Print top 5 configurations
    print("\nTop Parameter Configurations:")
    print("=" * 80)
    for rank, res in enumerate(results[:5], 1):
        p = res["params"]
        print(
            f"Rank {rank}: Score={res['score']:.1f} | Acc={res['accuracy']*100:.1f}% | "
            f"{res['sims_per_sec']:.1f} sims/s | cpuct={p['cpuct']}, alpha={p['dirichlet_alpha']}, "
            f"eps={p['dirichlet_eps']}, VL={p['virtual_loss']}, BS={p['batch_size']}, workers={p['cpu_workers']}"
        )
    print("=" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(description="UniChessTransformer MCTS Hyperparameter Search")
    parser.add_argument("--samples", "--rounds", dest="samples", type=int, default=12, help="Number of configurations/rounds to test")
    parser.add_argument("--puzzles", type=int, default=3, help="Number of positions per evaluation")
    parser.add_argument("--sims", type=int, default=120, help="Simulations per position")
    parser.add_argument("--preset", type=str, default="transformer_tiny", help="Model preset")
    parser.add_argument("--output", type=str, default="logs/hyperparam_search.json", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_hyperparam_search(
        num_samples=args.samples,
        puzzles_count=args.puzzles,
        simulations=args.sims,
        model_preset=args.preset,
        output_file=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
