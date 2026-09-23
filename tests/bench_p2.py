"""Quick benchmark for Stage P2 features."""
from __future__ import annotations

import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import torch
from unichess_t.search.cpp import load_cpp_mcts


def dummy_eval(tensor):
    b = tensor.shape[0]
    p = torch.zeros(b, 4096)
    pr = torch.zeros(b, 4)
    wdl = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32).repeat(b, 1)
    return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl


def benchmark_no_reuse(sims=100, num_moves=10):
    engine = load_cpp_mcts()()
    board = chess.Board()
    times = []
    for _ in range(num_moves):
        fen = board.fen()
        t0 = time.perf_counter()
        move, metrics = engine.search(fen, dummy_eval, simulations=sims, batch_size=16)
        dt = time.perf_counter() - t0
        times.append(dt)
        if move:
            board.push_uci(move)
        if board.is_game_over():
            break
    return times


def benchmark_with_reuse(sims=100, num_moves=10):
    engine = load_cpp_mcts()()
    board = chess.Board()
    prev_fen = board.fen()
    times = []
    for _ in range(num_moves):
        fen = board.fen()
        t0 = time.perf_counter()
        if _ == 0:
            move, metrics = engine.search(fen, dummy_eval, simulations=sims, batch_size=16)
        else:
            move, metrics = engine.search(
                fen, dummy_eval, simulations=sims, batch_size=16,
                reuse=True, previous_root_fen=prev_fen
            )
        dt = time.perf_counter() - t0
        times.append(dt)
        if move:
            board.push_uci(move)
            prev_fen = fen
        if board.is_game_over():
            break
    return times


def benchmark_contempt():
    engine = load_cpp_mcts()()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def draw_eval(tensor):
        b = tensor.shape[0]
        p = torch.zeros(b, 4096)
        pr = torch.zeros(b, 4)
        wdl = torch.zeros(b, 3)
        wdl[:, 1] = 1.0
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl
    
    _, m0 = engine.search(fen, draw_eval, simulations=200, batch_size=32, contempt=0.0)
    _, m1 = engine.search(fen, draw_eval, simulations=200, batch_size=32, contempt=0.5)
    _, m2 = engine.search(fen, draw_eval, simulations=200, batch_size=32, contempt=-0.5)
    print(f"contempt=0.0: root_value={m0['root_value']:.4f}, visits={m0['visits']}")
    print(f"contempt=0.5: root_value={m1['root_value']:.4f}, visits={m1['visits']}")
    print(f"contempt=-0.5: root_value={m2['root_value']:.4f}, visits={m2['visits']}")
    
    # With all draws (WDL = [0,1,0]), the network returns v = 0 for all leaves.
    # Contempt should bias these draws during backup.
    # Positive contempt should make root_value slightly positive (draws favored for root).
    # Negative contempt should make root_value slightly negative.
    print(f"Contempt difference (0.5 - 0.0): {m1['root_value'] - m0['root_value']:.4f}")
    print(f"Contempt difference (-0.5 - 0.0): {m2['root_value'] - m0['root_value']:.4f}")


if __name__ == "__main__":
    print("Benchmarking P2 features...")
    
    print("\n--- No Tree Reuse ---")
    times_no_reuse = benchmark_no_reuse(sims=100, num_moves=10)
    avg_no_reuse = sum(times_no_reuse) / len(times_no_reuse)
    print(f"Times: {[f'{t:.3f}' for t in times_no_reuse]}")
    print(f"Average: {avg_no_reuse:.3f}s")
    
    print("\n--- With Tree Reuse ---")
    times_reuse = benchmark_with_reuse(sims=100, num_moves=10)
    avg_reuse = sum(times_reuse) / len(times_reuse)
    print(f"Times: {[f'{t:.3f}' for t in times_reuse]}")
    print(f"Average: {avg_reuse:.3f}s")
    
    if avg_no_reuse > 0:
        speedup = avg_no_reuse / avg_reuse
        print(f"\nSpeedup: {speedup:.2f}x")
    
    print("\n--- Contempt Effect (all-draw evaluator) ---")
    benchmark_contempt()
    
    print("\nBenchmark complete.")
