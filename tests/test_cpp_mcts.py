"""Comprehensive verification and stress test suite for search/cpp (C++ MCTS & Bitboard)."""
from __future__ import annotations

from pathlib import Path
import random
import sys
import time

import chess
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.encoding import encode
from search.cpp import encode_planes, get_legal_moves, load_cpp_mcts, perft


def test_perft():
    print("--- Running Perft Tests ---")
    # Startpos
    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    startpos_expected = {1: 20, 2: 400, 3: 8902, 4: 197281}
    for d, expected in startpos_expected.items():
        res = perft(startpos, d)
        print(f"Startpos Depth {d}: got {res}, expected {expected}")
        assert res == expected, f"Startpos Depth {d} failed: got {res}, expected {expected}"

    # Kiwipete
    kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    kiwipete_expected = {1: 48, 2: 2039, 3: 97862}
    for d, expected in kiwipete_expected.items():
        res = perft(kiwipete, d)
        print(f"Kiwipete Depth {d}: got {res}, expected {expected}")
        assert res == expected, f"Kiwipete Depth {d} failed: got {res}, expected {expected}"

    # Position 3
    pos3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    pos3_expected = {1: 14, 2: 191, 3: 2812}
    for d, expected in pos3_expected.items():
        res = perft(pos3, d)
        print(f"Position 3 Depth {d}: got {res}, expected {expected}")
        assert res == expected, f"Position 3 Depth {d} failed: got {res}, expected {expected}"

    # Position 4
    pos4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    pos4_expected = {1: 6, 2: 264, 3: 9467}
    for d, expected in pos4_expected.items():
        res = perft(pos4, d)
        print(f"Position 4 Depth {d}: got {res}, expected {expected}")
        assert res == expected, f"Position 4 Depth {d} failed: got {res}, expected {expected}"
    print("PASS: Perft Tests")


def generate_diverse_fens(count: int = 50) -> list[str]:
    rng = random.Random(1337)
    fens = set()
    # Canonical positions to ensure edge cases are present
    presets = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqk2r/pppp1ppp/2n5/4p3/1bB1n3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 6",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1",
        "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
        "8/8/8/8/8/5k2/4p3/4K3 w - - 0 1",
    ]
    for p in presets:
        fens.add(p)

    while len(fens) < count:
        board = chess.Board()
        steps = rng.randint(1, 100)
        for _ in range(steps):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            board.push(rng.choice(moves))
        fens.add(board.fen())

    return sorted(list(fens))[:count]


def test_move_legality_and_parity():
    print("--- Running Move Legality & Parity Tests ---")
    fens = generate_diverse_fens(50)
    assert len(fens) == 50

    for idx, fen in enumerate(fens):
        b = chess.Board(fen)
        expected = sorted([m.uci() for m in b.legal_moves])
        cpp_moves = sorted(get_legal_moves(fen))
        assert cpp_moves == expected, (
            f"Parity mismatch on position {idx} ({fen}):\n"
            f"C++ ({len(cpp_moves)}): {cpp_moves}\n"
            f"Py  ({len(expected)}): {expected}"
        )
    print(f"PASS: Move Legality & Parity on all {len(fens)} diverse positions")


def test_19plane_encoding_parity():
    print("--- Running 19-Plane Encoding Parity Tests ---")
    fens = generate_diverse_fens(50)
    for idx, fen in enumerate(fens):
        b = chess.Board(fen)
        py_planes = torch.from_numpy(encode(b))
        cpp_planes = encode_planes(fen)
        assert cpp_planes.shape == (19, 8, 8)
        assert torch.allclose(cpp_planes, py_planes, atol=1e-6), (
            f"Encoding mismatch on position {idx} ({fen}):\n"
            f"Max diff: {torch.max(torch.abs(cpp_planes - py_planes)).item()}"
        )
    print(f"PASS: 19-Plane Encoding Parity on all {len(fens)} diverse positions")


def test_terminal_states_and_edge_cases():
    print("--- Running Terminal States & Edge Cases Tests ---")
    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def dummy_eval(tensor):
        b = tensor.shape[0]
        p = torch.zeros(b, 4096)
        pr = torch.zeros(b, 4)
        wdl = torch.tensor([[0.33, 0.34, 0.33]], dtype=torch.float32).repeat(b, 1)
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    # 1. Checkmate position (Fool's Mate: 1. f3 e5 2. g4 Qh4#)
    mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert chess.Board(mate_fen).is_checkmate()
    best_move, metrics = engine.search(mate_fen, dummy_eval, simulations=50, batch_size=8)
    assert best_move == "", f"Expected empty move on checkmate, got '{best_move}'"
    assert metrics["visits"] == 0
    assert metrics["root_value"] == -1.0, f"Expected terminal value -1.0, got {metrics['root_value']}"

    # 2. Stalemate position
    stalemate_fen = "k7/8/1Q6/8/8/8/8/7K b - - 0 1"
    assert chess.Board(stalemate_fen).is_stalemate()
    best_move, metrics = engine.search(stalemate_fen, dummy_eval, simulations=50, batch_size=8)
    assert best_move == "", f"Expected empty move on stalemate, got '{best_move}'"
    assert metrics["visits"] == 0
    assert metrics["root_value"] == 0.0, f"Expected terminal value 0.0, got {metrics['root_value']}"

    # 3. Exactly 1 legal move position
    # White King on a1, Black King on c1, Black Rook on b3: only legal move is Ka1-a2
    one_move_fen = "8/8/8/8/8/1r6/8/K1k5 w - - 0 1"
    b_one = chess.Board(one_move_fen)
    legal_moves = [m.uci() for m in b_one.legal_moves]
    assert len(legal_moves) == 1, f"Expected 1 legal move, got {legal_moves}"
    best_move, metrics = engine.search(one_move_fen, dummy_eval, simulations=50, batch_size=8)
    assert best_move == legal_moves[0], f"Expected single legal move {legal_moves[0]}, got {best_move}"
    assert metrics["visits"] == 50

    print("PASS: Terminal States & Edge Cases")


def test_memory_and_stability_stress():
    print("--- Running Memory & Stability Stress Test (100 searches) ---")
    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def dummy_eval(tensor):
        b = tensor.shape[0]
        p = torch.randn(b, 4096)
        pr = torch.randn(b, 4)
        wdl = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32).repeat(b, 1)
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    fens = generate_diverse_fens(50)
    rng = random.Random(42)

    t0 = time.perf_counter()
    for i in range(100):
        fen = rng.choice(fens)
        sims = rng.choice([32, 64, 128])
        batch_size = rng.choice([8, 16, 32])
        best_move, metrics = engine.search(fen, dummy_eval, simulations=sims, batch_size=batch_size)
        legals = [m.uci() for m in chess.Board(fen).legal_moves]
        if legals:
            assert best_move in legals, f"Search returned illegal move '{best_move}' in position: {fen}"
            assert metrics["visits"] == sims, f"Expected {sims} visits, got {metrics['visits']}"
        else:
            assert best_move == ""
    elapsed = time.perf_counter() - t0
    print(f"PASS: Completed 100 sequential MCTS searches in {elapsed:.2f}s without crash or leak")


def test_invariants_and_policy_validity():
    print("--- Running Invariants & Policy Validity Tests ---")
    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def dummy_eval(tensor):
        b = tensor.shape[0]
        p = torch.ones(b, 4096)
        pr = torch.ones(b, 4)
        wdl = torch.tensor([[0.4, 0.3, 0.3]], dtype=torch.float32).repeat(b, 1)
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    sims = 200
    best_move, metrics = engine.search(fen, dummy_eval, simulations=sims, batch_size=32, add_noise=True)

    assert metrics["visits"] == sims
    policy = metrics["policy"]
    total_policy_visits = sum(policy.values())
    assert total_policy_visits == sims, f"Policy visits sum {total_policy_visits} != {sims}"

    b = chess.Board(fen)
    legal_uci = set(m.uci() for m in b.legal_moves)
    for move_uci, visits in policy.items():
        assert move_uci in legal_uci, f"Move {move_uci} in policy not legal"
        assert visits >= 0

    assert best_move in legal_uci
    print("PASS: Invariants & Policy Validity")


if __name__ == "__main__":
    test_perft()
    test_move_legality_and_parity()
    test_19plane_encoding_parity()
    test_terminal_states_and_edge_cases()
    test_memory_and_stability_stress()
    test_invariants_and_policy_validity()
    print("\nALL C++ MCTS TESTS PASSED SUCCESSFULLY!")
