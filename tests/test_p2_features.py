"""Tests for Stage P2: Tree Reuse, Dynamic FPU, Contempt."""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unichess_t.search.cpp import load_cpp_mcts


def test_tree_reuse_basic():
    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def dummy_eval(tensor):
        b = tensor.shape[0]
        p = torch.zeros(b, 4096)
        pr = torch.zeros(b, 4)
        wdl = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32).repeat(b, 1)
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)
    legal_moves = [m.uci() for m in board.legal_moves]

    move1, metrics1 = engine.search(fen, dummy_eval, simulations=100, batch_size=16)
    assert move1 in legal_moves
    assert metrics1["visits"] == 100
    assert metrics1["reused_root"] == False

    board.push_uci(move1)
    fen2 = board.fen()

    success = engine.reuse_root(move1)
    assert success, "reuse_root should succeed"

    move2, metrics2 = engine.search(fen2, dummy_eval, simulations=100, batch_size=16, reuse=True)
    assert move2 in [m.uci() for m in board.legal_moves]
    assert metrics2["reused_root"] == True
    assert metrics2["visits"] == 100


def test_tree_reuse_auto_detect():
    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def dummy_eval(tensor):
        b = tensor.shape[0]
        p = torch.zeros(b, 4096)
        pr = torch.zeros(b, 4)
        wdl = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32).repeat(b, 1)
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)
    move1, _ = engine.search(fen, dummy_eval, simulations=100, batch_size=16)
    board.push_uci(move1)
    fen2 = board.fen()

    move2, metrics2 = engine.search(
        fen2, dummy_eval, simulations=100, batch_size=16,
        reuse=True, previous_root_fen=fen
    )
    assert metrics2["reused_root"] == True
    assert metrics2["visits"] == 100


def test_contempt_nonzero():
    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def draw_eval(tensor):
        b = tensor.shape[0]
        p = torch.zeros(b, 4096)
        pr = torch.zeros(b, 4)
        wdl = torch.zeros(b, 3)
        wdl[:, 1] = 1.0
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    stalemate_fen = "k7/8/1Q6/8/8/8/8/7K b - - 0 1"
    _, metrics_no_contempt = engine.search(stalemate_fen, draw_eval, simulations=10, batch_size=4, contempt=0.0)
    assert metrics_no_contempt["root_value"] == 0.0

    _, metrics_contempt = engine.search(stalemate_fen, draw_eval, simulations=10, batch_size=4, contempt=0.5)
    assert metrics_contempt["reused_root"] == False


def test_c_fpu_parameter():
    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def dummy_eval(tensor):
        b = tensor.shape[0]
        p = torch.zeros(b, 4096)
        pr = torch.zeros(b, 4)
        wdl = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32).repeat(b, 1)
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move1, metrics1 = engine.search(fen, dummy_eval, simulations=100, batch_size=16, c_fpu=0.8)
    assert move1 in [m.uci() for m in chess.Board(fen).legal_moves]
    assert metrics1["visits"] == 100


def test_reset_clears_tree():
    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def dummy_eval(tensor):
        b = tensor.shape[0]
        p = torch.zeros(b, 4096)
        pr = torch.zeros(b, 4)
        wdl = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32).repeat(b, 1)
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.search(fen, dummy_eval, simulations=100, batch_size=16)
    success = engine.reuse_root("e2e4")
    assert success, "reuse_root should succeed after search"

    engine.reset()
    success = engine.reuse_root("e2e4")
    assert not success, "reuse_root should fail after reset"


if __name__ == "__main__":
    test_tree_reuse_basic()
    print("PASS: test_tree_reuse_basic")
    test_tree_reuse_auto_detect()
    print("PASS: test_tree_reuse_auto_detect")
    test_contempt_nonzero()
    print("PASS: test_contempt_nonzero")
    test_c_fpu_parameter()
    print("PASS: test_c_fpu_parameter")
    test_reset_clears_tree()
    print("PASS: test_reset_clears_tree")
    print("\nALL P2 FEATURE TESTS PASSED!")
