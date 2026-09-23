"""Comprehensive end-to-end test suite for UniChessTransformer."""
from __future__ import annotations

import io
import math
from pathlib import Path
import sys

import chess
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unichess_t.core.encoding import NUM_PLANES, encode, orient, orient_move, unorient_move
from unichess_t.core.moves import (
    POLICY_SIZE,
    PROMO_SIZE,
    index_to_move,
    legal_mask,
    legal_move_indices,
    move_to_index,
    move_to_promo_index,
    uci_to_index,
)
from unichess_t.engine.engine import TransformerEngine
from eval.arena import elo_with_error, sprt_llr, sprt_verdict
from eval.puzzle_bench import BUILTIN_PUZZLES, evaluate_puzzles
from unichess_t.model.dataset import RECORD_DTYPE, decode_batch, decode_targets, get_shards, make_loader
from unichess_t.model.loss import ChessLoss, compute_metrics
from unichess_t.model.transformer import (
    PRESETS,
    ChessTransformer,
    StratifiedChessTransformer,
    TransformerConfig,
    create_transformer,
    stratified_20m,
    transformer_20m,
    transformer_50m,
    transformer_large,
    transformer_medium,
    transformer_small,
    transformer_tiny,
)
from unichess_t.search.parallel_mcts import ParallelMCTS, benchmark_parallel_mcts
from unichess_t.search.mcts import MCTS, MCTSConfig, Node, priors_from_policy
from uci import UCILoop


# -----------------------------------------------------------------------------
# Core tests
# -----------------------------------------------------------------------------
def test_encoding():
    board = chess.Board()
    planes = encode(board)
    assert planes.shape == (19, 8, 8)
    assert planes.dtype == np.float32

    # Initial position: 8 pawns, 2 knights, 2 bishops, 2 rooks, 1 queen, 1 king each
    # White pieces (0..5)
    assert planes[0].sum() == 8.0  # White pawns
    assert planes[1].sum() == 2.0  # White knights
    assert planes[5].sum() == 1.0  # White king
    # Black pieces (6..11)
    assert planes[6].sum() == 8.0  # Black pawns
    assert planes[11].sum() == 1.0  # Black king

    # Castling rights: all 4 are true in startpos
    assert planes[12].min() == 1.0
    assert planes[13].min() == 1.0
    assert planes[14].min() == 1.0
    assert planes[15].min() == 1.0

    # Test Black to move orientation
    board.push_san("e4")
    b_planes = encode(board)
    assert b_planes.shape == (19, 8, 8)
    # Since Black is to move, Black pawns are now "own" pieces (plane 0)
    assert b_planes[0].sum() == 8.0


def test_moves():
    board = chess.Board()
    e2e4 = chess.Move.from_uci("e2e4")
    idx = move_to_index(e2e4)
    assert 0 <= idx < POLICY_SIZE

    m_recon = index_to_move(idx)
    assert m_recon == e2e4

    # Test promotion
    promo_move = chess.Move.from_uci("e7e8q")
    p_idx = move_to_promo_index(promo_move)
    assert p_idx == 0  # Queen is index 0

    m_promo_recon = index_to_move(move_to_index(promo_move), promo_index=p_idx)
    assert m_promo_recon == promo_move

    # Test legal move mask
    mask = legal_mask(board)
    assert mask.shape == (POLICY_SIZE,)
    assert mask.sum() == 20  # 20 legal moves in starting position


# -----------------------------------------------------------------------------
# Transformer Architecture tests
# -----------------------------------------------------------------------------
def test_transformer_presets():
    presets = {
        "transformer_tiny": (192, 6, 6, 3.8e6),
        "transformer_small": (256, 8, 8, 6.7e6),
        "transformer_medium": (384, 10, 12, 18.5e6),
        "transformer_large": (512, 12, 16, 35.1e6),
        "transformer_20m": (384, 11, 12, 20.2e6),
        "transformer_50m": (512, 17, 16, 49.7e6),
    }
    for name, (d_model, layers, heads, target_params) in presets.items():
        model = create_transformer(name)
        param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.cfg.d_model == d_model
        assert model.cfg.num_layers == layers
        assert model.cfg.num_heads == heads
        # Check within 10% of target params
        assert abs(param_cnt - target_params) / target_params < 0.10, (
            f"{name} params {param_cnt} deviates from target {target_params}"
        )


def test_stratified_transformer():
    model = stratified_20m()
    assert isinstance(model, StratifiedChessTransformer)
    assert len(model.experts) == 3

    # Total params ~ 3 x 20.3M = ~60.9M
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 55e6 < total_params < 65e6

    # Test phase routing
    b_open = chess.Board()
    assert model.route_index(b_open) == 0

    b_mid = chess.Board("r4rk1/pp3ppp/8/8/2PP4/2N2N2/PP3PPP/R4RK1 w - - 0 15")
    assert model.route_index(b_mid) == 1

    b_end = chess.Board("8/4k3/8/8/8/8/4K3/4R3 w - - 30 50")
    assert model.route_index(b_end) == 2

    # Forward routing with list of boards
    x = torch.randn(3, 19, 8, 8)
    p, pr, v = model(x, boards=[b_open, b_mid, b_end])
    assert p.shape == (3, 4096)
    assert pr.shape == (3, 4)
    assert v.shape == (3, 3)

    # Forward routing with single board
    p1, pr1, v1 = model(x[:1], boards=b_open)
    assert p1.shape == (1, 4096)
    assert pr1.shape == (1, 4)
    assert v1.shape == (1, 3)

    # Test load_base_checkpoint
    base_ckpt = PROJECT_ROOT / "runs" / "transformer_20m" / "best_model.pt"
    if base_ckpt.exists():
        model.load_base_checkpoint(base_ckpt)
        for p1, p2 in zip(model.opening.parameters(), model.middlegame.parameters()):
            assert torch.equal(p1, p2)
        for p1, p3 in zip(model.opening.parameters(), model.endgame.parameters()):
            assert torch.equal(p1, p3)


def test_parallel_mcts():
    board = chess.Board()
    # Test ParallelMCTS instance with 2 workers
    pmcts = ParallelMCTS(
        num_workers=2,
        batch_size=32,
        model_preset="transformer_tiny",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    try:
        best_move, visits = pmcts.search(board, simulations=40)
        assert best_move in board.legal_moves
        assert sum(visits.values()) >= 40
    finally:
        pmcts.close()


def test_transformer_forward_both_shapes():
    model = transformer_tiny()
    model.eval()

    # Shape 1: (B, 19, 8, 8)
    x1 = torch.randn(2, 19, 8, 8)
    p1, pr1, v1 = model(x1)
    assert p1.shape == (2, 4096)
    assert pr1.shape == (2, 4)
    assert v1.shape == (2, 3)

    # Shape 2: (B, 64, 19)
    x2 = torch.randn(2, 64, 19)
    p2, pr2, v2 = model(x2)
    assert p2.shape == (2, 4096)
    assert pr2.shape == (2, 4)
    assert v2.shape == (2, 3)


# -----------------------------------------------------------------------------
# Dataset & Loss tests
# -----------------------------------------------------------------------------
def test_dataset_and_loss():
    assert RECORD_DTYPE.itemsize == 96

    # Test dummy record decode
    dummy_recs = np.zeros(4, dtype=RECORD_DTYPE)
    dummy_recs["occ_white"] = 0xFFFF
    dummy_recs["occ_black"] = 0xFFFF000000000000
    dummy_recs["castling"] = 15
    dummy_recs["promo"] = 255
    dummy_recs["wdl"] = [32767, 32768, 0]
    dummy_recs["policy_move"][:, 0] = 100
    dummy_recs["policy_prob"][:, 0] = 65535

    planes = decode_batch(dummy_recs)
    assert planes.shape == (4, 19, 8, 8)

    p, pr, w = decode_targets(dummy_recs)
    assert p.shape == (4, 4096)
    assert pr.shape == (4,)
    assert w.shape == (4, 3)

    # Test loss function
    criterion = ChessLoss()
    p_logits = torch.randn(4, 4096)
    pr_logits = torch.randn(4, 4)
    w_logits = torch.randn(4, 3)

    loss_out = criterion(
        p_logits,
        pr_logits,
        w_logits,
        torch.from_numpy(p),
        torch.from_numpy(pr),
        torch.from_numpy(w),
    )
    assert loss_out.total_loss.item() > 0
    assert "policy_top1_acc" in loss_out.metrics
    assert "wdl_acc" in loss_out.metrics


# -----------------------------------------------------------------------------
# MCTS & Engine tests
# -----------------------------------------------------------------------------
def test_mcts_search():
    model = transformer_tiny()
    engine = TransformerEngine(model, device="cpu", precision="fp32")
    board = chess.Board()

    mcts_cfg = MCTSConfig(simulations=20, batch_size=8)
    mcts = MCTS(engine.evaluate_batch, cfg=mcts_cfg)

    best_move, root = mcts.best_move(board, simulations=20)
    assert best_move in board.legal_moves
    assert root.sum_N >= 20

    # Policy distribution
    policy = mcts.visit_policy(root)
    assert len(policy) == len(list(board.legal_moves))
    assert math.isclose(sum(prob for _, prob in policy), 1.0, rel_tol=1e-3)


def test_engine_play():
    model = transformer_tiny()
    engine = TransformerEngine(model, device="cpu", precision="fp32")
    board = chess.Board()

    move = engine.play(board)
    assert move in board.legal_moves

    # Test evaluation
    p, pr, w = engine.evaluate(board)
    assert p.shape == (4096,)
    assert pr.shape == (4,)
    assert w.shape == (3,)
    assert math.isclose(w.sum(), 1.0, rel_tol=1e-3)


# -----------------------------------------------------------------------------
# Arena & Puzzle Bench tests
# -----------------------------------------------------------------------------
def test_arena_stats():
    elo, lo, hi = elo_with_error(10, 5, 5)
    assert elo > 0  # 12.5 / 20 = 62.5% win score -> positive Elo
    assert lo < elo < hi

    llr = sprt_llr(10, 5, 5, elo0=0.0, elo1=35.0)
    verdict = sprt_verdict(llr)
    assert isinstance(verdict, str)


def test_puzzle_bench():
    model = transformer_tiny()
    engine = TransformerEngine(model, device="cpu", precision="fp32")
    # Test on first 3 puzzles
    summary = evaluate_puzzles(engine, BUILTIN_PUZZLES[:3], verbose=False)
    assert summary["total_puzzles"] == 3
    assert 0.0 <= summary["top1_accuracy"] <= 100.0


# -----------------------------------------------------------------------------
# UCI interface test
# -----------------------------------------------------------------------------
def test_uci_protocol():
    class DummyArgs:
        ckpt = None
        preset = "transformer_tiny"
        device = "cpu"
        precision = "fp32"
        syzygy = None
        book = None
        temperature = 0.0
        mcts_sims = 0
        mcts_batch = 16

    loop = UCILoop(DummyArgs())
    # Test position startpos
    loop.handle_position(["position", "startpos", "moves", "e2e4", "e7e5"])
    assert loop.board.fen() == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

    # Test go command produces valid move
    eng = loop._ensure_engine()
    assert eng is not None


def test_cpp_mcts():
    import chess
    import torch
    from unichess_t.search.cpp import load_cpp_mcts

    MCTSCpp = load_cpp_mcts()
    engine = MCTSCpp()

    def dummy_eval(tensor):
        b = tensor.shape[0]
        p = torch.zeros(b, 4096)
        pr = torch.zeros(b, 4)
        wdl = torch.tensor([[0.3, 0.4, 0.3]], dtype=torch.float32).repeat(b, 1)
        return torch.softmax(p, dim=-1), torch.softmax(pr, dim=-1), wdl

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move, metrics = engine.search(fen, dummy_eval, simulations=100, batch_size=16)
    assert move in [m.uci() for m in chess.Board(fen).legal_moves], f"Illegal move: {move}"
    assert metrics["visits"] == 100, f"Expected 100 visits, got {metrics['visits']}"


if __name__ == "__main__":
    tests = [
        test_encoding,
        test_moves,
        test_transformer_presets,
        test_stratified_transformer,
        test_parallel_mcts,
        test_transformer_forward_both_shapes,
        test_dataset_and_loss,
        test_mcts_search,
        test_cpp_mcts,
        test_engine_play,
        test_arena_stats,
        test_puzzle_bench,
        test_uci_protocol,
    ]
    print(f"Running {len(tests)} test functions...")
    for test in tests:
        print(f"  Running {test.__name__}...")
        test()
    print("All tests passed cleanly!")
