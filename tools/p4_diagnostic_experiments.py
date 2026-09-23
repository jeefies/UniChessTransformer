#!/usr/bin/env python3
"""P4 Self-Play Regression Diagnostic Experiments.

Runs 5 small-scale corrective experiments to identify and fix the self-play
policy target distribution shift causing regression from 10-0 (P1) to 5-5 (P4).

Base model: runs/stratified_p1_opening/best_model.pt
Baseline match: 10-0 vs Model R (P1)
P4 regression: 5-5 vs Model R

Experiments:
  A: KL divergence loss instead of CE for policy
  B: 50% self-play + 50% original supervised data
  C: Lower LR (5e-6) with gradient accumulation (effective LR ~1.25e-6)
  D: Freeze policy head, only train value head on self-play
  E: Temperature smoothing on MCTS visit counts (T=2.0)

Each experiment:
  - 500 training steps
  - 4-game mini match vs Model R
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, '/home/jeefy/UniChess/Server')
from models import create_engine, _load_engine_class, resolve_kwargs

from unichess_t.core.encoding import encode
from unichess_t.core.moves import move_to_index, move_to_promo_index
from unichess_t.engine.engine import TransformerEngine
from unichess_t.model.dataset import get_shards, make_loader
from unichess_t.model.loss import ChessLoss, LossOutput, compute_metrics
from unichess_t.model.transformer import create_transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("P4Diagnostics")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_CKPT = "/home/jeefy/UniChess/Transformer/runs/stratified_p1_opening/best_model.pt"
DATA_DIR = "/home/jeefy/UniChess/data/shards_evals"
RESULT_DIR = PROJECT_ROOT / "logs" / "p4_diagnostics"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Self-play data collection (reuse existing logic)
# ---------------------------------------------------------------------------
class SelfPlayDataset(Dataset):
    def __init__(self, positions):
        states = np.stack([p[0] for p in positions])
        policy_targets = np.stack([p[1] for p in positions])
        promo_targets = np.array([p[2] for p in positions])
        scalar_values = np.array([p[3] for p in positions])

        wdl_targets = np.zeros((len(positions), 3), dtype=np.float32)
        wdl_targets[scalar_values > 0] = [1, 0, 0]
        wdl_targets[scalar_values < 0] = [0, 0, 1]
        wdl_targets[scalar_values == 0] = [0, 1, 0]

        self.states = torch.from_numpy(states).float()
        self.policy_targets = torch.from_numpy(policy_targets).float()
        self.promo_targets = torch.tensor(promo_targets, dtype=torch.long)
        self.wdl_targets = torch.from_numpy(wdl_targets).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policy_targets[idx], self.promo_targets[idx], self.wdl_targets[idx]


def evaluate_tensor_callback(engine: TransformerEngine):
    @torch.no_grad()
    def evaluator(x: torch.Tensor):
        return engine.evaluate_tensor(x)
    return evaluator


def play_game_cpp(engine, cpp_mcts, simulations=800, batch_size=64, add_noise=True, max_plies=500):
    board = chess.Board()
    game_data = []
    evaluator = evaluate_tensor_callback(engine)
    first_move = True

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        fen = board.fen()
        if first_move:
            best_move, metrics = cpp_mcts.search(
                fen, evaluator, simulations=simulations, batch_size=batch_size,
                add_noise=add_noise, temperature=0.0,
            )
            first_move = False
        else:
            best_move, metrics = cpp_mcts.search(
                fen, evaluator, simulations=simulations, batch_size=batch_size,
                add_noise=False, temperature=0.0, reuse=True,
            )
        if not best_move:
            break

        visit_counts = metrics.get("policy", {})
        policy_target = np.zeros(4096, dtype=np.float32)
        total_visits = 0
        for move_uci, visits in visit_counts.items():
            try:
                mv = chess.Move.from_uci(move_uci)
                idx = move_to_index(mv)
                policy_target[idx] = float(visits)
                total_visits += float(visits)
            except Exception:
                continue
        if total_visits > 0:
            policy_target /= total_visits

        state = encode(board).astype(np.float32)
        chosen_move = chess.Move.from_uci(best_move)
        promo_target = -100
        pi = move_to_promo_index(chosen_move)
        if pi is not None:
            promo_target = pi
        game_data.append((state, policy_target, promo_target, 0.0, chosen_move))
        board.push(chosen_move)
        try:
            cpp_mcts.reuse_root(best_move)
        except Exception:
            cpp_mcts.reset()
            first_move = True

    outcome = board.outcome(claim_draw=True)
    z = 0.0
    if outcome is not None and outcome.winner is not None:
        z = 1.0 if outcome.winner == chess.WHITE else -1.0

    board_replay = chess.Board()
    result = []
    for state, policy_target, promo_target, _, move in game_data:
        turn = board_replay.turn
        scalar_value = z if turn == chess.WHITE else -z
        result.append((state, policy_target, promo_target, scalar_value))
        if move in board_replay.legal_moves:
            board_replay.push(move)
        else:
            break
    return result


def play_game_python(engine, simulations=800, batch_size=128, add_noise=True, max_plies=500):
    board = chess.Board()
    game_data = []
    from unichess_t.search.mcts import MCTS, MCTSConfig
    mcts_cfg = MCTSConfig(
        simulations=simulations, batch_size=batch_size, temperature=0.0,
        dirichlet_eps=0.25 if add_noise else 0.0,
    )
    mcts = MCTS(engine.evaluate_batch, cfg=mcts_cfg, tablebase=engine.tablebase, rng=engine.np_rng)
    root = None

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mv, root = mcts.best_move(
            board, simulations=simulations, temperature=0.0, root=root,
            add_noise=add_noise and (root is None or not root.expanded),
        )
        visit_policy = MCTS.visit_policy(root)
        policy_target = np.zeros(4096, dtype=np.float32)
        total_visits = 0
        for move, visits in visit_policy:
            idx = move_to_index(move)
            policy_target[idx] = float(visits)
            total_visits += float(visits)
        if total_visits > 0:
            policy_target /= total_visits

        state = encode(board).astype(np.float32)
        promo_target = -100
        pi = move_to_promo_index(mv)
        if pi is not None:
            promo_target = pi
        game_data.append((state, policy_target, promo_target, 0.0, mv))
        board.push(mv)
        root = MCTS.advance_root(root, mv)

    outcome = board.outcome(claim_draw=True)
    z = 0.0
    if outcome is not None and outcome.winner is not None:
        z = 1.0 if outcome.winner == chess.WHITE else -1.0

    board_replay = chess.Board()
    result = []
    for state, policy_target, promo_target, _, move in game_data:
        turn = board_replay.turn
        scalar_value = z if turn == chess.WHITE else -z
        result.append((state, policy_target, promo_target, scalar_value))
        if move in board_replay.legal_moves:
            board_replay.push(move)
        else:
            break
    return result


def collect_selfplay_games(engine, cpp_mcts, num_games=10, simulations=800, batch_size=64):
    all_positions = []
    use_cpp = cpp_mcts is not None
    for game_idx in range(num_games):
        game_start = time.time()
        if use_cpp:
            try:
                game_data = play_game_cpp(engine, cpp_mcts, simulations=simulations, batch_size=batch_size)
            except Exception as e:
                logger.warning(f"C++ MCTS game {game_idx} failed ({e}), falling back to Python MCTS")
                cpp_mcts.reset()
                game_data = play_game_python(engine, simulations=simulations, batch_size=batch_size)
        else:
            game_data = play_game_python(engine, simulations=simulations, batch_size=batch_size)
        all_positions.extend(game_data)
        game_elapsed = time.time() - game_start
        logger.info(
            f"Self-play Game {game_idx+1}/{num_games}: {len(game_data)} plies, {game_elapsed:.1f}s "
            f"(total positions: {len(all_positions)})"
        )
    return all_positions


# ---------------------------------------------------------------------------
# Custom loss functions for experiments
# ---------------------------------------------------------------------------
class ChessLossKL(ChessLoss):
    """Experiment A: KL divergence loss for policy instead of CE."""

    def forward(self, policy_logits, promo_logits, value_wdl, policy_target, promo_target, wdl_target,
                mlh_logits=None, mlh_target=None):
        log_pred = F.log_softmax(policy_logits, dim=-1)
        loss_policy = F.kl_div(log_pred, policy_target, reduction="batchmean")

        valid_promo = (promo_target != -100)
        if valid_promo.any():
            loss_promo = F.cross_entropy(promo_logits[valid_promo], promo_target[valid_promo])
        else:
            loss_promo = torch.tensor(0.0, device=promo_logits.device, dtype=promo_logits.dtype)

        loss_wdl = F.cross_entropy(value_wdl, wdl_target)

        total_loss = (
            self.policy_weight * loss_policy
            + self.promo_weight * loss_promo
            + self.wdl_weight * loss_wdl
        )

        loss_mlh = None
        if mlh_logits is not None and mlh_target is not None:
            loss_mlh = F.smooth_l1_loss(mlh_logits, mlh_target)
            total_loss = total_loss + self.mlh_weight * loss_mlh

        with torch.no_grad():
            metrics = compute_metrics(policy_logits, promo_logits, value_wdl, policy_target, promo_target, wdl_target)
            if loss_mlh is not None:
                metrics["mlh_loss"] = loss_mlh.item()
                metrics["mlh_mae"] = F.l1_loss(mlh_logits, mlh_target).item()

        return LossOutput(
            total_loss=total_loss, policy_loss=loss_policy, promo_loss=loss_promo,
            wdl_loss=loss_wdl, metrics=metrics, mlh_loss=loss_mlh,
        )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train_steps(model, loader, optimizer, criterion, device, num_steps, amp_dtype=None, grad_accum_steps=1, freeze_policy=False):
    """Train for exactly num_steps optimizer steps."""
    model.train()
    if freeze_policy:
        if hasattr(model, 'experts'):
            for expert in model.experts:
                for p in expert.policy_head.parameters():
                    p.requires_grad = False

    total_loss = 0.0
    total_policy_loss = 0.0
    total_wdl_loss = 0.0
    total_p1 = 0.0
    total_p5 = 0.0
    total_wdl_acc = 0.0
    steps_done = 0

    data_iter = iter(loader)
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)

    while steps_done < num_steps:
        try:
            x, p, pr, w = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, p, pr, w = next(data_iter)

        x = x.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        pr = pr.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                p_out, pr_out, w_out = model(x)
                out = criterion(p_out, pr_out, w_out, p, pr, w)
                loss = out.total_loss / grad_accum_steps
            loss.backward()
        else:
            p_out, pr_out, w_out = model(x)
            out = criterion(p_out, pr_out, w_out, p, pr, w)
            loss = out.total_loss / grad_accum_steps
            loss.backward()

        accum_count += 1
        total_loss += out.total_loss.item()
        total_policy_loss += out.policy_loss.item()
        total_wdl_loss += out.wdl_loss.item()
        total_p1 += out.metrics.get("policy_top1_acc", 0.0)
        total_p5 += out.metrics.get("policy_top5_acc", 0.0)
        total_wdl_acc += out.metrics.get("wdl_acc", 0.0)

        if accum_count % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_done += 1
            accum_count = 0

            if steps_done % 100 == 0:
                logger.info(
                    f"  Step {steps_done}/{num_steps}: Loss={total_loss/steps_done:.4f} "
                    f"(Pol={total_policy_loss/steps_done:.4f}, WDL={total_wdl_loss/steps_done:.4f}) | "
                    f"P1={total_p1/steps_done*100:.2f}% WDL={total_wdl_acc/steps_done*100:.2f}%"
                )

    return {
        "total_loss": total_loss / max(1, steps_done),
        "policy_loss": total_policy_loss / max(1, steps_done),
        "wdl_loss": total_wdl_loss / max(1, steps_done),
        "policy_top1_acc": total_p1 / max(1, steps_done),
        "policy_top5_acc": total_p5 / max(1, steps_done),
        "wdl_acc": total_wdl_acc / max(1, steps_done),
        "steps": steps_done,
    }


def run_mini_match(ckpt_path, t_sims=2400, r_sims=800, num_games=4, log_path=None, pgn_path=None):
    """Run a mini match (4 games) vs Model R."""
    if log_path is None:
        log_path = RESULT_DIR / f"mini_match_{Path(ckpt_path).parent.name}.log"
    if pgn_path is None:
        pgn_path = RESULT_DIR / f"mini_match_{Path(ckpt_path).parent.name}.pgn"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    match_logger = logging.getLogger("MiniMatch")

    OPENINGS = [
        {"name": "Italian Game (Giuoco Piano)", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"], "eco": "C50"},
        {"name": "Ruy Lopez (Morphy Defense)", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"], "eco": "C70"},
        {"name": "Sicilian Defense (Open Sicilian)", "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"], "eco": "B90"},
        {"name": "French Defense (Classical)", "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"], "eco": "C11"},
        {"name": "Queen's Gambit Declined (Orthodox)", "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"], "eco": "D60"},
    ]

    match_logger.info(f"Initializing mini match: T={ckpt_path} (sims={t_sims}) vs R (sims={r_sims})")
    t_kwargs = resolve_kwargs('T', 'max_mcts')
    t_kwargs['ckpt'] = ckpt_path
    t_kwargs['mcts_sims'] = t_sims
    engine_t_cls = _load_engine_class('T')
    eng_t = engine_t_cls(**t_kwargs)

    r_kwargs = resolve_kwargs('R', 'max_mcts')
    r_kwargs['mcts_sims'] = r_sims
    engine_r_cls = _load_engine_class('R')
    eng_r = engine_r_cls(**r_kwargs)

    match_pairs = []
    game_num = 1
    for op in OPENINGS[:max(1, (num_games + 1) // 2)]:
        match_pairs.append((game_num, op, 'T', 'R'))
        game_num += 1
        if game_num > num_games:
            break
        match_pairs.append((game_num, op, 'R', 'T'))
        game_num += 1
        if game_num > num_games:
            break

    games_results = []
    with open(pgn_path, "w", encoding="utf-8") as f_pgn:
        pass

    for g_idx, op, w_name, b_name in match_pairs:
        eng_w = eng_t if w_name == 'T' else eng_r
        eng_b = eng_t if b_name == 'T' else eng_r
        board = chess.Board()
        for mv_uci in op["moves"]:
            board.push_uci(mv_uci)
        start_fen = board.fen()
        eng_w.setup(start_fen)
        eng_b.setup(start_fen)

        w_sims = t_sims if w_name == 'T' else r_sims
        b_sims = t_sims if b_name == 'T' else r_sims

        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = "P4 Diagnostic Mini Match"
        pgn_game.headers["Site"] = "Localhost"
        pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        pgn_game.headers["Round"] = str(g_idx)
        pgn_game.headers["White"] = f"Model {w_name} (sims {w_sims})"
        pgn_game.headers["Black"] = f"Model {b_name} (sims {b_sims})"
        pgn_game.headers["FEN"] = start_fen
        pgn_game.headers["SetUp"] = "1"
        pgn_game.headers["Opening"] = op["name"]
        pgn_game.headers["ECO"] = op["eco"]

        node = pgn_game
        white_times = []
        black_times = []
        plies = 0
        max_plies = 300

        match_logger.info(f"=== Game {g_idx}: W={w_name} vs B={b_name} | {op['name']} ===")
        while not board.is_game_over(claim_draw=True) and plies < max_plies:
            turn = board.turn
            current_eng = eng_w if turn == chess.WHITE else eng_b
            other_eng = eng_b if turn == chess.WHITE else eng_w
            current_name = w_name if turn == chess.WHITE else b_name

            t0 = time.time()
            res = current_eng.engine_move()
            dt = time.time() - t0
            move_uci = res.get("engine_move")
            if not move_uci:
                match_logger.warning(f"Engine {current_name} returned no move!")
                break
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                match_logger.error(f"Engine {current_name} made illegal move: {move_uci}")
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

        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                result_str = "1-0"
                winner = w_name
                reason = outcome.termination.name
            elif outcome.winner == chess.BLACK:
                result_str = "0-1"
                winner = b_name
                reason = outcome.termination.name
            else:
                result_str = "1/2-1/2"
                winner = "Draw"
                reason = outcome.termination.name
        else:
            result_str = "1/2-1/2"
            winner = "Draw"
            reason = "Move limit reached"

        pgn_game.headers["Result"] = result_str
        pgn_game.headers["Termination"] = reason
        avg_w_time = sum(white_times) / len(white_times) if white_times else 0.0
        avg_b_time = sum(black_times) / len(black_times) if black_times else 0.0

        game_summary = {
            "round": g_idx, "opening": op["name"], "white": w_name, "black": b_name,
            "result": result_str, "winner": winner, "reason": reason,
            "plies": plies, "moves": (plies + 1) // 2,
            "avg_w_time": avg_w_time, "avg_b_time": avg_b_time, "pgn": str(pgn_game),
        }
        games_results.append(game_summary)
        match_logger.info(
            f"Game {g_idx}: {result_str} ({winner}) via {reason} in {(plies+1)//2} moves | "
            f"W_time={avg_w_time:.2f}s B_time={avg_b_time:.2f}s"
        )
        with open(pgn_path, "a", encoding="utf-8") as f_pgn:
            f_pgn.write(str(pgn_game) + "\n\n")

    t_wins = sum(1 for g in games_results if g["winner"] == 'T')
    r_wins = sum(1 for g in games_results if g["winner"] == 'R')
    draws = sum(1 for g in games_results if g["winner"] == 'Draw')
    total = len(games_results)
    t_score = t_wins + 0.5 * draws
    r_score = r_wins + 0.5 * draws

    match_logger.info("=" * 60)
    match_logger.info(f"MINI MATCH COMPLETE: T {t_score} - {r_score} R ({total} games)")
    match_logger.info(f"T Wins: {t_wins}, R Wins: {r_wins}, Draws: {draws}")
    match_logger.info("=" * 60)
    for g in games_results:
        match_logger.info(f"Game {g['round']:2d} | W: {g['white']} vs B: {g['black']} | {g['result']:7s} | {g['opening']}")

    return {"t_score": t_score, "r_score": r_score, "t_wins": t_wins, "r_wins": r_wins, "draws": draws, "total": total}


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def run_experiment_a(model, selfplay_loader, original_loader, device, num_steps=500):
    """A: KL divergence loss for policy."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT A: KL Divergence Policy Loss")
    logger.info("=" * 60)
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=False).get("model", {}), strict=False)
    model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = ChessLossKL(policy_weight=1.0, promo_weight=0.1, wdl_weight=1.0)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    metrics = train_steps(model, selfplay_loader, optimizer, criterion, device, num_steps, amp_dtype, grad_accum_steps=1)
    ckpt_path = RESULT_DIR / "exp_a_kl.pt"
    torch.save({"model": model.state_dict(), "metrics": metrics}, ckpt_path)
    logger.info(f"Experiment A complete. Saved to {ckpt_path}")
    return ckpt_path, metrics


def run_experiment_b(model, selfplay_loader, original_loader, device, num_steps=500):
    """B: 50% self-play + 50% original mixed training."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT B: Mixed Self-Play + Original (50/50)")
    logger.info("=" * 60)
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=False).get("model", {}), strict=False)
    model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = ChessLoss(policy_weight=1.0, promo_weight=0.1, wdl_weight=1.0)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    sp_iter = iter(selfplay_loader)
    orig_iter = iter(original_loader)
    total_loss = 0.0
    total_policy_loss = 0.0
    total_wdl_loss = 0.0
    total_p1 = 0.0
    total_p5 = 0.0
    total_wdl_acc = 0.0
    steps_done = 0
    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    while steps_done < num_steps:
        use_sp = steps_done % 2 == 0
        try:
            x, p, pr, w = next(sp_iter if use_sp else orig_iter)
        except StopIteration:
            if use_sp:
                sp_iter = iter(selfplay_loader)
                x, p, pr, w = next(sp_iter)
            else:
                orig_iter = iter(original_loader)
                x, p, pr, w = next(orig_iter)

        x = x.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        pr = pr.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                p_out, pr_out, w_out = model(x)
                out = criterion(p_out, pr_out, w_out, p, pr, w)
                loss = out.total_loss
            loss.backward()
        else:
            p_out, pr_out, w_out = model(x)
            out = criterion(p_out, pr_out, w_out, p, pr, w)
            loss = out.total_loss
            loss.backward()

        accum_count += 1
        total_loss += out.total_loss.item()
        total_policy_loss += out.policy_loss.item()
        total_wdl_loss += out.wdl_loss.item()
        total_p1 += out.metrics.get("policy_top1_acc", 0.0)
        total_p5 += out.metrics.get("policy_top5_acc", 0.0)
        total_wdl_acc += out.metrics.get("wdl_acc", 0.0)

        if accum_count % 1 == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_done += 1
            accum_count = 0

            if steps_done % 100 == 0:
                logger.info(
                    f"  Step {steps_done}/{num_steps}: Loss={total_loss/steps_done:.4f} "
                    f"(Pol={total_policy_loss/steps_done:.4f}, WDL={total_wdl_loss/steps_done:.4f}) | "
                    f"P1={total_p1/steps_done*100:.2f}% WDL={total_wdl_acc/steps_done*100:.2f}%"
                )

    metrics = {
        "total_loss": total_loss / max(1, steps_done),
        "policy_loss": total_policy_loss / max(1, steps_done),
        "wdl_loss": total_wdl_loss / max(1, steps_done),
        "policy_top1_acc": total_p1 / max(1, steps_done),
        "policy_top5_acc": total_p5 / max(1, steps_done),
        "wdl_acc": total_wdl_acc / max(1, steps_done),
        "steps": steps_done,
    }
    ckpt_path = RESULT_DIR / "exp_b_mixed.pt"
    torch.save({"model": model.state_dict(), "metrics": metrics}, ckpt_path)
    logger.info(f"Experiment B complete. Saved to {ckpt_path}")
    return ckpt_path, metrics


def run_experiment_c(model, selfplay_loader, original_loader, device, num_steps=500):
    """C: Lower LR (5e-6) with gradient accumulation (4 steps, effective ~1.25e-6)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT C: Low LR (5e-6) + Grad Accum (4 steps)")
    logger.info("=" * 60)
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=False).get("model", {}), strict=False)
    model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    criterion = ChessLoss(policy_weight=1.0, promo_weight=0.1, wdl_weight=1.0)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    metrics = train_steps(model, selfplay_loader, optimizer, criterion, device, num_steps, amp_dtype, grad_accum_steps=4)
    ckpt_path = RESULT_DIR / "exp_c_low_lr.pt"
    torch.save({"model": model.state_dict(), "metrics": metrics}, ckpt_path)
    logger.info(f"Experiment C complete. Saved to {ckpt_path}")
    return ckpt_path, metrics


def run_experiment_d(model, selfplay_loader, original_loader, device, num_steps=500):
    """D: Freeze policy head, only train value head (and promo) on self-play."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT D: Freeze Policy Head, Train Value Only")
    logger.info("=" * 60)
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=False).get("model", {}), strict=False)
    model.to(device).train()

    if hasattr(model, 'experts'):
        for expert in model.experts:
            for p in expert.policy_head.parameters():
                p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    criterion = ChessLoss(policy_weight=0.0, promo_weight=0.1, wdl_weight=1.0)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    metrics = train_steps(model, selfplay_loader, optimizer, criterion, device, num_steps, amp_dtype, grad_accum_steps=1)
    ckpt_path = RESULT_DIR / "exp_d_freeze_policy.pt"
    torch.save({"model": model.state_dict(), "metrics": metrics}, ckpt_path)
    logger.info(f"Experiment D complete. Saved to {ckpt_path}")
    return ckpt_path, metrics


def run_experiment_e(model, selfplay_loader, original_loader, device, num_steps=500):
    """E: Temperature smoothing on MCTS visit counts (T=2.0) to make targets softer."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT E: Temperature Smoothing (T=2.0) on Visit Counts")
    logger.info("=" * 60)
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=False).get("model", {}), strict=False)
    model.to(device).train()

    sp_data = selfplay_loader.dataset
    temp = 2.0
    smoothed_policy = sp_data.policy_targets.clone()
    nonzero = smoothed_policy > 0
    smoothed_policy[nonzero] = smoothed_policy[nonzero] ** (1.0 / temp)
    row_sums = smoothed_policy.sum(dim=-1, keepdim=True)
    smoothed_policy = smoothed_policy / row_sums.clamp(min=1e-8)

    smoothed_dataset = torch.utils.data.TensorDataset(
        sp_data.states, smoothed_policy, sp_data.promo_targets, sp_data.wdl_targets
    )
    smoothed_loader = DataLoader(smoothed_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = ChessLoss(policy_weight=1.0, promo_weight=0.1, wdl_weight=1.0)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    metrics = train_steps(model, smoothed_loader, optimizer, criterion, device, num_steps, amp_dtype, grad_accum_steps=1)
    ckpt_path = RESULT_DIR / "exp_e_temp_smooth.pt"
    torch.save({"model": model.state_dict(), "metrics": metrics}, ckpt_path)
    logger.info(f"Experiment E complete. Saved to {ckpt_path}")
    return ckpt_path, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="P4 Self-Play Diagnostic Experiments")
    parser.add_argument("--base-ckpt", type=str, default="/home/jeefy/UniChess/Transformer/runs/stratified_p1_opening/best_model.pt")
    parser.add_argument("--num-selfplay-games", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--t-sims", type=int, default=2400)
    parser.add_argument("--r-sims", type=int, default=800)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-match", action="store_true", help="Skip mini match for quick testing")
    args = parser.parse_args()

    BASE_CKPT = args.base_ckpt
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Base checkpoint: {BASE_CKPT}")

    # Phase 1: Collect self-play data
    logger.info("=" * 60)
    logger.info("Phase 1: Self-Play Data Collection")
    logger.info("=" * 60)
    model = create_transformer("stratified_20m")
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    clean_sd = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    engine = TransformerEngine(model, device=args.device, precision="bf16", use_cpp_mcts=True, mcts_sims=0)
    cpp_mcts = None
    try:
        from unichess_t.search.cpp import load_cpp_mcts
        CppMCTSClass = load_cpp_mcts()
        cpp_mcts = CppMCTSClass()
        cpp_mcts.set_seed(42)
        logger.info("C++ MCTS loaded for self-play data collection")
    except Exception as e:
        logger.warning(f"Failed to load C++ MCTS: {e}")

    positions = collect_selfplay_games(engine, cpp_mcts, num_games=args.num_selfplay_games, simulations=800, batch_size=64)
    logger.info(f"Collected {len(positions)} positions")

    sp_data_path = RESULT_DIR / "shared_selfplay.pt"
    sp_states = np.stack([p[0] for p in positions])
    sp_policy = np.stack([p[1] for p in positions])
    sp_promo = np.array([p[2] for p in positions])
    sp_scalar = np.array([p[3] for p in positions])
    sp_wdl = np.zeros((len(positions), 3), dtype=np.float32)
    sp_wdl[sp_scalar > 0] = [1, 0, 0]
    sp_wdl[sp_scalar < 0] = [0, 0, 1]
    sp_wdl[sp_scalar == 0] = [0, 1, 0]
    torch.save({
        "states": torch.from_numpy(sp_states),
        "policy_targets": torch.from_numpy(sp_policy),
        "promo_targets": torch.tensor(sp_promo, dtype=torch.long),
        "wdl_targets": torch.from_numpy(sp_wdl),
    }, sp_data_path)

    sp_ds = SelfPlayDataset(positions)
    sp_loader = DataLoader(sp_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    logger.info("=" * 60)
    logger.info("Loading original training data")
    logger.info("=" * 60)
    all_shards = get_shards(DATA_DIR)
    train_shards = all_shards[:4]
    _, original_loader = make_loader(train_shards, batch_size=256, num_workers=4, pin_memory=True, shuffle=True)
    logger.info(f"Loaded {len(train_shards)} original shards")

    results = {}
    experiments = [
        ("A_KL", run_experiment_a),
        ("B_Mixed", run_experiment_b),
        ("C_LowLR", run_experiment_c),
        ("D_FreezePolicy", run_experiment_d),
        ("E_TempSmooth", run_experiment_e),
    ]

    for name, runner in experiments:
        logger.info(f"\n{'#' * 60}\nRunning Experiment {name}\n{'#' * 60}")
        model = create_transformer("stratified_20m")
        ckpt_path, train_metrics = runner(model, sp_loader, original_loader, DEVICE, args.num_steps)
        logger.info(f"Training metrics for {name}: {train_metrics}")

        logger.info(f"Running 4-game mini match for {name}...")
        if not args.skip_match:
            match_result = run_mini_match(
                str(ckpt_path), t_sims=args.t_sims, r_sims=args.r_sims,
                num_games=4, log_path=RESULT_DIR / f"match_{name}.log",
                pgn_path=RESULT_DIR / f"match_{name}.pgn",
            )
        else:
            match_result = {"t_score": 0, "r_score": 0, "t_wins": 0, "r_wins": 0, "draws": 0, "total": 0, "skipped": True}
        results[name] = {
            "train_metrics": {k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v for k, v in train_metrics.items()},
            "match_result": match_result,
            "ckpt": str(ckpt_path),
        }

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    summary_path = RESULT_DIR / "diagnostic_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    for name, res in results.items():
        m = res["match_result"]
        logger.info(
            f"{name:20s}: Train Loss={res['train_metrics']['total_loss']:.4f} "
            f"P1={res['train_metrics']['policy_top1_acc']*100:.1f}% "
            f"WDL={res['train_metrics']['wdl_acc']*100:.1f}% | "
            f"Match: T {m['t_score']} - {m['r_score']} R ({m['t_wins']}W/{m['draws']}D/{m['r_wins']}L)"
        )
    logger.info(f"\nDetailed results saved to {summary_path}")


if __name__ == "__main__":
    main()
