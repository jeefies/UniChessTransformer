#!/usr/bin/env python3
"""Gumbel AlphaZero self-play RL pipeline with C++ MCTS tree reuse."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.encoding import encode
from core.moves import move_to_index, move_to_promo_index
from engine.engine import TransformerEngine
from model.loss import ChessLoss
from model.transformer import create_transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("GumbelSelfPlay")


class SelfPlayDataset(Dataset):
    def __init__(self, states, policy_targets, promo_targets, wdl_targets):
        self.states = torch.from_numpy(np.stack(states)).float()
        self.policy_targets = torch.from_numpy(np.stack(policy_targets)).float()
        self.promo_targets = torch.tensor(promo_targets, dtype=torch.long)
        self.wdl_targets = torch.from_numpy(np.stack(wdl_targets)).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policy_targets[idx], self.promo_targets[idx], self.wdl_targets[idx]


def evaluate_tensor_callback(engine: TransformerEngine):
    @torch.no_grad()
    def evaluator(x: torch.Tensor):
        return engine.evaluate_tensor(x)
    return evaluator


def play_game_cpp(
    engine: TransformerEngine,
    cpp_mcts,
    simulations: int = 800,
    batch_size: int = 64,
    add_noise: bool = True,
    max_plies: int = 500,
) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
    board = chess.Board()
    game_data: list[tuple[np.ndarray, np.ndarray, int, float, chess.Move]] = []

    evaluator = evaluate_tensor_callback(engine)
    first_move = True

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        fen = board.fen()

        if first_move:
            best_move, metrics = cpp_mcts.search(
                fen, evaluator,
                simulations=simulations,
                batch_size=batch_size,
                add_noise=add_noise,
                temperature=0.0,
            )
            first_move = False
        else:
            best_move, metrics = cpp_mcts.search(
                fen, evaluator,
                simulations=simulations,
                batch_size=batch_size,
                add_noise=False,
                temperature=0.0,
                reuse=True,
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
    if outcome is not None and outcome.winner is not None:
        z = 1.0 if outcome.winner == chess.WHITE else -1.0
    else:
        z = 0.0

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


def play_game_python(
    engine: TransformerEngine,
    simulations: int = 800,
    batch_size: int = 128,
    add_noise: bool = True,
    max_plies: int = 500,
) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
    board = chess.Board()
    game_data: list[tuple[np.ndarray, np.ndarray, int, float, chess.Move]] = []

    from search.mcts import MCTS, MCTSConfig
    mcts_cfg = MCTSConfig(
        simulations=simulations,
        batch_size=batch_size,
        temperature=0.0,
        dirichlet_eps=0.25 if add_noise else 0.0,
    )
    mcts = MCTS(engine.evaluate_batch, cfg=mcts_cfg, tablebase=engine.tablebase, rng=engine.np_rng)
    root = None

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mv, root = mcts.best_move(
            board,
            simulations=simulations,
            temperature=0.0,
            root=root,
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
    if outcome is not None and outcome.winner is not None:
        z = 1.0 if outcome.winner == chess.WHITE else -1.0
    else:
        z = 0.0

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


def collect_selfplay_games(
    engine: TransformerEngine,
    cpp_mcts,
    num_games: int,
    simulations: int = 800,
    batch_size: int = 64,
    add_noise: bool = True,
    max_plies: int = 500,
) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
    all_positions: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    use_cpp = cpp_mcts is not None

    for game_idx in range(num_games):
        game_start = time.time()
        if use_cpp:
            try:
                game_data = play_game_cpp(
                    engine, cpp_mcts,
                    simulations=simulations,
                    batch_size=batch_size,
                    add_noise=add_noise,
                    max_plies=max_plies,
                )
            except Exception as e:
                logger.warning(f"C++ MCTS game {game_idx} failed ({e}), falling back to Python MCTS")
                cpp_mcts.reset()
                game_data = play_game_python(
                    engine,
                    simulations=simulations,
                    batch_size=batch_size,
                    add_noise=add_noise,
                    max_plies=max_plies,
                )
        else:
            game_data = play_game_python(
                engine,
                simulations=simulations,
                batch_size=batch_size,
                add_noise=add_noise,
                max_plies=max_plies,
            )

        all_positions.extend(game_data)
        game_elapsed = time.time() - game_start
        logger.info(
            f"Game {game_idx+1}/{num_games}: {len(game_data)} plies, {game_elapsed:.1f}s "
            f"(total positions: {len(all_positions)})"
        )

    return all_positions


def train_on_selfplay(
    model: nn.Module,
    positions: list[tuple[np.ndarray, np.ndarray, int, float]],
    epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_steps: int | None = None,
    device: torch.device = torch.device("cuda"),
) -> dict:
    if not positions:
        raise ValueError("No positions to train on")

    states, policy_targets, promo_targets, scalar_values = zip(*positions)

    wdl_targets = []
    for sv in scalar_values:
        if sv > 0:
            wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif sv < 0:
            wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        wdl_targets.append(wdl)

    dataset = SelfPlayDataset(states, policy_targets, promo_targets, wdl_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ChessLoss(policy_weight=1.0, promo_weight=0.1, wdl_weight=1.0)

    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    total_loss = 0.0
    total_policy_loss = 0.0
    total_wdl_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_wdl_loss = 0.0
        epoch_batches = 0

        for x, p, pr, w in loader:
            if max_steps is not None and num_batches >= max_steps:
                break

            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            pr = pr.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    p_out, pr_out, w_out = model(x)
                    out = criterion(p_out, pr_out, w_out, p, pr, w)
                    loss = out.total_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                p_out, pr_out, w_out = model(x)
                out = criterion(p_out, pr_out, w_out, p, pr, w)
                loss = out.total_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += out.total_loss.item()
            epoch_policy_loss += out.policy_loss.item()
            epoch_wdl_loss += out.wdl_loss.item()
            epoch_batches += 1
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(1, epoch_batches)
        avg_epoch_policy = epoch_policy_loss / max(1, epoch_batches)
        avg_epoch_wdl = epoch_wdl_loss / max(1, epoch_batches)

        total_loss += avg_epoch_loss
        total_policy_loss += avg_epoch_policy
        total_wdl_loss += avg_epoch_wdl

        logger.info(
            f"Epoch {epoch+1}/{epochs}: Loss={avg_epoch_loss:.4f} "
            f"(Policy={avg_epoch_policy:.4f}, WDL={avg_epoch_wdl:.4f})"
        )

        if max_steps is not None and num_batches >= max_steps:
            break

    return {
        "total_loss": total_loss / max(1, num_batches),
        "policy_loss": total_policy_loss / max(1, num_batches),
        "wdl_loss": total_wdl_loss / max(1, num_batches),
        "num_batches": num_batches,
    }


def save_checkpoint(model: nn.Module, save_path: str | Path, preset: str, metrics: dict):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = getattr(model, "_orig_mod", model)
    if hasattr(raw_model, "cfg"):
        cfg_dict = raw_model.cfg.to_dict() if hasattr(raw_model.cfg, "to_dict") else raw_model.cfg.__dict__
    else:
        cfg_dict = {}

    save_dict = {
        "step": metrics.get("step", 0),
        "epoch": metrics.get("epoch", 0),
        "preset": preset,
        "model": raw_model.state_dict(),
        "cfg": cfg_dict,
        "best_val_loss": metrics.get("loss", 0.0),
    }
    torch.save(save_dict, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Gumbel AlphaZero Self-Play RL Pipeline")
    parser.add_argument("--ckpt", type=str, default="/home/jeefy/UniChess/Transformer/runs/stratified_p1_opening/best_model.pt")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--selfplay-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-plies", type=int, default=500)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/stratified_p4_selfplay")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    args = parser.parse_args()

    if args.fast:
        args.num_games = min(args.num_games, 10)
        logger.info("Fast mode: reducing to %d games", args.num_games)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading model from {ckpt_path}")
    model = create_transformer("stratified_20m")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    clean_sd = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    engine = TransformerEngine(
        model,
        device=args.device,
        precision="bf16",
        use_cpp_mcts=True,
        mcts_sims=0,
    )

    cpp_mcts = None
    try:
        from search.cpp import load_cpp_mcts
        CppMCTSClass = load_cpp_mcts()
        cpp_mcts = CppMCTSClass()
        if args.seed is not None:
            cpp_mcts.set_seed(args.seed)
        logger.info("C++ MCTS loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load C++ MCTS: {e}. Will use Python MCTS fallback.")

    logger.info("=" * 60)
    logger.info(f"Phase 1: Self-Play Data Collection ({args.num_games} games)")
    logger.info("=" * 60)

    positions = collect_selfplay_games(
        engine=engine,
        cpp_mcts=cpp_mcts,
        num_games=args.num_games,
        simulations=args.sims,
        batch_size=args.batch_size,
        add_noise=not args.no_noise,
        max_plies=args.max_plies,
    )

    logger.info(f"Collected {len(positions)} positions from {args.num_games} games")

    if len(positions) == 0:
        logger.error("No positions collected. Exiting.")
        sys.exit(1)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_data_path = checkpoint_dir / "selfplay_data.pt"
    states = np.stack([p[0] for p in positions])
    policy_targets = np.stack([p[1] for p in positions])
    promo_targets = np.array([p[2] for p in positions])
    scalar_values = np.array([p[3] for p in positions])

    wdl_targets = np.zeros((len(positions), 3), dtype=np.float32)
    wdl_targets[scalar_values > 0] = [1, 0, 0]
    wdl_targets[scalar_values < 0] = [0, 0, 1]
    wdl_targets[scalar_values == 0] = [0, 1, 0]

    torch.save({
        "states": torch.from_numpy(states),
        "policy_targets": torch.from_numpy(policy_targets),
        "promo_targets": torch.from_numpy(promo_targets),
        "wdl_targets": torch.from_numpy(wdl_targets),
        "scalar_values": torch.from_numpy(scalar_values),
    }, raw_data_path)
    logger.info(f"Saved raw self-play data to {raw_data_path}")

    logger.info("=" * 60)
    logger.info(f"Phase 2: Training on Self-Play Data ({args.epochs} epochs)")
    logger.info("=" * 60)

    model.train()
    train_metrics = train_on_selfplay(
        model=model,
        positions=positions,
        epochs=args.epochs,
        batch_size=args.selfplay_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        device=device,
    )

    logger.info("=" * 60)
    logger.info("Phase 3: Saving Updated Model")
    logger.info("=" * 60)

    best_model_path = checkpoint_dir / "best_model.pt"
    save_checkpoint(model, best_model_path, preset="stratified_20m", metrics={
        "step": 0,
        "epoch": args.epochs,
        "loss": train_metrics["total_loss"],
    })

    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    save_checkpoint(model, latest_path, preset="stratified_20m", metrics={
        "step": 0,
        "epoch": args.epochs,
        "loss": train_metrics["total_loss"],
    })

    logger.info("=" * 60)
    logger.info("SELF-PLAY PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Games played: {args.num_games}")
    logger.info(f"Positions collected: {len(positions)}")
    logger.info(f"Training epochs: {args.epochs}")
    logger.info(f"Training steps: {train_metrics['num_batches']}")
    logger.info(f"Final loss: {train_metrics['total_loss']:.4f}")
    logger.info(f"Model saved to: {best_model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
