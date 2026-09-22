"""Dedicated pretraining script for P3 Stage 1 & 2: Stratified Chess Transformer with Moves-Left Head (MLH).

Base checkpoint: runs/stratified_middlegame_curriculum/best_model.pt
Uses StratifiedChessTransformer with return_mlh=True.
Targets:
    - Policy (soft CE)
    - Promotion (CE)
    - WDL (soft CE)
    - Moves-Left Head: target_mlh = (2.0 * piece_counts + 20.0 * (1.0 - (wdl[:, 0] - wdl[:, 2]).abs())) / 100.0
      Loss: mlh_weight * F.smooth_l1_loss(mlh, target_mlh)
Multi-shard streaming training across 10,000 steps with batch_size 512 (~5.12M positions).
Saves checkpoints to runs/stratified_p3_pretrain/best_model.pt and latest_checkpoint.pt.
Cleans intermediate non-milestone checkpoints to prevent disk bloat.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import get_shards, make_loader
from model.loss import ChessLoss
from model.transformer import StratifiedChessTransformer, stratified_20m


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="P3 Stage Pretraining with Moves-Left Head (MLH)")
    parser.add_argument("--base-ckpt", type=str, default="runs/stratified_middlegame_curriculum/best_model.pt")
    parser.add_argument("--data-dir", type=str, default="/home/jeefy/UniChess/data/shards_evals")
    parser.add_argument("--output-dir", type=str, default="runs/stratified_p3_pretrain")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mlh-weight", type=float, default=0.05)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_checkpoint(
    output_dir: Path,
    filename: str,
    model: StratifiedChessTransformer,
    step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    base_ckpt: dict,
    best_loss: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = raw_model.state_dict()

    save_dict = dict(base_ckpt) if base_ckpt else {}
    save_dict.update({
        "step": step,
        "preset": "stratified_20m",
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "p3_pretrain": True,
        "has_mlh": True,
    })

    save_path = output_dir / filename
    torch.save(save_dict, save_path)
    print(f"Saved checkpoint: {save_path} (step {step})")


def clean_intermediate_checkpoints(output_dir: Path, keep_steps: set[int]):
    """Removes intermediate step checkpoints not in keep_steps to avoid disk bloat."""
    for p in output_dir.glob("step_*.pt"):
        try:
            step_num = int(p.stem.split("_")[1])
            if step_num not in keep_steps:
                p.unlink()
        except (ValueError, IndexError):
            pass


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "p3_train_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device} ({device_name})")

    # Precision setup
    if args.precision == "bf16" and device.type == "cuda":
        amp_dtype = torch.bfloat16
        scaler = None
    elif args.precision == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = None
        scaler = None

    # Load base model & weights
    print(f"Instantiating StratifiedChessTransformer and loading base checkpoint: {args.base_ckpt}")
    model = stratified_20m()

    base_path = Path(args.base_ckpt)
    base_ckpt = {}
    if base_path.exists():
        loaded_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
        base_ckpt = loaded_ckpt if isinstance(loaded_ckpt, dict) else {}
        model.load_base_checkpoint(base_path)
        print("Base checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Base checkpoint not found at: {base_path}")

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # Data Loader
    shards = get_shards(args.data_dir)
    print(f"Found {len(shards)} shards in {args.data_dir}")
    _, loader = make_loader(
        shards,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    data_iter = iter(loader)

    # Optimizer, Scheduler, Loss
    fused = (device.type == "cuda" and hasattr(torch.optim.AdamW, "_step_supports_fused"))
    try:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused)
    except Exception:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps), eta_min=args.min_lr)
    criterion = ChessLoss(
        policy_weight=1.0,
        promo_weight=0.1,
        wdl_weight=1.0,
        mlh_weight=args.mlh_weight,
    )

    # Milestone steps to retain
    milestone_steps = {5000, 10000}

    # Tracking
    step = 0
    best_loss = float("inf")
    start_time = time.time()
    recent_losses: list[float] = []
    recent_policy_losses: list[float] = []
    recent_wdl_losses: list[float] = []
    recent_mlh_losses: list[float] = []
    recent_p1: list[float] = []
    recent_wdl_acc: list[float] = []

    print(f"Starting P3 pretraining: max_steps={args.max_steps}, batch_size={args.batch_size}, lr={args.lr} -> {args.min_lr}")

    model.train()
    while step < args.max_steps:
        try:
            x, p_tgt, pr_tgt, w_tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, p_tgt, pr_tgt, w_tgt = next(data_iter)

        x = x.to(device, non_blocking=True)
        p_tgt = p_tgt.to(device, non_blocking=True)
        pr_tgt = pr_tgt.to(device, non_blocking=True)
        w_tgt = w_tgt.to(device, non_blocking=True)

        # Dynamic pseudo moves-left target
        # target_mlh = (2.0 * piece_counts + 20.0 * (1.0 - (wdl[:, 0] - wdl[:, 2]).abs())) / 100.0
        piece_counts = x[:, :12].sum(dim=(1, 2, 3))
        q_target = (w_tgt[:, 0] - w_tgt[:, 2]).abs()
        target_mlh = (2.0 * piece_counts + 20.0 * (1.0 - q_target)) / 100.0

        optimizer.zero_grad(set_to_none=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                p_out, pr_out, w_out, mlh_out = model(x, return_mlh=True)
                loss_out = criterion(
                    p_out,
                    pr_out,
                    w_out,
                    p_tgt,
                    pr_tgt,
                    w_tgt,
                    mlh_logits=mlh_out,
                    mlh_target=target_mlh,
                )

            if scaler is not None:
                scaler.scale(loss_out.total_loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_out.total_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        else:
            p_out, pr_out, w_out, mlh_out = model(x, return_mlh=True)
            loss_out = criterion(
                p_out,
                pr_out,
                w_out,
                p_tgt,
                pr_tgt,
                w_tgt,
                mlh_logits=mlh_out,
                mlh_target=target_mlh,
            )
            loss_out.total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        scheduler.step()
        step += 1

        recent_losses.append(loss_out.total_loss.item())
        recent_policy_losses.append(loss_out.policy_loss.item())
        recent_wdl_losses.append(loss_out.wdl_loss.item())
        if loss_out.mlh_loss is not None:
            recent_mlh_losses.append(loss_out.mlh_loss.item())
        recent_p1.append(loss_out.metrics.get("policy_top1_acc", 0.0))
        recent_wdl_acc.append(loss_out.metrics.get("wdl_acc", 0.0))

        if step % args.log_every == 0 or step == args.max_steps:
            avg_loss = float(np.mean(recent_losses))
            avg_pol = float(np.mean(recent_policy_losses))
            avg_wdl = float(np.mean(recent_wdl_losses))
            avg_mlh = float(np.mean(recent_mlh_losses)) if recent_mlh_losses else 0.0
            avg_p1 = float(np.mean(recent_p1))
            avg_wacc = float(np.mean(recent_wdl_acc))

            recent_losses.clear()
            recent_policy_losses.clear()
            recent_wdl_losses.clear()
            recent_mlh_losses.clear()
            recent_p1.clear()
            recent_wdl_acc.clear()

            elapsed = time.time() - start_time
            lr_curr = scheduler.get_last_lr()[0]
            throughput = (step * args.batch_size) / max(1e-3, elapsed)

            print(
                f"Step {step:6d}/{args.max_steps} | Loss: {avg_loss:.4f} (Pol: {avg_pol:.3f}, WDL: {avg_wdl:.3f}, MLH: {avg_mlh:.4f}) | "
                f"P@1: {avg_p1*100:5.2f}% | WDL: {avg_wacc*100:5.2f}% | LR: {lr_curr:.2e} | "
                f"{throughput:.0f} pos/s"
            )

            log_entry = {
                "step": step,
                "loss": avg_loss,
                "policy_loss": avg_pol,
                "wdl_loss": avg_wdl,
                "mlh_loss": avg_mlh,
                "policy_top1_acc": avg_p1,
                "wdl_acc": avg_wacc,
                "lr": float(lr_curr),
                "throughput": float(throughput),
                "elapsed": float(elapsed),
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    output_dir,
                    "best_model.pt",
                    model,
                    step,
                    optimizer,
                    scheduler,
                    base_ckpt,
                    best_loss,
                )

        if step % args.save_every == 0:
            if step in milestone_steps:
                save_checkpoint(
                    output_dir,
                    f"step_{step}.pt",
                    model,
                    step,
                    optimizer,
                    scheduler,
                    base_ckpt,
                    best_loss,
                )
            clean_intermediate_checkpoints(output_dir, milestone_steps)

    # Save latest checkpoint
    save_checkpoint(
        output_dir,
        "latest_checkpoint.pt",
        model,
        step,
        optimizer,
        scheduler,
        base_ckpt,
        best_loss,
    )
    clean_intermediate_checkpoints(output_dir, milestone_steps)
    print(f"P3 Pretraining complete! Final best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
