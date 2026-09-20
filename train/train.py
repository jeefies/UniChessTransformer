"""Full training script with AdamW, Cosine/OneCycle LR scheduler, AMP mixed precision,
checkpointing, validation metrics, resume capability, gradient accumulation, and logging.
"""
from __future__ import annotations

import argparse
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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import get_shards, make_loader
from model.loss import ChessLoss
from model.transformer import PRESETS, ChessTransformer, StratifiedChessTransformer, TransformerConfig, create_transformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train UniChessTransformer")
    parser.add_argument("--preset", type=str, default="transformer_20m", choices=list(PRESETS.keys()))
    parser.add_argument("--data-dir", type=str, default="/home/jeefy/UniChess/data/shards_evals")
    parser.add_argument("--num-val-shards", type=int, default=1, help="Number of last shards for validation")
    parser.add_argument("--max-train-shards", type=int, default=None, help="Optional limit on train shards")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile(model)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after max-steps if set")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle"])
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--init-from", type=str, default=None, help="Path to base 20M checkpoint to initialize model weights")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=50, help="Max batches to evaluate during validation")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True, help="DataLoader pin_memory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--reset-lr", action="store_true", help="Reset LR schedule from current step to max-steps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    criterion: ChessLoss,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int = 50,
) -> dict[str, float]:
    model.eval()
    total_losses: list[float] = []
    policy_losses: list[float] = []
    promo_losses: list[float] = []
    wdl_losses: list[float] = []

    accumulated_metrics: dict[str, list[float]] = {}

    for i, (x, p, pr, w) in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        pr = pr.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                p_out, pr_out, w_out = model(x)
                out = criterion(p_out, pr_out, w_out, p, pr, w)
        else:
            p_out, pr_out, w_out = model(x)
            out = criterion(p_out, pr_out, w_out, p, pr, w)

        total_losses.append(out.total_loss.item())
        policy_losses.append(out.policy_loss.item())
        promo_losses.append(out.promo_loss.item())
        wdl_losses.append(out.wdl_loss.item())

        for k, v in out.metrics.items():
            accumulated_metrics.setdefault(k, []).append(v)

    summary = {
        "val_total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "val_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "val_promo_loss": float(np.mean(promo_losses)) if promo_losses else 0.0,
        "val_wdl_loss": float(np.mean(wdl_losses)) if wdl_losses else 0.0,
    }
    for k, v in accumulated_metrics.items():
        summary[f"val_{k}"] = float(np.mean(v))

    return summary


def train():
    args = parse_args()
    set_seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = ckpt_dir / "train_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device} ({device_name})")

    # AMP setup
    amp_enabled = (args.precision in ["bf16", "fp16"]) and (device.type == "cuda")
    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        scaler = None
    elif args.precision == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = None
        scaler = None

    # Load data shards
    all_shards = get_shards(args.data_dir)
    num_val = max(1, args.num_val_shards)
    train_shards = all_shards[:-num_val]
    val_shards = all_shards[-num_val:]
    if args.max_train_shards:
        train_shards = train_shards[:args.max_train_shards]

    print(f"Dataset: {len(train_shards)} train shards, {len(val_shards)} val shards")
    print(f"Data loader config: num_workers={args.num_workers}, pin_memory={args.pin_memory}, precision={args.precision}, grad_accum_steps={args.grad_accum_steps}")
    _, train_loader = make_loader(
        train_shards,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
    )
    _, val_loader = make_loader(
        val_shards,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
    )

    # Initialize model
    model = create_transformer(args.preset).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.preset} | Parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    # Initialize from base checkpoint if requested and not resuming
    if args.init_from and not args.resume:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from checkpoint not found at: {init_path}")
        print(f"Initializing model from base checkpoint: {init_path}")
        raw_model = getattr(model, "_orig_mod", model)
        if isinstance(raw_model, StratifiedChessTransformer):
            raw_model.load_base_checkpoint(init_path)
            print("Successfully loaded base checkpoint into all 3 phase experts (opening, middlegame, endgame).")
        elif isinstance(raw_model, ChessTransformer):
            ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
            sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            clean_sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            raw_model.load_state_dict(clean_sd)
            print("Successfully loaded base checkpoint into ChessTransformer.")
        else:
            raise TypeError(f"Unsupported model type for init_from: {type(raw_model)}")

    # Optimizer & Criterion
    fused = (device.type == "cuda" and hasattr(torch.optim.AdamW, "_step_supports_fused"))
    try:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused)
    except Exception:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    criterion = ChessLoss()

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    if args.max_steps:
        needed_epochs = math.ceil(args.max_steps / steps_per_epoch)
        if needed_epochs > args.epochs:
            args.epochs = needed_epochs
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    # Scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=args.min_lr)
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=max(1, total_steps),
            pct_start=min(0.1, max(0.01, args.warmup_steps / max(1, total_steps))),
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )

    # Resume capability
    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    best_p1_acc = 0.0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model = getattr(model, "_orig_mod", model)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        clean_sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        raw_model.load_state_dict(clean_sd)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            if args.reset_lr:
                for g in optimizer.param_groups:
                    g["lr"] = args.lr
        if "scheduler" in ckpt and not args.reset_lr:
            scheduler.load_state_dict(ckpt["scheduler"])
        elif not args.reset_lr and ckpt.get("step", 0) > 0:
            for _ in range(ckpt.get("step", 0)):
                scheduler.step()
        if scaler and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        start_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_p1_acc = ckpt.get("best_p1_acc", 0.0)
        print(f"Resumed at step {start_step}, epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    global_step = start_step
    print("Beginning training...")

    model.train()
    start_time = time.perf_counter()
    running_loss = 0.0
    running_policy_loss = 0.0
    running_wdl_loss = 0.0
    running_p1 = 0.0
    running_p5 = 0.0
    running_wdl_acc = 0.0
    log_batches = 0

    accum_count = 0
    optimizer.zero_grad(set_to_none=True)

    def get_model_state_dict():
        raw = getattr(model, "_orig_mod", model)
        return raw.state_dict()

    def get_cfg_dict():
        raw = getattr(model, "_orig_mod", model)
        if hasattr(raw, "cfg"):
            if hasattr(raw.cfg, "__dict__"):
                return raw.cfg.__dict__
            if hasattr(raw.cfg, "to_dict"):
                return raw.cfg.to_dict()
        return {}

    batches_to_skip = max(0, (start_step - start_epoch * steps_per_epoch) * args.grad_accum_steps)
    if batches_to_skip > 0:
        print(f"Fast-forwarding {batches_to_skip} batches to resume at step {start_step}...")

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, (x, p, pr, w) in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < batches_to_skip:
                continue
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            pr = pr.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)

            if amp_enabled and amp_dtype == torch.bfloat16:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    p_out, pr_out, w_out = model(x)
                    out = criterion(p_out, pr_out, w_out, p, pr, w)
                    loss = out.total_loss / args.grad_accum_steps
                loss.backward()
            elif amp_enabled and amp_dtype == torch.float16 and scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    p_out, pr_out, w_out = model(x)
                    out = criterion(p_out, pr_out, w_out, p, pr, w)
                    loss = out.total_loss / args.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                p_out, pr_out, w_out = model(x)
                out = criterion(p_out, pr_out, w_out, p, pr, w)
                loss = out.total_loss / args.grad_accum_steps
                loss.backward()

            accum_count += 1
            running_loss += out.total_loss.item()
            running_policy_loss += out.policy_loss.item()
            running_wdl_loss += out.wdl_loss.item()
            running_p1 += out.metrics.get("policy_top1_acc", 0.0)
            running_p5 += out.metrics.get("policy_top5_acc", 0.0)
            running_wdl_acc += out.metrics.get("wdl_acc", 0.0)
            log_batches += 1

            if accum_count % args.grad_accum_steps == 0:
                if scaler is not None:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_step += 1

                # Logging
                if global_step % args.log_every == 0:
                    elapsed = time.perf_counter() - start_time
                    throughput = (log_batches * args.batch_size) / max(1e-5, elapsed)
                    avg_loss = running_loss / log_batches
                    avg_p_loss = running_policy_loss / log_batches
                    avg_w_loss = running_wdl_loss / log_batches
                    avg_p1 = running_p1 / log_batches
                    avg_p5 = running_p5 / log_batches
                    avg_wdl_acc = running_wdl_acc / log_batches
                    current_lr = optimizer.param_groups[0]["lr"]

                    vram_alloc = torch.cuda.memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
                    vram_max = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

                    log_entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "total_loss": round(avg_loss, 4),
                        "policy_loss": round(avg_p_loss, 4),
                        "wdl_loss": round(avg_w_loss, 4),
                        "policy_top1_acc": round(avg_p1, 4),
                        "policy_top5_acc": round(avg_p5, 4),
                        "wdl_acc": round(avg_wdl_acc, 4),
                        "vram_allocated_mb": round(vram_alloc, 1),
                        "vram_max_mb": round(vram_max, 1),
                        "lr": current_lr,
                        "throughput": round(throughput, 1),
                    }
                    print(
                        f"Step {global_step:6d} | Epoch {epoch:2d} | Loss: {avg_loss:6.4f} "
                        f"(Pol: {avg_p_loss:6.4f}, WDL: {avg_w_loss:6.4f}) | "
                        f"P-Top1: {avg_p1*100:5.2f}% | P-Top5: {avg_p5*100:5.2f}% | "
                        f"WDL-Acc: {avg_wdl_acc*100:5.2f}% | VRAM: {vram_alloc:.0f}M/{vram_max:.0f}M | "
                        f"LR: {current_lr:.2e} | Speed: {throughput:6.1f} pos/s"
                    )

                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    running_loss = 0.0
                    running_policy_loss = 0.0
                    running_wdl_loss = 0.0
                    running_p1 = 0.0
                    running_p5 = 0.0
                    running_wdl_acc = 0.0
                    log_batches = 0
                    start_time = time.perf_counter()

                # Validation
                if global_step % args.val_every == 0:
                    print(f"--- Running Validation at Step {global_step} ---")
                    val_metrics = evaluate(
                        model,
                        val_loader,
                        criterion,
                        device,
                        amp_dtype,
                        max_batches=args.val_steps,
                    )
                    print(
                        f"Val Loss: {val_metrics['val_total_loss']:6.4f} "
                        f"(Pol: {val_metrics['val_policy_loss']:6.4f}, WDL: {val_metrics['val_wdl_loss']:6.4f}) | "
                        f"Val P-Top1: {val_metrics.get('val_policy_top1_acc', 0)*100:5.2f}% | "
                        f"Val P-Top5: {val_metrics.get('val_policy_top5_acc', 0)*100:5.2f}% | "
                        f"Val WDL-Acc: {val_metrics.get('val_wdl_acc', 0)*100:5.2f}% | "
                        f"Val Q-MSE: {val_metrics.get('val_q_mse', 0):6.4f}"
                    )

                    val_entry = {"step": global_step, **val_metrics}
                    with open(log_file, "a") as f:
                        f.write(json.dumps(val_entry) + "\n")

                    # Save best checkpoint
                    val_loss = val_metrics["val_total_loss"]
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_ckpt_path = ckpt_dir / "best_model.pt"
                        save_dict = {
                            "step": global_step,
                            "epoch": epoch,
                            "preset": args.preset,
                            "model": get_model_state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict() if scaler else None,
                            "cfg": get_cfg_dict(),
                            "best_val_loss": best_val_loss,
                            "best_p1_acc": val_metrics.get("val_policy_top1_acc", 0.0),
                        }
                        torch.save(save_dict, best_ckpt_path)
                        print(f"Saved new best checkpoint to {best_ckpt_path}")

                    model.train()

                # Periodic saving
                if global_step % args.save_every == 0:
                    step_ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pt"
                    save_dict = {
                        "step": global_step,
                        "epoch": epoch,
                        "preset": args.preset,
                        "model": get_model_state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler else None,
                        "cfg": get_cfg_dict(),
                        "best_val_loss": best_val_loss,
                        "best_p1_acc": best_p1_acc,
                    }
                    torch.save(save_dict, step_ckpt_path)
                    torch.save(save_dict, latest_ckpt_path)
                    print(f"Saved checkpoint to {step_ckpt_path} and {latest_ckpt_path}")

                if args.max_steps and global_step >= args.max_steps:
                    break

        if args.max_steps and global_step >= args.max_steps:
            break

    # Save final model
    final_path = ckpt_dir / "final_model.pt"
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pt"
    final_dict = {
        "step": global_step,
        "epoch": min(epoch, args.epochs - 1) if "epoch" in locals() else args.epochs,
        "preset": args.preset,
        "model": get_model_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "cfg": get_cfg_dict(),
        "best_val_loss": best_val_loss,
        "best_p1_acc": best_p1_acc,
    }
    torch.save(final_dict, final_path)
    torch.save(final_dict, latest_ckpt_path)
    print(f"Training completed at step {global_step}. Final model saved to {final_path} and {latest_ckpt_path}")


if __name__ == "__main__":
    train()
