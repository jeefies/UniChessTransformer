"""Curriculum fine-tuning for StratifiedChessTransformer endgame expert.

Scores endgame positions by difficulty (L_policy + lambda * L_wdl) in chunks,
sorts them from easiest to hardest, and fine-tunes the endgame expert network.
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unichess_t.model.dataset import RECORD_DTYPE, decode_batch, decode_targets, get_shards
from unichess_t.model.loss import ChessLoss
from unichess_t.model.transformer import StratifiedChessTransformer, transformer_20m


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Endgame Curriculum Fine-tuning")
    parser.add_argument("--base-ckpt", type=str, default="runs/stratified_20m/best_model.pt")
    parser.add_argument("--data-dir", type=str, default="/home/jeefy/UniChess/data/shards_evals")
    parser.add_argument("--output-dir", type=str, default="runs/stratified_curriculum")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--scoring-microbatch", type=int, default=2048)
    parser.add_argument("--lambda-val", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_endgame_from_checkpoint(ckpt_path: str | Path, device: torch.device) -> tuple[StratifiedChessTransformer, nn.Module, dict]:
    """Loads StratifiedChessTransformer from ckpt and extracts the endgame expert."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    full_model = StratifiedChessTransformer()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    clean_sd = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    full_model.load_state_dict(clean_sd)

    endgame_expert = full_model.endgame.to(device)
    return full_model, endgame_expert, ckpt if isinstance(ckpt, dict) else {}


def get_endgame_records_generator(shards: list[Path], chunk_size: int):
    """Iterates across shards, filters pieces <= 12, yields numpy record chunks."""
    buffer: list[np.ndarray] = []
    total_buffered = 0

    for shard_path in shards:
        mmap_records = np.memmap(shard_path, dtype=RECORD_DTYPE, mode="r")
        occ = mmap_records["occ_white"] | mmap_records["occ_black"]
        counts = np.bitwise_count(occ)
        endgame_indices = np.nonzero(counts <= 12)[0]

        if len(endgame_indices) == 0:
            continue

        endgame_recs = np.array(mmap_records[endgame_indices])
        buffer.append(endgame_recs)
        total_buffered += len(endgame_recs)

        while total_buffered >= chunk_size:
            concatenated = np.concatenate(buffer, axis=0)
            chunk = concatenated[:chunk_size]
            remaining = concatenated[chunk_size:]
            buffer = [remaining] if len(remaining) > 0 else []
            total_buffered = len(remaining)
            yield chunk

    if total_buffered > 0:
        yield np.concatenate(buffer, axis=0)


@torch.no_grad()
def score_chunk(
    model: nn.Module,
    records: np.ndarray,
    scoring_microbatch: int,
    lambda_val: float,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    """Scores records by difficulty: L_policy + lambda * L_wdl."""
    model.eval()
    n = len(records)
    scores = np.zeros(n, dtype=np.float32)

    for start_idx in range(0, n, scoring_microbatch):
        end_idx = min(start_idx + scoring_microbatch, n)
        batch_recs = records[start_idx:end_idx]

        x = torch.from_numpy(decode_batch(batch_recs)).to(device, non_blocking=True)
        p_tgt, _, w_tgt = decode_targets(batch_recs)
        p_tgt = torch.from_numpy(p_tgt).to(device, non_blocking=True)
        w_tgt = torch.from_numpy(w_tgt).to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                p_logits, _, w_logits = model(x)
                l_p = F.cross_entropy(p_logits, p_tgt, reduction="none")
                l_w = F.cross_entropy(w_logits, w_tgt, reduction="none")
        else:
            p_logits, _, w_logits = model(x)
            l_p = F.cross_entropy(p_logits, p_tgt, reduction="none")
            l_w = F.cross_entropy(w_logits, w_tgt, reduction="none")

        batch_scores = (l_p + lambda_val * l_w).float().cpu().numpy()
        scores[start_idx:end_idx] = batch_scores

    return scores


def save_curriculum_checkpoint(
    output_dir: Path,
    filename: str,
    full_model: StratifiedChessTransformer,
    endgame_expert: nn.Module,
    step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    base_ckpt: dict,
):
    """Updates full_model and ckpt state dict with the refined endgame weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full_model.endgame.load_state_dict(endgame_expert.state_dict())

    new_state_dict = full_model.state_dict()
    save_dict = dict(base_ckpt)
    save_dict.update({
        "step": step,
        "model": new_state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "endgame_curriculum": True,
    })

    save_path = output_dir / filename
    torch.save(save_dict, save_path)
    print(f"Saved checkpoint: {save_path} (step {step})")


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "curriculum_train_log.jsonl"

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

    # Load base checkpoint & extract endgame expert
    print(f"Loading base checkpoint: {args.base_ckpt}")
    full_model, endgame_expert, base_ckpt = load_endgame_from_checkpoint(args.base_ckpt, device)
    param_count = sum(p.numel() for p in endgame_expert.parameters() if p.requires_grad)
    print(f"Extracted endgame expert: {param_count:,} parameters ({param_count/1e6:.2f}M)")

    # Shards
    shards = get_shards(args.data_dir)
    print(f"Found {len(shards)} shards in {args.data_dir}")

    # Optimizer & Scheduler & Loss
    fused = (device.type == "cuda" and hasattr(torch.optim.AdamW, "_step_supports_fused"))
    try:
        optimizer = AdamW(endgame_expert.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused)
    except Exception:
        optimizer = AdamW(endgame_expert.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.min_lr)
    criterion = ChessLoss(policy_weight=1.0, promo_weight=0.1, wdl_weight=args.lambda_val)

    # Training loop across chunks
    step = 0
    start_time = time.time()
    recent_losses: list[float] = []
    recent_p1: list[float] = []
    recent_p5: list[float] = []
    recent_wdl: list[float] = []

    print(f"Starting endgame curriculum training: max_steps={args.max_steps}, batch_size={args.batch_size}, chunk_size={args.chunk_size}")

    chunk_gen = get_endgame_records_generator(shards, args.chunk_size)

    while step < args.max_steps:
        try:
            chunk = next(chunk_gen)
        except StopIteration:
            # Re-cycle over shards if max_steps not reached
            chunk_gen = get_endgame_records_generator(shards, args.chunk_size)
            chunk = next(chunk_gen)

        print(f"\n--- Scoring chunk of {len(chunk)} endgame records ---")
        t_score_start = time.time()
        difficulty_scores = score_chunk(
            endgame_expert,
            chunk,
            args.scoring_microbatch,
            args.lambda_val,
            device,
            amp_dtype,
        )
        t_score_end = time.time()
        min_s = float(np.min(difficulty_scores))
        med_s = float(np.median(difficulty_scores))
        max_s = float(np.max(difficulty_scores))
        print(f"Scored {len(chunk)} records in {t_score_end - t_score_start:.2f}s | Difficulty min: {min_s:.3f}, med: {med_s:.3f}, max: {max_s:.3f}")

        # Sort chunk by difficulty (curriculum: easy -> hard)
        sorted_indices = np.argsort(difficulty_scores)
        sorted_chunk = chunk[sorted_indices]

        # Step through sorted minibatches
        endgame_expert.train()
        n_records = len(sorted_chunk)
        for b_start in range(0, n_records - args.batch_size + 1, args.batch_size):
            if step >= args.max_steps:
                break

            b_records = sorted_chunk[b_start:b_start + args.batch_size]

            x_np = decode_batch(b_records)
            p_np, pr_np, w_np = decode_targets(b_records)

            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            p_tgt = torch.from_numpy(p_np).to(device, non_blocking=True)
            pr_tgt = torch.from_numpy(pr_np).to(device, non_blocking=True)
            w_tgt = torch.from_numpy(w_np).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    p_out, pr_out, w_out = endgame_expert(x)
                    loss_out = criterion(p_out, pr_out, w_out, p_tgt, pr_tgt, w_tgt)

                if scaler is not None:
                    scaler.scale(loss_out.total_loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(endgame_expert.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_out.total_loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(endgame_expert.parameters(), args.grad_clip)
                    optimizer.step()
            else:
                p_out, pr_out, w_out = endgame_expert(x)
                loss_out = criterion(p_out, pr_out, w_out, p_tgt, pr_tgt, w_tgt)
                loss_out.total_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(endgame_expert.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            step += 1

            recent_losses.append(loss_out.total_loss.item())
            recent_p1.append(loss_out.metrics.get("policy_top1_acc", 0.0))
            recent_p5.append(loss_out.metrics.get("policy_top5_acc", 0.0))
            recent_wdl.append(loss_out.metrics.get("wdl_acc", 0.0))

            if step % args.log_every == 0 or step == args.max_steps:
                avg_loss = np.mean(recent_losses)
                avg_p1 = np.mean(recent_p1)
                avg_p5 = np.mean(recent_p5)
                avg_wdl = np.mean(recent_wdl)
                recent_losses.clear()
                recent_p1.clear()
                recent_p5.clear()
                recent_wdl.clear()

                elapsed = time.time() - start_time
                lr_curr = scheduler.get_last_lr()[0]
                throughput = (step * args.batch_size) / max(1e-3, elapsed)

                print(
                    f"Step {step:6d}/{args.max_steps} | Loss: {avg_loss:.4f} | "
                    f"P@1: {avg_p1*100:5.2f}% | P@5: {avg_p5*100:5.2f}% | "
                    f"WDL: {avg_wdl*100:5.2f}% | LR: {lr_curr:.2e} | "
                    f"{throughput:.0f} pos/s"
                )

                log_entry = {
                    "step": step,
                    "loss": float(avg_loss),
                    "policy_top1_acc": float(avg_p1),
                    "policy_top5_acc": float(avg_p5),
                    "wdl_acc": float(avg_wdl),
                    "lr": float(lr_curr),
                    "throughput": float(throughput),
                    "elapsed": float(elapsed),
                }
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            if step % args.save_every == 0:
                save_curriculum_checkpoint(
                    output_dir,
                    f"step_{step}.pt",
                    full_model,
                    endgame_expert,
                    step,
                    optimizer,
                    scheduler,
                    base_ckpt,
                )

    # Final checkpoint save
    save_curriculum_checkpoint(
        output_dir,
        "latest_checkpoint.pt",
        full_model,
        endgame_expert,
        step,
        optimizer,
        scheduler,
        base_ckpt,
    )
    save_curriculum_checkpoint(
        output_dir,
        "best_model.pt",
        full_model,
        endgame_expert,
        step,
        optimizer,
        scheduler,
        base_ckpt,
    )
    print("Curriculum endgame training run completed successfully.")


if __name__ == "__main__":
    main()
