"""Loss function for Chess Transformer: Policy CrossEntropy/KL, WDL CrossEntropy, and Promotion CrossEntropy."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    promo_loss: torch.Tensor
    wdl_loss: torch.Tensor
    metrics: dict[str, float]
    mlh_loss: torch.Tensor | None = None


class ChessLoss(nn.Module):
    """Combined loss module for policy (soft CE / KL), promotion (CE), and WDL value (soft CE)."""

    def __init__(
        self,
        policy_weight: float = 1.0,
        promo_weight: float = 0.1,
        wdl_weight: float = 1.0,
        mlh_weight: float = 0.05,
        policy_loss_type: str = "cross_entropy",
    ):
        super().__init__()
        self.policy_weight = policy_weight
        self.promo_weight = promo_weight
        self.wdl_weight = wdl_weight
        self.mlh_weight = mlh_weight
        if policy_loss_type not in ("cross_entropy", "kl_divergence"):
            raise ValueError(f"Unsupported policy_loss_type: {policy_loss_type}")
        self.policy_loss_type = policy_loss_type

    def forward(
        self,
        policy_logits: torch.Tensor,
        promo_logits: torch.Tensor,
        value_wdl: torch.Tensor,
        policy_target: torch.Tensor,
        promo_target: torch.Tensor,
        wdl_target: torch.Tensor,
        mlh_logits: torch.Tensor | None = None,
        mlh_target: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute loss and accuracy metrics.

        Args:
            policy_logits: (B, 4096)
            promo_logits: (B, 4)
            value_wdl: (B, 3)
            policy_target: (B, 4096) soft probability targets
            promo_target: (B,) promotion targets with -100 for non-promotions
            wdl_target: (B, 3) soft WDL probability targets
            mlh_logits: (B,) predicted moves-left logits (optional)
            mlh_target: (B,) target moves-left pseudo values (optional)
        """
        # 1. Policy Loss
        if self.policy_loss_type == "kl_divergence":
            log_pred = F.log_softmax(policy_logits, dim=-1)
            loss_policy = F.kl_div(log_pred, policy_target, reduction="batchmean", log_target=False)
        else:
            loss_policy = F.cross_entropy(policy_logits, policy_target)

        # 2. Promotion Loss (CrossEntropy on valid promotion targets)
        valid_promo = (promo_target != -100)
        if valid_promo.any():
            loss_promo = F.cross_entropy(promo_logits[valid_promo], promo_target[valid_promo])
        else:
            loss_promo = torch.tensor(0.0, device=promo_logits.device, dtype=promo_logits.dtype)

        # 3. WDL Value Loss (CrossEntropy on soft probability targets)
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

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(
                policy_logits,
                promo_logits,
                value_wdl,
                policy_target,
                promo_target,
                wdl_target,
            )
            if loss_mlh is not None:
                metrics["mlh_loss"] = loss_mlh.item()
                metrics["mlh_mae"] = F.l1_loss(mlh_logits, mlh_target).item()

        return LossOutput(
            total_loss=total_loss,
            policy_loss=loss_policy,
            promo_loss=loss_promo,
            wdl_loss=loss_wdl,
            metrics=metrics,
            mlh_loss=loss_mlh,
        )


def compute_metrics(
    policy_logits: torch.Tensor,
    promo_logits: torch.Tensor,
    value_wdl: torch.Tensor,
    policy_target: torch.Tensor,
    promo_target: torch.Tensor,
    wdl_target: torch.Tensor,
) -> dict[str, float]:
    """Compute validation / monitoring metrics."""
    metrics: dict[str, float] = {}

    # Policy Top-1 Accuracy: predicted top move matches target top move
    pred_top1 = policy_logits.argmax(dim=-1)
    target_top1 = policy_target.argmax(dim=-1)
    # Only evaluate where target has valid moves
    valid_policy = policy_target.sum(dim=-1) > 0
    if valid_policy.any():
        p1_acc = (pred_top1[valid_policy] == target_top1[valid_policy]).float().mean().item()
        metrics["policy_top1_acc"] = p1_acc

        # Top-5 Accuracy: predicted move is among top-5 target moves with non-zero probability
        _, top5_target_idx = torch.topk(policy_target, k=min(5, policy_target.shape[-1]), dim=-1)
        in_top5 = (pred_top1.unsqueeze(-1) == top5_target_idx).any(dim=-1)
        p5_acc = in_top5[valid_policy].float().mean().item()
        metrics["policy_top5_acc"] = p5_acc

    # Promotion Accuracy
    valid_promo = (promo_target != -100)
    if valid_promo.any():
        pred_promo = promo_logits[valid_promo].argmax(dim=-1)
        promo_acc = (pred_promo == promo_target[valid_promo]).float().mean().item()
        metrics["promo_acc"] = promo_acc

    # WDL Accuracy & MSE
    pred_wdl = value_wdl.argmax(dim=-1)
    target_wdl = wdl_target.argmax(dim=-1)
    metrics["wdl_acc"] = (pred_wdl == target_wdl).float().mean().item()

    pred_probs = F.softmax(value_wdl, dim=-1)
    metrics["wdl_mse"] = F.mse_loss(pred_probs, wdl_target).item()

    # Scalar Value Error: Q = P(win) - P(loss)
    pred_q = pred_probs[:, 0] - pred_probs[:, 2]
    target_q = wdl_target[:, 0] - wdl_target[:, 2]
    metrics["q_mse"] = F.mse_loss(pred_q, target_q).item()

    return metrics
