"""Loss wrappers with class-imbalance handling."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def make_weighted_bce(pos_weight: torch.Tensor | float | None = None) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    weight_tensor = pos_weight if isinstance(pos_weight, torch.Tensor) else torch.tensor(pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def configure_binary_loss(
    targets: torch.Tensor,
    imbalance_threshold: float = 0.35,
    verbose: bool = True,
) -> Tuple[nn.Module, str]:
    """Configure BCE or FocalLoss based on class imbalance.

    For balanced tasks (~50% positive), standard BCE works better and is more stable.
    For imbalanced tasks, FocalLoss focuses on hard examples and prevents bias.

    Args:
        targets: Binary target tensor (values in [0, 1])
        imbalance_threshold: Use FocalLoss if pos_rate < threshold or > 1-threshold
        verbose: Whether to print configuration details

    Returns:
        Tuple of (loss_fn, description_string)
    """
    # Compute positive rate
    positives = (targets > 0.5).sum().item()
    total = targets.numel()
    pos_rate = positives / total if total > 0 else 0.5

    is_imbalanced = pos_rate < imbalance_threshold or pos_rate > (1 - imbalance_threshold)

    if is_imbalanced:
        # Use FocalLoss for imbalanced data
        # Dynamic alpha based on class imbalance
        if pos_rate < 0.5:
            alpha = max(0.25, pos_rate)  # Weight rare positive class
        else:
            alpha = max(0.25, 1 - pos_rate)  # Weight rare negative class

        # Higher gamma for extreme imbalance
        gamma = 3.0 if pos_rate < 0.1 or pos_rate > 0.9 else 2.0

        loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        desc = f"FocalLoss(alpha={alpha:.3f}, gamma={gamma:.1f}) - pos_rate={pos_rate:.2%} (IMBALANCED)"
    else:
        # Use BCE for balanced data (more stable, faster convergence)
        loss_fn = nn.BCEWithLogitsLoss()
        desc = f"BCEWithLogitsLoss - pos_rate={pos_rate:.2%} (BALANCED)"

    if verbose:
        print(f"  {desc}")

    return loss_fn, desc


__all__ = ["make_weighted_bce", "FocalLoss", "configure_binary_loss"]
