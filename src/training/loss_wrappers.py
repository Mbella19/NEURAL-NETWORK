"""Loss wrappers with class-imbalance handling."""
from __future__ import annotations

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


__all__ = ["make_weighted_bce", "FocalLoss"]
