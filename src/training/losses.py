"""Custom trading-aware losses (Phase 6.3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance in binary classification.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Focal Loss = -α * (1 - p_t)^γ * log(p_t)

    Where:
    - p_t is the probability of the correct class
    - α balances positive/negative examples (default 0.25)
    - γ focuses on hard examples (default 2.0)

    This loss down-weights easy examples and focuses training on hard negatives,
    which is especially useful for imbalanced datasets.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples.
                   Set to None for no weighting.
            gamma: Focusing parameter γ ≥ 0. Higher values give more weight to hard examples.
                   γ=0 reduces to standard cross-entropy.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs (before sigmoid), shape [batch_size] or [batch_size, 1]
            targets: Binary targets (0 or 1), shape [batch_size] or [batch_size, 1]

        Returns:
            Focal loss value
        """
        # Ensure proper shapes
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute p_t (probability of true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Apply focal term
        loss = focal_term * bce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def sharpe_ratio_loss(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = returns.mean()
    std = returns.std(unbiased=False)
    return -(mean / (std + eps))


def directional_accuracy_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.sign(predictions)
    return ((preds != torch.sign(targets)).float()).mean()


def risk_adjusted_return_loss(returns: torch.Tensor, risk_free: float = 0.0, penalty: float = 0.5) -> torch.Tensor:
    excess = returns - risk_free
    downside = torch.clamp(-excess, min=0)
    return -(excess.mean() - penalty * downside.mean())


def multi_objective_loss(
    logits: torch.Tensor,
    returns: torch.Tensor,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> torch.Tensor:
    sharpe = sharpe_ratio_loss(returns)
    directional = directional_accuracy_loss(logits, returns)
    risk = risk_adjusted_return_loss(returns)
    return weights[0] * sharpe + weights[1] * directional + weights[2] * risk


__all__ = [
    "FocalLoss",
    "directional_accuracy_loss",
    "multi_objective_loss",
    "risk_adjusted_return_loss",
    "sharpe_ratio_loss",
]
