"""Model utility helpers (Phase 4.5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch import nn


def sharpe_ratio_loss(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = returns.mean()
    std = returns.std(unbiased=False)
    return -(mean / (std + eps))


def directional_accuracy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.sign(logits)
    return torch.mean((preds != torch.sign(targets)).float())


def combined_trading_loss(
    logits: torch.Tensor,
    returns: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> torch.Tensor:
    return alpha * sharpe_ratio_loss(returns) + beta * directional_accuracy_loss(logits, returns)


def get_scheduler(optimizer: torch.optim.Optimizer, name: str, **kwargs):
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name == "cosine_warmup":
        # Cosine annealing with linear warmup
        # Expects: T_max (total epochs), warmup_epochs (default 10)
        warmup_epochs = kwargs.pop("warmup_epochs", 10)
        T_max = kwargs.get("T_max", 100)

        # Linear warmup from 0.2x to 1.0x of base LR over warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,  # Start at 20% of base LR (e.g., 1e-4 if base is 5e-4)
            end_factor=1.0,    # Reach full base LR
            total_iters=warmup_epochs
        )

        # Cosine annealing for remaining epochs
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max - warmup_epochs,
            **{k: v for k, v in kwargs.items() if k != "T_max"}
        )

        # Sequential: warmup first, then cosine
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler {name}")


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    min_epochs: int = 0
    mode: str = "min"  # "min" for loss, "max" for accuracy/score
    best_score: Optional[float] = None
    counter: int = 0
    epochs_run: int = 0
    should_stop: bool = False

    def step(self, metric: float) -> bool:
        self.epochs_run += 1

        # Determine if current metric is better than best_score
        if self.mode == "min":
            is_better = self.best_score is None or metric < self.best_score - self.min_delta
        else:  # mode == "max"
            is_better = self.best_score is None or metric > self.best_score + self.min_delta

        # Don't stop before min_epochs
        if self.epochs_run < self.min_epochs:
            if is_better:
                self.best_score = metric
                self.counter = 0
            return False

        # Normal early stopping logic after min_epochs
        if is_better:
            self.best_score = metric
            self.counter = 0
            self.should_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointManager:
    def __init__(self, directory: Path, keep: int = 3) -> None:
        self.directory = directory
        self.keep = keep
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.directory / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        self._prune()
        return path

    def load(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, path: Optional[Path] = None):
        if path is None:
            checkpoints = sorted(self.directory.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints available")
            path = checkpoints[-1]
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint

    def _prune(self) -> None:
        checkpoints = sorted(self.directory.glob("checkpoint_epoch_*.pt"))
        while len(checkpoints) > self.keep:
            old = checkpoints.pop(0)
            old.unlink(missing_ok=True)


__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "clip_gradients",
    "combined_trading_loss",
    "directional_accuracy_loss",
    "get_scheduler",
    "sharpe_ratio_loss",
]
