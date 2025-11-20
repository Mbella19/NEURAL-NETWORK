"""Dedicated checkpoint system (Phase 6.5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class CheckpointConfig:
    directory: Path = Path("checkpoints")
    keep: int = 3
    monitor: str = "val_loss"
    mode: str = "min"


class CheckpointSystem:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, config: CheckpointConfig | None = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config or CheckpointConfig()
        self.config.directory.mkdir(parents=True, exist_ok=True)
        self.best_metric: Optional[float] = None
        self.best_path: Optional[Path] = None

    def save(self, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.config.directory / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        self._update_best(path, metrics.get(self.config.monitor))
        self._prune()
        return path

    def load(self, path: Optional[Path] = None, map_location: str = "cpu") -> Dict[str, float]:
        if path is None:
            checkpoints = sorted(self.config.directory.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            path = checkpoints[-1]
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["metrics"]

    def _update_best(self, path: Path, metric: Optional[float]) -> None:
        if metric is None:
            return
        if self.best_metric is None:
            self.best_metric = metric
            self.best_path = path
            return
        better = metric < self.best_metric if self.config.mode == "min" else metric > self.best_metric
        if better:
            self.best_metric = metric
            self.best_path = path

    def _prune(self) -> None:
        checkpoints = sorted(self.config.directory.glob("checkpoint_epoch_*.pt"))
        while len(checkpoints) > self.config.keep:
            old = checkpoints.pop(0)
            if self.best_path and old == self.best_path:
                continue
            old.unlink(missing_ok=True)


__all__ = ["CheckpointConfig", "CheckpointSystem"]
