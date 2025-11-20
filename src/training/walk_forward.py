"""Walk-forward validation utilities (Phase 6.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import pandas as pd

from .trainer import TrainingLoop, TrainerConfig


@dataclass
class WalkForwardConfig:
    train_window: int
    test_window: int
    step: int


def walk_forward_validation(
    dataset: pd.DataFrame,
    build_trainer: Callable[[pd.DataFrame, pd.DataFrame], TrainingLoop],
    config: WalkForwardConfig,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    start = 0
    end = config.train_window
    while end + config.test_window <= len(dataset):
        train_slice = dataset.iloc[start:end]
        test_slice = dataset.iloc[end : end + config.test_window]
        trainer = build_trainer(train_slice, test_slice)
        metrics = trainer.train()
        results.append(metrics)
        start += config.step
        end = start + config.train_window
    return results


__all__ = ["WalkForwardConfig", "walk_forward_validation"]
