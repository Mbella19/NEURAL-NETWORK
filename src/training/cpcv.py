"""Combinatorial Purged Cross-Validation (CPCV) utility for time-series evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd


@dataclass
class CPCVConfig:
    n_splits: int = 5
    test_window: int = 5_000
    purge_window: int = 500


def cpcv_validation(
    dataset: pd.DataFrame,
    build_trainer: Callable[[pd.DataFrame, pd.DataFrame], object],
    config: CPCVConfig,
) -> List[Dict[str, float]]:
    """Run a simplified CPCV: generate combinatorial train/test folds with purge."""

    n = len(dataset)
    fold_results: List[Dict[str, float]] = []
    step = max(1, (n - config.test_window) // config.n_splits)
    starts = list(range(0, n - config.test_window + 1, step))[: config.n_splits]

    for start in starts:
        test_start = start
        test_end = start + config.test_window
        test_slice = dataset.iloc[test_start:test_end]

        # Purge overlapping region to avoid leakage
        purge_start = max(0, test_start - config.purge_window)
        train_slice = pd.concat(
            [
                dataset.iloc[0:purge_start],
                dataset.iloc[test_end:],
            ],
            ignore_index=True,
        )
        trainer = build_trainer(train_slice, test_slice)
        metrics = trainer.train()
        fold_results.append(metrics)
    return fold_results


__all__ = ["CPCVConfig", "cpcv_validation"]
