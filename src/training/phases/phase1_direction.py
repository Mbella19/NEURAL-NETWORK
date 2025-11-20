"""Curriculum Phase 1: Direction prediction (Phase 5.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase1Config:
    feature_subset: tuple[str, ...] = (
        "TREND_DIRECTION_L5",
        "SMA_14", "SMA_50", "SMA_200",
        "EMA_20",
        "VOLATILITY_REGIME"
    )
    threshold: float = 0.55
    max_epochs: int = 5
    forecast_horizon: int = 10  # Medium horizon (direction)
    take_profit: float = 0.0020  # ~20 pips
    stop_loss: float = 0.0010    # ~10 pips
    time_limit: int = 50         # bars


class Phase1DirectionTask:
    def __init__(self, config: Phase1Config | None = None) -> None:
        self.config = config or Phase1Config()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.output_dim = 1

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        base_df = feature_frame if feature_frame is not None else run_feature_pipeline(
            frame, pipeline=pipeline
        ).dataframe
        return base_df.loc[:, self.config.feature_subset].copy()

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        close = frame["CLOSE"].reset_index(drop=True)
        high = frame["HIGH"].reset_index(drop=True)
        low = frame["LOW"].reset_index(drop=True)

        tp_mult = 1.0 + self.config.take_profit
        sl_mult = 1.0 - self.config.stop_loss
        horizon = max(1, self.config.time_limit)

        labels = torch.zeros(len(close), dtype=torch.float32)

        for idx in range(len(close)):
            if idx + 1 >= len(close):
                break
            end = min(len(close), idx + horizon + 1)
            window_high = high.iloc[idx + 1 : end].to_numpy()
            window_low = low.iloc[idx + 1 : end].to_numpy()

            tp_level = close.iloc[idx] * tp_mult  # long take-profit
            sl_level = close.iloc[idx] * sl_mult  # long stop-loss / short take-profit

            tp_hits = np.where(window_high >= tp_level)[0]
            sl_hits = np.where(window_low <= sl_level)[0]

            tp_first = tp_hits[0] if tp_hits.size > 0 else None
            sl_first = sl_hits[0] if sl_hits.size > 0 else None

            if tp_first is None and sl_first is None:
                # Neither hit: fall back to direction of terminal price in window
                terminal = close.iloc[end - 1]
                labels[idx] = 1.0 if terminal > close.iloc[idx] else 0.0
            elif tp_first is not None and sl_first is None:
                labels[idx] = 1.0  # up move dominated
            elif tp_first is None and sl_first is not None:
                labels[idx] = 0.0  # down move dominated
            else:
                labels[idx] = 1.0 if tp_first <= sl_first else 0.0

        return labels.unsqueeze(1)

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds = torch.sigmoid(logits)
        accuracy = ((preds > 0.5) == (targets > 0.5)).float().mean().item()
        loss = self.loss_fn(logits, targets).item()
        return {"accuracy": accuracy, "loss": loss}

    @property
    def _horizon(self) -> int:
        return max(1, min(10, self.config.forecast_horizon))
