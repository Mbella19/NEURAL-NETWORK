"""Curriculum Phase 2: Indicator behavior (Phase 5.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase2Config:
    feature_subset: Tuple[str, ...] = (
        "SMA_10",
        "EMA_12",
        "RSI",
        "MACD_LINE",
        "MACD_SIGNAL",
        "ATR",
        "VOLATILITY_REGIME",
        "MARKET_EFFICIENCY",
    )
    threshold: float = 0.6
    retention_drop: float = 0.05
    forecast_horizon: int = 20  # Medium horizon (regime)
    # Longer horizon may be more predictable than short-term noise


class Phase2IndicatorTask:
    def __init__(self, config: Phase2Config | None = None) -> None:
        self.config = config or Phase2Config()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.output_dim = 1

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        base_df = feature_frame if feature_frame is not None else run_feature_pipeline(
            frame, pipeline=pipeline
        ).dataframe
        return base_df.loc[:, self.config.feature_subset].copy()

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        if feature_frame is None:
            raise ValueError("feature_frame is required to avoid recomputing indicators with future data.")
        close_series = frame["CLOSE"]
        # Single horizon prediction for focused gradient signal
        future_ret = ((close_series.shift(-self._horizon) - close_series) / close_series).fillna(0)
        target = torch.tensor((future_ret > 0).astype(float).values, dtype=torch.float32)
        return target.unsqueeze(1)  # [N, 1] instead of [N, 10]

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds = torch.sigmoid(logits)
        accuracy = ((preds > 0.5) == (targets > 0.5)).float().mean().item()
        loss = self.loss_fn(logits, targets).item()
        return {"accuracy": accuracy, "loss": loss}

    @property
    def _horizon(self) -> int:
        return max(1, min(10, self.config.forecast_horizon))
