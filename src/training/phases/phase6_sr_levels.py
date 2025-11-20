"""Curriculum Phase 6: Support/Resistance respect (Phase 5.7)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase6Config:
    feature_subset: Tuple[str, ...] = (
        "SR_SUPPORT_STRENGTH",
        "SR_RESISTANCE_STRENGTH",
        "SR_PSYCHO_CONFLUENCE",
        "SR_DISTANCE",
        "TREND_DIRECTION_L5",
        "ATR",  # Context for volatility
        "VOLATILITY_REGIME",  # Market conditions
    )
    threshold: float = 0.68
    forecast_horizon: int = 5  # Short horizon (micro behavior)
    sr_proximity: float = 0.002  # 0.2% distance to S/R
    atr_multiplier: float = 1.0  # Volatility-relative bounce threshold


class Phase6SupportResistanceTask:
    def __init__(self, config: Phase6Config | None = None) -> None:
        self.config = config or Phase6Config()
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
            raise ValueError("feature_frame is required to avoid recomputing S/R features with future data.")

        base_df = feature_frame
        close = frame["CLOSE"]
        horizon = self._horizon

        high = frame["HIGH"]
        low = frame["LOW"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().bfill()

        future_return = (close.shift(-horizon) - close) / close
        atr_pct = atr / close
        significant_move = (future_return.abs() > (self.config.atr_multiplier * atr_pct)).astype(float)
        target = torch.tensor(significant_move.fillna(0).values, dtype=torch.float32)

        return target.unsqueeze(1)

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds = torch.sigmoid(logits)
        accuracy = ((preds > 0.5) == (targets > 0.5)).float().mean().item()
        loss = self.loss_fn(logits, targets).item()
        return {"accuracy": accuracy, "loss": loss}

    @property
    def _horizon(self) -> int:
        return max(1, min(10, self.config.forecast_horizon))
