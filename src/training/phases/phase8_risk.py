"""Curriculum Phase 8: Risk management (Phase 5.10)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase8RiskConfig:
    feature_subset: Tuple[str, ...] = (
        "SR_DISTANCE",
        "SR_SUPPORT_STRENGTH",
        "SR_RESISTANCE_STRENGTH",
        "VOLATILITY_REGIME",
        "ATR",
        "STRUCTURE_STRENGTH",  # Using STRUCTURE_STRENGTH_L5 from MarketStructureFeatures
    )
    threshold: float = 0.75
    forecast_horizon: int = 30  # Longer horizon (risk)


class Phase8RiskTask:
    def __init__(self, config: Phase8RiskConfig | None = None) -> None:
        self.config = config or Phase8RiskConfig()
        self.loss_fn = nn.SmoothL1Loss()
        self.output_dim = self._horizon

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        base_df = feature_frame if feature_frame is not None else run_feature_pipeline(
            frame, pipeline=pipeline
        ).dataframe
        return base_df.loc[:, self.config.feature_subset].copy()

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        if feature_frame is None:
            raise ValueError("feature_frame is required to avoid recomputing ATR features with future data.")
        base_df = feature_frame

        # Compute pandas operations first, then convert to tensor
        close_series = frame["CLOSE"]
        cummax_series = close_series.cummax()
        drawdown_series = (close_series - cummax_series).abs()

        horizon = self._horizon
        targets = []
        for step in range(1, horizon + 1):
            shifted_drawdown = torch.tensor(drawdown_series.shift(-step).bfill().fillna(0).values, dtype=torch.float32)
            atr = torch.tensor(base_df["ATR"].shift(-step).bfill().fillna(0).values, dtype=torch.float32)
            tgt = (shifted_drawdown / (atr + 1e-6)).clamp(0, 5)
            targets.append(tgt)
        return torch.stack(targets, dim=1)

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        mse = self.loss_fn(logits, targets).item()
        mae = torch.mean(torch.abs(logits - targets)).item()
        return {"mse": mse, "mae": mae}

    @property
    def _horizon(self) -> int:
        return max(1, min(10, self.config.forecast_horizon))
