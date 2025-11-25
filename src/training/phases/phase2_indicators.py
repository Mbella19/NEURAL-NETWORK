"""Curriculum Phase 2: Indicator behavior (Phase 5.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
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
        """Compute targets for indicator-based trend continuation prediction.

        FIX: Instead of simple "price goes up", predict if the CURRENT TREND
        indicated by technical indicators will CONTINUE. This is more learnable
        because the indicators provide a baseline prediction.

        - If current trend is UP (SMA rising) AND price continues up = 1 (trend holds)
        - If current trend is DOWN (SMA falling) AND price continues down = 1 (trend holds)
        - If trend reverses = 0
        """
        if feature_frame is None:
            raise ValueError("feature_frame is required to avoid recomputing indicators with future data.")

        close = frame["CLOSE"]
        horizon = self._horizon

        # Future return direction
        future_ret = (close.shift(-horizon) - close) / close
        price_went_up = (future_ret > 0).astype(float)

        # Detect current trend using SMA crossovers or slope
        # Look for SMA columns
        sma_trend = pd.Series(0.5, index=frame.index)  # Default neutral

        for prefix in ["", "M5_"]:
            sma_fast = f"{prefix}SMA_10" if f"{prefix}SMA_10" in feature_frame.columns else f"{prefix}SMA_14"
            sma_slow = f"{prefix}SMA_50" if f"{prefix}SMA_50" in feature_frame.columns else None

            if sma_fast in feature_frame.columns:
                fast_ma = feature_frame[sma_fast].fillna(method='ffill')
                # Trend = SMA slope (compare to 5 bars ago)
                sma_slope = (fast_ma - fast_ma.shift(5)) / fast_ma.shift(5).abs().clip(lower=1e-8)
                sma_trend = (sma_slope > 0).astype(float)
                break  # Use first found

        # Trend continuation: current trend direction matches future direction
        trend_is_up = sma_trend.astype(float)
        trend_is_down = 1 - trend_is_up

        # Target = 1 if trend continues, 0 if reverses
        uptrend_continues = trend_is_up * price_went_up
        downtrend_continues = trend_is_down * (1 - price_went_up)
        trend_continuation = uptrend_continues + downtrend_continues

        return torch.tensor(trend_continuation.fillna(0.5).values, dtype=torch.float32).unsqueeze(1)

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds = torch.sigmoid(logits)
        accuracy = ((preds > 0.5) == (targets > 0.5)).float().mean().item()
        loss = self.loss_fn(logits, targets).item()
        return {"accuracy": accuracy, "loss": loss}

    @property
    def _horizon(self) -> int:
        # FIX: Removed arbitrary cap at 10 - use configured forecast_horizon
        return max(1, self.config.forecast_horizon)
