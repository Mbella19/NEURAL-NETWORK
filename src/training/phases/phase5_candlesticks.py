"""Curriculum Phase 5: Candlestick pattern mastery (Phase 5.6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase5Config:
    feature_subset: Tuple[str, ...] = (
        "DOJI",
        "SPINNING_TOP",
        "HAMMER",
        "SHOOTING_STAR",
        "MARUBOZU",
        "BULLISH_ENGULFING",
        "BEARISH_ENGULFING",
        "PATTERN_STRENGTH",
        "ATR",  # Context for volatility
        "VOLATILITY_REGIME",  # Market conditions
    )
    threshold: float = 0.66
    forecast_horizon: int = 5  # Short horizon (micro behavior)
    pattern_threshold: float = 0.1  # Threshold for "pattern exists"
    # Multiplier on ATR to define significant move
    atr_multiplier: float = 1.0


class Phase5CandlestickTask:
    def __init__(self, config: Phase5Config | None = None) -> None:
        self.config = config or Phase5Config()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.output_dim = 1

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        base_df = feature_frame if feature_frame is not None else run_feature_pipeline(
            frame, pipeline=pipeline
        ).dataframe
        return base_df.loc[:, self.config.feature_subset].copy()

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        """Compute targets for candlestick pattern confirmation.

        REDESIGNED: Only compute targets for samples with candlestick patterns.
        - Bullish pattern + price goes UP = 1 (pattern confirmed)
        - Bearish pattern + price goes DOWN = 1 (pattern confirmed)
        - Pattern present but not confirmed = 0
        - No pattern = NaN (masked in loss computation)
        """
        if feature_frame is None:
            raise ValueError("feature_frame is required")

        close = frame["CLOSE"]
        horizon = self._horizon

        # Get future return direction
        future_return = (close.shift(-horizon) - close) / close
        price_went_up = (future_return > 0).astype(float)

        # Detect bullish and bearish patterns from features
        # Bullish patterns: hammer, bullish_engulfing, marubozu (green)
        # Bearish patterns: shooting_star, bearish_engulfing, marubozu (red)
        bullish_pattern = pd.Series(0.0, index=frame.index)
        bearish_pattern = pd.Series(0.0, index=frame.index)

        # Check for pattern columns (handle both prefixed and unprefixed)
        for prefix in ["", "M5_"]:
            if f"{prefix}HAMMER" in feature_frame.columns:
                bullish_pattern += feature_frame[f"{prefix}HAMMER"].fillna(0)
            if f"{prefix}BULLISH_ENGULFING" in feature_frame.columns:
                bullish_pattern += feature_frame[f"{prefix}BULLISH_ENGULFING"].fillna(0)
            if f"{prefix}SHOOTING_STAR" in feature_frame.columns:
                bearish_pattern += feature_frame[f"{prefix}SHOOTING_STAR"].fillna(0)
            if f"{prefix}BEARISH_ENGULFING" in feature_frame.columns:
                bearish_pattern += feature_frame[f"{prefix}BEARISH_ENGULFING"].fillna(0)

        # Pattern confirmed: bullish pattern + up, OR bearish pattern + down
        has_bullish = (bullish_pattern > 0).astype(float)
        has_bearish = (bearish_pattern > 0).astype(float)

        # Target = 1 if pattern is confirmed by price action
        bullish_confirmed = has_bullish * price_went_up
        bearish_confirmed = has_bearish * (1 - price_went_up)

        # Combined: either pattern type confirmed
        pattern_exists = ((has_bullish + has_bearish) > 0).astype(float)
        pattern_confirmed = bullish_confirmed + bearish_confirmed

        # FIXED: Use NaN for samples WITHOUT patterns (will be masked in loss computation)
        # This prevents mixing two different prediction problems
        target = pattern_confirmed.copy()
        target[pattern_exists == 0] = float('nan')

        return torch.tensor(target.values, dtype=torch.float32).unsqueeze(1)

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds = torch.sigmoid(logits)
        accuracy = ((preds > 0.5) == (targets > 0.5)).float().mean().item()
        loss = self.loss_fn(logits, targets).item()
        return {"accuracy": accuracy, "loss": loss}

    @property
    def _horizon(self) -> int:
        # FIX: Removed arbitrary cap at 10 - use configured forecast_horizon
        return max(1, self.config.forecast_horizon)
