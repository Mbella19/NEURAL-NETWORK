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
        """Compute targets for S/R level behavior prediction.

        REDESIGNED: Only compute targets for samples near S/R levels.
        - Near support + bounces up = 1 (support respected)
        - Near resistance + bounces down = 1 (resistance respected)
        - Near support + breaks down = 0 (support broken)
        - Near resistance + breaks up = 0 (resistance broken)
        - NOT near S/R = NaN (masked in loss computation)
        """
        if feature_frame is None:
            raise ValueError("feature_frame is required to avoid recomputing S/R features with future data.")

        close = frame["CLOSE"]
        horizon = self._horizon

        # Compute ATR for distance thresholds
        high = frame["HIGH"]
        low = frame["LOW"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().bfill()
        atr_pct = atr / close

        # Get S/R features
        support_strength = pd.Series(0.0, index=frame.index)
        resistance_strength = pd.Series(0.0, index=frame.index)
        sr_distance = pd.Series(1.0, index=frame.index)  # default far from S/R

        for prefix in ["", "M5_"]:
            dist_col = f"{prefix}SR_DISTANCE" if f"{prefix}SR_DISTANCE" in feature_frame.columns else f"{prefix}SR_DISTANCE_W10"
            supp_col = f"{prefix}SR_SUPPORT_STRENGTH" if f"{prefix}SR_SUPPORT_STRENGTH" in feature_frame.columns else f"{prefix}SR_SUPPORT_STRENGTH_W10"
            res_col = f"{prefix}SR_RESISTANCE_STRENGTH" if f"{prefix}SR_RESISTANCE_STRENGTH" in feature_frame.columns else f"{prefix}SR_RESISTANCE_STRENGTH_W10"

            if dist_col in feature_frame.columns:
                sr_distance = feature_frame[dist_col].fillna(1.0).abs()
            if supp_col in feature_frame.columns:
                support_strength = feature_frame[supp_col].fillna(0)
            if res_col in feature_frame.columns:
                resistance_strength = feature_frame[res_col].fillna(0)

        # Use ATR-based proximity threshold for better adaptability
        # Instead of fixed 0.2%, use percentage of ATR for dynamic threshold
        atr_based_threshold = self.config.atr_multiplier * atr_pct
        near_sr_threshold = atr_based_threshold.clip(lower=0.001, upper=0.02)  # Clamp to reasonable range

        # Determine if near support or resistance
        near_support = ((sr_distance < near_sr_threshold) & (support_strength > 0.3)).astype(float)
        near_resistance = ((sr_distance < near_sr_threshold) & (resistance_strength > 0.3)).astype(float)

        # Future price movement
        future_return = (close.shift(-horizon) - close) / close
        price_went_up = (future_return > 0).astype(float)

        # S/R respected: near support and bounced up, OR near resistance and bounced down
        support_respected = near_support * price_went_up
        resistance_respected = near_resistance * (1 - price_went_up)

        # Combined S/R behavior target
        near_any_sr = ((near_support + near_resistance) > 0).astype(float)
        sr_respected = support_respected + resistance_respected

        # FIXED: Use NaN for samples NOT near S/R (will be masked in loss computation)
        # This prevents the fallback mechanism that was causing 99.95% positive rate
        target = sr_respected.copy()
        target[near_any_sr == 0] = float('nan')

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
