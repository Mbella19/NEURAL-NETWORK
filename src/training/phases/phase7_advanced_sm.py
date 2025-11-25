"""Curriculum Phase 7: Advanced smart money (Phase 5.8)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase7Config:
    feature_subset: Tuple[str, ...] = (
        "BOS_BULLISH",
        "BOS_BEARISH",
        "CHOCH_BULLISH",
        "CHOCH_BEARISH",
        "ORDER_BLOCK_BULL",
        "ORDER_BLOCK_BEAR",
        "FVG_UP",
        "FVG_DOWN",
        "LIQUIDITY_SWEEP",
        "IMBALANCE_ZONE",
        "ATR",  # Context for volatility
        "VOLATILITY_REGIME",  # Market conditions
    )
    threshold: float = 0.7
    forecast_horizon: int = 5  # Short horizon (micro behavior)
    signal_threshold: float = 0.1  # Threshold for "signal exists"
    atr_multiplier: float = 1.0


class Phase7AdvancedSMTask:
    def __init__(self, config: Phase7Config | None = None) -> None:
        self.config = config or Phase7Config()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.output_dim = 1

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        base_df = feature_frame if feature_frame is not None else run_feature_pipeline(
            frame, pipeline=pipeline
        ).dataframe
        return base_df.loc[:, self.config.feature_subset].copy()

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        """Compute targets for SMC-based directional prediction.

        REDESIGNED: Only compute targets for samples with SMC signals.
        - Bullish SMC context + price UP = 1 (signal confirmed)
        - Bearish SMC context + price DOWN = 1 (signal confirmed)
        - SMC signal present but not confirmed = 0
        - No SMC signal = NaN (masked in loss computation)
        """
        if feature_frame is None:
            raise ValueError("feature_frame is required")

        close = frame["CLOSE"]
        horizon = self._horizon

        # Future price direction
        future_return = (close.shift(-horizon) - close) / close
        price_went_up = (future_return > 0).astype(float)

        # Aggregate bullish SMC signals
        bullish_signal = pd.Series(0.0, index=frame.index)
        bearish_signal = pd.Series(0.0, index=frame.index)

        # Check for SMC columns (handle both suffixed and unsuffixed)
        for prefix in ["", "M5_"]:
            # Break of Structure
            bos_bull = f"{prefix}BOS_BULLISH" if f"{prefix}BOS_BULLISH" in feature_frame.columns else f"{prefix}BOS_BULLISH_SW5"
            bos_bear = f"{prefix}BOS_BEARISH" if f"{prefix}BOS_BEARISH" in feature_frame.columns else f"{prefix}BOS_BEARISH_SW5"
            if bos_bull in feature_frame.columns:
                bullish_signal += feature_frame[bos_bull].fillna(0)
            if bos_bear in feature_frame.columns:
                bearish_signal += feature_frame[bos_bear].fillna(0)

            # Fair Value Gaps
            fvg_up = f"{prefix}FVG_UP" if f"{prefix}FVG_UP" in feature_frame.columns else f"{prefix}FVG_UP_SW5"
            fvg_down = f"{prefix}FVG_DOWN" if f"{prefix}FVG_DOWN" in feature_frame.columns else f"{prefix}FVG_DOWN_SW5"
            if fvg_up in feature_frame.columns:
                bullish_signal += feature_frame[fvg_up].fillna(0)
            if fvg_down in feature_frame.columns:
                bearish_signal += feature_frame[fvg_down].fillna(0)

            # Order Blocks
            ob_bull = f"{prefix}ORDER_BLOCK_BULL" if f"{prefix}ORDER_BLOCK_BULL" in feature_frame.columns else f"{prefix}ORDER_BLOCK_BULL_SW5"
            ob_bear = f"{prefix}ORDER_BLOCK_BEAR" if f"{prefix}ORDER_BLOCK_BEAR" in feature_frame.columns else f"{prefix}ORDER_BLOCK_BEAR_SW5"
            if ob_bull in feature_frame.columns:
                bullish_signal += feature_frame[ob_bull].fillna(0)
            if ob_bear in feature_frame.columns:
                bearish_signal += feature_frame[ob_bear].fillna(0)

            # Change of Character (reversal signal)
            choch_bull = f"{prefix}CHOCH_BULLISH" if f"{prefix}CHOCH_BULLISH" in feature_frame.columns else f"{prefix}CHOCH_BULLISH_SW5"
            choch_bear = f"{prefix}CHOCH_BEARISH" if f"{prefix}CHOCH_BEARISH" in feature_frame.columns else f"{prefix}CHOCH_BEARISH_SW5"
            if choch_bull in feature_frame.columns:
                bullish_signal += feature_frame[choch_bull].fillna(0) * 2  # Weight CHoCH higher (reversal)
            if choch_bear in feature_frame.columns:
                bearish_signal += feature_frame[choch_bear].fillna(0) * 2

        # Net SMC bias
        smc_bias = bullish_signal - bearish_signal
        has_bullish_bias = (smc_bias > self.config.signal_threshold).astype(float)
        has_bearish_bias = (smc_bias < -self.config.signal_threshold).astype(float)

        # SMC signal confirmed by price action
        bullish_confirmed = has_bullish_bias * price_went_up
        bearish_confirmed = has_bearish_bias * (1 - price_went_up)

        # Combined: whether SMC signal exists
        has_smc_signal = ((has_bullish_bias + has_bearish_bias) > 0).astype(float)
        smc_confirmed = bullish_confirmed + bearish_confirmed

        # FIXED: Use NaN for samples WITHOUT SMC signals (will be masked in loss computation)
        # This prevents mixing two different prediction problems
        target = smc_confirmed.copy()
        target[has_smc_signal == 0] = float('nan')

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
