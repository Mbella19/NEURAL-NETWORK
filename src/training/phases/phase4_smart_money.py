"""Curriculum Phase 4: Smart money concepts (Phase 5.5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase4Config:
    feature_subset: Tuple[str, ...] = (
        "BOS_BULLISH",
        "BOS_BEARISH",
        "CHOCH_BULLISH",
        "CHOCH_BEARISH",
        "ORDER_BLOCK_BULL",
        "ORDER_BLOCK_BEAR",
        "LIQUIDITY_SWEEP",
        "FVG_UP",
        "FVG_DOWN",
    )
    threshold: float = 0.64
    forecast_horizon: int = 20  # Medium horizon (context)


class Phase4SmartMoneyTask:
    def __init__(self, config: Phase4Config | None = None) -> None:
        self.config = config or Phase4Config()
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_dim = 3

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        base_df = feature_frame if feature_frame is not None else run_feature_pipeline(
            frame, pipeline=pipeline
        ).dataframe
        return base_df.loc[:, self.config.feature_subset].copy()

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        """REDESIGNED: Predict if CURRENT SMC signals correctly predict future price direction.

        This fixes the data leakage issue where we were predicting what SMC patterns
        would appear in the future (look-ahead bias).

        Target classes:
        - 0: Bearish signal was correct (bearish SMC + price went down)
        - 1: No clear signal OR signal was wrong
        - 2: Bullish signal was correct (bullish SMC + price went up)
        """
        if feature_frame is None:
            raise ValueError("feature_frame is required to avoid recomputing SMC features with future data.")

        base_df = feature_frame
        horizon = self._horizon

        # Get CURRENT SMC signals (NO shift - this is the key fix!)
        def get_col(name, default=0):
            """Helper to get column with fallback."""
            for suffix in ["", "_SW5"]:
                col = f"{name}{suffix}"
                if col in base_df.columns:
                    return base_df[col].fillna(0)
            return default

        # Aggregate bullish signals
        bullish_signals = (
            get_col("BOS_BULLISH") +
            get_col("FVG_UP") +
            get_col("ORDER_BLOCK_BULL") +
            2 * get_col("CHOCH_BULLISH")  # CHOCH is stronger signal
        )

        # Aggregate bearish signals
        bearish_signals = (
            get_col("BOS_BEARISH") +
            get_col("FVG_DOWN") +
            get_col("ORDER_BLOCK_BEAR") +
            2 * get_col("CHOCH_BEARISH")  # CHOCH is stronger signal
        )

        # Compute net SMC bias: positive = bullish, negative = bearish, zero = neutral
        smc_bias = bullish_signals - bearish_signals

        # Future price direction (this is the ONLY look-ahead, which is necessary for targets)
        close = frame["CLOSE"]
        future_return = (close.shift(-horizon) - close) / close
        price_went_up = (future_return > 0).astype(float)

        # Classify SMC signal correctness
        # Threshold for "having a signal" - need some minimum signal strength
        signal_threshold = 0.5

        has_bullish = (smc_bias > signal_threshold).astype(float)
        has_bearish = (smc_bias < -signal_threshold).astype(float)
        has_signal = has_bullish + has_bearish

        # Target classes:
        # Class 2: Bullish signal + price went up (bullish correct)
        # Class 0: Bearish signal + price went down (bearish correct)
        # Class 1: No signal OR wrong signal
        bullish_correct = (has_bullish * price_went_up).astype(int) * 2
        bearish_correct = (has_bearish * (1 - price_went_up)).astype(int) * 0  # multiply by 0 to get class 0

        # Everything else is class 1 (neutral/wrong)
        is_wrong_or_neutral = 1 - (has_bullish * price_went_up + has_bearish * (1 - price_went_up))
        neutral_or_wrong = (is_wrong_or_neutral > 0).astype(int)

        # Combine: bullish_correct gives 2, bearish_correct stays 0, neutral/wrong gets 1
        target = bullish_correct + neutral_or_wrong

        return torch.tensor(target.values, dtype=torch.long)

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        loss = self.loss_fn(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        return {"accuracy": accuracy, "loss": loss}

    @property
    def _horizon(self) -> int:
        # FIX: Removed arbitrary cap at 10 - use configured forecast_horizon
        return max(1, self.config.forecast_horizon)
