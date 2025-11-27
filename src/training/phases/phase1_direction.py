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
        "VOLATILITY_REGIME",
        "ATR",  # Added for volatility context
        "RSI",  # Added for momentum context
    )
    threshold: float = 0.55
    max_epochs: int = 5
    forecast_horizon: int = 20  # Increased from 10 (more time to hit target)
    # FIX: More balanced take_profit/stop_loss for better label distribution
    take_profit: float = 0.0015  # ~15 pips (reduced from 20)
    stop_loss: float = 0.0015    # ~15 pips (increased from 10) - now 1:1 ratio
    time_limit: int = 50         # bars
    use_atr_scaling: bool = True  # Scale TP/SL by ATR


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
        """Compute triple-barrier labels with optional ATR scaling.

        FIX: Use ATR-scaled take-profit/stop-loss for volatility-adaptive barriers.
        This helps normalize the target across different volatility regimes.
        """
        close = frame["CLOSE"].reset_index(drop=True)
        high = frame["HIGH"].reset_index(drop=True)
        low = frame["LOW"].reset_index(drop=True)

        # FIX: Use forecast_horizon to match data trimming (20 bars) instead of time_limit (50 bars)
        horizon = max(1, self.config.forecast_horizon)
        labels = torch.zeros(len(close), dtype=torch.float32)

        # Compute ATR for volatility-adaptive barriers
        if self.config.use_atr_scaling and feature_frame is not None:
            # Look for ATR column in feature frame
            atr_col = None
            for col in ["ATR", "M5_ATR"]:
                if col in feature_frame.columns:
                    atr_col = col
                    break

            if atr_col:
                atr = feature_frame[atr_col].reset_index(drop=True).fillna(method='ffill').fillna(method='bfill')
                # ATR-scaled barriers: 1.5x ATR for both TP and SL
                atr_multiplier = 1.5
                tp_distances = atr * atr_multiplier
                sl_distances = atr * atr_multiplier
            else:
                # Fallback to fixed percentages
                tp_distances = close * self.config.take_profit
                sl_distances = close * self.config.stop_loss
        else:
            # Fixed percentage barriers
            tp_distances = close * self.config.take_profit
            sl_distances = close * self.config.stop_loss

        for idx in range(len(close)):
            if idx + 1 >= len(close):
                break
            end = min(len(close), idx + horizon + 1)
            window_high = high.iloc[idx + 1 : end].to_numpy()
            window_low = low.iloc[idx + 1 : end].to_numpy()

            entry_price = close.iloc[idx]
            tp_dist = tp_distances.iloc[idx] if hasattr(tp_distances, 'iloc') else tp_distances[idx]
            sl_dist = sl_distances.iloc[idx] if hasattr(sl_distances, 'iloc') else sl_distances[idx]

            tp_level = entry_price + tp_dist  # long take-profit
            sl_level = entry_price - sl_dist  # long stop-loss

            tp_hits = np.where(window_high >= tp_level)[0]
            sl_hits = np.where(window_low <= sl_level)[0]

            tp_first = tp_hits[0] if tp_hits.size > 0 else None
            sl_first = sl_hits[0] if sl_hits.size > 0 else None

            if tp_first is None and sl_first is None:
                # Neither hit: use direction of terminal price in window
                terminal = close.iloc[end - 1]
                labels[idx] = 1.0 if terminal > entry_price else 0.0
            elif tp_first is not None and sl_first is None:
                labels[idx] = 1.0  # up move dominated (TP hit first)
            elif tp_first is None and sl_first is not None:
                labels[idx] = 0.0  # down move dominated (SL hit first)
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
        # FIX: Removed arbitrary cap at 10 - use configured forecast_horizon
        return max(1, self.config.forecast_horizon)
