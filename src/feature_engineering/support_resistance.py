"""Support and resistance detection (Phase 3.7)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .base import BaseFeatureCalculator


@dataclass
class SupportResistanceLevel:
    price: float
    strength: float


class SupportResistanceFeatures(BaseFeatureCalculator):
    name = "support_resistance"
    required_columns = ("HIGH", "LOW", "CLOSE")
    # Note: produces is dynamically generated in __init__ to include suffix

    def __init__(
        self,
        *,
        window: int = 20,
        prominence: float = 0.0005,
        psychological_step: float = 0.50,
    ) -> None:
        super().__init__(window=window)
        self.window = window
        self.prominence = prominence
        self.psychological_step = psychological_step

        # Dynamically set produces with suffix
        suffix = f"_W{self.window}"
        self.produces = (
            f"SR_SUPPORT_STRENGTH{suffix}",
            f"SR_RESISTANCE_STRENGTH{suffix}",
            f"SR_PSYCHO_CONFLUENCE{suffix}",
            f"SR_DISTANCE{suffix}",
            f"SR_NEARBY{suffix}",
        )

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        high = frame["HIGH"]
        low = frame["LOW"]

        # Identify local extrema
        rolling_high = high.rolling(window=self.window, center=True, min_periods=1)
        rolling_low = low.rolling(window=self.window, center=True, min_periods=1)

        local_highs = (high == rolling_high.max()).astype(int)
        local_lows = (low == rolling_low.min()).astype(int)

        levels = self._cluster_levels(high, local_highs, kind="high") + self._cluster_levels(low, local_lows, kind="low")

        # Memory-efficient batched approach to avoid OOM on low-RAM systems
        n_rows = len(frame)
        close_values = frame["CLOSE"].values.astype(np.float32)  # Use float32 to save memory

        # Initialize output arrays
        sr_support = np.zeros(n_rows, dtype=np.float32)
        sr_resistance = np.zeros(n_rows, dtype=np.float32)
        sr_confluence = np.zeros(n_rows, dtype=np.float32)
        sr_distance = np.full(n_rows, np.inf, dtype=np.float32)

        if levels:
            # Process in batches to limit memory usage
            batch_size = min(50, len(levels))  # Process max 50 levels at a time
            level_prices = np.array([level.price for level in levels], dtype=np.float32)
            level_strengths = np.array([level.strength for level in levels], dtype=np.float32)
            is_psych = np.abs(level_prices * 100 - np.round(level_prices * 100)) <= 0.05

            for i in range(0, len(levels), batch_size):
                batch_end = min(i + batch_size, len(levels))
                batch_prices = level_prices[i:batch_end]
                batch_strengths = level_strengths[i:batch_end]
                batch_psych = is_psych[i:batch_end]

                # Compute distances for this batch (shape: batch_size × n_rows)
                distances = np.abs(close_values[np.newaxis, :] - batch_prices[:, np.newaxis])
                sr_distance = np.minimum(sr_distance, np.min(distances, axis=0))

                # Determine support vs resistance
                is_support = batch_prices[:, np.newaxis] <= close_values[np.newaxis, :]

                # Update support strengths
                support_strengths = np.where(is_support, batch_strengths[:, np.newaxis], 0)
                sr_support = np.maximum(sr_support, np.max(support_strengths, axis=0))

                # Update resistance strengths
                resistance_strengths = np.where(~is_support, batch_strengths[:, np.newaxis], 0)
                sr_resistance = np.maximum(sr_resistance, np.max(resistance_strengths, axis=0))

                # Update psychological confluence
                if np.any(batch_psych):
                    psych_strengths = np.where(batch_psych[:, np.newaxis], batch_strengths[:, np.newaxis], 0)
                    sr_confluence = np.maximum(sr_confluence, np.max(psych_strengths, axis=0))

        # Add suffix to avoid overwrites when using multiple windows/prominence values
        suffix = f"_W{self.window}"
        return pd.DataFrame(
            {
                f"SR_SUPPORT_STRENGTH{suffix}": sr_support,
                f"SR_RESISTANCE_STRENGTH{suffix}": sr_resistance,
                f"SR_PSYCHO_CONFLUENCE{suffix}": sr_confluence,
                f"SR_DISTANCE{suffix}": pd.Series(sr_distance, index=frame.index).replace(np.inf, np.nan).bfill().ffill(),
                f"SR_NEARBY{suffix}": (sr_distance <= self.prominence).astype(int),
            },
            index=frame.index,
        )

    def _cluster_levels(self, series: pd.Series, mask: pd.Series, kind: str) -> List[SupportResistanceLevel]:
        points = series[mask.astype(bool)].values
        if len(points) == 0:
            return []

        # Sort points for more efficient clustering
        points_sorted = np.sort(points)

        levels: List[SupportResistanceLevel] = []
        for value in points_sorted:
            # Since points are sorted, only check the last few levels (they're most likely to match)
            matched = False
            # Check levels in reverse order (most recent first) - more likely to match
            for i in range(len(levels) - 1, -1, -1):
                level = levels[i]
                if abs(level.price - value) <= self.prominence:
                    level.price = (level.price + value) / 2
                    level.strength += 1
                    matched = True
                    break
                # If we're past the proximity window, no need to check older levels
                elif value - level.price > self.prominence:
                    break
            if not matched:
                levels.append(SupportResistanceLevel(price=float(value), strength=1.0))
        return levels


__all__ = ["SupportResistanceFeatures"]
