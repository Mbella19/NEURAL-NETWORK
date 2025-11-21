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

        # Identify local extrema using only past data to avoid look-ahead leakage
        rolling_high = high.rolling(window=self.window, center=False, min_periods=1)
        rolling_low = low.rolling(window=self.window, center=False, min_periods=1)

        local_highs = (high == rolling_high.max()).astype(int)
        local_lows = (low == rolling_low.min()).astype(int)

        n_rows = len(frame)
        close_values = frame["CLOSE"].values.astype(np.float32)  # Use float32 to save memory

        # Initialize output arrays
        sr_support = np.zeros(n_rows, dtype=np.float32)
        sr_resistance = np.zeros(n_rows, dtype=np.float32)
        sr_confluence = np.zeros(n_rows, dtype=np.float32)
        sr_distance = np.full(n_rows, np.inf, dtype=np.float32)  # stays inf until a level forms

        # Build levels incrementally so each row only sees history up to that point (prevents leakage)
        support_levels: List[SupportResistanceLevel] = []
        resistance_levels: List[SupportResistanceLevel] = []

        def _update_levels(levels: List[SupportResistanceLevel], value: float) -> None:
            """Merge the new value into existing clustered levels or create a new one."""
            for level in reversed(levels):
                if abs(level.price - value) <= self.prominence:
                    level.price = (level.price + value) / 2
                    level.strength += 1
                    return
            levels.append(SupportResistanceLevel(price=float(value), strength=1.0))

        for idx, close_price in enumerate(close_values):
            if local_lows.iloc[idx]:
                _update_levels(support_levels, float(low.iloc[idx]))
            if local_highs.iloc[idx]:
                _update_levels(resistance_levels, float(high.iloc[idx]))

            all_levels: List[SupportResistanceLevel] = support_levels + resistance_levels
            if not all_levels:
                continue

            prices = np.fromiter((lvl.price for lvl in all_levels), dtype=np.float32)
            strengths = np.fromiter((lvl.strength for lvl in all_levels), dtype=np.float32)

            distances = np.abs(prices - close_price)
            sr_distance[idx] = float(distances.min())

            sr_support[idx] = max((lvl.strength for lvl in support_levels if lvl.price <= close_price), default=0.0)
            sr_resistance[idx] = max((lvl.strength for lvl in resistance_levels if lvl.price > close_price), default=0.0)

            psych_mask = np.abs(prices * 100 - np.round(prices * 100)) <= 0.05
            if np.any(psych_mask):
                sr_confluence[idx] = float(strengths[psych_mask].max())

        # Add suffix to avoid overwrites when using multiple windows/prominence values
        suffix = f"_W{self.window}"
        return pd.DataFrame(
            {
                f"SR_SUPPORT_STRENGTH{suffix}": sr_support,
                f"SR_RESISTANCE_STRENGTH{suffix}": sr_resistance,
                f"SR_PSYCHO_CONFLUENCE{suffix}": sr_confluence,
                f"SR_DISTANCE{suffix}": pd.Series(sr_distance, index=frame.index).replace(np.inf, np.nan),
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
