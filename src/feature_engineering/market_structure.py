"""Market structure feature calculator (Phase 3.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .base import BaseFeatureCalculator


@dataclass
class StructureState:
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    last_label: str = "NONE"
    trend: int = 0  # 1 up, -1 down, 0 undefined


class MarketStructureFeatures(BaseFeatureCalculator):
    name = "market_structure"
    required_columns = ("HIGH", "LOW", "CLOSE")
    # Note: produces is dynamically generated in __init__ to include suffix

    def __init__(self, *, lookback: int = 5, tolerance: float = 1e-6) -> None:
        super().__init__(window=lookback)
        self.lookback = lookback
        self.tolerance = tolerance

        # Dynamically set produces with suffix
        suffix = f"_L{self.lookback}"
        self.produces = (
            f"SWING_HIGH{suffix}",
            f"SWING_LOW{suffix}",
            f"STRUCTURE_LABEL{suffix}",
            f"TREND_DIRECTION{suffix}",
            f"STRUCTURE_STRENGTH{suffix}",
        )

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        high = frame["HIGH"]
        low = frame["LOW"]

        rolling_high = high.rolling(window=self.lookback, min_periods=1).max()
        rolling_low = low.rolling(window=self.lookback, min_periods=1).min()

        swing_high = ((high - rolling_high).abs() <= self.tolerance).astype(int)
        swing_low = ((low - rolling_low).abs() <= self.tolerance).astype(int)

        labels, trend, strength = self._build_structure_labels(high, low, swing_high, swing_low)

        suffix = f"_L{self.lookback}"
        return pd.DataFrame(
            {
                f"SWING_HIGH{suffix}": swing_high,
                f"SWING_LOW{suffix}": swing_low,
                f"STRUCTURE_LABEL{suffix}": labels,
                f"TREND_DIRECTION{suffix}": trend,
                f"STRUCTURE_STRENGTH{suffix}": strength,
            },
            index=frame.index,
        )

    def _build_structure_labels(
        self,
        high: pd.Series,
        low: pd.Series,
        swing_high: pd.Series,
        swing_low: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        state = StructureState()
        labels: List[str] = []
        trend_dir: List[int] = []
        strengths: List[float] = []

        for idx in range(len(high)):
            label = state.last_label
            trend = state.trend
            strength = 0.0

            if swing_high.iat[idx]:
                price = high.iat[idx]
                if state.last_swing_high is None or price > state.last_swing_high + self.tolerance:
                    label = "HH"
                    trend = 1
                    denom = max(abs(price), 1.0)
                    strength = (price - (state.last_swing_high or price)) / denom
                else:
                    label = "LH"
                    trend = -1 if trend <= 0 else trend
                    denom = max(abs(state.last_swing_high), 1.0)
                    strength = (state.last_swing_high - price) / denom
                state.last_swing_high = price
            elif swing_low.iat[idx]:
                price = low.iat[idx]
                if state.last_swing_low is None or price < state.last_swing_low - self.tolerance:
                    label = "LL"
                    trend = -1
                    base = state.last_swing_low if state.last_swing_low is not None else price
                    denom = max(abs(base), 1.0)
                    strength = (base - price) / denom
                else:
                    label = "HL"
                    trend = 1 if trend >= 0 else trend
                    denom = max(abs(price), 1.0)
                    strength = (price - state.last_swing_low) / denom
                state.last_swing_low = price

            state.last_label = label
            state.trend = trend

            labels.append(label)
            trend_dir.append(trend)
            strengths.append(strength)

        return pd.Series(labels, index=high.index), pd.Series(trend_dir, index=high.index), pd.Series(
            strengths, index=high.index
        )


__all__ = ["MarketStructureFeatures"]
