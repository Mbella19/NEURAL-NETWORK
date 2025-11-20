"""Wick analysis feature calculator (Phase 3.2)."""
from __future__ import annotations

import pandas as pd

from .base import BaseFeatureCalculator


class WickFeatures(BaseFeatureCalculator):
    name = "wick_features"
    required_columns = ("OPEN", "HIGH", "LOW", "CLOSE")
    produces = (
        "BODY_SIZE",
        "UPPER_WICK",
        "LOWER_WICK",
        "CANDLE_RANGE",
        "UPPER_WICK_RATIO",
        "LOWER_WICK_RATIO",
        "BODY_RATIO",
        "WICK_DOMINANCE",
    )

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        body = (frame["CLOSE"] - frame["OPEN"]).abs()
        upper = frame["HIGH"] - frame[["OPEN", "CLOSE"]].max(axis=1)
        lower = frame[["OPEN", "CLOSE"]].min(axis=1) - frame["LOW"]
        candle_range = frame["HIGH"] - frame["LOW"]

        range_safe = candle_range.replace(0, pd.NA)
        upper_ratio = upper / range_safe
        lower_ratio = lower / range_safe
        body_ratio = body / range_safe

        dominance = (upper - lower) / candle_range.replace(0, pd.NA)

        return pd.DataFrame(
            {
                "BODY_SIZE": body,
                "UPPER_WICK": upper,
                "LOWER_WICK": lower,
                "CANDLE_RANGE": candle_range,
                "UPPER_WICK_RATIO": upper_ratio.fillna(0.0),
                "LOWER_WICK_RATIO": lower_ratio.fillna(0.0),
                "BODY_RATIO": body_ratio.fillna(0.0),
                "WICK_DOMINANCE": dominance.fillna(0.0),
            },
            index=frame.index,
        )


__all__ = ["WickFeatures"]
