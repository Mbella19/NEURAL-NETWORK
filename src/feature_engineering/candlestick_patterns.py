"""Candlestick pattern detection (Phase 3.3)."""
from __future__ import annotations

import pandas as pd

from .base import BaseFeatureCalculator


class CandlestickPatternFeatures(BaseFeatureCalculator):
    name = "candlestick_patterns"
    required_columns = ("OPEN", "HIGH", "LOW", "CLOSE")
    produces = (
        "DOJI",
        "SPINNING_TOP",
        "HAMMER",
        "SHOOTING_STAR",
        "MARUBOZU",
        "BULLISH_ENGULFING",
        "BEARISH_ENGULFING",
        "PATTERN_STRENGTH",
    )

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        high = frame["HIGH"]
        low = frame["LOW"]
        open_ = frame["OPEN"]
        close = frame["CLOSE"]

        body = (close - open_).abs()
        candle_range = (high - low).replace(0, pd.NA)
        upper = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower = pd.concat([open_, close], axis=1).min(axis=1) - low

        body_ratio = (body / candle_range).fillna(0.0)
        upper_ratio = (upper / candle_range).fillna(0.0)
        lower_ratio = (lower / candle_range).fillna(0.0)

        # Base patterns
        doji = (body_ratio <= 0.1).astype(float)
        spinning_top = (
            (body_ratio <= 0.3) & (upper_ratio >= 0.25) & (lower_ratio >= 0.25)
        ).astype(float)
        hammer = (
            (lower_ratio >= 0.6) & (upper_ratio <= 0.1) & (close >= open_)
        ).astype(float)
        shooting_star = (
            (upper_ratio >= 0.6) & (lower_ratio <= 0.1) & (close <= open_)
        ).astype(float)
        marubozu = (body_ratio >= 0.9).astype(float)

        # Engulfing patterns require prior candle
        prev_open = open_.shift(1)
        prev_close = close.shift(1)

        bullish_engulfing = (
            (close > open_)
            & (prev_close < prev_open)
            & (close >= prev_open)
            & (open_ <= prev_close)
        ).astype(float)
        bearish_engulfing = (
            (close < open_)
            & (prev_close > prev_open)
            & (close <= prev_open)
            & (open_ >= prev_close)
        ).astype(float)

        pattern_strength = (
            0.4 * hammer
            + 0.4 * shooting_star
            + 0.3 * marubozu
            + 0.2 * bullish_engulfing
            + 0.2 * bearish_engulfing
            + 0.1 * spinning_top
        )

        return pd.DataFrame(
            {
                "DOJI": doji,
                "SPINNING_TOP": spinning_top,
                "HAMMER": hammer,
                "SHOOTING_STAR": shooting_star,
                "MARUBOZU": marubozu,
                "BULLISH_ENGULFING": bullish_engulfing,
                "BEARISH_ENGULFING": bearish_engulfing,
                "PATTERN_STRENGTH": pattern_strength,
            },
            index=frame.index,
        )


__all__ = ["CandlestickPatternFeatures"]
