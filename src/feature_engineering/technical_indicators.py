"""Technical indicator features (Phase 3.6)."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from .base import BaseFeatureCalculator


class TechnicalIndicatorFeatures(BaseFeatureCalculator):
    name = "technical_indicators"
    required_columns = ("OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL")
    produces = ()

    def __init__(
        self,
        sma_windows: Iterable[int] = (10, 20, 50),
        ema_windows: Iterable[int] = (12, 26),
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_k: int = 14,
        stoch_d: int = 3,
        atr_window: int = 14,
        adx_window: int = 14,
    ) -> None:
        super().__init__()
        self.sma_windows = tuple(sma_windows)
        self.ema_windows = tuple(ema_windows)
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.atr_window = atr_window
        self.adx_window = adx_window

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["CLOSE"]
        high = frame["HIGH"]
        low = frame["LOW"]
        volume = frame["TICKVOL"]

        features: Dict[str, pd.Series] = {}

        for window in self.sma_windows:
            sma = close.rolling(window=window, min_periods=window).mean().bfill()
            features[f"SMA_{window}"] = (close - sma) / sma
        for window in self.ema_windows:
            ema = close.ewm(span=window, adjust=False).mean()
            features[f"EMA_{window}"] = (close - ema) / ema

        features["RSI"] = self._rsi(close, self.rsi_window)

        macd_line = close.ewm(span=self.macd_fast, adjust=False).mean() - close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_signal = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        features["MACD_LINE"] = macd_line
        features["MACD_SIGNAL"] = macd_signal
        features["MACD_HIST"] = macd_line - macd_signal

        features["STOCH_K"] = self._stochastic_k(high, low, close, self.stoch_k)
        features["STOCH_D"] = features["STOCH_K"].rolling(window=self.stoch_d, min_periods=self.stoch_d).mean().bfill()

        features["ATR"] = self._atr(high, low, close, self.atr_window)
        features["ADX"] = self._adx(high, low, close, self.adx_window)

        features["VOLUME_MA_20"] = volume.rolling(window=20, min_periods=1).mean()
        features["EFFICIENCY_RATIO"] = self._efficiency_ratio(close, window=20)

        return pd.DataFrame(features, index=frame.index)

    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
        loss = -delta.clip(upper=0).rolling(window=window, min_periods=window).mean()
        rs = gain / (loss.replace(0, pd.NA))
        rsi = 100 - (100 / (1 + rs.abs()))
        return rsi.ffill().fillna(50.0)

    @staticmethod
    def _stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        lowest = low.rolling(window=window, min_periods=window).min()
        highest = high.rolling(window=window, min_periods=window).max()
        k = 100 * (close - lowest) / (highest - lowest).replace(0, pd.NA)
        return k.ffill().fillna(50.0)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=window, min_periods=window).mean()
        return atr.fillna(0.0)

    @staticmethod
    def _dx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

        atr = TechnicalIndicatorFeatures._atr(high, low, close, window)
        plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).sum() / atr.replace(0, pd.NA))
        minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).sum() / atr.replace(0, pd.NA))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, pd.NA)) * 100
        return dx.fillna(0.0)

    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        dx = self._dx(high, low, close, window)
        adx = dx.rolling(window=window, min_periods=window).mean()
        return adx.fillna(0.0)

    @staticmethod
    def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
        change = (close - close.shift(window)).abs()
        volatility = close.diff().abs().rolling(window=window, min_periods=window).sum()
        ratio = change / volatility.replace(0, pd.NA)
        return ratio.fillna(0.0)


__all__ = ["TechnicalIndicatorFeatures"]
