"""Market regime classification (Phase 3.9)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .base import BaseFeatureCalculator
try:  # pragma: no cover - optional dependency for richer regimes
    from hmmlearn import hmm  # type: ignore
    _HMM_AVAILABLE = True
except Exception:  # pragma: no cover
    _HMM_AVAILABLE = False


class MarketRegimeFeatures(BaseFeatureCalculator):
    name = "market_regime"
    required_columns = ("CLOSE",)
    produces = (
        "TREND_STRENGTH",
        "TREND_DIRECTION",
        "VOLATILITY_REGIME",
        "MARKET_EFFICIENCY",
        "REGIME_TRANSITION",
        "REGIME_STATE",
        "REGIME_HMM_STATE",
    )

    def __init__(
        self,
        *,
        momentum_window: int = 20,
        volatility_window: int = 20,
        efficiency_window: int = 20,
        volatility_threshold: float = 0.0005,
    ) -> None:
        super().__init__(window=momentum_window)
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.efficiency_window = efficiency_window
        self.volatility_threshold = volatility_threshold

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["CLOSE"]
        returns = close.pct_change()

        momentum = close.diff(self.momentum_window)
        trend_strength = (momentum / close.shift(self.momentum_window)).fillna(0.0)
        trend_direction = np.sign(trend_strength)

        volatility = returns.rolling(window=self.volatility_window, min_periods=1).std().fillna(0.0)
        volatility_regime = (volatility > self.volatility_threshold).astype(int)

        efficiency = self._efficiency_ratio(close, self.efficiency_window)
        regime_transition = volatility_regime.diff().fillna(0).abs()

        # Multi-state regime label (6 buckets: bull/bear/range x low/high vol)
        trend_label = np.sign(trend_strength)
        regime_state = []
        for t_dir, vol_flag in zip(trend_label, volatility_regime):
            if abs(t_dir) < 1e-6:
                label = 2 if vol_flag else 3  # range/high vol vs range/low vol
            elif t_dir > 0:
                label = 0 if not vol_flag else 1  # bull low vol / bull high vol
            else:
                label = 4 if not vol_flag else 5  # bear low vol / bear high vol
            regime_state.append(label)

        # Optional Hidden Markov Model refinement if available
        if _HMM_AVAILABLE and len(returns.dropna()) > 100:
            feats = np.column_stack(
                [
                    returns.fillna(0).values,
                    volatility.fillna(0).values,
                    efficiency.fillna(0).values,
                ]
            )
            model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=20)
            try:
                model.fit(feats)
                hmm_states = model.predict(feats)
            except Exception:
                hmm_states = np.full(len(frame), -1)
        else:
            hmm_states = np.full(len(frame), -1)

        return pd.DataFrame(
            {
                "TREND_STRENGTH": trend_strength,
                "TREND_DIRECTION": trend_direction,
                "VOLATILITY_REGIME": volatility_regime,
                "MARKET_EFFICIENCY": efficiency,
                "REGIME_TRANSITION": regime_transition,
                "REGIME_STATE": pd.Series(regime_state, index=frame.index),
                "REGIME_HMM_STATE": pd.Series(hmm_states, index=frame.index),
            },
            index=frame.index,
        )

    @staticmethod
    def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
        change = (close - close.shift(window)).abs()
        volatility = close.diff().abs().rolling(window=window, min_periods=1).sum()
        ratio = change / volatility.replace(0, pd.NA)
        return ratio.fillna(0.0)


__all__ = ["MarketRegimeFeatures"]
