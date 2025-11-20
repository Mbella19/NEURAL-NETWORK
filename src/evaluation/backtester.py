"""Backtesting engine (Phase 7.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .metrics import max_drawdown, sharpe_ratio


@dataclass
class BacktestConfig:
    transaction_cost: float = 0.0002
    slippage: float = 0.0001
    price_column: str = "CLOSE"
    vol_window: int = 20
    max_risk_per_trade: float = 0.02
    max_risk_per_day: float = 0.05


class Backtester:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, price_data: pd.DataFrame, signals: pd.Series) -> Dict[str, float]:
        prices = price_data[self.config.price_column].values
        returns = np.diff(prices) / prices[:-1]
        aligned_signals = signals.values[1:]
        aligned_signals = np.asarray(aligned_signals, dtype=float)

        # Dynamic position sizing based on confidence and volatility.
        vol = pd.Series(returns).rolling(self.config.vol_window, min_periods=1).std().values
        base_conf = np.clip(np.abs(aligned_signals), 0, 1)
        direction = np.sign(aligned_signals)
        size = base_conf * np.clip(self.config.max_risk_per_trade / (vol + 1e-6), 0, 1)

        # Enforce per-day risk cap: zero out positions once daily PnL breaches the cap.
        # Handle both DatetimeIndex and RangeIndex
        if isinstance(price_data.index, pd.DatetimeIndex):
            timestamps = price_data.index[1:]
            day_ids = timestamps.date
        elif "TIMESTAMP" in price_data.columns:
            timestamps = pd.to_datetime(price_data["TIMESTAMP"].values[1:])
            day_ids = timestamps.date
        else:
            # Fallback: treat each row as a separate day (no daily risk grouping)
            timestamps = np.arange(len(returns))
            day_ids = timestamps  # Each timestep is its own "day"
        daily_pnl = 0.0
        pnl_list = []
        equity = []
        eq = 1.0
        for i, r in enumerate(returns):
            if i > 0 and day_ids[i] != day_ids[i - 1]:
                daily_pnl = 0.0
            trade_cost = (self.config.transaction_cost + self.config.slippage) * size[i]
            step_pnl = direction[i] * size[i] * r - trade_cost
            daily_pnl += step_pnl
            if daily_pnl < -self.config.max_risk_per_day:
                step_pnl = -self.config.transaction_cost - self.config.slippage  # flat
            pnl_list.append(step_pnl)
            eq *= (1 + step_pnl)
            equity.append(eq)
        pnl = np.asarray(pnl_list)
        equity = np.asarray(equity)

        # Risk diagnostics
        if isinstance(price_data.index, pd.DatetimeIndex):
            eq_series = pd.Series(equity, index=price_data.index[1:])
            monthly_returns = eq_series.resample("M").last().pct_change().dropna()
            worst_month = float(monthly_returns.min()) if not monthly_returns.empty else 0.0
        elif "TIMESTAMP" in price_data.columns:
            eq_series = pd.Series(equity, index=pd.to_datetime(price_data["TIMESTAMP"].values[1:]))
            monthly_returns = eq_series.resample("M").last().pct_change().dropna()
            worst_month = float(monthly_returns.min()) if not monthly_returns.empty else 0.0
        else:
            # Can't compute monthly returns without timestamps
            worst_month = 0.0
        dd = max_drawdown(equity)
        underwater = 1 - equity / np.maximum.accumulate(equity)
        time_in_dd = float((underwater > 0).mean())
        tail_loss = float(np.percentile(pnl, 5)) if len(pnl) else 0.0

        return {
            "total_return": float(equity[-1] - 1),
            "sharpe": sharpe_ratio(pnl),
            "max_drawdown": dd,
            "worst_month": worst_month,
            "time_in_drawdown": time_in_dd,
            "tail_loss_p5": tail_loss,
        }


__all__ = ["Backtester", "BacktestConfig"]
