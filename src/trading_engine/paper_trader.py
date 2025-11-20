"""Paper trading interface (Phase 8.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import pandas as pd
import torch

from models.base_model import BaseModel
from .dashboard import TradingDashboard
from .order_manager import OrderManager
from .risk_manager import RiskManager, RiskConfig
from .safety import SafetySystem, SafetyConfig
from utils.drift import DriftDetector


@dataclass
class PaperTraderConfig:
    price_column: str = "CLOSE"
    signal_threshold: float = 0.5
    spread_pips: float = 1.2
    slippage_pips: float = 0.5
    latency_seconds: float = 0.8
    pip_scale: float = 1e-4  # EURUSD pip converter


class PaperTrader:
    def __init__(
        self,
        model: BaseModel,
        config: PaperTraderConfig | None = None,
        risk_manager: Optional[RiskManager] = None,
        order_manager: Optional[OrderManager] = None,
        safety_system: Optional[SafetySystem] = None,
        dashboard: Optional[TradingDashboard] = None,
        drift_detector: Optional[DriftDetector] = None,
    ) -> None:
        self.model = model
        self.config = config or PaperTraderConfig()
        self.positions = []
        self.equity_curve = []
        self.risk_manager = risk_manager or RiskManager(RiskConfig())
        self.order_manager = order_manager or OrderManager()
        self.safety = safety_system or SafetySystem(SafetyConfig())
        self.dashboard = dashboard or TradingDashboard()
        self.drift_detector = drift_detector or DriftDetector()

    def run(self, data: pd.DataFrame, feature_fn: Callable[[pd.DataFrame], torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        equity = 1.0
        last_price = data[self.config.price_column].iloc[0]
        for _, row in data.iterrows():
            x = feature_fn(pd.DataFrame([row]))
            signal = torch.sigmoid(self.model.predict(x)).item()
            raw_price = row[self.config.price_column]
            # Latency + slippage model: adjust execution price against the trade direction.
            slip = (self.config.slippage_pips + self.config.latency_seconds * 0.25) * self.config.pip_scale
            spread = self.config.spread_pips * self.config.pip_scale
            position = 1 if signal > self.config.signal_threshold else -1
            exec_price = raw_price + slip * position + (spread / 2) * position

            size = self.risk_manager.position_size(equity, row.get("ATR", 0.0005), exec_price)
            order = self.order_manager.place_order("long" if position == 1 else "short", size, exec_price)

            # Apply transaction costs per trade
            txn_cost = spread * size
            pnl = position * size * (exec_price - last_price) / last_price - txn_cost
            equity *= (1 + pnl)
            self.order_manager.fill_order(order.id)
            self.positions.append(position)
            self.equity_curve.append(equity)
            last_price = exec_price
            alerts = {}
            if self.drift_detector.update(float(pnl)):
                alerts["drift"] = "concept drift detected; halting trades"
                break
            if not self.risk_manager.update_equity(equity):
                alerts["drawdown"] = "max drawdown exceeded"
                break
            if not self.safety.check_daily_loss(equity):
                alerts["daily_loss"] = "daily loss limit hit"
                break
            self.dashboard.update(equity, position, alerts)
        return {
            "final_equity": equity,
            "total_return": equity - 1,
            "max_drawdown": float(min(self.equity_curve) / max(self.equity_curve) - 1),
        }


__all__ = ["PaperTrader", "PaperTraderConfig"]
