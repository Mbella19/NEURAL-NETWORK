"""Monitoring dashboard (Phase 8.5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class DashboardState:
    equity: float
    positions: int
    alerts: Dict[str, str]


class TradingDashboard:
    def __init__(self) -> None:
        self.history = []

    def update(self, equity: float, position: int, alerts: Dict[str, str]) -> DashboardState:
        state = DashboardState(equity=equity, positions=position, alerts=alerts)
        self.history.append(state)
        return state

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([state.__dict__ for state in self.history])


__all__ = ["TradingDashboard", "DashboardState"]
