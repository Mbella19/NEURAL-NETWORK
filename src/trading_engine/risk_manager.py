"""Risk management utilities (Phase 8.2)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    max_drawdown: float = 0.2
    risk_per_trade: float = 0.01
    exposure_limit: float = 0.2
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.equity = 1.0
        self.peak_equity = 1.0

    def position_size(self, equity: float, atr: float, price: float) -> float:
        """ATR-aware sizing with exposure cap to avoid runaway lot sizes."""

        atr_pct = atr / max(price, 1e-9)  # normalize ATR to percentage move
        effective_risk_per_unit = price * atr_pct * self.config.stop_loss_atr
        dollar_risk = equity * self.config.risk_per_trade

        if effective_risk_per_unit <= 0:
            return 0.0

        units_by_risk = dollar_risk / effective_risk_per_unit
        units_by_exposure = (equity * self.config.exposure_limit) / max(price, 1e-9)
        # Extra guardrail: never exceed 10 standard lots (1.0m notional) on retail constraints.
        return float(min(units_by_risk, units_by_exposure, 1_000_000))

    def update_equity(self, equity: float) -> bool:
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        return drawdown <= self.config.max_drawdown

    def kelly_fraction(self, win_rate: float, avg_win_pips: float, avg_loss_pips: float, *, fraction: float = 0.5) -> float:
        """Compute Kelly fraction (default half-Kelly)."""

        if avg_win_pips <= 0 or avg_loss_pips <= 0:
            return 0.0
        kelly = (win_rate * avg_win_pips - (1 - win_rate) * avg_loss_pips) / avg_win_pips
        return max(0.0, kelly * fraction)

    def kelly_position_size(
        self,
        equity: float,
        price: float,
        win_rate: float,
        avg_win_pips: float,
        avg_loss_pips: float,
        pip_scale: float = 1e-4,
        fraction: float = 0.5,
    ) -> float:
        """Kelly-based sizing using pip estimates."""

        k = self.kelly_fraction(win_rate, avg_win_pips, avg_loss_pips, fraction=fraction)
        dollar_risk = equity * k
        pip_value_per_unit = pip_scale * price
        if pip_value_per_unit <= 0:
            return 0.0
        units = dollar_risk / (avg_loss_pips * pip_value_per_unit)
        exposure_cap = (equity * self.config.exposure_limit) / max(price, 1e-9)
        return float(max(0.0, min(units, exposure_cap, 1_000_000)))


__all__ = ["RiskConfig", "RiskManager"]
