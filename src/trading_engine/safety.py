"""Safety controls (Phase 8.4)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyConfig:
    daily_loss_limit: float = 0.05
    anomaly_threshold: float = 3.0


class SafetySystem:
    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self.starting_equity = 1.0

    def check_daily_loss(self, current_equity: float) -> bool:
        loss = (self.starting_equity - current_equity) / self.starting_equity
        return loss <= self.config.daily_loss_limit

    def detect_anomaly(self, metric: float, mean: float, std: float) -> bool:
        if std == 0:
            return False
        z_score = abs(metric - mean) / std
        return z_score > self.config.anomaly_threshold
