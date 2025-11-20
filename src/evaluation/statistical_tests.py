"""Statistical significance testing (Phase 7.4)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def t_test_performance(returns: np.ndarray, benchmark: float = 0.0) -> float:
    t_stat, p_value = stats.ttest_1samp(returns, benchmark)
    return float(p_value)


def monte_carlo_simulation(returns: np.ndarray, n: int = 1000) -> np.ndarray:
    sims = np.random.choice(returns, size=(n, len(returns)), replace=True)
    return sims.mean(axis=1)


def bootstrap_confidence_interval(returns: np.ndarray, alpha: float = 0.95, n: int = 1000) -> tuple[float, float]:
    sims = monte_carlo_simulation(returns, n)
    lower = np.quantile(sims, (1 - alpha) / 2)
    upper = np.quantile(sims, 1 - (1 - alpha) / 2)
    return float(lower), float(upper)


__all__ = ["bootstrap_confidence_interval", "monte_carlo_simulation", "t_test_performance"]
