"""Evaluation metrics (Phase 7.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, eps: float = 1e-9) -> float:
    excess = returns - risk_free
    return float(excess.mean() / (excess.std(ddof=1) + eps))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0, eps: float = 1e-9) -> float:
    excess = returns - risk_free
    downside = excess[excess < 0]
    denom = downside.std(ddof=1) if len(downside) else eps
    return float(excess.mean() / (denom + eps))


def calmar_ratio(returns: np.ndarray) -> float:
    cum_returns = np.cumprod(1 + returns) - 1
    max_drawdown = np.max(np.maximum.accumulate(cum_returns) - cum_returns)
    annual_return = (1 + returns.mean()) ** 252 - 1
    return float(annual_return / (max_drawdown + 1e-9))


def max_drawdown(equity_curve: np.ndarray) -> float:
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cumulative_max) / (cumulative_max + 1e-9)
    return float(drawdown.min())


def precision_recall_f1(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    tp = np.logical_and(preds == 1, targets == 1).sum()
    fp = np.logical_and(preds == 1, targets == 0).sum()
    fn = np.logical_and(preds == 0, targets == 1).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def value_at_risk(returns: np.ndarray, alpha: float = 0.95) -> float:
    return float(np.quantile(returns, 1 - alpha))


def conditional_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    var = value_at_risk(returns, alpha)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else var


__all__ = [
    "calmar_ratio",
    "conditional_var",
    "max_drawdown",
    "precision_recall_f1",
    "sharpe_ratio",
    "sortino_ratio",
    "value_at_risk",
]
