"""Evaluation package exports."""

from .metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    precision_recall_f1,
    value_at_risk,
    conditional_var,
)
from .backtester import Backtester, BacktestConfig
from .explainability import saliency_map, shap_values, lime_explanation

__all__ = [
    # Metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "precision_recall_f1",
    "value_at_risk",
    "conditional_var",
    # Backtesting
    "Backtester",
    "BacktestConfig",
    # Explainability
    "saliency_map",
    "shap_values",
    "lime_explanation",
]
