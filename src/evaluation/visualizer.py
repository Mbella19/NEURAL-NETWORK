"""Visualization utilities (Phase 7.3)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curves(history: pd.DataFrame) -> None:
    history.plot(figsize=(10, 5))
    plt.title("Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.show()


def plot_equity_curve(equity: pd.Series) -> None:
    equity.plot(figsize=(10, 5))
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()


def plot_drawdown(equity: pd.Series) -> None:
    cumulative_max = equity.cummax()
    drawdown = (equity - cumulative_max) / cumulative_max
    drawdown.plot(figsize=(10, 4), color="red")
    plt.title("Drawdown")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.show()


__all__ = ["plot_drawdown", "plot_equity_curve", "plot_learning_curves"]
