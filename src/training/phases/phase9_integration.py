"""Curriculum Phase 9: Integration (Phase 5.9)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase9IntegrationConfig:
    threshold: float = 0.72
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    forecast_horizon: int = 30  # Longer horizon (integration)


class Phase9IntegrationTask:
    def __init__(self, config: Phase9IntegrationConfig | None = None) -> None:
        self.config = config or Phase9IntegrationConfig()
        self.quantiles = torch.tensor(self.config.quantiles, dtype=torch.float32)
        self.loss_fn = self._quantile_loss
        self.output_dim = self._horizon * len(self.quantiles)

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        result = (
            feature_frame
            if feature_frame is not None
            else run_feature_pipeline(frame, pipeline=pipeline).dataframe
        )

        # Filter out non-numeric columns (e.g., STRUCTURE_LABEL_L5 which contains strings)
        numeric_cols = result.select_dtypes(include=["number"]).columns
        df_numeric = result[numeric_cols].copy()
        df_numeric = df_numeric.replace([float("inf"), float("-inf")], 0).fillna(0)

        return df_numeric

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        close = frame["CLOSE"]
        horizon = self._horizon
        returns: List[torch.Tensor] = []
        for step in range(1, horizon + 1):
            step_ret = ((close.shift(-step) - close) / close).fillna(0) * 100.0
            returns.append(torch.tensor(step_ret.values, dtype=torch.float32).clamp(min=-10.0, max=10.0))
        stacked = torch.stack(returns, dim=1)  # [N, H]
        # Repeat along quantiles for pinball loss; flatten to [N, H*Q]
        expanded = stacked.unsqueeze(-1).repeat(1, 1, len(self.quantiles))
        return expanded.reshape(expanded.shape[0], -1)

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        q = len(self.quantiles)
        horizon = self._horizon
        preds = logits.reshape(-1, horizon, q)
        t = targets.reshape(-1, horizon, q)[..., 0]  # true returns
        median = preds[..., self.quantiles.tolist().index(0.5) if 0.5 in self.quantiles.tolist() else 0]
        mse = torch.mean((median - t) ** 2).item()
        directional = ((torch.sign(median) == torch.sign(t)).float().mean().item())
        risk = self._risk_metrics(t)
        return {"mse": mse, "directional_accuracy": directional, **risk}

    @property
    def _horizon(self) -> int:
        # FIX: Removed arbitrary cap at 10 - use configured forecast_horizon
        return max(1, self.config.forecast_horizon)

    def _quantile_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        q = self.quantiles.to(logits.device)
        qn = len(q)
        preds = logits.reshape(targets.shape[0], -1, qn)
        target = targets.reshape_as(preds)[..., 0]  # broadcast true return
        # pinball loss
        errors = target.unsqueeze(-1) - preds
        loss = torch.maximum(q * errors, (q - 1) * errors)
        return loss.mean()

    @staticmethod
    def _risk_metrics(returns: torch.Tensor) -> Dict[str, float]:
        # returns: [batch, horizon]
        eq = torch.cumprod(1 + returns / 100.0, dim=1)
        peak = torch.maximum.accumulate(eq, dim=1)
        dd = (peak - eq) / peak
        max_dd = dd.max(dim=1).values.mean().item()
        tail = torch.quantile(returns, 0.05).item()
        time_in_dd = (dd > 0).float().mean().item()
        return {"max_drawdown_est": max_dd, "tail_loss_p5_est": tail, "time_in_drawdown_est": time_in_dd}
