"""Policy execution head built on triple-barrier-style trade outcome."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class PolicyExecutionConfig:
    time_limit: int = 50           # vertical barrier (bars)
    take_profit: float = 0.0020    # ~20 pips on FX-style pricing
    stop_loss: float = 0.0010      # ~10 pips


class PolicyExecutionTask:
    """Labels a trade outcome by which barrier hits first."""

    def __init__(self, config: PolicyExecutionConfig | None = None) -> None:
        self.config = config or PolicyExecutionConfig()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.output_dim = 1

    def prepare_batch(self, frame, feature_frame=None):
        # This task uses backbone features only; nothing to mask here.
        return feature_frame

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        close = frame["CLOSE"].reset_index(drop=True)
        high = frame["HIGH"].reset_index(drop=True)
        low = frame["LOW"].reset_index(drop=True)

        tp_mult = 1.0 + self.config.take_profit
        sl_mult = 1.0 - self.config.stop_loss
        horizon = max(1, self.config.time_limit)

        targets = np.zeros(len(close), dtype=np.float32)

        for idx in range(len(close)):
            if idx + 1 >= len(close):
                break
            end = min(len(close), idx + horizon + 1)
            window_high = high.iloc[idx + 1 : end].values
            window_low = low.iloc[idx + 1 : end].values

            tp_level = close.iloc[idx] * tp_mult
            sl_level = close.iloc[idx] * sl_mult

            # Find first touch of TP or SL
            tp_hits = np.where(window_high >= tp_level)[0]
            sl_hits = np.where(window_low <= sl_level)[0]

            tp_first = tp_hits[0] if tp_hits.size > 0 else None
            sl_first = sl_hits[0] if sl_hits.size > 0 else None

            if tp_first is None and sl_first is None:
                targets[idx] = 0.0  # no trade success
            elif tp_first is not None and sl_first is None:
                targets[idx] = 1.0
            elif tp_first is None and sl_first is not None:
                targets[idx] = 0.0
            else:
                targets[idx] = 1.0 if tp_first <= sl_first else 0.0

        return torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds = torch.sigmoid(logits)
        accuracy = ((preds > 0.5) == (targets > 0.5)).float().mean().item()
        loss = self.loss_fn(logits, targets).item()
        return {"accuracy": accuracy, "loss": loss}

    @property
    def _horizon(self) -> int:
        return self.config.time_limit
