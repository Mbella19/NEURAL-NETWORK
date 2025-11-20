"""Curriculum Phase 4: Smart money concepts (Phase 5.5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase4Config:
    feature_subset: Tuple[str, ...] = (
        "BOS_BULLISH",
        "BOS_BEARISH",
        "CHOCH_BULLISH",
        "CHOCH_BEARISH",
        "ORDER_BLOCK_BULL",
        "ORDER_BLOCK_BEAR",
        "LIQUIDITY_SWEEP",
        "FVG_UP",
        "FVG_DOWN",
    )
    threshold: float = 0.64
    forecast_horizon: int = 20  # Medium horizon (context)


class Phase4SmartMoneyTask:
    def __init__(self, config: Phase4Config | None = None) -> None:
        self.config = config or Phase4Config()
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_dim = 3

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        base_df = feature_frame if feature_frame is not None else run_feature_pipeline(
            frame, pipeline=pipeline
        ).dataframe
        return base_df.loc[:, self.config.feature_subset].copy()

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        if feature_frame is None:
            raise ValueError("feature_frame is required to avoid recomputing SMC features with future data.")
        base_df = feature_frame
        horizon = self._horizon

        # Handle both suffixed and unsuffixed column names
        # attach_mapping creates aliases, but check both to be safe
        fvg_up_col = "FVG_UP" if "FVG_UP" in base_df.columns else "FVG_UP_SW5"
        fvg_down_col = "FVG_DOWN" if "FVG_DOWN" in base_df.columns else "FVG_DOWN_SW5"

        if fvg_up_col not in base_df.columns:
            raise KeyError(f"FVG_UP column not found. Available columns: {list(base_df.columns[:20])}")
        if fvg_down_col not in base_df.columns:
            raise KeyError(f"FVG_DOWN column not found. Available columns: {list(base_df.columns[:20])}")

        classes = []
        for step in range(1, horizon + 1):
            fvg_up = base_df[fvg_up_col].shift(-step).fillna(0)
            fvg_down = base_df[fvg_down_col].shift(-step).fillna(0)
            cls = (torch.tensor(fvg_up.values, dtype=torch.long) - torch.tensor(fvg_down.values, dtype=torch.long)) + 1
            classes.append(cls.clamp(0, 2))
        stacked = torch.stack(classes, dim=1)
        # Use majority class over horizon to keep output_dim stable.
        mode_vals, _ = torch.mode(stacked, dim=1)
        return mode_vals

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        loss = self.loss_fn(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        return {"accuracy": accuracy, "loss": loss}

    @property
    def _horizon(self) -> int:
        return max(1, min(10, self.config.forecast_horizon))
