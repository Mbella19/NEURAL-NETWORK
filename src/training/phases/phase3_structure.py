"""Curriculum Phase 3: Market structure (Phase 5.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline


@dataclass
class Phase3Config:
    feature_subset: Tuple[str, ...] = (
        "SWING_HIGH_L5",
        "SWING_LOW_L5",
        # "STRUCTURE_LABEL_L5",  # Removed - this is the target, not a feature (contains strings)
        "TREND_DIRECTION_L5",
        "STRUCTURE_STRENGTH_L5",
        "BOS_BULLISH",
        "BOS_BEARISH",
    )
    threshold: float = 0.62
    forecast_horizon: int = 20  # Medium horizon (structure)


class Phase3StructureTask:
    def __init__(self, config: Phase3Config | None = None) -> None:
        self.config = config or Phase3Config()
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_dim = 5

    def prepare_batch(self, frame, feature_frame=None):
        pipeline = build_default_feature_pipeline()
        base_df = feature_frame if feature_frame is not None else run_feature_pipeline(
            frame, pipeline=pipeline
        ).dataframe
        df = base_df.loc[:, self.config.feature_subset].copy()

        # FIX: Map string structure labels to integers explicitly
        if "STRUCTURE_LABEL_L5" in df.columns:
            mapping = {"HH": 1, "HL": 2, "LH": -1, "LL": -2, "NONE": 0}
            # Apply map and handle unexpected values
            df["STRUCTURE_LABEL_L5"] = df["STRUCTURE_LABEL_L5"].map(mapping).fillna(0)

        # Check for non-numeric columns and convert if needed
        for col in df.columns:
            if df[col].dtype == "object":
                print(f"Warning: Column '{col}' has object dtype, attempting conversion to numeric")
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def compute_targets(self, frame, feature_frame=None) -> torch.Tensor:
        if feature_frame is None:
            raise ValueError("feature_frame is required to avoid recomputing structure features with future data.")
        labels = feature_frame["STRUCTURE_LABEL_L5"].fillna("NONE")
        mapping = {"HH": 0, "HL": 1, "LH": 2, "LL": 3, "NONE": 4}
        mapped = labels.map(mapping)
        horizon = self._horizon
        rolled = []
        for step in range(1, horizon + 1):
            rolled.append(mapped.shift(-step))
        stacked = pd.concat(rolled, axis=1)
        mode_labels = stacked.mode(axis=1).iloc[:, 0].fillna(4)
        targets = torch.tensor(mode_labels.values, dtype=torch.long)
        return targets

    @property
    def _horizon(self) -> int:
        return max(1, min(10, self.config.forecast_horizon))

    def evaluate(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        loss = self.loss_fn(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        return {"accuracy": accuracy, "loss": loss}
