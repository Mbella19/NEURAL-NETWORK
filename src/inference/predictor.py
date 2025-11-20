"""Sliding-window inference utilities for the multi-task model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch

from config import get_settings
from .model_loader import ModelBundle


@dataclass
class CandleRecord:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_volume: float
    predictions: Dict[str, float]


class MultiTaskPredictor:
    """Generates streaming predictions over unseen feature frames."""

    def __init__(
        self,
        bundle: ModelBundle,
        feature_stats: Dict[str, torch.Tensor],
        *,
        sequence_length: int | None = None,
        tasks: Sequence[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.bundle = bundle
        self.feature_stats = feature_stats
        self.sequence_length = sequence_length or settings.model.sequence_length
        self.tasks = list(tasks or ("Phase1DirectionTask",))

    def _align_frames(
        self,
        candles_df: pd.DataFrame,
        feature_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        candles = candles_df.copy()
        candles["TIMESTAMP"] = pd.to_datetime(candles["TIMESTAMP"])
        features = feature_frame.copy()
        features["TIMESTAMP"] = pd.to_datetime(features["TIMESTAMP"])
        merged = candles.merge(features, on="TIMESTAMP", how="inner", suffixes=("", "_feat"))
        return merged

    def _prepare_feature_tensor(self, merged: pd.DataFrame) -> torch.Tensor:
        feature_df = merged.reindex(columns=self.bundle.feature_columns, fill_value=0.0)
        feature_df = feature_df.fillna(0.0)
        tensor = torch.tensor(feature_df.values, dtype=torch.float32)
        mean = self.feature_stats["mean"]
        std = self.feature_stats["std"].clamp(min=1e-6)
        tensor = (tensor - mean) / std
        return tensor

    def generate_records(
        self,
        candles_df: pd.DataFrame,
        feature_frame: pd.DataFrame,
    ) -> List[Dict[str, float]]:
        merged = self._align_frames(candles_df, feature_frame)
        if len(merged) < self.sequence_length:
            raise ValueError(
                f"Not enough samples ({len(merged)}) to form sequence length {self.sequence_length}."
            )

        features_tensor = self._prepare_feature_tensor(merged)
        device = self.bundle.device
        features_tensor = features_tensor.to(device)

        ohlc_cols = ["OPEN", "HIGH", "LOW", "CLOSE"]
        candles_subset = merged[["TIMESTAMP", *ohlc_cols, "VOL", "TICKVOL"]].reset_index(drop=True)

        records: List[Dict[str, float]] = []
        for idx in range(self.sequence_length - 1, len(features_tensor)):
            window = features_tensor[idx - self.sequence_length + 1 : idx + 1]
            if window.shape[0] != self.sequence_length:
                continue
            window = window.unsqueeze(0)
            with torch.no_grad():
                outputs = self.bundle.model(window)

            predictions: Dict[str, float] = {}
            for task in self.tasks:
                if task not in outputs:
                    continue
                logits = outputs[task]
                prob = torch.sigmoid(logits).mean().item()
                predictions[task] = prob

            row = candles_subset.iloc[idx]
            records.append(
                {
                    "timestamp": pd.Timestamp(row["TIMESTAMP"]).isoformat(),
                    "open": float(row["OPEN"]),
                    "high": float(row["HIGH"]),
                    "low": float(row["LOW"]),
                    "close": float(row["CLOSE"]),
                    "volume": float(row["VOL"]),
                    "tick_volume": float(row["TICKVOL"]),
                    "predictions": predictions,
                }
            )

        return records
