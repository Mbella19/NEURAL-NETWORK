"""Multi-task dataset for efficient joint training across all phases."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class MultiTaskDataset(Dataset):
    """Dataset that provides features and targets for all tasks simultaneously.

    This dataset creates sliding windows over the feature data and computes targets
    for all tasks at once, enabling efficient multi-task training.
    """

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        raw_frame: pd.DataFrame,
        tasks: List,
        sequence_length: int = 60,
        stride: int = 1,
        feature_stats: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize multi-task dataset.

        Args:
            feature_frame: DataFrame with computed features (all timeframes)
            raw_frame: DataFrame with raw OHLCV data (for target computation)
            tasks: List of task objects (Phase1Task, Phase2Task, etc.)
            sequence_length: Length of input sequences
            stride: Stride between consecutive sequences
            feature_stats: Dict with 'mean' and 'std' tensors for normalization
        """
        self.feature_frame = feature_frame
        self.raw_frame = raw_frame
        self.tasks = tasks
        self.sequence_length = sequence_length
        self.stride = stride

        # Normalize price-level features to percentage distances to avoid regime shift
        feature_frame = self._normalize_price_features(feature_frame)
        self.feature_frame = feature_frame

        # Get feature columns (numeric only)
        self.feature_columns = [
            c for c in feature_frame.columns
            if pd.api.types.is_numeric_dtype(feature_frame[c])
        ]

        # Convert features to tensor once
        self.features_2d = torch.tensor(
            feature_frame[self.feature_columns].values,
            dtype=torch.float32
        )
        self.features_2d = torch.nan_to_num(self.features_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features (CRITICAL for training stability)
        if feature_stats is not None:
            mean = feature_stats["mean"]
            std = feature_stats["std"]
            self.features_2d = (self.features_2d - mean) / std.clamp(min=1e-6)
            self.features_2d = torch.nan_to_num(self.features_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # Build per-task feature masks based on config.feature_subset (applied at __getitem__)
        self.task_feature_masks: Dict[str, torch.Tensor] = {}
        for task in tasks:
            task_name = task.__class__.__name__
            active_subset = getattr(task, "config", None)
            active_features = getattr(active_subset, "feature_subset", None)
            if active_features:
                mask = torch.zeros(len(self.feature_columns), dtype=torch.float32)
                for feat in active_features:
                    if feat in self.feature_columns:
                        mask[self.feature_columns.index(feat)] = 1.0
                    for prefix in ["M5_", "M15_", "H1_"]:
                        prefixed = f"{prefix}{feat}"
                        matches = [i for i, c in enumerate(self.feature_columns) if c == prefixed or c.startswith(prefixed + "_")]
                        for mi in matches:
                            mask[mi] = 1.0
                self.task_feature_masks[task_name] = mask

        # Compute all task targets once
        print(f"Computing targets for {len(tasks)} tasks...")
        self.task_targets = {}
        max_horizon = 0
        for task in tasks:
            task_name = task.__class__.__name__
            print(f"  Computing {task_name}...")
            targets = task.compute_targets(raw_frame, feature_frame=feature_frame)
            horizon = max(0, getattr(task, "_horizon", 0))
            max_horizon = max(max_horizon, horizon)
            self.task_targets[task_name] = targets

        # Trim tail to avoid leakage of future info for the longest horizon
        trim_len = max_horizon
        usable_len = max(0, len(self.features_2d) - trim_len)
        self.features_2d = self.features_2d[:usable_len]
        for task_name, targets in list(self.task_targets.items()):
            if len(targets) > usable_len:
                self.task_targets[task_name] = targets[:usable_len]
            else:
                self.task_targets[task_name] = targets

        self.min_length = usable_len

        # Calculate number of valid sequences
        self.num_sequences = max(0, (self.min_length - sequence_length) // stride + 1)
        print(f"Created {self.num_sequences} sequences of length {sequence_length} (stride {stride})")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        """Get a single sequence and all its targets.

        Returns:
            features: Tensor of shape [sequence_length, num_features]
            targets: Dict mapping task_name -> target tensor
        """
        start = idx * self.stride
        end = start + self.sequence_length

        # Get feature sequence
        features = self.features_2d[start:end]

        # Get targets for all tasks (at the final timestep)
        targets = {}
        for task_name, task_targets in self.task_targets.items():
            tgt = task_targets[end - 1]
            if task_name in self.task_feature_masks:
                mask = self.task_feature_masks[task_name].to(features.device)
                masked = features * mask
            else:
                masked = features
            targets[task_name] = tgt
            # stash masked features per task for caller if needed
        return features, targets

    def get_task_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each task's targets.

        Returns:
            Dict mapping task_name -> {mean, std, pos_rate, etc.}
        """
        stats = {}
        for task_name, targets in self.task_targets.items():
            # Convert to float for computing statistics if needed
            if targets.dtype in (torch.long, torch.int, torch.int64, torch.int32):
                targets_float = targets.float()
            else:
                targets_float = targets

            task_stats = {
                "mean": float(targets_float.mean()),
                "std": float(targets_float.std()),
                "min": float(targets.min()),
                "max": float(targets.max()),
                "dtype": str(targets.dtype),
            }

            # For binary tasks, compute positive rate
            if targets.dtype == torch.float32 and targets.min() >= 0.0 and targets.max() <= 1.0:
                task_stats["pos_rate"] = float((targets > 0.5).float().mean())
            # For integer categorical tasks, compute class distribution
            elif targets.dtype in (torch.long, torch.int64):
                unique_classes = targets.unique()
                task_stats["num_classes"] = len(unique_classes)
                task_stats["classes"] = unique_classes.tolist()

            stats[task_name] = task_stats

        return stats

    @staticmethod
    def _normalize_price_features(features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()
        prefixes = ["", "M5_", "M15_", "H1_"]
        price_markers = ("OPEN", "HIGH", "LOW", "CLOSE", "SMA", "EMA")
        for prefix in prefixes:
            close_col = f"{prefix}CLOSE" if prefix else "CLOSE"
            if close_col not in df.columns:
                continue
            close_series = df[close_col].replace(0, np.nan)
            price_cols = [c for c in df.columns if c.startswith(prefix) and any(marker in c for marker in price_markers)]
            for col in price_cols:
                if col == close_col:
                    df[col] = 0.0
                else:
                    df[col] = (df[col] - close_series) / close_series
        return df.fillna(0.0)


__all__ = ["MultiTaskDataset"]
