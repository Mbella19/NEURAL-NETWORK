"""Data preprocessing pipeline (Phase 2.4).

Responsibilities pulled from TASK.md + AGENTS.md:
- Remove weekends/holidays to avoid misleading features
- Handle missing values and temporal gaps
- Detect and clamp outliers without leaking future information
- Normalize/standardize features using training-window statistics only
- Create chronological train/validation/test splits
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import get_settings
from data_loader import MarketDataLoader

HOLIDAY_LIST: Tuple[str, ...] = (
    "2023-12-25",
    "2024-01-01",
    "2024-12-25",
)  # Extend as needed.


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scalers: Dict[str, Dict[str, float]]


class DataPreprocessor:
    """Full cleaning + normalization pipeline for EURUSD data."""

    def __init__(
        self,
        frame: Optional[pd.DataFrame] = None,
        *,
        drop_weekends: bool = True,
        holidays: Tuple[str, ...] = HOLIDAY_LIST,
    ) -> None:
        settings = get_settings()
        self.output_dir = settings.data_paths.processed / "preprocessed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if frame is None:
            loader = MarketDataLoader()
            frame = loader.load_dataframe().frame
        if "TIMESTAMP" not in frame.columns:
            raise ValueError("TIMESTAMP column is required for preprocessing.")
        self.frame = frame.copy().sort_values("TIMESTAMP").reset_index(drop=True)
        self.drop_weekends = drop_weekends
        self.holidays = set(pd.to_datetime(date) for date in holidays)

    def run(
        self,
        *,
        outlier_method: str = "zscore",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> SplitFrames:
        """Run preprocessing pipeline: clean, handle outliers, split data.

        NOTE: Normalization is intentionally NOT performed here. Features should be computed
        on raw prices, then normalized once in the training pipeline (prepare_phase_dataset)
        using train-only statistics. This prevents computing scale-sensitive indicators
        (ATR, volatility, etc.) on already-normalized prices, which would flatten ranges
        and destroy meaningful signals.
        """
        df = self.frame.copy()
        df = self._remove_weekends_and_holidays(df)
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df, method=outlier_method)
        splits = self._temporal_split(df, ratios=split_ratios)
        scalers = {}  # Empty - normalization happens in training pipeline

        return SplitFrames(train=splits.train, val=splits.val, test=splits.test, scalers=scalers)

    # ------------------------------------------------------------------ helpers
    def _remove_weekends_and_holidays(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        if self.drop_weekends:
            dow = df["TIMESTAMP"].dt.dayofweek
            mask &= ~dow.isin([5, 6])
        if self.holidays:
            mask &= ~df["TIMESTAMP"].dt.floor("D").isin(self.holidays)
        removed = len(df) - mask.sum()
        if removed:
            logger.bind(source="preprocessor").info("Removed %s rows due to weekends/holidays", removed)
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        df = df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
        return df.reset_index(drop=True)

    @staticmethod
    def _handle_outliers(df: pd.DataFrame, method: str = "zscore", z_thresh: float = 5.0) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == "zscore":
            for col in numeric_cols:
                series = df[col]
                mean = series.mean()
                std = series.std()
                if std == 0 or np.isnan(std):
                    continue
                z = (series - mean) / std
                mask = z.abs() > z_thresh
                if mask.any():
                    df.loc[mask, col] = mean + np.sign(z[mask]) * z_thresh * std
        elif method == "iqr":
            for col in numeric_cols:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        return df

    @staticmethod
    def _temporal_split(df: pd.DataFrame, ratios: Tuple[float, float, float]) -> SplitFrames:
        train_ratio, val_ratio, test_ratio = ratios
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError("Split ratios must sum to 1.")
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return SplitFrames(
            train=df.iloc[:train_end].copy(),
            val=df.iloc[train_end:val_end].copy(),
            test=df.iloc[val_end:].copy(),
            scalers={},
        )

    @staticmethod
    def _fit_normalization(train_df: pd.DataFrame, method: str = "zscore") -> Dict[str, Dict[str, float]]:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        scalers: Dict[str, Dict[str, float]] = {}
        for col in numeric_cols:
            if method == "zscore":
                scalers[col] = {
                    "mean": train_df[col].mean(),
                    "std": train_df[col].std() or 1.0,
                }
            elif method == "minmax":
                scalers[col] = {
                    "min": train_df[col].min(),
                    "max": train_df[col].max(),
                }
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        return scalers

    @staticmethod
    def _apply_normalization(
        df: pd.DataFrame,
        scalers: Dict[str, Dict[str, float]],
        *,
        method: str = "zscore",
    ) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.copy()
        for col in numeric_cols:
            if col not in scalers:
                continue
            params = scalers[col]
            if method == "zscore":
                df[col] = (df[col] - params["mean"]) / params["std"]
            else:  # minmax
                denom = params["max"] - params["min"] or 1.0
                df[col] = (df[col] - params["min"]) / denom
        return df


def preprocess_dataset(
    *,
    drop_weekends: bool = True,
    holidays: Tuple[str, ...] = HOLIDAY_LIST,
    outlier_method: str = "zscore",
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> SplitFrames:
    """Preprocess dataset: clean, handle outliers, and split into train/val/test.

    Note: This function does NOT normalize the data. Normalization happens later
    in the training pipeline after features are computed on raw prices.
    """
    processor = DataPreprocessor(drop_weekends=drop_weekends, holidays=holidays)
    results = processor.run(
        outlier_method=outlier_method,
        split_ratios=split_ratios,
    )
    _save_split(results.train, processor.output_dir / "train.parquet")
    _save_split(results.val, processor.output_dir / "val.parquet")
    _save_split(results.test, processor.output_dir / "test.parquet")
    _save_scalers(results.scalers, processor.output_dir / "scalers.json")
    logger.bind(source="preprocessor").info("Saved normalized splits to %s", processor.output_dir)
    return results


def _save_split(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)


def _save_scalers(scalers: Dict[str, Dict[str, float]], path: Path) -> None:
    import json

    serializable = {
        col: {k: float(v) for k, v in params.items()} for col, params in scalers.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


__all__ = ["DataPreprocessor", "SplitFrames", "preprocess_dataset"]
