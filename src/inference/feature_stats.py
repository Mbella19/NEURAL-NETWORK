"""Helpers for loading training feature statistics for inference normalization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch

from config import get_settings


def _read_feature_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature cache format: {path.suffix}")


def _resolve_feature_cache() -> Path:
    settings = get_settings()
    features_dir = settings.data_paths.features
    parquet_path = features_dir / "mtf_features.parquet"
    csv_path = features_dir / "mtf_features.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(
        "Could not locate cached feature statistics. Expected 'mtf_features.parquet' or 'mtf_features.csv'"
    )


def load_feature_stats(feature_columns: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Compute mean/std tensors aligned with training features."""

    cache_path = _resolve_feature_cache()
    cached = _read_feature_cache(cache_path)

    if "SPLIT" in cached.columns:
        train_numeric = cached[cached["SPLIT"] == "train"].drop(columns=["SPLIT"])
    else:
        train_numeric = cached

    train_numeric = train_numeric.select_dtypes(include=["number"]).reindex(columns=list(feature_columns), fill_value=0.0)

    mean_series = train_numeric.mean().fillna(0.0)
    std_series = train_numeric.std().replace(0, 1e-6).fillna(1e-6)

    mean = torch.tensor(mean_series.values, dtype=torch.float32)
    std = torch.tensor(std_series.values, dtype=torch.float32)
    return {"mean": mean, "std": std}
