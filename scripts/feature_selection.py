#!/usr/bin/env python3
"""One-off feature selection using RandomForest to prune noisy features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import get_settings
from data_loader.data_loader import ingest_training_data
from data_loader.timeframe_aggregator import aggregate_timeframes_from_raw
from data_loader.preprocessor import preprocess_dataset
from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline
from training.phases.phase1_direction import Phase1DirectionTask


def _aggregate_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    freq = "5T"
    agg = df.set_index("TIMESTAMP").resample(freq, label="right", closed="right").agg(
        {
            "OPEN": "first",
            "HIGH": "max",
            "LOW": "min",
            "CLOSE": "last",
            "TICKVOL": "sum",
            "VOL": "sum",
            "SPREAD": "mean",
        }
    )
    agg = agg.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
    agg["SPREAD"] = agg["SPREAD"].fillna(method="ffill").fillna(method="bfill")
    return agg.reset_index().rename(columns={"TIMESTAMP": "TIMESTAMP"})


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


def main() -> None:
    settings = get_settings()
    ingest_training_data()
    aggregate_timeframes_from_raw()
    preprocess_dataset()

    processed_dir = settings.data_paths.processed / "preprocessed"
    train_df = pd.read_parquet(processed_dir / "train.parquet") if (processed_dir / "train.parquet").exists() else pd.read_csv(processed_dir / "train.csv")
    train_df["TIMESTAMP"] = pd.to_datetime(train_df["TIMESTAMP"])
    train_df.sort_values("TIMESTAMP", inplace=True)

    train_df = _aggregate_to_5m(train_df)

    base_pipeline = build_default_feature_pipeline()
    feats = run_feature_pipeline(train_df, pipeline=base_pipeline).dataframe
    feats["TIMESTAMP"] = pd.to_datetime(train_df["TIMESTAMP"].values)

    # Align features and target
    task = Phase1DirectionTask()
    targets = task.compute_targets(train_df, feature_frame=feats)
    min_len = min(len(feats), len(targets))
    feats = feats.iloc[:min_len]
    targets = targets[:min_len].squeeze().numpy()

    numeric_feats = feats.select_dtypes(include=["number"])
    numeric_feats = _normalize_price_features(numeric_feats)

    # Sample to keep runtime modest
    sample_size = min(20000, len(numeric_feats))
    sample_idx = np.linspace(0, len(numeric_feats) - 1, sample_size, dtype=int)
    X = numeric_feats.iloc[sample_idx]
    y = targets[sample_idx]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_

    ranked = sorted(zip(numeric_feats.columns, importances), key=lambda x: x[1], reverse=True)
    keep_count = int(len(ranked) * 0.7)  # drop bottom 30%
    selected = [name for name, _ in ranked[:keep_count]]

    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)
    out_path = config_dir / "selected_features.json"
    with out_path.open("w") as f:
        json.dump(selected, f, indent=2)

    print(f"Saved {len(selected)} selected features (of {len(ranked)}) to {out_path}")


if __name__ == "__main__":
    main()
