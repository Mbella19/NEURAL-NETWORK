"""Utilities for preparing inference features from unseen CSV data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline

_HTF_RULES: Dict[str, str] = {"M15": "15T", "H1": "1H"}


def _normalize_column_name(name: str) -> str:
    return name.strip().strip("<>").upper()


def load_testing_csv(path: Path) -> pd.DataFrame:
    """Load the unseen CSV export and normalize columns for downstream use."""

    if not path.exists():
        raise FileNotFoundError(f"Testing CSV not found: {path}")

    frame = pd.read_csv(path, sep="\t")
    frame.columns = [_normalize_column_name(col) for col in frame.columns]

    if not {"DATE", "TIME"}.issubset(frame.columns):
        raise ValueError("Testing CSV must contain <DATE> and <TIME> columns")

    frame["TIMESTAMP"] = pd.to_datetime(frame["DATE"].astype(str) + " " + frame["TIME"].astype(str))
    numeric_cols = [col for col in frame.columns if col not in {"DATE", "TIME", "TIMESTAMP"}]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.drop(columns=[col for col in ["DATE", "TIME"] if col in frame.columns])
    frame = frame.sort_values("TIMESTAMP").dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"]).reset_index(drop=True)
    return frame


def _aggregate_timeframe(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = (
        df.set_index("TIMESTAMP")
        .resample(freq, label="right", closed="right")
        .agg(
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
    )
    agg = agg.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
    agg["SPREAD"] = agg["SPREAD"].fillna(method="ffill").fillna(method="bfill")
    agg = agg.reset_index().rename(columns={"TIMESTAMP": "TIMESTAMP"})
    return agg


def _attach_alias_features(base_feats_raw: pd.DataFrame) -> pd.DataFrame:
    attach_mapping = {
        "STRUCTURE_LABEL_L5": "STRUCTURE_LABEL_L5",
        "FVG_UP": "FVG_UP_SW5",
        "FVG_DOWN": "FVG_DOWN_SW5",
        "PATTERN_STRENGTH": "PATTERN_STRENGTH",
        "SR_DISTANCE": "SR_DISTANCE_W10",
        "LIQUIDITY_SWEEP": "LIQUIDITY_SWEEP_SW5",
        "IMBALANCE_ZONE": "IMBALANCE_ZONE_SW5",
        "ATR": "ATR",
        "BOS_BULLISH": "BOS_BULLISH_SW5",
        "BOS_BEARISH": "BOS_BEARISH_SW5",
    }

    available = {}
    for alias, source in attach_mapping.items():
        if source in base_feats_raw.columns:
            available[alias] = base_feats_raw[source]

    if not available:
        return pd.DataFrame(index=base_feats_raw.index)

    alias_df = pd.DataFrame(available, index=base_feats_raw.index)
    alias_df["TIMESTAMP"] = base_feats_raw["TIMESTAMP"]
    return alias_df


def _build_mtf_features(df_5m: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    pipeline = pipeline or build_default_feature_pipeline()

    base_feats_raw = run_feature_pipeline(df_5m, pipeline=pipeline).dataframe.copy()
    base_feats_raw["TIMESTAMP"] = pd.to_datetime(df_5m["TIMESTAMP"].values)
    base_feats = base_feats_raw.add_prefix("M5_")
    base_feats = base_feats.rename(columns={"M5_TIMESTAMP": "TIMESTAMP"})
    merged = base_feats

    for tf_name, freq in _HTF_RULES.items():
        tf_df = _aggregate_timeframe(df_5m, freq)
        tf_feats = run_feature_pipeline(tf_df, pipeline=pipeline).dataframe.copy()
        tf_feats["TIMESTAMP"] = pd.to_datetime(tf_df["TIMESTAMP"].values)
        tf_feats = tf_feats.add_prefix(f"{tf_name}_")
        tf_feats = tf_feats.rename(columns={f"{tf_name}_TIMESTAMP": "TIMESTAMP"})
        merged = pd.merge_asof(
            merged.sort_values("TIMESTAMP"),
            tf_feats.sort_values("TIMESTAMP"),
            on="TIMESTAMP",
            direction="backward",
            allow_exact_matches=True,
        )

    alias_df = _attach_alias_features(base_feats_raw)
    if not alias_df.empty:
        merged = merged.merge(alias_df, on="TIMESTAMP", how="left")

    merged = merged.sort_values("TIMESTAMP").reset_index(drop=True)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]
    return merged


def prepare_feature_frame(test_csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return base 5m candles and aligned multi-timeframe feature frame."""

    raw_df = load_testing_csv(test_csv_path)
    df_5m = _aggregate_timeframe(raw_df, "5T")
    feature_pipeline = build_default_feature_pipeline()
    feature_frame = _build_mtf_features(df_5m, pipeline=feature_pipeline)
    feature_frame["TIMESTAMP"] = pd.to_datetime(feature_frame["TIMESTAMP"])
    return df_5m.reset_index(drop=True), feature_frame.reset_index(drop=True)
