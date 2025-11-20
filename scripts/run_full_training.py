#!/usr/bin/env python3
"""End-to-end training orchestrator."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
import sys
import json

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT))

from config import get_settings
from data_loader.data_loader import ingest_training_data
from data_loader.timeframe_aggregator import aggregate_timeframes_from_raw
from data_loader.preprocessor import preprocess_dataset
from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline
from models.hybrid_model import HybridModel
from models.temporal_fusion import TemporalFusionTransformer
from models.multitask import MultiTaskModel
from training.ppo_runner import run_ppo_episode, PPORunnerConfig, train_ppo, ppo_policy_signals
from training.walk_forward import WalkForwardConfig, walk_forward_validation
from training.cpcv import CPCVConfig, cpcv_validation
from evaluation.explainability import saliency_map, shap_values, lime_explanation
try:
    import torch._dynamo  # type: ignore
except Exception:  # pragma: no cover
    torch_dynamo_available = False
else:
    torch_dynamo_available = True

torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass
# Disable torch.compile for now to avoid MPS/stride/FFT graph issues.
torch_compile = lambda x: x

SEQUENCE_STRIDE = 4  # reduce sequence count for faster training on MPS/CPU
from training.phases.phase1_direction import Phase1DirectionTask
from training.phases.phase2_indicators import Phase2IndicatorTask
from training.phases.phase3_structure import Phase3StructureTask
from training.phases.phase4_smart_money import Phase4SmartMoneyTask
from training.phases.phase5_candlesticks import Phase5CandlestickTask
from training.phases.phase6_sr_levels import Phase6SupportResistanceTask
from training.phases.phase7_advanced_sm import Phase7AdvancedSMTask
from training.phases.phase8_risk import Phase8RiskTask
from training.phases.phase9_integration import Phase9IntegrationTask
from training.phases.policy_execution import PolicyExecutionTask
from training.trainer import TrainerConfig, TrainingLoop
from training.loss_wrappers import make_weighted_bce, FocalLoss
from evaluation.backtester import Backtester
from evaluation.metrics import sharpe_ratio


class SlidingWindowDataset(Dataset):
    """Lazy sliding-window dataset to avoid materializing huge 3D tensors."""

    def __init__(self, features_2d: torch.Tensor, targets: torch.Tensor, sequence_length: int, stride: int = 1):
        if features_2d.ndim != 2:
            raise ValueError("features_2d must be 2D [N, F]")
        if targets.shape[0] != features_2d.shape[0]:
            raise ValueError("Targets and features must share first dimension.")
        self.features = features_2d
        self.targets = targets
        self.sequence_length = sequence_length
        self.stride = max(1, stride)
        self.num_sequences = max(0, (len(self.features) - sequence_length) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.sequence_length
        window = self.features[start:end]
        target = self.targets[end - 1]
        return window, target


def logits_to_signal_with_strength(
    logits: torch.Tensor,
    *,
    upper: float = 0.6,
    lower: float = 0.4,
) -> torch.Tensor:
    """Map logits to [-1,1] with a no-trade band; magnitude encodes confidence."""
    # Reduce multi-step outputs to a scalar confidence
    if logits.ndim >= 2 and logits.shape[-1] > 1:
        logits_scalar = logits.mean(dim=-1, keepdim=True)
    else:
        logits_scalar = logits

    if logits_scalar.shape[-1] == 1:
        probs = torch.sigmoid(logits_scalar.squeeze(-1))
    elif logits_scalar.shape[-1] == 2:
        probs = torch.softmax(logits_scalar, dim=-1)[:, 1]
    else:
        top = torch.argmax(logits_scalar, dim=-1)
        long_idx = logits_scalar.shape[-1] - 1
        return torch.where(
            top == long_idx,
            torch.tensor(1.0, device=logits.device),
            torch.where(top == 0, torch.tensor(-1.0, device=logits.device), torch.tensor(0.0, device=logits.device)),
        )

    longs = probs > upper
    shorts = probs < lower
    mag_long = ((probs - upper) / max(1e-6, 1 - upper)).clamp(0, 1)
    mag_short = ((lower - probs) / max(lower, 1e-6)).clamp(0, 1)
    signal = torch.zeros_like(probs)
    signal = torch.where(longs, mag_long, signal)
    signal = torch.where(shorts, -mag_short, signal)
    return signal


def _normalize_price_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Convert price-level features to percentage distances to remove scale drift."""
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


def _maybe_apply_feature_selection(feature_frames: Dict[str, pd.DataFrame], feature_columns: list[str]) -> tuple[Dict[str, pd.DataFrame], list[str]]:
    """Apply offline-selected features if config/selected_features.json exists."""
    selected_path = Path("config/selected_features.json")
    if not selected_path.exists():
        return feature_frames, feature_columns

    try:
        with selected_path.open("r") as f:
            selected = json.load(f)
        selected_set = [c for c in feature_columns if c in selected]
        if not selected_set:
            print("⚠ selected_features.json present but no overlap with current features; skipping selection")
            return feature_frames, feature_columns
        filtered_frames = {
            split: df.reindex(columns=selected_set, fill_value=0.0)
            for split, df in feature_frames.items()
        }
        print(f"Applied feature selection: {len(selected_set)} features retained (from {len(feature_columns)})")
        return filtered_frames, selected_set
    except Exception as exc:
        print(f"⚠ Failed to apply feature selection: {exc}")
        return feature_frames, feature_columns


def prepare_phase_dataset(
    task,
    frame: pd.DataFrame,
    sequence_length: int = 60,
    *,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    feature_stats: Dict[str, torch.Tensor],
    stride: int = 1,
    use_feature_masking: bool = True,
) -> SlidingWindowDataset:
    """Prepare a normalized, finite dataset with consistent feature dimensions."""
    # feature_frame is expected precomputed; avoid recomputation here
    features_df = feature_frame.select_dtypes(include=["number"]).reindex(columns=feature_columns, fill_value=0.0)
    features_df = _normalize_price_features(features_df)

    features_2d = torch.tensor(features_df.values, dtype=torch.float32)
    features_2d = features_2d.contiguous(memory_format=torch.contiguous_format)
    features_2d = torch.nan_to_num(features_2d, nan=0.0, posinf=0.0, neginf=0.0)

    mean = feature_stats["mean"]
    std = feature_stats["std"]
    if mean.shape[0] != features_2d.shape[1]:
        raise ValueError("Feature mean/std shape mismatch with feature tensor dimension.")

    normalized = (features_2d - mean) / std.clamp(min=1e-6)

    # FIXED: Feature masking breaks shared learning in multi-task training
    # Only apply masking when explicitly enabled (during warmup phase)
    active_subset = getattr(task, "config", None)
    active_features = getattr(active_subset, "feature_subset", None)
    if active_features and use_feature_masking:
        mask = torch.zeros(normalized.shape[1], dtype=normalized.dtype)
        idx = []
        features_found = []
        features_missing = []

        # For each requested feature, find exact match or prefixed/suffixed versions
        for feat in active_features:
            found_any = False

            # 1. Check for exact match (unprefixed features from attach_mapping)
            if feat in feature_columns:
                idx.append(feature_columns.index(feat))
                features_found.append(feat)
                found_any = True

            # 2. Look for multi-timeframe prefixed versions with optional suffixes
            # Suffixes: _SW{n} (SmartMoney), _W{n} (SupportResistance), _L{n} (MarketStructure)
            for prefix in ["M5_", "M15_", "H1_"]:
                # Try exact prefix + feature name
                prefixed_name = f"{prefix}{feat}"
                if prefixed_name in feature_columns:
                    idx.append(feature_columns.index(prefixed_name))
                    features_found.append(prefixed_name)
                    found_any = True
                else:
                    # Try with common suffixes (for features that have lookback parameters)
                    # This handles features like M5_FVG_UP_SW5, M15_SR_DISTANCE_W10, etc.
                    for col in feature_columns:
                        # Check if column starts with prefix+feat and has a suffix
                        if col.startswith(prefixed_name + "_"):
                            # It's a suffixed version - include it
                            if feature_columns.index(col) not in idx:  # Avoid duplicates
                                idx.append(feature_columns.index(col))
                                features_found.append(col)
                                found_any = True

            if not found_any:
                features_missing.append(feat)

        if idx:
            mask[idx] = 1.0
            normalized = normalized * mask
            print(f"  Features active for {task.__class__.__name__}: {len(idx)} columns")
            if features_missing:
                print(f"  WARNING: Could not find features: {features_missing}")
        else:
            print(f"  ERROR: No features found for {task.__class__.__name__}!")
            print(f"  Requested: {active_features}")
            print(f"  Available sample: {feature_columns[:10]}...")
            raise ValueError(f"No matching features found for {task.__class__.__name__}")

    normalized = torch.clamp(normalized, -8.0, 8.0)
    if not torch.isfinite(normalized).all():
        raise ValueError(f"Non-finite features detected for {task.__class__.__name__}")

    targets = task.compute_targets(frame, feature_frame=feature_frame)
    if not torch.isfinite(targets).all():
        raise ValueError(f"Non-finite targets detected for {task.__class__.__name__}")

    # Trim tail to avoid using padded targets where future data is unavailable.
    horizon = getattr(task, "_horizon", 0)
    if horizon > 0:
        normalized = normalized[:-horizon]
        targets = targets[:-horizon]

    n_samples = len(normalized)
    if n_samples < sequence_length:
        raise ValueError(f"Not enough data: {n_samples} samples < {sequence_length} sequence length")

    dataset = SlidingWindowDataset(normalized, targets, sequence_length, stride=stride)
    print(f"  Created {len(dataset)} sequences of length {sequence_length} (stride {stride})")
    print(f"  Feature shape: torch.Size([{dataset.num_sequences}, {sequence_length}, {normalized.shape[1]}]), "
          f"Target shape: torch.Size([{dataset.num_sequences}, {targets.shape[1] if targets.ndim > 1 else 1}])")

    return dataset


def build_trainers(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    feature_frames: Dict[str, pd.DataFrame],
    feature_columns: list[str],
    feature_stats: Dict[str, torch.Tensor],
    shared_model: HybridModel,
    sequence_length: int,
    use_lstm: bool,
) -> list[tuple[str, TrainingLoop]]:
    phases = [
        Phase1DirectionTask(),
        Phase2IndicatorTask(),
        Phase3StructureTask(),
        Phase4SmartMoneyTask(),
        Phase5CandlestickTask(),
        Phase6SupportResistanceTask(),
        Phase7AdvancedSMTask(),
        Phase8RiskTask(),         # Risk management now in Phase 8
        Phase9IntegrationTask(),  # Integration now in Phase 9
        PolicyExecutionTask(),    # Execution head
    ]
    # Determine device (MPS for M2 GPU, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (no GPU acceleration)")
    
    gpu_friendly = device.type in {"mps", "cuda"}
    batch_size = 128 if gpu_friendly else 32
    grad_accum = 1 if gpu_friendly else 2
    mixed_precision = gpu_friendly
    
    trainers = []
    for phase in phases:
        train_ds = prepare_phase_dataset(
            phase,
            train_frame,
            sequence_length=sequence_length,
            feature_frame=feature_frames["train"],
            feature_columns=feature_columns,
            feature_stats=feature_stats,
            stride=SEQUENCE_STRIDE,
        )
        val_ds = prepare_phase_dataset(
            phase,
            val_frame,
            sequence_length=sequence_length,
            feature_frame=feature_frames["val"],
            feature_columns=feature_columns,
            feature_stats=feature_stats,
            stride=SEQUENCE_STRIDE,
        )
        
        if hasattr(phase, "output_dim"):
            output_dim = phase.output_dim
        elif isinstance(phase, Phase3StructureTask):
            output_dim = 5
        else:
            output_dim = 1

        model = shared_model
        model.reset_head(output_dim)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        loss_fn = phase.loss_fn
        # FIXED: Use FocalLoss ONLY for imbalanced tasks (expert recommendation)
        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            try:
                # Compute class balance on the full training dataset
                all_train_targets = []
                for i in range(len(train_ds)):
                    _, target = train_ds[i]
                    all_train_targets.append(target)
                all_targets = torch.stack(all_train_targets, dim=0)
                positives = (all_targets > 0.5).sum().item()
                negatives = all_targets.numel() - positives
                pos_rate = positives / (positives + negatives) if (positives + negatives) > 0 else 0.5

                # Imbalance threshold
                IMBALANCE_THRESHOLD = 0.35
                is_imbalanced = pos_rate < IMBALANCE_THRESHOLD or pos_rate > (1 - IMBALANCE_THRESHOLD)

                if is_imbalanced:
                    # Use FocalLoss for imbalanced data
                    alpha = min(0.5, pos_rate) if pos_rate < 0.5 else min(0.5, 1 - pos_rate)
                    gamma = 3.0 if pos_rate < 0.1 or pos_rate > 0.9 else 2.0
                    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
                    print(f"  FocalLoss(alpha={alpha:.3f}, gamma={gamma:.1f}) - pos_rate={pos_rate:.2%} (IMBALANCED)")
                else:
                    # Use BCE for balanced data
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    print(f"  BCEWithLogitsLoss - pos_rate={pos_rate:.2%} (BALANCED)")
            except Exception as e:
                print(f"  Warning: Failed to configure loss: {e}, using default BCE")
                loss_fn = torch.nn.BCEWithLogitsLoss()

        trainer = TrainingLoop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=TrainerConfig(
                epochs=50,
                batch_size=batch_size,
                gradient_accumulation=grad_accum,
                mixed_precision=mixed_precision,
                resource_log_interval=1000,
                scheduler="cosine",
                lr=1e-4,
                show_progress=False,
            ),
            train_dataset=train_ds,
            val_dataset=val_ds,
        )
        trainers.append((phase.__class__.__name__, trainer))
    return trainers


def _load_split(path: Path) -> pd.DataFrame:
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    raise FileNotFoundError(f"Expected split not found at {path}")


def _compute_feature_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pipeline,
) -> tuple[Dict[str, pd.DataFrame], list[str], Dict[str, torch.Tensor]]:
    """Run the pipeline per split (with aligned multi-timeframe features) and compute train-only stats."""

    tf_rules = {"M15": "15T", "H1": "1H"}

    def _aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Aggregate base to HTF without leakage; timestamp is window close."""
        df = df.sort_values("TIMESTAMP").set_index("TIMESTAMP")
        agg = df.resample(freq, label="right", closed="right").agg(
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
        agg = agg.reset_index().rename(columns={"TIMESTAMP": "TIMESTAMP"})
        return agg

    def _build_mtf_features(df: pd.DataFrame) -> pd.DataFrame:
        """Attach HTF features (15m/1h) aligned backward to the 5m timestamps."""
        base_feats_raw = run_feature_pipeline(df, pipeline=pipeline).dataframe.copy()
        base_feats_raw["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"].values)
        base_feats = base_feats_raw.add_prefix("M5_")
        base_feats = base_feats.rename(columns={"M5_TIMESTAMP": "TIMESTAMP"})

        merged = base_feats
        for tf_name, freq in tf_rules.items():
            agg_df = _aggregate(df, freq)
            tf_feats = run_feature_pipeline(agg_df, pipeline=pipeline).dataframe.copy()
            tf_feats["TIMESTAMP"] = pd.to_datetime(agg_df["TIMESTAMP"].values)
            tf_feats = tf_feats.add_prefix(f"{tf_name}_")
            tf_feats = tf_feats.rename(columns={f"{tf_name}_TIMESTAMP": "TIMESTAMP"})
            merged = pd.merge_asof(
                merged.sort_values("TIMESTAMP"),
                tf_feats.sort_values("TIMESTAMP"),
                on="TIMESTAMP",
                direction="backward",  # use last completed HTF bar to avoid leakage
                allow_exact_matches=True,
            )
        # Reattach un-prefixed versions of key features for downstream tasks
        # Note: SmartMoney and SR features now have suffixes (_SW5, _W10, etc.)
        # We attach the primary lookback versions without suffix for backward compatibility.
        # This happens BEFORE prefixing, so base_feats_raw still has the original column names.
        attach_mapping = {
            "STRUCTURE_LABEL_L5": "STRUCTURE_LABEL_L5",
            "FVG_UP": "FVG_UP_SW5",           # From SmartMoneyFeatures(swing_lookback=5)
            "FVG_DOWN": "FVG_DOWN_SW5",
            "PATTERN_STRENGTH": "PATTERN_STRENGTH",
            "SR_DISTANCE": "SR_DISTANCE_W10",  # From SupportResistanceFeatures(window=10)
            "LIQUIDITY_SWEEP": "LIQUIDITY_SWEEP_SW5",
            "IMBALANCE_ZONE": "IMBALANCE_ZONE_SW5",
            "ATR": "ATR",
            "BOS_BULLISH": "BOS_BULLISH_SW5",
            "BOS_BEARISH": "BOS_BEARISH_SW5",
        }
        available = {}
        missing_features = []
        for unprefixed, suffixed in attach_mapping.items():
            if suffixed in base_feats_raw.columns:
                available[unprefixed] = base_feats_raw[suffixed]
            else:
                missing_features.append(f"{unprefixed} (expected: {suffixed})")

        if missing_features:
            print(f"  WARNING: Some expected features not found in pipeline output: {missing_features}")
            print(f"  This may cause issues for phases that reference these features.")
            print(f"  Available columns in base_feats_raw: {list(base_feats_raw.columns[:20])}...")

        if available:
            attach_df = pd.DataFrame(available, index=base_feats_raw.index)
            attach_df["TIMESTAMP"] = base_feats_raw["TIMESTAMP"]
            merged = merged.merge(attach_df, on="TIMESTAMP", how="left")
            print(f"  Attached {len(available)} unprefixed feature aliases for backward compatibility")
        merged = merged.sort_values("TIMESTAMP").reset_index(drop=True)
        return merged

    train_feats_full = _build_mtf_features(train_df)
    val_feats_full = _build_mtf_features(val_df)
    test_feats_full = _build_mtf_features(test_df)

    # Restrict to numeric columns and align columns across splits.
    numeric_cols = [c for c in train_feats_full.columns if pd.api.types.is_numeric_dtype(train_feats_full[c])]
    feature_columns = numeric_cols

    def _align_numeric(df: pd.DataFrame) -> pd.DataFrame:
        aligned = df.reindex(columns=feature_columns, fill_value=0.0)
        return aligned

    train_numeric = _align_numeric(train_feats_full)
    val_numeric = _align_numeric(val_feats_full)
    test_numeric = _align_numeric(test_feats_full)

    std_series = train_numeric.std().replace(0, 1e-6).fillna(1e-6)
    feature_stats: Dict[str, torch.Tensor] = {
        "mean": torch.tensor(train_numeric.mean().values, dtype=torch.float32),
        "std": torch.tensor(std_series.values, dtype=torch.float32),
    }
    return (
        {"train": train_feats_full, "val": val_feats_full, "test": test_feats_full},
        feature_columns,
        feature_stats,
    )


def _build_timeframe_slices(feature_columns: list[str]) -> Dict[str, list[int]]:
    """Build index slices for MTF encoders based on column prefixes."""
    mapping = {"H1_": "H1", "M15_": "M15", "M5_": "M5"}
    slices: Dict[str, list[int]] = {}
    for idx, col in enumerate(feature_columns):
        matched = False
        for prefix, name in mapping.items():
            if col.startswith(prefix):
                slices.setdefault(name, []).append(idx)
                matched = True
                break
    return {k: v for k, v in slices.items() if v}


def main() -> None:
    RUN_SEQUENTIAL = False
    RUN_CURRICULUM = False  # Disabled - using new multi-task training instead
    settings = get_settings()
    ingest_training_data()
    aggregate_timeframes_from_raw()
    preprocess_dataset()

    processed_dir = settings.data_paths.processed / "preprocessed"
    cache_dir = settings.data_paths.features
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path_parquet = cache_dir / "mtf_features.parquet"
    cache_path_csv = cache_dir / "mtf_features.csv"

    train_df_raw = _load_split(processed_dir / "train")
    val_df_raw = _load_split(processed_dir / "val")
    test_df_raw = _load_split(processed_dir / "test")

    # Ensure timestamps are datetime and ordered
    for df in (train_df_raw, val_df_raw, test_df_raw):
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df.sort_values("TIMESTAMP", inplace=True)
        df.reset_index(drop=True, inplace=True)

    def _aggregate_to_5m(df: pd.DataFrame) -> pd.DataFrame:
        """Convert base 1m data to 5m bars for training/inference."""
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
        agg = agg.reset_index().rename(columns={"TIMESTAMP": "TIMESTAMP"})
        return agg

    train_df = _aggregate_to_5m(train_df_raw)
    val_df = _aggregate_to_5m(val_df_raw)
    test_df = _aggregate_to_5m(test_df_raw)

    if cache_path_parquet.exists():
        cached_feats = pd.read_parquet(cache_path_parquet)
        feature_frames = {
            "train": cached_feats[cached_feats["SPLIT"] == "train"].drop(columns=["SPLIT"]),
            "val": cached_feats[cached_feats["SPLIT"] == "val"].drop(columns=["SPLIT"]),
            "test": cached_feats[cached_feats["SPLIT"] == "test"].drop(columns=["SPLIT"]),
        }
    elif cache_path_csv.exists():
        cached_feats = pd.read_csv(cache_path_csv)
        feature_frames = {
            "train": cached_feats[cached_feats["SPLIT"] == "train"].drop(columns=["SPLIT"]),
            "val": cached_feats[cached_feats["SPLIT"] == "val"].drop(columns=["SPLIT"]),
            "test": cached_feats[cached_feats["SPLIT"] == "test"].drop(columns=["SPLIT"]),
        }
    else:
        base_pipeline = build_default_feature_pipeline()
        feature_frames, feature_columns, feature_stats = _compute_feature_frames(
            train_df, val_df, test_df, base_pipeline
        )
        feature_frames, feature_columns = _maybe_apply_feature_selection(feature_frames, feature_columns)
        train_numeric = feature_frames["train"].reindex(columns=feature_columns, fill_value=0.0)
        norm_train = _normalize_price_features(train_numeric)
        std_series = norm_train.std().replace(0, 1e-6).fillna(1e-6)
        feature_stats = {
            "mean": torch.tensor(norm_train.mean().values, dtype=torch.float32),
            "std": torch.tensor(std_series.values, dtype=torch.float32),
        }
        concat_cache = pd.concat(
            [
                feature_frames["train"].assign(SPLIT="train"),
                feature_frames["val"].assign(SPLIT="val"),
                feature_frames["test"].assign(SPLIT="test"),
            ],
            axis=0,
            ignore_index=True,
        )
        try:
            concat_cache.to_parquet(cache_path_parquet)
        except Exception:
            concat_cache.to_csv(cache_path_csv, index=False)
    if cache_path_parquet.exists() or cache_path_csv.exists():
        # compute stats from cached features
        train_numeric = feature_frames["train"].select_dtypes(include=["number"])
        feature_columns = train_numeric.columns.tolist()
        feature_frames, feature_columns = _maybe_apply_feature_selection(feature_frames, feature_columns)
        train_numeric = feature_frames["train"].reindex(columns=feature_columns, fill_value=0.0)
        norm_train = _normalize_price_features(train_numeric)
        std_series = norm_train.std().replace(0, 1e-6).fillna(1e-6)
        feature_stats = {
            "mean": torch.tensor(norm_train.mean().values, dtype=torch.float32),
            "std": torch.tensor(std_series.values, dtype=torch.float32),
        }
    tf_slices = _build_timeframe_slices(feature_frames["train"].columns.tolist())
    use_mps = torch.backends.mps.is_available()
    hybrid_lstm_enabled = not use_mps  # disable LSTM branch on MPS to avoid CPU fallback

    all_tasks = [
        Phase1DirectionTask(),
        Phase2IndicatorTask(),
        Phase3StructureTask(),
        Phase4SmartMoneyTask(),
        Phase5CandlestickTask(),
        Phase6SupportResistanceTask(),
        Phase7AdvancedSMTask(),
        Phase8RiskTask(),         # Risk management now before integration
        Phase9IntegrationTask(),  # Integration after risk
        PolicyExecutionTask(),
    ]

    if RUN_SEQUENTIAL or RUN_CURRICULUM:
        print("Running curriculum Phase1->Phase9 sequentially with shared backbone...")
        shared_model = HybridModel(
            input_dim=len(feature_columns),
            output_dim=1,
            dropout=0.5,
            use_fsatten=True,
            use_lstm=hybrid_lstm_enabled,
            timeframe_slices=tf_slices,
        )
        phase_order = [
            Phase1DirectionTask(),
            Phase2IndicatorTask(),
            Phase3StructureTask(),
            Phase4SmartMoneyTask(),
            Phase5CandlestickTask(),
            Phase6SupportResistanceTask(),
            Phase7AdvancedSMTask(),
            Phase8RiskTask(),
            Phase9IntegrationTask(),
        ]
        backtester = Backtester()
        for phase in phase_order:
            phase_name = phase.__class__.__name__
            print(f"Curriculum phase: {phase_name}")
            train_ds = prepare_phase_dataset(
                phase,
                train_df,
                sequence_length=settings.model.sequence_length,
                feature_frame=feature_frames["train"],
                feature_columns=feature_columns,
                feature_stats=feature_stats,
                stride=SEQUENCE_STRIDE,
            )
            val_ds = prepare_phase_dataset(
                phase,
                val_df,
                sequence_length=settings.model.sequence_length,
                feature_frame=feature_frames["val"],
                feature_columns=feature_columns,
                feature_stats=feature_stats,
                stride=SEQUENCE_STRIDE,
            )
            if hasattr(phase, "output_dim"):
                output_dim = phase.output_dim
            elif isinstance(phase, Phase3StructureTask):
                output_dim = 5
            else:
                output_dim = 1

            shared_model.reset_head(output_dim)
            shared_model.to(torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")))

            optimizer = torch.optim.Adam(shared_model.parameters(), lr=1e-4, weight_decay=1e-4)
            loss_fn = phase.loss_fn
            # FIXED: Use FocalLoss ONLY for imbalanced tasks (expert recommendation)
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                try:
                    # Compute class balance on the full training dataset
                    all_train_targets = []
                    for i in range(len(train_ds)):
                        _, target = train_ds[i]
                        all_train_targets.append(target)
                    all_targets = torch.stack(all_train_targets, dim=0)
                    positives = (all_targets > 0.5).sum().item()
                    negatives = all_targets.numel() - positives
                    pos_rate = positives / (positives + negatives) if (positives + negatives) > 0 else 0.5

                    # Imbalance threshold
                    IMBALANCE_THRESHOLD = 0.35
                    is_imbalanced = pos_rate < IMBALANCE_THRESHOLD or pos_rate > (1 - IMBALANCE_THRESHOLD)

                    if is_imbalanced:
                        # Use FocalLoss for imbalanced data
                        alpha = min(0.5, pos_rate) if pos_rate < 0.5 else min(0.5, 1 - pos_rate)
                        gamma = 3.0 if pos_rate < 0.1 or pos_rate > 0.9 else 2.0
                        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
                        print(f"  FocalLoss(alpha={alpha:.3f}, gamma={gamma:.1f}) - pos_rate={pos_rate:.2%} (IMBALANCED)")
                    else:
                        # Use BCE for balanced data
                        loss_fn = torch.nn.BCEWithLogitsLoss()
                        print(f"  BCEWithLogitsLoss - pos_rate={pos_rate:.2%} (BALANCED)")
                except Exception as e:
                    print(f"  Warning: Failed to configure loss: {e}, using default BCE")
                    loss_fn = torch.nn.BCEWithLogitsLoss()

            trainer = TrainingLoop(
                model=shared_model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                config=TrainerConfig(
                    epochs=50,
                    batch_size=128 if torch.backends.mps.is_available() or torch.cuda.is_available() else 32,
                    gradient_accumulation=1 if torch.backends.mps.is_available() or torch.cuda.is_available() else 2,
                    mixed_precision=torch.backends.mps.is_available() or torch.cuda.is_available(),
                    resource_log_interval=1000,
                    scheduler="cosine",
                    lr=1e-4,
                    show_progress=False,
                ),
                train_dataset=train_ds,
                val_dataset=val_ds,
            )
            metrics = trainer.train()
            print(f"  Metrics: {metrics}")
            # Quick backtest on val split using phase model outputs as signals if binary
            try:
                if "val_loss" in metrics:
                    val_signals = []
                    ds = val_ds
                    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
                    trainer.model.eval()
                    with torch.no_grad():
                        for feats, _ in loader:
                            feats = feats.to(trainer._device())
                            out = trainer.model(feats)
                            sig = logits_to_signal_with_strength(out)
                            val_signals.extend(sig.cpu().numpy().tolist())
                    price_series = val_df["CLOSE"].iloc[settings.model.sequence_length : settings.model.sequence_length + len(val_signals)]
                    signals_series = pd.Series(val_signals[: len(price_series)], index=price_series.index)
                    bt_metrics = backtester.run(price_data=val_df.iloc[settings.model.sequence_length : settings.model.sequence_length + len(signals_series)], signals=signals_series)
                    print(f"  Backtest (val) -> return: {bt_metrics['total_return']:.4f}, sharpe: {bt_metrics['sharpe']:.4f}, maxdd: {bt_metrics['max_drawdown']:.4f}")
            except Exception as exc:
                print(f"  Backtest skipped for {phase_name}: {exc}")

    # Walk-forward and CPCV validation (DISABLED for faster multi-task training)
    # Uncomment below to enable validation
    """
    if True:  # Validation disabled
        # Lightweight walk-forward validation using the first phase task to ensure out-of-sample checks.
        wf_config = WalkForwardConfig(train_window=len(train_df), test_window=len(test_df), step=len(test_df))

        def build_phase1_trainer(train_slice: pd.DataFrame, test_slice: pd.DataFrame) -> TrainingLoop:
            # Reuse stats from the *original* train set to avoid leakage across windows.
            train_feats = run_feature_pipeline(train_slice, pipeline=base_pipeline).dataframe
            train_feats = train_feats.reindex(columns=feature_columns, fill_value=0.0)
            test_feats = run_feature_pipeline(test_slice, pipeline=base_pipeline).dataframe
            test_feats = test_feats.reindex(columns=feature_columns, fill_value=0.0)

            phase = Phase1DirectionTask()
            sequence_len = get_settings().model.sequence_length
            train_ds = prepare_phase_dataset(
                phase,
                train_slice,
                sequence_length=sequence_len,
                feature_frame=train_feats,
                feature_columns=feature_columns,
                feature_stats=feature_stats,
                stride=SEQUENCE_STRIDE,
            )
            test_ds = prepare_phase_dataset(
                phase,
                test_slice,
                sequence_length=sequence_len,
                feature_frame=test_feats,
                feature_columns=feature_columns,
                feature_stats=feature_stats,
                stride=SEQUENCE_STRIDE,
            )
            model = HybridModel(
                input_dim=len(feature_columns),
                output_dim=phase.output_dim,
                dropout=0.5,
                use_fsatten=True,
                use_lstm=hybrid_lstm_enabled,
                timeframe_slices=tf_slices,
            )
            model = torch_compile(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            trainer = TrainingLoop(
                model=model,
                optimizer=optimizer,
                loss_fn=phase.loss_fn,
                config=TrainerConfig(
                    epochs=10,
                    batch_size=64,
                    mixed_precision=False,
                    scheduler="cosine",
                    lr=1e-4,
                    device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
                ),
                train_dataset=train_ds,
                val_dataset=test_ds,
            )
            return trainer

        print("Running walk-forward validation (Phase1 sanity check)...")
        wf_results = walk_forward_validation(
            dataset=pd.concat([train_df, test_df], ignore_index=True),
            build_trainer=build_phase1_trainer,
            config=wf_config,
        )
        print("Walk-forward results:", wf_results)

        # CPCV (enabled for deeper OOS checks)
        cpcv_results = cpcv_validation(
            dataset=pd.concat([train_df, val_df, test_df], ignore_index=True),
            build_trainer=build_phase1_trainer,
            config=CPCVConfig(n_splits=3, test_window=len(test_df), purge_window=500),
        )
        print("CPCV results:", cpcv_results)

    # PPO training (multi-episode) using the same feature stats; eval on test set.
    # Note: PPO runs independently, not part of validation block
    RUN_PPO = False  # Set to True to run PPO training
    if RUN_PPO and RUN_CURRICULUM:  # PPO needs shared_model from curriculum
        try:
            mean_vec = feature_stats["mean"]
            std_vec = feature_stats["std"]

            def ppo_feature_fn(obs: torch.Tensor) -> torch.Tensor:
            # obs shape: (1, lookback, features) -> flatten then normalize
            flat = obs.reshape(obs.shape[0], -1)
            # Pad/truncate to match stats length if needed
            if flat.shape[1] != len(mean_vec):
                if flat.shape[1] > len(mean_vec):
                    flat = flat[:, : len(mean_vec)]
                else:
                    pad = torch.zeros(flat.shape[0], len(mean_vec) - flat.shape[1], device=flat.device, dtype=flat.dtype)
                    flat = torch.cat([flat, pad], dim=1)
            norm = (flat - mean_vec.to(flat.device)) / std_vec.to(flat.device).clamp(min=1e-6)
            return norm

        ppo_cfg = PPORunnerConfig(
            device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
            rollout_steps=512,
            episodes=5,
        )
        ppo_out = train_ppo(train_df, ppo_feature_fn, ppo_cfg)
        print("PPO train metrics:", ppo_out["metrics"])
        # Generate PPO policy signals on validation data
        ppo_probs, ppo_actions = ppo_policy_signals(val_df, ppo_out["agent"], ppo_feature_fn, ppo_cfg)
        # Map actions: 0->short(-1),1->flat(0),2->long(1)
        ppo_signals = pd.Series((ppo_actions.numpy() - 1).astype(float))
        # Supervised direction head (Phase1) signals
        phase1 = Phase1DirectionTask()
        phase1_ds = prepare_phase_dataset(
            phase1,
            val_df,
            sequence_length=settings.model.sequence_length,
            feature_frame=feature_frames["val"],
            feature_columns=feature_columns,
            feature_stats=feature_stats,
            stride=SEQUENCE_STRIDE,
        )
        sup_sigs = []
        sup_loader = torch.utils.data.DataLoader(phase1_ds, batch_size=256, shuffle=False)
        shared_model.eval()
        with torch.no_grad():
            for feats, _ in sup_loader:
                out = shared_model(feats.to(shared_model.device if hasattr(shared_model, "device") else "cpu"))  # type: ignore[arg-type]
                sup_sigs.extend(logits_to_signal_with_strength(out).cpu().numpy().tolist())
        sup_signals = pd.Series(sup_sigs[: len(ppo_signals)], index=ppo_signals.index)
        # Agreement filter: trade only when PPO action equals supervised direction
        fusion_signals = ppo_signals.copy()
        fusion_signals.loc[sup_signals != ppo_signals] = 0.0
        price_series = val_df["CLOSE"].iloc[: len(fusion_signals)]
        fusion_signals = fusion_signals.iloc[: len(price_series)]
        bt_metrics = Backtester().run(
            price_data=val_df.iloc[: len(fusion_signals)],
            signals=fusion_signals,
        )
        print(f"PPO+Supervised agreement (val) -> return: {bt_metrics['total_return']:.4f}, sharpe: {bt_metrics['sharpe']:.4f}, maxdd: {bt_metrics['max_drawdown']:.4f}")
    except Exception as exc:  # pragma: no cover
        print("PPO training failed:", exc)
    """  # End of disabled validation and PPO section

    # ============================================================================
    # 3-Phase Hybrid Multi-Task Training
    # ============================================================================
    # Phase 1: Optional short curriculum warmup (5-10 epochs per task)
    # Phase 2: Multi-task primary training (50+ epochs)
    # Phase 3: Optional task-specific fine-tuning with EWC
    # ============================================================================

    print("\n" + "="*80)
    print("PHASE 1: SHORT CURRICULUM WARMUP (OPTIONAL)")
    print("="*80)

    # Define all tasks
    all_tasks = [
        Phase1DirectionTask(),
        Phase2IndicatorTask(),
        Phase3StructureTask(),
        Phase4SmartMoneyTask(),
        Phase5CandlestickTask(),
        Phase6SupportResistanceTask(),
        Phase7AdvancedSMTask(),
        Phase8RiskTask(),
        Phase9IntegrationTask(),
    ]

    # Create multi-task datasets using the new MultiTaskDataset infrastructure
    from training.multitask_dataset import MultiTaskDataset

    print("Creating multi-task datasets...")
    mt_train_ds = MultiTaskDataset(
        feature_frame=feature_frames["train"],
        raw_frame=train_df,
        tasks=all_tasks,
        sequence_length=settings.model.sequence_length,
        stride=SEQUENCE_STRIDE,
        feature_stats=feature_stats,  # CRITICAL: Normalize features for training stability
    )
    mt_val_ds = MultiTaskDataset(
        feature_frame=feature_frames["val"],
        raw_frame=val_df,
        tasks=all_tasks,
        sequence_length=settings.model.sequence_length,
        stride=SEQUENCE_STRIDE,
        feature_stats=feature_stats,  # Use training stats for validation
    )

    # Print dataset statistics
    train_stats = mt_train_ds.get_task_statistics()
    print("\nTask statistics (train):")
    for task_name, stats in train_stats.items():
        print(f"  {task_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        if "pos_rate" in stats:
            print(f"    pos_rate={stats['pos_rate']:.2%}")

    # Initialize shared backbone model in FEATURE MODE for multi-task learning
    # output_features=True makes TFT output features (d_model dim) instead of predictions
    # This allows MultiTaskModel aux_heads to transform features -> task predictions
    tft_backbone = TemporalFusionTransformer(
        input_dim=len(feature_columns),
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.4,  # FIX: Increased from 0.3 to 0.4
        use_fsatten=True,
        output_features=True,  # CRITICAL: Output features for multi-task heads
    )

    # Create multi-task model with auxiliary heads
    aux_heads = {}
    for task in all_tasks:
        # Use explicit output_dim if available (preferred for classification tasks with 1D targets)
        if hasattr(task, "output_dim"):
            head_dim = task.output_dim
        else:
            # Fallback to target shape for regression tasks
            head_dim = mt_train_ds.task_targets[task.__class__.__name__].shape[-1]
        aux_heads[task.__class__.__name__] = head_dim
    mt_model = MultiTaskModel(tft_backbone, aux_heads=aux_heads, dropout=0.6)  # FIX: Increased dropout from 0.5 to 0.6 in head
    mt_optimizer = torch.optim.Adam(mt_model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Define device for training
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Define task loss functions with proper class balancing
    mt_loss_fns = {}
    for task in all_tasks:
        task_name = task.__class__.__name__
        lf = task.loss_fn

        # FIXED: Use FocalLoss ONLY for imbalanced tasks (expert recommendation)
        # For balanced tasks (~50%), standard BCE works better and is more stable
        if isinstance(lf, torch.nn.BCEWithLogitsLoss):
            sample_targets = mt_train_ds.task_targets[task_name]
            positives = (sample_targets > 0.5).sum().item()
            negatives = sample_targets.numel() - positives
            pos_rate = positives / sample_targets.numel()

            # Imbalance threshold: use FocalLoss only if significantly imbalanced
            IMBALANCE_THRESHOLD = 0.40  # Imbalanced if pos_rate < 0.40 or > 0.60
            is_imbalanced = pos_rate < IMBALANCE_THRESHOLD or pos_rate > (1 - IMBALANCE_THRESHOLD)

            if is_imbalanced:
                # IMBALANCED task → Use FocalLoss
                # Dynamic alpha based on class imbalance
                if pos_rate < 0.5:
                    alpha = max(0.25, pos_rate)  # Weight rare positive class, min 0.25
                else:
                    alpha = max(0.25, 1 - pos_rate)  # Weight rare negative class, min 0.25

                # Higher gamma for extreme imbalance
                gamma = 3.0 if pos_rate < 0.1 or pos_rate > 0.9 else 2.0

                lf = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
                print(f"  {task_name}: FocalLoss(alpha={alpha:.3f}, gamma={gamma:.1f}) - pos_rate={pos_rate:.2%} (IMBALANCED)")
            else:
                # BALANCED task → Use standard BCE (more stable, faster convergence)
                lf = torch.nn.BCEWithLogitsLoss()
                print(f"  {task_name}: BCEWithLogitsLoss - pos_rate={pos_rate:.2%} (BALANCED, no FocalLoss needed)")

        elif isinstance(lf, torch.nn.CrossEntropyLoss):
            # Handle multiclass imbalance for Phase 4 (Smart Money)
            sample_targets = mt_train_ds.task_targets[task_name]
            if sample_targets.ndim > 1:
                sample_targets = sample_targets.squeeze()
            
            # Compute class weights: total / (n_classes * count)
            # Ensure we handle all classes defined in output_dim
            num_classes = getattr(task, "output_dim", 3)
            counts = torch.zeros(num_classes)
            unique, u_counts = torch.unique(sample_targets, return_counts=True)
            
            for cls, count in zip(unique, u_counts):
                if cls < num_classes:
                    counts[cls.long()] = count
            
            # Avoid division by zero
            counts = counts.clamp(min=1.0)
            total = counts.sum()
            weights = total / (num_classes * counts)
            
            # Normalize weights so they sum to num_classes (optional, but keeps scale similar)
            weights = weights / weights.mean()
            
            # Clamp weights to prevent exploding gradients from extremely rare classes
            weights = weights.clamp(max=10.0)
            
            print(f"  {task_name}: CrossEntropyLoss with weights {weights.tolist()}")
            lf = torch.nn.CrossEntropyLoss(weight=weights.to(device))

        mt_loss_fns[task_name] = lf

    # Define base task weights (importance)
    # CHANGE THIS:
    WEIGHTING_STRATEGY = "uncertainty"  # Was "static"

    # You can keep base weights as 1.0, the model will adjust them automatically
    # But providing a slight hint helps convergence speed.
    task_weights = {
        "Phase1DirectionTask": 1.0,
        "Phase2IndicatorTask": 1.0,
        "Phase3StructureTask": 1.0,
        "Phase4SmartMoneyTask": 1.0,
        "Phase5CandlestickTask": 1.0,
        "Phase6SupportResistanceTask": 1.0,
        "Phase7AdvancedSMTask": 1.0,
        "Phase8RiskTask": 1.0,
        "Phase9IntegrationTask": 1.0,
        "PolicyExecutionTask": 1.0,
    }

    # OPTIONAL: Phase 1 - Short curriculum warmup (5 epochs per task)
    # This helps initialize the shared representations before multi-task learning
    # FIXED: Now trains the actual mt_model (Phase 2 model) directly, ensuring perfect knowledge transfer
    # Each task trains for 5 epochs, updating the backbone + task-specific head
    ENABLE_WARMUP = False  # Warmup disabled - trains mt_model one task at a time

    if ENABLE_WARMUP:
        print("\nRunning curriculum warmup on all 9 tasks...")
        print("(Training the multi-task model directly, one task at a time)")
        print("This ensures perfect knowledge transfer from Phase 1 to Phase 2!\n")

        device = torch.device("mps" if torch.backends.mps.is_available() else
                             ("cuda" if torch.cuda.is_available() else "cpu"))
        mt_model.to(device)

        for task in all_tasks:
            task_name = task.__class__.__name__
            print(f"\n  Warmup: {task_name}")

            # Prepare single-task dataset
            warmup_train_ds = prepare_phase_dataset(
                task,
                train_df,
                sequence_length=settings.model.sequence_length,
                feature_frame=feature_frames["train"],
                feature_columns=feature_columns,
                feature_stats=feature_stats,
                stride=SEQUENCE_STRIDE,
            )
            warmup_val_ds = prepare_phase_dataset(
                task,
                val_df,
                sequence_length=settings.model.sequence_length,
                feature_frame=feature_frames["val"],
                feature_columns=feature_columns,
                feature_stats=feature_stats,
                stride=SEQUENCE_STRIDE,
            )

            # Create DataLoaders
            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                warmup_train_ds,
                batch_size=128 if torch.backends.mps.is_available() or torch.cuda.is_available() else 32,
                shuffle=True,
            )
            val_loader = DataLoader(
                warmup_val_ds,
                batch_size=128 if torch.backends.mps.is_available() or torch.cuda.is_available() else 32,
                shuffle=False,
            )

            # Fresh optimizer for each task warmup
            warmup_optimizer = torch.optim.Adam(mt_model.parameters(), lr=1e-4, weight_decay=1e-4)

            # Train for 5 epochs on this task only
            for epoch in range(5):
                mt_model.train()
                total_loss = 0.0
                num_batches = 0

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Forward pass through all task heads
                    outputs = mt_model(inputs)

                    # Compute loss ONLY for current task (other heads exist but aren't trained)
                    loss = mt_loss_fns[task_name](outputs[task_name], targets)

                    # Backward pass (updates backbone + current task head)
                    warmup_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mt_model.parameters(), 1.0)
                    warmup_optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / max(num_batches, 1)
                print(f"    Epoch {epoch+1}/5: loss={avg_loss:.4f}")

            print(f"    ✓ {task_name} warmup complete")

        print("\n✓ Phase 1 warmup complete! mt_model is now ready for Phase 2 multi-task training.")
        print("  → The backbone and all task heads have been initialized with task-specific knowledge.\n")

    # ============================================================================
    # PHASE 2: MULTI-TASK PRIMARY TRAINING (50+ EPOCHS)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 2: MULTI-TASK PRIMARY TRAINING")
    print("="*80)

    from training.multitask_trainer import MultiTaskTrainingLoop, MultiTaskConfig

    # Configure multi-task training
    # Options: "static", "uncertainty", "gradnorm", "adaptive"
    # FIXED: Using "uncertainty" to automatically balance task weights based on loss magnitude
    # This solves the "Gradient Bully" problem where high-loss tasks dominate training
    WEIGHTING_STRATEGY = "uncertainty"  # CHANGED from "static"
    MONITOR_HEALTH = True  # Enable task health monitoring

    mt_config = MultiTaskConfig(
        epochs=150,
        batch_size=128 if torch.backends.mps.is_available() or torch.cuda.is_available() else 32,
        lr=5e-4,  # FIXED: Set to 5e-4 (safer than 1e-3 for multi-task, higher than 3e-4 for better convergence) + warmup below
        gradient_accumulation=1,
        mixed_precision=torch.backends.mps.is_available() or torch.cuda.is_available(),
        scheduler="cosine_warmup",  # FIXED: Using cosine_warmup (10 epochs warmup from 1e-4 → 5e-4, then cosine decay)
        max_grad_norm=1.0,
        resource_log_interval=1000,
        # Early stopping configuration
        early_stopping_patience=10,
        early_stopping_min_epochs=30,
        # Multi-task specific
        weighting_strategy=WEIGHTING_STRATEGY,
        gradnorm_alpha=1.5,
        gradnorm_lr=0.025,
        adaptive_rate=0.1,
        # Task health monitoring
        monitor_task_health=MONITOR_HEALTH,
        health_check_interval=1,
        stuck_patience=20,  # FIXED: Increased from 5 to 20 - multi-task learning needs more patience
        stuck_variance_threshold=1e-6,  # FIX: Lower threshold significantly. 0.001 was too high for variance check.
        overfitting_ratio_threshold=3.0,  # FIXED: Increased from 2.0 to 3.0 - allow more train/val gap
        loss_explosion_threshold=100.0,
        loss_explosion_ratio=10.0,
        bad_task_action="log_only",  # FIXED: Changed from "reduce_weight" to "log_only" - prevents weight collapse
        weight_reduction_factor=0.5,
        update_task_weights_interval=1,
        # FIXED: Enable per-task gradient normalization to balance gradient magnitudes
        normalize_gradients=True,  # Normalize gradients across tasks (prevents 100-300x imbalance)
        gradient_norm_target=1.0,  # Target norm for gradient normalization
        checkpoint_dir=Path("checkpoints/multitask"),
    )

    print(f"\nMulti-task configuration:")
    print(f"  Weighting strategy: {WEIGHTING_STRATEGY}")
    print(f"  Gradient normalization: {mt_config.normalize_gradients}")
    print(f"  Health monitoring: {MONITOR_HEALTH} (bad_task_action={mt_config.bad_task_action})")
    print(f"  Epochs: {mt_config.epochs}")
    print(f"  Learning rate: {mt_config.lr} (with {mt_config.scheduler} scheduler)")
    print(f"  Batch size: {mt_config.batch_size}")
    print(f"  Health thresholds: stuck_patience={mt_config.stuck_patience}, stuck_variance={mt_config.stuck_variance_threshold}")

    # Create multi-task training loop
    mt_trainer = MultiTaskTrainingLoop(
        model=mt_model,
        optimizer=mt_optimizer,
        task_loss_fns=mt_loss_fns,
        task_weights=task_weights,
        config=mt_config,
        train_dataset=mt_train_ds,
        val_dataset=mt_val_ds,
    )

    # Run multi-task training
    print("\nStarting multi-task training...")
    mt_metrics = mt_trainer.train()

    print(f"\nMulti-task training completed!")
    print(f"Final metrics: {mt_metrics}")

    # Save the multitask model
    mt_ckpt_dir = Path("checkpoints/multitask")
    mt_ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": mt_model.state_dict(),
            "optimizer_state": mt_optimizer.state_dict(),
            "feature_columns": feature_columns,
            "task_order": [t.__class__.__name__ for t in all_tasks],
            "task_weights": task_weights,
            "weighting_strategy": WEIGHTING_STRATEGY,
        },
        mt_ckpt_dir / "multitask_tft.pt",
    )
    print(f"Saved multi-task model to {mt_ckpt_dir / 'multitask_tft.pt'}")

    # ============================================================================
    # PHASE 3: OPTIONAL TASK-SPECIFIC FINE-TUNING WITH EWC
    # ============================================================================
    # Note: EWC implementation will be added separately
    # For now, we skip this phase
    ENABLE_FINE_TUNING = False

    if ENABLE_FINE_TUNING:
        print("\n" + "="*80)
        print("PHASE 3: TASK-SPECIFIC FINE-TUNING WITH EWC")
        print("="*80)
        print("EWC fine-tuning not yet implemented - skipping...")
        # TODO: Implement EWC-based fine-tuning for individual tasks

    # Explainability quick checks on multitask backbone
    try:
        print("\nRunning Explainability...")
        sample_feats, _ = mt_train_ds[0]
        # Ensure tensors and model share the same device
        sample_feats = sample_feats.unsqueeze(0).to(device)
        mt_model.to(device)
        
        smap = saliency_map(mt_model, sample_feats)
        print(f"  Saliency sample mean: {float(smap['saliency'].mean()):.6f}")
        
        # SHAP
        shap_bg = sample_feats[:, :5, :].reshape(1, -1).to(device)
        shap_vals = shap_values(
            lambda x: mt_model(torch.tensor(x, dtype=sample_feats.dtype, device=device))["main"],
            shap_bg,
            shap_bg
        )
        print(f"  SHAP values computed")

        # LIME on one instance
        lime_exp = lime_explanation(
            lambda x: mt_model(torch.tensor(x, dtype=sample_feats.dtype, device=device))["main"],
            sample_feats.reshape(1, -1)
        )
        print("  LIME explanation available")
    except Exception as exc:  # pragma: no cover
        print(f"Explainability step failed: {exc}")


if __name__ == "__main__":
    main()
