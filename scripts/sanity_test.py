#!/usr/bin/env python3
"""Sanity test: Verify each task can overfit a small dataset.

This script tests if each task is fundamentally learnable by attempting
to overfit 1000 samples. If a task cannot achieve low loss on a tiny dataset,
it indicates one of the following problems:
- Label computation bug (compute_targets)
- Feature/target misalignment
- Loss function mismatch
- Broken task definition

Expected behavior:
- Binary classification tasks: Should achieve loss < 0.3, accuracy > 0.80
- Multi-class tasks: Should achieve loss < 0.5
- Regression tasks: Should achieve MSE < 0.1
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT))

from data_loader.data_loader import ingest_training_data
from data_loader.preprocessor import preprocess_dataset
from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline
from training.phases.phase1_direction import Phase1DirectionTask
from training.phases.phase2_indicators import Phase2IndicatorTask
from training.phases.phase3_structure import Phase3StructureTask
from training.phases.phase4_smart_money import Phase4SmartMoneyTask
from training.phases.phase5_candlesticks import Phase5CandlestickTask
from training.phases.phase6_sr_levels import Phase6SupportResistanceTask
from training.phases.phase7_advanced_sm import Phase7AdvancedSMTask
from training.phases.phase8_risk import Phase8RiskTask
from training.phases.phase9_integration import Phase9IntegrationTask
from training.loss_wrappers import make_weighted_bce
from config import get_settings


class SimpleDataset(Dataset):
    """Simple dataset for sanity testing."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor, sequence_length: int = 60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.num_sequences = max(0, len(features) - sequence_length + 1)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        start = idx
        end = start + self.sequence_length
        window = self.features[start:end]
        target = self.targets[end - 1]
        return window, target


class SimpleHead(nn.Module):
    """Simple model head for sanity testing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        # Use last timestep
        last_step = x[:, -1, :]  # [batch, features]
        return self.network(last_step)


def sanity_test_task(
    task,
    task_name: str,
    train_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    n_samples: int = 1000,
    sequence_length: int = 60,
    epochs: int = 100,
    device: str = "cpu",
) -> dict:
    """Test if a task can overfit a small dataset.

    Args:
        task: Task instance (e.g., Phase1DirectionTask())
        task_name: Task name for logging
        train_df: Raw OHLCV dataframe
        feature_frame: Precomputed features dataframe
        feature_columns: List of feature column names
        n_samples: Number of samples to use
        sequence_length: Sequence length
        epochs: Number of epochs to train
        device: Device to use

    Returns:
        Dict with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing {task_name}...")
    print(f"{'='*80}")

    # Prepare small dataset
    small_df = train_df.iloc[:n_samples].copy()
    small_features = feature_frame.iloc[:n_samples].copy()

    # Select relevant features for this task
    if hasattr(task.config, 'feature_subset'):
        relevant_features = list(task.config.feature_subset)
    else:
        relevant_features = feature_columns[:50]  # Use first 50 features as fallback

    # Ensure all features exist, fill missing with 0
    features_df = small_features.select_dtypes(include=["number"]).reindex(
        columns=relevant_features, fill_value=0.0
    )

    # Compute targets
    try:
        targets_tensor = task.compute_targets(small_df, feature_frame=small_features)
    except Exception as e:
        print(f"  ✗ FAILED: Cannot compute targets - {e}")
        return {"status": "failed", "reason": f"compute_targets error: {e}"}

    # Convert features to tensor
    features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
    features_tensor = torch.nan_to_num(features_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize features
    mean = features_tensor.mean(dim=0)
    std = features_tensor.std(dim=0).clamp(min=1e-6)
    features_tensor = (features_tensor - mean) / std

    # Create dataset
    dataset = SimpleDataset(features_tensor, targets_tensor, sequence_length=sequence_length)

    if len(dataset) < 10:
        print(f"  ✗ FAILED: Insufficient samples after windowing ({len(dataset)} samples)")
        return {"status": "failed", "reason": "insufficient_samples"}

    # Check target statistics
    if isinstance(task.loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
        pos_rate = (targets_tensor > 0.5).float().mean().item()
        print(f"  Target statistics: pos_rate={pos_rate:.2%}")

        if pos_rate < 0.01 or pos_rate > 0.99:
            print(f"  ⚠ WARNING: Extreme imbalance (pos_rate={pos_rate:.2%})")

    # Create simple model
    input_dim = features_tensor.shape[1]
    output_dim = task.output_dim
    model = SimpleHead(input_dim, hidden_dim=128, output_dim=output_dim).to(device)

    # Setup loss function with proper pos_weight for BCE
    if isinstance(task.loss_fn, nn.BCEWithLogitsLoss):
        pos_rate = (targets_tensor > 0.5).float().mean().item()
        raw_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)
        pos_weight = torch.tensor(min(20.0, max(1.0, raw_pos_weight)))
        loss_fn = make_weighted_bce(pos_weight=pos_weight)
        print(f"  Using pos_weight={pos_weight.item():.2f}")
    else:
        loss_fn = task.loss_fn

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train
    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_losses = []
        epoch_accs = []

        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # Compute accuracy for classification tasks
            if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                with torch.no_grad():
                    preds = torch.sigmoid(logits) > 0.5
                    acc = (preds == (targets > 0.5)).float().mean().item()
                    epoch_accs.append(acc)
            elif isinstance(loss_fn, nn.CrossEntropyLoss):
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == targets).float().mean().item()
                    epoch_accs.append(acc)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)

        if epoch_accs:
            avg_acc = sum(epoch_accs) / len(epoch_accs)
            accuracies.append(avg_acc)

        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            if epoch_accs:
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={avg_acc:.4f}")
            else:
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}")

    # Evaluate final performance
    final_loss = losses[-1]

    # Determine success criteria based on task type
    if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
        # Binary classification
        final_acc = accuracies[-1] if accuracies else 0.0
        success = final_loss < 0.3 and final_acc > 0.70

        if success:
            print(f"  ✓ PASSED: Overfits successfully (loss={final_loss:.4f}, acc={final_acc:.4f})")
            return {"status": "passed", "final_loss": final_loss, "final_accuracy": final_acc}
        else:
            print(f"  ✗ FAILED: Cannot overfit (loss={final_loss:.4f}, acc={final_acc:.4f})")
            return {"status": "failed", "reason": "cannot_overfit", "final_loss": final_loss, "final_accuracy": final_acc}

    elif isinstance(loss_fn, nn.CrossEntropyLoss):
        # Multi-class classification
        final_acc = accuracies[-1] if accuracies else 0.0
        success = final_loss < 0.5 and final_acc > 0.60

        if success:
            print(f"  ✓ PASSED: Overfits successfully (loss={final_loss:.4f}, acc={final_acc:.4f})")
            return {"status": "passed", "final_loss": final_loss, "final_accuracy": final_acc}
        else:
            print(f"  ✗ FAILED: Cannot overfit (loss={final_loss:.4f}, acc={final_acc:.4f})")
            return {"status": "failed", "reason": "cannot_overfit", "final_loss": final_loss, "final_accuracy": final_acc}

    else:
        # Regression
        success = final_loss < 0.5

        if success:
            print(f"  ✓ PASSED: Overfits successfully (loss={final_loss:.4f})")
            return {"status": "passed", "final_loss": final_loss}
        else:
            print(f"  ✗ FAILED: Cannot overfit (loss={final_loss:.4f})")
            return {"status": "failed", "reason": "cannot_overfit", "final_loss": final_loss}


def main():
    """Run sanity tests on all tasks."""
    print("\n" + "="*80)
    print("SANITY TEST: Verifying task learnability")
    print("="*80)

    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    settings = get_settings()
    dfs = ingest_training_data(settings)

    train_df = dfs["BTCUSDT_5m_train"]
    print(f"  Loaded {len(train_df)} samples")

    # Preprocess
    print("Preprocessing...")
    out = preprocess_dataset(train_df, test_df=None, val_df=None)
    train_df = out["train"]

    # Compute features
    print("Computing features...")
    pipeline = build_default_feature_pipeline()
    feature_frame = run_feature_pipeline(train_df, pipeline=pipeline).dataframe
    feature_columns = list(feature_frame.select_dtypes(include=["number"]).columns)
    print(f"  Computed {len(feature_columns)} features")

    # Define all tasks
    tasks = [
        (Phase1DirectionTask(), "Phase1DirectionTask"),
        (Phase2IndicatorTask(), "Phase2IndicatorTask"),
        (Phase3StructureTask(), "Phase3StructureTask"),
        (Phase4SmartMoneyTask(), "Phase4SmartMoneyTask"),
        (Phase5CandlestickTask(), "Phase5CandlestickTask"),
        (Phase6SupportResistanceTask(), "Phase6SupportResistanceTask"),
        (Phase7AdvancedSMTask(), "Phase7AdvancedSMTask"),
        (Phase8RiskTask(), "Phase8RiskTask"),
        (Phase9IntegrationTask(), "Phase9IntegrationTask"),
    ]

    # Run sanity test on each task
    results = {}
    for task, task_name in tasks:
        try:
            result = sanity_test_task(
                task=task,
                task_name=task_name,
                train_df=train_df,
                feature_frame=feature_frame,
                feature_columns=feature_columns,
                n_samples=1500,  # Use 1500 samples to get enough after windowing
                sequence_length=60,
                epochs=100,
                device=device,
            )
            results[task_name] = result
        except Exception as e:
            print(f"  ✗ FAILED: Unexpected error - {e}")
            import traceback
            traceback.print_exc()
            results[task_name] = {"status": "error", "reason": str(e)}

    # Print summary
    print("\n" + "="*80)
    print("SANITY TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    failed = sum(1 for r in results.values() if r.get("status") in ["failed", "error"])

    print(f"\nResults: {passed} passed, {failed} failed (out of {len(results)} tasks)")
    print("\nPer-task results:")

    for task_name, result in results.items():
        status = result.get("status", "unknown")
        if status == "passed":
            symbol = "✓"
        elif status in ["failed", "error"]:
            symbol = "✗"
        else:
            symbol = "?"

        print(f"  {symbol} {task_name:35s} - {status}")
        if status == "failed" and "reason" in result:
            print(f"      Reason: {result['reason']}")
            if "final_loss" in result:
                print(f"      Final loss: {result['final_loss']:.4f}")
            if "final_accuracy" in result:
                print(f"      Final accuracy: {result['final_accuracy']:.4f}")

    print("\n" + "="*80)

    if failed == 0:
        print("✓ All tasks passed sanity test!")
    else:
        print(f"✗ {failed} task(s) failed sanity test - these tasks need debugging")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
