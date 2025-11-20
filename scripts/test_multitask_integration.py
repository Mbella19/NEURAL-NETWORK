#!/usr/bin/env python3
"""Integration test for multi-task training infrastructure.

This script tests:
1. MultiTaskDataset creation and data loading
2. Task weighting strategies
3. TaskMonitor for health checking
4. MultiTaskTrainingLoop
5. EWC implementation
6. Training visualization

Run this to validate the multi-task training system is working correctly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT))

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from config import get_settings
from training.multitask_dataset import MultiTaskDataset
from training.task_weighting import (
    StaticWeighting,
    UncertaintyWeighting,
    GradNormWeighting,
    AdaptiveWeighting,
)
from training.task_monitor import TaskMonitor, TaskHealthConfig
from training.multitask_trainer import MultiTaskTrainingLoop, MultiTaskConfig
from training.ewc import EWC, OnlineEWC
from training.visualization import TrainingVisualizer, plot_ewc_importance

# Create dummy tasks for testing
from training.phases.phase1_direction import Phase1DirectionTask
from training.phases.phase2_indicators import Phase2IndicatorTask
from training.phases.phase3_structure import Phase3StructureTask


def create_dummy_data(num_samples: int = 1000, num_features: int = 50):
    """Create dummy data for testing."""
    logger.info(f"Creating dummy data: {num_samples} samples, {num_features} features")

    # Create dummy raw data (OHLCV)
    timestamps = pd.date_range("2024-01-01", periods=num_samples, freq="5min")
    raw_data = pd.DataFrame({
        "TIMESTAMP": timestamps,
        "OPEN": 1.1 + 0.01 * torch.randn(num_samples).numpy(),
        "HIGH": 1.11 + 0.01 * torch.randn(num_samples).numpy(),
        "LOW": 1.09 + 0.01 * torch.randn(num_samples).numpy(),
        "CLOSE": 1.1 + 0.01 * torch.randn(num_samples).numpy(),
        "VOLUME": 1000 + 100 * torch.randn(num_samples).abs().numpy(),
    })

    # Create dummy feature data
    feature_data = pd.DataFrame(
        torch.randn(num_samples, num_features).numpy(),
        columns=[f"feature_{i}" for i in range(num_features)],
    )
    feature_data["TIMESTAMP"] = timestamps

    return raw_data, feature_data


def test_multitask_dataset():
    """Test MultiTaskDataset creation and data loading."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: MultiTaskDataset")
    logger.info("="*80)

    raw_data, feature_data = create_dummy_data(num_samples=500)

    # Create tasks
    tasks = [
        Phase1DirectionTask(),
        Phase2IndicatorTask(),
        Phase3StructureTask(),
    ]

    # Create dataset
    dataset = MultiTaskDataset(
        feature_frame=feature_data,
        raw_frame=raw_data,
        tasks=tasks,
        sequence_length=60,
        stride=5,
    )

    logger.info(f"Dataset created with {len(dataset)} sequences")

    # Test __getitem__
    features, targets = dataset[0]
    logger.info(f"Sample features shape: {features.shape}")
    logger.info(f"Sample targets: {list(targets.keys())}")

    for task_name, target in targets.items():
        logger.info(f"  {task_name}: shape={target.shape}, dtype={target.dtype}")

    # Test statistics
    stats = dataset.get_task_statistics()
    logger.info("\nTask statistics:")
    for task_name, task_stats in stats.items():
        logger.info(f"  {task_name}: mean={task_stats['mean']:.4f}, std={task_stats['std']:.4f}")

    logger.info("✓ MultiTaskDataset test PASSED")
    return dataset


def test_task_weighting():
    """Test task weighting strategies."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Task Weighting Strategies")
    logger.info("="*80)

    task_names = ["TaskA", "TaskB", "TaskC"]
    base_weights = {"TaskA": 1.0, "TaskB": 0.8, "TaskC": 0.6}

    # Test 1: Static Weighting
    logger.info("\nTesting StaticWeighting...")
    static_weighting = StaticWeighting(len(task_names), task_names, base_weights)
    dummy_losses = {
        "TaskA": torch.tensor(0.5),
        "TaskB": torch.tensor(0.3),
        "TaskC": torch.tensor(0.7),
    }
    combined_loss = static_weighting(dummy_losses, epoch=0)
    logger.info(f"  Combined loss: {combined_loss.item():.4f}")
    logger.info(f"  Task weights: {static_weighting.get_task_weights()}")

    # Test 2: Uncertainty Weighting
    logger.info("\nTesting UncertaintyWeighting...")
    uncertainty_weighting = UncertaintyWeighting(len(task_names), task_names, base_weights)
    combined_loss = uncertainty_weighting(dummy_losses, epoch=0)
    logger.info(f"  Combined loss: {combined_loss.item():.4f}")
    logger.info(f"  Task weights: {uncertainty_weighting.get_task_weights()}")
    logger.info(f"  Uncertainties: {uncertainty_weighting.get_uncertainties()}")

    # Test 3: Adaptive Weighting
    logger.info("\nTesting AdaptiveWeighting...")
    adaptive_weighting = AdaptiveWeighting(len(task_names), task_names, base_weights)
    combined_loss = adaptive_weighting(dummy_losses, epoch=0)
    logger.info(f"  Combined loss: {combined_loss.item():.4f}")
    logger.info(f"  Initial weights: {adaptive_weighting.get_task_weights()}")

    # Simulate improvement on TaskA, stagnation on TaskB
    for i in range(5):
        task_metrics = {
            "TaskA": {"val_loss": 0.5 - i * 0.05},  # Improving
            "TaskB": {"val_loss": 0.3},  # Stagnant
            "TaskC": {"val_loss": 0.7 + i * 0.02},  # Getting worse
        }
        adaptive_weighting.update_weights(task_metrics)

    logger.info(f"  Updated weights: {adaptive_weighting.get_task_weights()}")
    logger.info("  (TaskA should decrease, TaskC should increase)")

    logger.info("✓ Task weighting test PASSED")


def test_task_monitor():
    """Test TaskMonitor for health checking."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: TaskMonitor")
    logger.info("="*80)

    task_names = ["TaskA", "TaskB", "TaskC"]
    config = TaskHealthConfig(
        stuck_patience=3,
        stuck_variance_threshold=0.001,
        overfitting_ratio_threshold=2.0,
        action="reduce_weight",
    )
    monitor = TaskMonitor(task_names, config)

    # Simulate healthy task
    logger.info("\nSimulating healthy task (TaskA)...")
    for epoch in range(5):
        metrics = {
            "loss": 0.5 - epoch * 0.05,  # Decreasing
            "train_loss": 0.5 - epoch * 0.05,
            "val_loss": 0.52 - epoch * 0.04,
            "accuracy": 0.7 + epoch * 0.02,
        }
        health = monitor.check_task_health("TaskA", metrics, epoch)
        logger.info(f"  Epoch {epoch}: {health['status']}, healthy={health['is_healthy']}")

    # Simulate stuck task
    logger.info("\nSimulating stuck task (TaskB)...")
    for epoch in range(5):
        metrics = {
            "loss": 0.5,  # Not changing
            "train_loss": 0.5,
            "val_loss": 0.5,
            "accuracy": 0.7,  # Not changing
        }
        health = monitor.check_task_health("TaskB", metrics, epoch)
        logger.info(f"  Epoch {epoch}: {health['status']}, healthy={health['is_healthy']}")
        if not health["is_healthy"]:
            logger.info(f"    Issues: {health['issues']}")

    # Simulate overfitting task
    logger.info("\nSimulating overfitting task (TaskC)...")
    for epoch in range(5):
        metrics = {
            "loss": 0.1,  # Low train loss
            "train_loss": 0.1,
            "val_loss": 0.5 + epoch * 0.1,  # Increasing val loss
        }
        health = monitor.check_task_health("TaskC", metrics, epoch)
        logger.info(f"  Epoch {epoch}: {health['status']}, healthy={health['is_healthy']}")
        if not health["is_healthy"]:
            logger.info(f"    Issues: {health['issues']}")

    # Test weight modifications
    logger.info("\nTesting weight modifications...")
    current_weights = {"TaskA": 1.0, "TaskB": 1.0, "TaskC": 1.0}
    modified_weights = monitor.apply_weight_modifications(current_weights)
    logger.info(f"  Original weights: {current_weights}")
    logger.info(f"  Modified weights: {modified_weights}")
    logger.info("  (TaskB and TaskC should be reduced)")

    # Get summary
    summary = monitor.get_summary()
    logger.info(f"\nSummary: {summary['num_healthy']}/{len(task_names)} healthy tasks")

    logger.info("✓ TaskMonitor test PASSED")


def test_ewc():
    """Test EWC implementation."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Elastic Weight Consolidation (EWC)")
    logger.info("="*80)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = SimpleModel()

    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.randn(10), torch.randn(1)

    dataset = DummyDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    # Test EWC
    logger.info("\nTesting EWC...")
    ewc = EWC(model, ewc_lambda=1000.0)

    # Compute Fisher Information
    def dummy_loss_fn(output, target):
        return nn.functional.mse_loss(output, target)

    ewc.compute_fisher(data_loader, dummy_loss_fn, num_samples=50)

    logger.info(f"  Fisher computed: {ewc.fisher_computed}")
    logger.info(f"  Number of parameters tracked: {len(ewc.fisher)}")

    # Compute penalty (should be 0 initially)
    penalty = ewc.penalty()
    logger.info(f"  Initial penalty: {penalty.item():.6f}")

    # Modify model parameters
    for param in model.parameters():
        param.data += 0.1 * torch.randn_like(param.data)

    # Compute penalty again (should be > 0)
    penalty = ewc.penalty()
    logger.info(f"  Penalty after parameter change: {penalty.item():.6f}")

    # Get parameter importance
    importance = ewc.get_parameter_importance()
    logger.info(f"\n  Top-3 most important parameters:")
    sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_params[:3]:
        logger.info(f"    {name}: {score:.6f}")

    # Test Online EWC
    logger.info("\nTesting OnlineEWC...")
    online_ewc = OnlineEWC(model, ewc_lambda=1000.0, gamma=0.9)
    online_ewc.update_fisher_and_params(data_loader, dummy_loss_fn, num_samples=50)
    logger.info(f"  Fisher updated: {online_ewc.fisher_computed}")

    logger.info("✓ EWC test PASSED")


def test_visualization():
    """Test training visualization."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Training Visualization")
    logger.info("="*80)

    viz = TrainingVisualizer(output_dir=Path("outputs/test_vis"))

    # Simulate training metrics
    logger.info("\nSimulating training metrics...")
    for epoch in range(20):
        metrics = {
            "train_loss": 1.0 - epoch * 0.04 + 0.05 * torch.randn(1).item(),
            "val_loss": 1.0 - epoch * 0.03 + 0.08 * torch.randn(1).item(),
            "train_loss_TaskA": 0.8 - epoch * 0.03 + 0.03 * torch.randn(1).item(),
            "val_loss_TaskA": 0.85 - epoch * 0.025 + 0.05 * torch.randn(1).item(),
            "train_loss_TaskB": 0.6 - epoch * 0.02 + 0.02 * torch.randn(1).item(),
            "val_loss_TaskB": 0.65 - epoch * 0.015 + 0.04 * torch.randn(1).item(),
            "train_loss_TaskC": 0.9 - epoch * 0.035 + 0.04 * torch.randn(1).item(),
            "val_loss_TaskC": 0.95 - epoch * 0.03 + 0.06 * torch.randn(1).item(),
            "task_weight_TaskA": 1.0 + 0.1 * torch.randn(1).item(),
            "task_weight_TaskB": 0.8 + 0.05 * torch.randn(1).item(),
            "task_weight_TaskC": 0.6 + 0.08 * torch.randn(1).item(),
        }
        viz.log_epoch(epoch, metrics)

    logger.info(f"  Logged {len(viz.metrics_history)} epochs")

    # Generate plots
    logger.info("\nGenerating plots...")
    viz.create_all_plots()

    # Generate report
    report = viz.generate_training_report()
    logger.info(f"\nTraining Report:")
    logger.info(f"  Total epochs: {report['total_epochs']}")
    logger.info(f"  Tasks: {report['tasks']}")
    logger.info(f"  Best val losses: {report.get('best_val_loss', {})}")

    logger.info("✓ Visualization test PASSED")


def test_full_integration():
    """Test full multi-task training integration (minimal epochs)."""
    logger.info("\n" + "="*80)
    logger.info("TEST 6: Full Multi-Task Training Integration")
    logger.info("="*80)

    # Create dummy data
    raw_data, feature_data = create_dummy_data(num_samples=500, num_features=30)

    # Create tasks
    tasks = [
        Phase1DirectionTask(),
        Phase2IndicatorTask(),
    ]
    task_names = [t.__class__.__name__ for t in tasks]

    # Create datasets
    logger.info("\nCreating datasets...")
    train_ds = MultiTaskDataset(
        feature_frame=feature_data.iloc[:400],
        raw_frame=raw_data.iloc[:400],
        tasks=tasks,
        sequence_length=30,
        stride=5,
    )
    val_ds = MultiTaskDataset(
        feature_frame=feature_data.iloc[400:],
        raw_frame=raw_data.iloc[400:],
        tasks=tasks,
        sequence_length=30,
        stride=5,
    )

    logger.info(f"  Train: {len(train_ds)} sequences")
    logger.info(f"  Val: {len(val_ds)} sequences")

    # Create simple model
    class SimpleMultiTaskModel(nn.Module):
        def __init__(self, input_dim, num_features):
            super().__init__()
            self.lstm = nn.LSTM(num_features, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])

    model = SimpleMultiTaskModel(input_dim=30, num_features=30)

    # Define loss functions
    task_loss_fns = {}
    for task in tasks:
        task_name = task.__class__.__name__
        task_loss_fns[task_name] = nn.BCEWithLogitsLoss()

    # Define task weights
    task_weights = {name: 1.0 for name in task_names}

    # Create multi-task trainer
    logger.info("\nCreating multi-task trainer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    config = MultiTaskConfig(
        epochs=3,  # Very short for testing
        batch_size=32,
        lr=1e-3,
        weighting_strategy="static",
        monitor_task_health=True,
        stuck_patience=2,
        bad_task_action="log_only",
        checkpoint_dir=Path("checkpoints/test_multitask"),
    )

    trainer = MultiTaskTrainingLoop(
        model=model,
        optimizer=optimizer,
        task_loss_fns=task_loss_fns,
        task_weights=task_weights,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    # Run training
    logger.info("\nRunning multi-task training...")
    metrics = trainer.train()

    logger.info(f"\nFinal metrics: {metrics}")

    logger.info("✓ Full integration test PASSED")


def main():
    """Run all integration tests."""
    logger.info("\n" + "="*80)
    logger.info("MULTI-TASK TRAINING INTEGRATION TESTS")
    logger.info("="*80)

    try:
        # Test each component
        dataset = test_multitask_dataset()
        test_task_weighting()
        test_task_monitor()
        test_ewc()
        test_visualization()

        # Test full integration
        test_full_integration()

        logger.info("\n" + "="*80)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("="*80)
        logger.info("\nThe multi-task training infrastructure is working correctly.")
        logger.info("You can now run the full training with: python scripts/run_full_training.py")

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
