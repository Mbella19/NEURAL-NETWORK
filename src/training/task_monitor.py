"""Task monitoring and health checking for multi-task learning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class TaskHealthConfig:
    """Configuration for task health monitoring."""

    # Stuck accuracy detection
    stuck_patience: int = 5  # Number of epochs to check
    stuck_variance_threshold: float = 0.001  # Max variance for "stuck"

    # Overfitting detection
    overfitting_ratio_threshold: float = 2.0  # val_loss / train_loss

    # Loss explosion detection
    loss_explosion_threshold: float = 100.0  # Absolute loss value
    loss_explosion_ratio: float = 10.0  # Relative to median task loss

    # NaN/Inf detection
    check_nan_inf: bool = True

    # Actions to take when bad task detected
    action: str = "reduce_weight"  # Options: "disable", "reduce_weight", "log_only"
    weight_reduction_factor: float = 0.5  # Multiply weight by this when reducing


class TaskMonitor:
    """Monitor task health during multi-task training and handle problematic tasks."""

    def __init__(self, task_names: List[str], config: Optional[TaskHealthConfig] = None):
        """Initialize task monitor.

        Args:
            task_names: List of task names to monitor
            config: Health checking configuration
        """
        self.task_names = task_names
        self.config = config or TaskHealthConfig()

        # Track metrics history for each task
        self.history: Dict[str, List[Dict[str, float]]] = {
            name: [] for name in task_names
        }

        # Track task health flags
        self.is_healthy: Dict[str, bool] = {name: True for name in task_names}
        self.health_issues: Dict[str, List[str]] = {name: [] for name in task_names}

        # Track task weight modifications
        self.weight_modifications: Dict[str, float] = {name: 1.0 for name in task_names}

    def check_task_health(
        self,
        task_name: str,
        metrics: Dict[str, float],
        epoch: int,
    ) -> Dict[str, any]:
        """Check if a task is behaving healthily.

        Args:
            task_name: Name of the task to check
            metrics: Dict of metrics (loss, accuracy, etc.)
            epoch: Current training epoch

        Returns:
            Dict with health status and any issues detected
        """
        # Record metrics
        self.history[task_name].append({**metrics, "epoch": epoch})

        issues = []
        status = "healthy"

        # Check 1: Stuck accuracy (not learning)
        stuck_issue = self._check_stuck_accuracy(task_name)
        if stuck_issue:
            issues.append(stuck_issue)
            status = "stuck_accuracy"

        # Check 2: Overfitting (val >> train)
        overfit_issue = self._check_overfitting(task_name, metrics)
        if overfit_issue:
            issues.append(overfit_issue)
            if status == "healthy":
                status = "overfitting"

        # Check 3: Loss explosion
        explosion_issue = self._check_loss_explosion(task_name, metrics)
        if explosion_issue:
            issues.append(explosion_issue)
            status = "exploding_loss"

        # Check 4: NaN/Inf values
        nan_issue = self._check_nan_inf(metrics)
        if nan_issue:
            issues.append(nan_issue)
            status = "nan_detected"

        # Update health status
        if issues:
            self.is_healthy[task_name] = False
            self.health_issues[task_name].extend(issues)
        else:
            # Clear issues if task has recovered
            if not self.is_healthy[task_name] and len(self.history[task_name]) >= self.config.stuck_patience:
                # Check if recent epochs show improvement
                recent = self.history[task_name][-self.config.stuck_patience:]
                losses = [m.get("loss", float("inf")) for m in recent]
                if losses[-1] < losses[0] * 0.95:  # 5% improvement
                    self.is_healthy[task_name] = True
                    self.health_issues[task_name] = []
                    status = "recovered"

        return {
            "status": status,
            "is_healthy": self.is_healthy[task_name],
            "issues": issues,
            "weight_modification": self.weight_modifications[task_name],
        }

    def _check_stuck_accuracy(self, task_name: str) -> Optional[str]:
        """Check if accuracy/loss has been stuck (constant) for too long."""
        history = self.history[task_name]

        if len(history) < self.config.stuck_patience:
            return None

        # Get recent accuracy or loss values
        recent = history[-self.config.stuck_patience:]

        # Check accuracy variance if available
        if "accuracy" in recent[0]:
            accuracies = [m["accuracy"] for m in recent]
            variance = float(np.var(accuracies))
            if variance < self.config.stuck_variance_threshold:
                return f"Stuck accuracy (var={variance:.6f}, last={accuracies[-1]:.4f})"

        # Check loss variance
        if "loss" in recent[0]:
            losses = [m["loss"] for m in recent]
            variance = float(np.var(losses))
            mean_loss = float(np.mean(losses))
            # Relative variance (coefficient of variation)
            rel_variance = variance / (mean_loss + 1e-8)
            if rel_variance < self.config.stuck_variance_threshold:
                return f"Stuck loss (rel_var={rel_variance:.6f}, loss={mean_loss:.4f})"

        return None

    def _check_overfitting(self, task_name: str, metrics: Dict[str, float]) -> Optional[str]:
        """Check if validation loss is much higher than training loss."""
        if "val_loss" not in metrics or "train_loss" not in metrics:
            return None

        val_loss = metrics["val_loss"]
        train_loss = metrics["train_loss"]

        if train_loss > 0 and val_loss / train_loss > self.config.overfitting_ratio_threshold:
            ratio = val_loss / train_loss
            return f"Overfitting (val/train={ratio:.2f}, val={val_loss:.4f}, train={train_loss:.4f})"

        return None

    def _check_loss_explosion(self, task_name: str, metrics: Dict[str, float]) -> Optional[str]:
        """Check if loss has exploded to very high values."""
        if "loss" not in metrics:
            return None

        loss = metrics["loss"]

        # Absolute threshold
        if loss > self.config.loss_explosion_threshold:
            return f"Exploding loss (absolute={loss:.4f})"

        # Relative threshold (compared to task's own history)
        history = self.history[task_name]
        if len(history) >= 5:
            recent_losses = [m["loss"] for m in history[-5:] if "loss" in m]
            if recent_losses:
                median_loss = float(np.median(recent_losses))
                if median_loss > 0 and loss > median_loss * self.config.loss_explosion_ratio:
                    return f"Exploding loss (relative, current={loss:.4f}, median={median_loss:.4f})"

        return None

    def _check_nan_inf(self, metrics: Dict[str, float]) -> Optional[str]:
        """Check for NaN or Inf values in metrics."""
        if not self.config.check_nan_inf:
            return None

        for key, value in metrics.items():
            if not np.isfinite(value):
                return f"Non-finite value detected ({key}={value})"

        return None

    def apply_weight_modifications(self, task_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight modifications based on task health.

        Args:
            task_weights: Current task weights

        Returns:
            Modified task weights with bad tasks handled
        """
        modified_weights = task_weights.copy()

        for task_name in self.task_names:
            if not self.is_healthy[task_name]:
                if self.config.action == "disable":
                    # Set weight to 0 (disable task)
                    modified_weights[task_name] = 0.0
                    self.weight_modifications[task_name] = 0.0

                elif self.config.action == "reduce_weight":
                    # Reduce weight
                    modified_weights[task_name] *= self.config.weight_reduction_factor
                    self.weight_modifications[task_name] *= self.config.weight_reduction_factor

                # "log_only" doesn't modify weights

        return modified_weights

    def get_summary(self) -> Dict[str, any]:
        """Get a summary of task health status.

        Returns:
            Dict with health statistics for all tasks
        """
        summary = {
            "num_healthy": sum(self.is_healthy.values()),
            "num_unhealthy": len(self.task_names) - sum(self.is_healthy.values()),
            "tasks": {},
        }

        for task_name in self.task_names:
            summary["tasks"][task_name] = {
                "is_healthy": self.is_healthy[task_name],
                "issues": self.health_issues[task_name],
                "weight_modification": self.weight_modifications[task_name],
                "num_epochs_tracked": len(self.history[task_name]),
            }

        return summary

    def reset_task(self, task_name: str):
        """Reset health status for a specific task.

        Args:
            task_name: Name of task to reset
        """
        self.is_healthy[task_name] = True
        self.health_issues[task_name] = []
        self.weight_modifications[task_name] = 1.0

    def get_active_tasks(self) -> List[str]:
        """Get list of currently healthy/active tasks.

        Returns:
            List of task names that are healthy or have non-zero weight
        """
        return [
            name for name in self.task_names
            if self.is_healthy[name] or self.weight_modifications[name] > 0.0
        ]


__all__ = ["TaskMonitor", "TaskHealthConfig"]
