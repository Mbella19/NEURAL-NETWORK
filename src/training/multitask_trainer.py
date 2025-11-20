"""Multi-task training loop with task weighting and health monitoring."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset
from loguru import logger

from models.base_model import BaseModel
from training.trainer import TrainingLoop, TrainerConfig
from training.task_weighting import TaskWeighting, StaticWeighting
from training.task_monitor import TaskMonitor, TaskHealthConfig


@dataclass
class MultiTaskConfig(TrainerConfig):
    """Configuration for multi-task training."""

    # Task weighting strategy
    weighting_strategy: str = "static"  # Options: "static", "uncertainty", "gradnorm", "adaptive"
    gradnorm_alpha: float = 1.5  # For GradNorm
    gradnorm_lr: float = 0.025  # For GradNorm
    adaptive_rate: float = 0.1  # For Adaptive weighting

    # Task health monitoring
    monitor_task_health: bool = True
    health_check_interval: int = 1  # Check every N epochs
    stuck_patience: int = 5  # Epochs to check for stuck accuracy
    stuck_variance_threshold: float = 0.001  # Max variance for "stuck"
    overfitting_ratio_threshold: float = 2.0  # val_loss / train_loss
    loss_explosion_threshold: float = 100.0  # Absolute loss value
    loss_explosion_ratio: float = 10.0  # Relative to median
    bad_task_action: str = "reduce_weight"  # Options: "disable", "reduce_weight", "log_only"
    weight_reduction_factor: float = 0.5  # Multiply weight by this when reducing

    # GradNorm specific
    update_task_weights_interval: int = 1  # Update weights every N epochs

    # Per-task gradient normalization
    normalize_gradients: bool = True  # Normalize gradient magnitudes across tasks
    gradient_norm_target: float = 1.0  # Target norm for gradient normalization


class MultiTaskTrainingLoop(TrainingLoop):
    """Training loop for multi-task learning with task weighting and health monitoring.

    Extends TrainingLoop to support:
    - Multiple task losses combined via task weighting strategies
    - Per-task metrics tracking
    - Task health monitoring (stuck accuracy, overfitting, loss explosion)
    - Automatic bad task handling (disable, reduce weight, or log only)
    - GradNorm gradient balancing
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        task_loss_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        task_weights: Dict[str, float],
        config: MultiTaskConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        task_weighting: Optional[TaskWeighting] = None,
    ) -> None:
        """Initialize multi-task training loop.

        Args:
            model: Model to train
            optimizer: Optimizer
            task_loss_fns: Dict mapping task_name -> loss function
            task_weights: Dict mapping task_name -> base weight
            config: Multi-task training configuration
            train_dataset: Training dataset (should be MultiTaskDataset)
            val_dataset: Validation dataset (should be MultiTaskDataset)
            task_weighting: Task weighting strategy (if None, will create based on config)
        """
        # Store multi-task configuration
        self.task_names = list(task_loss_fns.keys())
        self.task_loss_fns = task_loss_fns
        self.task_weights = task_weights
        self.mtl_config = config

        # Initialize parent first (this sets self.config, self.device, etc.)
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=lambda x, y: torch.tensor(0.0),  # Dummy, not used
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        
        # FIXED: Explicitly store dataset references for access to task_feature_masks
        self.train_dataset = train_dataset

        # Create task weighting strategy (after parent init so we have self.device)
        if task_weighting is None:
            task_weighting = self._create_weighting_strategy(config, task_weights)
        self.task_weighting = task_weighting
        self.task_weighting.to(self.device)

        # Create task monitor
        if config.monitor_task_health:
            health_config = TaskHealthConfig(
                stuck_patience=config.stuck_patience,
                stuck_variance_threshold=config.stuck_variance_threshold,
                overfitting_ratio_threshold=config.overfitting_ratio_threshold,
                loss_explosion_threshold=config.loss_explosion_threshold,
                loss_explosion_ratio=config.loss_explosion_ratio,
                action=config.bad_task_action,
                weight_reduction_factor=config.weight_reduction_factor,
            )
            self.task_monitor = TaskMonitor(self.task_names, health_config)
        else:
            self.task_monitor = None

        # Track per-task metrics history
        self.task_history: Dict[str, List[Dict[str, float]]] = {
            name: [] for name in self.task_names
        }

        # Override DataLoaders with custom collate_fn for multi-task batching
        # The default collate can't handle dict of targets
        from torch.utils.data import DataLoader

        def multitask_collate_fn(batch):
            """Custom collate function to batch multi-task data."""
            features = torch.stack([item[0] for item in batch])
            targets_dict = {}
            for task_name in batch[0][1].keys():
                targets_dict[task_name] = torch.stack([item[1][task_name] for item in batch])
            return features, targets_dict

        pin_memory = self.device.type == "cuda"
        if self.config.pin_memory is not None:
            pin_memory = self.config.pin_memory

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            collate_fn=multitask_collate_fn,  # CRITICAL: Handle dict targets
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=config.num_workers,
                persistent_workers=config.num_workers > 0,
                collate_fn=multitask_collate_fn,  # CRITICAL: Handle dict targets
            )

        logger.bind(source="multitask_trainer").info(
            f"Initialized multi-task trainer with {len(self.task_names)} tasks: {self.task_names}"
        )
        logger.bind(source="multitask_trainer").info(
            f"Weighting strategy: {config.weighting_strategy}, Health monitoring: {config.monitor_task_health}"
        )

    def _create_weighting_strategy(
        self, config: MultiTaskConfig, task_weights: Dict[str, float]
    ) -> TaskWeighting:
        """Create task weighting strategy based on config."""
        from training.task_weighting import (
            StaticWeighting,
            UncertaintyWeighting,
            GradNormWeighting,
            AdaptiveWeighting,
        )

        num_tasks = len(self.task_names)

        if config.weighting_strategy == "static":
            return StaticWeighting(num_tasks, self.task_names, task_weights)

        elif config.weighting_strategy == "uncertainty":
            return UncertaintyWeighting(num_tasks, self.task_names, task_weights)

        elif config.weighting_strategy == "gradnorm":
            return GradNormWeighting(
                num_tasks,
                self.task_names,
                task_weights,
                alpha=config.gradnorm_alpha,
                lr=config.gradnorm_lr,
            )

        elif config.weighting_strategy == "adaptive":
            return AdaptiveWeighting(
                num_tasks,
                self.task_names,
                task_weights,
                adapt_rate=config.adaptive_rate,
            )

        else:
            raise ValueError(f"Unknown weighting strategy: {config.weighting_strategy}")

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch with multi-task learning."""
        self.model.train()
        device = self._device()

        # Track totals for each task
        task_losses: Dict[str, float] = defaultdict(float)
        task_metrics: Dict[str, Dict[str, float]] = {
            name: defaultdict(float) for name in self.task_names
        }

        step_count = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            inputs, targets_dict = batch
            inputs = inputs.to(device).contiguous()
            task_masks = getattr(self.train_dataset, "task_feature_masks", {})

            # Move all task targets to device
            targets_dict = {
                task_name: target.to(device)
                for task_name, target in targets_dict.items()
            }

            if not torch.isfinite(inputs).all():
                raise RuntimeError(f"Non-finite inputs detected at epoch {epoch}, step {step}")

            # Compute loss for each task
            individual_losses = {}
            for task_name in self.task_names:
                if task_name not in targets_dict:
                    logger.bind(source="multitask_trainer").warning(
                        f"Task {task_name} missing from batch targets, skipping"
                    )
                    continue

                target = targets_dict[task_name]
                mask = task_masks.get(task_name)
                if mask is not None:
                    mask = mask.to(device)
                    task_inputs = inputs * mask
                else:
                    task_inputs = inputs

                # Forward pass per task (masked features if provided)
                logits_dict = self.model(task_inputs)

                task_logits = logits_dict.get(task_name, logits_dict.get("main"))
                if task_logits is None:
                    logger.bind(source="multitask_trainer").warning(
                        f"Task {task_name} missing from model outputs, skipping"
                    )
                    continue
                loss_fn = self.task_loss_fns[task_name]

                # Validate dimensions
                if task_logits.shape[0] != target.shape[0]:
                    raise RuntimeError(
                        f"Batch size mismatch for {task_name}: "
                        f"logits={task_logits.shape}, targets={target.shape}"
                    )

                # Check for non-finite values
                if not torch.isfinite(task_logits).all():
                    logger.bind(source="multitask_trainer").error(
                        f"Non-finite logits for task {task_name} at epoch {epoch}, step {step}"
                    )
                    raise RuntimeError(f"Non-finite logits for task {task_name}")

                # Compute task loss
                task_loss = loss_fn(task_logits, target)

                if not torch.isfinite(task_loss):
                    logger.bind(source="multitask_trainer").error(
                        f"Non-finite loss for task {task_name} at epoch {epoch}, step {step}"
                    )
                    raise RuntimeError(f"Non-finite loss for task {task_name}")

                individual_losses[task_name] = task_loss
                task_losses[task_name] += task_loss.item()

                # Compute task-specific metrics
                batch_task_metrics = self._compute_batch_metrics(task_logits, target)
                for key, value in batch_task_metrics.items():
                    task_metrics[task_name][key] += value

            # Compute gradient norms every 5 epochs (on first batch for efficiency)
            # This helps diagnose multi-task gradient interference
            if epoch % 5 == 0 and step == 1:
                grad_norms = self._compute_gradient_norms(individual_losses)
                # Store in task_metrics for later averaging
                for task_name, grad_norm in grad_norms.items():
                    task_metrics[task_name]["grad_norm"] = grad_norm
                    logger.bind(source="multitask_trainer").info(
                        f"Gradient norm for {task_name}: {grad_norm:.6f}"
                    )

            # FIXED: Apply per-task gradient normalization to balance gradient magnitudes
            if self.mtl_config.normalize_gradients:
                # Normalize gradients across tasks to prevent imbalance
                combined_loss = self._apply_gradient_normalization(individual_losses)
                combined_loss = combined_loss / self.config.gradient_accumulation
            else:
                # Standard approach: combine losses and do single backward pass
                combined_loss = self.task_weighting(individual_losses, epoch)
                combined_loss = combined_loss / self.config.gradient_accumulation

                if not torch.isfinite(combined_loss):
                    logger.bind(source="multitask_trainer").error(
                        f"Non-finite combined loss at epoch {epoch}, step {step}"
                    )
                    raise RuntimeError("Non-finite combined loss encountered")

                # Backward pass
                combined_loss.backward()

            # Update weights (every N steps OR at the end of epoch)
            if step % self.config.gradient_accumulation == 0 or step == step_count:
                from models.utils import clip_gradients
                clip_gradients(self.model, self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Average metrics over all steps
        averaged_metrics = {}

        # Overall train loss (average of all task losses)
        overall_loss = sum(task_losses.values()) / (len(self.task_names) * max(step_count, 1))
        averaged_metrics["train_loss"] = overall_loss

        # Per-task metrics
        for task_name in self.task_names:
            task_loss = task_losses[task_name] / max(step_count, 1)
            averaged_metrics[f"train_loss_{task_name}"] = task_loss

            # Add other task-specific metrics
            for metric_name, total_value in task_metrics[task_name].items():
                avg_value = total_value / max(step_count, 1)
                averaged_metrics[f"train_{metric_name}_{task_name}"] = avg_value

        # Add task weights to metrics
        current_weights = self.task_weighting.get_task_weights()
        for task_name, weight in current_weights.items():
            averaged_metrics[f"task_weight_{task_name}"] = weight

        return averaged_metrics

    def _run_validation(self) -> Dict[str, float]:
        """Run validation with per-task metrics."""
        if not self.val_loader:
            return {}

        self.model.eval()
        device = self._device()

        # Track totals for each task
        task_losses: Dict[str, float] = defaultdict(float)
        task_metrics: Dict[str, Dict[str, float]] = {
            name: defaultdict(float) for name in self.task_names
        }

        step_count = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets_dict in self.val_loader:
                inputs = inputs.to(device).contiguous()
                task_masks = getattr(self.val_loader.dataset, "task_feature_masks", {})

                # Move all task targets to device
                targets_dict = {
                    task_name: target.to(device)
                    for task_name, target in targets_dict.items()
                }

                # Compute loss for each task
                for task_name in self.task_names:
                    if task_name not in targets_dict:
                        continue

                    target = targets_dict[task_name]
                    mask = task_masks.get(task_name)
                    if mask is not None:
                        mask = mask.to(device)
                        task_inputs = inputs * mask
                    else:
                        task_inputs = inputs

                    logits_dict = self.model(task_inputs)
                    task_logits = logits_dict.get(task_name, logits_dict.get("main"))
                    if task_logits is None:
                        continue
                    loss_fn = self.task_loss_fns[task_name]

                    # Check for non-finite values
                    if not torch.isfinite(task_logits).all():
                        raise RuntimeError(f"Non-finite logits for task {task_name} during validation")

                    # Compute task loss
                    task_loss = loss_fn(task_logits, target)

                    if not torch.isfinite(task_loss):
                        raise RuntimeError(f"Non-finite validation loss for task {task_name}")

                    task_losses[task_name] += task_loss.item()

                    # Compute task-specific metrics
                    batch_task_metrics = self._compute_batch_metrics(task_logits, target)
                    for key, value in batch_task_metrics.items():
                        task_metrics[task_name][key] += value

        # Average metrics over all steps
        averaged_metrics = {}

        # Overall val loss (average of all task losses)
        overall_loss = sum(task_losses.values()) / (len(self.task_names) * max(step_count, 1))
        averaged_metrics["val_loss"] = overall_loss

        # Per-task metrics
        for task_name in self.task_names:
            task_loss = task_losses[task_name] / max(step_count, 1)
            averaged_metrics[f"val_loss_{task_name}"] = task_loss

            # Add other task-specific metrics
            for metric_name, total_value in task_metrics[task_name].items():
                avg_value = total_value / max(step_count, 1)
                averaged_metrics[f"val_{metric_name}_{task_name}"] = avg_value

        return averaged_metrics

    def train(self) -> Dict[str, float]:
        """Run multi-task training with health monitoring and adaptive weights."""
        for epoch in range(1, self.config.epochs + 1):
            # Run training and validation
            train_metrics = self._run_epoch(epoch)
            val_metrics = self._run_validation() if self.val_loader else {}
            metrics = {**train_metrics, **val_metrics}

            # Update task monitor with per-task metrics
            if self.task_monitor and epoch % self.mtl_config.health_check_interval == 0:
                self._update_task_health(epoch, train_metrics, val_metrics)

            # Update task weights for adaptive/gradnorm strategies
            if epoch % self.mtl_config.update_task_weights_interval == 0:
                self._update_task_weights(epoch, train_metrics, val_metrics)

            # Print epoch summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics)

            # Model checkpoint and early stopping
            self.model.update_performance(metrics)
            self.checkpoints.save(self.model, self.optimizer, epoch, metrics)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics.get("val_loss", train_metrics.get("train_loss", 0)))
                else:
                    self.scheduler.step()

            if self.val_loader and self.early_stopping.step(metrics.get("val_loss", train_metrics.get("train_loss", 0))):
                print(f"  ⚠ Early stopping triggered after {epoch} epochs")
                break

        # Print final task health summary
        if self.task_monitor:
            self._print_task_health_summary()

        return self.model.performance

    def _update_task_health(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ) -> None:
        """Update task health monitor and apply weight modifications."""
        for task_name in self.task_names:
            # Gather metrics for this task
            task_metrics = {
                "loss": train_metrics.get(f"train_loss_{task_name}", float("inf")),
                "train_loss": train_metrics.get(f"train_loss_{task_name}", float("inf")),
                "val_loss": val_metrics.get(f"val_loss_{task_name}", float("inf")),
            }

            # Add accuracy if available
            if f"train_accuracy_{task_name}" in train_metrics:
                task_metrics["accuracy"] = train_metrics[f"train_accuracy_{task_name}"]

            # Check task health
            health_status = self.task_monitor.check_task_health(task_name, task_metrics, epoch)

            # Log issues if any
            if not health_status["is_healthy"]:
                logger.bind(source="multitask_trainer").warning(
                    f"Task {task_name} unhealthy at epoch {epoch}: {health_status['status']} - {health_status['issues']}"
                )

        # Apply weight modifications based on task health
        current_weights = self.task_weighting.get_task_weights()
        modified_weights = self.task_monitor.apply_weight_modifications(current_weights)

        # Update weights in the weighting strategy (for static/adaptive strategies)
        if hasattr(self.task_weighting, "weights"):
            with torch.no_grad():
                for task_name in self.task_names:
                    idx = self.task_weighting.task_to_idx[task_name]
                    self.task_weighting.weights[idx] = modified_weights[task_name]

    def _compute_gradient_norms(self, individual_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute gradient norm for each task to diagnose multi-task interference.

        This method computes the L2 norm of gradients from each task individually,
        allowing us to identify which tasks dominate the gradient signal and which
        tasks have vanishing gradients.

        Args:
            individual_losses: Dict mapping task_name -> task loss tensor

        Returns:
            Dict mapping task_name -> gradient norm (float)
        """
        grad_norms = {}

        # Get backbone parameters (shared across all tasks)
        if hasattr(self.model, 'backbone'):
            backbone_params = list(self.model.backbone.parameters())
        elif hasattr(self.model, 'encoder'):
            backbone_params = list(self.model.encoder.parameters())
        else:
            # Fallback: use all model parameters
            backbone_params = list(self.model.parameters())

        for task_name, task_loss in individual_losses.items():
            # Zero existing gradients
            self.optimizer.zero_grad()

            # Compute gradients for this task only
            task_loss.backward(retain_graph=True)

            # Measure gradient norm on backbone parameters
            total_norm = 0.0
            for p in backbone_params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            grad_norms[task_name] = total_norm

        # Clear gradients after measurement (will recompute with combined loss)
        self.optimizer.zero_grad()

        return grad_norms

    def _apply_gradient_normalization(self, individual_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply per-task gradient normalization to balance gradient magnitudes.

        This method computes gradients for each task separately, normalizes their magnitude,
        and then combines them. This prevents tasks with large gradients from dominating
        the training signal.

        Args:
            individual_losses: Dict mapping task_name -> task loss tensor

        Returns:
            Combined normalized loss tensor
        """
        # Get backbone parameters (shared across all tasks)
        if hasattr(self.model, 'backbone'):
            backbone_params = list(self.model.backbone.parameters())
        elif hasattr(self.model, 'encoder'):
            backbone_params = list(self.model.encoder.parameters())
        else:
            backbone_params = list(self.model.parameters())

        # Accumulate normalized gradients
        self.optimizer.zero_grad()
        task_grad_norms = {}

        # First pass: compute gradient norms for each task
        for task_name, task_loss in individual_losses.items():
            # Get task weight
            task_weight = self.task_weighting.get_task_weights().get(task_name, 1.0)
            weighted_loss = task_weight * task_loss

            # Compute gradients for this task
            grads = torch.autograd.grad(
                weighted_loss,
                backbone_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )

            # Compute L2 norm
            total_norm = 0.0
            for g in grads:
                if g is not None:
                    total_norm += (g ** 2).sum().item()
            total_norm = total_norm ** 0.5
            task_grad_norms[task_name] = max(total_norm, 1e-8)  # Avoid division by zero

        # Compute mean gradient norm across all tasks
        mean_norm = max(sum(task_grad_norms.values()) / len(task_grad_norms), 1e-8)
        target_norm = self.mtl_config.gradient_norm_target

        # Second pass: apply normalized gradients
        for task_name, task_loss in individual_losses.items():
            # Get task weight
            task_weight = self.task_weighting.get_task_weights().get(task_name, 1.0)
            weighted_loss = task_weight * task_loss

            # Compute scaling factor to normalize gradient magnitude
            # Scale factor = (target_norm * mean_norm) / task_norm
            denom = max(task_grad_norms[task_name], 1e-8)
            scale = (target_norm * mean_norm) / denom
            # Safety clamp to avoid exploding/vanishing gradients
            scale = torch.tensor(scale, device=task_loss.device)
            scale = torch.clamp(scale, min=1e-3, max=10.0)

            # Apply scaled backward pass
            scaled_loss = scale * weighted_loss
            scaled_loss.backward(retain_graph=(task_name != list(individual_losses.keys())[-1]))

        # Return dummy loss for logging (actual gradients already applied)
        return sum(individual_losses.values()) / len(individual_losses)

    def _update_task_weights(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ) -> None:
        """Update task weights for adaptive strategies."""
        from training.task_weighting import AdaptiveWeighting, GradNormWeighting

        # Adaptive weighting: update based on validation performance
        if isinstance(self.task_weighting, AdaptiveWeighting):
            task_val_metrics = {}
            for task_name in self.task_names:
                task_val_metrics[task_name] = {
                    "val_loss": val_metrics.get(f"val_loss_{task_name}", float("inf")),
                }
                if f"val_accuracy_{task_name}" in val_metrics:
                    task_val_metrics[task_name]["val_accuracy"] = val_metrics[f"val_accuracy_{task_name}"]

            self.task_weighting.update_weights(task_val_metrics)

        # GradNorm: update based on gradient magnitudes
        # Note: This requires a separate backward pass, so we skip it in the epoch loop
        # and only update based on the last batch's gradients
        # For full GradNorm implementation, you'd need to modify _run_epoch to call update_weights

    def _print_epoch_summary(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ) -> None:
        """Print a summary of the epoch with per-task metrics."""
        train_loss = train_metrics.get("train_loss", 0.0)
        val_loss = val_metrics.get("val_loss", 0.0)

        print(f"\n  Epoch {epoch}/{self.config.epochs} - Overall Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Print per-task metrics
        print("  Per-task metrics:")
        for task_name in self.task_names:
            task_train_loss = train_metrics.get(f"train_loss_{task_name}", 0.0)
            task_val_loss = val_metrics.get(f"val_loss_{task_name}", 0.0)
            task_weight = train_metrics.get(f"task_weight_{task_name}", 1.0)

            # Check if task is healthy
            health_status = "✓" if (not self.task_monitor or self.task_monitor.is_healthy[task_name]) else "✗"

            print(f"    {health_status} {task_name:30s} - Train: {task_train_loss:.4f}, Val: {task_val_loss:.4f}, Weight: {task_weight:.3f}")

    def _print_task_health_summary(self) -> None:
        """Print final task health summary."""
        summary = self.task_monitor.get_summary()

        print(f"\n{'='*80}")
        print(f"Task Health Summary:")
        print(f"  Healthy tasks: {summary['num_healthy']}/{len(self.task_names)}")
        print(f"  Unhealthy tasks: {summary['num_unhealthy']}/{len(self.task_names)}")
        print(f"{'='*80}")

        for task_name, task_info in summary["tasks"].items():
            status_icon = "✓" if task_info["is_healthy"] else "✗"
            print(f"  {status_icon} {task_name:30s} - Weight mod: {task_info['weight_modification']:.3f}, Epochs tracked: {task_info['num_epochs_tracked']}")

            if task_info["issues"]:
                for issue in task_info["issues"][-3:]:  # Show last 3 issues
                    print(f"      Issue: {issue}")

        print(f"{'='*80}\n")


__all__ = ["MultiTaskTrainingLoop", "MultiTaskConfig"]
