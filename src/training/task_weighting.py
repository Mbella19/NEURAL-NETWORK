"""Task weighting strategies for multi-task learning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn


class TaskWeighting(ABC, nn.Module):
    """Base class for task weighting strategies."""

    def __init__(self, num_tasks: int, task_names: List[str]):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names
        self.task_to_idx = {name: idx for idx, name in enumerate(task_names)}

    @abstractmethod
    def forward(self, task_losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        """Compute weighted combination of task losses.

        Args:
            task_losses: Dict mapping task_name -> loss tensor
            epoch: Current training epoch

        Returns:
            Combined weighted loss
        """
        pass

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights as a dictionary.

        Returns:
            Dict mapping task_name -> weight value
        """
        pass


class StaticWeighting(TaskWeighting):
    """Fixed task weights throughout training."""

    def __init__(self, num_tasks: int, task_names: List[str], base_weights: Dict[str, float]):
        super().__init__(num_tasks, task_names)
        self.base_weights = base_weights

        # Convert to tensor for efficient computation
        weight_list = [base_weights.get(name, 1.0) for name in task_names]
        self.register_buffer("weights", torch.tensor(weight_list, dtype=torch.float32))

    def forward(self, task_losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total_loss = 0.0
        for task_name, loss in task_losses.items():
            idx = self.task_to_idx[task_name]
            total_loss = total_loss + self.weights[idx] * loss
        return total_loss / self.num_tasks

    def get_task_weights(self) -> Dict[str, float]:
        return {name: float(self.weights[idx]) for name, idx in self.task_to_idx.items()}


class UncertaintyWeighting(TaskWeighting):
    """Learn task weights based on homoscedastic uncertainty.

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., 2018)

    The model learns log(σ²) for each task, where σ² represents task-dependent uncertainty.
    Tasks with higher uncertainty get automatically lower effective weight.
    """

    def __init__(self, num_tasks: int, task_names: List[str], base_weights: Dict[str, float]):
        super().__init__(num_tasks, task_names)
        self.base_weights = base_weights

        # Learnable log variance for each task (initialized to log(1.0) = 0.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        # Store base weights as buffers
        weight_list = [base_weights.get(name, 1.0) for name in task_names]
        self.register_buffer("base_weight_tensor", torch.tensor(weight_list, dtype=torch.float32))

    def forward(self, task_losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        """Compute uncertainty-weighted loss.

        Loss = Σ (1 / (2σ²)) * L_i + log(σ²) + base_weight_i

        The log(σ²) term prevents the model from setting uncertainty to infinity.
        """
        total_loss = 0.0
        for task_name, loss in task_losses.items():
            idx = self.task_to_idx[task_name]

            # Precision = 1 / σ²  = exp(-log(σ²))
            precision = torch.exp(-self.log_vars[idx])

            # Weighted loss with uncertainty + regularization
            weighted_loss = (
                self.base_weight_tensor[idx] *  # Base importance weight
                (precision * loss + self.log_vars[idx])  # Uncertainty weighting
            )
            total_loss = total_loss + weighted_loss

        return total_loss / self.num_tasks

    def get_task_weights(self) -> Dict[str, float]:
        """Get effective task weights (including learned uncertainty)."""
        weights = {}
        for name, idx in self.task_to_idx.items():
            precision = torch.exp(-self.log_vars[idx]).item()
            base_weight = self.base_weight_tensor[idx].item()
            effective_weight = base_weight * precision
            weights[name] = effective_weight
        return weights

    def get_uncertainties(self) -> Dict[str, float]:
        """Get learned uncertainty (σ²) for each task."""
        uncertainties = {}
        for name, idx in self.task_to_idx.items():
            sigma_squared = torch.exp(self.log_vars[idx]).item()
            uncertainties[name] = sigma_squared
        return uncertainties


class GradNormWeighting(TaskWeighting):
    """Dynamically balance gradients across tasks using GradNorm.

    Reference: "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    (Chen et al., 2018)

    Balances gradient magnitudes across tasks to ensure all tasks contribute equally to learning.
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: List[str],
        base_weights: Dict[str, float],
        alpha: float = 1.5,
        lr: float = 0.025,
    ):
        super().__init__(num_tasks, task_names)
        self.alpha = alpha  # Controls target gradient norm adjustment based on loss ratios
        self.lr = lr  # Learning rate for weight updates

        # Learnable task weights (initialized to base weights)
        weight_list = [base_weights.get(name, 1.0) for name in task_names]
        self.weights = nn.Parameter(torch.tensor(weight_list, dtype=torch.float32))

        # Track initial task losses for relative progress
        self.register_buffer("initial_losses", None)

    def forward(self, task_losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        """Compute weighted loss (weights will be updated separately via update_weights)."""
        total_loss = 0.0
        for task_name, loss in task_losses.items():
            idx = self.task_to_idx[task_name]
            total_loss = total_loss + self.weights[idx] * loss
        return total_loss / self.num_tasks

    def update_weights(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_params: List[torch.nn.Parameter],
    ) -> Dict[str, float]:
        """Update task weights based on gradient magnitudes.

        Args:
            task_losses: Dict of task losses
            shared_params: List of shared backbone parameters

        Returns:
            Dict of gradient norms for logging
        """
        # Store initial losses on first call
        loss_tensor = torch.stack([task_losses[name] for name in self.task_names])
        if self.initial_losses is None:
            self.initial_losses = loss_tensor.detach().clone()

        # Compute gradient norms for each task
        grad_norms = []
        for task_name in self.task_names:
            idx = self.task_to_idx[task_name]
            task_loss = self.weights[idx] * task_losses[task_name]

            # Compute gradients w.r.t. shared parameters
            grads = torch.autograd.grad(
                task_loss,
                shared_params,
                retain_graph=True,
                create_graph=True,
            )

            # L2 norm of concatenated gradients
            grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
            grad_norms.append(grad_norm)

        grad_norm_tensor = torch.stack(grad_norms)

        # Compute relative inverse training rates
        loss_ratios = loss_tensor / (self.initial_losses + 1e-8)
        inverse_train_rates = loss_ratios / (loss_ratios.mean() + 1e-8)

        # Target gradient norms: G_avg * (r_i ^ alpha)
        mean_grad_norm = grad_norm_tensor.mean().detach()
        target_grad_norms = mean_grad_norm * (inverse_train_rates.detach() ** self.alpha)

        # GradNorm loss: minimize difference between actual and target gradient norms
        gradnorm_loss = torch.sum(torch.abs(grad_norm_tensor - target_grad_norms))

        # Update weights using gradient descent
        weight_grads = torch.autograd.grad(gradnorm_loss, self.weights)[0]
        with torch.no_grad():
            self.weights -= self.lr * weight_grads
            # Keep weights positive
            self.weights.clamp_(min=0.01)

            # Renormalize to maintain scale
            self.weights /= self.weights.sum() / self.num_tasks

        return {
            name: float(grad_norms[idx])
            for name, idx in self.task_to_idx.items()
        }

    def get_task_weights(self) -> Dict[str, float]:
        return {name: float(self.weights[idx]) for name, idx in self.task_to_idx.items()}


class AdaptiveWeighting(TaskWeighting):
    """Adapt task weights based on validation performance.

    Increase weight for tasks that are performing poorly,
    decrease weight for tasks that have converged.
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: List[str],
        base_weights: Dict[str, float],
        adapt_rate: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
    ):
        super().__init__(num_tasks, task_names)
        self.adapt_rate = adapt_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Initialize weights
        weight_list = [base_weights.get(name, 1.0) for name in task_names]
        self.register_buffer("weights", torch.tensor(weight_list, dtype=torch.float32))

        # Track performance history
        self.performance_history = {name: [] for name in task_names}

    def forward(self, task_losses: Dict[str, torch.Tensor], epoch: int = 0) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total_loss = 0.0
        for task_name, loss in task_losses.items():
            idx = self.task_to_idx[task_name]
            total_loss = total_loss + self.weights[idx] * loss
        return total_loss / self.num_tasks

    def update_weights(self, task_metrics: Dict[str, Dict[str, float]]):
        """Update weights based on task performance.

        Args:
            task_metrics: Dict mapping task_name -> {"val_loss": ..., "val_accuracy": ..., etc.}
        """
        with torch.no_grad():
            for task_name, metrics in task_metrics.items():
                if "val_loss" not in metrics:
                    continue

                idx = self.task_to_idx[task_name]
                self.performance_history[task_name].append(metrics["val_loss"])

                # Only adapt after seeing some history
                if len(self.performance_history[task_name]) < 3:
                    continue

                # Check if task is improving
                recent_losses = self.performance_history[task_name][-3:]
                is_improving = recent_losses[-1] < recent_losses[0]

                if is_improving:
                    # Decrease weight slightly (task is learning well)
                    self.weights[idx] *= (1.0 - self.adapt_rate * 0.5)
                else:
                    # Increase weight (task needs more attention)
                    self.weights[idx] *= (1.0 + self.adapt_rate)

                # Clamp to bounds
                self.weights[idx] = torch.clamp(
                    self.weights[idx],
                    self.min_weight,
                    self.max_weight
                )

    def get_task_weights(self) -> Dict[str, float]:
        return {name: float(self.weights[idx]) for name, idx in self.task_to_idx.items()}


__all__ = [
    "TaskWeighting",
    "StaticWeighting",
    "UncertaintyWeighting",
    "GradNormWeighting",
    "AdaptiveWeighting",
]
