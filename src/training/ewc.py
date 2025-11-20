"""Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.

Reference: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
https://arxiv.org/abs/1612.00796

EWC prevents catastrophic forgetting by:
1. Computing Fisher Information Matrix (FIM) after training on a task
2. Adding a penalty when fine-tuning to discourage changing important parameters
3. Penalty: λ/2 * Σ F_i * (θ_i - θ*_i)^2
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger


class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting.

    Usage:
        # 1. Train model on Task A
        model = MyModel()
        optimizer = torch.optim.Adam(model.parameters())
        train_on_task_a(model, optimizer, task_a_data)

        # 2. Compute Fisher Information and save optimal parameters
        ewc = EWC(model, ewc_lambda=1000.0)
        ewc.compute_fisher(task_a_data_loader, loss_fn)

        # 3. Fine-tune on Task B with EWC penalty
        for batch in task_b_data_loader:
            loss = compute_task_b_loss(model, batch)
            ewc_loss = ewc.penalty(model)  # Add EWC penalty
            total_loss = loss + ewc_loss
            total_loss.backward()
            optimizer.step()
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        """Initialize EWC.

        Args:
            model: The neural network model
            ewc_lambda: Regularization strength for EWC penalty (higher = stronger retention)
        """
        self.model = model
        self.ewc_lambda = ewc_lambda

        # Store Fisher Information Matrix (diagonal approximation)
        self.fisher: Dict[str, torch.Tensor] = {}

        # Store optimal parameters from previous task
        self.optimal_params: Dict[str, torch.Tensor] = {}

        # Track if Fisher has been computed
        self.fisher_computed = False

    def compute_fisher(
        self,
        data_loader: DataLoader,
        loss_fn: callable,
        num_samples: Optional[int] = None,
    ) -> None:
        """Compute Fisher Information Matrix on the given dataset.

        The Fisher Information measures how much each parameter affects the loss.
        We use a diagonal approximation: F_i ≈ E[(∂L/∂θ_i)²]

        Args:
            data_loader: DataLoader with training data from the task
            loss_fn: Loss function (should take model output and target)
            num_samples: Maximum number of samples to use (None = use all)
        """
        logger.bind(source="ewc").info("Computing Fisher Information Matrix...")

        # Initialize Fisher dict with zeros
        self.fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)

        # Set model to eval mode (no dropout, etc.)
        self.model.eval()

        num_processed = 0
        total_samples = num_samples if num_samples else len(data_loader.dataset)

        for batch in data_loader:
            # Unpack batch
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                raise ValueError("Expected batch to be (inputs, targets) tuple")

            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)

            # Handle multi-task targets (dict) or single-task targets (tensor)
            if isinstance(targets, dict):
                # For multi-task, we'll compute Fisher on all tasks
                # You could also compute separate Fisher for each task
                targets = {k: v.to(device) for k, v in targets.items()}
            else:
                targets = targets.to(device)

            # Zero gradients
            self.model.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            if isinstance(targets, dict):
                # Multi-task: sum losses from all tasks
                loss = 0.0
                for task_name, task_target in targets.items():
                    if isinstance(outputs, dict) and task_name in outputs:
                        task_output = outputs[task_name]
                    else:
                        task_output = outputs  # Single output for all tasks
                    task_loss = loss_fn(task_output, task_target)
                    loss = loss + task_loss
            else:
                # Single-task
                loss = loss_fn(outputs, targets)

            # Backward pass to compute gradients
            loss.backward()

            # Accumulate squared gradients (Fisher Information)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2

            num_processed += inputs.size(0)

            if num_samples and num_processed >= num_samples:
                break

        # Average Fisher over samples
        for name in self.fisher:
            self.fisher[name] /= num_processed

        # Save optimal parameters
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

        self.fisher_computed = True

        # Log statistics
        total_fisher = sum(f.sum().item() for f in self.fisher.values())
        logger.bind(source="ewc").info(
            f"Fisher Information computed on {num_processed} samples. "
            f"Total Fisher: {total_fisher:.6f}"
        )

        # Log top-5 most important parameters
        param_importance = {
            name: fisher.sum().item() for name, fisher in self.fisher.items()
        }
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        logger.bind(source="ewc").info("Top-5 most important parameters:")
        for name, importance in sorted_params[:5]:
            logger.bind(source="ewc").info(f"  {name}: {importance:.6f}")

    def penalty(self, model: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute EWC penalty for current model parameters.

        Penalty = λ/2 * Σ F_i * (θ_i - θ*_i)^2

        Args:
            model: Model to compute penalty for (uses self.model if None)

        Returns:
            EWC penalty term (scalar tensor)
        """
        if not self.fisher_computed:
            logger.bind(source="ewc").warning(
                "Fisher Information not computed yet. Call compute_fisher() first. Returning 0 penalty."
            )
            return torch.tensor(0.0)

        if model is None:
            model = self.model

        penalty = 0.0

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                # Penalty = Fisher * (current_param - optimal_param)^2
                fisher = self.fisher[name]
                optimal = self.optimal_params[name]
                penalty += (fisher * (param - optimal) ** 2).sum()

        # Scale by lambda and 1/2
        penalty = (self.ewc_lambda / 2.0) * penalty

        return penalty

    def consolidate_fisher(self, new_fisher: Dict[str, torch.Tensor], gamma: float = 0.9) -> None:
        """Consolidate new Fisher information with existing Fisher.

        When training on multiple tasks sequentially, you can accumulate Fisher
        information from all previous tasks:
        F_consolidated = γ * F_old + F_new

        Args:
            new_fisher: New Fisher information from latest task
            gamma: Decay factor for old Fisher (0 = forget old, 1 = keep all)
        """
        if not self.fisher_computed:
            # First task - just use new Fisher
            self.fisher = new_fisher
            self.fisher_computed = True
            return

        # Consolidate with existing Fisher
        for name in self.fisher:
            if name in new_fisher:
                self.fisher[name] = gamma * self.fisher[name] + new_fisher[name]

        logger.bind(source="ewc").info(f"Consolidated Fisher with gamma={gamma}")

    def save(self, filepath: str) -> None:
        """Save EWC state to disk.

        Args:
            filepath: Path to save EWC state
        """
        state = {
            "fisher": self.fisher,
            "optimal_params": self.optimal_params,
            "ewc_lambda": self.ewc_lambda,
            "fisher_computed": self.fisher_computed,
        }
        torch.save(state, filepath)
        logger.bind(source="ewc").info(f"Saved EWC state to {filepath}")

    def load(self, filepath: str) -> None:
        """Load EWC state from disk.

        Args:
            filepath: Path to load EWC state from
        """
        state = torch.load(filepath)
        self.fisher = state["fisher"]
        self.optimal_params = state["optimal_params"]
        self.ewc_lambda = state["ewc_lambda"]
        self.fisher_computed = state["fisher_computed"]
        logger.bind(source="ewc").info(f"Loaded EWC state from {filepath}")

    def get_parameter_importance(self) -> Dict[str, float]:
        """Get importance score for each parameter.

        Returns:
            Dict mapping parameter name -> importance score (sum of Fisher)
        """
        if not self.fisher_computed:
            return {}

        return {name: fisher.sum().item() for name, fisher in self.fisher.items()}


class OnlineEWC(EWC):
    """Online EWC for continual learning across multiple tasks.

    Unlike standard EWC which stores Fisher for each task separately,
    Online EWC maintains a running Fisher estimate that's updated after each task.

    Reference: "Continual Learning Through Synaptic Intelligence" (Zenke et al., 2017)
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0, gamma: float = 0.9):
        """Initialize Online EWC.

        Args:
            model: The neural network model
            ewc_lambda: Regularization strength
            gamma: Decay factor for consolidating Fisher across tasks
        """
        super().__init__(model, ewc_lambda)
        self.gamma = gamma

    def update_fisher_and_params(
        self,
        data_loader: DataLoader,
        loss_fn: callable,
        num_samples: Optional[int] = None,
    ) -> None:
        """Update Fisher information and optimal parameters after training on a task.

        This consolidates the new Fisher with existing Fisher and updates optimal parameters.

        Args:
            data_loader: DataLoader with training data from the latest task
            loss_fn: Loss function
            num_samples: Maximum number of samples to use
        """
        # Compute Fisher for new task
        new_ewc = EWC(self.model, self.ewc_lambda)
        new_ewc.compute_fisher(data_loader, loss_fn, num_samples)

        if not self.fisher_computed:
            # First task
            self.fisher = new_ewc.fisher
            self.optimal_params = new_ewc.optimal_params
            self.fisher_computed = True
        else:
            # Consolidate with previous Fisher
            for name in self.fisher:
                if name in new_ewc.fisher:
                    self.fisher[name] = self.gamma * self.fisher[name] + new_ewc.fisher[name]

            # Update optimal parameters to current values
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.optimal_params[name] = param.data.clone()

        logger.bind(source="online_ewc").info(
            f"Updated Online EWC (gamma={self.gamma})"
        )


__all__ = ["EWC", "OnlineEWC"]
