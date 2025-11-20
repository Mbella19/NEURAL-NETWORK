"""Training loop infrastructure (Phase 6.1)."""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from models.base_model import BaseModel
from models.utils import CheckpointManager, EarlyStopping, clip_gradients, get_scheduler
from utils.logger import ResourceMonitor, log_resource_snapshot
try:
    import torch._dynamo  # type: ignore
except Exception:  # pragma: no cover
    torch_dynamo_available = False
else:
    torch_dynamo_available = True


@dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    scheduler: Optional[str] = None
    checkpoint_dir: Path = Path("checkpoints")
    resource_log_interval: int = 200
    show_progress: bool = True
    adversarial_eps: float = 0.0
    device: Optional[str] = None
    pos_weight: Optional[torch.Tensor] = None
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    # Early stopping configuration
    early_stopping_patience: int = 5
    early_stopping_min_epochs: int = 10


class TrainingLoop:
    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainerConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = self._detect_device()
        self.model.to(self.device)
        pin_memory = self.device.type == "cuda"
        if self.config.pin_memory is not None:
            pin_memory = self.config.pin_memory
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,  # Enable shuffling to decorrelate batches and improve convergence
            pin_memory=pin_memory,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
        )
        self.val_loader = (
            DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                pin_memory=pin_memory,
                shuffle=False,
                num_workers=config.num_workers,
                persistent_workers=config.num_workers > 0,
            )
            if val_dataset
            else None
        )
        self.amp_device_type: Optional[str] = None
        if config.mixed_precision and self.device.type in {"cuda", "mps"}:
            self.amp_device_type = self.device.type
        self.use_amp = self.amp_device_type is not None
        if self.amp_device_type == "cuda":
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except AttributeError:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None
        self.checkpoints = CheckpointManager(config.checkpoint_dir)
        self.scheduler = (
            get_scheduler(optimizer, config.scheduler, **{"T_max": config.epochs}) if config.scheduler else None
        )
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_epochs=config.early_stopping_min_epochs,
            mode="min"
        )
        self.resource_monitor = ResourceMonitor()
        self.pos_weight = config.pos_weight

    def train(self) -> Dict[str, float]:
        for epoch in range(1, self.config.epochs + 1):
            log_resource_snapshot(f"train_epoch_{epoch}_start", self.resource_monitor)
            train_metrics = self._run_epoch(epoch)
            val_metrics = self._run_validation() if self.val_loader else {}
            metrics = {**train_metrics, **val_metrics}
            train_loss = train_metrics.get("train_loss", 0.0)
            val_loss = val_metrics.get("val_loss", 0.0)
            extra_bits = []
            if "train_accuracy" in train_metrics:
                extra_bits.append(f"Train Acc: {train_metrics['train_accuracy']:.4f}")
            if "val_accuracy" in val_metrics:
                extra_bits.append(f"Val Acc: {val_metrics['val_accuracy']:.4f}")
            if "train_mae" in train_metrics:
                extra_bits.append(f"Train MAE: {train_metrics['train_mae']:.4f}")
            if "val_mae" in val_metrics:
                extra_bits.append(f"Val MAE: {val_metrics['val_mae']:.4f}")
            extras = " | ".join(extra_bits)
            print(
                f"  Epoch {epoch}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                + (f" | {extras}" if extras else "")
            )
            self.model.update_performance(metrics)
            self.checkpoints.save(self.model, self.optimizer, epoch, metrics)
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics.get("val_loss", train_metrics.get("loss", 0)))
            if self.val_loader and self.early_stopping.step(metrics.get("val_loss", train_metrics.get("loss", 0))):
                print(f"  ⚠ Early stopping triggered after {epoch} epochs (no improvement for {self.early_stopping.patience} epochs)")
                break
        return self.model.performance

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        device = self._device()

        metrics_totals: Dict[str, float] = defaultdict(float)
        step_count = len(self.train_loader)
        iterator = enumerate(self.train_loader, start=1)
        progress = None  # tqdm disabled to reduce overhead

        for step, batch in iterator:
            inputs, targets = batch
            inputs = inputs.to(device).contiguous()
            targets = targets.to(device)

            if not torch.isfinite(inputs).all():
                raise RuntimeError(f"Non-finite inputs detected at epoch {epoch}, step {step}")

            autocast_context = nullcontext()
            if self.amp_device_type == "cuda":
                autocast_context = torch.cuda.amp.autocast()
            elif self.amp_device_type == "mps":
                autocast_context = torch.autocast(device_type="mps", dtype=torch.float16)

            with autocast_context:
                inputs.requires_grad_(self.config.adversarial_eps > 0)
                logits = self.model(inputs)
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite logits detected at epoch {epoch}, step {step}")
                loss = self.loss_fn(logits, targets) / self.config.gradient_accumulation

                # Adversarial (FGSM) smoothing
                if self.config.adversarial_eps > 0:
                    loss.backward(retain_graph=True)
                    perturb = self.config.adversarial_eps * inputs.grad.data.sign()
                    inputs.grad = None
                    adv_inputs = torch.clamp(inputs + perturb, -8.0, 8.0)
                    adv_logits = self.model(adv_inputs)
                    adv_loss = self.loss_fn(adv_logits, targets) / self.config.gradient_accumulation
                    loss = (loss + adv_loss) * 0.5

            if not torch.isfinite(loss):
                logger.bind(source="training_loop").error(
                    "Non-finite loss detected at epoch=%d step=%d loss=%s "
                    "(logits_has_nan=%s targets_has_nan=%s)",
                    epoch,
                    step,
                    loss.detach().cpu().item(),
                    bool(torch.isnan(logits).any().item()),
                    bool(torch.isnan(targets).any().item()),
                )
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}, step {step}")

            if self.amp_device_type == "cuda" and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            batch_metrics = self._compute_batch_metrics(logits.detach(), targets.detach())
            for key, value in batch_metrics.items():
                metrics_totals[key] += value

            if step % self.config.gradient_accumulation == 0:
                if self.amp_device_type == "cuda" and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradients(self.model, self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    clip_gradients(self.model, self.config.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()
            if step % self.config.resource_log_interval == 0:
                log_resource_snapshot(f"train_epoch_{epoch}_step_{step}", self.resource_monitor)
            total_loss += loss.item()

        averaged_metrics = {f"train_{k}": v / max(step_count, 1) for k, v in metrics_totals.items()}
        averaged_metrics["train_loss"] = total_loss / max(step_count, 1)
        return averaged_metrics

    def _run_validation(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        metrics_totals: Dict[str, float] = defaultdict(float)
        step_count = len(self.val_loader) if self.val_loader else 0
        pos_pred = 0
        total_pred = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self._device()).contiguous()
                targets = targets.to(self._device())
                logits = self.model(inputs)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite logits encountered during validation")
                loss = self.loss_fn(logits, targets)
                if not torch.isfinite(loss):
                    logger.bind(source="training_loop").error(
                        "Non-finite validation loss detected (loss=%s)",
                        loss.detach().cpu().item(),
                    )
                    raise RuntimeError("Non-finite loss encountered during validation")
                total_loss += loss.item()
                batch_metrics = self._compute_batch_metrics(logits, targets)
                for key, value in batch_metrics.items():
                    metrics_totals[key] += value
                if (
                    logits.shape[-1] == 1
                    and targets.shape == logits.shape
                    and targets.dtype.is_floating_point
                    and targets.min().item() >= 0.0
                    and targets.max().item() <= 1.0
                ):
                    preds = torch.sigmoid(logits)
                    pos_pred += (preds > 0.5).sum().item()
                    total_pred += preds.numel()
        averaged_metrics = {f"val_{k}": v / max(step_count, 1) for k, v in metrics_totals.items()}
        averaged_metrics["val_loss"] = total_loss / max(step_count, 1)
        if total_pred:
            averaged_metrics["val_pos_rate"] = pos_pred / total_pred
        return averaged_metrics

    def _detect_device(self) -> torch.device:
        if self.config.device:
            return torch.device(self.config.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _device(self) -> torch.device:
        return self.device

    def _compute_batch_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if targets.dtype in (torch.long, torch.int64):
            preds = torch.argmax(logits, dim=-1)
            metrics["accuracy"] = (preds == targets).float().mean().item()
        elif torch.is_floating_point(targets):
            binary_targets = (
                bool(torch.isfinite(targets).all())
                and targets.min().item() >= 0.0
                and targets.max().item() <= 1.0
            )
            same_shape = logits.shape == targets.shape
            if same_shape and binary_targets:
                preds = torch.sigmoid(logits)
                diff = preds - targets
                metrics["mae"] = diff.abs().mean().item()
                metrics["mse"] = diff.pow(2).mean().item()
                metrics["accuracy"] = ((preds > 0.5) == (targets > 0.5)).float().mean().item()
                metrics["pos_rate"] = (preds > 0.5).float().mean().item()
                metrics["target_pos_rate"] = (targets > 0.5).float().mean().item()
            elif same_shape:
                diff = logits - targets
                metrics["mae"] = diff.abs().mean().item()
                metrics["mse"] = diff.pow(2).mean().item()
        return metrics


__all__ = ["TrainerConfig", "TrainingLoop"]
