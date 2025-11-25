"""Training package exports."""

from .trainer import TrainerConfig, TrainingLoop
from .multitask_trainer import MultiTaskConfig, MultiTaskTrainingLoop
from .multitask_dataset import MultiTaskDataset
from .loss_wrappers import FocalLoss, make_weighted_bce, configure_binary_loss
from .checkpointing import CheckpointConfig, CheckpointSystem
from .task_weighting import TaskWeighting, StaticWeighting, UncertaintyWeighting
from .task_monitor import TaskMonitor, TaskHealthConfig
from .walk_forward import WalkForwardConfig, walk_forward_validation
from .cpcv import CPCVConfig, cpcv_validation
from .ewc import EWC, OnlineEWC

__all__ = [
    # Core training
    "TrainerConfig",
    "TrainingLoop",
    "MultiTaskConfig",
    "MultiTaskTrainingLoop",
    "MultiTaskDataset",
    # Loss functions
    "FocalLoss",
    "make_weighted_bce",
    "configure_binary_loss",
    # Checkpointing
    "CheckpointConfig",
    "CheckpointSystem",
    # Task management
    "TaskWeighting",
    "StaticWeighting",
    "UncertaintyWeighting",
    "TaskMonitor",
    "TaskHealthConfig",
    # Validation
    "WalkForwardConfig",
    "walk_forward_validation",
    "CPCVConfig",
    "cpcv_validation",
    # Continual learning
    "EWC",
    "OnlineEWC",
]
