"""Model loading helpers for inference/serving."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from models.temporal_fusion import TemporalFusionTransformer
from models.multitask import MultiTaskModel
from training.phases.phase1_direction import Phase1DirectionTask
from training.phases.phase2_indicators import Phase2IndicatorTask
from training.phases.phase3_structure import Phase3StructureTask
from training.phases.phase4_smart_money import Phase4SmartMoneyTask
from training.phases.phase5_candlesticks import Phase5CandlestickTask
from training.phases.phase6_sr_levels import Phase6SupportResistanceTask
from training.phases.phase7_advanced_sm import Phase7AdvancedSMTask
from training.phases.phase8_risk import Phase8RiskTask
from training.phases.phase9_integration import Phase9IntegrationTask

TASK_REGISTRY = {
    "Phase1DirectionTask": Phase1DirectionTask,
    "Phase2IndicatorTask": Phase2IndicatorTask,
    "Phase3StructureTask": Phase3StructureTask,
    "Phase4SmartMoneyTask": Phase4SmartMoneyTask,
    "Phase5CandlestickTask": Phase5CandlestickTask,
    "Phase6SupportResistanceTask": Phase6SupportResistanceTask,
    "Phase7AdvancedSMTask": Phase7AdvancedSMTask,
    "Phase8RiskTask": Phase8RiskTask,
    "Phase9IntegrationTask": Phase9IntegrationTask,
}


@dataclass
class ModelBundle:
    model: MultiTaskModel
    feature_columns: List[str]
    task_names: Sequence[str]
    device: torch.device


def build_all_tasks() -> List:
    return [cls() for cls in TASK_REGISTRY.values()]


def _task_output_dims(tasks: Sequence) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for task in tasks:
        name = task.__class__.__name__
        dim = getattr(task, "output_dim", 1)
        dims[name] = dim
    return dims


def _build_tasks_from_checkpoint(ckpt: Dict) -> List:
    task_order = ckpt.get("task_order")
    if not task_order:
        return build_all_tasks()
    tasks: List = []
    for name in task_order:
        cls = TASK_REGISTRY.get(name)
        if cls is None:
            continue
        tasks.append(cls())
    return tasks or build_all_tasks()


def _infer_aux_dims(tasks: Sequence, state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    defaults = _task_output_dims(tasks)
    for task in tasks:
        name = task.__class__.__name__
        key = f"aux_heads.{name}.2.weight"
        if key in state_dict:
            dims[name] = state_dict[key].shape[0]
        else:
            dims[name] = defaults.get(name, 1)
    return dims


def load_multitask_model(checkpoint_path: Path) -> ModelBundle:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    feature_columns: List[str] = ckpt.get("feature_columns")
    if not feature_columns:
        raise ValueError("Checkpoint missing 'feature_columns'")

    tasks = _build_tasks_from_checkpoint(ckpt)
    aux_heads = _infer_aux_dims(tasks, ckpt.get("model_state", {}))

    backbone = TemporalFusionTransformer(
        input_dim=len(feature_columns),
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.3,
        use_fsatten=True,
        output_features=True,
    )
    model = MultiTaskModel(backbone, aux_heads=aux_heads, dropout=0.5)
    model.load_state_dict(ckpt["model_state"])

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    model.eval()

    return ModelBundle(model=model, feature_columns=feature_columns, task_names=tuple(aux_heads.keys()), device=device)
