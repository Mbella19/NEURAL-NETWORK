"""Curriculum learning manager (Phase 5.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class CurriculumPhase:
    name: str
    module: object
    graduation_metric: str
    threshold: float
    max_epochs: int = 10
    status: str = "pending"
    history: List[Dict[str, float]] = field(default_factory=list)


class CurriculumManager:
    """Handles phase progression, evaluation, and catastrophic forgetting checks."""

    def __init__(self, phases: List[CurriculumPhase]) -> None:
        self.phases = phases
        self.current_index = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self.current_index]

    def record_metrics(self, metrics: Dict[str, float]) -> None:
        phase = self.current_phase
        phase.history.append(metrics)
        if len(phase.history) > phase.max_epochs:
            phase.history.pop(0)

    def should_graduate(self) -> bool:
        phase = self.current_phase
        if not phase.history:
            return False
        recent = phase.history[-1][phase.graduation_metric]
        return recent >= phase.threshold

    def advance_phase(self) -> None:
        phase = self.current_phase
        phase.status = "completed"
        if self.current_index < len(self.phases) - 1:
            self.current_index += 1
            self.current_phase.status = "in_progress"

    def run_phase(self, trainer_fn) -> None:
        phase = self.current_phase
        phase.status = "in_progress"
        while not self.should_graduate():
            metrics = trainer_fn(phase)
            self.record_metrics(metrics)
            if len(phase.history) >= phase.max_epochs:
                break
        self.advance_phase()

    def knowledge_retention_check(self, evaluate_fn) -> Dict[str, float]:
        results = {}
        for completed_phase in self.phases[: self.current_index]:
            results[completed_phase.name] = evaluate_fn(completed_phase)
        return results


__all__ = ["CurriculumManager", "CurriculumPhase"]
