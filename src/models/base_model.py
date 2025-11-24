"""Base model abstractions (Phase 4.1)."""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class ModelMetadata:
    model_name: str
    version: str = "0.1.0"
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    sequence_length: Optional[int] = None
    features: Optional[int] = None
    extra: Dict[str, Any] = None


class BaseModel(abc.ABC, torch.nn.Module):
    """Abstract base class for all trading models."""

    def __init__(self, metadata: Optional[ModelMetadata] = None) -> None:
        super().__init__()
        self.metadata = metadata or ModelMetadata(model_name=self.__class__.__name__)
        self._performance: Dict[str, float] = {}

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass producing logits or predictions."""

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(inputs, **kwargs)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "model.pt")
        (directory / "metadata.json").write_text(json.dumps(asdict(self.metadata), indent=2), encoding="utf-8")
        (directory / "performance.json").write_text(json.dumps(self._performance, indent=2), encoding="utf-8")

    def load(self, directory: Path, map_location: Optional[str] = None) -> None:
        state = torch.load(directory / "model.pt", map_location=map_location or "cpu")
        self.load_state_dict(state)
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.metadata = ModelMetadata(**data)
        perf_path = directory / "performance.json"
        if perf_path.exists():
            self._performance = json.loads(perf_path.read_text(encoding="utf-8"))

    def update_performance(self, metrics: Dict[str, float]) -> None:
        self._performance.update(metrics)

    @property
    def performance(self) -> Dict[str, float]:
        return dict(self._performance)


__all__ = ["BaseModel", "ModelMetadata"]
