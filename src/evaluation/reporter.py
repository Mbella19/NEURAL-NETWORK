"""Performance reporting (Phase 7.5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json


@dataclass
class ReportConfig:
    output_dir: Path = Path("reports")


class PerformanceReporter:
    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or ReportConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, metrics: Dict[str, float], name: str = "report") -> Path:
        path = self.config.output_dir / f"{name}.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path


__all__ = ["PerformanceReporter", "ReportConfig"]
