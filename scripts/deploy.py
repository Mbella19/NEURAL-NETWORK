#!/usr/bin/env python3
"""Deployment automation script (Phase 9.3)."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / "venv"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def setup_environment() -> None:
    if not ENV_PATH.exists():
        run(["python3", "-m", "venv", str(ENV_PATH)])
    pip = ENV_PATH / "bin" / "pip"
    run([str(pip), "install", "-r", str(REPO_ROOT / "requirements.txt")])


def deploy_model(model_path: Path) -> None:
    target = REPO_ROOT / "deployed_models"
    target.mkdir(parents=True, exist_ok=True)
    run(["cp", str(model_path), str(target / model_path.name)])


def main() -> None:
    setup_environment()
    deploy_model(REPO_ROOT / "checkpoints" / "best_model.pt")


if __name__ == "__main__":
    main()
