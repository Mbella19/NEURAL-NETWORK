"""Single-process runner for the live prediction dashboard."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from uvicorn import Config, Server

from inference.live_dashboard import create_app, LiveDashboardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live dashboard without uvicorn CLI reloads")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--testing-csv", type=str, default=None, help="Override testing CSV path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = LiveDashboardConfig()
    if args.testing_csv:
        cfg.testing_csv = os.path.expanduser(args.testing_csv)
    if args.checkpoint:
        cfg.checkpoint_path = os.path.expanduser(args.checkpoint)

    app = create_app(cfg)
    server = Server(
        Config(
            app=app,
            host=args.host,
            port=args.port,
            reload=False,
            loop="asyncio",
            http="h11",
        )
    )
    server.run()


if __name__ == "__main__":
    main()
