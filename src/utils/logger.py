"""Project-wide logging utilities tailored for the trading agent.

The implementation follows the guidance from AGENTS.md: logs are structured,
memory-aware, and emphasize observability from day one. This module centralizes
console/file logging, exposes convenience helpers for performance metrics, and
provides lightweight resource monitoring that future subsystems can reuse.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from loguru import logger

try:  # TensorFlow is optional during early development phases.
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - TensorFlow may be unavailable
    tf = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - Torch may be unavailable
    torch = None  # type: ignore

try:
    # Prefer runtime configuration if available.
    from config.settings import get_settings
except Exception:  # pragma: no cover - allow logger to work without config package
    get_settings = None


DISABLE_RESOURCE_MONITORING = os.getenv("DISABLE_RESOURCE_MONITORING", "0").lower() in {"1", "true", "yes", "on"}

if not DISABLE_RESOURCE_MONITORING:
    import psutil  # type: ignore
else:  # pragma: no cover - psutil unused when monitoring disabled
    psutil = None  # type: ignore


DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "PID={process} | "
    "<cyan>{extra[source]:<12}</cyan> | "
    "{message}"
)


@dataclass(frozen=True)
class LoggerConfig:
    """Configuration container for logger setup."""

    level: str = "INFO"
    log_dir: Path = Path("logs/training")
    rotation: str = "50 MB"
    retention: str = "14 days"
    enqueue: bool = False


class ResourceMonitor:
    """Snapshots CPU, memory, and GPU state for observability hooks."""

    def __init__(self) -> None:
        if DISABLE_RESOURCE_MONITORING:
            self.process = None
            self._enabled = False
            return
        try:
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(interval=None)  # Prime internal counters.
            self._enabled = True
        except Exception as exc:  # pragma: no cover - platform specific
            logger.warning("Resource monitor degraded: %s", exc)
            self.process = None
            self._enabled = False

    def snapshot(self) -> Dict[str, Any]:
        """Return a dict with instantaneous resource metrics."""

        if not self._enabled or self.process is None:
            snapshot = {"cpu.percent": None, "mem.rss_gb": None, "mem.percent": None}
        else:
            mem_info = self.process.memory_info()
            system_mem = psutil.virtual_memory()
            snapshot = {
                "cpu.percent": round(self.process.cpu_percent(interval=None), 2),
                "mem.rss_gb": round(mem_info.rss / (1024**3), 3),
                "mem.percent": system_mem.percent,
            }
        snapshot.update(self._gpu_snapshot())
        return snapshot

    def _gpu_snapshot(self) -> Dict[str, Any]:
        """Best-effort GPU statistics for MPS/Metal environments."""

        # PyTorch on Apple Metal.
        if torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / (1024**2)
                driver = torch.mps.driver_allocated_memory() / (1024**2)
            except Exception:
                allocated = driver = None  # MPS stats unsupported on some builds.

            return {
                "gpu.backend": "mps",
                "gpu.available": True,
                "gpu.memory_mb": allocated,
                "gpu.driver_memory_mb": driver,
            }

        # TensorFlow Metal backend (experimental introspection).
        if tf:
            try:
                devices = tf.config.list_physical_devices("GPU")
                if devices:
                    return {
                        "gpu.backend": "tensorflow-metal",
                        "gpu.available": True,
                        "gpu.devices": [d.name for d in devices],
                    }
            except Exception:
                pass

        return {"gpu.backend": None, "gpu.available": False}


def _resolve_default_config() -> LoggerConfig:
    """Use config.settings defaults when available."""

    if get_settings is None:
        return LoggerConfig()

    settings = get_settings()
    log_settings = settings.logging
    return LoggerConfig(
        level=log_settings.level,
        log_dir=log_settings.log_dir,
        rotation=log_settings.rotation,
        retention=log_settings.retention,
        enqueue=True,
    )


def configure_logging(config: Optional[LoggerConfig] = None) -> None:
    """Configure global loguru handlers for console + file output."""

    cfg = config or _resolve_default_config()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.configure(extra={"source": "system"})

    logger.add(
        sys.stdout,
        colorize=True,
        format=DEFAULT_LOG_FORMAT,
        level=cfg.level.upper(),
        enqueue=cfg.enqueue,
    )
    logger.add(
        cfg.log_dir / "app.log",
        rotation=cfg.rotation,
        retention=cfg.retention,
        level=cfg.level.upper(),
        enqueue=cfg.enqueue,
        backtrace=True,
        diagnose=False,
        format=DEFAULT_LOG_FORMAT,
    )
    logger.debug("Logger configured -> dir=%s level=%s", cfg.log_dir, cfg.level)


def log_performance_metrics(
    tag: str,
    *,
    step: Optional[int] = None,
    duration_ms: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured performance log entry."""

    payload = metrics or {}
    metric_str = " ".join(f"{key}={_format_metric(value)}" for key, value in payload.items())
    logger.bind(source=tag).info(
        "step=%s duration_ms=%s %s",
        step if step is not None else "-",
        f"{duration_ms:.2f}" if duration_ms is not None else "-",
        metric_str,
    )


def log_resource_snapshot(label: str, monitor: Optional[ResourceMonitor] = None) -> Dict[str, Any]:
    """Capture and log CPU/memory/GPU utilization."""

    if DISABLE_RESOURCE_MONITORING:
        logger.bind(source=label).debug("Resource monitoring disabled via env flag")
        return {}

    monitor = monitor or ResourceMonitor()
    snapshot = monitor.snapshot()
    logger.bind(source=label).info(
        "Resource usage -> %s",
        " ".join(f"{k}={_format_metric(v)}" for k, v in snapshot.items()),
    )
    return snapshot


def performance_timer(tag: str, metrics: Optional[Dict[str, Any]] = None):
    """Context manager yielding duration-aware logging around code blocks."""

    class _Timer:
        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            duration = (time.perf_counter() - self._start) * 1000
            log_performance_metrics(tag, duration_ms=duration, metrics=metrics)
            if exc:
                logger.bind(source=tag).exception("Error during timed block")
            return False

    return _Timer()


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.5f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_metric(v) for v in value) + "]"
    return str(value)


def iter_with_logging(
    iterable: Iterable[Any],
    *,
    tag: str,
    monitor: Optional[ResourceMonitor] = None,
    every: int = 1000,
) -> Iterable[Any]:
    """Yield items while periodically logging health metrics (for long loops)."""

    monitor = monitor or ResourceMonitor()
    for idx, item in enumerate(iterable, start=1):
        yield item
        if idx % every == 0:
            log_resource_snapshot(tag, monitor)


__all__ = [
    "LoggerConfig",
    "ResourceMonitor",
    "configure_logging",
    "iter_with_logging",
    "log_performance_metrics",
    "log_resource_snapshot",
    "performance_timer",
    "DISABLE_RESOURCE_MONITORING",
]
