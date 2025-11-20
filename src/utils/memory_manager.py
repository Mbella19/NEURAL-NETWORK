"""Memory management utilities (Phase 6.4)."""
from __future__ import annotations

import contextlib
import gc
import os
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, Optional

import psutil
import torch


@dataclass
class MemoryManagerConfig:
    target_batch_size: int = 64
    max_memory_gb: float = 6.0
    profile_interval: int = 100


class MemoryManager:
    def __init__(self, config: MemoryManagerConfig | None = None) -> None:
        self.config = config or MemoryManagerConfig()
        self.process = psutil.Process(os.getpid())

    def profile(self) -> dict[str, float]:
        mem = self.process.memory_info().rss / (1024**3)
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        return {"rss_gb": mem, "gpu_gb": gpu_mem}

    def adjust_batch_size(self, current_batch_size: int) -> int:
        stats = self.profile()
        if stats["rss_gb"] and stats["rss_gb"] > self.config.max_memory_gb:
            return max(current_batch_size // 2, 1)
        return current_batch_size

    @contextlib.contextmanager
    def manage(self) -> Generator[None, None, None]:
        try:
            yield
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def data_generator(data: Iterable, batch_size: int) -> Generator[list, None, None]:
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = ["MemoryManager", "MemoryManagerConfig", "data_generator"]
