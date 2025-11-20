"""Simple concept drift detector (ADWIN-like) to guard against regime shifts."""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class DriftDetector:
    def __init__(self, window_size: int = 500, delta: float = 0.02) -> None:
        self.window_size = window_size
        self.delta = delta
        self.recent: Deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> bool:
        """Return True when drift is detected."""

        self.recent.append(value)
        if len(self.recent) < self.window_size:
            return False

        arr = np.fromiter(self.recent, dtype=float)
        mid = len(arr) // 2
        left, right = arr[:mid], arr[mid:]
        if len(left) == 0 or len(right) == 0:
            return False

        mean_diff = abs(left.mean() - right.mean())
        pooled_std = np.sqrt(((left.std() ** 2) + (right.std() ** 2)) / 2)
        threshold = self.delta * (pooled_std + 1e-9)
        return bool(mean_diff > threshold)


__all__ = ["DriftDetector"]
