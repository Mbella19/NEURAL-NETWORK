"""Time-based features (Phase 3.8)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .base import BaseFeatureCalculator

HOLIDAYS = (
    "2023-12-25",
    "2024-01-01",
    "2024-07-04",
    "2024-12-25",
)


class TimeFeatures(BaseFeatureCalculator):
    name = "time_features"
    required_columns = ("TIMESTAMP",)
    produces = (
        "HOUR_SIN",
        "HOUR_COS",
        "DOW_SIN",
        "DOW_COS",
        "ASIA_SESSION",
        "LONDON_SESSION",
        "NEWYORK_SESSION",
        "SESSION_OVERLAP",
        "HOLIDAY_FLAG",
    )

    def __init__(self, *, holidays: Iterable[str] = HOLIDAYS) -> None:
        super().__init__()
        self.holidays = {pd.Timestamp(h).date() for h in holidays}

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(frame["TIMESTAMP"])
        hour = ts.dt.hour + ts.dt.minute / 60.0
        dow = ts.dt.dayofweek

        hour_angle = 2 * np.pi * hour / 24
        dow_angle = 2 * np.pi * dow / 7

        hour_sin = pd.Series(np.sin(hour_angle), index=frame.index)
        hour_cos = pd.Series(np.cos(hour_angle), index=frame.index)
        dow_sin = pd.Series(np.sin(dow_angle), index=frame.index)
        dow_cos = pd.Series(np.cos(dow_angle), index=frame.index)

        asia = self._session_flag(hour, 0, 8)
        london = self._session_flag(hour, 8, 17)
        newyork = self._session_flag(hour, 13, 22)
        overlap = ((london & newyork) | (asia & london)).astype(int)

        holiday_flag = ts.dt.date.isin(self.holidays).astype(int)

        return pd.DataFrame(
            {
                "HOUR_SIN": hour_sin,
                "HOUR_COS": hour_cos,
                "DOW_SIN": dow_sin,
                "DOW_COS": dow_cos,
                "ASIA_SESSION": asia,
                "LONDON_SESSION": london,
                "NEWYORK_SESSION": newyork,
                "SESSION_OVERLAP": overlap,
                "HOLIDAY_FLAG": holiday_flag,
            },
            index=frame.index,
        )

    @staticmethod
    def _session_flag(hour_series: pd.Series, start: float, end: float) -> pd.Series:
        if start < end:
            return ((hour_series >= start) & (hour_series < end)).astype(int)
        return (((hour_series >= start) | (hour_series < end))).astype(int)


__all__ = ["TimeFeatures"]
