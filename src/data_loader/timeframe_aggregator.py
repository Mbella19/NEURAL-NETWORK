"""Timeframe aggregation utilities for EURUSD data.

Phase 2.3 requirements (AGENTS.md + TASK.md):
- Convert the minute-level dataset into higher timeframes without leaking data
- Preserve OHLCV integrity (first open, last close, max high, min low, sum volume)
- Handle gaps/missing periods and report anomalies
- Operate memory efficiently (single resample pass, optional chunking)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from loguru import logger

from config import get_settings
from data_loader import DataLoadResult, MarketDataLoader

try:
    import pyarrow  # type: ignore  # noqa: F401

    _PARQUET_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _PARQUET_AVAILABLE = False

TIMEFRAME_MAP = {
    "M5": "5T",
    "M15": "15T",
    "H1": "1H",
}


@dataclass
class AggregationResult:
    timeframe: str
    frame: pd.DataFrame
    metadata: Dict[str, object]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.frame.copy()
        try:
            if path.suffix.lower() == ".parquet" and _PARQUET_AVAILABLE:
                frame.to_parquet(path, index=False)
            else:
                if path.suffix.lower() != ".csv":
                    path = path.with_suffix(".csv")
                frame.to_csv(path, index=False)
        except Exception as exc:
            logger.bind(source="aggregator").exception("Failed to save %s timeframe", self.timeframe)
            raise
        else:
            logger.bind(source="aggregator").info("Saved %s timeframe to %s", self.timeframe, path)
            return path


class TimeframeAggregator:
    """Aggregates minute data to higher timeframes with validation."""

    def __init__(
        self,
        base_data: Optional[pd.DataFrame] = None,
        *,
        instrument: str = "EURUSD",
        output_dir: Optional[Path] = None,
    ) -> None:
        settings = get_settings()
        self.instrument = instrument
        self.output_dir = output_dir or settings.data_paths.processed / "timeframes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if base_data is None:
            loader = MarketDataLoader()
            base_data = loader.load_dataframe().frame
        self.base_frame = base_data.copy()
        if "TIMESTAMP" not in self.base_frame.columns:
            raise ValueError("Base data requires a 'TIMESTAMP' column for resampling.")
        self.base_frame = self.base_frame.sort_values("TIMESTAMP").reset_index(drop=True)

    def aggregate(self, timeframe: str) -> AggregationResult:
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Expected one of {list(TIMEFRAME_MAP)}")

        freq = TIMEFRAME_MAP[timeframe]
        df = self.base_frame.set_index("TIMESTAMP")
        aggregated = df.resample(freq, label="right", closed="right").agg(
            {
                "OPEN": "first",
                "HIGH": "max",
                "LOW": "min",
                "CLOSE": "last",
                "TICKVOL": "sum",
                "VOL": "sum",
                "SPREAD": "mean",
            }
        )

        aggregated = aggregated.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
        aggregated["SPREAD"] = aggregated["SPREAD"].ffill().bfill()
        aggregated = aggregated.reset_index().rename(columns={"TIMESTAMP": "END_TIME"})
        aggregated["START_TIME"] = aggregated["END_TIME"] - pd.to_timedelta(freq)
        aggregated = aggregated[["START_TIME", "END_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "VOL", "SPREAD"]]

        gaps = self._detect_gaps(aggregated["END_TIME"], freq)
        self._validate_frame(aggregated)
        metadata = {
            "timeframe": timeframe,
            "rows": len(aggregated),
            "start": aggregated["START_TIME"].min(),
            "end": aggregated["END_TIME"].max(),
            "gaps_detected": len(gaps),
            "expected_freq": freq,
        }
        if gaps:
            metadata["gap_samples"] = gaps[:5]
            logger.bind(source="aggregator").warning(
                "Detected %s gaps in timeframe %s", len(gaps), timeframe
            )

        return AggregationResult(timeframe=timeframe, frame=aggregated, metadata=metadata)

    def aggregate_all(self, timeframes: Optional[Iterable[str]] = None) -> List[AggregationResult]:
        timeframes = list(timeframes or TIMEFRAME_MAP.keys())
        results = []
        for tf in timeframes:
            result = self.aggregate(tf)
            result_path = self.output_dir / f"{self.instrument}_{tf}.parquet"
            result.save(result_path)
            results.append(result)
        return results

    @staticmethod
    def _detect_gaps(timestamps: pd.Series, freq: str) -> List[str]:
        expected_delta = pd.to_timedelta(freq)
        diffs = timestamps.diff().dropna()
        gap_mask = diffs > expected_delta
        gaps = timestamps[gap_mask.index[gap_mask]].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        return gaps

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        issues = {}
        if (frame["HIGH"] < frame[["OPEN", "CLOSE"]].max(axis=1)).any():
            issues["high_violation"] = True
        if (frame["LOW"] > frame[["OPEN", "CLOSE"]].min(axis=1)).any():
            issues["low_violation"] = True
        if (frame["OPEN"] > frame["HIGH"]).any() or (frame["OPEN"] < frame["LOW"]).any():
            issues["open_out_of_range"] = True
        if (frame["CLOSE"] > frame["HIGH"]).any() or (frame["CLOSE"] < frame["LOW"]).any():
            issues["close_out_of_range"] = True
        if (frame[["TICKVOL", "VOL"]] < 0).any().any():
            issues["negative_volume"] = True
        if issues:
            raise ValueError(f"Aggregated frame failed validation: {issues}")


def aggregate_timeframes_from_raw(
    *,
    timeframes: Optional[Iterable[str]] = None,
    instrument: str = "EURUSD",
    raw_file: Optional[str] = None,
) -> List[AggregationResult]:
    settings = get_settings()
    raw_dir = settings.data_paths.raw
    file_name = raw_file or "EURUSD_M1_202306010000_202412302358.csv"
    loader = MarketDataLoader(training_dir=raw_dir)
    data = loader.load_dataframe(file_name=file_name, copy_to_raw=False)
    aggregator = TimeframeAggregator(base_data=data.frame, instrument=instrument)
    return aggregator.aggregate_all(timeframes)


__all__ = ["AggregationResult", "TimeframeAggregator", "aggregate_timeframes_from_raw"]
