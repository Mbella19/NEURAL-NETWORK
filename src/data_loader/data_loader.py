"""Market data loader utilities for the EURUSD curriculum dataset.

Design goals pulled from AGENTS.md + TASK.md:
- Memory awareness (stream data in chunks, avoid unnecessary copies)
- Strict validation to prevent data leakage or corrupted candles
- Clear abstractions for future MT5/live integrations
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import pandas as pd
from loguru import logger

from config import get_settings
from utils.logger import configure_logging, log_resource_snapshot

# Ensure logging is configured when this module is imported directly.
configure_logging()

DEFAULT_FILE = "eurusd_m1_5y_part2.csv"
REQUIRED_COLUMNS = ("TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "SPREAD")
DTYPE_HINTS = {
    "OPEN": "float32",
    "HIGH": "float32",
    "LOW": "float32",
    "CLOSE": "float32",
    "TICKVOL": "float32",  # Changed to float32 as input is float
    "VOL": "int32",
    "SPREAD": "float32",
}


class DataLoaderError(RuntimeError):
    """Base exception for loader failures."""


class DataValidationError(DataLoaderError):
    """Raised when incoming data fails quality checks."""


@dataclass
class DataLoadResult:
    """Container for loaded market data and accompanying metadata."""

    frame: pd.DataFrame
    metadata: Dict[str, object]

    def to_quality_report(self) -> Dict[str, object]:
        """Return serializable quality metrics for this dataset."""

        frame = self.frame
        timestamp = frame["TIMESTAMP"]
        duration = (timestamp.max() - timestamp.min()).days if not frame.empty else 0
        time_diffs = timestamp.diff().dropna().dt.total_seconds()
        duplicates = int(timestamp.duplicated().sum())
        missing = {col: int(frame[col].isna().sum()) for col in frame.columns}

        price_high_violation = bool((frame["HIGH"] < frame[["OPEN", "CLOSE"]].max(axis=1)).any())
        price_low_violation = bool((frame["LOW"] > frame[["OPEN", "CLOSE"]].min(axis=1)).any())

        weekend_rows = int(
            frame["TIMESTAMP"]
            .dt.dayofweek.isin([5, 6])
            .sum()
        )

        return {
            "source": self.metadata.get("source"),
            "rows": int(self.metadata.get("rows", len(frame))),
            "date_range": {
                "start": timestamp.min().isoformat() if not frame.empty else None,
                "end": timestamp.max().isoformat() if not frame.empty else None,
                "duration_days": duration,
            },
            "median_interval_seconds": float(time_diffs.median()) if not time_diffs.empty else None,
            "duplicate_timestamps": duplicates,
            "missing_values": missing,
            "price_integrity": {
                "high_ge_open_close": not price_high_violation,
                "low_le_open_close": not price_low_violation,
            },
            "weekend_rows": weekend_rows,
            "columns": self.metadata.get("columns"),
        }


class MarketDataLoader:
    """Handles loading and validating historical EURUSD data from disk."""

    def __init__(self, training_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.training_dir = training_dir or settings.trading.datasource_root
        self.raw_dir = settings.data_paths.raw
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        log_resource_snapshot("data_loader:init")

    # --------------------------------------------------------------------- API
    def load_dataframe(
        self,
        file_name: str = DEFAULT_FILE,
        *,
        chunk_size: Optional[int] = None,
        copy_to_raw: bool = False,
        validate: bool = True,
    ) -> DataLoadResult:
        """Load the requested historical file into memory or an iterator."""

        path = self._resolve_file(file_name)
        fmt = self._infer_format(path)
        logger.bind(source="data_loader").info("Loading file %s (format=%s)", path, fmt)

        if chunk_size:
            iterator = self.stream_batches(path, fmt=fmt, chunk_size=chunk_size, validate=validate)
            # Materialize generator if requested; otherwise return metadata only.
            aggregated = pd.concat(iterator, ignore_index=True)
            frame = aggregated
        else:
            frame = self._read_file(path, fmt=fmt)
            if validate:
                self._validate_frame(frame)

        if copy_to_raw:
            destination = self.raw_dir / path.name
            if destination.resolve() != path.resolve():
                destination.write_bytes(path.read_bytes())

        metadata = self._build_metadata(frame, source=str(path))
        return DataLoadResult(frame=frame, metadata=metadata)

    def ingest(
        self,
        file_name: str = DEFAULT_FILE,
        *,
        chunk_size: Optional[int] = None,
        quality_report: bool = True,
        copy_to_raw: bool = True,
    ) -> DataLoadResult:
        """Load, validate, copy to raw directory, and emit a quality report."""

        result = self.load_dataframe(
            file_name=file_name,
            chunk_size=chunk_size,
            copy_to_raw=copy_to_raw,
            validate=True,
        )
        if quality_report:
            self.write_quality_report(result)
        return result

    def stream_batches(
        self,
        path: Path,
        *,
        fmt: Optional[str] = None,
        chunk_size: int = 200_000,
        validate: bool = True,
    ) -> Iterator[pd.DataFrame]:
        """Yield chunks of data for memory-aware processing."""

        fmt = fmt or self._infer_format(path)
        if fmt not in {"csv", "tsv"}:
            raise DataLoaderError(f"Chunked streaming is only available for CSV/TSV; got {fmt}")

        sep = "," if fmt == "csv" else "\t"
        try:
            iterator = pd.read_csv(
                path,
                sep=sep,
                dtype=DTYPE_HINTS,
                chunksize=chunk_size,
                parse_dates={"timestamp": ["DATE", "TIME"]},
            )
        except Exception as exc:  # pragma: no cover - delegated to pandas
            raise DataLoaderError(f"Unable to stream file {path}") from exc

        for chunk in iterator:
            chunk = self._normalize_columns(chunk)
            if validate:
                self._validate_frame(chunk, raise_on_fail=True)
            yield chunk

    # ------------------------------------------------------------------ Helpers
    def _resolve_file(self, file_name: str) -> Path:
        candidate = Path(file_name)
        if not candidate.is_absolute():
            candidate = self.training_dir / file_name
        if not candidate.exists():
            raise DataLoaderError(f"Training file not found: {candidate}")
        return candidate

    @staticmethod
    def _infer_format(path: Path) -> str:
        ext = path.suffix.lower().lstrip(".")
        if ext in {"csv", "tsv", "parquet", "pq", "h5", "hdf5"}:
            return "tsv" if ext == "tsv" else ("parquet" if ext in {"parquet", "pq"} else ("hdf5" if ext in {"h5", "hdf5"} else "csv"))
        raise DataLoaderError(f"Unsupported file format: {path.suffix}")

    def _read_file(self, path: Path, *, fmt: Optional[str] = None) -> pd.DataFrame:
        fmt = fmt or self._infer_format(path)
        if fmt in {"csv", "tsv"}:
            sep = self._detect_separator(path) if fmt == "csv" else "\t"
            frame = pd.read_csv(
                path,
                sep=sep,
            )
        elif fmt == "parquet":
            frame = pd.read_parquet(path)
        elif fmt == "hdf5":
            frame = pd.read_hdf(path)
        else:  # pragma: no cover - guarded by _infer_format
            raise DataLoaderError(f"Unknown format: {fmt}")

        frame = self._normalize_columns(frame)
        return frame

    def write_quality_report(self, result: DataLoadResult, *, output_dir: Optional[Path] = None) -> Path:
        """Persist a JSON quality report for the provided dataset."""

        output_dir = output_dir or (self.raw_dir / "reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        report = result.to_quality_report()
        source_name = Path(report["source"]).stem if report.get("source") else DEFAULT_FILE
        report_path = output_dir / f"{source_name}_quality.json"

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.bind(source="data_loader").info("Quality report written to %s", report_path)
        return report_path

    @staticmethod
    def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        def _clean(name: str) -> str:
            return name.strip().strip("<>").upper()

        columns = {col: _clean(col) for col in frame.columns}
        frame = frame.rename(columns=columns)
        
        # Handle column mapping for new 5-year dataset
        if "TICK_VOLUME" in frame.columns and "TICKVOL" not in frame.columns:
            frame = frame.rename(columns={"TICK_VOLUME": "TICKVOL"})
            
        # Handle timestamp creation/conversion
        if "TIMESTAMP" not in frame.columns:
            if {"DATE", "TIME"} <= set(frame.columns):
                frame["TIMESTAMP"] = pd.to_datetime(frame["DATE"] + " " + frame["TIME"], errors="coerce")
            elif "timestamp" in frame.columns:
                 frame["TIMESTAMP"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        else:
            # Ensure existing TIMESTAMP is actually datetime
            frame["TIMESTAMP"] = pd.to_datetime(frame["TIMESTAMP"], errors="coerce")

        # Ensure VOL exists (fill with 0 if missing)
        if "VOL" not in frame.columns:
            frame["VOL"] = 0

        for col, dtype in DTYPE_HINTS.items():
            if col in frame.columns:
                frame[col] = frame[col].astype(dtype, errors="ignore")
        
        # Drop rows with invalid timestamps
        frame = frame.dropna(subset=["TIMESTAMP"])
        
        frame = frame.sort_values("TIMESTAMP").reset_index(drop=True)
        return frame

    @staticmethod
    def _detect_separator(path: Path, sample_bytes: int = 2048) -> str:
        """Heuristic to determine whether a .csv file is comma or tab separated."""

        with path.open("r", encoding="utf-8", newline="") as fh:
            sample = fh.read(sample_bytes)
        tab_count = sample.count("\t")
        comma_count = sample.count(",")
        return "\t" if tab_count > comma_count else ","

    def _validate_frame(self, frame: pd.DataFrame, raise_on_fail: bool = True) -> Tuple[bool, Dict[str, object]]:
        """Run sanity checks to prevent data leakage or corrupted candles."""

        issues = {}
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            issues["missing_columns"] = sorted(missing)

        if frame["TIMESTAMP"].isnull().any():
            issues["invalid_timestamps"] = int(frame["TIMESTAMP"].isnull().sum())

        if not frame["TIMESTAMP"].is_monotonic_increasing:
            issues["monotonic"] = False

        if (frame["HIGH"] < frame[["OPEN", "CLOSE"]].max(axis=1)).any():
            issues["high_violation"] = True
        if (frame["LOW"] > frame[["OPEN", "CLOSE"]].min(axis=1)).any():
            issues["low_violation"] = True

        if frame["TIMESTAMP"].duplicated().any():
            issues["duplicates"] = int(frame["TIMESTAMP"].duplicated().sum())

        if issues and raise_on_fail:
            raise DataValidationError(f"Data quality checks failed: {issues}")

        if issues:
            logger.bind(source="data_loader").warning("Validation issues detected: %s", issues)
        else:
            logger.bind(source="data_loader").debug("Validation checks passed")
        return (not issues, issues)

    @staticmethod
    def _build_metadata(frame: pd.DataFrame, *, source: str) -> Dict[str, object]:
        return {
            "source": source,
            "source_name": Path(source).name,
            "rows": len(frame),
            "start": frame["TIMESTAMP"].min(),
            "end": frame["TIMESTAMP"].max(),
            "columns": list(frame.columns),
        }


# Convenience functions for other modules --------------------------------------
def load_training_data(**kwargs) -> DataLoadResult:
    """Shortcut to load the canonical EURUSD dataset defined in TASK.md."""

    loader = MarketDataLoader()
    return loader.load_dataframe(**kwargs)


def ingest_training_data(**kwargs) -> DataLoadResult:
    """Load data, copy into raw directory, and generate a quality report."""

    loader = MarketDataLoader()
    return loader.ingest(**kwargs)


def stream_training_data(chunk_size: int = 200_000, **kwargs) -> Iterable[pd.DataFrame]:
    loader = MarketDataLoader()
    path = loader._resolve_file(kwargs.get("file_name", DEFAULT_FILE))
    return loader.stream_batches(path, chunk_size=chunk_size, validate=kwargs.get("validate", True))


__all__ = [
    "DataLoadResult",
    "DataLoaderError",
    "DataValidationError",
    "MarketDataLoader",
    "ingest_training_data",
    "load_training_data",
    "stream_training_data",
]
