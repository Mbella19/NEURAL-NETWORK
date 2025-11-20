"""Advanced data quality checks (Phase 2.5)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import get_settings
from data_loader import MarketDataLoader


@dataclass
class QualityReport:
    summary: Dict[str, Any]
    path: Optional[Path] = None

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary, indent=2, default=float), encoding="utf-8")
        self.path = path
        logger.bind(source="quality_checker").info("Quality report saved to %s", path)
        return path


class DataQualityChecker:
    """Runs sanity checks to detect look-ahead bias, discontinuities, and anomalies."""

    def __init__(
        self,
        frame: Optional[pd.DataFrame] = None,
        *,
        source_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        settings = get_settings()
        self.output_dir = output_dir or settings.data_paths.raw / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if frame is None:
            loader = MarketDataLoader(training_dir=settings.data_paths.raw)
            data = loader.load_dataframe(file_name=source_name or "EURUSD_M1_202306010000_202412302358.csv")
            frame = data.frame
            self.source = data.metadata.get("source_name")
        else:
            self.source = source_name or "in-memory"
        self.frame = frame.sort_values("TIMESTAMP").reset_index(drop=True)

    def run(self, *, price_gap_threshold: float = 0.0005) -> QualityReport:
        report = {
            "source": self.source,
            "rows": len(self.frame),
            "date_range": {
                "start": self.frame["TIMESTAMP"].min().isoformat() if not self.frame.empty else None,
                "end": self.frame["TIMESTAMP"].max().isoformat() if not self.frame.empty else None,
            },
        }
        report["lookahead_bias"] = self._check_lookahead_bias()
        report["price_continuity"] = self._check_price_continuity(price_gap_threshold)
        report["volume_validation"] = self._check_volume()
        report["anomalies"] = self._detect_anomalies()
        return QualityReport(summary=report)

    def _check_lookahead_bias(self) -> Dict[str, Any]:
        timestamps = self.frame["TIMESTAMP"]
        duplicates = int(timestamps.duplicated().sum())
        monotonic = bool(timestamps.is_monotonic_increasing)
        missing = int(timestamps.isna().sum())
        return {
            "monotonic_increasing": monotonic,
            "duplicate_timestamps": duplicates,
            "missing_timestamps": missing,
            "status": monotonic and duplicates == 0 and missing == 0,
        }

    def _check_price_continuity(self, threshold: float) -> Dict[str, Any]:
        open_prices = self.frame["OPEN"]
        prev_close = self.frame["CLOSE"].shift(1)
        gaps = (open_prices - prev_close).abs()
        mask = gaps > threshold
        num_gaps = int(mask.sum())
        max_gap = float(gaps.max() or 0.0)
        sample_rows = []
        if num_gaps:
            subset = self.frame.loc[mask, ["TIMESTAMP", "OPEN"]].head(5)
            gap_values = gaps[mask].head(5)
            for (ts, open_price), gap_val in zip(subset.values, gap_values):
                sample_rows.append(
                    {
                        "timestamp": pd.Timestamp(ts).isoformat(),
                        "open": float(open_price),
                        "gap": float(gap_val),
                    }
                )
        return {
            "threshold": threshold,
            "gaps_detected": num_gaps,
            "max_gap": max_gap,
            "sample_gaps": sample_rows,
        }

    def _check_volume(self) -> Dict[str, Any]:
        tickvol = self.frame.get("TICKVOL")
        vol = self.frame.get("VOL")
        result = {}
        if tickvol is not None:
            result["tickvol_negative"] = int((tickvol < 0).sum())
            result["tickvol_zero_pct"] = float(((tickvol == 0).mean()) * 100)
        if vol is not None:
            result["vol_negative"] = int((vol < 0).sum())
            result["vol_zero_pct"] = float(((vol == 0).mean()) * 100)
        result["status"] = all(value == 0 for key, value in result.items() if key.endswith("negative"))
        return result

    def _detect_anomalies(self, z_thresh: float = 6.0) -> Dict[str, Any]:
        close_prices = self.frame["CLOSE"]
        if close_prices.empty:
            return {"outliers": 0}
        mean = close_prices.mean()
        std = close_prices.std() or 1.0
        z_scores = (close_prices - mean) / std
        mask = z_scores.abs() > z_thresh
        return {
            "threshold": z_thresh,
            "outliers": int(mask.sum()),
            "sample": [float(val) for val in close_prices[mask].head(5).tolist()],
        }


def run_quality_checks(
    *,
    source_name: Optional[str] = None,
    price_gap_threshold: float = 0.0005,
) -> QualityReport:
    checker = DataQualityChecker(source_name=source_name)
    report = checker.run(price_gap_threshold=price_gap_threshold)
    out_path = checker.output_dir / "data_quality_summary.json"
    report.save(out_path)
    return report


__all__ = ["DataQualityChecker", "QualityReport", "run_quality_checks"]
