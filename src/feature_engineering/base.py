"""Feature engineering base classes (Phase 3.1)."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger


class FeatureCalculationError(RuntimeError):
    """Raised when a feature calculator fails."""


class BaseFeatureCalculator(abc.ABC):
    """Abstract calculator contract for all feature modules."""

    name: str = "base"
    required_columns: Sequence[str] = ()
    produces: Sequence[str] = ()

    def __init__(self, *, window: Optional[int] = None) -> None:
        self.window = window

    def validate_input(self, frame: pd.DataFrame) -> None:
        missing = set(self.required_columns) - set(frame.columns)
        if missing:
            raise FeatureCalculationError(f"{self.name}: missing columns {missing}")

    @abc.abstractmethod
    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with the new feature columns."""

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(frame)
        features = self.compute(frame)
        if self.produces:
            unexpected = set(features.columns) - set(self.produces)
            if unexpected:
                logger.bind(source=self.name).warning("Produced unexpected columns %s", unexpected)
        return features


@dataclass
class FeatureResult:
    dataframe: pd.DataFrame
    feature_names: List[str]


class FeaturePipeline:
    """Orchestrates multiple feature calculators with validation hooks."""

    def __init__(self, calculators: Optional[Iterable[BaseFeatureCalculator]] = None) -> None:
        self.calculators = list(calculators or [])

    def add(self, calculator: BaseFeatureCalculator) -> None:
        self.calculators.append(calculator)

    def run(self, frame: pd.DataFrame) -> FeatureResult:
        working = frame.copy()
        new_columns: List[str] = []
        seen_overwrites = set()

        for calculator in self.calculators:
            logger.bind(source="feature_pipeline").info("Running calculator %s", calculator.name)
            features = calculator(working)
            for col in features.columns:
                if col in working.columns and col not in seen_overwrites:
                    logger.bind(source=calculator.name).warning("Overwriting column %s", col)
                    seen_overwrites.add(col)
                working[col] = features[col]
            new_columns.extend(list(features.columns))
            self._validate_features(working, features.columns)

        return FeatureResult(dataframe=working, feature_names=new_columns)

    @staticmethod
    def _validate_features(frame: pd.DataFrame, columns: Iterable[str]) -> None:
        for col in columns:
            if frame[col].isna().any():
                logger.bind(source="feature_pipeline").warning("Feature %s contains NaN values", col)
            if pd.api.types.is_numeric_dtype(frame[col]):
                if not np.isfinite(frame[col]).all():
                    logger.bind(source="feature_pipeline").warning("Feature %s contains non-finite values", col)


__all__ = [
    "BaseFeatureCalculator",
    "FeatureCalculationError",
    "FeaturePipeline",
    "FeatureResult",
]
