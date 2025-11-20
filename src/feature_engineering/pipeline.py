"""Feature pipeline orchestration (Phase 3.10)."""
from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence

import pandas as pd

from .base import BaseFeatureCalculator, FeaturePipeline, FeatureResult
from .candlestick_patterns import CandlestickPatternFeatures
from .market_regime import MarketRegimeFeatures
from .market_structure import MarketStructureFeatures
from .smart_money import SmartMoneyFeatures
from .support_resistance import SupportResistanceFeatures
from .technical_indicators import TechnicalIndicatorFeatures
from .time_features import TimeFeatures
from .wick_features import WickFeatures


def build_default_feature_pipeline() -> FeaturePipeline:
    calculators: Iterable[BaseFeatureCalculator] = [
        TimeFeatures(),
        TechnicalIndicatorFeatures(
            sma_windows=(14, 50, 200),
            ema_windows=(20,),
        ),
        WickFeatures(),
        CandlestickPatternFeatures(),
        MarketStructureFeatures(lookback=5),
        MarketStructureFeatures(lookback=50),
        SmartMoneyFeatures(swing_lookback=5),
        SmartMoneyFeatures(swing_lookback=50),
        SupportResistanceFeatures(window=10, prominence=0.0004),
        SupportResistanceFeatures(window=60, prominence=0.0015),
        MarketRegimeFeatures(),
    ]
    return FeaturePipeline(calculators)


def run_feature_pipeline(
    frame: pd.DataFrame,
    *,
    pipeline: Optional[FeaturePipeline] = None,
    selected_features: Optional[Sequence[str]] = None,
) -> FeatureResult:
    pipeline = pipeline or build_default_feature_pipeline()
    result = pipeline.run(frame)
    # Deduplicate columns keeping the last occurrence to avoid noisy overwrite warnings downstream.
    result.dataframe = result.dataframe.loc[:, ~result.dataframe.columns.duplicated(keep="last")]
    if selected_features:
        # Convert to list to avoid pandas treating tuples as multi-index
        feature_list = list(selected_features)
        result.dataframe = result.dataframe[feature_list]
        result.feature_names = feature_list
    return result


def run_pipeline_on_iterator(
    frame_iterator: Iterator[pd.DataFrame],
    *,
    pipeline: Optional[FeaturePipeline] = None,
) -> Iterator[FeatureResult]:
    pipeline = pipeline or build_default_feature_pipeline()
    for chunk in frame_iterator:
        yield pipeline.run(chunk)


__all__ = ["build_default_feature_pipeline", "run_feature_pipeline", "run_pipeline_on_iterator"]
