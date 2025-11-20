"""Smart Money Concepts features (Phase 3.5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .base import BaseFeatureCalculator


@dataclass
class SMCState:
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    last_swing_high_idx: Optional[int] = None
    last_swing_low_idx: Optional[int] = None
    trend: int = 0


class SmartMoneyFeatures(BaseFeatureCalculator):
    name = "smart_money"
    required_columns = ("OPEN", "HIGH", "LOW", "CLOSE")
    # Note: produces is dynamically generated in __init__ to include suffix

    def __init__(self, *, swing_lookback: int = 5, tolerance: float = 1e-5) -> None:
        super().__init__(window=swing_lookback)
        self.swing_lookback = swing_lookback
        self.tolerance = tolerance
        self.min_swing_spacing = 10

        # Dynamically set produces with suffix
        suffix = f"_SW{self.swing_lookback}"
        self.produces = (
            f"BOS_BULLISH{suffix}",
            f"BOS_BEARISH{suffix}",
            f"CHOCH_BULLISH{suffix}",
            f"CHOCH_BEARISH{suffix}",
            f"ORDER_BLOCK_BULL{suffix}",
            f"ORDER_BLOCK_BEAR{suffix}",
            f"FVG_UP{suffix}",
            f"FVG_DOWN{suffix}",
            f"LIQUIDITY_SWEEP{suffix}",
            f"IMBALANCE_ZONE{suffix}",
            f"LIQUIDITY_POOL_BUY{suffix}",
            f"LIQUIDITY_POOL_SELL{suffix}",
        )

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        high = frame["HIGH"]
        low = frame["LOW"]
        open_ = frame["OPEN"]
        close = frame["CLOSE"]

        rolling_high = high.rolling(window=self.swing_lookback, min_periods=1).max()
        rolling_low = low.rolling(window=self.swing_lookback, min_periods=1).min()

        swing_high = ((high - rolling_high).abs() <= self.tolerance).astype(int)
        swing_low = ((low - rolling_low).abs() <= self.tolerance).astype(int)

        state = SMCState()
        bos_bull: List[int] = []
        bos_bear: List[int] = []
        choch_bull: List[int] = []
        choch_bear: List[int] = []
        ob_bull: List[int] = []
        ob_bear: List[int] = []
        liquidity_sweep: List[int] = []

        for idx in range(len(frame)):
            bull_bos = 0
            bear_bos = 0
            choch_b = 0
            choch_s = 0
            ob_b = 0
            ob_s = 0

            if swing_high.iat[idx]:
                if state.last_swing_high_idx is None or (idx - state.last_swing_high_idx) >= self.min_swing_spacing:
                    state.last_swing_high = high.iat[idx]
                    state.last_swing_high_idx = idx
            if swing_low.iat[idx]:
                if state.last_swing_low_idx is None or (idx - state.last_swing_low_idx) >= self.min_swing_spacing:
                    state.last_swing_low = low.iat[idx]
                    state.last_swing_low_idx = idx

            if state.last_swing_high is not None and close.iat[idx] > state.last_swing_high + self.tolerance:
                bull_bos = 1
                if state.trend <= 0:
                    choch_b = 1
                state.trend = 1
                state.last_swing_high = close.iat[idx]
                state.last_swing_high_idx = idx
                # Order block: last down candle before the break (up to 5 bars back)
                order_block_idx = idx
                for lookback in range(1, 6):
                    prev_idx = idx - lookback
                    if prev_idx < 0:
                        break
                    if close.iat[prev_idx] < open_.iat[prev_idx]:
                        order_block_idx = prev_idx
                        break
                ob_b = 1 if order_block_idx == idx else 0
                if order_block_idx < idx and order_block_idx >= 0:
                    # Ensure list has been populated up to this point
                    while len(ob_bull) <= order_block_idx:
                        ob_bull.append(0)
                        bos_bull.append(0)
                        bos_bear.append(0)
                        choch_bull.append(0)
                        choch_bear.append(0)
                        ob_bear.append(0)
                        liquidity_sweep.append(0)
                    ob_bull[order_block_idx] = 1
            elif state.last_swing_low is not None and close.iat[idx] < state.last_swing_low - self.tolerance:
                bear_bos = 1
                if state.trend >= 0:
                    choch_s = 1
                state.trend = -1
                state.last_swing_low = close.iat[idx]
                state.last_swing_low_idx = idx
                # Order block: last up candle before the break (up to 5 bars back)
                order_block_idx = idx
                for lookback in range(1, 6):
                    prev_idx = idx - lookback
                    if prev_idx < 0:
                        break
                    if close.iat[prev_idx] > open_.iat[prev_idx]:
                        order_block_idx = prev_idx
                        break
                ob_s = 1 if order_block_idx == idx else 0
                if order_block_idx < idx and order_block_idx >= 0:
                    while len(ob_bear) <= order_block_idx:
                        ob_bear.append(0)
                        bos_bull.append(0)
                        bos_bear.append(0)
                        choch_bull.append(0)
                        choch_bear.append(0)
                        ob_bull.append(0)
                        liquidity_sweep.append(0)
                    ob_bear[order_block_idx] = 1

            # Liquidity sweep: wick breaks previous extreme but closes opposite
            if idx > 0:
                if high.iat[idx] > high.iat[idx - 1] + self.tolerance and close.iat[idx] < close.iat[idx - 1]:
                    liquidity_sweep.append(1)
                elif low.iat[idx] < low.iat[idx - 1] - self.tolerance and close.iat[idx] > close.iat[idx - 1]:
                    liquidity_sweep.append(1)
                else:
                    liquidity_sweep.append(0)
            else:
                liquidity_sweep.append(0)

            bos_bull.append(bull_bos)
            bos_bear.append(bear_bos)
            choch_bull.append(choch_b)
            choch_bear.append(choch_s)
            ob_bull.append(ob_b)
            ob_bear.append(ob_s)

        fvg_up = (
            (low > high.shift(2) + self.tolerance)
            .fillna(0)
            .astype(int)
        )
        fvg_down = (
            (high < low.shift(2) - self.tolerance)
            .fillna(0)
            .astype(int)
        )

        body = (close - open_).abs()
        candle_range = (high - low).replace(0, pd.NA)
        body_ratio = (body / candle_range).fillna(0.0)
        imbalance_zone = (body_ratio >= 0.7).astype(int)

        pool_window = max(5, self.swing_lookback)
        rolling_high = high.rolling(window=pool_window, min_periods=pool_window).max()
        rolling_low = low.rolling(window=pool_window, min_periods=pool_window).min()
        liquidity_pool_buy = ((rolling_high - high).abs() <= self.tolerance).astype(int)
        liquidity_pool_sell = ((low - rolling_low).abs() <= self.tolerance).astype(int)

        # Add suffix to avoid overwrites when using multiple lookbacks
        suffix = f"_SW{self.swing_lookback}"
        return pd.DataFrame(
            {
                f"BOS_BULLISH{suffix}": bos_bull,
                f"BOS_BEARISH{suffix}": bos_bear,
                f"CHOCH_BULLISH{suffix}": choch_bull,
                f"CHOCH_BEARISH{suffix}": choch_bear,
                f"ORDER_BLOCK_BULL{suffix}": ob_bull,
                f"ORDER_BLOCK_BEAR{suffix}": ob_bear,
                f"FVG_UP{suffix}": fvg_up,
                f"FVG_DOWN{suffix}": fvg_down,
                f"LIQUIDITY_SWEEP{suffix}": liquidity_sweep,
                f"IMBALANCE_ZONE{suffix}": imbalance_zone,
                f"LIQUIDITY_POOL_BUY{suffix}": liquidity_pool_buy,
                f"LIQUIDITY_POOL_SELL{suffix}": liquidity_pool_sell,
            },
            index=frame.index,
        )


__all__ = ["SmartMoneyFeatures"]
