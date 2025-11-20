"""Simple gymnasium-style trading environment for PPO/A3C."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception as exc:  # pragma: no cover
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None


@dataclass
class TradingEnvConfig:
    price_column: str = "CLOSE"
    lookback: int = 60
    initial_equity: float = 1.0
    transaction_cost: float = 0.0001
    max_steps: Optional[int] = None


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data, config: TradingEnvConfig | None = None) -> None:
        if gym is None or spaces is None:
            raise ImportError(
                "gymnasium is required for TradingEnv. Install with `pip install gymnasium`."
            ) from _GYM_IMPORT_ERROR
        super().__init__()
        self.cfg = config or TradingEnvConfig()
        self.data = data.reset_index(drop=True)
        self.prices = self.data[self.cfg.price_column].values
        self.step_idx = self.cfg.lookback
        self.equity = self.cfg.initial_equity
        self.position = 0  # -1 short, 0 flat, 1 long
        self.action_space = spaces.Discrete(3)  # short, flat, long
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.lookback, len(self.data.columns)), dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        start = self.step_idx - self.cfg.lookback
        end = self.step_idx
        obs = self.data.iloc[start:end].select_dtypes(include=[np.number]).values
        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_price = self.prices[self.step_idx - 1]
        self.step_idx += 1
        terminated = self.step_idx >= len(self.prices) or (
            self.cfg.max_steps is not None and self.step_idx >= self.cfg.lookback + self.cfg.max_steps
        )

        reward = 0.0
        curr_price = self.prices[self.step_idx - 1]
        new_position = action - 1  # map 0->-1,1->0,2->1
        if new_position != self.position:
            reward -= self.cfg.transaction_cost
        pnl = (new_position * (curr_price - prev_price) / prev_price)
        reward += pnl
        self.position = new_position
        self.equity *= (1 + reward)

        obs = self._get_observation()
        info = {"equity": self.equity, "step": self.step_idx}
        return obs, reward, terminated, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_idx = self.cfg.lookback
        self.equity = self.cfg.initial_equity
        self.position = 0
        return self._get_observation(), {}

    def render(self):  # minimal stub
        return None


__all__ = ["TradingEnv", "TradingEnvConfig"]
