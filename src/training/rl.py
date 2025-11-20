"""PPO-style reinforcement learning scaffold for trading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
from torch import nn


@dataclass
class PPOConfig:
    gamma: float = 0.99
    clip_ratio: float = 0.2
    lr: float = 3e-4
    epochs: int = 10
    steps_per_rollout: int = 2048
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    batch_size: int = 256


class ActorCritic(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # long/short logits
        )
        self.value = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy(feat), self.value(feat)


class PPOTrainger:
    def __init__(self, feature_fn: Callable[[torch.Tensor], torch.Tensor], config: PPOConfig | None = None) -> None:
        self.cfg = config or PPOConfig()
        self.feature_fn = feature_fn
        sample = torch.zeros(1, feature_fn(torch.zeros(1, 1)).shape[-1])
        feature_dim = sample.shape[-1]
        self.model = ActorCritic(feature_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def _compute_advantage(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * gae * (1 - dones[t])
            adv[t] = gae
        return adv

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_logp = batch["logp"]
        returns = batch["returns"]
        advantage = batch["advantage"]

        logits, values = self.model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ratio = torch.exp(logp - old_logp)

        clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * advantage
        policy_loss = -(torch.min(ratio * advantage, clipped)).mean()
        value_loss = (returns - values.squeeze(-1)).pow(2).mean()
        entropy = dist.entropy().mean()

        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }


__all__ = ["PPOConfig", "PPOTrainger", "ActorCritic"]
