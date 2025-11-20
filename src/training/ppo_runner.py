"""PPO rollout + update loop wired to TradingEnv."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

from .env import TradingEnv, TradingEnvConfig
from .rl import PPOConfig, PPOTrainger


@dataclass
class PPORunnerConfig:
    rollout_steps: int = 2048
    update_epochs: int = 5
    device: str = "cpu"
    episodes: int = 3
    checkpoint_dir: Path = Path("checkpoints")


def _rollout(env: TradingEnv, agent: PPOTrainger, feature_fn: Callable[[torch.Tensor], torch.Tensor], cfg: PPORunnerConfig) -> Dict[str, torch.Tensor]:
    device = torch.device(cfg.device)
    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    values_list, logps, actions_list, rewards, dones, feats_list = [], [], [], [], [], []

    for _ in range(cfg.rollout_steps):
        feat = feature_fn(obs_t)
        logits, value = agent.model(feat)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        next_obs, reward, terminated, _, _ = env.step(action.item())
        rewards.append(torch.tensor(reward, device=device))
        dones.append(torch.tensor(float(terminated), device=device))
        values_list.append(value.squeeze(-1))
        logps.append(logp)
        actions_list.append(action.squeeze(-1))
        feats_list.append(feat.squeeze(0))

        if terminated:
            break
        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        final_feat = feature_fn(obs_t)
        _, final_value = agent.model(final_feat)

    rewards_t = torch.stack(rewards)
    dones_t = torch.stack(dones)
    values_t = torch.cat(values_list + [final_value.squeeze(-1)])
    adv = agent._compute_advantage(rewards_t, values_t, dones_t)
    returns_t = adv + values_t[:-1]

    batch = {
        "obs": torch.stack(feats_list),
        "actions": torch.stack(actions_list),
        "logp": torch.stack(logps),
        "returns": returns_t.detach(),
        "advantage": adv.detach(),
    }
    batch["episode_return"] = torch.tensor(env.equity - 1.0)
    return batch


def run_ppo_episode(env_data, feature_fn: Callable[[torch.Tensor], torch.Tensor], runner_cfg: PPORunnerConfig | None = None, agent: Optional[PPOTrainger] = None) -> Dict[str, float]:
    cfg = runner_cfg or PPORunnerConfig()
    env = TradingEnv(env_data, TradingEnvConfig())
    ppo_cfg = PPOConfig(steps_per_rollout=cfg.rollout_steps)
    agent = agent or PPOTrainger(feature_fn, config=ppo_cfg)
    device = torch.device(cfg.device)
    agent.model.to(device)

    batch = _rollout(env, agent, feature_fn, cfg)
    metrics = agent.update(batch)
    metrics["episode_return"] = float(batch["episode_return"].item())
    return metrics


def train_ppo(env_data, feature_fn: Callable[[torch.Tensor], torch.Tensor], runner_cfg: PPORunnerConfig | None = None) -> Dict[str, float]:
    cfg = runner_cfg or PPORunnerConfig()
    env = TradingEnv(env_data, TradingEnvConfig())
    ppo_cfg = PPOConfig(steps_per_rollout=cfg.rollout_steps)
    agent = PPOTrainger(feature_fn, config=ppo_cfg)
    device = torch.device(cfg.device)
    agent.model.to(device)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_return = -1e9
    last_metrics: Dict[str, float] = {}
    for ep in range(cfg.episodes):
        batch = _rollout(env, agent, feature_fn, cfg)
        metrics = agent.update(batch)
        ep_return = float(batch["episode_return"].item())
        metrics["episode_return"] = ep_return
        last_metrics = metrics
        if ep_return > best_return:
            best_return = ep_return
            ckpt = cfg.checkpoint_dir / "ppo_best.pt"
            torch.save({"model": agent.model.state_dict(), "optimizer": agent.optimizer.state_dict(), "return": ep_return}, ckpt)
        env.reset()
    return {"metrics": last_metrics, "agent": agent}


def ppo_policy_signals(env_data, agent: PPOTrainger, feature_fn: Callable[[torch.Tensor], torch.Tensor], cfg: PPORunnerConfig | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate PPO policy signals on a dataset: returns probabilities and discrete actions."""

    cfg = cfg or PPORunnerConfig()
    env = TradingEnv(env_data, TradingEnvConfig())
    device = torch.device(cfg.device)
    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    probs = []
    actions = []
    for _ in range(len(env_data) - env.cfg.lookback):
        feat = feature_fn(obs_t)
        logits, _ = agent.model(feat)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        probs.append(torch.softmax(logits, dim=-1).squeeze(0).detach().cpu())
        actions.append(action.detach().cpu())
        next_obs, _, terminated, _, _ = env.step(action.item())
        if terminated:
            break
        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
    return torch.stack(probs), torch.stack(actions)


__all__ = ["run_ppo_episode", "PPORunnerConfig"]
