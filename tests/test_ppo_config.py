from training.ppo_runner import PPORunnerConfig


def test_ppo_runner_config_defaults():
    cfg = PPORunnerConfig()
    assert cfg.rollout_steps > 0
    assert cfg.episodes > 0
    assert cfg.device in {"cpu", "cuda", "mps"} or isinstance(cfg.device, str)
