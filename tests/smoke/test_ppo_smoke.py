"""Tiny real-environment PPO training smoke test."""

import numpy as np

from rl_platform.agents import PPOAgent
from rl_platform.algorithms import PPOConfig
from rl_platform.environments import make_environment
from rl_platform.training import TrainingConfig, TrainingEngine
from rl_platform.utils.random import create_prng_key


def test_ppo_cartpole_training_smoke() -> None:
    config = PPOConfig(
        learning_rate=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        rollout_length=4,
        minibatch_size=2,
        epochs=2,
        hidden_layers=1,
        hidden_size=16,
        activation="tanh",
        optimizer="adam",
    )
    agent = PPOAgent.create(4, 2, config, create_prng_key(50))
    with make_environment("CartPole-v1") as environment:
        result = TrainingEngine(
            TrainingConfig(episodes=2, max_steps_per_episode=4, seed=51, moving_average_window=2)
        ).train(environment, agent)

    assert result.total_steps > 0
    assert len(result.metrics.episode_reward) == 2
    assert len(result.metrics.loss) > 0
    assert all(np.isfinite(loss) for loss in result.metrics.loss)
    assert agent.training_steps > 0
