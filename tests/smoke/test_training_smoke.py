"""Short real-environment DQN training and evaluation smoke test."""

import numpy as np

from rl_platform.agents import DQNAgent
from rl_platform.algorithms import DQNConfig
from rl_platform.environments import make_environment
from rl_platform.evaluation import EvaluationConfig, EvaluationEngine
from rl_platform.training import TrainingConfig, TrainingEngine
from rl_platform.utils.random import create_prng_key


def test_dqn_cartpole_training_and_evaluation_smoke() -> None:
    dqn_config = DQNConfig(
        learning_rate=0.01,
        gamma=0.99,
        batch_size=2,
        replay_capacity=32,
        min_replay_size=2,
        target_update_interval=2,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=10,
        hidden_layers=1,
        hidden_size=16,
        activation="relu",
        optimizer="adam",
    )
    agent = DQNAgent.create(4, 2, dqn_config, create_prng_key(100))
    training_config = TrainingConfig(
        episodes=2,
        max_steps_per_episode=4,
        seed=101,
        moving_average_window=2,
    )

    with make_environment("CartPole-v1") as environment:
        training = TrainingEngine(training_config).train(environment, agent)
        steps_after_training = agent.training_steps
        replay_after_training = len(agent.replay_buffer)
        evaluation = EvaluationEngine(
            EvaluationConfig(episodes=2, max_steps_per_episode=3, seed=202)
        ).evaluate(environment, agent)

    assert training.total_steps > 0
    assert len(training.metrics.episode_reward) == 2
    assert len(training.metrics.episode_length) == 2
    assert len(training.metrics.loss) > 0
    assert all(np.isfinite(loss) for loss in training.metrics.loss)
    assert evaluation.total_steps > 0
    assert len(evaluation.metrics.evaluation_reward) == 2
    assert agent.training_steps == steps_after_training
    assert len(agent.replay_buffer) == replay_after_training
