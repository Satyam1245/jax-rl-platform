"""Tiny Actor-Critic CartPole and checkpoint smoke test."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from rl_platform.agents import ActorCriticAgent
from rl_platform.algorithms import ActorCriticConfig
from rl_platform.environments import make_environment
from rl_platform.evaluation import EvaluationConfig, EvaluationEngine
from rl_platform.experiments import ExperimentManager
from rl_platform.training import TrainingConfig, TrainingEngine
from rl_platform.utils.random import create_prng_key


def test_actor_critic_cartpole_checkpoint_and_evaluation_smoke(tmp_path: Path) -> None:
    algorithm_config = ActorCriticConfig(
        learning_rate=0.01,
        gamma=0.99,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        hidden_layers=1,
        hidden_size=16,
        activation="tanh",
        optimizer="adam",
    )
    configuration = {
        "algorithm": "actor_critic",
        "environment": "CartPole-v1",
        "seed": 61,
        "actor_critic": asdict(algorithm_config),
    }
    manager = ExperimentManager(tmp_path / "experiments", tmp_path / "checkpoints", tmp_path / "results")
    experiment = manager.create_experiment(
        algorithm="actor_critic",
        environment="CartPole-v1",
        seed=61,
        configuration=configuration,
    )
    agent = ActorCriticAgent.create(4, 2, algorithm_config, create_prng_key(61))
    with make_environment("CartPole-v1") as environment:
        training = TrainingEngine(
            TrainingConfig(episodes=2, max_steps_per_episode=4, seed=61, moving_average_window=2)
        ).train(environment, agent)
        manager.persist_metrics(experiment, training.metrics)
        checkpoint = manager.save_checkpoint(
            experiment,
            agent,
            algorithm="actor_critic",
            environment="CartPole-v1",
            seed=61,
            configuration=configuration,
            training_steps=agent.training_steps,
            episodes_completed=training.episodes_completed,
            total_steps=training.total_steps,
        )
        loaded_agent = ActorCriticAgent.create(4, 2, algorithm_config, create_prng_key(999))
        metadata = manager.load_checkpoint(
            checkpoint,
            loaded_agent,
            algorithm="actor_critic",
            environment="CartPole-v1",
            seed=61,
            configuration=configuration,
        )
        evaluation = EvaluationEngine(
            EvaluationConfig(episodes=1, max_steps_per_episode=2, seed=62)
        ).evaluate(environment, loaded_agent)

    assert training.total_steps > 0
    assert agent.training_steps > 0
    assert len(training.metrics.episode_reward) == 2
    assert len(training.metrics.loss) == training.total_steps
    assert checkpoint.joinpath("metadata.json").is_file()
    assert checkpoint.joinpath("state.msgpack").is_file()
    assert metadata.training_steps == agent.training_steps
    assert evaluation.total_steps > 0
    assert len(evaluation.metrics.evaluation_reward) == 1
    assert loaded_agent.logits(np.zeros(4, dtype=np.float32)).shape == (2,)
    assert np.isfinite(np.asarray(loaded_agent.value(np.zeros(4, dtype=np.float32))))
