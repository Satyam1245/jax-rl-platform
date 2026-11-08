"""Integration tests for Phase 6 experiment and checkpoint management."""

from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from rl_platform.agents import DQNAgent
from rl_platform.algorithms import DQNConfig
from rl_platform.checkpoints import IncompatibleCheckpointError
from rl_platform.environments import make_environment
from rl_platform.evaluation import EvaluationConfig, EvaluationEngine
from rl_platform.experiments import ExperimentManager
from rl_platform.training import TrainingConfig, TrainingEngine
from rl_platform.utils.random import create_prng_key


def _dqn_config() -> DQNConfig:
    return DQNConfig(
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
        hidden_size=8,
        activation="relu",
        optimizer="adam",
    )


def _configuration(config: DQNConfig) -> dict[str, object]:
    return {"algorithm": "dqn", "environment": "CartPole-v1", "dqn": asdict(config)}


def test_dqn_checkpoint_load_evaluate_and_resume(tmp_path: Path) -> None:
    dqn_config = _dqn_config()
    configuration = _configuration(dqn_config)
    manager = ExperimentManager(
        tmp_path / "experiments",
        tmp_path / "checkpoints",
        tmp_path / "results",
    )
    experiment = manager.create_experiment(
        algorithm="dqn",
        environment="CartPole-v1",
        seed=41,
        configuration=configuration,
    )
    assert experiment.experiment_id
    assert experiment.configuration_path.is_file()
    assert experiment.checkpoint_directory.is_dir()
    assert experiment.result_directory.is_dir()
    assert yaml.safe_load(experiment.configuration_path.read_text(encoding="utf-8")) == configuration

    agent = DQNAgent.create(4, 2, dqn_config, create_prng_key(41))
    training_config = TrainingConfig(episodes=1, max_steps_per_episode=4, seed=41, moving_average_window=2)
    with make_environment("CartPole-v1") as environment:
        training = TrainingEngine(training_config).train(environment, agent)
        metrics_path = manager.persist_metrics(experiment, training.metrics)
        checkpoint = manager.save_checkpoint(
            experiment,
            agent,
            algorithm="dqn",
            environment="CartPole-v1",
            seed=41,
            configuration=configuration,
            training_steps=agent.training_steps,
            episodes_completed=training.episodes_completed,
            total_steps=training.total_steps,
        )

        loaded_agent = DQNAgent.create(4, 2, dqn_config, create_prng_key(999))
        metadata = manager.load_checkpoint(
            checkpoint,
            loaded_agent,
            algorithm="dqn",
            environment="CartPole-v1",
            seed=41,
            configuration=configuration,
        )
        evaluation = EvaluationEngine(
            EvaluationConfig(episodes=1, max_steps_per_episode=2, seed=42)
        ).evaluate(environment, loaded_agent)
        resumed = TrainingEngine(
            TrainingConfig(episodes=1, max_steps_per_episode=3, seed=41, moving_average_window=2)
        ).train(
            environment,
            loaded_agent,
            start_episode=metadata.episodes_completed,
            initial_total_steps=metadata.total_steps,
        )

    restored_metrics = manager.load_metrics(experiment)
    assert metrics_path.is_file()
    assert restored_metrics.episode_reward == training.metrics.episode_reward
    assert metadata.training_steps == agent.training_steps
    assert evaluation.total_steps > 0
    assert len(evaluation.metrics.evaluation_reward) == 1
    q_values = loaded_agent.q_values(np.zeros(4, dtype=np.float32))
    assert q_values.shape == (2,)
    assert np.isfinite(np.asarray(q_values)).all()
    assert resumed.episodes_completed == metadata.episodes_completed + 1
    assert resumed.total_steps > metadata.total_steps
    assert loaded_agent.training_steps > metadata.training_steps
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(loaded_agent.parameters)
    )


def test_checkpoint_rejects_incompatible_context(tmp_path: Path) -> None:
    dqn_config = _dqn_config()
    configuration = _configuration(dqn_config)
    manager = ExperimentManager(tmp_path / "experiments", tmp_path / "checkpoints", tmp_path / "results")
    experiment = manager.create_experiment(
        algorithm="dqn",
        environment="CartPole-v1",
        seed=5,
        configuration=configuration,
    )
    agent = DQNAgent.create(4, 2, dqn_config, create_prng_key(5))
    checkpoint = manager.save_checkpoint(
        experiment,
        agent,
        algorithm="dqn",
        environment="CartPole-v1",
        seed=5,
        configuration=configuration,
        training_steps=0,
        episodes_completed=0,
        total_steps=0,
    )

    with pytest.raises(IncompatibleCheckpointError, match="environment mismatch"):
        manager.load_checkpoint(
            checkpoint,
            DQNAgent.create(4, 2, dqn_config, create_prng_key(8)),
            algorithm="dqn",
            environment="MountainCar-v0",
            seed=5,
            configuration=configuration,
        )
