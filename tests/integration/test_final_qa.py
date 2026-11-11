"""Final-QA integration coverage for the production backend workflow."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from rl_platform.algorithms import ActorCriticConfig, DQNConfig, PPOConfig
from rl_platform.checkpoints import CheckpointError, IncompatibleCheckpointError, CheckpointManager
from rl_platform.configuration import ConfigurationError, validate_config
from rl_platform.environments import make_environment
from rl_platform.experiments import ExperimentManager
from rl_platform.services import (
    evaluate_checkpoint,
    load_algorithm_template,
    load_stored_results,
    train_experiment,
)
from rl_platform.visualization import ResultValidationError, load_experiment_results


PROJECT_ROOT = Path(__file__).parents[2]


def _small_hyperparameters(algorithm: str) -> dict[str, object]:
    values = load_algorithm_template(PROJECT_ROOT, algorithm)
    values.update({"hidden_layers": 1, "hidden_size": 8})
    if algorithm == "dqn":
        values.update(
            {
                "batch_size": 2,
                "replay_capacity": 8,
                "min_replay_size": 2,
                "target_update_interval": 2,
                "epsilon_decay_steps": 4,
            }
        )
    elif algorithm == "ppo":
        values.update({"rollout_length": 2, "minibatch_size": 2, "epochs": 1})
    return values


@pytest.mark.parametrize("algorithm", ["dqn", "ppo", "actor_critic"])
def test_backend_workflow_trains_checkpoints_and_evaluates(
    tmp_path: Path, algorithm: str
) -> None:
    execution = train_experiment(
        project_root=tmp_path,
        environment="CartPole-v1",
        algorithm=algorithm,
        hyperparameters=_small_hyperparameters(algorithm),
        seed=17,
        episodes=1,
        max_steps_per_episode=2,
        moving_average_window=2,
    )

    stored = load_stored_results(tmp_path)
    metadata, evaluation = evaluate_checkpoint(
        checkpoint_path=execution.checkpoint_path,
        episodes=2,
        max_steps_per_episode=2,
    )

    assert execution.result.total_steps > 0
    assert execution.result.metrics.episode_reward
    assert execution.checkpoint_path.joinpath("metadata.json").is_file()
    assert execution.checkpoint_path.joinpath("state.msgpack").is_file()
    assert len(stored) == 1
    assert stored[0].algorithm == algorithm
    assert metadata.algorithm == algorithm
    assert len(evaluation.metrics.evaluation_reward) == 2
    assert np.isfinite(evaluation.metrics.evaluation_reward).all()


def test_invalid_inputs_and_corrupt_artifacts_are_rejected(tmp_path: Path) -> None:
    raw_config = yaml.safe_load((PROJECT_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    raw_config["environment"]["name"] = "Unknown-v1"
    with pytest.raises(ConfigurationError, match="Unsupported environment"):
        validate_config(raw_config)
    with pytest.raises(ValueError, match="Unsupported environment"):
        make_environment("Unknown-v1")
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        train_experiment(
            project_root=tmp_path,
            environment="CartPole-v1",
            algorithm="unknown",
            hyperparameters={},
            seed=1,
            episodes=1,
            max_steps_per_episode=1,
            moving_average_window=1,
        )
    with pytest.raises(ValueError):
        DQNConfig.from_mapping({"gamma": 1.2})
    with pytest.raises(ValueError):
        PPOConfig.from_mapping({"clip_epsilon": -0.1})
    with pytest.raises(ValueError):
        ActorCriticConfig.from_mapping({"gamma": -0.1})

    with pytest.raises(CheckpointError, match="Invalid checkpoint directory"):
        CheckpointManager(tmp_path / "missing").load(
            tmp_path / "missing" / "step-1", object()
        )

    execution = train_experiment(
        project_root=tmp_path,
        environment="CartPole-v1",
        algorithm="dqn",
        hyperparameters=_small_hyperparameters("dqn"),
        seed=3,
        episodes=1,
        max_steps_per_episode=2,
        moving_average_window=1,
    )
    execution.checkpoint_path.joinpath("state.msgpack").write_bytes(b"corrupt")
    with pytest.raises(IncompatibleCheckpointError, match="incompatible"):
        evaluate_checkpoint(
            checkpoint_path=execution.checkpoint_path,
            episodes=1,
            max_steps_per_episode=1,
        )

    broken_manager = ExperimentManager(
        tmp_path / "broken-experiments",
        tmp_path / "broken-checkpoints",
        tmp_path / "broken-results",
    )
    broken_manager.create_experiment(
        algorithm="dqn",
        environment="CartPole-v1",
        seed=1,
        configuration={"test": True},
    )
    with pytest.raises(ResultValidationError, match="Missing experiment metrics"):
        load_experiment_results(tmp_path / "broken-experiments", tmp_path / "broken-results")


def test_same_seed_produces_consistent_tiny_dqn_results(tmp_path: Path) -> None:
    roots = [tmp_path / "run-a", tmp_path / "run-b"]
    results = []
    for root in roots:
        execution = train_experiment(
            project_root=root,
            environment="CartPole-v1",
            algorithm="dqn",
            hyperparameters=_small_hyperparameters("dqn"),
            seed=23,
            episodes=2,
            max_steps_per_episode=2,
            moving_average_window=2,
        )
        results.append(execution.result.metrics)

    assert results[0].episode_reward == results[1].episode_reward
    assert results[0].episode_length == results[1].episode_length
    assert np.allclose(results[0].loss, results[1].loss)
