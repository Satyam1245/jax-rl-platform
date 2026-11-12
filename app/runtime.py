"""Application-facing orchestration over the tested RL backend services."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np

from rl_platform.agents import ActorCriticAgent, DQNAgent, PPOAgent
from rl_platform.algorithms import ActorCriticConfig, DQNConfig, PPOConfig
from rl_platform.checkpoints import CheckpointManager, CheckpointMetadata
from rl_platform.configuration import load_config
from rl_platform.environments import make_environment
from rl_platform.evaluation import EvaluationConfig, EvaluationEngine, EvaluationResult
from rl_platform.experiments import ExperimentManager, ExperimentPaths
from rl_platform.training import TrainingConfig, TrainingEngine, TrainingResult
from rl_platform.utils.random import create_prng_key
from rl_platform.visualization import ExperimentResult, load_experiment_results


_CONFIG_FILES = {
    "dqn": "default.yaml",
    "ppo": "ppo.yaml",
    "actor_critic": "actor_critic.yaml",
}


@dataclass(frozen=True, slots=True)
class TrainingExecution:
    """Backend output needed by the UI after a training request."""

    experiment: ExperimentPaths
    result: TrainingResult
    algorithm: str
    environment: str
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    """Display-safe checkpoint metadata summary."""

    path: Path
    metadata: CheckpointMetadata


def available_algorithms() -> tuple[str, ...]:
    """Return the algorithms wired into the current backend."""

    return tuple(_CONFIG_FILES)


def load_algorithm_template(project_root: str | Path, algorithm: str) -> dict[str, Any]:
    """Load UI defaults from the existing validated algorithm configuration."""

    normalized = algorithm.strip().lower()
    try:
        filename = _CONFIG_FILES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported algorithm: {algorithm}") from exc
    config = load_config(Path(project_root) / "configs" / filename)
    return dict(config.algorithm.hyperparameters)


def train_experiment(
    *,
    project_root: str | Path,
    environment: str,
    algorithm: str,
    hyperparameters: dict[str, Any],
    seed: int,
    episodes: int,
    max_steps_per_episode: int,
    moving_average_window: int,
) -> TrainingExecution:
    """Create an experiment, train through the generic engine, and persist results."""

    root = Path(project_root)
    normalized_algorithm = algorithm.strip().lower()
    agent, typed_config = _create_agent(environment, normalized_algorithm, hyperparameters, seed)
    training_config = TrainingConfig(
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
        moving_average_window=moving_average_window,
    )
    configuration = {
        "algorithm": normalized_algorithm,
        "environment": environment,
        "seed": seed,
        "hyperparameters": asdict(typed_config),
        "training": asdict(training_config),
    }
    manager = ExperimentManager(root / "experiments", root / "checkpoints", root / "results")
    experiment = manager.create_experiment(
        algorithm=normalized_algorithm,
        environment=environment,
        seed=seed,
        configuration=configuration,
    )
    with make_environment(environment, seed=seed) as env:
        result = TrainingEngine(training_config).train(env, agent)
    manager.persist_metrics(
        experiment,
        result.metrics,
        training_duration_seconds=result.duration_seconds,
        total_steps=result.total_steps,
    )
    checkpoint_path = manager.save_checkpoint(
        experiment,
        agent,
        algorithm=normalized_algorithm,
        environment=environment,
        seed=seed,
        configuration=configuration,
        training_steps=getattr(agent, "training_steps", 0),
        episodes_completed=result.episodes_completed,
        total_steps=result.total_steps,
    )
    return TrainingExecution(
        experiment=experiment,
        result=result,
        algorithm=normalized_algorithm,
        environment=environment,
        checkpoint_path=checkpoint_path,
    )


def list_checkpoint_summaries(checkpoints_directory: str | Path) -> tuple[CheckpointSummary, ...]:
    """Return valid saved checkpoint metadata for the Checkpoints page."""

    root = Path(checkpoints_directory)
    if not root.exists():
        return ()
    if not root.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {root}")
    summaries: list[CheckpointSummary] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        try:
            values = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = CheckpointMetadata.from_dict(values)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        state_path = metadata_path.parent / "state.msgpack"
        if state_path.is_file():
            summaries.append(CheckpointSummary(metadata_path.parent, metadata))
    return tuple(summaries)


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    episodes: int,
    max_steps_per_episode: int,
) -> tuple[CheckpointMetadata, EvaluationResult]:
    """Restore a checkpoint and evaluate it deterministically through the backend."""

    path = Path(checkpoint_path)
    metadata_values = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    metadata = CheckpointMetadata.from_dict(metadata_values)
    configuration = metadata.configuration
    try:
        environment = str(configuration["environment"])
        algorithm = str(configuration["algorithm"])
        hyperparameters = dict(configuration["hyperparameters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint configuration cannot construct an evaluation agent") from exc
    agent, _ = _create_agent(environment, algorithm, hyperparameters, metadata.seed)
    CheckpointManager(path.parent).load(
        path,
        agent,
        expected_algorithm=algorithm,
        expected_environment=environment,
        expected_seed=metadata.seed,
        expected_configuration=configuration,
    )
    evaluation_config = EvaluationConfig(
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=metadata.seed,
    )
    with make_environment(environment, seed=metadata.seed) as env:
        result = EvaluationEngine(evaluation_config).evaluate(env, agent)
    return metadata, result


def load_stored_results(
    project_root: str | Path,
) -> tuple[ExperimentResult, ...]:
    """Load validated persisted results from the configured project directories."""

    root = Path(project_root)
    return load_experiment_results(root / "experiments", root / "results")


def _create_agent(
    environment: str,
    algorithm: str,
    hyperparameters: dict[str, Any],
    seed: int,
) -> tuple[Any, Any]:
    with make_environment(environment) as env:
        observation_shape = getattr(env.observation_space, "shape", None)
        action_count = getattr(env.action_space, "n", None)
    if observation_shape is None or action_count is None:
        raise ValueError("The current agent backends require vector observations and discrete actions")
    observation_size = int(np.prod(observation_shape))
    key = create_prng_key(seed)
    if algorithm == "dqn":
        dqn_config = DQNConfig.from_mapping(hyperparameters)
        return DQNAgent.create(observation_size, int(action_count), dqn_config, key), dqn_config
    if algorithm == "ppo":
        ppo_config = PPOConfig.from_mapping(hyperparameters)
        return PPOAgent.create(observation_size, int(action_count), ppo_config, key), ppo_config
    if algorithm == "actor_critic":
        actor_critic_config = ActorCriticConfig.from_mapping(hyperparameters)
        return (
            ActorCriticAgent.create(observation_size, int(action_count), actor_critic_config, key),
            actor_critic_config,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")
