"""Experiment IDs, artifact directories, and persisted run artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import uuid
from typing import Any

import yaml

from rl_platform.checkpoints import CheckpointManager, CheckpointMetadata
from rl_platform.metrics import MetricsRecorder
from rl_platform.utils.logging import get_logger


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    """Artifact paths belonging to one experiment ID."""

    experiment_id: str
    experiment_directory: Path
    checkpoint_directory: Path
    result_directory: Path
    configuration_path: Path
    experiment_metadata_path: Path
    metrics_path: Path


class ExperimentManager:
    """Create experiments and delegate checkpoint persistence safely."""

    def __init__(
        self,
        experiments_directory: str | Path,
        checkpoints_directory: str | Path,
        results_directory: str | Path,
    ) -> None:
        self.experiments_directory = Path(experiments_directory)
        self.checkpoints_directory = Path(checkpoints_directory)
        self.results_directory = Path(results_directory)
        self._logger = get_logger(__name__)

    def create_experiment(
        self,
        *,
        algorithm: str,
        environment: str,
        seed: int,
        configuration: Any,
    ) -> ExperimentPaths:
        """Create uniquely identified experiment, checkpoint, and result directories."""

        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError("algorithm must be a non-empty string")
        if not isinstance(environment, str) or not environment.strip():
            raise ValueError("environment must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        experiment_id = _new_experiment_id()
        experiment_directory = self.experiments_directory / experiment_id
        checkpoint_directory = self.checkpoints_directory / experiment_id
        result_directory = self.results_directory / experiment_id
        experiment_directory.mkdir(parents=True, exist_ok=False)
        checkpoint_directory.mkdir(parents=True, exist_ok=False)
        result_directory.mkdir(parents=True, exist_ok=False)
        paths = ExperimentPaths(
            experiment_id=experiment_id,
            experiment_directory=experiment_directory,
            checkpoint_directory=checkpoint_directory,
            result_directory=result_directory,
            configuration_path=experiment_directory / "configuration.yaml",
            experiment_metadata_path=experiment_directory / "metadata.json",
            metrics_path=result_directory / "metrics.json",
        )
        normalized_configuration = _json_safe(configuration)
        paths.configuration_path.write_text(
            yaml.safe_dump(normalized_configuration, sort_keys=False), encoding="utf-8"
        )
        paths.experiment_metadata_path.write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "algorithm": algorithm,
                    "environment": environment,
                    "seed": seed,
                    "configuration": normalized_configuration,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._logger.info("Experiment created: %s", experiment_id)
        return paths

    def persist_metrics(
        self,
        experiment: ExperimentPaths,
        metrics: MetricsRecorder,
        *,
        training_duration_seconds: float | None = None,
        total_steps: int | None = None,
    ) -> Path:
        """Persist metrics and optional measured training metadata as JSON."""

        values = metrics.to_dict()
        if training_duration_seconds is not None:
            if not isfinite(float(training_duration_seconds)) or training_duration_seconds < 0:
                raise ValueError("training_duration_seconds must be finite and non-negative")
            values["training_duration_seconds"] = float(training_duration_seconds)
        if total_steps is not None:
            if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 0:
                raise ValueError("total_steps must be a non-negative integer")
            values["total_steps"] = total_steps
        experiment.metrics_path.write_text(
            json.dumps(values, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._logger.info("Metrics persisted: %s", experiment.metrics_path)
        return experiment.metrics_path

    def load_metrics(self, experiment: ExperimentPaths) -> MetricsRecorder:
        """Load persisted metrics for an experiment."""

        try:
            values = json.loads(experiment.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read experiment metrics: {experiment.metrics_path}") from exc
        return MetricsRecorder.from_dict(values)

    def save_checkpoint(
        self,
        experiment: ExperimentPaths,
        agent: Any,
        *,
        algorithm: str,
        environment: str,
        seed: int,
        configuration: Any,
        training_steps: int,
        episodes_completed: int,
        total_steps: int,
        checkpoint_name: str | None = None,
    ) -> Path:
        """Create metadata and save one agent checkpoint for an experiment."""

        metadata = CheckpointMetadata.create(
            experiment_id=experiment.experiment_id,
            algorithm=algorithm,
            environment=environment,
            seed=seed,
            configuration=configuration,
            training_steps=training_steps,
            episodes_completed=episodes_completed,
            total_steps=total_steps,
        )
        return CheckpointManager(experiment.checkpoint_directory).save(
            agent, metadata, checkpoint_name=checkpoint_name
        )

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        agent: Any,
        *,
        algorithm: str,
        environment: str,
        seed: int,
        configuration: Any,
    ) -> CheckpointMetadata:
        """Validate and restore an experiment checkpoint into an agent."""

        return CheckpointManager(Path(checkpoint_path).parent).load(
            checkpoint_path,
            agent,
            expected_algorithm=algorithm,
            expected_environment=environment,
            expected_seed=seed,
            expected_configuration=configuration,
        )


def _new_experiment_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Configuration value is not serializable: {type(value).__name__}")
