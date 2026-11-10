"""Validation and loading of persisted experiment results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

from rl_platform.metrics import MetricsRecorder


class ResultValidationError(ValueError):
    """Raised when stored experiment metadata or metrics are malformed."""


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Validated data loaded from one experiment's stored artifacts."""

    experiment_id: str
    algorithm: str
    environment: str
    seed: int
    configuration: dict[str, Any]
    created_at: str
    metrics: MetricsRecorder
    training_duration_seconds: float | None
    total_steps: int | None

    @property
    def episode_rewards(self) -> tuple[float, ...]:
        """Return recorded training episode rewards."""

        return self.metrics.episode_reward

    @property
    def moving_average_rewards(self) -> tuple[float, ...]:
        """Return recorded moving-average rewards."""

        return self.metrics.moving_average_reward

    @property
    def episode_lengths(self) -> tuple[int, ...]:
        """Return recorded training episode lengths."""

        return self.metrics.episode_length

    @property
    def losses(self) -> tuple[float, ...]:
        """Return recorded optimizer losses."""

        return self.metrics.loss

    @property
    def evaluation_rewards(self) -> tuple[float, ...]:
        """Return recorded evaluation rewards."""

        return self.metrics.evaluation_reward


def load_experiment_result(
    experiment_directory: str | Path,
    results_directory: str | Path,
) -> ExperimentResult:
    """Load and validate one experiment metadata/metrics pair."""

    experiment_path = Path(experiment_directory)
    result_path = Path(results_directory)
    metadata = _read_json(experiment_path / "metadata.json", "experiment metadata")
    required_metadata = {"experiment_id", "algorithm", "environment", "seed", "configuration", "created_at"}
    _require_fields(metadata, required_metadata, "experiment metadata")
    if not all(isinstance(metadata[field], str) and metadata[field].strip() for field in ("experiment_id", "algorithm", "environment", "created_at")):
        raise ResultValidationError("Experiment metadata identity fields must be non-empty strings")
    if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int) or metadata["seed"] < 0:
        raise ResultValidationError("Experiment metadata seed must be a non-negative integer")
    if not isinstance(metadata["configuration"], dict):
        raise ResultValidationError("Experiment metadata configuration must be a mapping")

    metrics_values = _read_json(result_path / "metrics.json", "experiment metrics")
    _require_fields(metrics_values, {"moving_average_window", "training", "evaluation", "loss"}, "experiment metrics")
    try:
        metrics = MetricsRecorder.from_dict(metrics_values)
    except (TypeError, ValueError, KeyError) as exc:
        raise ResultValidationError("Experiment metrics contain invalid records") from exc
    duration = metrics_values.get("training_duration_seconds")
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not isfinite(float(duration)) or duration < 0:
            raise ResultValidationError("training_duration_seconds must be finite and non-negative")
        duration = float(duration)
    total_steps = metrics_values.get("total_steps")
    if total_steps is not None:
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 0:
            raise ResultValidationError("total_steps must be a non-negative integer")
    return ExperimentResult(
        experiment_id=metadata["experiment_id"],
        algorithm=metadata["algorithm"],
        environment=metadata["environment"],
        seed=metadata["seed"],
        configuration=metadata["configuration"],
        created_at=metadata["created_at"],
        metrics=metrics,
        training_duration_seconds=duration,
        total_steps=total_steps,
    )


def load_experiment_results(
    experiments_directory: str | Path,
    results_directory: str | Path,
) -> tuple[ExperimentResult, ...]:
    """Load every stored experiment result, raising on corrupt entries.

    Raises ``ResultValidationError`` when any experiment's metadata or
    metrics files are missing or malformed.  Only unexpected OS-level
    filesystem errors on individual directory entries are silently skipped
    so that one unreadable directory does not prevent others from loading.
    """

    root = Path(experiments_directory)
    if not root.exists():
        return ()
    if not root.is_dir():
        raise ResultValidationError(f"Experiments path is not a directory: {root}")
    experiment_directories = sorted(path for path in root.iterdir() if path.is_dir())
    results: list[ExperimentResult] = []
    for path in experiment_directories:
        try:
            results.append(
                load_experiment_result(path, Path(results_directory) / path.name)
            )
        except OSError:
            # Skip experiment directories that cannot be read at the OS level.
            continue
    return tuple(results)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ResultValidationError(f"Missing {description}: {path}")
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"Malformed {description}: {path}") from exc
    if not isinstance(values, dict):
        raise ResultValidationError(f"{description} must contain a mapping: {path}")
    return values


def _require_fields(values: Mapping[str, Any], fields: Iterable[str], description: str) -> None:
    missing = sorted(set(fields) - set(values))
    if missing:
        raise ResultValidationError(f"{description} is missing field(s): {', '.join(missing)}")
