"""Safe metadata and Flax state persistence for resumable experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from flax import serialization

from rl_platform.utils.logging import get_logger


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointableAgent(Protocol):
    """Structural interface for agents that can persist and restore state."""

    def checkpoint_state(self) -> dict[str, Any]: ...

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None: ...


class CheckpointError(ValueError):
    """Raised when checkpoint files are missing, malformed, or invalid."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when a checkpoint does not match the requested run context."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Identity and progress information required to validate a checkpoint."""

    experiment_id: str
    algorithm: str
    environment: str
    seed: int
    configuration: dict[str, Any]
    training_steps: int
    episodes_completed: int
    total_steps: int
    created_at: str
    format_version: int = CHECKPOINT_FORMAT_VERSION

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        algorithm: str,
        environment: str,
        seed: int,
        configuration: Any,
        training_steps: int,
        episodes_completed: int,
        total_steps: int,
    ) -> CheckpointMetadata:
        """Create metadata with a UTC timestamp and normalized configuration."""

        return cls(
            experiment_id=experiment_id,
            algorithm=algorithm,
            environment=environment,
            seed=seed,
            configuration=_json_safe(configuration),
            training_steps=training_steps,
            episodes_completed=episodes_completed,
            total_steps=total_steps,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def __post_init__(self) -> None:
        for text_value, text_field in (
            (self.experiment_id, "experiment_id"),
            (self.algorithm, "algorithm"),
            (self.environment, "environment"),
            (self.created_at, "created_at"),
        ):
            if not isinstance(text_value, str) or not text_value.strip():
                raise CheckpointError(f"Checkpoint {text_field} must be a non-empty string")
        for numeric_value, numeric_field in (
            (self.seed, "seed"),
            (self.training_steps, "training_steps"),
            (self.episodes_completed, "episodes_completed"),
            (self.total_steps, "total_steps"),
            (self.format_version, "format_version"),
        ):
            if isinstance(numeric_value, bool) or not isinstance(numeric_value, int) or numeric_value < 0:
                raise CheckpointError(f"Checkpoint {numeric_field} must be a non-negative integer")
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointError(
                f"Unsupported checkpoint format version: {self.format_version}"
            )
        if not isinstance(self.configuration, dict):
            raise CheckpointError("Checkpoint configuration must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CheckpointMetadata:
        """Parse and validate metadata read from JSON."""

        if not isinstance(values, Mapping):
            raise CheckpointError("Checkpoint metadata must be a mapping")
        required = {
            "experiment_id",
            "algorithm",
            "environment",
            "seed",
            "configuration",
            "training_steps",
            "episodes_completed",
            "total_steps",
            "created_at",
            "format_version",
        }
        missing = sorted(required - set(values))
        if missing:
            raise CheckpointError(f"Checkpoint metadata is missing: {', '.join(missing)}")
        try:
            return cls(**{key: values[key] for key in required})
        except TypeError as exc:
            raise CheckpointError(f"Invalid checkpoint metadata: {exc}") from exc


class CheckpointManager:
    """Persist and restore agent state below one checkpoint directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._logger = get_logger(__name__)

    def save(
        self,
        agent: CheckpointableAgent,
        metadata: CheckpointMetadata,
        *,
        checkpoint_name: str | None = None,
    ) -> Path:
        """Save metadata and serialized agent state, returning the checkpoint path."""

        name = checkpoint_name or f"step-{metadata.training_steps:08d}"
        _validate_checkpoint_name(name)
        checkpoint_path = self.directory / name
        if checkpoint_path.exists():
            raise CheckpointError(f"Checkpoint already exists: {checkpoint_path}")
        checkpoint_path.mkdir(parents=True, exist_ok=False)
        try:
            state = agent.checkpoint_state()
            state_bytes = serialization.msgpack_serialize(state)
            (checkpoint_path / "metadata.json").write_text(
                json.dumps(metadata.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
            (checkpoint_path / "state.msgpack").write_bytes(state_bytes)
        except Exception:
            _remove_checkpoint_directory(checkpoint_path)
            raise
        self._logger.info("Checkpoint saved: %s", checkpoint_path)
        return checkpoint_path

    def load(
        self,
        checkpoint_path: str | Path,
        agent: CheckpointableAgent,
        *,
        expected_algorithm: str | None = None,
        expected_environment: str | None = None,
        expected_seed: int | None = None,
        expected_configuration: Any | None = None,
    ) -> CheckpointMetadata:
        """Validate and restore a checkpoint into an initialized compatible agent."""

        path = Path(checkpoint_path)
        metadata_path = path / "metadata.json"
        state_path = path / "state.msgpack"
        if not path.is_dir() or not metadata_path.is_file() or not state_path.is_file():
            raise CheckpointError(f"Invalid checkpoint directory: {path}")
        try:
            metadata_values = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"Unable to read checkpoint metadata: {path}") from exc
        metadata = CheckpointMetadata.from_dict(metadata_values)
        _validate_compatibility(
            metadata,
            expected_algorithm=expected_algorithm,
            expected_environment=expected_environment,
            expected_seed=expected_seed,
            expected_configuration=expected_configuration,
        )
        try:
            state = serialization.msgpack_restore(state_path.read_bytes())
            agent.restore_checkpoint_state(state)
        except IncompatibleCheckpointError:
            raise
        except Exception as exc:
            raise IncompatibleCheckpointError(
                f"Checkpoint state is incompatible with the initialized agent: {path}"
            ) from exc
        self._logger.info("Checkpoint loaded: %s", path)
        return metadata


def _validate_compatibility(
    metadata: CheckpointMetadata,
    *,
    expected_algorithm: str | None,
    expected_environment: str | None,
    expected_seed: int | None,
    expected_configuration: Any | None,
) -> None:
    checks = (
        (expected_algorithm, metadata.algorithm, "algorithm"),
        (expected_environment, metadata.environment, "environment"),
        (expected_seed, metadata.seed, "seed"),
    )
    for expected, actual, field in checks:
        if expected is not None and expected != actual:
            raise IncompatibleCheckpointError(
                f"Checkpoint {field} mismatch: expected {expected!r}, found {actual!r}"
            )
    if expected_configuration is not None:
        if _json_safe(expected_configuration) != metadata.configuration:
            raise IncompatibleCheckpointError("Checkpoint configuration is incompatible")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Configuration value is not JSON-compatible: {type(value).__name__}")


def _validate_checkpoint_name(name: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise CheckpointError("checkpoint_name must be a simple directory name")


def _remove_checkpoint_directory(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
    path.rmdir()
