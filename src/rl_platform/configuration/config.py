"""YAML configuration loading and validation for the platform foundation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_ALGORITHMS = frozenset({"dqn", "ppo", "actor_critic"})
SUPPORTED_ENVIRONMENTS = frozenset({"CartPole-v1", "MountainCar-v0"})
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigurationError(ValueError):
    """Raised when a configuration file or mapping fails validation."""


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Environment settings that are independent of any algorithm."""

    name: str


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    """Algorithm identity and opaque algorithm-specific hyperparameters."""

    name: str
    hyperparameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Reproducibility and run-size settings."""

    seed: int
    episodes: int
    max_steps_per_episode: int
    evaluation_episodes: int
    moving_average_window: int = 100


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Locations for future experiment artifacts."""

    experiments: Path
    checkpoints: Path
    results: Path


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging settings."""

    level: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated platform configuration."""

    schema_version: int
    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    run: RunConfig
    paths: PathsConfig
    logging: LoggingConfig


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    The function performs no filesystem writes. Artifact directories are
    created by the future experiment/checkpoint layers when they are needed.
    """

    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("Configuration file must use .yaml or .yml: " f"{path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration file {path}: {exc}") from exc

    return validate_config(raw_config, source=path)


def validate_config(
    raw_config: Mapping[str, Any] | None,
    *,
    source: str | Path = "configuration",
) -> AppConfig:
    """Validate a raw configuration mapping and return typed settings."""

    if not isinstance(raw_config, Mapping):
        raise ConfigurationError(f"{source} must contain a top-level mapping")

    _check_keys(raw_config, {"schema_version", "environment", "algorithm", "run", "paths", "logging"}, source)

    schema_version = raw_config.get("schema_version", 1)
    _require_int(schema_version, "schema_version", minimum=1)

    environment_data = _require_mapping(raw_config, "environment")
    _check_keys(environment_data, {"name"}, "environment")
    environment_name = _require_text(environment_data.get("name"), "environment.name")
    if environment_name not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
        raise ConfigurationError(
            f"Unsupported environment {environment_name!r}; supported environments: {supported}"
        )

    algorithm_data = _require_mapping(raw_config, "algorithm")
    _check_keys(algorithm_data, {"name", "hyperparameters"}, "algorithm")
    algorithm_name = _require_text(algorithm_data.get("name"), "algorithm.name").lower()
    if algorithm_name not in SUPPORTED_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ConfigurationError(
            f"Unsupported algorithm {algorithm_name!r}; supported algorithms: {supported}"
        )
    hyperparameters = algorithm_data.get("hyperparameters", {})
    if not isinstance(hyperparameters, Mapping):
        raise ConfigurationError("algorithm.hyperparameters must be a mapping")

    run_data = _require_mapping(raw_config, "run")
    _check_keys(
        run_data,
        {"seed", "episodes", "max_steps_per_episode", "evaluation_episodes", "moving_average_window"},
        "run",
    )
    seed = _require_int(run_data.get("seed"), "run.seed", minimum=0)
    episodes = _require_int(run_data.get("episodes"), "run.episodes", minimum=1)
    max_steps = _require_int(
        run_data.get("max_steps_per_episode"), "run.max_steps_per_episode", minimum=1
    )
    evaluation_episodes = _require_int(
        run_data.get("evaluation_episodes"), "run.evaluation_episodes", minimum=1
    )
    moving_average_window = _require_int(
        run_data.get("moving_average_window", 100), "run.moving_average_window", minimum=1
    )

    paths_data = _require_mapping(raw_config, "paths")
    _check_keys(paths_data, {"experiments", "checkpoints", "results"}, "paths")
    paths = PathsConfig(
        experiments=_require_path(paths_data.get("experiments"), "paths.experiments"),
        checkpoints=_require_path(paths_data.get("checkpoints"), "paths.checkpoints"),
        results=_require_path(paths_data.get("results"), "paths.results"),
    )

    logging_data = raw_config.get("logging", {"level": "INFO"})
    logging_data = _require_mapping_value(logging_data, "logging")
    _check_keys(logging_data, {"level"}, "logging")
    logging_level = _require_text(logging_data.get("level"), "logging.level").upper()
    if logging_level not in _LOG_LEVELS:
        supported = ", ".join(sorted(_LOG_LEVELS))
        raise ConfigurationError(f"Unsupported logging.level {logging_level!r}; use: {supported}")

    return AppConfig(
        schema_version=schema_version,
        environment=EnvironmentConfig(name=environment_name),
        algorithm=AlgorithmConfig(
            name=algorithm_name,
            hyperparameters=dict(hyperparameters),
        ),
        run=RunConfig(
            seed=seed,
            episodes=episodes,
            max_steps_per_episode=max_steps,
            evaluation_episodes=evaluation_episodes,
            moving_average_window=moving_average_window,
        ),
        paths=paths,
        logging=LoggingConfig(level=logging_level),
    )


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], section: str | Path) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        names = ", ".join(str(name) for name in unknown)
        raise ConfigurationError(f"Unknown configuration key(s) in {section}: {names}")


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in mapping:
        raise ConfigurationError(f"Missing required configuration section: {key}")
    return _require_mapping_value(mapping[key], key)


def _require_mapping_value(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} must be a mapping")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field} must be an integer")
    if value < minimum:
        raise ConfigurationError(f"{field} must be >= {minimum}")
    return value


def _require_path(value: Any, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigurationError(f"{field} must be a non-empty path")
    return Path(value)
