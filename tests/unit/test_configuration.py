"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from rl_platform.configuration import ConfigurationError, load_config, validate_config


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "environment": {"name": "CartPole-v1"},
        "algorithm": {"name": "dqn", "hyperparameters": {"learning_rate": 0.001}},
        "run": {
            "seed": 7,
            "episodes": 3,
            "max_steps_per_episode": 10,
            "evaluation_episodes": 2,
        },
        "paths": {
            "experiments": "experiments",
            "checkpoints": "checkpoints",
            "results": "results",
        },
        "logging": {"level": "debug"},
    }


def test_load_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_valid_config()), encoding="utf-8")

    config = load_config(config_path)

    assert config.environment.name == "CartPole-v1"
    assert config.algorithm.name == "dqn"
    assert config.algorithm.hyperparameters["learning_rate"] == 0.001
    assert config.run.seed == 7
    assert config.run.moving_average_window == 100
    assert config.paths.results == Path("results")
    assert config.logging.level == "DEBUG"


def test_validate_config_rejects_unsupported_environment() -> None:
    raw_config = _valid_config()
    raw_config["environment"] = {"name": "UnknownEnv-v0"}

    with pytest.raises(ConfigurationError, match="Unsupported environment"):
        validate_config(raw_config)


def test_validate_config_rejects_invalid_run_value() -> None:
    raw_config = _valid_config()
    raw_config["run"] = {
        "seed": -1,
        "episodes": 3,
        "max_steps_per_episode": 10,
        "evaluation_episodes": 2,
    }

    with pytest.raises(ConfigurationError, match=r"run\.seed must be >= 0"):
        validate_config(raw_config)


def test_validate_config_rejects_unknown_key() -> None:
    raw_config = _valid_config()
    raw_config["unexpected"] = True

    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        validate_config(raw_config)
