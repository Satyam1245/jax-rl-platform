"""Unit tests for the Gymnasium environment layer."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rl_platform.environments import (
    EnvironmentAdapter,
    EnvironmentConfigurationError,
    describe_space,
    make_environment,
    supported_environment_names,
    validate_environment_name,
)


@pytest.mark.parametrize("name", ["CartPole-v1", "MountainCar-v0"])
def test_supported_environment_names_are_allow_listed(name: str) -> None:
    assert name in supported_environment_names()
    assert validate_environment_name(name) == name


def test_supported_environment_names_are_sorted() -> None:
    assert supported_environment_names() == ("CartPole-v1", "MountainCar-v0")


@pytest.mark.parametrize("invalid_name", ["", "CartPole", "Acrobot-v1", 42])
def test_validate_environment_name_rejects_invalid_names(invalid_name: Any) -> None:
    with pytest.raises(EnvironmentConfigurationError, match="Environment name|Unsupported environment"):
        validate_environment_name(invalid_name)


def test_describe_space_handles_box_and_discrete_spaces() -> None:
    box_info = describe_space(spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32))
    discrete_info = describe_space(spaces.Discrete(3))

    assert box_info.kind == "Box"
    assert box_info.shape == (2,)
    assert box_info.dtype == "float32"
    assert box_info.low == (-1.0, -1.0)
    assert box_info.high == (1.0, 1.0)
    assert discrete_info.kind == "Discrete"
    assert discrete_info.shape == ()
    assert discrete_info.dtype == "int64"
    assert discrete_info.n == 3


@pytest.mark.parametrize(
    ("name", "observation_shape", "action_count"),
    [("CartPole-v1", (4,), 2), ("MountainCar-v0", (2,), 3)],
)
def test_make_environment_exposes_spaces(
    name: str, observation_shape: tuple[int, ...], action_count: int
) -> None:
    environment = make_environment(name)
    try:
        assert environment.name == name
        assert environment.observation_space_info.shape == observation_shape
        assert environment.action_space_info.n == action_count
        assert environment.spec.name == name
        assert environment.spec.observation_space.shape == observation_shape
    finally:
        environment.close()


@pytest.mark.parametrize("name", ["CartPole-v1", "MountainCar-v0"])
def test_make_environment_applies_explicit_seed_on_first_reset(name: str) -> None:
    first = make_environment(name, seed=123)
    second = make_environment(name, seed=123)
    try:
        first_reset = first.reset()
        second_reset = second.reset()
        np.testing.assert_array_equal(first_reset.observation, second_reset.observation)
        assert first_reset.info == second_reset.info
    finally:
        first.close()
        second.close()


class _FakeEnvironment:
    """Minimal fake used to test adapter semantics without an algorithm."""

    observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, *, terminated: bool, truncated: bool) -> None:
        self.terminated = terminated
        self.truncated = truncated
        self.reset_args: tuple[int | None, dict[str, Any] | None] | None = None
        self.close_calls = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.reset_args = (seed, options)
        return np.array([0.0], dtype=np.float32), {"reset": True}

    def step(self, action: int):
        assert self.action_space.contains(action)
        return (
            np.array([1.0], dtype=np.float32),
            1.5,
            self.terminated,
            self.truncated,
            {"action": action},
        )

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_step_preserves_terminated_and_truncated(
    terminated: bool, truncated: bool
) -> None:
    fake = _FakeEnvironment(terminated=terminated, truncated=truncated)
    environment = EnvironmentAdapter(fake, "Fake-v0")
    try:
        reset_result = environment.reset(seed=9, options={"test": True})
        result = environment.step(1)
    finally:
        environment.close()

    assert reset_result.info == {"reset": True}
    assert fake.reset_args == (9, {"test": True})
    assert result.reward == 1.5
    assert result.terminated is terminated
    assert result.truncated is truncated
    assert result.done is (terminated or truncated)
    assert result.info == {"action": 1}


def test_close_is_idempotent_and_rejects_later_lifecycle_calls() -> None:
    fake = _FakeEnvironment(terminated=False, truncated=False)
    environment = EnvironmentAdapter(fake, "Fake-v0")

    environment.close()
    environment.close()

    assert environment.closed
    assert fake.close_calls == 1
    with pytest.raises(RuntimeError, match="already closed"):
        environment.reset()
    with pytest.raises(RuntimeError, match="already closed"):
        environment.step(0)


def test_context_manager_closes_environment() -> None:
    fake = _FakeEnvironment(terminated=False, truncated=False)
    with EnvironmentAdapter(fake, "Fake-v0") as environment:
        assert not environment.closed
    assert environment.closed
    assert fake.close_calls == 1


def test_make_environment_rejects_invalid_name_before_creation() -> None:
    with pytest.raises(EnvironmentConfigurationError, match="Unsupported environment"):
        make_environment("NotAnEnvironment-v0")
