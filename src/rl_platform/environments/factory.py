"""Environment creation and lifecycle handling for the supported Gymnasium environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SUPPORTED_ENVIRONMENTS = frozenset({"CartPole-v1", "MountainCar-v0"})


class EnvironmentConfigurationError(ValueError):
    """Raised when an environment name or environment configuration is invalid."""


@dataclass(frozen=True, slots=True)
class SpaceInfo:
    """Serializable information about a Gymnasium space."""

    kind: str
    shape: tuple[int, ...] | None
    dtype: str
    n: int | None = None
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Public metadata for an instantiated environment."""

    name: str
    observation_space: SpaceInfo
    action_space: SpaceInfo


@dataclass(frozen=True, slots=True)
class ResetResult:
    """Result returned by a normalized environment reset."""

    observation: Any
    info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result returned by a normalized environment step.

    ``terminated`` and ``truncated`` remain separate so callers can apply
    correct reinforcement-learning bootstrapping semantics. ``done`` is only
    a convenience for episode-control code.
    """

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        """Return whether the episode ended for either Gymnasium reason."""

        return self.terminated or self.truncated


class EnvironmentAdapter:
    """Small lifecycle adapter around one Gymnasium environment instance."""

    def __init__(self, environment: gym.Env, name: str, initial_seed: int | None = None) -> None:
        self._environment = environment
        self._name = name
        self._initial_seed = initial_seed
        self._first_reset = True
        self._closed = False

    @property
    def name(self) -> str:
        """Return the configured Gymnasium environment name."""

        return self._name

    @property
    def observation_space(self) -> spaces.Space:
        """Return the raw Gymnasium observation space."""

        return self._environment.observation_space

    @property
    def action_space(self) -> spaces.Space:
        """Return the raw Gymnasium action space."""

        return self._environment.action_space

    @property
    def observation_space_info(self) -> SpaceInfo:
        """Return normalized observation-space metadata."""

        return describe_space(self.observation_space)

    @property
    def action_space_info(self) -> SpaceInfo:
        """Return normalized action-space metadata."""

        return describe_space(self.action_space)

    @property
    def spec(self) -> EnvironmentSpec:
        """Return the environment name and both space descriptions."""

        return EnvironmentSpec(
            name=self.name,
            observation_space=self.observation_space_info,
            action_space=self.action_space_info,
        )

    @property
    def closed(self) -> bool:
        """Return whether this adapter has been closed."""

        return self._closed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        """Reset the environment and return the observation plus reset info.

        If a seed was supplied to :func:`make_environment`, it is used for the
        first reset when no reset seed is supplied. Later resets preserve the
        environment's normal random-state progression unless explicitly
        reseeded by the caller.
        """

        self._ensure_open()
        reset_seed = self._initial_seed if self._first_reset and seed is None else seed
        self._first_reset = False
        observation, info = self._environment.reset(seed=reset_seed, options=options)
        return ResetResult(observation=observation, info=dict(info))

    def step(self, action: Any) -> StepResult:
        """Advance the environment by one action and preserve both end flags."""

        self._ensure_open()
        observation, reward, terminated, truncated, info = self._environment.step(action)
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def close(self) -> None:
        """Close the underlying environment safely and idempotently."""

        if self._closed:
            return
        try:
            self._environment.close()
        finally:
            self._closed = True

    def __enter__(self) -> EnvironmentAdapter:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Environment {self.name!r} is already closed")


def supported_environment_names() -> tuple[str, ...]:
    """Return supported environment names in deterministic order."""

    return tuple(sorted(SUPPORTED_ENVIRONMENTS))


def validate_environment_name(name: str) -> str:
    """Validate and return an allow-listed Gymnasium environment name."""

    if not isinstance(name, str) or not name.strip():
        raise EnvironmentConfigurationError("Environment name must be a non-empty string")
    normalized_name = name.strip()
    if normalized_name not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(supported_environment_names())
        raise EnvironmentConfigurationError(
            f"Unsupported environment {normalized_name!r}; supported environments: {supported}"
        )
    return normalized_name


def make_environment(
    name: str,
    *,
    seed: int | None = None,
    render_mode: str | None = None,
) -> EnvironmentAdapter:
    """Create one supported Gymnasium environment.

    The supplied seed is applied on the adapter's first reset. No environment
    is created for an invalid name.
    """

    normalized_name = validate_environment_name(name)
    make_kwargs: dict[str, Any] = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    environment = gym.make(normalized_name, **make_kwargs)
    return EnvironmentAdapter(environment, normalized_name, initial_seed=seed)


def describe_space(space: spaces.Space) -> SpaceInfo:
    """Convert a Gymnasium space into stable, inspection-friendly metadata."""

    if isinstance(space, spaces.Box):
        return SpaceInfo(
            kind=type(space).__name__,
            shape=tuple(int(dimension) for dimension in space.shape),
            dtype=str(np.dtype(space.dtype)),
            low=tuple(float(value) for value in np.asarray(space.low).reshape(-1)),
            high=tuple(float(value) for value in np.asarray(space.high).reshape(-1)),
        )
    if isinstance(space, spaces.Discrete):
        return SpaceInfo(
            kind=type(space).__name__,
            shape=tuple(int(dimension) for dimension in (space.shape or ())),
            dtype=str(np.dtype(space.dtype)),
            n=int(space.n),
        )

    shape = getattr(space, "shape", None)
    normalized_shape = None if shape is None else tuple(int(dimension) for dimension in shape)
    dtype = getattr(space, "dtype", None)
    return SpaceInfo(
        kind=type(space).__name__,
        shape=normalized_shape,
        dtype=str(np.dtype(dtype)) if dtype is not None else "unknown",
    )
