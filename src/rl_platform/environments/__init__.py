"""Gymnasium environment factory and lifecycle adapter."""

from rl_platform.environments.factory import (
    SUPPORTED_ENVIRONMENTS,
    EnvironmentAdapter,
    EnvironmentConfigurationError,
    EnvironmentSpec,
    ResetResult,
    SpaceInfo,
    StepResult,
    describe_space,
    make_environment,
    supported_environment_names,
    validate_environment_name,
)

__all__ = [
    "SUPPORTED_ENVIRONMENTS",
    "EnvironmentAdapter",
    "EnvironmentConfigurationError",
    "EnvironmentSpec",
    "ResetResult",
    "SpaceInfo",
    "StepResult",
    "describe_space",
    "make_environment",
    "supported_environment_names",
    "validate_environment_name",
]
