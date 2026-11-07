"""Validated configuration for independent evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from rl_platform.configuration import RunConfig


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Episode count, step limit, and seed for evaluation."""

    episodes: int
    max_steps_per_episode: int
    seed: int

    @classmethod
    def from_run_config(cls, run_config: RunConfig) -> EvaluationConfig:
        """Create evaluation settings from validated application settings."""

        return cls(
            episodes=run_config.evaluation_episodes,
            max_steps_per_episode=run_config.max_steps_per_episode,
            seed=run_config.seed,
        )

    def __post_init__(self) -> None:
        _positive_int(self.episodes, "episodes")
        _positive_int(self.max_steps_per_episode, "max_steps_per_episode")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")

