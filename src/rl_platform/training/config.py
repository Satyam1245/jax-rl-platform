"""Validated configuration for the generic training engine."""

from __future__ import annotations

from dataclasses import dataclass

from rl_platform.configuration import RunConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Run-size, reproducibility, and metric settings for training."""

    episodes: int
    max_steps_per_episode: int
    seed: int
    moving_average_window: int = 100

    @classmethod
    def from_run_config(cls, run_config: RunConfig) -> TrainingConfig:
        """Create training settings from the validated application config."""

        return cls(
            episodes=run_config.episodes,
            max_steps_per_episode=run_config.max_steps_per_episode,
            seed=run_config.seed,
            moving_average_window=run_config.moving_average_window,
        )

    def __post_init__(self) -> None:
        _positive_int(self.episodes, "episodes")
        _positive_int(self.max_steps_per_episode, "max_steps_per_episode")
        _non_negative_int(self.seed, "seed")
        _positive_int(self.moving_average_window, "moving_average_window")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _non_negative_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")

