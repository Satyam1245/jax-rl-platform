"""Generic environment interaction and training loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import jax
import numpy as np

from rl_platform.environments import EnvironmentAdapter
from rl_platform.metrics import MetricsRecorder
from rl_platform.training.config import TrainingConfig
from rl_platform.utils.logging import get_logger
from rl_platform.utils.random import create_prng_key


class TrainableAgent(Protocol):
    """Structural interface required by the generic training engine."""

    def select_action(
        self,
        observation: Any,
        *,
        key: Any,
        deterministic: bool = False,
    ) -> int: ...

    def observe(self, transition: Any) -> None: ...

    def update(self, *, key: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class EnvironmentTransition:
    """Algorithm-neutral transition passed from an environment to an agent."""

    observation: Any
    action: int
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        """Return whether the transition ended the episode."""

        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Output of one training run."""

    metrics: MetricsRecorder
    total_steps: int
    duration_seconds: float
    episodes_completed: int


class TrainingEngine:
    """Train any agent implementing the small structural agent interface."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._logger = get_logger(__name__)

    def train(
        self,
        environment: EnvironmentAdapter,
        agent: TrainableAgent,
        *,
        start_episode: int = 0,
        initial_total_steps: int = 0,
    ) -> TrainingResult:
        """Run configured episodes, optionally continuing saved progress."""

        _validate_non_negative_int(start_episode, "start_episode")
        _validate_non_negative_int(initial_total_steps, "initial_total_steps")

        started_at = perf_counter()
        metrics = MetricsRecorder(self.config.moving_average_window)
        rng_key = create_prng_key(self.config.seed)
        total_steps = initial_total_steps
        self._logger.info(
            "Training started: episodes=%d seed=%d environment=%s",
            self.config.episodes,
            self.config.seed,
            environment.name,
        )

        for episode_index in range(self.config.episodes):
            global_episode = start_episode + episode_index
            reset_result = environment.reset(seed=self.config.seed + global_episode)
            observation = reset_result.observation
            episode_reward = 0.0
            episode_losses: list[float] = []
            episode_length = 0

            for _ in range(self.config.max_steps_per_episode):
                rng_key, action_key = jax.random.split(rng_key)
                action = agent.select_action(observation, key=action_key, deterministic=False)
                step_result = environment.step(action)
                transition = EnvironmentTransition(
                    observation=np.asarray(observation),
                    action=int(action),
                    reward=step_result.reward,
                    next_observation=np.asarray(step_result.observation),
                    terminated=step_result.terminated,
                    truncated=step_result.truncated,
                )
                agent.observe(transition)
                rng_key, update_key = jax.random.split(rng_key)
                update = agent.update(key=update_key)
                loss = _extract_loss(update)
                if loss is not None:
                    episode_losses.append(loss)

                observation = step_result.observation
                episode_reward += step_result.reward
                episode_length += 1
                total_steps += 1
                if step_result.done:
                    break

            metrics.record_training_episode(
                episode=global_episode + 1,
                reward=episode_reward,
                length=episode_length,
                losses=episode_losses,
            )
            self._logger.info(
                "Training episode %d/%d: reward=%.3f length=%d",
                episode_index + 1,
                self.config.episodes,
                episode_reward,
                episode_length,
            )

        duration = perf_counter() - started_at
        self._logger.info("Training finished: steps=%d duration=%.3fs", total_steps, duration)
        return TrainingResult(
            metrics=metrics,
            total_steps=total_steps,
            duration_seconds=duration,
            episodes_completed=start_episode + self.config.episodes,
        )


def _extract_loss(update: Any) -> float | None:
    if update is None:
        return None
    loss_value = getattr(update, "loss", None)
    if loss_value is None:
        return None
    loss = float(loss_value)
    if not np.isfinite(loss):
        raise FloatingPointError("Agent update returned a NaN or Inf loss")
    return loss


def _validate_non_negative_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
