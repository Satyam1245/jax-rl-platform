"""Independent deterministic evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import jax

from rl_platform.environments import EnvironmentAdapter
from rl_platform.evaluation.config import EvaluationConfig
from rl_platform.metrics import MetricsRecorder
from rl_platform.utils.logging import get_logger
from rl_platform.utils.random import create_prng_key


class EvaluatableAgent(Protocol):
    """Structural interface required by the evaluation engine."""

    def select_action(
        self,
        observation: Any,
        *,
        key: Any,
        deterministic: bool = False,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Output of one independent evaluation run."""

    metrics: MetricsRecorder
    total_steps: int
    duration_seconds: float


class EvaluationEngine:
    """Evaluate an agent without updating it or adding replay transitions."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self._logger = get_logger(__name__)

    def evaluate(
        self,
        environment: EnvironmentAdapter,
        agent: EvaluatableAgent,
    ) -> EvaluationResult:
        """Run deterministic episodes and return evaluation-reward metrics."""

        started_at = perf_counter()
        metrics = MetricsRecorder()
        rng_key = create_prng_key(self.config.seed)
        total_steps = 0
        self._logger.info(
            "Evaluation started: episodes=%d seed=%d environment=%s",
            self.config.episodes,
            self.config.seed,
            environment.name,
        )

        for episode_index in range(self.config.episodes):
            reset_result = environment.reset(seed=self.config.seed + episode_index)
            observation = reset_result.observation
            episode_reward = 0.0
            episode_length = 0
            for _ in range(self.config.max_steps_per_episode):
                rng_key, action_key = jax.random.split(rng_key)
                action = agent.select_action(observation, key=action_key, deterministic=True)
                step_result = environment.step(action)
                observation = step_result.observation
                episode_reward += step_result.reward
                episode_length += 1
                total_steps += 1
                if step_result.done:
                    break

            metrics.record_evaluation_episode(
                episode=episode_index + 1,
                reward=episode_reward,
                length=episode_length,
            )
            self._logger.info(
                "Evaluation episode %d/%d: reward=%.3f length=%d",
                episode_index + 1,
                self.config.episodes,
                episode_reward,
                episode_length,
            )

        duration = perf_counter() - started_at
        self._logger.info("Evaluation finished: steps=%d duration=%.3fs", total_steps, duration)
        return EvaluationResult(metrics=metrics, total_steps=total_steps, duration_seconds=duration)

