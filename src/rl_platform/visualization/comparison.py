"""Comparison statistics computed only from loaded experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from statistics import median, pstdev
from typing import Iterable

from rl_platform.visualization.results import ExperimentResult


@dataclass(frozen=True, slots=True)
class AlgorithmComparison:
    """Aggregate comparison metrics for one algorithm."""

    algorithm: str
    mean_reward: float | None
    median_reward: float | None
    standard_deviation: float | None
    best_reward: float | None
    evaluation_reward: float | None
    training_duration_seconds: float | None


def compare_experiments(results: Iterable[ExperimentResult]) -> tuple[AlgorithmComparison, ...]:
    """Aggregate actual stored results by algorithm without filling missing data."""

    groups: dict[str, list[ExperimentResult]] = {}
    for result in results:
        groups.setdefault(result.algorithm, []).append(result)
    comparisons: list[AlgorithmComparison] = []
    for algorithm in sorted(groups):
        group = groups[algorithm]
        rewards = [reward for result in group for reward in result.episode_rewards]
        evaluation_rewards = [reward for result in group for reward in result.evaluation_rewards]
        durations = [result.training_duration_seconds for result in group]
        comparisons.append(
            AlgorithmComparison(
                algorithm=algorithm,
                mean_reward=(fsum(rewards) / len(rewards)) if rewards else None,
                median_reward=median(rewards) if rewards else None,
                standard_deviation=pstdev(rewards) if len(rewards) > 1 else (0.0 if rewards else None),
                best_reward=max(rewards) if rewards else None,
                evaluation_reward=(fsum(evaluation_rewards) / len(evaluation_rewards)) if evaluation_rewards else None,
                training_duration_seconds=(
                    fsum(duration for duration in durations if duration is not None) / len(durations)
                    if durations and all(duration is not None for duration in durations)
                    else None
                ),
            )
        )
    return tuple(comparisons)
