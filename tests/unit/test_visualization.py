"""Tests for stored-result loading, comparison, and Plotly visualizations."""

import json
from pathlib import Path

import plotly.graph_objects as go
import pytest

from rl_platform.experiments import ExperimentManager
from rl_platform.metrics import MetricsRecorder
from rl_platform.visualization import (
    ResultValidationError,
    compare_experiments,
    load_experiment_results,
    plot_algorithm_comparison,
    plot_episode_length,
    plot_evaluation_reward,
    plot_loss,
    plot_moving_average_reward,
    plot_reward_curves,
    plot_reward_distribution,
    plot_training_duration,
)


def _write_result(
    root: Path,
    algorithm: str,
    rewards: list[float],
    *,
    duration: float | None = 12.0,
) -> None:
    manager = ExperimentManager(root / "experiments", root / "checkpoints", root / "results")
    experiment = manager.create_experiment(
        algorithm=algorithm,
        environment="CartPole-v1",
        seed=7,
        configuration={"algorithm": algorithm, "seed": 7},
    )
    metrics = MetricsRecorder(moving_average_window=2)
    for index, reward in enumerate(rewards, start=1):
        metrics.record_training_episode(
            episode=index,
            reward=reward,
            length=index + 1,
            losses=[float(index)],
        )
    metrics.record_evaluation_episode(episode=1, reward=max(rewards) + 1.0, length=3)
    manager.persist_metrics(
        experiment,
        metrics,
        training_duration_seconds=duration,
        total_steps=sum(record.episode_length for record in metrics.training_records),
    )


def test_valid_results_load_and_build_all_visualizations(tmp_path: Path) -> None:
    _write_result(tmp_path, "dqn", [1.0, 3.0])
    results = load_experiment_results(tmp_path / "experiments", tmp_path / "results")

    assert len(results) == 1
    result = results[0]
    assert result.algorithm == "dqn"
    assert result.episode_rewards == (1.0, 3.0)
    assert result.training_duration_seconds == 12.0
    figures = [
        plot_reward_curves(results),
        plot_moving_average_reward(results),
        plot_episode_length(results),
        plot_loss(results),
        plot_evaluation_reward(results),
        plot_reward_distribution(results),
        plot_algorithm_comparison(results),
        plot_training_duration(results),
    ]
    assert all(isinstance(figure, go.Figure) for figure in figures)
    assert all(len(figure.data) == 1 for figure in figures)


def test_empty_results_are_returned_without_fabricated_data(tmp_path: Path) -> None:
    results = load_experiment_results(tmp_path / "missing-experiments", tmp_path / "results")
    assert results == ()
    assert compare_experiments(results) == ()
    assert len(plot_reward_curves(results).data) == 0
    assert len(plot_algorithm_comparison(results).data[0].x) == 0


def test_multiple_algorithms_produce_comparison_metrics(tmp_path: Path) -> None:
    _write_result(tmp_path, "dqn", [1.0, 3.0], duration=10.0)
    _write_result(tmp_path, "ppo", [2.0, 4.0], duration=20.0)
    _write_result(tmp_path, "actor_critic", [3.0, 5.0], duration=30.0)
    results = load_experiment_results(tmp_path / "experiments", tmp_path / "results")

    comparisons = compare_experiments(results)

    assert [item.algorithm for item in comparisons] == ["actor_critic", "dqn", "ppo"]
    dqn = next(item for item in comparisons if item.algorithm == "dqn")
    assert dqn.mean_reward == pytest.approx(2.0)
    assert dqn.median_reward == pytest.approx(2.0)
    assert dqn.standard_deviation == pytest.approx(1.0)
    assert dqn.best_reward == pytest.approx(3.0)
    assert dqn.evaluation_reward == pytest.approx(4.0)
    assert dqn.training_duration_seconds == pytest.approx(10.0)
    assert len(plot_algorithm_comparison(results, metric="evaluation_reward").data) == 1


def test_missing_result_fields_raise_validation_error(tmp_path: Path) -> None:
    _write_result(tmp_path, "dqn", [1.0, 2.0])
    metrics_path = next((tmp_path / "results").glob("*/metrics.json"))
    values = json.loads(metrics_path.read_text(encoding="utf-8"))
    del values["training"]
    metrics_path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ResultValidationError, match="missing field"):
        load_experiment_results(tmp_path / "experiments", tmp_path / "results")


def test_malformed_result_file_raises_validation_error(tmp_path: Path) -> None:
    _write_result(tmp_path, "dqn", [1.0, 2.0])
    metrics_path = next((tmp_path / "results").glob("*/metrics.json"))
    metrics_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ResultValidationError, match="Malformed experiment metrics"):
        load_experiment_results(tmp_path / "experiments", tmp_path / "results")

