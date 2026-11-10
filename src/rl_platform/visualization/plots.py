"""Plotly figures built from validated stored experiment results."""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from rl_platform.visualization.comparison import compare_experiments
from rl_platform.visualization.results import ExperimentResult


def plot_reward_curves(results: Sequence[ExperimentResult]) -> go.Figure:
    """Plot training episode rewards for each stored experiment."""

    figure = go.Figure()
    for result in results:
        figure.add_trace(go.Scatter(
            x=list(range(1, len(result.episode_rewards) + 1)),
            y=list(result.episode_rewards),
            mode="lines+markers",
            name=f"{result.algorithm} ({result.experiment_id})",
        ))
    return _finish(figure, "Reward curves", "Episode", "Reward")


def plot_moving_average_reward(results: Sequence[ExperimentResult]) -> go.Figure:
    """Plot stored moving-average reward series."""

    figure = go.Figure()
    for result in results:
        figure.add_trace(go.Scatter(
            x=list(range(1, len(result.moving_average_rewards) + 1)),
            y=list(result.moving_average_rewards),
            mode="lines",
            name=f"{result.algorithm} ({result.experiment_id})",
        ))
    return _finish(figure, "Moving-average reward", "Episode", "Moving-average reward")


def plot_episode_length(results: Sequence[ExperimentResult]) -> go.Figure:
    """Plot training episode lengths."""

    figure = go.Figure()
    for result in results:
        figure.add_trace(go.Scatter(
            x=list(range(1, len(result.episode_lengths) + 1)),
            y=list(result.episode_lengths),
            mode="lines+markers",
            name=f"{result.algorithm} ({result.experiment_id})",
        ))
    return _finish(figure, "Episode length", "Episode", "Steps")


def plot_loss(results: Sequence[ExperimentResult]) -> go.Figure:
    """Plot stored optimizer losses by update index."""

    figure = go.Figure()
    for result in results:
        figure.add_trace(go.Scatter(
            x=list(range(1, len(result.losses) + 1)),
            y=list(result.losses),
            mode="lines",
            name=f"{result.algorithm} ({result.experiment_id})",
        ))
    return _finish(figure, "Training loss", "Update", "Loss")


def plot_evaluation_reward(results: Sequence[ExperimentResult]) -> go.Figure:
    """Plot stored evaluation episode rewards."""

    figure = go.Figure()
    for result in results:
        figure.add_trace(go.Scatter(
            x=list(range(1, len(result.evaluation_rewards) + 1)),
            y=list(result.evaluation_rewards),
            mode="lines+markers",
            name=f"{result.algorithm} ({result.experiment_id})",
        ))
    return _finish(figure, "Evaluation reward", "Evaluation episode", "Reward")


def plot_reward_distribution(results: Sequence[ExperimentResult]) -> go.Figure:
    """Plot reward distributions using actual episode rewards."""

    figure = go.Figure()
    for result in results:
        figure.add_trace(go.Histogram(
            x=list(result.episode_rewards),
            name=f"{result.algorithm} ({result.experiment_id})",
            opacity=0.7,
        ))
    figure.update_layout(barmode="overlay")
    return _finish(figure, "Reward distribution", "Reward", "Count")


def plot_algorithm_comparison(
    results: Sequence[ExperimentResult],
    *,
    metric: str = "mean_reward",
) -> go.Figure:
    """Plot one aggregate comparison metric per algorithm."""

    allowed = {
        "mean_reward", "median_reward", "standard_deviation", "best_reward",
        "evaluation_reward", "training_duration_seconds",
    }
    if metric not in allowed:
        raise ValueError(f"Unsupported comparison metric: {metric}")
    comparisons = compare_experiments(results)
    values = [(comparison.algorithm, getattr(comparison, metric)) for comparison in comparisons]
    values = [(algorithm, value) for algorithm, value in values if value is not None]
    figure = go.Figure(go.Bar(x=[item[0] for item in values], y=[item[1] for item in values]))
    return _finish(figure, f"Algorithm comparison: {metric}", "Algorithm", metric)


def plot_training_duration(results: Sequence[ExperimentResult]) -> go.Figure:
    """Plot measured training duration for experiments that recorded it."""

    values = [
        (f"{result.algorithm} ({result.experiment_id})", result.training_duration_seconds)
        for result in results
        if result.training_duration_seconds is not None
    ]
    figure = go.Figure(go.Bar(x=[item[0] for item in values], y=[item[1] for item in values]))
    return _finish(figure, "Training duration", "Experiment", "Seconds")


def _finish(figure: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    figure.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return figure
