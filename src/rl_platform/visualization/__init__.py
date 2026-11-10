"""Stored-result loading, comparison metrics, and Plotly visualizations."""

from rl_platform.visualization.comparison import AlgorithmComparison, compare_experiments
from rl_platform.visualization.plots import (
    plot_algorithm_comparison,
    plot_episode_length,
    plot_evaluation_reward,
    plot_loss,
    plot_moving_average_reward,
    plot_reward_curves,
    plot_reward_distribution,
    plot_training_duration,
)
from rl_platform.visualization.results import (
    ExperimentResult,
    ResultValidationError,
    load_experiment_result,
    load_experiment_results,
)

__all__ = [
    "AlgorithmComparison",
    "ExperimentResult",
    "ResultValidationError",
    "compare_experiments",
    "load_experiment_result",
    "load_experiment_results",
    "plot_algorithm_comparison",
    "plot_episode_length",
    "plot_evaluation_reward",
    "plot_loss",
    "plot_moving_average_reward",
    "plot_reward_curves",
    "plot_reward_distribution",
    "plot_training_duration",
]
