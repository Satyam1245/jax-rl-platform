"""Streamlit presentation layer for the JAX RL platform backend."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from statistics import mean
from typing import Any

import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_platform.configuration import load_config
from rl_platform.environments import supported_environment_names
from rl_platform.services import (
    available_algorithms,
    evaluate_checkpoint,
    list_checkpoint_summaries,
    load_algorithm_template,
    load_stored_results,
    train_experiment,
)
from rl_platform.visualization import (
    compare_experiments,
    plot_algorithm_comparison,
    plot_episode_length,
    plot_evaluation_reward,
    plot_loss,
    plot_moving_average_reward,
    plot_reward_curves,
    plot_reward_distribution,
    plot_training_duration,
)


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(page_title="JAX RL Platform", page_icon="🤖", layout="wide")
    app_config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    st.sidebar.title("JAX RL Platform")
    section = st.sidebar.radio(
        "Navigate",
        (
            "Overview",
            "Train Agent",
            "Training Progress",
            "Evaluate Agent",
            "Experiment History",
            "Algorithm Comparison",
            "Checkpoints",
        ),
    )
    st.sidebar.caption(f"Environment scope: {', '.join(supported_environment_names())}")

    if section == "Overview":
        render_overview(app_config)
    elif section == "Train Agent":
        render_train_agent(app_config)
    elif section == "Training Progress":
        render_training_progress()
    elif section == "Evaluate Agent":
        render_evaluate_agent(app_config)
    elif section == "Experiment History":
        render_experiment_history()
    elif section == "Algorithm Comparison":
        render_algorithm_comparison()
    else:
        render_checkpoints(app_config)


def render_overview(app_config: Any) -> None:
    """Render project summary and recent stored experiments."""

    st.title("JAX Reinforcement Learning Platform")
    st.write(
        "A modular platform for training, evaluating, comparing, and visualizing "
        "JAX reinforcement-learning agents."
    )
    first_column, second_column = st.columns(2)
    with first_column:
        st.subheader("Available environments")
        st.write(list(supported_environment_names()))
    with second_column:
        st.subheader("Available algorithms")
        st.write(list(available_algorithms()))
    st.subheader("Recent experiments")
    results = _stored_results()
    if not results:
        st.info("No stored experiment results are available yet.")
        return
    st.dataframe(
        [
            {
                "experiment_id": result.experiment_id,
                "algorithm": result.algorithm,
                "environment": result.environment,
                "episodes": len(result.episode_rewards),
                "best_reward": max(result.episode_rewards) if result.episode_rewards else None,
            }
            for result in results[-5:]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_train_agent(app_config: Any) -> None:
    """Render training controls and delegate execution to the backend service."""

    st.title("Train Agent")
    algorithms = available_algorithms()
    with st.form("training_form"):
        env_names = supported_environment_names()
        default_env = app_config.environment.name
        default_env_index = env_names.index(default_env) if default_env in env_names else 0
        environment = st.selectbox(
            "Environment",
            env_names,
            index=default_env_index,
        )
        algorithm = st.selectbox("Algorithm", algorithms)
        defaults = load_algorithm_template(PROJECT_ROOT, algorithm)
        st.subheader("Algorithm hyperparameters")
        hyperparameters = _hyperparameter_controls(algorithm, defaults)
        st.subheader("Training settings")
        seed = st.number_input("Seed", min_value=0, value=int(app_config.run.seed), step=1)
        episodes = st.number_input("Episodes", min_value=1, value=int(app_config.run.episodes), step=1)
        max_steps = st.number_input(
            "Maximum steps per episode",
            min_value=1,
            value=int(app_config.run.max_steps_per_episode),
            step=1,
        )
        moving_window = st.number_input(
            "Moving-average window",
            min_value=1,
            value=int(app_config.run.moving_average_window),
            step=1,
        )
        submitted = st.form_submit_button("Start training")

    if not submitted:
        return
    try:
        with st.spinner("Training in progress..."):
            execution = train_experiment(
                project_root=PROJECT_ROOT,
                environment=environment,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                seed=int(seed),
                episodes=int(episodes),
                max_steps_per_episode=int(max_steps),
                moving_average_window=int(moving_window),
            )
        st.session_state["last_training"] = execution
        st.success(
            f"Training completed: {execution.experiment.experiment_id} "
            f"(checkpoint: {execution.checkpoint_path.name})"
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Training failed: {exc}")


def render_training_progress() -> None:
    """Render metrics from the most recent backend training execution."""

    st.title("Training Progress")
    execution = st.session_state.get("last_training")
    if execution is None:
        st.info("Start a training run to view progress here.")
        return
    result = execution.result
    metrics = result.metrics
    st.success("Training status: completed")
    columns = st.columns(4)
    columns[0].metric("Episodes", len(metrics.episode_reward))
    columns[1].metric("Steps", result.total_steps)
    columns[2].metric("Latest reward", _latest(metrics.episode_reward))
    columns[3].metric("Latest loss", _latest(metrics.loss))
    st.write("Average reward", _latest(metrics.mean_reward))
    st.write("Moving-average reward", _latest(metrics.moving_average_reward))
    st.dataframe(
        [
            {
                "episode": record.episode,
                "reward": record.episode_reward,
                "length": record.episode_length,
                "mean_reward": record.mean_reward,
                "moving_average_reward": record.moving_average_reward,
                "loss": record.loss,
            }
            for record in metrics.training_records
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_evaluate_agent(app_config: Any) -> None:
    """Render checkpoint selection and deterministic evaluation results."""

    st.title("Evaluate Agent")
    summaries = list_checkpoint_summaries(PROJECT_ROOT / "checkpoints")
    if not summaries:
        st.info("No valid checkpoints are available.")
        return
    selected = st.selectbox(
        "Checkpoint",
        summaries,
        format_func=lambda item: _checkpoint_label(item),
    )
    episodes = st.number_input(
        "Evaluation episodes", min_value=1, value=int(app_config.run.evaluation_episodes), step=1
    )
    max_steps = st.number_input(
        "Maximum steps per episode",
        min_value=1,
        value=int(app_config.run.max_steps_per_episode),
        step=1,
    )
    if not st.button("Evaluate checkpoint"):
        return
    try:
        with st.spinner("Evaluating checkpoint..."):
            _, result = evaluate_checkpoint(
                checkpoint_path=selected.path,
                episodes=int(episodes),
                max_steps_per_episode=int(max_steps),
            )
        rewards = result.metrics.evaluation_reward
        st.metric("Mean reward", mean(rewards) if rewards else None)
        st.plotly_chart(_evaluation_distribution(rewards), use_container_width=True)
        st.dataframe(
            [
                {"episode": record.episode, "reward": record.evaluation_reward, "length": record.episode_length}
                for record in result.metrics.evaluation_records
            ],
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")


def render_experiment_history() -> None:
    """Render stored experiment records and their persisted curves."""

    st.title("Experiment History")
    results = _stored_results()
    if not results:
        st.info("No stored experiment results are available yet.")
        return
    st.dataframe(
        [
            {
                "experiment_id": result.experiment_id,
                "algorithm": result.algorithm,
                "environment": result.environment,
                "seed": result.seed,
                "episodes": len(result.episode_rewards),
                "duration_seconds": result.training_duration_seconds,
            }
            for result in results
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(plot_reward_curves(results), use_container_width=True)
    st.plotly_chart(plot_moving_average_reward(results), use_container_width=True)
    st.plotly_chart(plot_episode_length(results), use_container_width=True)
    st.plotly_chart(plot_loss(results), use_container_width=True)
    st.plotly_chart(plot_evaluation_reward(results), use_container_width=True)


def render_algorithm_comparison() -> None:
    """Render comparison statistics and interactive plots from stored results."""

    st.title("Algorithm Comparison")
    results = _stored_results()
    if not results:
        st.info("No stored experiment results are available yet.")
        return
    comparisons = compare_experiments(results)
    st.dataframe([asdict(item) for item in comparisons], use_container_width=True, hide_index=True)
    metric = st.selectbox(
        "Comparison metric",
        ("mean_reward", "median_reward", "standard_deviation", "best_reward", "evaluation_reward", "training_duration_seconds"),
    )
    st.plotly_chart(plot_algorithm_comparison(results, metric=metric), use_container_width=True)
    st.plotly_chart(plot_reward_distribution(results), use_container_width=True)
    st.plotly_chart(plot_training_duration(results), use_container_width=True)


def render_checkpoints(app_config: Any) -> None:
    """Render saved checkpoint metadata."""

    del app_config
    st.title("Checkpoints")
    summaries = list_checkpoint_summaries(PROJECT_ROOT / "checkpoints")
    if not summaries:
        st.info("No valid checkpoints are available.")
        return
    st.dataframe(
        [
            {
                "path": str(item.path),
                "experiment_id": item.metadata.experiment_id,
                "algorithm": item.metadata.algorithm,
                "environment": item.metadata.environment,
                "seed": item.metadata.seed,
                "training_steps": item.metadata.training_steps,
                "episodes_completed": item.metadata.episodes_completed,
                "total_steps": item.metadata.total_steps,
                "created_at": item.metadata.created_at,
            }
            for item in summaries
        ],
        use_container_width=True,
        hide_index=True,
    )


def _hyperparameter_controls(algorithm: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Render editable controls using the selected configuration template."""

    values: dict[str, Any] = {}
    for name, default in defaults.items():
        label = name.replace("_", " ").title()
        if isinstance(default, bool):
            values[name] = st.checkbox(label, value=default)
        elif isinstance(default, int):
            minimum = 0 if name in {"hidden_layers"} else 1
            # Explicitly cast to int: st.number_input may return int|float
            raw = st.number_input(label, min_value=minimum, value=default, step=1, key=f"{algorithm}_{name}")
            values[name] = int(raw)
        elif isinstance(default, float):
            minimum = 0.0 if name in {"gamma", "gae_lambda", "epsilon_end", "epsilon_start", "value_coefficient", "entropy_coefficient"} else 0.00000001
            raw = st.number_input(label, min_value=minimum, value=default, format="%.8f", key=f"{algorithm}_{name}")
            values[name] = float(raw)
        else:
            values[name] = st.text_input(label, value=str(default), key=f"{algorithm}_{name}")
    return values


def _stored_results() -> tuple[Any, ...]:
    try:
        return load_stored_results(PROJECT_ROOT)
    except Exception as exc:
        st.error(f"Unable to load stored experiment results: {exc}")
        return ()


def _latest(values: tuple[Any, ...]) -> Any:
    return values[-1] if values else None


def _checkpoint_label(summary: Any) -> str:
    metadata = summary.metadata
    return f"{metadata.algorithm} | {metadata.environment} | step {metadata.training_steps} | {summary.path.name}"


def _evaluation_distribution(rewards: tuple[float, ...]) -> go.Figure:
    figure = go.Figure(go.Histogram(x=list(rewards), name="Evaluation reward"))
    figure.update_layout(title="Evaluation reward distribution", xaxis_title="Reward", yaxis_title="Count")
    return figure


if __name__ == "__main__":
    main()
