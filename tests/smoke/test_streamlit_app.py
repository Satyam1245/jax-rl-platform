"""Application-level smoke checks for the Streamlit presentation layer."""

from app import streamlit_app
from rl_platform.services import available_algorithms, load_algorithm_template, train_experiment


def test_streamlit_entry_point_imports_without_running_ui() -> None:
    assert callable(streamlit_app.main)


def test_streamlit_backend_templates_cover_all_algorithms() -> None:
    assert set(available_algorithms()) == {"dqn", "ppo", "actor_critic"}
    for algorithm in available_algorithms():
        assert load_algorithm_template(streamlit_app.PROJECT_ROOT, algorithm)


def test_backend_training_service_creates_ui_visible_artifacts(tmp_path) -> None:
    execution = train_experiment(
        project_root=tmp_path,
        environment="CartPole-v1",
        algorithm="dqn",
        hyperparameters={
            "learning_rate": 0.01,
            "gamma": 0.99,
            "batch_size": 2,
            "replay_capacity": 4,
            "min_replay_size": 2,
            "target_update_interval": 2,
            "epsilon_start": 0.2,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 4,
            "hidden_layers": 1,
            "hidden_size": 8,
            "activation": "relu",
            "optimizer": "adam",
        },
        seed=3,
        episodes=1,
        max_steps_per_episode=2,
        moving_average_window=2,
    )
    assert execution.experiment.metrics_path.is_file()
    assert execution.checkpoint_path.joinpath("metadata.json").is_file()
    assert execution.checkpoint_path.joinpath("state.msgpack").is_file()
