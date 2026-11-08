"""Unit tests for shared metric recording."""

import pytest

from rl_platform.metrics import MetricsRecorder


def test_metrics_recorder_records_training_series_and_loss() -> None:
    recorder = MetricsRecorder(moving_average_window=2)
    first = recorder.record_training_episode(
        episode=1, reward=1.0, length=3, losses=[2.0, 4.0]
    )
    second = recorder.record_training_episode(episode=2, reward=3.0, length=4, losses=[6.0])
    recorder.record_training_episode(episode=3, reward=5.0, length=2)

    assert first.mean_reward == pytest.approx(1.0)
    assert second.mean_reward == pytest.approx(2.0)
    assert second.moving_average_reward == pytest.approx(2.0)
    assert recorder.episode_reward == (1.0, 3.0, 5.0)
    assert recorder.episode_length == (3, 4, 2)
    assert recorder.mean_reward == pytest.approx((1.0, 2.0, 3.0))
    assert recorder.moving_average_reward == pytest.approx((1.0, 2.0, 4.0))
    assert recorder.loss == pytest.approx((2.0, 4.0, 6.0))


def test_metrics_recorder_records_independent_evaluation_rewards() -> None:
    recorder = MetricsRecorder()
    recorder.record_evaluation_episode(episode=1, reward=7.0, length=5)

    assert recorder.training_records == ()
    assert recorder.evaluation_reward == (7.0,)
    assert recorder.evaluation_records[0].episode_length == 5


def test_metrics_recorder_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="moving_average_window"):
        MetricsRecorder(moving_average_window=0)
    recorder = MetricsRecorder()
    with pytest.raises(ValueError, match="finite"):
        recorder.record_training_episode(episode=1, reward=float("nan"), length=1)

