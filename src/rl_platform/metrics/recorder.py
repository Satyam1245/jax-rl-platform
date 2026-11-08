"""Metric recording primitives shared by training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import fsum
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Metrics recorded after one training episode."""

    episode: int
    episode_reward: float
    episode_length: int
    mean_reward: float
    moving_average_reward: float
    loss: float | None


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeMetrics:
    """Metrics recorded after one evaluation episode."""

    episode: int
    evaluation_reward: float
    episode_length: int


@dataclass
class MetricsRecorder:
    """Collect immutable per-episode records and derived metric series."""

    moving_average_window: int = 100
    _training_records: list[EpisodeMetrics] = field(default_factory=list, init=False)
    _evaluation_records: list[EvaluationEpisodeMetrics] = field(default_factory=list, init=False)
    _loss_values: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.moving_average_window, bool) or not isinstance(
            self.moving_average_window, int
        ) or self.moving_average_window < 1:
            raise ValueError("moving_average_window must be a positive integer")

    def record_training_episode(
        self,
        *,
        episode: int,
        reward: float,
        length: int,
        losses: Iterable[float] = (),
    ) -> EpisodeMetrics:
        """Record reward, length, episode loss, mean, and moving average."""

        _validate_episode(episode)
        _validate_length(length)
        episode_reward = _finite_float(reward, "episode reward")
        episode_losses = [_finite_float(loss, "loss") for loss in losses]
        self._loss_values.extend(episode_losses)
        rewards = [record.episode_reward for record in self._training_records]
        rewards.append(episode_reward)
        moving_rewards = rewards[-self.moving_average_window :]
        record = EpisodeMetrics(
            episode=episode,
            episode_reward=episode_reward,
            episode_length=length,
            mean_reward=fsum(rewards) / len(rewards),
            moving_average_reward=fsum(moving_rewards) / len(moving_rewards),
            loss=(fsum(episode_losses) / len(episode_losses)) if episode_losses else None,
        )
        self._training_records.append(record)
        return record

    def record_evaluation_episode(
        self,
        *,
        episode: int,
        reward: float,
        length: int,
    ) -> EvaluationEpisodeMetrics:
        """Record one independent evaluation episode."""

        _validate_episode(episode)
        _validate_length(length)
        record = EvaluationEpisodeMetrics(
            episode=episode,
            evaluation_reward=_finite_float(reward, "evaluation reward"),
            episode_length=length,
        )
        self._evaluation_records.append(record)
        return record

    @property
    def training_records(self) -> tuple[EpisodeMetrics, ...]:
        """Return training records in episode order."""

        return tuple(self._training_records)

    @property
    def evaluation_records(self) -> tuple[EvaluationEpisodeMetrics, ...]:
        """Return evaluation records in episode order."""

        return tuple(self._evaluation_records)

    @property
    def episode_reward(self) -> tuple[float, ...]:
        """Return the training episode-reward series."""

        return tuple(record.episode_reward for record in self._training_records)

    @property
    def episode_length(self) -> tuple[int, ...]:
        """Return the training episode-length series."""

        return tuple(record.episode_length for record in self._training_records)

    @property
    def mean_reward(self) -> tuple[float, ...]:
        """Return the cumulative mean-reward series."""

        return tuple(record.mean_reward for record in self._training_records)

    @property
    def moving_average_reward(self) -> tuple[float, ...]:
        """Return the configured-window moving-average reward series."""

        return tuple(record.moving_average_reward for record in self._training_records)

    @property
    def loss(self) -> tuple[float, ...]:
        """Return every finite optimizer loss reported during training."""

        return tuple(self._loss_values)

    @property
    def evaluation_reward(self) -> tuple[float, ...]:
        """Return the independent evaluation-reward series."""

        return tuple(record.evaluation_reward for record in self._evaluation_records)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of all recorded metrics."""

        return {
            "moving_average_window": self.moving_average_window,
            "training": [asdict(record) for record in self._training_records],
            "evaluation": [asdict(record) for record in self._evaluation_records],
            "loss": list(self._loss_values),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> MetricsRecorder:
        """Restore metrics previously returned by :meth:`to_dict`."""

        window = values.get("moving_average_window", 100)
        if not isinstance(window, int):
            raise ValueError("Metric moving_average_window must be an integer")
        recorder = cls(window)
        training = values.get("training", [])
        evaluation = values.get("evaluation", [])
        losses = values.get("loss", [])
        if not isinstance(training, list) or not isinstance(evaluation, list) or not isinstance(losses, list):
            raise ValueError("Metric records must be lists")
        recorder._training_records = [EpisodeMetrics(**record) for record in training]
        recorder._evaluation_records = [EvaluationEpisodeMetrics(**record) for record in evaluation]
        recorder._loss_values = [_finite_float(loss, "loss") for loss in losses]
        return recorder


def _validate_episode(episode: int) -> None:
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
        raise ValueError("episode must be a positive integer")


def _validate_length(length: int) -> None:
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("episode length must be a positive integer")


def _finite_float(value: float, field: str) -> float:
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    return result
