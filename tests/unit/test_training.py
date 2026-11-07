"""Unit tests for the generic training engine and configuration."""

from dataclasses import dataclass

import pytest

from rl_platform.environments import make_environment
from rl_platform.training import EnvironmentTransition, TrainingConfig, TrainingEngine


@dataclass
class StubUpdate:
    loss: float


class StubAgent:
    def __init__(self) -> None:
        self.observed: list[EnvironmentTransition] = []
        self.action_calls = 0
        self.update_calls = 0

    def select_action(self, observation, *, key, deterministic=False) -> int:
        del observation, key, deterministic
        self.action_calls += 1
        return 0

    def observe(self, transition) -> None:
        self.observed.append(transition)

    def update(self, *, key):
        del key
        self.update_calls += 1
        return StubUpdate(loss=0.25)


def test_environment_transition_done_is_algorithm_neutral() -> None:
    assert not EnvironmentTransition(0, 0, 0.0, 1, False, False).done
    assert EnvironmentTransition(0, 0, 0.0, 1, True, False).done
    assert EnvironmentTransition(0, 0, 0.0, 1, False, True).done


def test_training_engine_records_interaction_and_metrics() -> None:
    agent = StubAgent()
    config = TrainingConfig(
        episodes=2,
        max_steps_per_episode=2,
        seed=21,
        moving_average_window=2,
    )
    with make_environment("CartPole-v1") as environment:
        result = TrainingEngine(config).train(environment, agent)

    assert result.total_steps == 4
    assert result.metrics.episode_length == (2, 2)
    assert len(result.metrics.episode_reward) == 2
    assert result.metrics.loss == (0.25, 0.25, 0.25, 0.25)
    assert len(agent.observed) == 4
    assert agent.action_calls == 4
    assert agent.update_calls == 4


def test_training_config_rejects_invalid_episode_settings() -> None:
    with pytest.raises(ValueError, match="episodes"):
        TrainingConfig(episodes=0, max_steps_per_episode=1, seed=0)

