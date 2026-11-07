"""Unit tests for independent deterministic evaluation."""

from dataclasses import dataclass

from rl_platform.environments import make_environment
from rl_platform.evaluation import EvaluationConfig, EvaluationEngine


@dataclass
class SpyAgent:
    deterministic_calls: list[bool]

    def select_action(self, observation, *, key, deterministic=False) -> int:
        del observation, key
        self.deterministic_calls.append(deterministic)
        return 0


def test_evaluation_is_deterministic_by_default_and_does_not_update_agent() -> None:
    agent = SpyAgent(deterministic_calls=[])
    config = EvaluationConfig(episodes=2, max_steps_per_episode=2, seed=31)
    with make_environment("CartPole-v1") as environment:
        result = EvaluationEngine(config).evaluate(environment, agent)

    assert result.total_steps == 4
    assert result.metrics.evaluation_reward[0] == 2.0
    assert len(result.metrics.evaluation_reward) == 2
    assert agent.deterministic_calls == [True, True, True, True]
    assert result.metrics.training_records == ()

