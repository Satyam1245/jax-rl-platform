"""Short real-environment smoke tests."""

import pytest

from rl_platform.environments import make_environment


@pytest.mark.parametrize("name", ["CartPole-v1", "MountainCar-v0"])
def test_real_environment_can_reset_step_and_close(name: str) -> None:
    environment = make_environment(name, seed=2026)
    try:
        reset_result = environment.reset()
        assert environment.observation_space.contains(reset_result.observation)

        for _ in range(5):
            result = environment.step(0)
            assert environment.observation_space.contains(result.observation)
            assert isinstance(result.reward, float)
            assert isinstance(result.terminated, bool)
            assert isinstance(result.truncated, bool)
            if result.done:
                break
    finally:
        environment.close()

    assert environment.closed
