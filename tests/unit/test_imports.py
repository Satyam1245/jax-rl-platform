"""Import smoke tests for the Phase 1 package foundation."""

import rl_platform
from rl_platform.agents import DQNAgent
from rl_platform.algorithms import DQNConfig, ReplayBuffer
from rl_platform import configuration, utils
from rl_platform.agents import ActorCriticAgent
from rl_platform.algorithms import ActorCriticConfig
from rl_platform.configuration import AppConfig, ConfigurationError
from rl_platform.networks import MLP, MLPConfig


def test_core_package_imports() -> None:
    assert rl_platform.__version__ == "0.1.0"
    assert configuration.AppConfig is AppConfig
    assert issubclass(ConfigurationError, ValueError)
    assert callable(utils.configure_logging)
    assert MLPConfig is not None
    assert MLP is not None
    assert ActorCriticAgent is not None
    assert ActorCriticConfig is not None
    assert DQNAgent is not None
    assert DQNConfig is not None
    assert ReplayBuffer is not None
