"""Q-network wrapper built on the Phase 3 Flax MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from rl_platform.networks.mlp import MLP, MLPConfig, initialize_mlp, mlp_forward


@dataclass(frozen=True, slots=True)
class QNetworkConfig:
    """Architecture configuration for a discrete-action Q-network."""

    observation_size: int
    action_count: int
    hidden_layers: int
    hidden_size: int
    activation: str

    def __post_init__(self) -> None:
        if isinstance(self.action_count, bool) or not isinstance(self.action_count, int):
            raise ValueError("action_count must be a positive integer")
        if self.action_count < 1:
            raise ValueError("action_count must be a positive integer")
        MLPConfig(
            input_size=self.observation_size,
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
            output_size=self.action_count,
            activation=self.activation,
        )

    @property
    def mlp_config(self) -> MLPConfig:
        """Return the equivalent reusable MLP configuration."""

        return MLPConfig(
            input_size=self.observation_size,
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
            output_size=self.action_count,
            activation=self.activation,
        )


@dataclass(frozen=True, slots=True)
class QNetwork:
    """Functional Q-network facade over a Flax MLP."""

    config: QNetworkConfig
    model: MLP

    @classmethod
    def create(cls, config: QNetworkConfig) -> QNetwork:
        """Create an uninitialized Q-network."""

        return cls(config=config, model=MLP(config=config.mlp_config))

    def initialize(self, key: Any) -> Any:
        """Initialize and return the network parameter PyTree."""

        _, params = initialize_mlp(self.config.mlp_config, key)
        return params

    def apply(self, params: Any, observations: Any) -> jax.Array:
        """Return Q-values for one observation or a batch of observations."""

        return mlp_forward(self.model, params, observations)
