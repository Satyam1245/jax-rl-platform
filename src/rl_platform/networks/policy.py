"""Actor and critic MLP facades used by PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from rl_platform.networks.mlp import MLP, MLPConfig, initialize_mlp, mlp_forward


@dataclass(frozen=True, slots=True)
class PolicyNetworkConfig:
    """Shared architecture settings for discrete actor and scalar critic."""

    observation_size: int
    action_count: int
    hidden_layers: int
    hidden_size: int
    activation: str

    def __post_init__(self) -> None:
        if isinstance(self.action_count, bool) or not isinstance(self.action_count, int):
            raise ValueError("action_count must be a positive integer")
        if self.action_count < 2:
            raise ValueError("action_count must be at least two for a categorical policy")
        _actor_config(self)
        _critic_config(self)


@dataclass(frozen=True, slots=True)
class Actor:
    """Categorical-policy network producing one logit per action."""

    config: PolicyNetworkConfig
    model: MLP

    @classmethod
    def create(cls, config: PolicyNetworkConfig) -> Actor:
        return cls(config=config, model=MLP(_actor_config(config)))

    def initialize(self, key: Any) -> Any:
        _, params = initialize_mlp(_actor_config(self.config), key)
        return params

    def logits(self, params: Any, observations: Any) -> jax.Array:
        """Return policy logits for one observation or a batch."""

        return mlp_forward(self.model, params, observations)


@dataclass(frozen=True, slots=True)
class Critic:
    """Value network producing one scalar state value."""

    config: PolicyNetworkConfig
    model: MLP

    @classmethod
    def create(cls, config: PolicyNetworkConfig) -> Critic:
        return cls(config=config, model=MLP(_critic_config(config)))

    def initialize(self, key: Any) -> Any:
        _, params = initialize_mlp(_critic_config(self.config), key)
        return params

    def values(self, params: Any, observations: Any) -> jax.Array:
        """Return scalar values with shape ``(batch,)`` or scalar ``()``."""

        values = mlp_forward(self.model, params, observations)
        return jnp.squeeze(values, axis=-1)


def _actor_config(config: PolicyNetworkConfig) -> MLPConfig:
    return MLPConfig(
        input_size=config.observation_size,
        hidden_layers=config.hidden_layers,
        hidden_size=config.hidden_size,
        output_size=config.action_count,
        activation=config.activation,
    )


def _critic_config(config: PolicyNetworkConfig) -> MLPConfig:
    return MLPConfig(
        input_size=config.observation_size,
        hidden_layers=config.hidden_layers,
        hidden_size=config.hidden_size,
        output_size=1,
        activation=config.activation,
    )
