"""Standard one-step Actor-Critic objectives and optimizer integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from rl_platform.networks.distributions import policy_entropy, policy_log_prob
from rl_platform.networks.policy import Actor, Critic


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    """Validated hyperparameters and network settings for one-step Actor-Critic."""

    learning_rate: float
    gamma: float
    value_coefficient: float
    entropy_coefficient: float
    hidden_layers: int
    hidden_size: int
    activation: str
    optimizer: str

    def __post_init__(self) -> None:
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        _unit_interval(self.gamma, "gamma")
        if isinstance(self.value_coefficient, bool) or not isinstance(self.value_coefficient, (int, float)) or self.value_coefficient < 0:
            raise ValueError("value_coefficient must be non-negative")
        if isinstance(self.entropy_coefficient, bool) or not isinstance(self.entropy_coefficient, (int, float)) or self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative")
        if isinstance(self.hidden_layers, bool) or not isinstance(self.hidden_layers, int) or self.hidden_layers < 0:
            raise ValueError("hidden_layers must be a non-negative integer")
        _positive_int(self.hidden_size, "hidden_size")
        if not isinstance(self.activation, str) or not self.activation.strip():
            raise ValueError("activation must be a non-empty string")
        if not isinstance(self.optimizer, str) or not self.optimizer.strip():
            raise ValueError("optimizer must be a non-empty string")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ActorCriticConfig:
        """Create settings from a strict algorithm hyperparameter mapping."""

        required = {
            "learning_rate", "gamma", "value_coefficient", "entropy_coefficient",
            "hidden_layers", "hidden_size", "activation", "optimizer",
        }
        missing = sorted(required - set(values))
        unknown = sorted(set(values) - required)
        if missing:
            raise ValueError(f"Missing Actor-Critic hyperparameter(s): {', '.join(missing)}")
        if unknown:
            raise ValueError(f"Unknown Actor-Critic hyperparameter(s): {', '.join(unknown)}")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid Actor-Critic hyperparameters: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ActorCriticBatch:
    """JAX batch for one-step Actor-Critic updates."""

    observations: jax.Array
    actions: jax.Array
    returns: jax.Array
    advantages: jax.Array


def actor_critic_returns(
    rewards: Any,
    next_values: Any,
    terminated: Any,
    gamma: float,
) -> jax.Array:
    """Compute one-step TD returns, masking true terminals only."""

    _unit_interval(gamma, "gamma")
    return (
        jnp.asarray(rewards, dtype=jnp.float32)
        + gamma
        * (1.0 - jnp.asarray(terminated, dtype=jnp.float32))
        * jnp.asarray(next_values, dtype=jnp.float32)
    )


def compute_advantages(returns: Any, values: Any) -> jax.Array:
    """Compute the one-step advantage estimate ``return - V(s)``."""

    return jnp.asarray(returns, dtype=jnp.float32) - jnp.asarray(values, dtype=jnp.float32)


def actor_critic_policy_loss(logits: Any, actions: Any, advantages: Any) -> jax.Array:
    """Return policy-gradient loss with a stopped-gradient advantage."""

    log_probs = policy_log_prob(logits, actions)
    return -jnp.mean(log_probs * jax.lax.stop_gradient(jnp.asarray(advantages)))


def actor_critic_value_loss(values: Any, returns: Any) -> jax.Array:
    """Return half mean-squared error between values and TD returns."""

    return 0.5 * jnp.mean(
        (jnp.asarray(values, dtype=jnp.float32) - jnp.asarray(returns, dtype=jnp.float32)) ** 2
    )


def actor_critic_entropy(logits: Any) -> jax.Array:
    """Return mean categorical policy entropy."""

    return jnp.mean(policy_entropy(logits))


def actor_critic_loss(
    actor_params: Any,
    critic_params: Any,
    actor: Actor,
    critic: Critic,
    batch: ActorCriticBatch,
    config: ActorCriticConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute total Actor-Critic loss and its component terms."""

    logits = actor.logits(actor_params, batch.observations)
    values = critic.values(critic_params, batch.observations)
    policy_loss = actor_critic_policy_loss(logits, batch.actions, batch.advantages)
    current_value_loss = actor_critic_value_loss(values, batch.returns)
    entropy = actor_critic_entropy(logits)
    total = policy_loss + config.value_coefficient * current_value_loss - config.entropy_coefficient * entropy
    return total, {
        "policy_loss": policy_loss,
        "value_loss": current_value_loss,
        "entropy": entropy,
    }


def update_actor_critic_parameters(
    actor_state: Any,
    critic_state: Any,
    actor: Actor,
    critic: Critic,
    batch: ActorCriticBatch,
    config: ActorCriticConfig,
) -> tuple[Any, Any, jax.Array, dict[str, jax.Array]]:
    """Apply one joint gradient update to actor and critic states."""

    def loss_fn(actor_params: Any, critic_params: Any):
        return actor_critic_loss(actor_params, critic_params, actor, critic, batch, config)

    (loss, components), (actor_gradients, critic_gradients) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(actor_state.params, critic_state.params)
    return (
        actor_state.apply_gradients(grads=actor_gradients),
        critic_state.apply_gradients(grads=critic_gradients),
        loss,
        components,
    )


def _unit_interval(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
