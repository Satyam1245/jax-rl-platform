"""PPO objectives, return estimation, and optimizer integration.

Note: policy_log_prob, policy_entropy, and sample_policy_action are imported
from networks.distributions for internal use only. They are re-exported at the
rl_platform.algorithms package level directly from networks.distributions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from rl_platform.algorithms.rollout import RolloutBatch
from rl_platform.networks.distributions import (
    policy_entropy,
    policy_log_prob,
    sample_policy_action,
)
from rl_platform.networks.policy import Actor, Critic


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Validated PPO hyperparameters and network settings."""

    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    rollout_length: int
    minibatch_size: int
    epochs: int
    hidden_layers: int
    hidden_size: int
    activation: str
    optimizer: str
    advantage_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        _unit_interval(self.gamma, "gamma")
        _unit_interval(self.gae_lambda, "gae_lambda")
        if not isinstance(self.clip_epsilon, (int, float)) or self.clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be greater than zero")
        if not isinstance(self.value_coefficient, (int, float)) or self.value_coefficient < 0:
            raise ValueError("value_coefficient must be non-negative")
        if not isinstance(self.entropy_coefficient, (int, float)) or self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative")
        _positive_int(self.rollout_length, "rollout_length")
        _positive_int(self.minibatch_size, "minibatch_size")
        _positive_int(self.epochs, "epochs")
        if self.minibatch_size > self.rollout_length:
            raise ValueError("minibatch_size must be <= rollout_length")
        if isinstance(self.hidden_layers, bool) or not isinstance(self.hidden_layers, int) or self.hidden_layers < 0:
            raise ValueError("hidden_layers must be a non-negative integer")
        _positive_int(self.hidden_size, "hidden_size")
        if not isinstance(self.activation, str) or not self.activation.strip():
            raise ValueError("activation must be a non-empty string")
        if not isinstance(self.optimizer, str) or not self.optimizer.strip():
            raise ValueError("optimizer must be a non-empty string")
        if not isinstance(self.advantage_epsilon, (int, float)) or self.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be greater than zero")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> PPOConfig:
        """Create PPO settings from a strict hyperparameter mapping."""

        required = {
            "learning_rate", "gamma", "gae_lambda", "clip_epsilon", "value_coefficient",
            "entropy_coefficient", "rollout_length", "minibatch_size", "epochs",
            "hidden_layers", "hidden_size", "activation", "optimizer",
        }
        missing = sorted(required - set(values))
        unknown = sorted(set(values) - required - {"advantage_epsilon"})
        if missing:
            raise ValueError(f"Missing PPO hyperparameter(s): {', '.join(missing)}")
        if unknown:
            raise ValueError(f"Unknown PPO hyperparameter(s): {', '.join(unknown)}")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid PPO hyperparameters: {exc}") from exc


@dataclass(frozen=True, slots=True)
class PPOBatch:
    """JAX batch used by the PPO optimizer."""

    observations: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    advantages: jax.Array
    returns: jax.Array


def compute_returns(
    rewards: Any,
    next_values: Any,
    terminated: Any,
    gamma: float,
) -> jax.Array:
    """Compute one-step bootstrapped returns, masking true terminals only."""

    _unit_interval(gamma, "gamma")
    return (
        jnp.asarray(rewards, dtype=jnp.float32)
        + gamma
        * (1.0 - jnp.asarray(terminated, dtype=jnp.float32))
        * jnp.asarray(next_values, dtype=jnp.float32)
    )


def compute_gae(
    rewards: Any,
    values: Any,
    next_values: Any,
    terminated: Any,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Compute generalized advantage estimates with reverse-time recursion."""

    _unit_interval(gamma, "gamma")
    _unit_interval(gae_lambda, "gae_lambda")
    rewards_array = jnp.asarray(rewards, dtype=jnp.float32)
    values_array = jnp.asarray(values, dtype=jnp.float32)
    next_values_array = jnp.asarray(next_values, dtype=jnp.float32)
    terminal_array = jnp.asarray(terminated, dtype=jnp.float32)
    deltas = rewards_array + gamma * (1.0 - terminal_array) * next_values_array - values_array

    def scan_step(gae, inputs):
        delta, terminal = inputs
        next_gae = delta + gamma * gae_lambda * (1.0 - terminal) * gae
        return next_gae, next_gae

    _, reversed_advantages = jax.lax.scan(
        scan_step,
        jnp.asarray(0.0, dtype=jnp.float32),
        (deltas[::-1], terminal_array[::-1]),
    )
    return reversed_advantages[::-1]


def normalize_advantages(advantages: Any, epsilon: float = 1e-8) -> jax.Array:
    """Normalize advantages to approximately zero mean and unit variance."""

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")
    values = jnp.asarray(advantages, dtype=jnp.float32)
    return (values - jnp.mean(values)) / (jnp.std(values) + epsilon)


def clipped_policy_objective(
    new_log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
    clip_epsilon: float,
) -> jax.Array:
    """Return the minimized clipped PPO policy loss."""

    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be greater than zero")
    ratio = jnp.exp(jnp.asarray(new_log_probs) - jnp.asarray(old_log_probs))
    advantage_values = jnp.asarray(advantages)
    unclipped = ratio * advantage_values
    clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_values
    return -jnp.mean(jnp.minimum(unclipped, clipped))


def value_loss(values: Any, returns: Any) -> jax.Array:
    """Return half mean-squared value error."""

    return 0.5 * jnp.mean((jnp.asarray(values) - jnp.asarray(returns)) ** 2)


def entropy_bonus(logits: Any) -> jax.Array:
    """Return mean policy entropy for the entropy bonus term."""

    return jnp.mean(policy_entropy(logits))


def ppo_loss(
    actor_params: Any,
    critic_params: Any,
    actor: Actor,
    critic: Critic,
    batch: PPOBatch,
    config: PPOConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute total PPO loss and its policy/value/entropy components."""

    logits = actor.logits(actor_params, batch.observations)
    new_log_probs = policy_log_prob(logits, batch.actions)
    policy_loss = clipped_policy_objective(
        new_log_probs, batch.old_log_probs, batch.advantages, config.clip_epsilon
    )
    values = critic.values(critic_params, batch.observations)
    current_value_loss = value_loss(values, batch.returns)
    entropy = entropy_bonus(logits)
    total = policy_loss + config.value_coefficient * current_value_loss - config.entropy_coefficient * entropy
    return total, {"policy_loss": policy_loss, "value_loss": current_value_loss, "entropy": entropy}


def update_ppo_parameters(
    actor_state: Any,
    critic_state: Any,
    actor: Actor,
    critic: Critic,
    batch: PPOBatch,
    config: PPOConfig,
) -> tuple[Any, Any, jax.Array, dict[str, jax.Array]]:
    """Apply one joint actor/critic optimizer update."""

    def loss_fn(actor_params: Any, critic_params: Any):
        return ppo_loss(actor_params, critic_params, actor, critic, batch, config)

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
