"""Reusable DQN mathematics and optimizer integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from rl_platform.algorithms.replay import ReplayBatch
from rl_platform.networks.optimization import optimizer_step
from rl_platform.networks.q_network import QNetwork


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Validated hyperparameters and network settings for a DQN agent."""

    learning_rate: float
    gamma: float
    batch_size: int
    replay_capacity: int
    min_replay_size: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    hidden_layers: int
    hidden_size: int
    activation: str
    optimizer: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> DQNConfig:
        """Create a DQN configuration from validated YAML hyperparameters."""

        required = {
            "learning_rate",
            "gamma",
            "batch_size",
            "replay_capacity",
            "min_replay_size",
            "target_update_interval",
            "epsilon_start",
            "epsilon_end",
            "epsilon_decay_steps",
            "hidden_layers",
            "hidden_size",
            "activation",
            "optimizer",
        }
        missing = sorted(required - set(values))
        unknown = sorted(set(values) - required)
        if missing:
            raise ValueError(f"Missing DQN hyperparameter(s): {', '.join(missing)}")
        if unknown:
            raise ValueError(f"Unknown DQN hyperparameter(s): {', '.join(unknown)}")
        try:
            return cls(**dict(values))
        except TypeError as exc:
            raise ValueError(f"Invalid DQN hyperparameters: {exc}") from exc

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.replay_capacity, "replay_capacity")
        _positive_int(self.min_replay_size, "min_replay_size")
        _positive_int(self.target_update_interval, "target_update_interval")
        if self.min_replay_size < self.batch_size:
            raise ValueError("min_replay_size must be >= batch_size")
        if self.min_replay_size > self.replay_capacity:
            raise ValueError("min_replay_size must be <= replay_capacity")
        _probability(self.epsilon_start, "epsilon_start")
        _probability(self.epsilon_end, "epsilon_end")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must be <= epsilon_start")
        _positive_int(self.epsilon_decay_steps, "epsilon_decay_steps")
        _non_negative_int(self.hidden_layers, "hidden_layers")
        _positive_int(self.hidden_size, "hidden_size")
        if not isinstance(self.activation, str) or not self.activation.strip():
            raise ValueError("activation must be a non-empty string")
        if not isinstance(self.optimizer, str) or not self.optimizer.strip():
            raise ValueError("optimizer must be a non-empty string")


@dataclass(frozen=True, slots=True)
class EpsilonScheduler:
    """Linear epsilon schedule with explicit, bounded decay."""

    start: float
    end: float
    decay_steps: int

    def __post_init__(self) -> None:
        _probability(self.start, "epsilon start")
        _probability(self.end, "epsilon end")
        if self.end > self.start:
            raise ValueError("epsilon end must be <= epsilon start")
        _positive_int(self.decay_steps, "epsilon decay_steps")

    def value(self, step: int) -> float:
        """Return epsilon at a non-negative training step."""

        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Epsilon step must be a non-negative integer")
        progress = min(step, self.decay_steps) / self.decay_steps
        return self.start + (self.end - self.start) * progress


def epsilon_greedy_action(q_values: Any, key: Any, epsilon: float) -> jax.Array:
    """Select a random or greedy discrete action using an explicit key."""

    _probability(epsilon, "epsilon")
    values = jnp.asarray(q_values)
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError("q_values must have shape (action_count,)")
    explore_key, action_key = jax.random.split(key)
    explore = jax.random.uniform(explore_key) < epsilon
    random_action = jax.random.randint(action_key, shape=(), minval=0, maxval=values.shape[0])
    greedy_action = jnp.argmax(values).astype(jnp.int32)
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


def bellman_target(
    rewards: Any,
    next_q_values: Any,
    terminated: Any,
    gamma: float,
) -> jax.Array:
    """Compute one-step Bellman targets.

    The target does not bootstrap after ``terminated=True``. A truncated
    transition remains bootstrap-able because truncation is an external time
    limit rather than an absorbing MDP terminal state.
    """

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    reward_values = jnp.asarray(rewards, dtype=jnp.float32)
    next_values = jnp.asarray(next_q_values, dtype=jnp.float32)
    terminal_values = jnp.asarray(terminated, dtype=jnp.float32)
    max_next_values = next_values if next_values.ndim == 1 else jnp.max(next_values, axis=-1)
    return reward_values + gamma * (1.0 - terminal_values) * max_next_values


def dqn_loss(
    online_params: Any,
    target_params: Any,
    q_network: QNetwork,
    batch: ReplayBatch,
    gamma: float,
) -> jax.Array:
    """Compute the mean-squared temporal-difference loss for a batch."""

    online_q_values = q_network.apply(online_params, batch.observations)
    actions = jnp.asarray(batch.actions, dtype=jnp.int32).reshape((-1, 1))
    selected_q_values = jnp.take_along_axis(online_q_values, actions, axis=1).squeeze(axis=1)
    next_q_values = q_network.apply(target_params, batch.next_observations)
    targets = jax.lax.stop_gradient(
        bellman_target(batch.rewards, next_q_values, batch.terminated, gamma)
    )
    return jnp.mean((selected_q_values - targets) ** 2)


def update_dqn_parameters(
    state: Any,
    target_params: Any,
    q_network: QNetwork,
    batch: ReplayBatch,
    gamma: float,
) -> tuple[Any, jax.Array]:
    """Apply one generic Optax update to online DQN parameters."""

    def loss_fn(online_params: Any) -> jax.Array:
        return dqn_loss(online_params, target_params, q_network, batch, gamma)

    return optimizer_step(state, loss_fn)


def update_target_network(online_params: Any, target_params: Any, tau: float = 1.0) -> Any:
    """Update target parameters by Polyak averaging; ``tau=1`` is a hard copy."""

    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in the interval (0, 1]")
    return jax.tree_util.tree_map(
        lambda online, target: tau * online + (1.0 - tau) * target,
        online_params,
        target_params,
    )


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _non_negative_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _probability(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
