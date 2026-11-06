"""Streamlit-independent DQN agent built from the reusable foundation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training import train_state

from rl_platform.algorithms.dqn import (
    DQNConfig,
    EpsilonScheduler,
    bellman_target,
    dqn_loss,
    epsilon_greedy_action,
    update_dqn_parameters,
    update_target_network,
)
from rl_platform.algorithms.replay import ReplayBatch, ReplayBuffer, Transition
from rl_platform.networks.optimization import create_train_state
from rl_platform.networks.q_network import QNetwork, QNetworkConfig


@dataclass(frozen=True, slots=True)
class DQNUpdate:
    """Summary of one successful DQN optimizer update."""

    step: int
    loss: float
    epsilon: float
    target_updated: bool


@dataclass
class DQNAgent:
    """A discrete-action DQN agent with online and target networks."""

    q_network: QNetwork
    online_state: train_state.TrainState
    target_params: Any
    replay_buffer: ReplayBuffer
    epsilon_scheduler: EpsilonScheduler
    gamma: float
    batch_size: int
    min_replay_size: int
    target_update_interval: int
    rng_key: Any
    training_steps: int = 0

    @classmethod
    def create(
        cls,
        observation_size: int,
        action_count: int,
        config: DQNConfig,
        key: Any,
    ) -> DQNAgent:
        """Initialize an agent and identical online/target parameter trees."""

        network = QNetwork.create(
            QNetworkConfig(
                observation_size=observation_size,
                action_count=action_count,
                hidden_layers=config.hidden_layers,
                hidden_size=config.hidden_size,
                activation=config.activation,
            )
        )
        initialization_key, agent_key = jax.random.split(key)
        params = network.initialize(initialization_key)
        online_state = create_train_state(
            network.model,
            params,
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer,
        )
        target_params = jax.tree_util.tree_map(lambda value: value, params)
        return cls(
            q_network=network,
            online_state=online_state,
            target_params=target_params,
            replay_buffer=ReplayBuffer(config.replay_capacity),
            epsilon_scheduler=EpsilonScheduler(
                config.epsilon_start,
                config.epsilon_end,
                config.epsilon_decay_steps,
            ),
            gamma=config.gamma,
            batch_size=config.batch_size,
            min_replay_size=config.min_replay_size,
            target_update_interval=config.target_update_interval,
            rng_key=agent_key,
        )

    @property
    def parameters(self) -> Any:
        """Return current online parameters."""

        return self.online_state.params

    @property
    def action_count(self) -> int:
        """Return the number of discrete actions."""

        return self.q_network.config.action_count

    @property
    def epsilon(self) -> float:
        """Return epsilon at the current optimizer-step count."""

        return self.epsilon_scheduler.value(self.training_steps)

    def q_values(self, observation: Any) -> jax.Array:
        """Return online Q-values for one observation."""

        return self.q_network.apply(self.online_state.params, jnp.asarray(observation))

    def select_action(
        self,
        observation: Any,
        *,
        key: Any | None = None,
        deterministic: bool = False,
    ) -> int:
        """Select an epsilon-greedy action using an explicit or agent key."""

        action_key = self._take_key() if key is None else key
        epsilon = 0.0 if deterministic else self.epsilon
        action = epsilon_greedy_action(self.q_values(observation), action_key, epsilon)
        return int(action)

    def observe(self, transition: Transition) -> None:
        """Add one environment transition to replay memory."""

        self.replay_buffer.add_transition(transition)

    def update(self, *, key: Any | None = None) -> DQNUpdate | None:
        """Sample replay data and apply one DQN update when warmup is complete."""

        if len(self.replay_buffer) < self.min_replay_size:
            return None
        sample_key = self._take_key() if key is None else key
        batch = self.replay_buffer.sample(self.batch_size, sample_key)
        updated_state, loss = update_dqn_parameters(
            self.online_state,
            self.target_params,
            self.q_network,
            batch,
            self.gamma,
        )
        loss_value = float(loss)
        self._ensure_finite(updated_state.params, loss_value)
        self.online_state = updated_state
        self.training_steps += 1
        target_updated = self.training_steps % self.target_update_interval == 0
        if target_updated:
            self.update_target_network()
        return DQNUpdate(
            step=self.training_steps,
            loss=loss_value,
            epsilon=self.epsilon,
            target_updated=target_updated,
        )

    def update_target_network(self, *, tau: float = 1.0) -> None:
        """Update target parameters from the online parameters."""

        self.target_params = update_target_network(
            self.online_state.params,
            self.target_params,
            tau=tau,
        )

    def bellman_target(self, batch: ReplayBatch) -> jax.Array:
        """Compute targets using this agent's target-network parameters."""

        next_q_values = self.q_network.apply(self.target_params, batch.next_observations)
        return bellman_target(batch.rewards, next_q_values, batch.terminated, self.gamma)

    def loss(self, batch: ReplayBatch) -> jax.Array:
        """Compute the current online/target-network loss for a batch."""

        return dqn_loss(
            self.online_state.params,
            self.target_params,
            self.q_network,
            batch,
            self.gamma,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """Return online, target, optimizer, RNG, progress, and replay state."""

        return {
            "online_state": serialization.to_state_dict(self.online_state),
            "target_params": serialization.to_state_dict(self.target_params),
            "rng_key": np.asarray(self.rng_key),
            "training_steps": self.training_steps,
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore a compatible checkpoint state into this initialized agent."""

        required = {"online_state", "target_params", "rng_key", "training_steps", "replay_buffer"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"Checkpoint state is missing: {', '.join(missing)}")
        restored_online_state = serialization.from_state_dict(
            self.online_state, state["online_state"]
        )
        restored_target_params = serialization.from_state_dict(
            self.target_params, state["target_params"]
        )
        training_steps = state["training_steps"]
        if isinstance(training_steps, bool) or not isinstance(training_steps, (int, np.integer)):
            raise ValueError("Checkpoint training_steps must be a non-negative integer")
        if int(training_steps) < 0:
            raise ValueError("Checkpoint training_steps must be a non-negative integer")
        self.replay_buffer.load_state_dict(state["replay_buffer"])
        self.online_state = restored_online_state
        self.target_params = restored_target_params
        self.rng_key = jnp.asarray(state["rng_key"], dtype=jnp.uint32)
        self.training_steps = int(training_steps)
        self._ensure_finite(self.online_state.params, 0.0)
        self._ensure_finite(self.target_params, 0.0)

    def _take_key(self) -> Any:
        self.rng_key, action_key = jax.random.split(self.rng_key)
        return action_key

    @staticmethod
    def _ensure_finite(params: Any, loss: float) -> None:
        if not np.isfinite(loss):
            raise FloatingPointError("DQN loss is NaN or Inf")
        if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(params)):
            raise FloatingPointError("DQN parameters contain NaN or Inf")
