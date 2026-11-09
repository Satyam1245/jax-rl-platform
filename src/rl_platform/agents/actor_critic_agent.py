"""Streamlit-independent standard one-step Actor-Critic agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training import train_state

from rl_platform.algorithms.actor_critic import (
    ActorCriticBatch,
    ActorCriticConfig,
    actor_critic_returns,
    compute_advantages,
    update_actor_critic_parameters,
)
from rl_platform.networks.optimization import create_train_state
from rl_platform.networks.distributions import sample_policy_action
from rl_platform.networks.policy import Actor, Critic, PolicyNetworkConfig


@dataclass(frozen=True, slots=True)
class ActorCriticUpdate:
    """Summary of one Actor-Critic transition update."""

    step: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float


@dataclass
class ActorCriticAgent:
    """A discrete-action one-step Actor-Critic agent."""

    actor: Actor
    critic: Critic
    actor_state: train_state.TrainState
    critic_state: train_state.TrainState
    config: ActorCriticConfig
    rng_key: Any
    training_steps: int = 0
    _transition: Any | None = None

    @classmethod
    def create(
        cls,
        observation_size: int,
        action_count: int,
        config: ActorCriticConfig,
        key: Any,
    ) -> ActorCriticAgent:
        """Initialize actor, critic, and their independent optimizer states."""

        network_config = PolicyNetworkConfig(
            observation_size=observation_size,
            action_count=action_count,
            hidden_layers=config.hidden_layers,
            hidden_size=config.hidden_size,
            activation=config.activation,
        )
        actor = Actor.create(network_config)
        critic = Critic.create(network_config)
        actor_key, critic_key, agent_key = jax.random.split(key, 3)
        actor_params = actor.initialize(actor_key)
        critic_params = critic.initialize(critic_key)
        return cls(
            actor=actor,
            critic=critic,
            actor_state=create_train_state(actor.model, actor_params, config.learning_rate, optimizer_name=config.optimizer),
            critic_state=create_train_state(critic.model, critic_params, config.learning_rate, optimizer_name=config.optimizer),
            config=config,
            rng_key=agent_key,
        )

    @property
    def action_count(self) -> int:
        """Return the discrete action count."""

        return self.actor.config.action_count

    @property
    def parameters(self) -> dict[str, Any]:
        """Return actor and critic parameter trees."""

        return {"actor": self.actor_state.params, "critic": self.critic_state.params}

    def select_action(
        self,
        observation: Any,
        *,
        key: Any | None = None,
        deterministic: bool = False,
    ) -> int:
        """Select a stochastic training action or deterministic policy mode."""

        action_key = self._take_key() if key is None else key
        logits = self.actor.logits(self.actor_state.params, jnp.asarray(observation))
        return int(sample_policy_action(logits, action_key, deterministic=deterministic))

    def observe(self, transition: Any) -> None:
        """Store the latest transition for the next one-step update."""

        self._transition = transition

    def update(self, *, key: Any | None = None) -> ActorCriticUpdate | None:
        """Apply one update for the observed transition."""

        if self._transition is None:
            return None
        transition = self._transition
        observation = jnp.asarray(transition.observation)
        next_observation = jnp.asarray(transition.next_observation)
        value = self.critic.values(self.critic_state.params, observation)
        next_value = self.critic.values(self.critic_state.params, next_observation)
        returns = actor_critic_returns(
            transition.reward,
            next_value,
            transition.terminated,
            self.config.gamma,
        )
        advantages = compute_advantages(returns, value)
        batch = ActorCriticBatch(
            observations=observation[None, ...],
            actions=jnp.asarray([transition.action], dtype=jnp.int32),
            returns=jnp.asarray([returns], dtype=jnp.float32),
            advantages=jnp.asarray([advantages], dtype=jnp.float32),
        )
        self.actor_state, self.critic_state, loss, components = update_actor_critic_parameters(
            self.actor_state,
            self.critic_state,
            self.actor,
            self.critic,
            batch,
            self.config,
        )
        self._ensure_finite(loss, self.actor_state.params, self.critic_state.params)
        self._transition = None
        self.training_steps += 1
        return ActorCriticUpdate(
            step=self.training_steps,
            loss=float(loss),
            policy_loss=float(components["policy_loss"]),
            value_loss=float(components["value_loss"]),
            entropy=float(components["entropy"]),
        )

    def logits(self, observation: Any) -> jax.Array:
        """Return actor logits for inspection."""

        return self.actor.logits(self.actor_state.params, jnp.asarray(observation))

    def value(self, observation: Any) -> jax.Array:
        """Return the critic's scalar value estimate."""

        return self.critic.values(self.critic_state.params, jnp.asarray(observation))

    def checkpoint_state(self) -> dict[str, Any]:
        """Return model, optimizer, RNG, and progress state for persistence."""

        return {
            "actor_state": serialization.to_state_dict(self.actor_state),
            "critic_state": serialization.to_state_dict(self.critic_state),
            "rng_key": np.asarray(self.rng_key),
            "training_steps": self.training_steps,
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore a compatible persisted state into this initialized agent."""

        required = {"actor_state", "critic_state", "rng_key", "training_steps"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"Checkpoint state is missing: {', '.join(missing)}")
        training_steps = state["training_steps"]
        if isinstance(training_steps, bool) or not isinstance(training_steps, (int, np.integer)) or training_steps < 0:
            raise ValueError("Checkpoint training_steps must be a non-negative integer")
        self.actor_state = serialization.from_state_dict(self.actor_state, state["actor_state"])
        self.critic_state = serialization.from_state_dict(self.critic_state, state["critic_state"])
        self.rng_key = jnp.asarray(state["rng_key"], dtype=jnp.uint32)
        self.training_steps = int(training_steps)
        self._ensure_finite(0.0, self.actor_state.params, self.critic_state.params)

    def _take_key(self) -> Any:
        self.rng_key, action_key = jax.random.split(self.rng_key)
        return action_key

    @staticmethod
    def _ensure_finite(loss: Any, *parameter_trees: Any) -> None:
        if not np.isfinite(float(loss)):
            raise FloatingPointError("Actor-Critic loss is NaN or Inf")
        if not all(
            np.isfinite(np.asarray(leaf)).all()
            for tree in parameter_trees
            for leaf in jax.tree_util.tree_leaves(tree)
        ):
            raise FloatingPointError("Actor-Critic parameters contain NaN or Inf")
