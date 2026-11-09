"""Streamlit-independent PPO agent compatible with the common agent interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training import train_state

from rl_platform.algorithms.ppo import (
    PPOBatch,
    PPOConfig,
    compute_gae,
    normalize_advantages,
    policy_log_prob,
    sample_policy_action,
    update_ppo_parameters,
)
from rl_platform.algorithms.rollout import RolloutBuffer
from rl_platform.networks.optimization import create_train_state
from rl_platform.networks.policy import Actor, Critic, PolicyNetworkConfig


@dataclass(frozen=True, slots=True)
class PPOUpdate:
    """Summary of one PPO rollout update across all minibatch epochs."""

    step: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    minibatches: int


@dataclass(frozen=True, slots=True)
class _PendingAction:
    observation: np.ndarray
    action: int
    log_prob: float
    value: float


@dataclass
class PPOAgent:
    """On-policy categorical PPO agent using separate actor and critic networks."""

    actor: Actor
    critic: Critic
    actor_state: train_state.TrainState
    critic_state: train_state.TrainState
    rollout: RolloutBuffer
    config: PPOConfig
    rng_key: Any
    training_steps: int = 0
    _pending_action: _PendingAction | None = None

    @classmethod
    def create(
        cls,
        observation_size: int,
        action_count: int,
        config: PPOConfig,
        key: Any,
    ) -> PPOAgent:
        """Initialize actor, critic, optimizer states, and rollout storage."""

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
            actor_state=create_train_state(
                actor.model,
                actor_params,
                learning_rate=config.learning_rate,
                optimizer_name=config.optimizer,
            ),
            critic_state=create_train_state(
                critic.model,
                critic_params,
                learning_rate=config.learning_rate,
                optimizer_name=config.optimizer,
            ),
            rollout=RolloutBuffer(config.rollout_length),
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
        """Select a categorical policy action and cache its old-policy statistics."""

        action_key = self._take_key() if key is None else key
        observation_array = jnp.asarray(observation)
        logits = self.actor.logits(self.actor_state.params, observation_array)
        action = sample_policy_action(logits, action_key, deterministic=deterministic)
        log_prob = policy_log_prob(logits, action)
        value = self.critic.values(self.critic_state.params, observation_array)
        self._pending_action = _PendingAction(
            observation=np.asarray(observation_array),
            action=int(action),
            log_prob=float(log_prob),
            value=float(value),
        )
        return int(action)

    def observe(self, transition: Any) -> None:
        """Add an environment transition and old-policy statistics to the rollout."""

        observation = np.asarray(transition.observation, dtype=np.float32)
        action = int(transition.action)
        if (
            self._pending_action is not None
            and self._pending_action.action == action
            and np.array_equal(self._pending_action.observation, observation)
        ):
            old_log_prob = self._pending_action.log_prob
            value = self._pending_action.value
        else:
            observation_array = jnp.asarray(observation)
            logits = self.actor.logits(self.actor_state.params, observation_array)
            old_log_prob = float(policy_log_prob(logits, action))
            value = float(self.critic.values(self.critic_state.params, observation_array))
        self.rollout.add(
            observation=observation,
            action=action,
            reward=float(transition.reward),
            next_observation=transition.next_observation,
            terminated=bool(transition.terminated),
            truncated=bool(transition.truncated),
            old_log_prob=old_log_prob,
            value=value,
        )
        self._pending_action = None

    def update(self, *, key: Any | None = None) -> PPOUpdate | None:
        """Optimize a completed rollout for multiple shuffled PPO epochs."""

        if not self.rollout.ready:
            return None
        rollout_batch = self.rollout.as_batch()
        observations = jnp.asarray(rollout_batch.observations)
        next_values = self.critic.values(
            self.critic_state.params, jnp.asarray(rollout_batch.next_observations)
        )
        advantages = compute_gae(
            rollout_batch.rewards,
            rollout_batch.values,
            next_values,
            rollout_batch.terminated,
            self.config.gamma,
            self.config.gae_lambda,
        )
        returns = jnp.asarray(rollout_batch.values) + advantages
        advantages = normalize_advantages(advantages, self.config.advantage_epsilon)
        training_key = self._take_key() if key is None else key
        batch = PPOBatch(
            observations=observations,
            actions=jnp.asarray(rollout_batch.actions),
            old_log_probs=jnp.asarray(rollout_batch.old_log_probs),
            advantages=advantages,
            returns=returns,
        )

        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        minibatch_count = 0
        for _ in range(self.config.epochs):
            training_key, permutation_key = jax.random.split(training_key)
            permutation = jax.random.permutation(permutation_key, batch.observations.shape[0])
            for start in range(0, batch.observations.shape[0], self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                minibatch = PPOBatch(
                    observations=batch.observations[indices],
                    actions=batch.actions[indices],
                    old_log_probs=batch.old_log_probs[indices],
                    advantages=batch.advantages[indices],
                    returns=batch.returns[indices],
                )
                self.actor_state, self.critic_state, loss, components = update_ppo_parameters(
                    self.actor_state,
                    self.critic_state,
                    self.actor,
                    self.critic,
                    minibatch,
                    self.config,
                )
                self._ensure_finite(loss, self.actor_state.params, self.critic_state.params)
                losses.append(float(loss))
                policy_losses.append(float(components["policy_loss"]))
                value_losses.append(float(components["value_loss"]))
                entropies.append(float(components["entropy"]))
                minibatch_count += 1
                self.training_steps += 1
        self.rollout.clear()
        return PPOUpdate(
            step=self.training_steps,
            loss=float(np.mean(losses)),
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            minibatches=minibatch_count,
        )

    def logits(self, observation: Any) -> jax.Array:
        """Return actor logits for inspection and evaluation."""

        return self.actor.logits(self.actor_state.params, jnp.asarray(observation))

    def value(self, observation: Any) -> jax.Array:
        """Return critic value for inspection."""

        return self.critic.values(self.critic_state.params, jnp.asarray(observation))

    def checkpoint_state(self) -> dict[str, Any]:
        """Return actor, critic, rollout, RNG, and progress state for persistence."""

        pending_action = None
        if self._pending_action is not None:
            pending_action = {
                "observation": self._pending_action.observation,
                "action": self._pending_action.action,
                "log_prob": self._pending_action.log_prob,
                "value": self._pending_action.value,
            }
        return {
            "actor_state": serialization.to_state_dict(self.actor_state),
            "critic_state": serialization.to_state_dict(self.critic_state),
            "rng_key": np.asarray(self.rng_key),
            "training_steps": self.training_steps,
            "rollout": self.rollout.state_dict(),
            "pending_action": pending_action,
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore a compatible PPO checkpoint into this initialized agent."""

        required = {
            "actor_state",
            "critic_state",
            "rng_key",
            "training_steps",
            "rollout",
            "pending_action",
        }
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"Checkpoint state is missing: {', '.join(missing)}")
        training_steps = state["training_steps"]
        if (
            isinstance(training_steps, bool)
            or not isinstance(training_steps, (int, np.integer))
            or training_steps < 0
        ):
            raise ValueError("Checkpoint training_steps must be a non-negative integer")
        restored_actor_state = serialization.from_state_dict(
            self.actor_state, state["actor_state"]
        )
        restored_critic_state = serialization.from_state_dict(
            self.critic_state, state["critic_state"]
        )
        self.rollout.load_state_dict(state["rollout"])
        pending_action = state["pending_action"]
        if pending_action is None:
            restored_pending = None
        elif isinstance(pending_action, dict):
            pending_required = {"observation", "action", "log_prob", "value"}
            if not pending_required.issubset(pending_action):
                raise ValueError("Checkpoint pending action is malformed")
            restored_pending = _PendingAction(
                observation=np.asarray(pending_action["observation"], dtype=np.float32),
                action=int(pending_action["action"]),
                log_prob=float(pending_action["log_prob"]),
                value=float(pending_action["value"]),
            )
        else:
            raise ValueError("Checkpoint pending action is malformed")
        self.actor_state = restored_actor_state
        self.critic_state = restored_critic_state
        self.rng_key = jnp.asarray(state["rng_key"], dtype=jnp.uint32)
        self.training_steps = int(training_steps)
        self._pending_action = restored_pending
        self._ensure_finite(0.0, self.actor_state.params, self.critic_state.params)

    def _take_key(self) -> Any:
        self.rng_key, action_key = jax.random.split(self.rng_key)
        return action_key

    @staticmethod
    def _ensure_finite(loss: Any, *parameter_trees: Any) -> None:
        if not np.isfinite(float(loss)):
            raise FloatingPointError("PPO loss is NaN or Inf")
        if not all(
            np.isfinite(np.asarray(leaf)).all()
            for tree in parameter_trees
            for leaf in jax.tree_util.tree_leaves(tree)
        ):
            raise FloatingPointError("PPO parameters contain NaN or Inf")
