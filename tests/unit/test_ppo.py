"""Unit tests for PPO networks, estimators, objectives, and updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_platform.agents import PPOAgent
from rl_platform.algorithms import (
    PPOBatch,
    PPOConfig,
    clipped_policy_objective,
    compute_gae,
    compute_returns,
    entropy_bonus,
    normalize_advantages,
    policy_entropy,
    policy_log_prob,
    sample_policy_action,
    update_ppo_parameters,
    value_loss,
)
from rl_platform.networks import Actor, Critic, PolicyNetworkConfig
from rl_platform.training import EnvironmentTransition
from rl_platform.utils.random import create_prng_key


def make_config(**overrides: object) -> PPOConfig:
    values: dict[str, object] = {
        "learning_rate": 0.01,
        "gamma": 0.9,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "rollout_length": 4,
        "minibatch_size": 2,
        "epochs": 2,
        "hidden_layers": 1,
        "hidden_size": 8,
        "activation": "tanh",
        "optimizer": "adam",
    }
    values.update(overrides)
    return PPOConfig(**values)  # type: ignore[arg-type]


def make_networks() -> tuple[Actor, Critic, object, object]:
    network_config = PolicyNetworkConfig(
        observation_size=3,
        action_count=2,
        hidden_layers=1,
        hidden_size=8,
        activation="tanh",
    )
    actor = Actor.create(network_config)
    critic = Critic.create(network_config)
    actor_params = actor.initialize(create_prng_key(1))
    critic_params = critic.initialize(create_prng_key(2))
    return actor, critic, actor_params, critic_params


def test_actor_policy_and_critic_value_outputs() -> None:
    actor, critic, actor_params, critic_params = make_networks()
    observations = jnp.ones((4, 3))
    assert actor.logits(actor_params, observations).shape == (4, 2)
    assert critic.values(critic_params, observations).shape == (4,)
    assert actor.logits(actor_params, observations[0]).shape == (2,)
    assert critic.values(critic_params, observations[0]).shape == ()


def test_policy_log_probability_entropy_and_sampling() -> None:
    logits = jnp.asarray([[2.0, 0.0], [0.0, 2.0]])
    log_probs = policy_log_prob(logits, jnp.asarray([0, 1]))
    assert log_probs.shape == (2,)
    assert np.all(np.asarray(log_probs) < 0.0)
    assert policy_entropy(logits).shape == (2,)
    assert float(entropy_bonus(logits)) > 0.0
    assert int(sample_policy_action(logits[0], create_prng_key(3), deterministic=True)) == 0


def test_returns_and_gae_mask_terminated_transitions() -> None:
    rewards = jnp.asarray([1.0, 2.0, 3.0])
    values = jnp.asarray([0.5, 1.0, 1.5])
    next_values = jnp.asarray([1.0, 2.0, 4.0])
    terminated = jnp.asarray([False, True, False])
    returns = compute_returns(rewards, next_values, terminated, gamma=0.5)
    np.testing.assert_allclose(returns, [1.5, 2.0, 5.0])
    advantages = compute_gae(rewards, values, next_values, terminated, 0.5, 1.0)
    np.testing.assert_allclose(advantages, [1.5, 1.0, 3.5])


def test_advantage_normalization() -> None:
    normalized = normalize_advantages(jnp.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(jnp.mean(normalized), 0.0, atol=1e-6)
    np.testing.assert_allclose(jnp.std(normalized), 1.0, atol=1e-5)


def test_clipped_policy_objective_clips_large_positive_ratio() -> None:
    objective = clipped_policy_objective(
        new_log_probs=jnp.log(jnp.asarray([2.0])),
        old_log_probs=jnp.zeros(1),
        advantages=jnp.ones(1),
        clip_epsilon=0.2,
    )
    assert float(objective) == pytest.approx(-1.2)


def test_value_loss_and_joint_ppo_parameter_update() -> None:
    actor, critic, actor_params, critic_params = make_networks()
    batch = PPOBatch(
        observations=jnp.ones((4, 3)),
        actions=jnp.asarray([0, 1, 0, 1]),
        old_log_probs=jnp.zeros(4),
        advantages=jnp.asarray([1.0, -1.0, 0.5, -0.5]),
        returns=jnp.ones(4),
    )
    assert float(value_loss(jnp.zeros(2), jnp.ones(2))) == pytest.approx(0.5)
    from rl_platform.networks.optimization import create_train_state

    actor_state = create_train_state(actor.model, actor_params, 0.01)
    critic_state = create_train_state(critic.model, critic_params, 0.01)
    updated_actor, updated_critic, loss, components = update_ppo_parameters(
        actor_state, critic_state, actor, critic, batch, make_config()
    )
    assert np.isfinite(float(loss))
    assert set(components) == {"policy_loss", "value_loss", "entropy"}
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(actor_state.params),
            jax.tree_util.tree_leaves(updated_actor.params),
        )
    )
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(critic_state.params),
            jax.tree_util.tree_leaves(updated_critic.params),
        )
    )


def test_ppo_agent_collects_rollout_and_runs_multiple_epochs() -> None:
    config = make_config()
    agent = PPOAgent.create(3, 2, config, create_prng_key(8))
    for index in range(4):
        observation = np.asarray([index, 0.0, 1.0], dtype=np.float32)
        action = agent.select_action(observation, key=create_prng_key(20 + index))
        agent.observe(
            EnvironmentTransition(
                observation=observation,
                action=action,
                reward=1.0,
                next_observation=observation + 0.1,
                terminated=False,
                truncated=False,
            )
        )
    before = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), agent.parameters)

    update = agent.update(key=create_prng_key(30))

    assert update is not None
    assert update.minibatches == 4
    assert np.isfinite(update.loss)
    assert len(agent.rollout) == 0
    assert any(
        not np.array_equal(before_leaf, after_leaf)
        for before_leaf, after_leaf in zip(
            jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(agent.parameters)
        )
    )
