"""Unit tests for Actor-Critic networks, estimators, objectives, and updates."""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_platform.agents import ActorCriticAgent
from rl_platform.algorithms import (
    ActorCriticBatch,
    ActorCriticConfig,
    actor_critic_entropy,
    actor_critic_policy_loss,
    actor_critic_returns,
    actor_critic_value_loss,
    compute_advantages,
    update_actor_critic_parameters,
)
from rl_platform.networks import Actor, Critic, PolicyNetworkConfig
from rl_platform.training import EnvironmentTransition
from rl_platform.utils.random import create_prng_key


def make_config(**overrides: object) -> ActorCriticConfig:
    values: dict[str, object] = {
        "learning_rate": 0.01,
        "gamma": 0.9,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "hidden_layers": 1,
        "hidden_size": 8,
        "activation": "tanh",
        "optimizer": "adam",
    }
    values.update(overrides)
    return ActorCriticConfig(**values)  # type: ignore[arg-type]


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
    return actor, critic, actor.initialize(create_prng_key(1)), critic.initialize(create_prng_key(2))


def test_actor_critic_configuration_mapping() -> None:
    config = make_config()
    assert ActorCriticConfig.from_mapping(asdict(config)) == config
    with pytest.raises(ValueError, match="Unknown Actor-Critic hyperparameter"):
        ActorCriticConfig.from_mapping({**asdict(config), "extra": 1})


def test_actor_and_critic_outputs() -> None:
    actor, critic, actor_params, critic_params = make_networks()
    observations = jnp.ones((4, 3))
    assert actor.logits(actor_params, observations).shape == (4, 2)
    assert critic.values(critic_params, observations).shape == (4,)
    assert actor.logits(actor_params, observations[0]).shape == (2,)
    assert critic.values(critic_params, observations[0]).shape == ()


def test_returns_and_advantages() -> None:
    returns = actor_critic_returns(
        rewards=jnp.asarray([1.0, 2.0]),
        next_values=jnp.asarray([3.0, 4.0]),
        terminated=jnp.asarray([False, True]),
        gamma=0.5,
    )
    np.testing.assert_allclose(returns, [2.5, 2.0])
    np.testing.assert_allclose(compute_advantages(returns, [1.5, 1.0]), [1.0, 1.0])


def test_policy_loss_value_loss_and_entropy() -> None:
    logits = jnp.asarray([[0.0, 0.0]])
    assert float(actor_critic_policy_loss(logits, [0], [1.0])) == pytest.approx(np.log(2.0))
    assert float(actor_critic_value_loss([0.0], [2.0])) == pytest.approx(2.0)
    assert float(actor_critic_entropy(logits)) == pytest.approx(np.log(2.0))


def test_joint_actor_critic_gradient_update_changes_parameters() -> None:
    actor, critic, actor_params, critic_params = make_networks()
    batch = ActorCriticBatch(
        observations=jnp.ones((3, 3)),
        actions=jnp.asarray([0, 1, 0]),
        returns=jnp.asarray([1.0, 0.0, 2.0]),
        advantages=jnp.asarray([0.5, -0.5, 1.0]),
    )
    from rl_platform.networks.optimization import create_train_state

    actor_state = create_train_state(actor.model, actor_params, 0.01)
    critic_state = create_train_state(critic.model, critic_params, 0.01)
    updated_actor, updated_critic, loss, components = update_actor_critic_parameters(
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


def test_agent_observes_and_updates_one_transition() -> None:
    agent = ActorCriticAgent.create(3, 2, make_config(), create_prng_key(3))
    observation = np.asarray([0.0, 1.0, 0.5], dtype=np.float32)
    action = agent.select_action(observation, key=create_prng_key(4))
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
    update = agent.update(key=create_prng_key(5))
    assert update is not None
    assert update.step == 1
    assert np.isfinite(update.loss)
