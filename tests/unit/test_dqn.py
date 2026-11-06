"""Unit tests for the Phase 4 DQN components."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_platform.agents import DQNAgent
from rl_platform.algorithms import (
    DQNConfig,
    EpsilonScheduler,
    ReplayBuffer,
    Transition,
    bellman_target,
    dqn_loss,
    epsilon_greedy_action,
    update_target_network,
)
from rl_platform.networks import QNetwork, QNetworkConfig
from rl_platform.utils.random import create_prng_key


def make_config(**overrides: object) -> DQNConfig:
    values: dict[str, object] = {
        "learning_rate": 0.01,
        "gamma": 0.9,
        "batch_size": 2,
        "replay_capacity": 8,
        "min_replay_size": 2,
        "target_update_interval": 2,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 10,
        "hidden_layers": 1,
        "hidden_size": 8,
        "activation": "relu",
        "optimizer": "adam",
    }
    values.update(overrides)
    return DQNConfig(**values)  # type: ignore[arg-type]


def make_transition(index: int, *, terminated: bool = False, truncated: bool = False) -> Transition:
    observation = np.asarray([index, index + 1], dtype=np.float32)
    return Transition(
        observation=observation,
        action=index % 2,
        reward=float(index),
        next_observation=observation + 1.0,
        terminated=terminated,
        truncated=truncated,
    )


def make_batch(buffer: ReplayBuffer, count: int = 2):
    for index in range(count):
        buffer.add_transition(make_transition(index))
    return buffer.sample(count, create_prng_key(42))


def test_dqn_config_mapping_requires_known_hyperparameters() -> None:
    config = make_config()
    values = {
        field: getattr(config, field)
        for field in config.__dataclass_fields__
    }
    assert DQNConfig.from_mapping(values) == config
    with pytest.raises(ValueError, match="Unknown DQN hyperparameter"):
        DQNConfig.from_mapping({**values, "unexpected": 1})


def test_replay_buffer_insertion_and_capacity() -> None:
    buffer = ReplayBuffer(capacity=2)
    buffer.add_transition(make_transition(0))
    buffer.add_transition(make_transition(1))
    buffer.add_transition(make_transition(2))

    assert len(buffer) == 2
    batch = buffer.sample(2, create_prng_key(0))
    np.testing.assert_array_equal(np.sort(batch.rewards), np.asarray([1.0, 2.0]))


def test_replay_buffer_sampling_returns_expected_arrays() -> None:
    buffer = ReplayBuffer(capacity=4)
    for index in range(4):
        buffer.add_transition(make_transition(index))

    batch = buffer.sample(3, create_prng_key(1))
    assert batch.size == 3
    assert batch.observations.shape == (3, 2)
    assert batch.actions.shape == (3,)
    assert batch.terminated.dtype == np.bool_


def test_transition_done_handles_terminated_and_truncated() -> None:
    assert not make_transition(0).done
    assert make_transition(0, terminated=True).done
    assert make_transition(0, truncated=True).done


def test_bellman_target_masks_terminated_but_not_truncated() -> None:
    targets = bellman_target(
        rewards=jnp.asarray([1.0, 2.0, 3.0]),
        next_q_values=jnp.asarray([[10.0, 8.0], [4.0, 9.0], [1.0, 2.0]]),
        terminated=jnp.asarray([True, False, False]),
        gamma=0.5,
    )
    np.testing.assert_allclose(targets, np.asarray([1.0, 6.5, 4.0]))


def test_epsilon_greedy_action_is_greedy_when_epsilon_zero() -> None:
    action = epsilon_greedy_action(jnp.asarray([1.0, 3.0, 2.0]), create_prng_key(0), 0.0)
    assert int(action) == 1


def test_epsilon_scheduler_decays_and_is_bounded() -> None:
    scheduler = EpsilonScheduler(start=1.0, end=0.1, decay_steps=10)
    assert scheduler.value(0) == pytest.approx(1.0)
    assert scheduler.value(5) == pytest.approx(0.55)
    assert scheduler.value(10) == pytest.approx(0.1)
    assert scheduler.value(100) == pytest.approx(0.1)


def test_q_network_output_shapes_for_single_and_batched_inputs() -> None:
    network = QNetwork.create(
        QNetworkConfig(
            observation_size=2,
            action_count=3,
            hidden_layers=1,
            hidden_size=8,
            activation="tanh",
        )
    )
    params = network.initialize(create_prng_key(3))
    assert network.apply(params, jnp.zeros((2,))).shape == (3,)
    assert network.apply(params, jnp.zeros((5, 2))).shape == (5, 3)


def test_dqn_loss_is_finite_and_differentiable() -> None:
    config = make_config()
    agent = DQNAgent.create(2, 2, config, create_prng_key(4))
    batch = make_batch(agent.replay_buffer)

    loss = dqn_loss(agent.parameters, agent.target_params, agent.q_network, batch, config.gamma)
    gradients = jax.grad(
        lambda parameters: dqn_loss(
            parameters, agent.target_params, agent.q_network, batch, config.gamma
        )
    )(agent.parameters)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(gradients))


def test_target_network_hard_and_soft_updates() -> None:
    online = {"w": jnp.asarray([2.0, 4.0])}
    target = {"w": jnp.asarray([0.0, 2.0])}
    np.testing.assert_allclose(update_target_network(online, target)["w"], np.asarray([2.0, 4.0]))
    np.testing.assert_allclose(
        update_target_network(online, target, tau=0.25)["w"],
        np.asarray([0.5, 2.5]),
    )


def test_agent_update_changes_parameters_and_updates_target() -> None:
    config = make_config(target_update_interval=1)
    agent = DQNAgent.create(2, 2, config, create_prng_key(5))
    for index in range(2):
        agent.observe(make_transition(index))
    before = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), agent.parameters)

    update = agent.update(key=create_prng_key(6))

    assert update is not None
    assert update.target_updated
    assert np.isfinite(update.loss)
    changed = any(
        not np.array_equal(before_leaf, after_leaf)
        for before_leaf, after_leaf in zip(
            jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(agent.parameters)
        )
    )
    assert changed
    for leaf in jax.tree_util.tree_leaves(agent.parameters):
        assert np.isfinite(np.asarray(leaf)).all()
    for online_leaf, target_leaf in zip(
        jax.tree_util.tree_leaves(agent.parameters), jax.tree_util.tree_leaves(agent.target_params)
    ):
        np.testing.assert_allclose(online_leaf, target_leaf)


def test_agent_update_waits_for_replay_warmup() -> None:
    agent = DQNAgent.create(2, 2, make_config(), create_prng_key(7))
    assert agent.update(key=create_prng_key(8)) is None
