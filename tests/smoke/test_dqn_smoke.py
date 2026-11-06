"""Tiny end-to-end smoke test for DQN data interaction and learning."""

import jax
import numpy as np

from rl_platform.agents import DQNAgent
from rl_platform.algorithms import DQNConfig, Transition
from rl_platform.utils.random import create_prng_key


def test_synthetic_dqn_update_is_finite_and_changes_parameters() -> None:
    config = DQNConfig(
        learning_rate=0.01,
        gamma=0.95,
        batch_size=2,
        replay_capacity=4,
        min_replay_size=2,
        target_update_interval=2,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=5,
        hidden_layers=1,
        hidden_size=8,
        activation="relu",
        optimizer="adam",
    )
    agent = DQNAgent.create(2, 2, config, create_prng_key(10))
    observation = np.asarray([0.0, 1.0], dtype=np.float32)
    for index in range(2):
        agent.observe(
            Transition(
                observation=observation + index,
                action=index % 2,
                reward=1.0,
                next_observation=observation + index + 1.0,
                terminated=index == 1,
                truncated=False,
            )
        )

    action = agent.select_action(observation, key=create_prng_key(11))
    before = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), agent.parameters)
    update = agent.update(key=create_prng_key(12))

    assert action in (0, 1)
    assert update is not None
    assert np.isfinite(update.loss)
    assert any(
        not np.array_equal(before_leaf, after_leaf)
        for before_leaf, after_leaf in zip(
            jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(agent.parameters)
        )
    )
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(agent.parameters)
    )

