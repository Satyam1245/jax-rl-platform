"""Generic Optax optimization integration for JAX parameter trees."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import optax
from flax.training import train_state

from rl_platform.networks.mlp import MLP


def create_optimizer(
    learning_rate: float,
    *,
    name: str = "adam",
) -> optax.GradientTransformation:
    """Create a supported Optax optimizer without owning algorithm logic."""

    if learning_rate <= 0:
        raise ValueError("Learning rate must be greater than zero")
    normalized_name = name.strip().lower()
    if normalized_name == "adam":
        return optax.adam(learning_rate)
    if normalized_name == "sgd":
        return optax.sgd(learning_rate)
    raise ValueError("Unsupported optimizer; supported optimizers: adam, sgd")


def create_train_state(
    model: MLP,
    params: Any,
    learning_rate: float,
    *,
    optimizer_name: str = "adam",
) -> train_state.TrainState:
    """Create a Flax ``TrainState`` for a parameterized network."""

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=create_optimizer(learning_rate, name=optimizer_name),
    )


def optimizer_step(
    state: train_state.TrainState,
    loss_fn: Callable[[Any], Any],
) -> tuple[train_state.TrainState, Any]:
    """Compute gradients of ``loss_fn`` and apply one Optax update."""

    loss, gradients = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=gradients), loss
