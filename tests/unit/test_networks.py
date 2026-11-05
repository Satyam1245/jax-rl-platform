"""Tests for the reusable JAX/Flax ML foundation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_platform.networks import (
    MLPConfig,
    batched_mlp_forward,
    create_train_state,
    get_activation,
    initialize_mlp,
    mlp_forward,
    optimizer_step,
)
from rl_platform.utils import create_prng_key


@pytest.fixture
def network() -> tuple[object, object, MLPConfig]:
    config = MLPConfig(
        input_size=4,
        hidden_layers=2,
        hidden_size=8,
        output_size=3,
        activation="tanh",
    )
    model, params = initialize_mlp(config, create_prng_key(11))
    return model, params, config


def test_initialization_creates_expected_parameter_tree(network: tuple[object, object, MLPConfig]) -> None:
    model, params, _ = network

    assert model is not None
    assert set(params) == {"hidden_0", "hidden_1", "output"}
    assert params["hidden_0"]["kernel"].shape == (4, 8)
    assert params["hidden_1"]["kernel"].shape == (8, 8)
    assert params["output"]["kernel"].shape == (8, 3)


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu", "swish", "linear"])
def test_activation_handling_returns_jax_callable(activation: str) -> None:
    function = get_activation(activation)
    values = function(jnp.array([-1.0, 0.0, 1.0]))

    assert values.shape == (3,)
    assert isinstance(values, jax.Array)


def test_unknown_activation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("not_an_activation")


def test_forward_pass_has_configured_output_shape(network: tuple[object, object, MLPConfig]) -> None:
    model, params, _ = network

    output = mlp_forward(model, params, jnp.ones((4,)))

    assert output.shape == (3,)
    assert np.isfinite(np.asarray(output)).all()


def test_batched_inference_has_batch_preserved(network: tuple[object, object, MLPConfig]) -> None:
    model, params, _ = network
    inputs = jnp.ones((5, 4))

    output = batched_mlp_forward(model, params, inputs)

    assert output.shape == (5, 3)


def test_invalid_input_shapes_are_rejected(network: tuple[object, object, MLPConfig]) -> None:
    model, params, _ = network

    with pytest.raises(ValueError, match="input size"):
        mlp_forward(model, params, jnp.ones((3,)))
    with pytest.raises(ValueError, match="Batched MLP inputs"):
        batched_mlp_forward(model, params, jnp.ones((4,)))


def test_gradient_is_computable_and_non_zero(network: tuple[object, object, MLPConfig]) -> None:
    model, params, _ = network
    inputs = jnp.ones((4,))

    def loss_fn(current_params: object) -> jax.Array:
        prediction = mlp_forward(model, current_params, inputs)
        return jnp.mean(prediction**2)

    loss, gradients = jax.value_and_grad(loss_fn)(params)

    assert np.isfinite(np.asarray(loss))
    assert any(np.any(np.asarray(gradient) != 0.0) for gradient in jax.tree.leaves(gradients))


def test_jit_forward_execution_matches_eager(network: tuple[object, object, MLPConfig]) -> None:
    model, params, _ = network
    inputs = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)

    eager_output = batched_mlp_forward(model, params, inputs)
    jitted_forward = jax.jit(lambda values: batched_mlp_forward(model, params, values))
    jitted_output = jitted_forward(inputs)

    np.testing.assert_allclose(eager_output, jitted_output)


def test_optimizer_update_changes_parameters(network: tuple[object, object, MLPConfig]) -> None:
    model, params, _ = network
    state = create_train_state(model, params, learning_rate=0.01)
    inputs = jnp.ones((4,))
    target = jnp.ones((3,))

    def loss_fn(current_params: object) -> jax.Array:
        prediction = mlp_forward(model, current_params, inputs)
        return jnp.mean((prediction - target) ** 2)

    updated_state, loss = optimizer_step(state, loss_fn)

    assert np.isfinite(np.asarray(loss))
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree.leaves(state.params),
            jax.tree.leaves(updated_state.params),
            strict=True,
        )
    )


def test_jitted_optimizer_update_changes_parameters(network: tuple[object, object, MLPConfig]) -> None:
    model, params, _ = network
    state = create_train_state(model, params, learning_rate=0.01)
    inputs = jnp.ones((4,))
    target = jnp.ones((3,))

    def loss_fn(current_params: object) -> jax.Array:
        prediction = mlp_forward(model, current_params, inputs)
        return jnp.mean((prediction - target) ** 2)

    jitted_step = jax.jit(optimizer_step, static_argnums=1)
    updated_state, _ = jitted_step(state, loss_fn)

    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree.leaves(state.params),
            jax.tree.leaves(updated_state.params),
            strict=True,
        )
    )
