"""Configurable MLP implemented with Flax Linen and JAX."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import linen as nn

Activation = Callable[[jax.Array], jax.Array]


def _identity(values: jax.Array) -> jax.Array:
    return values


_ACTIVATIONS: dict[str, Activation] = {
    "gelu": jax.nn.gelu,
    "identity": _identity,
    "linear": _identity,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "swish": jax.nn.swish,
    "tanh": jax.nn.tanh,
}


def get_activation(name: str) -> Activation:
    """Return a supported JAX activation function by name."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Activation name must be a non-empty string")
    normalized_name = name.strip().lower().replace("-", "_")
    try:
        return _ACTIVATIONS[normalized_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"Unsupported activation {name!r}; supported activations: {supported}") from exc


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Static architecture configuration for an MLP."""

    input_size: int
    hidden_layers: int
    hidden_size: int
    output_size: int
    activation: str = "relu"

    def __post_init__(self) -> None:
        _validate_positive_int(self.input_size, "input_size")
        _validate_non_negative_int(self.hidden_layers, "hidden_layers")
        _validate_positive_int(self.hidden_size, "hidden_size")
        _validate_positive_int(self.output_size, "output_size")
        get_activation(self.activation)


class MLP(nn.Module):
    """A configurable multilayer perceptron.

    The same module accepts a single input with shape ``(input_size,)`` or a
    batch with shape ``(batch_size, input_size)``. Parameters are created by
    :func:`initialize_mlp` and applied through :func:`mlp_forward`.
    """

    config: MLPConfig

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        values = jnp.asarray(inputs)
        if values.ndim not in (1, 2):
            raise ValueError("MLP inputs must have shape (input_size,) or (batch_size, input_size)")
        if values.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected input size {self.config.input_size}, received {values.shape[-1]}"
            )

        activation = get_activation(self.config.activation)
        for layer_index in range(self.config.hidden_layers):
            values = nn.Dense(
                features=self.config.hidden_size,
                name=f"hidden_{layer_index}",
            )(values)
            values = activation(values)
        return nn.Dense(features=self.config.output_size, name="output")(values)


def initialize_mlp(
    config: MLPConfig,
    key: Any,
    *,
    dtype: Any = jnp.float32,
) -> tuple[MLP, Any]:
    """Create an MLP and initialize its parameter PyTree with an explicit key."""

    model = MLP(config=config)
    sample_input = jnp.zeros((config.input_size,), dtype=dtype)
    variables = model.init(key, sample_input)
    return model, variables["params"]


def mlp_forward(model: MLP, params: Any, inputs: Any) -> jax.Array:
    """Run a single or batched forward pass using the supplied parameters."""

    return cast(jax.Array, model.apply({"params": params}, jnp.asarray(inputs)))


def batched_mlp_forward(model: MLP, params: Any, inputs: Any) -> jax.Array:
    """Run a forward pass over a rank-two batch using ``jax.vmap``."""

    batch = jnp.asarray(inputs)
    if batch.ndim != 2:
        raise ValueError("Batched MLP inputs must have shape (batch_size, input_size)")
    return cast(
        jax.Array,
        jax.vmap(lambda sample: model.apply({"params": params}, sample))(batch),
    )


def _validate_positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _validate_non_negative_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
