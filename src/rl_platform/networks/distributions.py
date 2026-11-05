"""JAX categorical policy distribution helpers shared by policy algorithms."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def policy_log_prob(logits: Any, actions: Any) -> jax.Array:
    """Return categorical log-probabilities for selected actions."""

    log_probs = jax.nn.log_softmax(jnp.asarray(logits), axis=-1)
    action_values = jnp.asarray(actions, dtype=jnp.int32)
    return jnp.take_along_axis(log_probs, action_values[..., None], axis=-1).squeeze(axis=-1)


def policy_entropy(logits: Any) -> jax.Array:
    """Return categorical entropy per observation."""

    log_probs = jax.nn.log_softmax(jnp.asarray(logits), axis=-1)
    probabilities = jnp.exp(log_probs)
    return -jnp.sum(probabilities * log_probs, axis=-1)


def sample_policy_action(logits: Any, key: Any, *, deterministic: bool = False) -> jax.Array:
    """Sample a categorical action or select its mode."""

    values = jnp.asarray(logits)
    if values.ndim != 1 or values.shape[0] < 2:
        raise ValueError("Policy logits must have shape (action_count,) with at least two actions")
    if deterministic:
        return jnp.argmax(values).astype(jnp.int32)
    return jax.random.categorical(key, values).astype(jnp.int32)
