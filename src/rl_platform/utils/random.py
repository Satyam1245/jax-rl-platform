"""Explicit JAX pseudo-random-number generator helpers."""

from __future__ import annotations

from typing import Any

import jax


def create_prng_key(seed: int) -> Any:
    """Create a JAX PRNG key from an explicit non-negative seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("PRNG seed must be an integer")
    if seed < 0:
        raise ValueError("PRNG seed must be non-negative")
    return jax.random.PRNGKey(seed)


def split_prng_key(key: Any, num: int = 2) -> tuple[Any, ...]:
    """Split a JAX PRNG key into ``num`` independent keys."""

    if isinstance(num, bool) or not isinstance(num, int):
        raise TypeError("Number of PRNG keys must be an integer")
    if num < 1:
        raise ValueError("Number of PRNG keys must be >= 1")
    return tuple(jax.random.split(key, num))
