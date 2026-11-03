"""Tests for explicit JAX PRNG handling."""

import jax
import numpy as np
import pytest

from rl_platform.utils import create_prng_key, split_prng_key


def test_prng_key_creation_is_reproducible() -> None:
    first = create_prng_key(42)
    second = create_prng_key(42)

    assert isinstance(first, jax.Array)
    np.testing.assert_array_equal(first, second)


def test_prng_key_splitting_returns_distinct_keys() -> None:
    keys = split_prng_key(create_prng_key(42), num=3)

    assert len(keys) == 3
    assert not np.array_equal(keys[0], keys[1])


@pytest.mark.parametrize("seed", [-1, True, "42"])
def test_invalid_prng_seed_is_rejected(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_prng_key(seed)  # type: ignore[arg-type]
