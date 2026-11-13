# JAX ML Foundation

Phase 3 provides reusable JAX/Flax/Optax building blocks without implementing
an RL algorithm.

## MLP

`rl_platform.networks.MLPConfig` defines `input_size`, `hidden_layers`,
`hidden_size`, `output_size`, and `activation`. Supported activations include
ReLU, tanh, sigmoid, GELU, swish, softplus, and identity/linear.

```python
from rl_platform.networks import MLPConfig, initialize_mlp, mlp_forward
from rl_platform.utils import create_prng_key

config = MLPConfig(
    input_size=4,
    hidden_layers=2,
    hidden_size=64,
    output_size=2,
    activation="relu",
)
model, params = initialize_mlp(config, create_prng_key(42))
values = mlp_forward(model, params, observation)
```

The same forward function accepts one observation or a rank-two batch. The
explicit `batched_mlp_forward` helper uses `jax.vmap` for batched inference.

## PRNG handling

Randomness is explicit. Callers create a key from a seed and split it when
independent operations need separate keys. No global random state is used.

## Optax integration

`create_train_state` creates a Flax `TrainState` with an Optax Adam or SGD
optimizer. `optimizer_step` accepts a generic parameter-only loss function,
computes gradients with `jax.value_and_grad`, and applies the update. The
foundation does not define DQN, PPO, Actor-Critic, or training loops.
