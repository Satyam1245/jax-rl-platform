# DQN

Phase 4 provides a complete, Streamlit-independent DQN agent for discrete
action environments. It uses the Phase 3 Flax MLP and Optax integration.

## Components

- `ReplayBuffer` stores transitions in a bounded host-side ring buffer and
  samples with explicit JAX PRNG keys.
- `QNetwork` produces one Q-value per discrete action.
- `DQNAgent` maintains separate online and target parameter trees.
- `EpsilonScheduler` linearly decays exploration between configured bounds.
- `epsilon_greedy_action` selects random or greedy actions from explicit keys.
- `bellman_target` computes one-step targets.
- `dqn_loss` computes mean-squared temporal-difference loss.
- `update_dqn_parameters` applies a generic Optax update.
- `update_target_network` supports hard or Polyak target updates.

## Terminal semantics

Bellman bootstrapping stops on `terminated=True`. `truncated=True` is retained
in replay records but does not by itself suppress bootstrapping, matching the
Gymnasium distinction between an absorbing terminal state and an external time
limit.

## Minimal construction

```python
from rl_platform.agents import DQNAgent
from rl_platform.algorithms import DQNConfig
from rl_platform.utils import create_prng_key

config = DQNConfig.from_mapping({
    "learning_rate": 0.001,
    "gamma": 0.99,
    "batch_size": 32,
    "replay_capacity": 10000,
    "min_replay_size": 32,
    "target_update_interval": 100,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 10000,
    "hidden_layers": 2,
    "hidden_size": 64,
    "activation": "relu",
    "optimizer": "adam",
})
agent = DQNAgent.create(4, 2, config, create_prng_key(42))
```

The Phase 4 agent can select actions, accept transitions, and update its
parameters from replay data. Episode orchestration and long-running training
remain deferred to Phase 5.
