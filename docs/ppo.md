# PPO

Phase 7 adds a categorical PPO agent while reusing the platform's existing
MLP, Optax, training, evaluation, metrics, and configuration layers.

## Components

- `Actor` produces categorical policy logits.
- `Critic` produces scalar state values.
- `RolloutBuffer` collects on-policy transitions and old-policy statistics.
- `compute_returns` and `compute_gae` implement bootstrapped returns and GAE.
- `normalize_advantages` standardizes each rollout's advantages.
- `clipped_policy_objective`, `value_loss`, and `entropy_bonus` implement the
  PPO objective terms.
- `PPOAgent` shuffles rollout data into minibatches and performs multiple PPO
  epochs through the common `select_action`, `observe`, and `update` interface.

The GAE and return estimators mask `terminated=True` while continuing to
bootstrap across external `truncated=True` time limits. Training actions are
sampled from the categorical policy; the existing evaluation engine requests
deterministic actions by default.

PPO rollout collection remains inside `PPOAgent`; generic episode management
and metric recording remain in Phase 5's shared engines. Actor-Critic is not
implemented.

