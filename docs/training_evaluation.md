# Training and evaluation engine

Phase 5 adds algorithm-independent episode orchestration.

## Training

`TrainingEngine` accepts an `EnvironmentAdapter`, a `TrainingConfig`, and any
agent implementing `select_action`, `observe`, and `update`. The engine owns
episode resets, explicit JAX key splitting, environment interaction, transition
forwarding, step counting, and metric recording. It does not import or encode
DQN-specific behavior.

```python
from rl_platform.training import TrainingConfig, TrainingEngine

config = TrainingConfig(
    episodes=5,
    max_steps_per_episode=500,
    seed=0,
    moving_average_window=100,
)
result = TrainingEngine(config).train(environment, agent)
```

Each environment transition preserves `terminated` and `truncated` in the
algorithm-neutral `EnvironmentTransition`. The configured step limit controls
episode orchestration without changing the environment's termination flags.

## Evaluation

`EvaluationEngine` is a separate component. It resets the environment with
explicit per-episode seeds, calls `select_action(..., deterministic=True)` by
default, and never calls `observe` or `update`. This keeps evaluation
independent from training and disables DQN exploration by default.

## Metrics

`MetricsRecorder` records training episode reward, episode length, cumulative
mean reward, moving-average reward, and every finite optimizer loss. Evaluation
episodes are recorded separately as evaluation rewards and lengths. Results
expose immutable tuples through `episode_reward`, `episode_length`,
`mean_reward`, `moving_average_reward`, `loss`, and `evaluation_reward`.

Long-running training, checkpointing, experiment tracking, and visualization
remain separate future phases.

