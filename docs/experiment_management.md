# Experiment management and checkpoints

Phase 6 adds persistent experiment artifacts and resumable DQN state.

## Experiment directories

`ExperimentManager.create_experiment` creates a unique UTC timestamp/UUID
experiment ID and corresponding directories:

- `experiments/<experiment-id>/configuration.yaml`
- `experiments/<experiment-id>/metadata.json`
- `checkpoints/<experiment-id>/`
- `results/<experiment-id>/metrics.json`

The configuration is normalized before persistence so dataclasses and `Path`
values can be represented in YAML/JSON.

## Checkpoints

`CheckpointManager` stores `metadata.json` and `state.msgpack` in each named
checkpoint directory. Metadata identifies the experiment, algorithm,
environment, seed, configuration, format version, timestamp, and training
progress. The serialized agent state includes the online Flax `TrainState`
(model parameters and optimizer state), target parameters, RNG key, training
steps, and replay buffer.

Loading requires an initialized compatible agent. Optional expected algorithm,
environment, seed, and configuration values are compared before state is
restored; mismatches raise `IncompatibleCheckpointError`.

```python
metadata = manager.load_checkpoint(
    checkpoint_path,
    agent,
    algorithm="dqn",
    environment="CartPole-v1",
    seed=0,
    configuration=configuration,
)
resumed = training_engine.train(
    environment,
    agent,
    start_episode=metadata.episodes_completed,
    initial_total_steps=metadata.total_steps,
)
```

Dashboard integration remains a future phase.
