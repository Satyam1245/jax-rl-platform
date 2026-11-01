# JAX Reinforcement Learning Platform

This project is a modular foundation for training, evaluating, comparing, and
visualizing reinforcement-learning agents with JAX. The initial environment
scope is CartPole-v1 and MountainCar-v0, with CartPole-v1 as the primary
demonstration.

## Phase 11 status

Phase 1 established the package layout, dependency declarations, YAML
configuration validation, logging, artifact directories, and baseline tests.
Phase 2 added the allow-listed Gymnasium environment factory and lifecycle
adapter for CartPole-v1 and MountainCar-v0. Phase 3 added the reusable
JAX/Flax MLP, explicit PRNG helpers, and generic Optax integration. Phase 4
adds the independent DQN components and agent. It intentionally does not
contain PPO, Actor-Critic, checkpointing, experiment tracking, visualization,
or Streamlit code. Phase 5 adds generic training and independent deterministic
evaluation engines that operate through agent interfaces. Phase 6 adds
experiment IDs, artifact persistence, DQN checkpoints, and resume support.
Phase 7 adds the categorical PPO agent with rollout collection, GAE, clipped
updates, and multiple minibatch epochs. Phase 8 adds the standard one-step
Actor-Critic agent with bootstrapped returns, advantages, entropy regularizing,
and checkpoint-compatible actor/critic state.
Phase 9 adds stored-result validation, DQN/PPO/Actor-Critic comparison
metrics, and Plotly visualizations. Phase 10 adds the Streamlit dashboard as a
presentation layer over the tested
backend services. It does not duplicate any RL algorithm or training logic.
The final QA pass covers all three algorithms, checkpoint/evaluation
round-trips, stored-result visualization, invalid-input handling, imports, and
the dashboard startup path.

## Setup

Use an isolated Python 3.11+ environment. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The dependency versions are minimum compatibility bounds, with protobuf pinned
below version 7 for Streamlit compatibility. Verify the resolved
JAX/Flax/Optax/Gymnasium combination in the target environment before a run.

## Configuration

The sample configuration is at `configs/default.yaml`. Configuration is
loaded and validated before any future environment or training code runs.
Required run values, including the seed, episode counts, and artifact paths,
are supplied by YAML rather than hard-coded in Python.

## Algorithms

The supported algorithms are:

- DQN — replay buffer, target network, and epsilon-greedy exploration.
- PPO — categorical actor-critic policy with clipped updates, GAE, and
  minibatch epochs.
- Actor-Critic — one-step bootstrapped actor-critic updates with entropy
  regularization.

All algorithms are independent of Streamlit and use the common environment,
training, evaluation, metrics, and checkpoint interfaces.

## Training and evaluation

The backend service `rl_platform.services.train_experiment` creates a
validated experiment, trains the selected agent through the generic training
engine, records metrics, and saves a checkpoint. The independent service
`rl_platform.services.evaluate_checkpoint` restores a checkpoint and runs
deterministic evaluation without exploration or training updates. Both
services accept explicit environment, algorithm, hyperparameters, seed, and
run-size values and can be called from Python without starting Streamlit.

## Experiments and checkpoints

Each run creates an experiment directory under `experiments/`, a checkpoint
directory under `checkpoints/`, and a persisted metrics file under `results/`.
Metadata records the experiment ID, algorithm, environment, seed, resolved
configuration, timestamp, training progress, and measured duration. Checkpoint
loading validates the algorithm, environment, seed, configuration, and state
schema before restoring model and optimizer state. See
[`docs/experiment_management.md`](docs/experiment_management.md) for resume
examples.

## Dashboard

Start the presentation layer from the project root with:

```powershell
python -m streamlit run app/streamlit_app.py
```

The dashboard provides overview, training, progress, checkpoint evaluation,
experiment history, comparison, and checkpoint sections. It reads backend
services and stored results; algorithm mathematics and training loops remain
outside the UI.

## Environment layer

Create and use an environment independently of any RL algorithm:

```python
from rl_platform.environments import make_environment

with make_environment("CartPole-v1", seed=42) as environment:
    reset_result = environment.reset()
    step_result = environment.step(0)
    print(environment.spec, step_result.terminated, step_result.truncated)
```

The adapter preserves Gymnasium's separate `terminated` and `truncated`
signals and closes the underlying environment safely.

## Tests

Run the complete test suite with:

```powershell
python -m pytest
```

The test layout is divided into `tests/unit`, `tests/integration`, and
`tests/smoke`. Unit tests cover individual components, integration tests cover
artifact and backend workflows, and smoke tests use tiny real environments.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system boundaries and
  architectural decisions.
- [`docs/development_plan.md`](docs/development_plan.md) — audit findings,
  phase dependencies, risks, and implementation order.
- [`docs/environment_layer.md`](docs/environment_layer.md) — environment-layer
  API and lifecycle semantics.
- [`docs/jax_foundation.md`](docs/jax_foundation.md) — JAX/Flax/Optax foundation
  API and PRNG conventions.
- [`docs/dqn.md`](docs/dqn.md) — DQN components, update semantics, and usage.
- [`docs/training_evaluation.md`](docs/training_evaluation.md) — generic
  training/evaluation engines and recorded metrics.
- [`docs/experiment_management.md`](docs/experiment_management.md) — experiment
  artifacts, checkpoint validation, and resume support.
- [`docs/ppo.md`](docs/ppo.md) — PPO architecture, estimators, and objectives.
- [`docs/actor_critic.md`](docs/actor_critic.md) — Actor-Critic objectives,
  updates, and checkpoint compatibility.
- [`docs/visualization.md`](docs/visualization.md) — stored results,
  comparison metrics, and Plotly visualizations.
- [`docs/streamlit.md`](docs/streamlit.md) — dashboard sections and backend
  service integration.
