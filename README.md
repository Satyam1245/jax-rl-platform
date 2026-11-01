# JAX Reinforcement Learning Platform

A modular reinforcement-learning platform built with **JAX, Flax, Optax, and Gymnasium**. It provides implementations of DQN, Proximal Policy Optimization (PPO), and one-step Actor-Critic, together with shared training and evaluation engines, experiment persistence, checkpoint management, result comparison, and a Streamlit dashboard.

The primary demonstration environment is `CartPole-v1`. The environment layer also supports `MountainCar-v0`.

## Features

- **Three RL algorithms**
  - **DQN** — experience replay, target network, and epsilon-greedy exploration.
  - **PPO** — categorical policy, clipped policy updates, generalized advantage estimation (GAE), and minibatch epochs.
  - **Actor-Critic** — one-step bootstrapped actor-critic updates with entropy regularization.
- **Shared training and evaluation** engines so the algorithms use common workflows.
- **Gymnasium environment adapter** with validated environment names and separate `terminated` and `truncated` signals.
- **YAML configuration** for environment selection, algorithm hyperparameters, run settings, and artifact paths.
- **Experiment records and metrics** persisted for later inspection and comparison.
- **Checkpoint save, restore, and deterministic evaluation** workflows.
- **Plotly visualizations** for training results and algorithm comparisons.
- **Streamlit dashboard** for training, progress review, checkpoint evaluation, experiment history, and comparisons.
- **Automated tests** organized into unit, integration, and smoke-test suites.

## Tech stack

- Python 3.11+
- JAX
- Flax
- Optax
- Gymnasium
- NumPy and pandas
- Plotly
- Streamlit
- PyYAML
- pytest (development and testing)

## Project structure

```text
jax-rl-platform/
├── app/
│   └── streamlit_app.py       # Streamlit dashboard
├── configs/
│   ├── default.yaml           # DQN defaults
│   ├── ppo.yaml               # PPO defaults
│   └── actor_critic.yaml      # Actor-Critic defaults
├── docs/                      # Architecture and implementation notes
├── src/
│   └── rl_platform/
│       ├── agents/            # Agent interfaces and implementations
│       ├── algorithms/        # DQN, PPO, Actor-Critic, replay and rollout logic
│       ├── checkpoints/       # Checkpoint persistence and validation
│       ├── configuration/     # Configuration loading and validation
│       ├── environments/      # Supported Gymnasium environments
│       ├── evaluation/        # Evaluation configuration and engine
│       ├── experiments/       # Experiment metadata and artifact management
│       ├── metrics/           # Training metric recording
│       ├── networks/          # JAX/Flax networks and optimization helpers
│       ├── services/          # Backend orchestration used by the dashboard
│       ├── training/          # Generic training engine
│       ├── utils/             # Logging and PRNG utilities
│       └── visualization/     # Result loading, plots and comparisons
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Getting started

These instructions use **Windows PowerShell**. Run the commands from the project root (the directory containing `pyproject.toml`).

### 1. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `py -3.11` is unavailable, install Python 3.11 or newer and use that interpreter to create the environment.

### 2. Install the project

```powershell
python -m pip install -e ".[dev]"
```

The project declares CPU JAX dependencies by default. Available JAX devices and performance depend on your Python, operating system, and installed runtime.

### 3. Start the dashboard

```powershell
python -m streamlit run app/streamlit_app.py
```

Streamlit will print a local URL in the terminal; open that URL in your browser. Use the dashboard to configure training runs, inspect progress and stored experiments, evaluate checkpoints, and compare results.

## Configuration

Configuration files are stored in `configs/`:

- `configs/default.yaml` — DQN configuration and general run defaults.
- `configs/ppo.yaml` — PPO configuration.
- `configs/actor_critic.yaml` — Actor-Critic configuration.

The YAML files define settings such as the environment, random seed, episode count, maximum steps per episode, algorithm hyperparameters, moving-average window, and artifact directories. Configuration values are loaded and validated by the backend before they are used.

Supported environments:

- `CartPole-v1`
- `MountainCar-v0`

## Training, evaluation, and artifacts

The backend service layer coordinates agent creation, training, metric persistence, checkpoint saving, checkpoint restoration, and evaluation. It can be used independently of the Streamlit interface.

Training runs write artifacts to the project directories:

- `experiments/` — experiment metadata and run information.
- `checkpoints/` — saved agent and optimizer state plus checkpoint metadata.
- `results/` — persisted training metrics used by the result-loading and visualization components.

These directories are intended for generated run artifacts. The repository's `.gitignore` excludes their contents except for optional `.gitkeep` files, so local training output does not need to be committed.

## Use the environment adapter

The environment adapter can be used independently of the agents and dashboard:

```python
from rl_platform.environments import make_environment

with make_environment("CartPole-v1", seed=42) as environment:
    reset_result = environment.reset()
    print("Observation:", reset_result.observation)

    step_result = environment.step(0)
    print("Reward:", step_result.reward)
    print("Terminated:", step_result.terminated)
    print("Truncated:", step_result.truncated)
```

The adapter provides context-manager cleanup and preserves Gymnasium's `terminated` and `truncated` signals separately.

## Run the tests

From the project root, with the virtual environment activated:

```powershell
python -m pytest
```

The test suite is divided into:

- `tests/unit/` — focused tests for individual components.
- `tests/integration/` — backend workflows and artifact handling.
- `tests/smoke/` — small end-to-end checks, including environment, algorithm, training, and dashboard-related paths.

The presence of tests in the repository does not itself guarantee that they pass in every environment. Run the suite locally after installation to verify your setup.

## Documentation

Implementation and architecture notes are available in [`docs/`](docs/):

- [Architecture](docs/architecture.md)
- [Development plan](docs/development_plan.md)
- [Environment layer](docs/environment_layer.md)
- [JAX foundation](docs/jax_foundation.md)
- [DQN](docs/dqn.md)
- [Training and evaluation](docs/training_evaluation.md)
- [Experiment management and checkpoints](docs/experiment_management.md)
- [PPO](docs/ppo.md)
- [Actor-Critic](docs/actor_critic.md)
- [Visualization](docs/visualization.md)
- [Streamlit dashboard](docs/streamlit.md)

## Current scope and limitations

This project currently focuses on the supported Gymnasium environments listed above and the three implemented algorithms. It is intended as a modular learning and experimentation platform, not as a claim that any algorithm will outperform another on every environment. Training outcomes depend on configuration, random seed, runtime, and environment.

## License

No license file was present in the supplied project archive. Add a `LICENSE` file before describing the project as open source or specifying reuse permissions.
