# JAX Reinforcement Learning Platform

A modular reinforcement learning platform built with **JAX, Flax, Optax, and Gymnasium**, featuring implementations of **Deep Q-Network (DQN), Proximal Policy Optimization (PPO), and Actor-Critic** algorithms.

The platform provides shared training and evaluation engines, experiment management, checkpoint handling, performance visualization, and an interactive Streamlit dashboard.

**The application is publicly deployed and available to explore.**

<p align="center">
  <a href="https://jax-rl-platform-9ctlgkcc9yc47p34njfqzf.streamlit.app/">
    <strong>🚀 Launch Live Application</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/Satyam1245/jax-rl-platform">
    <strong>View Source Code</strong>
  </a>
</p>

The primary demonstration environment is `CartPole-v1`. The environment layer also supports `MountainCar-v0`.

---

## Table of Contents

- [Live Demo](#live-demo)
- [Features](#features)
- [Supported Algorithms](#supported-algorithms)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Getting Started](#getting-started)
- [Running the Dashboard](#running-the-dashboard)
- [Configuration](#configuration)
- [Training and Evaluation](#training-and-evaluation)
- [Experiment and Checkpoint Management](#experiment-and-checkpoint-management)
- [Environment Adapter](#environment-adapter)
- [Running Tests](#running-tests)
- [Documentation](#documentation)
- [Deployment](#deployment)
- [Current Scope and Limitations](#current-scope-and-limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Live Demo

**Try the deployed application:**

[https://jax-rl-platform-9ctlgkcc9yc47p34njfqzf.streamlit.app/](https://jax-rl-platform-9ctlgkcc9yc47p34njfqzf.streamlit.app/)

The interactive dashboard provides access to the platform's reinforcement learning workflows, including:

- Algorithm selection and training configuration.
- Training execution and progress inspection.
- Experiment history and metric visualization.
- Checkpoint management and evaluation.
- Training result comparison.

The application is deployed using **Streamlit Community Cloud**.

---

## Features

### Reinforcement Learning Algorithms

- **DQN (Deep Q-Network)**
  - Experience replay.
  - Target network.
  - Epsilon-greedy exploration.
  - Q-value learning.

- **PPO (Proximal Policy Optimization)**
  - Categorical policy.
  - Clipped policy updates.
  - Generalized Advantage Estimation (GAE).
  - Minibatch optimization epochs.

- **Actor-Critic**
  - One-step bootstrapped actor-critic updates.
  - Separate actor and critic networks.
  - Entropy regularization.

### Shared Infrastructure

- Common training and evaluation engines.
- Gymnasium environment adapter.
- Validated YAML configuration system.
- Experiment metadata and metric persistence.
- Model checkpoint saving and restoration.
- Deterministic checkpoint evaluation.
- Plotly-based training visualizations.
- Algorithm performance comparison.
- Streamlit dashboard.
- Unit, integration, and smoke-test suites.

---

## Supported Algorithms

| Algorithm | Key Components | Learning Approach |
|---|---|---|
| DQN | Replay buffer, target network, epsilon-greedy policy | Value-based |
| PPO | Clipped objective, categorical policy, GAE | Policy gradient |
| Actor-Critic | Actor network, critic network, entropy regularization | Actor-Critic |

The platform provides a shared infrastructure for training and evaluating these algorithms, making it easier to experiment with different reinforcement learning approaches.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Core programming language |
| JAX | Numerical computation and automatic differentiation |
| Flax | Neural network modules |
| Optax | Gradient-based optimization |
| Gymnasium | Reinforcement learning environments |
| NumPy | Numerical operations |
| pandas | Experiment and metric handling |
| Plotly | Interactive visualizations |
| Streamlit | Web dashboard |
| PyYAML | Configuration management |
| pytest | Automated testing |

---

## Project Architecture

The project follows a modular architecture that separates reinforcement learning algorithms, environment interaction, training workflows, evaluation, experiment management, and visualization.

```text

jax-rl-platform/
│
├── .devcontainer/
│   └── devcontainer.json       # Development container configuration
│
├── .github/
│   └── workflows/
│       └── tests.yml           # GitHub Actions test workflow
│
├── app/
│   └── streamlit_app.py        # Streamlit dashboard
│
├── configs/
│   ├── default.yaml            # DQN defaults
│   ├── ppo.yaml                # PPO configuration
│   └── actor_critic.yaml       # Actor-Critic configuration
│
├── docs/                       # Architecture and implementation docs
│
├── src/
│   └── rl_platform/
│       ├── agents/
│       ├── algorithms/
│       ├── checkpoints/
│       ├── configuration/
│       ├── environments/
│       ├── evaluation/
│       ├── experiments/
│       ├── metrics/
│       ├── networks/
│       ├── services/
│       ├── training/
│       ├── utils/
│       └── visualization/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
│
├── .gitignore
├── conftest.py
├── pyproject.toml
├── requirements.txt
└── README.md


```

### Architectural Components

| Component | Responsibility |
|---|---|
| `agents/` | Agent interfaces and implementations |
| `algorithms/` | DQN, PPO, Actor-Critic, replay and rollout logic |
| `checkpoints/` | Checkpoint persistence and validation |
| `configuration/` | Configuration loading and validation |
| `environments/` | Gymnasium environment integration |
| `evaluation/` | Evaluation configuration and execution |
| `experiments/` | Experiment metadata and artifact management |
| `metrics/` | Training metric recording |
| `networks/` | JAX/Flax neural networks and optimization helpers |
| `services/` | Backend orchestration for application workflows |
| `training/` | Shared training engine |
| `utils/` | Logging and PRNG utilities |
| `visualization/` | Training plots, result loading, and comparisons |

---

## Getting Started

These instructions use **Windows PowerShell**. Run all commands from the project root directory containing `pyproject.toml`.

### 1. Clone the repository

```powershell
git clone https://github.com/Satyam1245/jax-rl-platform.git

cd jax-rl-platform
```

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

If `py -3.11` is unavailable, install Python 3.11 or newer and use that interpreter.

### 3. Install dependencies

```powershell
python -m pip install -e ".[dev]"
```

The project declares CPU JAX dependencies by default. Available JAX devices and performance depend on your Python version, operating system, and installed runtime.

---

## Running the Dashboard

Start the Streamlit application:

```powershell
python -m streamlit run app/streamlit_app.py
```

Streamlit will provide a local URL in the terminal. Open that address in your browser.

The dashboard provides an interface for interacting with the platform's training, evaluation, experiment, checkpoint, and visualization workflows.

You can also access the deployed version directly:

**[Open JAX RL Platform](https://jax-rl-platform-9ctlgkcc9yc47p34njfqzf.streamlit.app/)**

---

## Configuration

Configuration files are maintained in the `configs/` directory.

| File | Description |
|---|---|
| `default.yaml` | DQN and general run defaults |
| `ppo.yaml` | PPO-specific configuration |
| `actor_critic.yaml` | Actor-Critic configuration |

The configuration system supports settings such as:

- Environment selection.
- Random seed.
- Number of training episodes.
- Maximum steps per episode.
- Algorithm-specific hyperparameters.
- Moving-average window.
- Artifact directories.

Configuration values are loaded and validated by the backend before use.

### Supported Environments

- `CartPole-v1`
- `MountainCar-v0`

---

## Training and Evaluation

The backend service layer coordinates the major reinforcement learning workflows.

The shared infrastructure supports:

1. Agent initialization.
2. Environment interaction.
3. Training execution.
4. Metric collection.
5. Experiment recording.
6. Checkpoint saving.
7. Checkpoint restoration.
8. Agent evaluation.
9. Result visualization and comparison.

The shared training and evaluation engines help maintain consistent workflows across the implemented algorithms.

---

## Experiment and Checkpoint Management

Training runs can generate artifacts in the following directories:

```text
experiments/
    Experiment metadata and run information.

checkpoints/
    Saved agent and optimizer state,
    together with checkpoint metadata.

results/
    Persisted training metrics and results.
```

These directories are intended for generated runtime artifacts rather than source code.

The repository's `.gitignore` excludes generated artifact contents, preventing local training outputs from unnecessarily increasing the repository size.

**Deployment consideration:** Streamlit Community Cloud does not guarantee persistence of files written to the application's local filesystem across restarts or redeployments. Therefore, checkpoints and experiment results created in the hosted application should be treated as temporary unless an external persistent storage solution is configured.

For long-term storage, consider integrating cloud object storage or a database.

---

## Environment Adapter

The environment adapter can be used independently of the agents and dashboard.

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

The adapter provides:

- Context-manager-based cleanup.
- Environment initialization and reset.
- Action execution.
- Structured step results.
- Separate `terminated` and `truncated` signals.

---

## Running Tests

From the project root, with the virtual environment activated:

```powershell
python -m pytest
```

The test suite is organized into three categories.

| Directory | Purpose |
|---|---|
| `tests/unit/` | Focused tests for individual components |
| `tests/integration/` | Backend workflows and artifact handling |
| `tests/smoke/` | Small end-to-end application and algorithm checks |

The presence of tests in the repository does not guarantee that they pass in every environment. Run the complete test suite after installation to verify your local setup.

---

## Documentation

Additional architecture and implementation documentation is available in the [`docs/`](docs/) directory.

- [Architecture](docs/architecture.md)
- [Development Plan](docs/development_plan.md)
- [Environment Layer](docs/environment_layer.md)
- [JAX Foundation](docs/jax_foundation.md)
- [DQN](docs/dqn.md)
- [PPO](docs/ppo.md)
- [Actor-Critic](docs/actor_critic.md)
- [Training and Evaluation](docs/training_evaluation.md)
- [Experiment Management and Checkpoints](docs/experiment_management.md)
- [Visualization](docs/visualization.md)
- [Streamlit Dashboard](docs/streamlit.md)

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

**Live application:**  
https://jax-rl-platform-9ctlgkcc9yc47p34njfqzf.streamlit.app/

The deployment is connected to the GitHub repository.

For your own deployment:

1. Fork or use the repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select the repository and deployment branch.
5. Set the application entry point to:

   ```text
   app/streamlit_app.py
   ```

6. Configure the Python and dependency environment as required.
7. Deploy the application.

Keep credentials and sensitive configuration values out of source files. Use Streamlit's secrets management for sensitive settings.

---

## Current Scope and Limitations

The platform currently focuses on:

- Three implemented reinforcement learning algorithms: DQN, PPO, and Actor-Critic.
- Two supported Gymnasium environments: CartPole-v1 and MountainCar-v0.
- Shared training and evaluation infrastructure.
- Experiment and checkpoint management.
- Interactive Streamlit visualizations.

The platform is intended as a modular learning and experimentation environment, not as a claim that one algorithm will outperform another in every scenario.

Training outcomes depend on:

- Algorithm and hyperparameter configuration.
- Random seed.
- Environment characteristics.
- Number of training episodes.
- Runtime and available computational resources.

Performance comparisons should be conducted using consistent evaluation conditions and, where possible, multiple random seeds.

Hosted runtime storage should not be treated as permanent storage for generated experiments and checkpoints without an external persistence solution.

---

## Future Improvements

Potential directions for extending the platform include:

- Additional Gymnasium environments.
- More reinforcement learning algorithms.
- Multi-seed experiment evaluation.
- Hyperparameter optimization.
- Persistent cloud checkpoint storage.
- Advanced experiment analytics.
- Automated continuous integration and testing.
- Expanded performance benchmarking.

---

## License

No license file is currently included in this repository.

Until a license is added, the default copyright restrictions apply. Add an appropriate `LICENSE` file before granting others explicit permission to reuse, modify, or distribute the project.

---

## Author

**Satyam**

GitHub: [@Satyam1245](https://github.com/Satyam1245)

**Project:** JAX Reinforcement Learning Platform

**Live Demo:** [Launch Application](https://jax-rl-platform-9ctlgkcc9yc47p34njfqzf.streamlit.app/)

---

<p align="center">
  <strong>Built with JAX, Flax, Gymnasium, and Streamlit.</strong>
  <br/>
  Exploring reinforcement learning through modular implementations and interactive experimentation.
</p>
