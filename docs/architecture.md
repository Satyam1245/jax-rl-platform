# JAX Reinforcement Learning Platform Architecture

## Phase 0 audit result

The workspace does not currently contain a JAX reinforcement-learning project.
The top-level directory is a collection of unrelated projects, including two
cricket analytics applications, computer-vision projects, and data-analysis
artifacts. There is no root Python package, root dependency manifest, root
configuration, or root test suite for the requested platform.

The platform is therefore isolated under `jax-rl-platform/`. Existing projects
are not dependencies of this system and will not be modified by it.

## Goals and initial scope

The platform will provide a reproducible, modular workflow for training,
evaluating, recording, comparing, and visualizing reinforcement-learning
agents. CartPole-v1 is the primary demonstration environment. MountainCar-v0
is the only additional environment in the initial scope.

Algorithms will be delivered sequentially:

1. DQN
2. PPO
3. Actor-Critic

Each algorithm must operate independently of Streamlit and must use JAX for
model computation and gradient-based learning.

## Layered architecture

```text
Configuration
     |
Environment factory --> Environment adapters
     |
Agent interface <--> Algorithm implementation <--> JAX networks/Optax
     |
Training engine --> Evaluation engine
     |
Checkpointing --> Experiment tracking --> Recorded results
     |
Visualization
     |
Streamlit dashboard
```

### Responsibilities and boundaries

| Layer | Responsibility | Must not do |
| --- | --- | --- |
| `configuration` | Load YAML, apply no hidden defaults, validate values and paths | Run training or import Streamlit |
| `environments` | Create the allow-listed Gymnasium environments and expose normalized reset/step behavior | Know algorithm hyperparameters |
| `networks` | Define JAX neural networks and pure forward functions | Own episode loops or filesystem state |
| `agents` | Select actions and hold agent-facing state/RNG interfaces | Contain the training loop |
| `algorithms` | Implement DQN, then PPO, then Actor-Critic update mathematics | Depend on Streamlit or visualization |
| `training` | Coordinate episodes/rollouts, updates, logging, and metrics | Reimplement algorithm losses |
| `evaluation` | Run deterministic or configured evaluation episodes and return records | Mutate training state |
| `checkpoints` | Persist and restore model/training state with metadata | Decide when an agent is trained |
| `experiments` | Record configuration, seed, timing, environment, algorithm, and metrics | Calculate algorithm-specific losses |
| `metrics` | Aggregate episode and evaluation measurements | Write UI-specific objects |
| `visualization` | Turn recorded results into Plotly figures/tables | Call an RL algorithm directly |
| `app` | Present controls and recorded outputs in Streamlit | Implement RL computation |

The dependency direction is one-way from the user interface toward recorded
results. Core RL modules must remain importable and usable from a command-line
entry point without importing Streamlit.

## Technical decisions

### JAX and neural networks

The implementation will use JAX arrays and explicit PRNG keys for numerical
computation, including network forward passes, loss functions, gradients, and
updates. `jax.jit`, `jax.vmap`, or equivalent transformations will be used
only where they improve a well-defined numerical path and remain compatible
with the environment boundary.

Flax is the provisional neural-network choice because it provides a clear
parameter-tree representation that integrates with Optax and future
checkpointing. The exact compatible versions will be selected during Phase 1
after checking the target Python interpreter and package availability.

### Randomness

No global mutable random state will be used by platform code. A seed will be
accepted explicitly, converted into a root JAX key, and split for agent
decisions, network initialization, replay sampling, and environment resets.
Environment seeding will use Gymnasium's supported reset options.

### Environment boundary

The factory will allow only `CartPole-v1` and `MountainCar-v0` initially and
will reject unsupported names with a clear configuration error. Both initial
environments have discrete actions, which makes them suitable for the first
DQN implementation. Observation and action-space details will be discovered
from the instantiated environment rather than hard-coded into algorithm
logic.

### Configuration

YAML configuration will be the source of experiment settings. The schema will
separate common run settings (environment, seed, output directories, episode
limits) from algorithm-specific settings. Validation will happen before an
environment is created or training starts. Configuration values such as
learning rate, gamma, batch size, episode count, seed, and output directories
will never be embedded in algorithm code.

### Data and persistence

Training and evaluation will produce structured records containing at least
episode number, reward, length, termination/truncation status, and elapsed
time. Experiment metadata will additionally capture the seed, environment,
algorithm, resolved hyperparameters, timestamp, training steps, and training
duration. Later phases will define the on-disk format and checkpoint schema;
this phase intentionally does not create those files.

## Planned project layout

The clean project setup in Phase 1 is expected to establish:

```text
jax-rl-platform/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
├── src/rl_platform/
│   ├── environments/
│   ├── networks/
│   ├── agents/
│   ├── algorithms/
│   ├── training/
│   ├── evaluation/
│   ├── checkpoints/
│   ├── experiments/
│   ├── metrics/
│   ├── visualization/
│   ├── configuration/
│   └── utils/
├── app/streamlit_app.py
├── tests/{unit,integration,smoke}/
├── experiments/
├── checkpoints/
├── results/
└── docs/
```

The runtime directories for generated artifacts will be ignored by version
control and created on demand. No generated artifacts belong in the source
tree.

## Verification strategy

The test suite will be fast and layered:

- Unit tests for configuration validation, environment construction, network
  shapes, loss functions, replay behavior, metrics, and serialization helpers.
- Integration tests for environment-to-agent interaction and training/evaluation
  orchestration.
- Smoke tests using tiny episode counts and rollout sizes to prove an actual
  end-to-end run without asserting a particular reward.

Every phase will run its targeted tests followed by the complete suite, import
checks, configuration checks, and an application-start check when applicable.
Failures will be diagnosed and fixed rather than suppressed or hidden by
weakening tests.

## Phase progression and gates

| Phase | Deliverable | Gate before proceeding |
| --- | --- | --- |
| 0 | Audit and architecture | Scope, boundaries, and isolation documented |
| 1 | Clean project setup | Package imports, dependency configuration, and baseline tests work |
| 2 | Environment layer | Both allow-listed environments reset and step correctly |
| 3 | JAX/RL foundations | Networks, PRNG handling, transitions, and shared interfaces are tested |
| 4 | DQN | DQN update math and small actual training run pass |
| 5 | Training/evaluation | CLI training and independent evaluation pass |
| 6 | Checkpoints/experiments | Save/load and metadata round-trip pass |
| 7 | PPO | PPO is implemented and tested without regressing DQN |
| 8 | Actor-Critic | Actor-Critic is implemented and tested without regressing earlier algorithms |
| 9 | Comparison/visualization | Recorded results produce comparable Plotly outputs |
| 10 | Streamlit dashboard | UI consumes results and does not own RL logic |
| 11 | Final QA | Full integration, imports, configuration, smoke tests, and documentation pass |

No implementation beyond Phase 0 is included in this document or the current
workspace changes.

## Phase 0 audit checks

The audit was performed against the workspace root on 2026-08-30.

- Python available: 3.13.5, satisfying the Python 3.11+ requirement.
- Present in the current interpreter: NumPy 2.1.3, Pandas 2.2.3, Plotly
  5.24.1, Streamlit 1.45.1, PyYAML 6.0.2, and pytest 8.3.4.
- Not present in the current interpreter: JAX, Optax, Gymnasium, Flax, and
  Equinox.
- Workspace-root pytest collection: no tests collected.
- No existing JAX/RL source or configuration was found.

Dependency installation and baseline project setup are intentionally deferred
to Phase 1.
