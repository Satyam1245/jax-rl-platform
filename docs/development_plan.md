# JAX Reinforcement Learning Platform Development Plan

## Phase 0 repository audit

Audit date: 2026-08-30.

The workspace root is a directory containing several unrelated projects, not a
single Git repository. No existing JAX reinforcement-learning implementation
was found. The requested platform is isolated under `jax-rl-platform/` so it
does not share imports, dependency environments, generated artifacts, or
application entry points with those projects.

### Existing directories and files

The relevant non-generated inventory is:

| Location | Contents | Approximate non-generated inventory |
| --- | --- | ---: |
| `AI/` | Standalone scripts, a face-mask project, and a cricket analytics application | 113 files, 61 Python files |
| `cricket-ai-platform-phase1/` | Separate cricket streaming platform with backend, AI services, frontend, streaming, and workers | 153 files, 77 Python files |
| `Data-Analytics-Internship/` | Notebooks, CSV datasets, reports, images, and one utility module | 19 files, 1 Python file |
| `Digit Recognizer/` | Notebook and test images | 13 files |
| `Face_Recog_Attendance/` | Webcam capture, training, recognition, and attendance data | 8 files, 4 Python files |
| `multimodal-deepfake-detector-backup/` | FastAPI app, PyTorch models, media processing, scripts, and frontend assets | 39 files, 26 Python files |
| `jax-rl-platform/` | Phase 0 architecture documentation only | 2 documentation files |

The workspace also contains `.vscode/`, `.sixth/`, a generated
`.pytest_cache/`, notebook metadata, datasets, model artifacts, and a bundled
virtual environment. These are not source dependencies of the RL platform.

### Existing Python code

Existing Python code belongs to unrelated applications:

- `AI/TimeTable.py` is a Streamlit timetable application.
- `AI/Face-Mask-Detection/src/` contains TensorFlow/Keras training,
  prediction, webcam detection, and dataset-audit scripts.
- `AI/cricket-ai-platform/` contains FastAPI backend and AI-service modules,
  repositories, services, schemas, database models, workers, and Alembic
  migration code.
- `cricket-ai-platform-phase1/` contains a second, more complete cricket
  platform with FastAPI services, SQLAlchemy models, repositories, workers,
  Redis integration, and AI pipelines.
- `Data-Analytics-Internship/src/utils.py` contains a Pandas utility.
- `Face_Recog_Attendance/` contains OpenCV/geocoder scripts.
- `multimodal-deepfake-detector-backup/` contains PyTorch, OpenCV, audio,
  video, FastAPI, and command-line inference/training code.

There are no `rl_platform` modules, Gymnasium environment adapters, JAX
networks, agents, algorithms, replay buffers, rollout collectors, or RL
training engines to reuse.

### Existing dependencies

Existing dependency manifests are project-local rather than workspace-wide:

- The two cricket projects use `pyproject.toml` files with FastAPI,
  SQLAlchemy/Alembic, Redis, Pydantic, Uvicorn, NumPy, and optional PyTorch or
  Transformers dependencies.
- The face-mask project pins TensorFlow, Keras, OpenCV, NumPy, Matplotlib,
  scikit-learn, and imutils.
- The data-analysis project uses Pandas, NumPy, Matplotlib, Seaborn,
  scikit-learn, and SciPy.
- The face-recognition project uses OpenCV and geocoder.
- The deepfake project uses PyTorch, torchvision, OpenCV, facenet-pytorch,
  EfficientNet, Albumentations, Pillow, Transformers, SciPy, PyYAML, FastAPI,
  and Uvicorn.

The RL platform should not import or inherit these application dependencies.
Its dependency set should remain limited to the requested Python/JAX stack
and standard library unless a later phase proves a genuine requirement.

### Python version and installed packages

The current interpreter is:

- Python 3.13.5
- Executable: `C:\Users\Satya\anaconda3\python.exe`

Relevant packages visible to the current interpreter are:

| Package | Installed version |
| --- | --- |
| NumPy | 2.1.3 |
| Pandas | 2.2.3 |
| Plotly | 5.24.1 |
| Streamlit | 1.45.1 |
| PyYAML | 6.0.2 |
| pytest | 8.3.4 |
| TensorFlow | 2.21.0 |
| scikit-learn | 1.6.1 |
| JAX | Not installed |
| jaxlib | Not installed |
| Optax | Not installed |
| Gymnasium | Not installed |
| Flax | Not installed |
| Equinox | Not installed |
| Ruff | Not installed |

The project must use an isolated environment. Existing manifests require
Python `>=3.12,<3.13` in the cricket applications, while the current
interpreter is Python 3.13.5. This mismatch is a direct reason not to reuse
those environments. Phase 1 must verify a compatible JAX/Flax/Optax/Gymnasium
combination before writing pins.

### Existing tests

There is no workspace-level test suite. The cricket Phase 1 project contains
empty `ai-services/tests/` and `backend/tests/` directories. The deepfake
project has `scripts/test.py`, which is an application test/inference script,
not a pytest suite. Image directories named `test_images` contain data rather
than tests. Workspace-root pytest collection found zero tests.

The RL platform must create its own `tests/unit`, `tests/integration`, and
`tests/smoke` suites in Phase 1 onward.

### Existing configuration

No root `pyproject.toml`, `requirements.txt`, `setup.py`, `setup.cfg`,
`.gitignore`, pytest configuration, YAML experiment configuration, or lint
configuration exists. Existing project-local configuration includes:

- Four cricket `pyproject.toml` manifests with Ruff settings and, in one
  backend, pytest settings.
- Two cricket Docker Compose files and Dockerfiles.
- Two Alembic configurations.
- Frontend `package.json`, TypeScript, Vite, and Tailwind configuration.
- A deepfake Python configuration module supporting YAML/JSON.
- No configuration specific to the RL platform.

RL configuration must be introduced in the isolated project and validated
before training starts. It must not rely on environment variables or defaults
from the existing applications.

### Existing documentation

Documentation exists for the unrelated cricket platforms, face-mask project,
face-recognition project, data-analysis project, and deepfake project. It
covers their own architecture, setup, Docker services, streaming, APIs, and
model workflows. No existing documentation describes JAX, Gymnasium, DQN,
PPO, Actor-Critic, or the requested RL platform. The RL platform currently has
only its Phase 0 architecture document and this development plan.

### Existing Git configuration

No `.git` directory was found at the workspace root or in the inspected
project directories, and `git rev-parse --show-toplevel` does not resolve a
repository. Git only reported the system-level configuration supplied by the
runtime, including CRLF normalization and the credential helper. There is no
repository-local branch, remote, ignore file, or commit history available for
the RL platform.

### Existing application entry points

The existing applications expose unrelated entry points:

- Streamlit: `AI/TimeTable.py`.
- FastAPI: the two cricket backends, the two cricket AI services, and
  `multimodal-deepfake-detector-backup/app.py`.
- CLI scripts: face-mask training/prediction/audit, face-recognition capture/
  training/recognition, and deepfake augmentation/training/inference/testing.
- Node/Vite: cricket frontend development/build/preview and WebSocket gateway
  development/build/start scripts.
- Docker: cricket backend/AI-service Uvicorn commands, workers, Nginx
  streaming, frontend, and WebSocket gateway services.

None should be treated as the RL platform entry point. The RL platform should
eventually expose a command-line training/evaluation entry point and a
separate Streamlit dashboard, with the dashboard importing recorded results
only.

## Proposed architecture

```text
YAML configuration
        |
Environment factory -> Gymnasium adapters
        |
Agent interface <-> Algorithm implementation <-> JAX networks + Optax
        |
Training engine -> Evaluation engine
        |
Checkpointing -> Experiment tracking -> Recorded results
        |
Metrics -> Plotly visualization -> Streamlit dashboard
```

The core dependency direction is one-way. Environment adapters know about
Gymnasium; algorithms know about agent state, networks, and Optax; training
coordinates algorithms and environments; evaluation consumes a trained agent
without owning training; visualization consumes records; and Streamlit is a
presentation layer. No core RL module may import Streamlit.

The initial environment allow-list is `CartPole-v1` and `MountainCar-v0`, with
CartPole-v1 as the primary demonstration. The algorithms are implemented in
the required order: DQN, then PPO, then Actor-Critic.

## Module responsibilities

| Module | Responsibility | Boundary |
| --- | --- | --- |
| `configuration` | Load YAML, merge explicitly documented sections, validate types/ranges/paths | Must reject invalid configuration before training |
| `environments` | Allow-listed environment factory, reset/step normalization, space metadata, optional wrappers | Must not contain algorithm hyperparameters |
| `networks` | Flax/JAX parameterized networks and pure forward functions | Must not own episodes or filesystem state |
| `agents` | Action-selection interface, policy/Q-value access, explicit RNG state | Must not implement optimizer update mathematics |
| `algorithms` | DQN, PPO, and Actor-Critic losses, targets, advantages, and Optax updates | Must be usable without Streamlit or training orchestration |
| `training` | Episode/rollout coordination, update scheduling, logging, and metric emission | Must call algorithms rather than duplicate their losses |
| `evaluation` | Independent rollout execution, deterministic evaluation mode, evaluation records | Must not mutate training state |
| `checkpoints` | Save/load parameter and optimizer state plus metadata | Must not infer performance claims |
| `experiments` | Run identity, resolved configuration, timestamps, seed, duration, and artifact locations | Must record facts, not fabricate metrics |
| `metrics` | Episode summaries, moving statistics, comparison-ready tabular records | Must remain UI-independent |
| `visualization` | Plotly figures and comparison tables from recorded results | Must not call environment or agent methods |
| `utils` | Small shared helpers such as logging, paths, and serialization primitives | Must avoid becoming a second service layer |
| `app` | Streamlit controls and presentation of recorded runs | Must contain no RL algorithm implementation |

## Dependency decisions

- Use Python 3.11+ as the supported language baseline; select and pin the
  exact compatible interpreter/package matrix in Phase 1.
- Use JAX and `jax.numpy` for numerical computation, model forward passes,
  losses, gradients, and optimizer-facing updates.
- Use explicit `jax.random.PRNGKey` values and key splitting. Do not use global
  mutable randomness.
- Use Optax for optimizer definitions and parameter updates.
- Use Gymnasium for the two initial environments and preserve the distinction
  between `terminated` and `truncated`.
- Use Flax provisionally for neural-network parameter trees and integration
  with JAX and Optax. Equinox remains an alternative only if the Phase 1
  compatibility check shows a material advantage.
- Use NumPy/Pandas at the environment and recorded-results boundaries where
  ordinary Python data structures are appropriate; JAX remains responsible
  for actual learning computation.
- Use PyYAML for configuration loading and Plotly for result visualization.
- Use Streamlit only for the later dashboard phase.
- Use pytest for tests and Python logging for controlled diagnostics.
- Do not add databases, web APIs, experiment-tracking services, deep-learning
  frameworks, or extra environments unless a later phase establishes a real
  requirement.

## Phase dependencies

| Phase | Depends on | Deliverable and completion gate |
| --- | --- | --- |
| 0. Audit and architecture | None | Scope, isolation, architecture, and risks documented |
| 1. Clean project setup | Phase 0 | Isolated package, dependency manifests, baseline config, imports, and tests |
| 2. Environment layer | Phase 1 | CartPole and MountainCar construction/reset/step tests |
| 3. JAX/RL foundations | Phase 2 | Networks, transitions, PRNG handling, shared interfaces, and optimizer utilities |
| 4. DQN | Phase 3 | Actual DQN update path, replay buffer, target network, and tiny training smoke test |
| 5. Training/evaluation | Phase 4 | CLI training and independent evaluation engines |
| 6. Checkpointing/experiments | Phase 5 | Round-trip checkpoints and complete experiment metadata |
| 7. PPO | Phase 6 | PPO rollout, clipped objective, value loss, and regression-safe tests |
| 8. Actor-Critic | Phase 7 | Actor-Critic update path and regression-safe tests |
| 9. Comparison/visualization | Phase 8 | Result comparison and Plotly figures from recorded data |
| 10. Streamlit dashboard | Phase 9 | Dashboard that consumes results without owning RL logic |
| 11. Full integration/final QA | Phase 10 | Full suite, smoke run, import/config checks, startup checks, and documentation |

No phase should begin until the preceding phase's gate passes. Algorithms must
not be developed in parallel: DQN must be complete before PPO, and PPO before
Actor-Critic.

## Testing strategy

### Unit tests

Test small, deterministic responsibilities in isolation:

- Configuration schema, validation errors, and path handling.
- Environment allow-listing and Gymnasium reset/step normalization.
- Network initialization, output shapes, and pure forward functions.
- PRNG key splitting and reproducibility properties.
- Transition/replay-buffer insertion and sampling behavior.
- DQN/PPO/Actor-Critic loss terms and target/advantage calculations.
- Metrics aggregation, result schemas, and checkpoint metadata helpers.

### Integration tests

Test component boundaries without long training runs:

- Environment factory to agent action selection.
- Agent/algorithm update with a small synthetic batch or rollout.
- Training engine coordination with a tiny real Gymnasium environment run.
- Evaluation independently loading and using a trained state.
- Experiment record and checkpoint round trips.
- Visualization consuming saved records rather than algorithm objects.

### Smoke tests

Use real computation and tiny configuration values, such as 2–5 episodes,
small batches, and short rollouts. Smoke tests should verify that the complete
path starts, steps an actual environment, updates actual parameters, records
real rewards, and writes expected artifacts. They must not require a reward
threshold or claim learning from an unvalidated short run.

Every phase should run targeted tests first, then the complete suite, import
checks, configuration checks, and linting if configured. The dashboard and
other application startup checks apply once those components exist.

## Known risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| JAX/Flax/Optax/Python compatibility | Installation or import failures | Use an isolated environment and verify the exact matrix in Phase 1 |
| CPU/GPU differences and JAX compilation | Slow or nondeterministic runs | Start with CPU-compatible code, explicit seeds, small tests, and document hardware nondeterminism |
| Incorrect PRNG reuse | Hidden reproducibility and exploration bugs | Pass keys explicitly and test deterministic key derivation |
| Gymnasium API semantics | Incorrect bootstrapping at truncation/termination | Preserve separate flags in transitions and test both cases |
| DQN instability | Non-learning or divergent training | Keep target-network, replay, epsilon, and update rules independently testable; avoid performance claims from smoke tests |
| Discrete-action limitation | PPO/Actor-Critic design may overfit initial environments | Define a clear discrete-action interface first and document extension points |
| Host/device data transfer | Slow replay sampling or compilation issues | Keep the replay buffer host-side initially and convert validated batches at the algorithm boundary |
| Checkpoint schema changes | Inability to resume experiments | Store versioned metadata and test round trips before adding new fields |
| Configuration drift | Runs that cannot be reproduced | Persist resolved configuration, seed, environment, algorithm, timestamp, steps, and duration |
| UI/core coupling | Training becomes unavailable outside Streamlit | Keep dashboard imports at the edge and add a command-line smoke test |
| Existing workspace contamination | Import collisions and dependency conflicts | Keep all RL code under `jax-rl-platform`, use a dedicated environment, and avoid modifying existing projects |
| No repository metadata | No branch/remote/history safety net | Do not assume Git operations; keep changes limited to the requested project path |

## Recommended implementation order

1. Complete the audit and freeze the architecture boundaries (this phase).
2. Create the isolated package, manifests, configuration skeleton, logging
   conventions, and baseline tests in Phase 1.
3. Implement and test the two-environment factory in Phase 2.
4. Establish JAX/Flax network conventions, transition types, PRNG handling,
   and shared agent/algorithm interfaces in Phase 3.
5. Implement DQN fully and prove an actual tiny run in Phase 4.
6. Add reusable training and independent evaluation engines in Phase 5.
7. Add checkpointing and experiment metadata in Phase 6.
8. Implement PPO only after DQN and shared infrastructure are stable.
9. Implement Actor-Critic only after PPO regression tests pass.
10. Add comparison records and Plotly visualization.
11. Add the Streamlit dashboard as a consumer of recorded results.
12. Run final integration and QA across all supported algorithms and
    environments.

Phase 1 is intentionally not implemented by this document.
