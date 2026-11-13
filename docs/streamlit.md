# Streamlit dashboard

Phase 10 adds `app/streamlit_app.py` as a presentation layer over the tested
backend. It does not implement DQN, PPO, Actor-Critic, training loops,
evaluation loops, checkpoint serialization, or result parsing.

The dashboard provides these sections:

- **Overview** — project description, available environments/algorithms, and
  recent stored experiments.
- **Train Agent** — editable environment, algorithm, hyperparameter, seed, and
  run controls. Training is delegated to `rl_platform.services.runtime`.
- **Training Progress** — completed-run episode/step, reward, average reward,
  loss, status, and metric table.
- **Evaluate Agent** — checkpoint selection and deterministic evaluation with
  mean reward, distribution, and episode statistics.
- **Experiment History** — persisted experiment table and stored metric plots.
- **Algorithm Comparison** — DQN/PPO/Actor-Critic comparison statistics and
  Plotly figures.
- **Checkpoints** — saved checkpoint metadata and paths.

The runtime service assembles the selected existing agent and delegates to the
generic training/evaluation engines and experiment/checkpoint managers. A UI
training request persists metrics and saves a checkpoint for subsequent
evaluation.

Run locally from the project directory:

```powershell
python -m streamlit run app/streamlit_app.py
```

