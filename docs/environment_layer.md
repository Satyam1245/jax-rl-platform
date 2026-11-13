# Environment Layer

Phase 2 provides the only environment boundary used by later RL components.
It supports `CartPole-v1` and `MountainCar-v0` through
`rl_platform.environments.make_environment`.

## Factory

```python
from rl_platform.environments import make_environment

environment = make_environment("CartPole-v1", seed=42)
try:
    reset_result = environment.reset()
    step_result = environment.step(0)
finally:
    environment.close()
```

The factory validates names before calling Gymnasium. A seed supplied to the
factory is applied to the first reset when that reset has no explicit seed.
Callers can reseed any later reset with `environment.reset(seed=...)`.

## Spaces and results

`environment.observation_space` and `environment.action_space` expose the raw
Gymnasium spaces. Their stable inspection forms are available through
`observation_space_info`, `action_space_info`, and `spec`. The initial spaces
are:

| Environment | Observation | Action |
| --- | --- | --- |
| CartPole-v1 | `Box`, shape `(4,)` | `Discrete(2)` |
| MountainCar-v0 | `Box`, shape `(2,)` | `Discrete(3)` |

`reset()` returns a `ResetResult` containing `observation` and `info`.
`step(action)` returns a `StepResult` containing `observation`, `reward`,
`terminated`, `truncated`, and `info`. The two end flags are deliberately not
collapsed; `StepResult.done` is only a convenience that returns their logical
OR.

The adapter is a context manager and `close()` is idempotent. Calls to
`reset()` or `step()` after close raise `RuntimeError`.

No agent, algorithm, training, or Streamlit code belongs in this layer.
