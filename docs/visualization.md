# Experiment comparison and visualization

Phase 9 reads the persisted experiment artifacts created by Phase 6. It does
not import or execute any RL algorithm.

## Stored results

`load_experiment_results(experiments_directory, results_directory)` validates
each experiment's `metadata.json` and `metrics.json`, returning
`ExperimentResult` objects. Missing or malformed fields raise
`ResultValidationError`; a missing experiment root returns an empty result
collection. Missing optional duration data remains `None` and is never
reconstructed.

## Comparison metrics

`compare_experiments` groups actual loaded results by algorithm and computes
mean reward, median reward, population standard deviation, best reward, mean
evaluation reward, and training duration. If a metric has no stored source
data, it remains `None`.

## Plotly figures

The visualization package provides Plotly builders for reward curves,
moving-average reward, episode length, loss, evaluation reward, reward
distribution, algorithm comparison, and training duration. Every builder
accepts validated `ExperimentResult` objects and returns a Plotly `Figure`.

Training duration is persisted by passing the measured value from
`TrainingResult.duration_seconds` to `ExperimentManager.persist_metrics`:

```python
manager.persist_metrics(
    experiment,
    training.metrics,
    training_duration_seconds=training.duration_seconds,
    total_steps=training.total_steps,
)
```

