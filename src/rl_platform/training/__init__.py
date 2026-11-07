"""Generic, algorithm-independent training engine."""

from rl_platform.training.config import TrainingConfig
from rl_platform.training.engine import (
    EnvironmentTransition,
    TrainableAgent,
    TrainingEngine,
    TrainingResult,
)

__all__ = [
    "EnvironmentTransition",
    "TrainableAgent",
    "TrainingConfig",
    "TrainingEngine",
    "TrainingResult",
]
