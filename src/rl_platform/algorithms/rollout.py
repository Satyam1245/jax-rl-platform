"""Host-side rollout storage for on-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """A completed rollout represented as dense NumPy arrays."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    old_log_probs: np.ndarray
    values: np.ndarray

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True, slots=True)
class _RolloutItem:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    old_log_prob: float
    value: float


class RolloutBuffer:
    """Fixed-length transition collection with explicit episode-end support."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Rollout capacity must be a positive integer")
        self.capacity = capacity
        self._items: list[_RolloutItem] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def ready(self) -> bool:
        """Return whether the rollout should trigger an update."""

        return bool(self._items) and (
            len(self._items) >= self.capacity or self._items[-1].terminated or self._items[-1].truncated
        )

    @property
    def last_done(self) -> bool:
        """Return whether the most recent transition ended an episode."""

        return bool(self._items) and (self._items[-1].terminated or self._items[-1].truncated)

    def add(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
        old_log_prob: float,
        value: float,
    ) -> None:
        """Append one transition to the current rollout."""

        if len(self._items) >= self.capacity:
            raise ValueError("Rollout is full; consume it before adding another transition")
        self._items.append(
            _RolloutItem(
                observation=np.asarray(observation, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_observation=np.asarray(next_observation, dtype=np.float32).copy(),
                terminated=bool(terminated),
                truncated=bool(truncated),
                old_log_prob=float(old_log_prob),
                value=float(value),
            )
        )

    def as_batch(self) -> RolloutBatch:
        """Return the current rollout as dense arrays."""

        if not self._items:
            raise ValueError("Cannot create a batch from an empty rollout")
        return RolloutBatch(
            observations=np.stack([item.observation for item in self._items]),
            actions=np.asarray([item.action for item in self._items], dtype=np.int32),
            rewards=np.asarray([item.reward for item in self._items], dtype=np.float32),
            next_observations=np.stack([item.next_observation for item in self._items]),
            terminated=np.asarray([item.terminated for item in self._items], dtype=np.bool_),
            truncated=np.asarray([item.truncated for item in self._items], dtype=np.bool_),
            old_log_probs=np.asarray([item.old_log_prob for item in self._items], dtype=np.float32),
            values=np.asarray([item.value for item in self._items], dtype=np.float32),
        )

    def clear(self) -> None:
        """Discard the consumed rollout."""

        self._items.clear()

    def state_dict(self) -> dict[str, Any]:
        """Return the in-progress rollout in a checkpoint-safe representation."""

        return {
            "capacity": self.capacity,
            "items": [
                {
                    "observation": item.observation,
                    "action": item.action,
                    "reward": item.reward,
                    "next_observation": item.next_observation,
                    "terminated": item.terminated,
                    "truncated": item.truncated,
                    "old_log_prob": item.old_log_prob,
                    "value": item.value,
                }
                for item in self._items
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an in-progress rollout with capacity and shape validation."""

        if not isinstance(state, Mapping):
            raise ValueError("Rollout checkpoint state must be a mapping")
        capacity = state.get("capacity")
        items = state.get("items")
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)):
            raise ValueError("Rollout checkpoint capacity must be an integer")
        if int(capacity) != self.capacity:
            raise ValueError(
                f"Rollout capacity mismatch: expected {self.capacity}, found {int(capacity)}"
            )
        if not isinstance(items, list) or len(items) > self.capacity:
            raise ValueError("Rollout checkpoint items must be a list within capacity")

        restored: list[_RolloutItem] = []
        required = {
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
            "old_log_prob",
            "value",
        }
        for item in items:
            if not isinstance(item, Mapping) or not required.issubset(item):
                raise ValueError("Rollout checkpoint item is malformed")
            restored.append(
                _RolloutItem(
                    observation=np.asarray(item["observation"], dtype=np.float32).copy(),
                    action=int(item["action"]),
                    reward=float(item["reward"]),
                    next_observation=np.asarray(item["next_observation"], dtype=np.float32).copy(),
                    terminated=bool(item["terminated"]),
                    truncated=bool(item["truncated"]),
                    old_log_prob=float(item["old_log_prob"]),
                    value=float(item["value"]),
                )
            )
        self._items = restored
