"""Host-side replay buffer primitives for value-based reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any

import jax
import numpy as np


@dataclass(frozen=True, slots=True)
class Transition:
    """One environment transition recorded by a DQN agent."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        """Return whether the episode ended for either Gymnasium reason."""

        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A sampled batch suitable for conversion to JAX arrays."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of transitions in the batch."""

        return int(self.actions.shape[0])


class ReplayBuffer:
    """Fixed-capacity ring buffer with explicit-key sampling."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Replay capacity must be a positive integer")
        self._capacity = capacity
        self._storage: list[Transition | None] = [None] * capacity
        self._next_index = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        """Return the maximum number of stored transitions."""

        return self._capacity

    def __len__(self) -> int:
        return self._size

    def state_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly snapshot of the replay contents."""

        return {
            "capacity": self._capacity,
            "transitions": [
                {
                    "observation": transition.observation,
                    "action": transition.action,
                    "reward": transition.reward,
                    "next_observation": transition.next_observation,
                    "terminated": transition.terminated,
                    "truncated": transition.truncated,
                }
                for transition in self.transitions
            ],
        }

    @property
    def transitions(self) -> tuple[Transition, ...]:
        """Return stored transitions from oldest to newest."""

        if self._size == 0:
            return ()
        if self._size < self._capacity:
            indices: Iterable[int] = range(self._size)
        else:
            indices = ((self._next_index + offset) % self._capacity for offset in range(self._size))
        transitions = [self._storage[index] for index in indices]
        if any(transition is None for transition in transitions):
            raise RuntimeError("Replay buffer contains an unexpected empty slot")
        return tuple(transition for transition in transitions if transition is not None)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore replay contents after validating capacity compatibility."""

        if not isinstance(state, Mapping):
            raise ValueError("Replay state must be a mapping")
        if state.get("capacity") != self._capacity:
            raise ValueError("Replay capacity is incompatible with the checkpoint")
        transitions = state.get("transitions")
        if not isinstance(transitions, (list, tuple)) or len(transitions) > self._capacity:
            raise ValueError("Checkpoint replay transitions are invalid")
        restored: list[Transition] = []
        for item in transitions:
            if not isinstance(item, Mapping):
                raise ValueError("Checkpoint replay transition is invalid")
            restored.append(
                Transition(
                    observation=np.asarray(item["observation"], dtype=np.float32),
                    action=int(item["action"]),
                    reward=float(item["reward"]),
                    next_observation=np.asarray(item["next_observation"], dtype=np.float32),
                    terminated=bool(item["terminated"]),
                    truncated=bool(item["truncated"]),
                )
            )
        self._storage = [None] * self._capacity
        self._next_index = 0
        self._size = 0
        for transition in restored:
            self.add_transition(transition)

    def add(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Insert one transition, overwriting the oldest item when full."""

        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("Replay action must be an integer")
        transition = Transition(
            observation=np.asarray(observation, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_observation=np.asarray(next_observation, dtype=np.float32).copy(),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        self._storage[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def add_transition(self, transition: Transition) -> None:
        """Insert an existing transition."""

        self.add(
            transition.observation,
            transition.action,
            transition.reward,
            transition.next_observation,
            transition.terminated,
            transition.truncated,
        )

    def sample(self, batch_size: int, key: Any) -> ReplayBatch:
        """Sample unique transitions using an explicit JAX PRNG key."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Replay batch size must be a positive integer")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer containing {self._size}"
            )

        indices = np.asarray(
            jax.random.choice(key, self._size, shape=(batch_size,), replace=False),
            dtype=np.int64,
        )
        transitions = [self._storage[int(index)] for index in indices]
        if any(transition is None for transition in transitions):
            raise RuntimeError("Replay buffer contains an unexpected empty slot")
        valid_transitions = [transition for transition in transitions if transition is not None]
        return ReplayBatch(
            observations=np.stack([transition.observation for transition in valid_transitions]),
            actions=np.asarray([transition.action for transition in valid_transitions], dtype=np.int32),
            rewards=np.asarray([transition.reward for transition in valid_transitions], dtype=np.float32),
            next_observations=np.stack(
                [transition.next_observation for transition in valid_transitions]
            ),
            terminated=np.asarray(
                [transition.terminated for transition in valid_transitions], dtype=np.bool_
            ),
            truncated=np.asarray(
                [transition.truncated for transition in valid_transitions], dtype=np.bool_
            ),
        )
