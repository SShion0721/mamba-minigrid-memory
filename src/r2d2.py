"""R2D2-style recurrent replay primitives.

This module intentionally keeps the learner policy-agnostic. It provides the
sequence storage, burn-in slicing, and priority update mechanics needed by a
future recurrent Q learner without coupling that learner to MiniGrid wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReplaySequence:
    data: dict[str, np.ndarray]
    priority: float


class RecurrentSequenceReplay:
    """Small prioritized replay buffer for fixed-format recurrent sequences."""

    def __init__(
        self,
        capacity: int,
        sequence_len: int,
        burn_in_len: int,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        seed: int = 0,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if sequence_len <= 0:
            raise ValueError("sequence_len must be positive.")
        if burn_in_len < 0 or burn_in_len >= sequence_len:
            raise ValueError("burn_in_len must satisfy 0 <= burn_in_len < sequence_len.")
        self.capacity = capacity
        self.sequence_len = sequence_len
        self.burn_in_len = burn_in_len
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self._items: list[ReplaySequence] = []
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, data: dict[str, np.ndarray], priority: float | None = None) -> int:
        self._validate_data(data)
        if priority is None:
            priority = max((item.priority for item in self._items), default=1.0)
        item = ReplaySequence(data={key: np.asarray(value).copy() for key, value in data.items()}, priority=float(priority))
        if len(self._items) < self.capacity:
            self._items.append(item)
            idx = len(self._items) - 1
        else:
            idx = self._next_idx
            self._items[idx] = item
        self._next_idx = (idx + 1) % self.capacity
        return idx

    def sample(self, batch_size: int) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
        if not self._items:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        priorities = np.asarray([item.priority for item in self._items], dtype=np.float64)
        probs = np.power(priorities + self.epsilon, self.alpha)
        probs /= probs.sum()
        indices = self.rng.choice(len(self._items), size=batch_size, replace=len(self._items) < batch_size, p=probs)
        weights = np.power(len(self._items) * probs[indices], -self.beta)
        weights = weights / weights.max()
        batch = [self._items[int(idx)].data for idx in indices]
        return batch, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self._items[int(idx)].priority = float(max(priority, self.epsilon))

    def _validate_data(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            arr = np.asarray(value)
            if arr.shape[0] != self.sequence_len:
                raise ValueError(f"{key} has first dimension {arr.shape[0]}, expected {self.sequence_len}.")
