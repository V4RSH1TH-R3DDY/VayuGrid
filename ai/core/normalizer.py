from __future__ import annotations

import numpy as np


class ObservationNormalizer:
    """Running mean/std normalizer for observations.

    Normalisation statistics are stored alongside the model checkpoint
    so that deployed agents normalise observations identically to training.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.count = 0

    def update(self, obs: np.ndarray) -> None:
        batch = obs if obs.ndim > 1 else obs[np.newaxis, :]
        self.count += len(batch)
        delta = batch - self.mean
        self.mean += delta.sum(axis=0) / self.count
        delta2 = batch - self.mean
        self.std = np.sqrt(((delta * delta2).sum(axis=0) + 1e-6) / self.count)

    def normalize(self, obs: np.ndarray, inplace: bool = False) -> np.ndarray:
        if inplace:
            obs -= self.mean
            obs /= np.clip(self.std, 1e-6, None)
            return obs
        return (obs - self.mean) / np.clip(self.std, 1e-6, None)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.std = state["std"]
        self.count = state.get("count", 0)
