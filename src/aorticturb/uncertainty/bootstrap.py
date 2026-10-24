from __future__ import annotations

from typing import Callable
import numpy as np


def bootstrap_realizations(
    fields: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray | float],
    *,
    samples: int = 1000,
    seed: int = 20261024,
    confidence: float = 0.95,
) -> dict[str, np.ndarray]:
    """Bootstrap only across the realization axis."""
    fields = np.asarray(fields)
    if fields.ndim < 1:
        raise ValueError("fields must have a realization axis")
    n = fields.shape[0]
    if n < 2:
        raise ValueError("At least two realizations are required.")

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        estimates.append(np.asarray(statistic(fields[idx])))
    stacked = np.stack(estimates)
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": np.quantile(stacked, alpha, axis=0),
        "median": np.quantile(stacked, 0.5, axis=0),
        "upper": np.quantile(stacked, 1.0 - alpha, axis=0),
    }
