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


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int = 1000,
    seed: int = 20261024,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (np.nan, np.nan, np.nan)
    result = bootstrap_realizations(
        values[:, None],
        lambda x: float(np.mean(x[:, 0])),
        samples=samples,
        seed=seed,
        confidence=confidence,
    )
    return (float(result["lower"]), float(result["median"]), float(result["upper"]))


def bootstrap_phase_energy(
    fields: np.ndarray,
    support: np.ndarray,
    *,
    samples: int = 1000,
    seed: int = 20261024,
    confidence: float = 0.95,
    node_cap: int = 4096,
) -> tuple[float, float, float]:
    """Bootstrap phase-conditioned k, recomputing the mean on a deterministic node sample."""
    fields = np.asarray(fields, dtype=float)
    support = np.asarray(support, dtype=bool)
    ids = np.flatnonzero(support)
    if ids.size < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    if ids.size > node_cap:
        ids = np.sort(rng.choice(ids, size=node_cap, replace=False))
    sample_fields = fields[:, ids, :]
    n = sample_fields.shape[0]
    estimates = np.empty(samples, dtype=float)
    for s in range(samples):
        rid = rng.integers(0, n, size=n)
        draw = sample_fields[rid]
        mean = np.mean(draw, axis=0)
        residuals = draw - mean[None, ...]
        estimates[s] = 0.5 * float(np.mean(np.sum(residuals**2, axis=-1)))
    alpha = (1.0 - confidence) / 2.0
    q = np.quantile(estimates, [alpha, 0.5, 1.0 - alpha])
    return tuple(float(x) for x in q)
