from __future__ import annotations

import numpy as np


def leave_one_out_mean_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {
            "full_mean": np.nan,
            "loo_min": np.nan,
            "loo_max": np.nan,
            "loo_max_abs_deviation": np.nan,
            "loo_max_relative_deviation": np.nan,
        }
    full = float(np.mean(values))
    loo = np.asarray(
        [np.mean(np.delete(values, idx)) for idx in range(values.size)], dtype=float
    )
    max_abs = float(np.max(np.abs(loo - full)))
    if full != 0.0 and np.isfinite(full):
        max_rel = max_abs / abs(full)
    else:
        max_rel = np.nan
    return {
        "full_mean": full,
        "loo_min": float(np.min(loo)),
        "loo_max": float(np.max(loo)),
        "loo_max_abs_deviation": max_abs,
        "loo_max_relative_deviation": float(max_rel),
    }


def subset_mean_convergence(
    values: np.ndarray,
    *,
    sizes: tuple[int, ...] = (4, 8, 12, 16),
    draws: int = 256,
    seed: int = 20261024,
    confidence: float = 0.95,
) -> list[dict[str, float | int]]:
    """Descriptive ensemble-size convergence using subsets without replacement."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return []
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict[str, float | int]] = []
    for size in sorted({int(x) for x in sizes if 2 <= int(x) <= n}):
        if size == n:
            estimates = np.asarray([np.mean(values)], dtype=float)
        else:
            estimates = np.empty(int(draws), dtype=float)
            for idx in range(int(draws)):
                ids = rng.choice(n, size=size, replace=False)
                estimates[idx] = float(np.mean(values[ids]))
        rows.append(
            {
                "ensemble_size": size,
                "lower": float(np.quantile(estimates, alpha)),
                "median": float(np.quantile(estimates, 0.5)),
                "upper": float(np.quantile(estimates, 1.0 - alpha)),
                "draws": int(estimates.size),
            }
        )
    return rows
