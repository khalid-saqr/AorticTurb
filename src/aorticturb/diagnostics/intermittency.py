from __future__ import annotations

import numpy as np


def flatness_from_moments(s2: np.ndarray, s4: np.ndarray) -> np.ndarray:
    s2 = np.asarray(s2, dtype=float)
    s4 = np.asarray(s4, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = s4 / (s2**2)
    out[~np.isfinite(out)] = np.nan
    return out


def increment_flatness(
    velocity_grid: np.ndarray,
    *,
    axis: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(velocity_grid, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("Expected velocity_grid shape (nx, ny, nz, 3).")
    if not np.isfinite(u).all():
        raise ValueError("Flatness window contains non-finite values.")
    lags = []
    values = []
    for lag in range(1, min(max_lag, u.shape[axis] - 1) + 1):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis] = slice(0, -lag)
        b[axis] = slice(lag, None)
        d = (u[tuple(b) + (axis,)] - u[tuple(a) + (axis,)]).ravel()
        d = d[np.isfinite(d)]
        if d.size:
            s2 = np.mean(d**2)
            s4 = np.mean(d**4)
            value = s4 / (s2**2) if s2 > 0 else np.nan
        else:
            value = np.nan
        lags.append(lag)
        values.append(value)
    return np.asarray(lags), np.asarray(values, dtype=float)
