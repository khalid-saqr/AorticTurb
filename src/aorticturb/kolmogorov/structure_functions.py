from __future__ import annotations

import numpy as np


def longitudinal_increments(
    positions: np.ndarray,
    velocity: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    pairs = np.asarray(pairs, dtype=int)
    a = pairs[:, 0]
    b = pairs[:, 1]
    dr = positions[b] - positions[a]
    r = np.linalg.norm(dr, axis=1)
    valid = r > 0
    rhat = np.zeros_like(dr)
    rhat[valid] = dr[valid] / r[valid, None]
    du = velocity[b] - velocity[a]
    dul = np.einsum("ij,ij->i", du, rhat)
    return r, dul


def structure_function(
    r: np.ndarray,
    delta_u_l: np.ndarray,
    bin_edges: np.ndarray,
    *,
    order: int,
    absolute: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    d = np.asarray(delta_u_l, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(centers.shape, np.nan)
    counts = np.zeros(centers.shape, dtype=int)
    q = np.abs(d) ** order if absolute else d**order
    for idx in range(len(centers)):
        mask = (r >= edges[idx]) & (r < edges[idx + 1]) & np.isfinite(q)
        counts[idx] = int(mask.sum())
        if counts[idx]:
            values[idx] = float(np.mean(q[mask]))
    return centers, values, counts


def directional_structure_functions(
    velocity_grid: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    max_lag: int,
    orders: tuple[int, ...] = (2, 3, 4, 6),
) -> dict[int, dict[str, np.ndarray]]:
    """Directional longitudinal structure functions without assuming isotropy."""
    u = np.asarray(velocity_grid, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("Expected velocity_grid shape (nx, ny, nz, 3).")
    if not np.isfinite(u).all():
        raise ValueError("Structure-function window contains non-finite values.")

    result: dict[int, dict[str, np.ndarray]] = {}
    for axis in range(3):
        available = min(max_lag, u.shape[axis] - 1)
        rvals = []
        counts = []
        moments = {p: [] for p in orders}
        for lag in range(1, available + 1):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -lag)
            b[axis] = slice(lag, None)
            du_l = u[tuple(b) + (axis,)] - u[tuple(a) + (axis,)]
            flat = du_l.ravel()
            finite = flat[np.isfinite(flat)]
            rvals.append(lag * spacing[axis])
            counts.append(finite.size)
            for p in orders:
                if finite.size:
                    value = np.mean(np.abs(finite) ** p) if p != 3 else np.mean(finite**3)
                else:
                    value = np.nan
                moments[p].append(float(value))
        result[axis] = {
            "r": np.asarray(rvals, dtype=float),
            "count": np.asarray(counts, dtype=int),
            **{f"S{p}": np.asarray(values, dtype=float) for p, values in moments.items()},
        }
    return result


def ess_exponent(s3: np.ndarray, sp: np.ndarray) -> float | None:
    s3 = np.asarray(s3, dtype=float)
    sp = np.asarray(sp, dtype=float)
    valid = np.isfinite(s3) & np.isfinite(sp) & (np.abs(s3) > 0) & (sp > 0)
    if valid.sum() < 4:
        return None
    slope, _ = np.polyfit(np.log(np.abs(s3[valid])), np.log(sp[valid]), 1)
    return float(slope)


def local_log_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.log(x)
        ly = np.log(np.abs(y))
    return np.gradient(ly, lx)
