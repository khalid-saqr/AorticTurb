from __future__ import annotations

import numpy as np


def longitudinal_increments(
    positions: np.ndarray,
    velocity: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute separation magnitude and longitudinal velocity increment for node pairs."""
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


def local_log_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.log(x)
        ly = np.log(np.abs(y))
    return np.gradient(ly, lx)
