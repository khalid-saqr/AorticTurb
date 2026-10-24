from __future__ import annotations

import numpy as np


def velocity_gradient(
    velocity_grid: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    u = np.asarray(velocity_grid, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("Expected velocity_grid shape (nx, ny, nz, 3).")
    if not np.isfinite(u).all():
        raise ValueError("Gradient window contains non-finite values.")
    grad = np.empty(u.shape[:3] + (3, 3), dtype=float)
    for i in range(3):
        d = np.gradient(u[..., i], *spacing, edge_order=2)
        for j in range(3):
            grad[..., i, j] = d[j]
    return grad


def strain_rate(grad: np.ndarray) -> np.ndarray:
    return 0.5 * (grad + np.swapaxes(grad, -1, -2))


def vorticity(grad: np.ndarray) -> np.ndarray:
    grad = np.asarray(grad, dtype=float)
    wx = grad[..., 2, 1] - grad[..., 1, 2]
    wy = grad[..., 0, 2] - grad[..., 2, 0]
    wz = grad[..., 1, 0] - grad[..., 0, 1]
    return np.stack([wx, wy, wz], axis=-1)


def enstrophy(omega: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.asarray(omega, dtype=float) ** 2, axis=-1)


def resolved_dissipation_surrogate(strain: np.ndarray) -> np.ndarray:
    return 2.0 * np.einsum("...ij,...ij->...", strain, strain)
