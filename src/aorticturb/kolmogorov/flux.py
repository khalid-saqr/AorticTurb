from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def coarse_grained_flux(
    velocity_grid: np.ndarray,
    spacing: tuple[float, float, float],
    sigma: float,
) -> np.ndarray:
    """Compute Pi_ell = -tau_ij Sbar_ij using a Gaussian filter.

    The caller is responsible for providing a fully finite local rectangular window and
    relating `sigma` to the physical filter scale.
    """
    u = np.asarray(velocity_grid, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("Expected velocity_grid shape (nx, ny, nz, 3).")
    if not np.isfinite(u).all():
        raise ValueError("Flux window contains non-finite values.")

    ubar = np.stack(
        [gaussian_filter(u[..., i], sigma=sigma, mode="nearest") for i in range(3)],
        axis=-1,
    )

    tau = np.empty(u.shape[:3] + (3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            filtered_product = gaussian_filter(
                u[..., i] * u[..., j], sigma=sigma, mode="nearest"
            )
            tau[..., i, j] = filtered_product - ubar[..., i] * ubar[..., j]

    grad = np.empty(u.shape[:3] + (3, 3), dtype=float)
    for i in range(3):
        derivatives = np.gradient(ubar[..., i], *spacing, edge_order=2)
        for j in range(3):
            grad[..., i, j] = derivatives[j]

    strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    return -np.einsum("...ij,...ij->...", tau, strain)
