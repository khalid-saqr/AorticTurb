from __future__ import annotations

import numpy as np


def phase_decomposition(
    fields: np.ndarray,
    *,
    support: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-conditioned ensemble mean and residuals.

    Parameters
    ----------
    fields:
        Array shaped (N_realizations, N_nodes, 3). Each call must contain one phase only.
    support:
        Optional boolean node mask. Nodes outside support are returned as NaN.
    """
    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 3 or fields.shape[-1] != 3:
        raise ValueError("Expected fields shaped (N_realizations, N_nodes, 3).")
    if fields.shape[0] < 2:
        raise ValueError("At least two realizations are required.")

    mean = np.nanmean(fields, axis=0)
    residuals = fields - mean[None, :, :]

    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != (fields.shape[1],):
            raise ValueError("Support mask must have shape (N_nodes,).")
        mean = mean.copy()
        residuals = residuals.copy()
        mean[~support] = np.nan
        residuals[:, ~support, :] = np.nan

    return mean, residuals


def reynolds_stress(
    residuals: np.ndarray,
    *,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Reynolds-stress tensor and k at each node.

    residuals shape: (N_realizations, N_nodes, 3)
    returns R shape (N_nodes, 3, 3), k shape (N_nodes,)
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 3 or residuals.shape[-1] != 3:
        raise ValueError("Expected residuals shaped (N_realizations, N_nodes, 3).")

    n = np.isfinite(residuals).all(axis=-1).sum(axis=0)
    centered = np.where(np.isfinite(residuals), residuals, np.nan)
    R = np.full((residuals.shape[1], 3, 3), np.nan, dtype=float)

    for i in range(3):
        for j in range(3):
            products = centered[:, :, i] * centered[:, :, j]
            num = np.nansum(products, axis=0)
            denom = n - ddof
            with np.errstate(invalid="ignore", divide="ignore"):
                R[:, i, j] = np.where(denom > 0, num / denom, np.nan)

    k = 0.5 * np.trace(R, axis1=1, axis2=2)
    return R, k
