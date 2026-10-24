from __future__ import annotations

import numpy as np


def anisotropy_tensor(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=float)
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must end with shape (3, 3).")
    k = 0.5 * np.trace(R, axis1=-2, axis2=-1)
    eye = np.eye(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = R / (2.0 * k[..., None, None]) - eye / 3.0
    b[~np.isfinite(b)] = np.nan
    return b, k


def invariants(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(b, dtype=float)
    if b.shape[-2:] != (3, 3):
        raise ValueError("b must end with shape (3, 3).")
    ii = -0.5 * np.einsum("...ij,...ji->...", b, b)
    iii = (1.0 / 3.0) * np.einsum("...ij,...jk,...ki->...", b, b, b)
    return ii, iii


def eigenvalues(b: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(np.asarray(b, dtype=float))
