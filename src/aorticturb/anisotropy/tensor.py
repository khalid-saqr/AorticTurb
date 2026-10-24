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
    """Return descending eigenvalues, preserving nonfinite nodes as NaN.

    The anisotropy field legitimately contains NaNs outside finite measurement support.
    NumPy's batched ``eigvalsh`` can fail the entire batch when any matrix is nonfinite,
    so only fully finite 3x3 tensors are submitted to the eigensolver. Invalid nodes
    retain NaN eigenvalues and are excluded downstream by the existing support/validity
    masks.
    """
    b = np.asarray(b, dtype=float)
    if b.shape[-2:] != (3, 3):
        raise ValueError("b must end with shape (3, 3).")

    flat = b.reshape((-1, 3, 3))
    values = np.full((flat.shape[0], 3), np.nan, dtype=float)
    finite = np.isfinite(flat).all(axis=(1, 2))

    if finite.any():
        solved = np.linalg.eigvalsh(flat[finite])
        values[finite] = np.flip(solved, axis=-1)

    return values.reshape(b.shape[:-2] + (3,))


def barycentric_coordinates(b: np.ndarray) -> np.ndarray:
    """Return (C1, C2, C3) realizability weights from descending eigenvalues."""
    lam = eigenvalues(b)
    c1 = lam[..., 0] - lam[..., 1]
    c2 = 2.0 * (lam[..., 1] - lam[..., 2])
    c3 = 3.0 * lam[..., 2] + 1.0
    return np.stack([c1, c2, c3], axis=-1)


def realizability_mask(b: np.ndarray, *, atol: float = 1e-10) -> np.ndarray:
    c = barycentric_coordinates(b)
    finite = np.isfinite(c).all(axis=-1)
    nonnegative = (c >= -atol).all(axis=-1)
    closure = np.isclose(np.sum(c, axis=-1), 1.0, atol=atol, rtol=0.0)
    return finite & nonnegative & closure
