from __future__ import annotations

import numpy as np


def hann3d(shape: tuple[int, int, int]) -> np.ndarray:
    wx = np.hanning(shape[0])
    wy = np.hanning(shape[1])
    wz = np.hanning(shape[2])
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def isotropic_spectrum(
    velocity_grid: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    bins: int = 40,
    taper: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local isotropic 3-D spectrum for a fully finite rectangular window.

    This routine intentionally rejects NaNs and should only be called on vetted local
    windows. It is not a whole-domain spectrum driver.
    """
    u = np.asarray(velocity_grid, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("Expected velocity_grid shape (nx, ny, nz, 3).")
    if not np.isfinite(u).all():
        raise ValueError("Local spectrum window contains non-finite values.")

    shape = u.shape[:3]
    w = hann3d(shape) if taper else np.ones(shape)
    u0 = u - np.mean(u, axis=(0, 1, 2), keepdims=True)
    norm = np.mean(w**2)

    energy = np.zeros(shape, dtype=float)
    for comp in range(3):
        F = np.fft.fftn(u0[..., comp] * w)
        energy += 0.5 * np.abs(F) ** 2 / (np.prod(shape) ** 2 * norm)

    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=spacing[0])
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=spacing[1])
    kz = 2 * np.pi * np.fft.fftfreq(shape[2], d=spacing[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    positive = kmag > 0
    edges = np.linspace(0.0, float(kmag.max()), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    spectrum = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)

    for i in range(bins):
        mask = positive & (kmag >= edges[i]) & (kmag < edges[i + 1])
        counts[i] = int(mask.sum())
        if counts[i]:
            spectrum[i] = float(energy[mask].sum())

    return centers, spectrum, counts
