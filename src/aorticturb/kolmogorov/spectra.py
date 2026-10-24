from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class ScalingFit:
    slope: float
    intercept: float
    r2: float
    start: int
    stop: int
    points: int

    def to_dict(self) -> dict:
        return asdict(self)


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
    """Local shell-integrated 3-D spectrum for a fully finite rectangular window."""
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


def directional_spectrum(
    velocity_grid: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    axis: int,
    taper: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One-dimensional spectrum along one grid axis, averaged over transverse lines."""
    u = np.asarray(velocity_grid, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("Expected velocity_grid shape (nx, ny, nz, 3).")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not np.isfinite(u).all():
        raise ValueError("Directional spectrum window contains non-finite values.")

    n = u.shape[axis]
    window = np.hanning(n) if taper else np.ones(n)
    shape = [1, 1, 1, 1]
    shape[axis] = n
    w = window.reshape(shape)
    u0 = u - np.mean(u, axis=axis, keepdims=True)
    F = np.fft.rfft(u0 * w, axis=axis)
    power = 0.5 * np.sum(np.abs(F) ** 2, axis=-1) / (n**2 * np.mean(window**2))
    transverse = tuple(i for i in range(3) if i != axis)
    E = np.mean(power, axis=transverse)
    k = 2 * np.pi * np.fft.rfftfreq(n, d=spacing[axis])
    return k[1:], E[1:]


def fit_scaling_range(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int = 4,
) -> ScalingFit | None:
    """Select a power-law interval by linearity and length, never by a target exponent."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (np.abs(y) > 0)
    indices = np.flatnonzero(valid)
    if indices.size < min_points:
        return None
    # Restrict to the longest contiguous valid block.
    blocks = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    block = max(blocks, key=len)
    if len(block) < min_points:
        return None

    best: tuple[float, ScalingFit] | None = None
    for a_pos in range(0, len(block) - min_points + 1):
        for b_pos in range(a_pos + min_points, len(block) + 1):
            ids = block[a_pos:b_pos]
            lx = np.log(x[ids])
            ly = np.log(np.abs(y[ids]))
            slope, intercept = np.polyfit(lx, ly, 1)
            pred = slope * lx + intercept
            ss_res = float(np.sum((ly - pred) ** 2))
            ss_tot = float(np.sum((ly - ly.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            points = len(ids)
            score = max(r2, 0.0) * np.sqrt(points)
            fit = ScalingFit(float(slope), float(intercept), float(r2), int(ids[0]), int(ids[-1] + 1), points)
            if best is None or score > best[0] or (np.isclose(score, best[0]) and points > best[1].points):
                best = (float(score), fit)
    return best[1] if best else None
