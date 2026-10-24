from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from ..grid import StructuredGrid, WindowSpec
from ..uncertainty.bootstrap import bootstrap_mean_ci


@dataclass(frozen=True)
class HomogeneityAssessment:
    not_rejected: bool | None
    octant_deviation_ci: list[tuple[float, float, float]]
    windows: int
    realizations: int

    def to_dict(self) -> dict:
        return asdict(self)


def _octant_energy_deviation(u: np.ndarray) -> np.ndarray:
    """Relative residual-energy deviations among eight spatial octants."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3 or not np.isfinite(u).all():
        raise ValueError("Expected a fully finite velocity window shaped (nx,ny,nz,3)")
    mids = tuple(n // 2 for n in u.shape[:3])
    if min(mids) < 2:
        raise ValueError("Window is too small for an octant homogeneity assessment")
    energies = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                xs = slice(0, mids[0]) if ix == 0 else slice(mids[0], None)
                ys = slice(0, mids[1]) if iy == 0 else slice(mids[1], None)
                zs = slice(0, mids[2]) if iz == 0 else slice(mids[2], None)
                block = u[xs, ys, zs]
                energies.append(0.5 * float(np.mean(np.sum(block**2, axis=-1))))
    values = np.asarray(energies, dtype=float)
    mean = float(np.mean(values))
    if not np.isfinite(mean) or mean <= 0:
        return np.full(8, np.nan)
    return (values - mean) / mean


def assess_local_homogeneity(
    residuals: np.ndarray,
    grid: StructuredGrid,
    windows: list[WindowSpec],
    *,
    samples: int = 1000,
    seed: int = 20261024,
) -> HomogeneityAssessment:
    """Test whether local energy inhomogeneity is statistically detected.

    The test has no arbitrary amplitude tolerance. For each realization, relative
    residual-energy deviations are computed in eight octants of every admissible local
    window and averaged across windows. Realization bootstrap CIs are then formed. Local
    homogeneity is *not rejected* only when all eight CIs contain zero. Two or more
    statistically non-zero octants reject local homogeneity; a single such octant is
    treated as unresolved rather than decisive.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 3 or residuals.shape[-1] != 3:
        raise ValueError("residuals must have shape (N_realizations,N_nodes,3)")
    if not windows:
        return HomogeneityAssessment(None, [], 0, residuals.shape[0])
    deviations = np.full((residuals.shape[0], 8), np.nan)
    for ridx in range(residuals.shape[0]):
        local = []
        for window in windows:
            local.append(_octant_energy_deviation(grid.extract_window(residuals[ridx], window)))
        deviations[ridx] = np.nanmean(np.asarray(local), axis=0)
    cis = [bootstrap_mean_ci(deviations[:, i], samples=samples, seed=seed + i) for i in range(8)]
    finite = [ci for ci in cis if np.isfinite(ci[0]) and np.isfinite(ci[2])]
    if len(finite) != 8:
        decision: bool | None = None
    else:
        exclusions = sum(not (ci[0] <= 0.0 <= ci[2]) for ci in cis)
        if exclusions == 0:
            decision = True
        elif exclusions >= 2:
            decision = False
        else:
            decision = None
    return HomogeneityAssessment(decision, cis, len(windows), residuals.shape[0])
