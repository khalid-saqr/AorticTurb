from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class TaylorAssessment:
    median_fluctuation_ratio: float | None
    fraction_below_threshold: float | None
    temporal_test_available: bool
    conclusion: str

    def to_dict(self) -> dict:
        return asdict(self)


def fluctuation_ratio(mean_velocity: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    mean_speed = np.linalg.norm(mean_velocity, axis=-1)
    u_rms = np.sqrt(np.nanmean(np.sum(residuals**2, axis=-1), axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = u_rms / mean_speed
    chi[~np.isfinite(chi)] = np.nan
    return chi


def assess(
    mean_velocity: np.ndarray,
    residuals: np.ndarray,
    *,
    threshold: float = 0.20,
) -> TaylorAssessment:
    chi = fluctuation_ratio(mean_velocity, residuals)
    finite = np.isfinite(chi)
    median = float(np.nanmedian(chi)) if finite.any() else None
    fraction = float(np.mean(chi[finite] < threshold)) if finite.any() else None
    return TaylorAssessment(
        median_fluctuation_ratio=median,
        fraction_below_threshold=fraction,
        temporal_test_available=False,
        conclusion=(
            "NECESSARY_SPATIAL_ADMISSIBILITY_ONLY: the acquired subset contains "
            "independent phase-locked realizations, not a continuous turbulent time series."
        ),
    )
