from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class GateAEvidence:
    realizations: int
    support_fraction: float
    fluctuation_ratio_median: float | None
    decision: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def fluctuation_ratio(mean_velocity: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    mean_speed = np.linalg.norm(mean_velocity, axis=-1)
    rms = np.sqrt(np.nanmean(np.sum(residuals**2, axis=-1), axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rms / mean_speed
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def evaluate_gate_a(
    mean_velocity: np.ndarray,
    residuals: np.ndarray,
    *,
    support_fraction_value: float,
    min_realizations: int = 12,
    min_support_fraction: float = 0.50,
) -> GateAEvidence:
    """Conservative readiness gate, not a turbulence detector.

    Gate A cannot establish turbulence from a scalar threshold. This function only
    decides whether the dataset is sufficiently supported to proceed to multiscale
    diagnostics. Scientific classification occurs later.
    """
    n = residuals.shape[0]
    reasons: list[str] = []

    if n < min_realizations:
        reasons.append(f"Only {n} realizations; require at least {min_realizations}.")
    if support_fraction_value < min_support_fraction:
        reasons.append(
            f"Common support fraction {support_fraction_value:.3f} "
            f"is below {min_support_fraction:.3f}."
        )

    ratio = fluctuation_ratio(mean_velocity, residuals)
    median = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else None

    decision = "PROCEED_TO_MULTISCALE_DIAGNOSTICS" if not reasons else "UNRESOLVED"
    return GateAEvidence(
        realizations=n,
        support_fraction=support_fraction_value,
        fluctuation_ratio_median=median,
        decision=decision,
        reasons=reasons,
    )
