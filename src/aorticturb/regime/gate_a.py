from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


LEGACY_SUPPORT_GATE_POLICY = "legacy_global_fraction"
LEGACY_SUPPORT_FRACTION_DEFINITION = (
    "common_intersection_nodes / total_cartesian_lattice_nodes"
)


@dataclass(frozen=True)
class GateAEvidence:
    realizations: int
    support_fraction: float
    support_fraction_definition: str
    support_gate_policy: str
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
    support_gate_policy: str = LEGACY_SUPPORT_GATE_POLICY,
    support_fraction_definition: str = LEGACY_SUPPORT_FRACTION_DEFINITION,
) -> GateAEvidence:
    """Conservative readiness gate, not a turbulence detector.

    Pass 1 deliberately preserves the historical Gate-A readiness decision. The support
    fraction used by that rule is now named explicitly so it cannot be confused with
    realization-to-realization mask agreement or local-window validity. Alternative
    support policies are not enabled here; they require the planned sensitivity analysis.
    """
    if support_gate_policy != LEGACY_SUPPORT_GATE_POLICY:
        raise ValueError(
            "Only the legacy_global_fraction Gate-A support policy is available in "
            "this pass; alternative policies require an explicit sensitivity analysis."
        )

    n = residuals.shape[0]
    reasons: list[str] = []

    if n < min_realizations:
        reasons.append(f"Only {n} realizations; require at least {min_realizations}.")
    if support_fraction_value < min_support_fraction:
        reasons.append(
            "Legacy global common-support fraction "
            f"{support_fraction_value:.3f} is below {min_support_fraction:.3f}; "
            f"definition: {support_fraction_definition}."
        )

    ratio = fluctuation_ratio(mean_velocity, residuals)
    median = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else None

    decision = "PROCEED_TO_MULTISCALE_DIAGNOSTICS" if not reasons else "UNRESOLVED"
    return GateAEvidence(
        realizations=n,
        support_fraction=support_fraction_value,
        support_fraction_definition=support_fraction_definition,
        support_gate_policy=support_gate_policy,
        fluctuation_ratio_median=median,
        decision=decision,
        reasons=reasons,
    )
