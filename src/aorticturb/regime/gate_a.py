from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .support_sensitivity import STRICT_MASK_AND_LOCAL_WINDOWS


PRIMARY_SUPPORT_GATE_POLICY = STRICT_MASK_AND_LOCAL_WINDOWS
LEGACY_SUPPORT_FRACTION_DEFINITION = (
    "common_intersection_nodes / total_cartesian_lattice_nodes"
)


@dataclass(frozen=True)
class GateAEvidence:
    realizations: int
    support_fraction: float
    support_fraction_definition: str
    support_gate_policy: str
    intersection_over_union_fraction: float | None
    fully_finite_window_count: int
    local_window_min_valid_fraction: float | None
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
    intersection_over_union_fraction: float | None,
    fully_finite_window_count: int,
    local_window_min_valid_fraction: float | None,
    min_realizations: int = 12,
    min_fully_finite_windows: int = 1,
    required_local_window_valid_fraction: float = 1.0,
    support_gate_policy: str = PRIMARY_SUPPORT_GATE_POLICY,
    support_fraction_definition: str = LEGACY_SUPPORT_FRACTION_DEFINITION,
) -> GateAEvidence:
    """Gate-A readiness check based on measurement support actually used by the diagnostics.

    The historical whole-lattice occupancy is retained as an audit quantity only. It is not
    a readiness criterion because it changes with the rectangular storage domain even when
    the measured velocity support is unchanged. Primary readiness instead requires:

    1. the minimum number of independent realizations;
    2. exact realization-mask consistency (intersection/union = 1); and
    3. at least one fully finite local rectangular window used by the multiscale diagnostics.

    This function remains a readiness gate, not a turbulence detector.
    """
    if support_gate_policy != PRIMARY_SUPPORT_GATE_POLICY:
        raise ValueError(
            "Canonical Gate-A support policy is strict_mask_and_local_windows."
        )
    if min_fully_finite_windows < 1:
        raise ValueError("min_fully_finite_windows must be at least 1")
    if required_local_window_valid_fraction != 1.0:
        raise ValueError(
            "Canonical Gate-A local-window valid fraction is fixed at 1.0."
        )

    n = residuals.shape[0]
    reasons: list[str] = []

    if n < min_realizations:
        reasons.append(f"Only {n} realizations; require at least {min_realizations}.")

    if intersection_over_union_fraction is None:
        reasons.append(
            "Realization-mask agreement is unavailable; exact mask consistency is required."
        )
    elif intersection_over_union_fraction != 1.0:
        reasons.append(
            "Realization masks are not exactly consistent: intersection/union="
            f"{intersection_over_union_fraction:.6f}, required 1.000000."
        )

    if fully_finite_window_count < min_fully_finite_windows:
        reasons.append(
            f"Only {fully_finite_window_count} fully finite local windows; require at least "
            f"{min_fully_finite_windows}."
        )
    elif local_window_min_valid_fraction is None:
        reasons.append("Local-window support fraction was not recorded.")
    elif local_window_min_valid_fraction < required_local_window_valid_fraction:
        reasons.append(
            "Minimum local-window valid fraction "
            f"{local_window_min_valid_fraction:.6f} is below required "
            f"{required_local_window_valid_fraction:.6f}."
        )

    ratio = fluctuation_ratio(mean_velocity, residuals)
    median = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else None

    decision = "PROCEED_TO_MULTISCALE_DIAGNOSTICS" if not reasons else "UNRESOLVED"
    return GateAEvidence(
        realizations=n,
        support_fraction=support_fraction_value,
        support_fraction_definition=support_fraction_definition,
        support_gate_policy=support_gate_policy,
        intersection_over_union_fraction=intersection_over_union_fraction,
        fully_finite_window_count=fully_finite_window_count,
        local_window_min_valid_fraction=local_window_min_valid_fraction,
        fluctuation_ratio_median=median,
        decision=decision,
        reasons=reasons,
    )
