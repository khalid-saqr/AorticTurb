from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


LEGACY_GLOBAL_FRACTION = "legacy_global_fraction"
STRICT_MASK_CONSISTENCY = "strict_mask_consistency"
FULLY_FINITE_LOCAL_WINDOWS = "fully_finite_local_windows"
STRICT_MASK_AND_LOCAL_WINDOWS = "strict_mask_and_local_windows"

DEFAULT_SUPPORT_SENSITIVITY_POLICIES = (
    LEGACY_GLOBAL_FRACTION,
    STRICT_MASK_CONSISTENCY,
    FULLY_FINITE_LOCAL_WINDOWS,
    STRICT_MASK_AND_LOCAL_WINDOWS,
)


@dataclass(frozen=True)
class SupportPolicyEvaluation:
    """Readiness result for one pre-specified support policy.

    ``ready`` is ``None`` when the required audit quantity was not recorded, so an older
    run cannot be silently interpreted as passing or failing a criterion it never measured.
    These evaluations are sensitivity diagnostics; they do not select the primary Gate-A
    policy or directly classify turbulence.
    """

    policy: str
    evaluable: bool
    ready: bool | None
    reasons: list[str]
    criteria: dict[str, Any]
    observed: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SupportSensitivityResult:
    primary_policy: str
    policies: dict[str, SupportPolicyEvaluation]
    evaluable_policies: list[str]
    readiness_values: list[bool]
    readiness_policy_sensitive: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["policies"] = {
            name: evaluation.to_dict() for name, evaluation in self.policies.items()
        }
        return payload


def _realization_reason(realizations: int, min_realizations: int) -> list[str]:
    if realizations >= min_realizations:
        return []
    return [f"Only {realizations} realizations; require at least {min_realizations}."]


def _legacy_global_fraction(
    *,
    realizations: int,
    min_realizations: int,
    common_intersection_fraction_total: float,
    min_common_support_fraction: float,
) -> SupportPolicyEvaluation:
    reasons = _realization_reason(realizations, min_realizations)
    if common_intersection_fraction_total < min_common_support_fraction:
        reasons.append(
            "Whole-lattice common-support occupancy "
            f"{common_intersection_fraction_total:.3f} is below "
            f"{min_common_support_fraction:.3f}."
        )
    return SupportPolicyEvaluation(
        policy=LEGACY_GLOBAL_FRACTION,
        evaluable=True,
        ready=not reasons,
        reasons=reasons,
        criteria={
            "min_realizations": min_realizations,
            "min_common_support_fraction_total_lattice": min_common_support_fraction,
            "fraction_definition": (
                "common_intersection_nodes / total_cartesian_lattice_nodes"
            ),
        },
        observed={
            "realizations": realizations,
            "common_intersection_fraction_total_lattice": common_intersection_fraction_total,
        },
    )


def _strict_mask_consistency(
    *,
    realizations: int,
    min_realizations: int,
    intersection_over_union_fraction: float | None,
) -> SupportPolicyEvaluation:
    reasons = _realization_reason(realizations, min_realizations)
    if intersection_over_union_fraction is None:
        return SupportPolicyEvaluation(
            policy=STRICT_MASK_CONSISTENCY,
            evaluable=False,
            ready=None,
            reasons=[
                *reasons,
                "Intersection/union mask agreement was not recorded for this decomposition.",
            ],
            criteria={
                "min_realizations": min_realizations,
                "required_intersection_over_union_fraction": 1.0,
                "interpretation": "all realization finite-velocity masks are identical",
            },
            observed={
                "realizations": realizations,
                "intersection_over_union_fraction": None,
            },
        )
    if intersection_over_union_fraction != 1.0:
        reasons.append(
            "Realization masks are not exactly consistent: intersection/union="
            f"{intersection_over_union_fraction:.6f}, required 1.000000."
        )
    return SupportPolicyEvaluation(
        policy=STRICT_MASK_CONSISTENCY,
        evaluable=True,
        ready=not reasons,
        reasons=reasons,
        criteria={
            "min_realizations": min_realizations,
            "required_intersection_over_union_fraction": 1.0,
            "interpretation": "all realization finite-velocity masks are identical",
        },
        observed={
            "realizations": realizations,
            "intersection_over_union_fraction": intersection_over_union_fraction,
        },
    )


def _fully_finite_local_windows(
    *,
    realizations: int,
    min_realizations: int,
    fully_finite_window_count: int,
    local_window_min_valid_fraction: float | None,
    min_fully_finite_windows: int,
    required_local_window_valid_fraction: float,
) -> SupportPolicyEvaluation:
    reasons = _realization_reason(realizations, min_realizations)
    if fully_finite_window_count < min_fully_finite_windows:
        reasons.append(
            f"Only {fully_finite_window_count} fully finite local windows; require at least "
            f"{min_fully_finite_windows}."
        )
    if fully_finite_window_count > 0:
        if local_window_min_valid_fraction is None:
            reasons.append("Local-window support fraction was not recorded.")
        elif local_window_min_valid_fraction < required_local_window_valid_fraction:
            reasons.append(
                "Minimum local-window valid fraction "
                f"{local_window_min_valid_fraction:.6f} is below required "
                f"{required_local_window_valid_fraction:.6f}."
            )
    return SupportPolicyEvaluation(
        policy=FULLY_FINITE_LOCAL_WINDOWS,
        evaluable=True,
        ready=not reasons,
        reasons=reasons,
        criteria={
            "min_realizations": min_realizations,
            "min_fully_finite_windows": min_fully_finite_windows,
            "required_local_window_valid_fraction": required_local_window_valid_fraction,
            "interpretation": (
                "support is sufficient on the actual local rectangular domains used by "
                "the spatial multiscale diagnostics"
            ),
        },
        observed={
            "realizations": realizations,
            "fully_finite_window_count": fully_finite_window_count,
            "local_window_min_valid_fraction": local_window_min_valid_fraction,
        },
    )


def evaluate_support_policy_sensitivity(
    *,
    realizations: int,
    min_realizations: int,
    common_intersection_fraction_total: float,
    min_common_support_fraction: float,
    intersection_over_union_fraction: float | None,
    fully_finite_window_count: int,
    local_window_min_valid_fraction: float | None,
    min_fully_finite_windows: int = 1,
    required_local_window_valid_fraction: float = 1.0,
    policies: tuple[str, ...] = DEFAULT_SUPPORT_SENSITIVITY_POLICIES,
) -> SupportSensitivityResult:
    """Compare pre-specified support-readiness policies without selecting a winner.

    The policies are defined from support semantics, not from the observed turbulence
    outcome. The historical whole-lattice policy remains the primary policy. Alternative
    policies are sensitivity comparators only.
    """
    if min_fully_finite_windows < 1:
        raise ValueError("min_fully_finite_windows must be at least 1")
    if not 0.0 <= required_local_window_valid_fraction <= 1.0:
        raise ValueError("required_local_window_valid_fraction must be in [0, 1]")

    local = _fully_finite_local_windows(
        realizations=realizations,
        min_realizations=min_realizations,
        fully_finite_window_count=fully_finite_window_count,
        local_window_min_valid_fraction=local_window_min_valid_fraction,
        min_fully_finite_windows=min_fully_finite_windows,
        required_local_window_valid_fraction=required_local_window_valid_fraction,
    )
    mask = _strict_mask_consistency(
        realizations=realizations,
        min_realizations=min_realizations,
        intersection_over_union_fraction=intersection_over_union_fraction,
    )
    available: dict[str, SupportPolicyEvaluation] = {
        LEGACY_GLOBAL_FRACTION: _legacy_global_fraction(
            realizations=realizations,
            min_realizations=min_realizations,
            common_intersection_fraction_total=common_intersection_fraction_total,
            min_common_support_fraction=min_common_support_fraction,
        ),
        STRICT_MASK_CONSISTENCY: mask,
        FULLY_FINITE_LOCAL_WINDOWS: local,
    }

    if not mask.evaluable:
        combined = SupportPolicyEvaluation(
            policy=STRICT_MASK_AND_LOCAL_WINDOWS,
            evaluable=False,
            ready=None,
            reasons=[
                "Combined policy is not evaluable because strict mask consistency is not evaluable.",
                *mask.reasons,
            ],
            criteria={
                "requires": [STRICT_MASK_CONSISTENCY, FULLY_FINITE_LOCAL_WINDOWS],
            },
            observed={
                "strict_mask_consistency_ready": None,
                "fully_finite_local_windows_ready": local.ready,
            },
        )
    else:
        assert mask.ready is not None and local.ready is not None
        combined_reasons = [] if (mask.ready and local.ready) else [
            *(mask.reasons if not mask.ready else []),
            *(local.reasons if not local.ready else []),
        ]
        combined = SupportPolicyEvaluation(
            policy=STRICT_MASK_AND_LOCAL_WINDOWS,
            evaluable=True,
            ready=bool(mask.ready and local.ready),
            reasons=combined_reasons,
            criteria={
                "requires": [STRICT_MASK_CONSISTENCY, FULLY_FINITE_LOCAL_WINDOWS],
            },
            observed={
                "strict_mask_consistency_ready": mask.ready,
                "fully_finite_local_windows_ready": local.ready,
            },
        )
    available[STRICT_MASK_AND_LOCAL_WINDOWS] = combined

    unknown = [policy for policy in policies if policy not in available]
    if unknown:
        raise ValueError(f"Unknown support sensitivity policies: {unknown}")
    selected = {policy: available[policy] for policy in policies}
    evaluable = [name for name, value in selected.items() if value.evaluable]
    readiness_values = [
        bool(selected[name].ready) for name in evaluable if selected[name].ready is not None
    ]
    return SupportSensitivityResult(
        primary_policy=LEGACY_GLOBAL_FRACTION,
        policies=selected,
        evaluable_policies=evaluable,
        readiness_values=readiness_values,
        readiness_policy_sensitive=len(set(readiness_values)) > 1,
    )


def project_gate_a_decision(
    readiness: SupportPolicyEvaluation,
    *,
    coherent_spatial_residual: bool,
    correlation_decay_supported: bool,
    forward_transfer_supported: bool,
    laminar_evidence: bool,
) -> dict[str, Any]:
    """Project the same downstream evidence through one readiness comparator.

    This is a counterfactual sensitivity result, not a primary classification. The primary
    Gate-A result remains governed by the historical policy until an explicit project-level
    decision changes it.
    """
    if not readiness.evaluable or readiness.ready is None:
        return {
            "evaluable": False,
            "decision": "UNRESOLVED",
            "turbulence_established": None,
            "reason": "Support policy could not be evaluated from the recorded quantities.",
        }
    if (
        readiness.ready
        and coherent_spatial_residual
        and correlation_decay_supported
        and forward_transfer_supported
    ):
        return {
            "evaluable": True,
            "decision": "STOCHASTIC_MULTISCALE_SUPPORTED",
            "turbulence_established": True,
            "reason": "Support readiness and all Gate-A multiscale evidence tests are satisfied.",
        }
    if laminar_evidence:
        return {
            "evaluable": True,
            "decision": "LAMINAR_UNSTEADY_SUPPORTED",
            "turbulence_established": False,
            "reason": "Residual RMS is bounded by the configured measurement-noise ceiling and forward transfer is not supported.",
        }
    return {
        "evaluable": True,
        "decision": "UNRESOLVED",
        "turbulence_established": None,
        "reason": (
            "At least one support-readiness or multiscale-evidence requirement is not satisfied."
        ),
    }
