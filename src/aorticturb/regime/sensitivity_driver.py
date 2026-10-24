from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..reporting.run_manifest import (
    append_log,
    create_run_directory,
    finalize_run,
    initialize_run,
    new_run_id,
    write_csv,
    write_json,
)
from .support_sensitivity import (
    DEFAULT_SUPPORT_SENSITIVITY_POLICIES,
    LEGACY_GLOBAL_FRACTION,
    evaluate_support_policy_sensitivity,
    project_gate_a_decision,
)


METHODS = """# Gate A support-policy sensitivity analysis

This run compares pre-specified support-readiness policies without selecting a policy from
its turbulence outcome. The historical whole-lattice common-support criterion remains the
primary Gate-A rule and is not overwritten.

The sensitivity comparators are:

1. `legacy_global_fraction`: common intersection nodes divided by all nodes in the deposited
   Cartesian lattice, using the exact threshold and minimum-realization rule recorded by the
   source Gate-A run.
2. `strict_mask_consistency`: exact equality of finite-velocity support across realizations,
   expressed as intersection/union = 1. No empirical near-unity threshold is introduced.
3. `fully_finite_local_windows`: at least one local rectangular analysis window with valid
   support fraction 1.0, matching the support requirement already used by the spatial
   multiscale diagnostics.
4. `strict_mask_and_local_windows`: conjunction of exact mask consistency and local-window
   admissibility.

The comparator set and the new local criterion are locked for this canonical sensitivity
pass. Changing them requires an explicit code/project change rather than a config edit after
observing the result.

For each policy, the already-computed Gate-A multiscale evidence (spatial coherence,
correlation decay, forward transfer, and measurement-noise criterion where available) is
projected through that readiness rule. These projected decisions are counterfactual
sensitivity results only. They do not replace the primary Gate-A decision, do not promote
results to `05_results`, and do not open the KO branch automatically.
"""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _intersection_over_union(data: dict[str, np.ndarray]) -> float | None:
    support = np.asarray(data["support"], dtype=bool)
    support_union = data.get("support_union")
    if support_union is None:
        return None
    union = np.asarray(support_union, dtype=bool)
    if union.shape != support.shape:
        raise ValueError("support_union must have the same shape as support")
    union_nodes = int(union.sum())
    return float(support.sum() / union_nodes) if union_nodes > 0 else None


def _window_support(phase_diag: dict[str, Any]) -> tuple[int, float | None]:
    windows = phase_diag.get("windows", [])
    if not isinstance(windows, list):
        raise ValueError("Gate-A diagnostics field 'windows' must be a list")
    fractions = [
        float(window["valid_fraction"])
        for window in windows
        if isinstance(window, dict) and window.get("valid_fraction") is not None
    ]
    return len(windows), (min(fractions) if fractions else None)


def _laminar_evidence(phase_diag: dict[str, Any], noise_ceiling: float | None) -> bool:
    if noise_ceiling is None:
        return False
    rms_ci = phase_diag.get("residual_rms_ci")
    if not isinstance(rms_ci, list) or len(rms_ci) < 3:
        return False
    upper = float(rms_ci[2])
    transfer = bool(phase_diag.get("forward_transfer_supported", False))
    return bool(np.isfinite(upper) and upper <= noise_ceiling and not transfer)


def _locked_sensitivity_config(config: dict) -> tuple[tuple[str, ...], int, float]:
    scfg = config.get("regime", {}).get("support_sensitivity", {})
    policies = tuple(
        str(value)
        for value in scfg.get(
            "policies",
            list(DEFAULT_SUPPORT_SENSITIVITY_POLICIES),
        )
    )
    min_windows = int(scfg.get("min_fully_finite_windows", 1))
    required_window_fraction = float(
        scfg.get("required_local_window_valid_fraction", 1.0)
    )
    if policies != DEFAULT_SUPPORT_SENSITIVITY_POLICIES:
        raise ValueError(
            "Canonical Pass-2 support sensitivity policies are locked to the pre-specified "
            f"set {DEFAULT_SUPPORT_SENSITIVITY_POLICIES}; received {policies}."
        )
    if min_windows != 1:
        raise ValueError(
            "Canonical Pass-2 local-window criterion is locked to at least one fully "
            "finite analysis window."
        )
    if required_window_fraction != 1.0:
        raise ValueError(
            "Canonical Pass-2 local-window valid fraction is locked to 1.0."
        )
    return policies, min_windows, required_window_fraction


def run_support_sensitivity(
    dataset_root: str | Path,
    decomposition_dir: str | Path,
    regime_dir: str | Path,
    output_root: str | Path,
    config: dict,
    *,
    run_id: str | None = None,
) -> dict:
    """Run support-policy sensitivity without changing the primary Gate-A decision."""
    dataset_root = Path(dataset_root)
    decomposition_dir = Path(decomposition_dir)
    regime_dir = Path(regime_dir)
    base_run_id = run_id or new_run_id()
    if not base_run_id.endswith("_support_sensitivity"):
        base_run_id = f"{base_run_id}_support_sensitivity"
    run_dir = create_run_directory(
        output_root,
        "01_laminar_vs_unsteady",
        run_id=base_run_id,
    )

    diagnostics_path = regime_dir / "diagnostics.json"
    regime_summary_path = regime_dir / "regime_summary.json"
    source_config_path = regime_dir / "config.json"
    decomposition_summary_path = decomposition_dir / "decomposition_summary.json"
    for path in (
        diagnostics_path,
        regime_summary_path,
        source_config_path,
        decomposition_summary_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    initialize_run(
        run_dir,
        branch="01_laminar_vs_unsteady",
        dataset_root=dataset_root,
        config=config,
        methods=METHODS,
        extra_inputs=[
            decomposition_summary_path,
            source_config_path,
            diagnostics_path,
            regime_summary_path,
        ],
    )
    append_log(run_dir, f"source decomposition={decomposition_dir}")
    append_log(run_dir, f"source regime={regime_dir}")

    regime_diagnostics = _load_json(diagnostics_path)
    regime_summary = _load_json(regime_summary_path)
    source_config = _load_json(source_config_path)
    source_rcfg = source_config.get("regime", {})
    min_realizations = int(source_rcfg.get("min_realizations", 12))
    min_global = float(source_rcfg.get("min_common_support_fraction", 0.50))
    noise_raw = source_rcfg.get("measurement_noise_rms_max")
    noise_ceiling = None if noise_raw is None else float(noise_raw)
    policies, min_windows, required_window_fraction = _locked_sensitivity_config(config)

    rows: list[dict[str, Any]] = []
    phase_payload: dict[str, Any] = {}

    for phase in PHASES:
        data = load_decomposition(decomposition_dir, phase)
        support = np.asarray(data["support"], dtype=bool)
        residuals = np.asarray(data["residuals"])
        phase_diag = regime_diagnostics.get("phases", {}).get(phase)
        if not isinstance(phase_diag, dict):
            raise ValueError(f"Missing Gate-A diagnostics for {phase}")

        mask_iou = _intersection_over_union(data)
        window_count, min_window_fraction = _window_support(phase_diag)
        sensitivity = evaluate_support_policy_sensitivity(
            realizations=int(residuals.shape[0]),
            min_realizations=min_realizations,
            common_intersection_fraction_total=float(support.mean()),
            min_common_support_fraction=min_global,
            intersection_over_union_fraction=mask_iou,
            fully_finite_window_count=window_count,
            local_window_min_valid_fraction=min_window_fraction,
            min_fully_finite_windows=min_windows,
            required_local_window_valid_fraction=required_window_fraction,
            policies=policies,
        )

        coherent = bool(phase_diag.get("coherent_spatial_residual", False))
        decays = bool(phase_diag.get("correlation_decay_supported", False))
        transfer = bool(phase_diag.get("forward_transfer_supported", False))
        laminar = _laminar_evidence(phase_diag, noise_ceiling)
        primary = regime_summary.get("phases", {}).get(phase, {})
        primary_decision = primary.get("decision", phase_diag.get("decision", "UNRESOLVED"))
        primary_turbulence = primary.get(
            "turbulence_established",
            phase_diag.get("turbulence_established"),
        )

        projected: dict[str, Any] = {}
        projected_decisions: list[str] = []
        for policy, evaluation in sensitivity.policies.items():
            projection = project_gate_a_decision(
                evaluation,
                coherent_spatial_residual=coherent,
                correlation_decay_supported=decays,
                forward_transfer_supported=transfer,
                laminar_evidence=laminar,
            )
            projected[policy] = projection
            if projection["evaluable"]:
                projected_decisions.append(str(projection["decision"]))
            rows.append(
                {
                    "phase": phase,
                    "policy": policy,
                    "is_primary_policy": policy == LEGACY_GLOBAL_FRACTION,
                    "support_evaluable": evaluation.evaluable,
                    "support_ready": evaluation.ready,
                    "projected_decision": projection["decision"],
                    "projected_turbulence_established": projection[
                        "turbulence_established"
                    ],
                    "primary_decision": primary_decision,
                    "matches_primary_decision": (
                        projection["decision"] == primary_decision
                        if projection["evaluable"]
                        else None
                    ),
                    "common_intersection_fraction_total": float(support.mean()),
                    "intersection_over_union_fraction": mask_iou,
                    "fully_finite_window_count": window_count,
                    "local_window_min_valid_fraction": min_window_fraction,
                    "coherent_spatial_residual": coherent,
                    "correlation_decay_supported": decays,
                    "forward_transfer_supported": transfer,
                }
            )

        projected_sensitive = len(set(projected_decisions)) > 1
        phase_payload[phase] = {
            "primary_gate_a": {
                "policy": LEGACY_GLOBAL_FRACTION,
                "decision": primary_decision,
                "turbulence_established": primary_turbulence,
                "unchanged_by_sensitivity_run": True,
            },
            "source_gate_a_parameters": {
                "min_realizations": min_realizations,
                "min_common_support_fraction": min_global,
                "measurement_noise_rms_max": noise_ceiling,
            },
            "support_sensitivity": sensitivity.to_dict(),
            "projected_gate_a_by_policy": projected,
            "projected_decision_policy_sensitive": projected_sensitive,
            "multiscale_evidence_reused_from_source_regime_run": {
                "coherent_spatial_residual": coherent,
                "correlation_decay_supported": decays,
                "forward_transfer_supported": transfer,
                "laminar_evidence": laminar,
            },
        }
        append_log(
            run_dir,
            f"{phase} readiness_policy_sensitive={sensitivity.readiness_policy_sensitive} "
            f"projected_decision_policy_sensitive={projected_sensitive}",
        )

    write_csv(run_dir / "support_sensitivity.csv", rows)
    write_json(
        run_dir / "support_sensitivity.json",
        {
            "source_decomposition_run": str(decomposition_dir),
            "source_regime_run": str(regime_dir),
            "primary_policy_unchanged": LEGACY_GLOBAL_FRACTION,
            "policies_are_sensitivity_comparators_only": True,
            "canonical_comparator_set_locked": True,
            "phases": phase_payload,
        },
    )
    summary = {
        "run_dir": str(run_dir),
        "source_regime_run": str(regime_dir),
        "primary_policy_unchanged": LEGACY_GLOBAL_FRACTION,
        "phases": {
            phase: {
                "primary_decision": payload["primary_gate_a"]["decision"],
                "readiness_policy_sensitive": payload["support_sensitivity"][
                    "readiness_policy_sensitive"
                ],
                "projected_decision_policy_sensitive": payload[
                    "projected_decision_policy_sensitive"
                ],
            }
            for phase, payload in phase_payload.items()
        },
    }
    write_json(run_dir / "sensitivity_summary.json", summary)
    finalize_run(run_dir)
    return summary
