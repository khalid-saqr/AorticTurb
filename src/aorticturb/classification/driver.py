from __future__ import annotations

from pathlib import Path
import json

from ..constants import PHASES
from ..reporting.run_manifest import create_run_directory, initialize_run, finalize_run, write_json, write_csv
from .decision import classify


METHODS = """# Integrated classification

Classification preserves the locked causal order. Gate A decides whether turbulence is
established; KO diagnostics are consulted only for phases with established turbulence.
Taylor admissibility, Lumley anisotropy, and canonical diagnostics are characterization
axes and cannot manufacture an upstream turbulence decision. Unresolved evidence stays
unresolved. The four allowed primary labels are exactly those in AGENTS.md.
"""


def run_classification(
    dataset_root: str | Path,
    regime_dir: str | Path,
    kolmogorov_dir: str | Path,
    output_root: str | Path,
    config: dict,
    *,
    taylor_dir: str | Path | None = None,
    anisotropy_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict:
    dataset_root = Path(dataset_root)
    regime_dir = Path(regime_dir)
    kolmogorov_dir = Path(kolmogorov_dir)
    regime = json.loads((regime_dir / "regime_summary.json").read_text(encoding="utf-8"))
    ko = json.loads((kolmogorov_dir / "kolmogorov_summary.json").read_text(encoding="utf-8"))
    run_dir = create_run_directory(output_root, "05_results", run_id=run_id)
    extras = [regime_dir / "regime_summary.json", kolmogorov_dir / "kolmogorov_summary.json"]
    for optional, filename in ((taylor_dir, "taylor_summary.json"), (anisotropy_dir, "anisotropy_summary.json"), (diagnostics_dir, "diagnostics_summary.json")):
        if optional is not None:
            extras.append(Path(optional) / filename)
    initialize_run(run_dir, branch="05_results", dataset_root=dataset_root, config=config, methods=METHODS, extra_inputs=extras)

    rows = []
    evidence_rows = []
    phase_results = {}
    trace_lines = ["# AorticTurb decision trace", ""]
    for phase in PHASES:
        turbulence = regime["phases"][phase].get("turbulence_established")
        kophase = ko["phases"].get(phase, {})
        decision = classify(
            turbulence_established=turbulence,
            ko_resolved=kophase.get("ko_resolved"),
            ko_supported=kophase.get("ko_supported"),
            adequate_power=True,
        )
        phase_results[phase] = decision.to_dict()
        rows.append({"phase": phase, "classification": decision.label, "rationale": " | ".join(decision.rationale)})
        evidence_rows.append({
            "phase": phase,
            "gate_a_decision": regime["phases"][phase].get("decision"),
            "turbulence_established": turbulence,
            "ko_resolved": kophase.get("ko_resolved"),
            "ko_supported": kophase.get("ko_supported"),
            "ko_core": json.dumps(kophase.get("core_diagnostics", {}), sort_keys=True),
            "classification": decision.label,
        })
        trace_lines.extend([
            f"## {phase}",
            f"- Gate A: `{regime['phases'][phase].get('decision')}`",
            f"- Turbulence established: `{turbulence}`",
            f"- KO resolved: `{kophase.get('ko_resolved')}`",
            f"- KO supported: `{kophase.get('ko_supported')}`",
            f"- Primary classification: **{decision.label}**",
            f"- Rationale: {' '.join(decision.rationale)}",
            "",
        ])

    unique = sorted(set(item["label"] for item in phase_results.values()))
    synthesis = {
        "phase_classifications": phase_results,
        "uniform_across_phases": len(unique) == 1,
        "unique_phase_labels": unique,
        "note": "The study reports phase-specific primary classifications; no unapproved fifth overall label is invented when phases differ.",
    }
    write_csv(run_dir / "metrics.csv", rows)
    write_csv(run_dir / "evidence_matrix.csv", evidence_rows)
    write_csv(run_dir / "uncertainty.csv", [])
    write_json(run_dir / "classification.json", synthesis)
    (run_dir / "decision_trace.md").write_text("\n".join(trace_lines), encoding="utf-8")
    summary = {"run_dir": str(run_dir), **synthesis}
    write_json(run_dir / "classification_summary.json", summary)
    finalize_run(run_dir)
    return summary
