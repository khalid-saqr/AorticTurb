from __future__ import annotations

from pathlib import Path
import csv
import json

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..grid import infer_structured_grid, WindowSpec
from ..reporting.run_manifest import finalize_run, write_csv, write_json, append_log
from ._core_driver import run_kolmogorov as _run_core
from .admissibility import assess_local_homogeneity


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_kolmogorov(
    dataset_root: str | Path,
    decomposition_dir: str | Path,
    regime_dir: str | Path,
    output_root: str | Path,
    config: dict,
    *,
    run_id: str | None = None,
) -> dict:
    """Run the KO stage and enforce local-homogeneity admissibility for K41 laws."""
    result = _run_core(dataset_root, decomposition_dir, regime_dir, output_root, config, run_id=run_id)
    run_dir = Path(result["run_dir"])
    diagnostics_path = run_dir / "diagnostics.json"
    summary_path = run_dir / "kolmogorov_summary.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = _read_csv(run_dir / "metrics.csv")
    uncertainty = _read_csv(run_dir / "uncertainty.csv")

    analysis = config.get("analysis", {})
    samples = int(analysis.get("bootstrap_samples", 1000))
    seed = int(analysis.get("random_seed", 20261024))

    for pidx, phase in enumerate(PHASES):
        phase_summary = summary["phases"].get(phase, {})
        phase_diag = diagnostics.get("phases", {}).get(phase)
        if not phase_diag or phase_summary.get("status") == "SKIPPED_GATE_A":
            continue
        data = load_decomposition(decomposition_dir, phase)
        grid = infer_structured_grid(data["coordinates"])
        windows = [
            WindowSpec(tuple(w["start"]), tuple(w["stop"]), float(w["valid_fraction"]))
            for w in phase_diag.get("windows", [])
        ]
        assessment = assess_local_homogeneity(
            data["residuals"],
            grid,
            windows,
            samples=samples,
            seed=seed + 60000 + pidx * 100,
        )
        phase_diag["local_homogeneity"] = assessment.to_dict()
        phase_diag["local_homogeneity_not_rejected"] = assessment.not_rejected
        for octant, ci in enumerate(assessment.octant_deviation_ci):
            uncertainty.append({
                "phase": phase,
                "direction": "all",
                "quantity": f"homogeneity_octant_{octant + 1}_relative_energy_deviation",
                "lower": ci[0],
                "median": ci[1],
                "upper": ci[2],
                "theory_target": 0.0,
                "bootstrap_axis": "realizations",
            })

        # The four-fifths law and an overall K41 classification require local homogeneity
        # in addition to the directional-isotropy admissibility already assessed by core.
        if assessment.not_rejected is not True:
            phase_diag.setdefault("core_diagnostics", {})["S3_four_fifths"] = "UNRESOLVED"
            phase_diag["ko_resolved"] = False
            phase_diag["ko_supported"] = None
            phase_summary["core_diagnostics"] = phase_diag["core_diagnostics"]
            phase_summary["local_homogeneity_not_rejected"] = assessment.not_rejected
            phase_summary["ko_resolved"] = False
            phase_summary["ko_supported"] = None
            phase_summary["status"] = "KO_UNRESOLVED"
        else:
            phase_summary["local_homogeneity_not_rejected"] = True

        for row in metrics:
            if row.get("phase") == phase:
                row["local_homogeneity_not_rejected"] = assessment.not_rejected
                row["status"] = phase_summary.get("status")
                row["ko_resolved"] = phase_summary.get("ko_resolved")
                row["ko_supported"] = phase_summary.get("ko_supported")
                row["status_S3_four_fifths"] = phase_diag.get("core_diagnostics", {}).get("S3_four_fifths")
        append_log(run_dir, f"{phase} local_homogeneity_not_rejected={assessment.not_rejected}")

    methods_path = run_dir / "methods.md"
    with methods_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Local-homogeneity admissibility\n\n"
            "Before the four-fifths law or an overall KO classification can be accepted, "
            "the residual energy in eight spatial octants of the selected local windows is "
            "compared across the 16 realizations. Homogeneity is not rejected only when all "
            "bootstrap intervals for relative octant-energy deviation contain zero; no "
            "amplitude tolerance is introduced.\n"
        )
    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_json(diagnostics_path, diagnostics)
    write_json(summary_path, summary)
    result = {"run_dir": str(run_dir), "phases": summary["phases"]}
    finalize_run(run_dir, status="COMPLETE")
    return result
