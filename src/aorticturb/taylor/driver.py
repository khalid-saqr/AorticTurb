from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..uncertainty.bootstrap import bootstrap_mean_ci
from ..reporting.run_manifest import create_run_directory, initialize_run, finalize_run, write_json, write_csv, append_log
from .admissibility import assess, fluctuation_ratio


METHODS = """# Taylor-hypothesis admissibility

Taylor's frozen-turbulence hypothesis is assessed only after Gate A. The acquired data
are independent phase-locked realizations and do not form continuous turbulent
trajectories, so a true space-time correlation ridge cannot be measured and no
frequency-to-wavenumber conversion is permitted. This stage reports only necessary
spatial/mean-flow conditions, including the fluctuation-to-mean ratio chi. The configured
chi warning value is an operational diagnostic, not proof of Taylor validity.
"""


def run_taylor(
    dataset_root: str | Path,
    decomposition_dir: str | Path,
    regime_dir: str | Path,
    output_root: str | Path,
    config: dict,
    *,
    run_id: str | None = None,
) -> dict:
    dataset_root = Path(dataset_root)
    decomposition_dir = Path(decomposition_dir)
    regime_dir = Path(regime_dir)
    regime = json.loads((regime_dir / "regime_summary.json").read_text(encoding="utf-8"))
    run_dir = create_run_directory(output_root, "03_taylor_hypothesis", run_id=run_id)
    initialize_run(run_dir, branch="03_taylor_hypothesis", dataset_root=dataset_root, config=config, methods=METHODS, extra_inputs=[decomposition_dir / "decomposition_summary.json", regime_dir / "regime_summary.json"])

    analysis = config.get("analysis", {})
    tcfg = config.get("taylor", {})
    threshold = float(tcfg.get("fluctuation_ratio_warn", 0.20))
    samples = int(analysis.get("bootstrap_samples", 1000))
    seed = int(analysis.get("random_seed", 20261024))
    rows: list[dict] = []
    uncertainty: list[dict] = []
    diagnostics: dict = {"temporal_test_available": False, "phases": {}}
    summary_phases: dict = {}

    for pidx, phase in enumerate(PHASES):
        if regime["phases"][phase]["turbulence_established"] is not True:
            summary_phases[phase] = {"status": "SKIPPED_GATE_A", "temporal_test_available": False}
            rows.append({"phase": phase, "status": "SKIPPED_GATE_A", "temporal_test_available": False})
            continue
        data = load_decomposition(decomposition_dir, phase)
        result = assess(data["mean"], data["residuals"], threshold=threshold)
        chi = fluctuation_ratio(data["mean"], data["residuals"])
        support = data["support"].astype(bool)
        mean_speed = np.linalg.norm(data["mean"], axis=-1)
        finite = support & np.isfinite(chi) & np.isfinite(mean_speed)

        per_realization = []
        for residual in data["residuals"]:
            mag = np.linalg.norm(residual, axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = mag / mean_speed
            vals = ratio[finite & np.isfinite(ratio)]
            per_realization.append(float(np.median(vals)) if vals.size else np.nan)
        chi_ci = bootstrap_mean_ci(np.asarray(per_realization), samples=samples, seed=seed + pidx)
        direction = data["mean"][finite]
        speed = np.linalg.norm(direction, axis=1)
        unit = direction / speed[:, None]
        preferred = np.nanmean(unit, axis=0)
        pref_norm = np.linalg.norm(preferred)
        alignment = float(pref_norm) if np.isfinite(pref_norm) else np.nan
        speed_cv = float(np.std(speed) / np.mean(speed)) if speed.size and np.mean(speed) != 0 else np.nan

        status = "NECESSARY_CONDITIONS_ASSESSED_TEMPORAL_TEST_UNAVAILABLE"
        rows.append({
            "phase": phase,
            "status": status,
            "median_chi": result.median_fluctuation_ratio,
            "fraction_chi_below_warning": result.fraction_below_threshold,
            "chi_warning_value": threshold,
            "mean_direction_alignment": alignment,
            "mean_speed_cv": speed_cv,
            "temporal_test_available": False,
        })
        uncertainty.append({"phase": phase, "quantity": "realization_median_chi", "lower": chi_ci[0], "median": chi_ci[1], "upper": chi_ci[2], "bootstrap_axis": "realizations"})
        diagnostics["phases"][phase] = {
            **result.to_dict(),
            "chi_bootstrap_ci": list(map(float, chi_ci)),
            "mean_direction_alignment": alignment,
            "mean_speed_cv": speed_cv,
            "prohibited_inference": "No turbulent frequency spectrum may be converted to wavenumber using Taylor from this dataset.",
        }
        summary_phases[phase] = {"status": status, "temporal_test_available": False, "necessary_spatial_assessment": result.to_dict()}
        append_log(run_dir, f"{phase} necessary conditions assessed; temporal test unavailable")

    write_csv(run_dir / "metrics.csv", rows)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_json(run_dir / "diagnostics.json", diagnostics)
    summary = {"run_dir": str(run_dir), "phases": summary_phases, "temporal_test_available": False}
    write_json(run_dir / "taylor_summary.json", summary)
    finalize_run(run_dir)
    return summary
