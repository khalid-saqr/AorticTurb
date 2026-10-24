from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..grid import infer_structured_grid, adaptive_windows
from ..kolmogorov.structure_functions import directional_structure_functions
from ..uncertainty.bootstrap import bootstrap_mean_ci
from ..reporting.run_manifest import create_run_directory, initialize_run, finalize_run, write_json, write_csv, append_log
from .gradients import velocity_gradient, strain_rate, vorticity, enstrophy, resolved_dissipation_surrogate
from .intermittency import flatness_from_moments


METHODS = """# Canonical turbulence diagnostics

Derivative-based diagnostics are evaluated only on local fully finite rectangular windows.
The code reports vorticity, strain, enstrophy, divergence quality, and the resolved
quantity `2 S'_ij S'_ij`. This is a dissipation surrogate unless a kinematic viscosity is
explicitly supplied; even then it is labeled resolved dissipation because the experiment
may not resolve all dissipative scales. Scale-dependent longitudinal-increment flatness is
reported with Gaussian reference value 3. Bootstrap uncertainty is across realizations.
"""


def run_diagnostics(
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
    run_dir = create_run_directory(output_root, "05_canonical_diagnostics", run_id=run_id)
    initialize_run(run_dir, branch="05_canonical_diagnostics", dataset_root=dataset_root, config=config, methods=METHODS, extra_inputs=[decomposition_dir / "decomposition_summary.json", regime_dir / "regime_summary.json"])

    analysis = config.get("analysis", {})
    kcfg = config.get("kolmogorov", {})
    dcfg = config.get("diagnostics", {})
    samples = int(analysis.get("bootstrap_samples", 1000))
    seed = int(analysis.get("random_seed", 20261024))
    preferred = tuple(int(x) for x in kcfg.get("window_shape", [32, 32, 32]))
    fallback = tuple(int(x) for x in kcfg.get("fallback_window_sizes", [24, 16, 12, 8]))
    max_windows = int(kcfg.get("max_windows", 4))
    max_lag_cfg = int(dcfg.get("flatness_max_lag", 8))
    nu = config.get("physical", {}).get("kinematic_viscosity")

    metrics: list[dict] = []
    uncertainty: list[dict] = []
    diagnostics: dict = {"phases": {}, "physical_dissipation_available": nu is not None}
    summary_phases: dict = {}

    for pidx, phase in enumerate(PHASES):
        if regime["phases"][phase]["turbulence_established"] is not True:
            summary_phases[phase] = {"status": "SKIPPED_GATE_A"}
            metrics.append({"phase": phase, "status": "SKIPPED_GATE_A"})
            continue
        data = load_decomposition(decomposition_dir, phase)
        grid = infer_structured_grid(data["coordinates"])
        support_grid = grid.reshape_nodes(data["support"].astype(bool))
        windows = adaptive_windows(support_grid, preferred=preferred, fallback_sizes=fallback, overlap=float(kcfg.get("overlap", 0.5)), min_valid_fraction=1.0, require_full=True, max_windows=max_windows)
        if not windows:
            summary_phases[phase] = {"status": "UNRESOLVED_NO_LOCAL_WINDOW"}
            metrics.append({"phase": phase, "status": "UNRESOLVED_NO_LOCAL_WINDOW"})
            continue

        residuals = data["residuals"]
        per_r = {"vorticity_rms": [], "enstrophy_mean": [], "resolved_dissipation_surrogate_mean": [], "divergence_ratio": []}
        flatness_by_axis = {a: [] for a in range(3)}
        r_by_axis: dict[int, np.ndarray] = {}
        max_lag = min(max_lag_cfg, max(3, min(windows[0].shape) // 4))

        for ridx in range(residuals.shape[0]):
            local_metrics = {k: [] for k in per_r}
            local_flat = {a: [] for a in range(3)}
            for window in windows:
                u = grid.extract_window(residuals[ridx], window)
                grad = velocity_gradient(u, grid.spacing)
                strain = strain_rate(grad)
                omega = vorticity(grad)
                ens = enstrophy(omega)
                surrogate = resolved_dissipation_surrogate(strain)
                div = np.trace(grad, axis1=-2, axis2=-1)
                grad_norm = np.sqrt(np.mean(np.sum(grad**2, axis=(-2, -1))))
                local_metrics["vorticity_rms"].append(float(np.sqrt(np.mean(np.sum(omega**2, axis=-1)))))
                local_metrics["enstrophy_mean"].append(float(np.mean(ens)))
                local_metrics["resolved_dissipation_surrogate_mean"].append(float(np.mean(surrogate)))
                local_metrics["divergence_ratio"].append(float(np.sqrt(np.mean(div**2)) / grad_norm) if grad_norm > 0 else np.nan)
                sf = directional_structure_functions(u, grid.spacing, max_lag=max_lag, orders=(2, 4))
                for axis in range(3):
                    r_by_axis[axis] = sf[axis]["r"]
                    local_flat[axis].append(flatness_from_moments(sf[axis]["S2"], sf[axis]["S4"]))
            for key in per_r:
                per_r[key].append(float(np.nanmean(local_metrics[key])))
            for axis in range(3):
                flatness_by_axis[axis].append(np.nanmean(np.asarray(local_flat[axis]), axis=0))

        phase_uncertainty = {}
        for qidx, (key, values) in enumerate(per_r.items()):
            ci = bootstrap_mean_ci(np.asarray(values), samples=samples, seed=seed + pidx * 100 + qidx)
            phase_uncertainty[key] = ci
            uncertainty.append({"phase": phase, "quantity": key, "lower": ci[0], "median": ci[1], "upper": ci[2], "bootstrap_axis": "realizations"})

        mean_metrics = {key: float(np.nanmean(values)) for key, values in per_r.items()}
        physical_resolved = float(nu) * mean_metrics["resolved_dissipation_surrogate_mean"] if nu is not None else None
        row = {"phase": phase, "status": "CHARACTERIZED", **mean_metrics, "resolved_dissipation_with_supplied_nu": physical_resolved, "window_shape": str(windows[0].shape), "spacing_native": str(grid.spacing)}
        metrics.append(row)
        flat_diag = {}
        for axis in range(3):
            arr = np.asarray(flatness_by_axis[axis])
            flat_diag[str(axis)] = {"r": r_by_axis[axis].tolist(), "mean": np.nanmean(arr, axis=0).tolist()}
        diagnostics["phases"][phase] = {"metrics": row, "uncertainty": {k: list(map(float, v)) for k, v in phase_uncertainty.items()}, "flatness": flat_diag, "derivative_resolution_statement": "Derivatives are finite-difference estimates on native local windows; no claim is made that the Kolmogorov scale is resolved."}
        summary_phases[phase] = {"status": "CHARACTERIZED", "divergence_ratio": mean_metrics["divergence_ratio"], "physical_resolved_dissipation_reported": nu is not None}

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for axis in range(3):
                arr = np.asarray(flatness_by_axis[axis])
                ax.plot(r_by_axis[axis], np.nanmean(arr, axis=0), marker="o", label="xyz"[axis])
            ax.axhline(3.0, linestyle="--", linewidth=0.8, label="Gaussian 3")
            ax.set_xlabel("Separation (native length)")
            ax.set_ylabel("Longitudinal increment flatness")
            ax.legend()
            fig.tight_layout()
            fig.savefig(run_dir / "figures" / f"{phase}_flatness.png", dpi=180)
            plt.close(fig)
        except Exception as exc:
            append_log(run_dir, f"figure warning {phase}: {exc}")

    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_json(run_dir / "diagnostics.json", diagnostics)
    summary = {"run_dir": str(run_dir), "phases": summary_phases}
    write_json(run_dir / "diagnostics_summary.json", summary)
    finalize_run(run_dir)
    return summary
