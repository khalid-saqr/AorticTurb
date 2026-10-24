from __future__ import annotations

from pathlib import Path
import numpy as np

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..grid import infer_structured_grid, adaptive_windows
from ..kolmogorov.flux import coarse_grained_flux
from ..uncertainty.bootstrap import bootstrap_mean_ci
from ..reporting.run_manifest import create_run_directory, initialize_run, finalize_run, write_json, write_csv, append_log
from .gate_a import evaluate_gate_a


METHODS = """# Gate A — turbulence versus deterministic unsteadiness

Gate A is conservative and does not use a `-5/3` slope. It first checks ensemble/sample
support, then tests whether fixed-phase residuals possess reproducible spatial coherence,
a decaying two-point correlation, and statistically positive forward scale-to-scale
energy transfer across the tested local filter scales. The transfer test is used only to
establish resolved stochastic multiscale dynamics; constancy across scale is reserved for
the later Kolmogorov–Obukhov stage. Bootstrap resampling is exclusively across the 16
independent realizations. A laminar/unsteady result requires an externally supplied
measurement-noise RMS ceiling; otherwise failure to establish turbulence is `UNRESOLVED`.
"""


def _correlation_curve(u: np.ndarray, max_lag: int) -> np.ndarray:
    values = []
    for lag in range(1, max_lag + 1):
        axis_values = []
        for axis in range(3):
            if u.shape[axis] <= lag:
                continue
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -lag)
            b[axis] = slice(lag, None)
            ua = u[tuple(a)]
            ub = u[tuple(b)]
            num = float(np.mean(np.sum(ua * ub, axis=-1)))
            ea = float(np.mean(np.sum(ua * ua, axis=-1)))
            eb = float(np.mean(np.sum(ub * ub, axis=-1)))
            denom = np.sqrt(ea * eb)
            axis_values.append(num / denom if denom > 0 else np.nan)
        values.append(float(np.nanmean(axis_values)) if axis_values else np.nan)
    return np.asarray(values, dtype=float)


def _mean_flux(u: np.ndarray, spacing: tuple[float, float, float], sigma: float) -> float:
    pi = coarse_grained_flux(u, spacing, sigma)
    margin = int(np.ceil(3 * sigma))
    if all(n > 2 * margin + 2 for n in pi.shape):
        core = pi[margin:-margin, margin:-margin, margin:-margin]
    else:
        core = pi
    return float(np.mean(core))


def run_regime(dataset_root: str | Path, decomposition_dir: str | Path, output_root: str | Path, config: dict, *, run_id: str | None = None) -> dict:
    dataset_root = Path(dataset_root)
    decomposition_dir = Path(decomposition_dir)
    run_dir = create_run_directory(output_root, "01_laminar_vs_unsteady", run_id=run_id)
    initialize_run(run_dir, branch="01_laminar_vs_unsteady", dataset_root=dataset_root, config=config, methods=METHODS, extra_inputs=[decomposition_dir / "decomposition_summary.json"])
    analysis = config.get("analysis", {})
    rcfg = config.get("regime", {})
    kcfg = config.get("kolmogorov", {})
    samples = int(analysis.get("bootstrap_samples", 1000))
    seed = int(analysis.get("random_seed", 20261024))
    preferred = tuple(int(x) for x in kcfg.get("window_shape", [32, 32, 32]))
    fallback = tuple(int(x) for x in kcfg.get("fallback_window_sizes", [24, 16, 12, 8]))
    max_windows = int(kcfg.get("max_windows", 4))
    overlap = float(kcfg.get("overlap", 0.5))
    sigmas = tuple(float(x) for x in rcfg.get("transfer_sigmas", [1.0, 2.0, 3.0]))
    max_lag_cfg = int(rcfg.get("correlation_max_lag", 8))
    noise_ceiling = rcfg.get("measurement_noise_rms_max")
    metrics: list[dict] = []
    uncertainty: list[dict] = []
    diagnostics: dict = {"phases": {}}
    phase_decisions: dict = {}
    for pidx, phase in enumerate(PHASES):
        data = load_decomposition(decomposition_dir, phase)
        xyz = data["coordinates"]
        support = data["support"].astype(bool)
        mean = data["mean"]
        residuals = data["residuals"]
        grid = infer_structured_grid(xyz)
        support_grid = grid.reshape_nodes(support)
        windows = adaptive_windows(support_grid, preferred=preferred, fallback_sizes=fallback, overlap=overlap, min_valid_fraction=1.0, require_full=True, max_windows=max_windows)
        readiness = evaluate_gate_a(mean, residuals, support_fraction_value=float(support.mean()), min_realizations=int(rcfg.get("min_realizations", 12)), min_support_fraction=float(rcfg.get("min_common_support_fraction", 0.50)))
        phase_diag = {"readiness": readiness.to_dict(), "grid": grid.describe(), "windows": [w.to_dict() for w in windows]}
        if not windows:
            decision = "UNRESOLVED"
            turbulence: bool | None = None
            phase_diag["reason"] = "No fully finite local rectangular window was available for multiscale tests."
            corr_mean = np.array([])
            flux_mean = np.array([])
        else:
            max_lag = min(max_lag_cfg, max(1, min(windows[0].shape) // 4))
            corr_by_r = np.full((residuals.shape[0], max_lag), np.nan)
            flux_by_r = np.full((residuals.shape[0], len(sigmas)), np.nan)
            for ridx in range(residuals.shape[0]):
                corr_windows = []
                flux_windows = []
                for window in windows:
                    uw = grid.extract_window(residuals[ridx], window)
                    corr_windows.append(_correlation_curve(uw, max_lag))
                    flux_windows.append([_mean_flux(uw, grid.spacing, sigma) for sigma in sigmas])
                corr_by_r[ridx] = np.nanmean(np.asarray(corr_windows), axis=0)
                flux_by_r[ridx] = np.nanmean(np.asarray(flux_windows), axis=0)
            corr_mean = np.nanmean(corr_by_r, axis=0)
            flux_mean = np.nanmean(flux_by_r, axis=0)
            corr_ci = [bootstrap_mean_ci(corr_by_r[:, i], samples=samples, seed=seed + pidx * 100 + i) for i in range(max_lag)]
            flux_ci = [bootstrap_mean_ci(flux_by_r[:, i], samples=samples, seed=seed + pidx * 1000 + i) for i in range(len(sigmas))]
            decay_ci = bootstrap_mean_ci(corr_by_r[:, 0] - corr_by_r[:, -1], samples=samples, seed=seed + 7000 + pidx)
            coherent = np.isfinite(corr_ci[0][0]) and corr_ci[0][0] > 0
            decays = np.isfinite(decay_ci[0]) and decay_ci[0] > 0
            transfer_positive = bool(flux_ci) and all(np.isfinite(ci[0]) and ci[0] > 0 for ci in flux_ci)
            per_realization_rms = np.sqrt(2.0 * np.asarray(data["per_realization_k"], dtype=float))
            rms_ci = bootstrap_mean_ci(per_realization_rms, samples=samples, seed=seed + 9000 + pidx)
            laminar_evidence = noise_ceiling is not None and np.isfinite(rms_ci[2]) and rms_ci[2] <= float(noise_ceiling) and not transfer_positive
            if readiness.decision == "PROCEED_TO_MULTISCALE_DIAGNOSTICS" and coherent and decays and transfer_positive:
                decision = "STOCHASTIC_MULTISCALE_SUPPORTED"
                turbulence = True
            elif laminar_evidence:
                decision = "LAMINAR_UNSTEADY_SUPPORTED"
                turbulence = False
            else:
                decision = "UNRESOLVED"
                turbulence = None
            phase_diag.update({"correlation_lags_native": [float((i + 1) * np.mean(grid.spacing)) for i in range(max_lag)], "correlation_mean": corr_mean.tolist(), "correlation_ci": [list(map(float, ci)) for ci in corr_ci], "correlation_decay_ci": list(map(float, decay_ci)), "transfer_sigma_grid": list(sigmas), "transfer_scale_native": [float(s * np.mean(grid.spacing)) for s in sigmas], "mean_flux": flux_mean.tolist(), "flux_ci": [list(map(float, ci)) for ci in flux_ci], "coherent_spatial_residual": bool(coherent), "correlation_decay_supported": bool(decays), "forward_transfer_supported": bool(transfer_positive), "measurement_noise_rms_max": noise_ceiling, "residual_rms_ci": list(map(float, rms_ci))})
            for i, ci in enumerate(corr_ci):
                uncertainty.append({"phase": phase, "quantity": f"correlation_lag_{i+1}", "lower": ci[0], "median": ci[1], "upper": ci[2], "bootstrap_axis": "realizations"})
            for i, ci in enumerate(flux_ci):
                uncertainty.append({"phase": phase, "quantity": f"mean_flux_sigma_{sigmas[i]:g}", "lower": ci[0], "median": ci[1], "upper": ci[2], "bootstrap_axis": "realizations"})
        metrics.append({"phase": phase, "decision": decision, "turbulence_established": turbulence, "fully_finite_windows": len(windows), "lag1_correlation": float(corr_mean[0]) if corr_mean.size else np.nan, "mean_forward_flux_native": float(np.nanmean(flux_mean)) if flux_mean.size else np.nan})
        phase_diag["decision"] = decision
        phase_diag["turbulence_established"] = turbulence
        diagnostics["phases"][phase] = phase_diag
        phase_decisions[phase] = {"decision": decision, "turbulence_established": turbulence}
        append_log(run_dir, f"{phase} decision={decision}")
    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_json(run_dir / "diagnostics.json", diagnostics)
    summary = {"run_dir": str(run_dir), "phases": phase_decisions}
    write_json(run_dir / "regime_summary.json", summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for phase in PHASES:
            d = diagnostics["phases"][phase]
            if "correlation_mean" in d:
                ax.plot(d["correlation_lags_native"], d["correlation_mean"], marker="o", label=phase)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlabel("Separation (native coordinate units)")
        ax.set_ylabel("Residual vector correlation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / "spatial_correlation.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        append_log(run_dir, f"figure warning: {exc}")
    finalize_run(run_dir)
    return summary
