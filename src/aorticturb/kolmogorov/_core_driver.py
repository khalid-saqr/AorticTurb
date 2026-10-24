from __future__ import annotations

from pathlib import Path
import numpy as np

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..grid import infer_structured_grid, adaptive_windows
from ..regime.ko_handoff import resolve_ko_handoff
from ..uncertainty.bootstrap import bootstrap_mean_ci
from ..reporting.run_manifest import create_run_directory, initialize_run, finalize_run, write_json, write_csv, append_log
from .spectra import directional_spectrum, isotropic_spectrum, fit_scaling_range, ScalingFit
from .structure_functions import directional_structure_functions, ess_exponent
from .flux import coarse_grained_flux


METHODS = """# Kolmogorov–Obukhov tests

This stage runs only for phases with an explicit Gate-A handoff authorization. The primary
Gate-A result is never rewritten. When a saved support-sensitivity run is explicitly supplied,
KO may proceed only where all locked non-primary support comparators concordantly project
STOCHASTIC_MULTISCALE_SUPPORTED; that handoff and its source files are recorded in the run.
All spectra, structure functions, and coarse-grained fluxes are computed directly in space
on local fully finite rectangular windows. No whole-domain FFT and no Taylor
frequency-to-wavenumber conversion is used. Directional spectra and longitudinal structure
functions are retained so isotropy is not assumed. Candidate scaling intervals are selected
by power-law linearity and interval length without reference to a target exponent, then fixed
for realization-level uncertainty. K41 targets are tested against realization-bootstrap
confidence intervals. The 4/5-law amplitude test uses the independently estimated positive
coarse-grained flux as epsilon. KO support requires concordance; a single `-5/3` slope can
never decide it.
"""

AXIS_NAMES = {0: "x", 1: "y", 2: "z"}


def _mean_flux(u: np.ndarray, spacing: tuple[float, float, float], sigma: float) -> float:
    pi = coarse_grained_flux(u, spacing, sigma)
    margin = int(np.ceil(3 * sigma))
    if all(n > 2 * margin + 2 for n in pi.shape):
        pi = pi[margin:-margin, margin:-margin, margin:-margin]
    return float(np.mean(pi))


def _slopes_on_fit(x: np.ndarray, values: np.ndarray, fit: ScalingFit | None) -> np.ndarray:
    if fit is None:
        return np.full(values.shape[0], np.nan)
    ids = slice(fit.start, fit.stop)
    out = np.full(values.shape[0], np.nan)
    for i, y in enumerate(values):
        xv = np.asarray(x[ids], dtype=float)
        yv = np.asarray(y[ids], dtype=float)
        valid = np.isfinite(xv) & np.isfinite(yv) & (xv > 0) & (np.abs(yv) > 0)
        if valid.sum() >= 3:
            out[i] = float(np.polyfit(np.log(xv[valid]), np.log(np.abs(yv[valid])), 1)[0])
    return out


def _target_status(ci: tuple[float, float, float], target: float) -> str:
    lo, _, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "UNRESOLVED"
    return "CONSISTENT" if lo <= target <= hi else "INCONSISTENT"


def _aggregate_directional(statuses: list[str]) -> str:
    if len(statuses) != 3 or any(s == "UNRESOLVED" for s in statuses):
        return "UNRESOLVED"
    if all(s == "CONSISTENT" for s in statuses):
        return "CONSISTENT"
    if sum(s == "INCONSISTENT" for s in statuses) >= 2:
        return "INCONSISTENT"
    return "UNRESOLVED"


def _interval_overlap(cis: list[tuple[float, float, float]]) -> bool | None:
    if len(cis) != 3 or any(not np.isfinite(ci[0]) or not np.isfinite(ci[2]) for ci in cis):
        return None
    return max(ci[0] for ci in cis) <= min(ci[2] for ci in cis)


def _flux_status(
    flux_by_realization: np.ndarray,
    scales: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[str, tuple[float, float, float], list[tuple[float, float, float]]]:
    slopes = np.full(flux_by_realization.shape[0], np.nan)
    for i, row in enumerate(flux_by_realization):
        valid = np.isfinite(row) & (np.abs(row) > 0)
        if valid.sum() >= 3:
            slopes[i] = float(np.polyfit(np.log(scales[valid]), np.log(np.abs(row[valid])), 1)[0])
    slope_ci = bootstrap_mean_ci(slopes, samples=samples, seed=seed)
    sign_ci = [bootstrap_mean_ci(flux_by_realization[:, j], samples=samples, seed=seed + 100 + j) for j in range(flux_by_realization.shape[1])]
    slope_flat = np.isfinite(slope_ci[0]) and slope_ci[0] <= 0 <= slope_ci[2]
    positive = all(np.isfinite(ci[0]) and ci[0] > 0 for ci in sign_ci)
    negative = all(np.isfinite(ci[2]) and ci[2] < 0 for ci in sign_ci)
    if slope_flat and positive:
        status = "CONSISTENT"
    elif negative or (np.isfinite(slope_ci[0]) and not slope_flat):
        status = "INCONSISTENT"
    else:
        status = "UNRESOLVED"
    return status, slope_ci, sign_ci


def run_kolmogorov(
    dataset_root: str | Path,
    decomposition_dir: str | Path,
    regime_dir: str | Path,
    output_root: str | Path,
    config: dict,
    *,
    support_sensitivity_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict:
    dataset_root = Path(dataset_root)
    decomposition_dir = Path(decomposition_dir)
    regime_dir = Path(regime_dir)
    sensitivity_dir = (
        Path(support_sensitivity_dir) if support_sensitivity_dir is not None else None
    )
    handoff = resolve_ko_handoff(decomposition_dir, regime_dir, sensitivity_dir)
    run_dir = create_run_directory(output_root, "02_kolmogorov_obukhov", run_id=run_id)
    extra_inputs = [
        decomposition_dir / "decomposition_summary.json",
        regime_dir / "regime_summary.json",
    ]
    if sensitivity_dir is not None:
        extra_inputs.append(sensitivity_dir / "support_sensitivity.json")
    initialize_run(
        run_dir,
        branch="02_kolmogorov_obukhov",
        dataset_root=dataset_root,
        config=config,
        methods=METHODS,
        extra_inputs=extra_inputs,
    )

    analysis = config.get("analysis", {})
    kcfg = config.get("kolmogorov", {})
    samples = int(analysis.get("bootstrap_samples", 1000))
    seed = int(analysis.get("random_seed", 20261024))
    preferred = tuple(int(x) for x in kcfg.get("window_shape", [32, 32, 32]))
    fallback = tuple(int(x) for x in kcfg.get("fallback_window_sizes", [24, 16, 12, 8]))
    overlap = float(kcfg.get("overlap", 0.5))
    max_windows = int(kcfg.get("max_windows", 4))
    max_lag_cfg = int(kcfg.get("max_lag", 8))
    sigmas = np.asarray(kcfg.get("flux_sigmas", [1.0, 1.5, 2.0, 3.0, 4.0]), dtype=float)
    spectrum_min_points = int(kcfg.get("minimum_scaling_points", 4))

    metrics: list[dict] = []
    uncertainty: list[dict] = []
    diagnostics: dict = {"phases": {}}
    phase_summary: dict = {}

    for pidx, phase in enumerate(PHASES):
        phase_handoff = handoff[phase]
        handoff_payload = phase_handoff.to_dict()
        if not phase_handoff.ko_authorized:
            phase_summary[phase] = {
                "status": "SKIPPED_GATE_A",
                "ko_resolved": None,
                "ko_supported": None,
                "gate_a_handoff": handoff_payload,
            }
            diagnostics["phases"][phase] = {"gate_a_handoff": handoff_payload}
            metrics.append({
                "phase": phase,
                "status": "SKIPPED_GATE_A",
                "ko_resolved": None,
                "ko_supported": None,
                "gate_a_handoff_authorized": False,
                "gate_a_handoff_source": phase_handoff.source,
            })
            append_log(run_dir, f"{phase} SKIPPED_GATE_A: {phase_handoff.reason}")
            continue

        data = load_decomposition(decomposition_dir, phase)
        grid = infer_structured_grid(data["coordinates"])
        support_grid = grid.reshape_nodes(data["support"].astype(bool))
        windows = adaptive_windows(support_grid, preferred=preferred, fallback_sizes=fallback, overlap=overlap, min_valid_fraction=1.0, require_full=True, max_windows=max_windows)
        if not windows:
            phase_summary[phase] = {
                "status": "UNRESOLVED_NO_LOCAL_WINDOW",
                "ko_resolved": False,
                "ko_supported": None,
                "gate_a_handoff": handoff_payload,
            }
            diagnostics["phases"][phase] = {"gate_a_handoff": handoff_payload}
            metrics.append({
                "phase": phase,
                "status": "UNRESOLVED_NO_LOCAL_WINDOW",
                "ko_resolved": False,
                "ko_supported": None,
                "gate_a_handoff_authorized": True,
                "gate_a_handoff_source": phase_handoff.source,
            })
            continue

        residuals = data["residuals"]
        nreal = residuals.shape[0]
        max_lag = min(max_lag_cfg, max(3, min(windows[0].shape) // 4))
        spec: dict[int, list[np.ndarray]] = {a: [] for a in range(3)}
        kvals: dict[int, np.ndarray] = {}
        sf: dict[int, dict[int, list[np.ndarray]]] = {a: {2: [], 3: [], 4: [], 6: []} for a in range(3)}
        rvals: dict[int, np.ndarray] = {}
        iso_specs: list[np.ndarray] = []
        iso_k: np.ndarray | None = None
        flux = np.full((nreal, len(sigmas)), np.nan)

        for ridx in range(nreal):
            per_window_spec = {a: [] for a in range(3)}
            per_window_sf = {a: {2: [], 3: [], 4: [], 6: []} for a in range(3)}
            per_window_flux = []
            per_window_iso = []
            for window in windows:
                u = grid.extract_window(residuals[ridx], window)
                for axis in range(3):
                    k, e = directional_spectrum(u, grid.spacing, axis=axis, taper=True)
                    kvals[axis] = k
                    per_window_spec[axis].append(e)
                dsf = directional_structure_functions(u, grid.spacing, max_lag=max_lag, orders=(2, 3, 4, 6))
                for axis in range(3):
                    rvals[axis] = dsf[axis]["r"]
                    for order in (2, 3, 4, 6):
                        per_window_sf[axis][order].append(dsf[axis][f"S{order}"])
                per_window_flux.append([_mean_flux(u, grid.spacing, s) for s in sigmas])
                ik, ie, _ = isotropic_spectrum(u, grid.spacing, bins=max(12, min(windows[0].shape) // 2), taper=True)
                iso_k = ik
                per_window_iso.append(ie)
            for axis in range(3):
                spec[axis].append(np.nanmean(np.asarray(per_window_spec[axis]), axis=0))
                for order in (2, 3, 4, 6):
                    sf[axis][order].append(np.nanmean(np.asarray(per_window_sf[axis][order]), axis=0))
            flux[ridx] = np.nanmean(np.asarray(per_window_flux), axis=0)
            iso_specs.append(np.nanmean(np.asarray(per_window_iso), axis=0))

        spec_arr = {a: np.asarray(spec[a]) for a in range(3)}
        sf_arr = {a: {p: np.asarray(sf[a][p]) for p in (2, 3, 4, 6)} for a in range(3)}
        iso_specs_arr = np.asarray(iso_specs)
        scales = sigmas * float(np.mean(grid.spacing))
        flux_status, flux_slope_ci, flux_sign_ci = _flux_status(flux, scales, samples=samples, seed=seed + 50000 + pidx)

        spectrum_axis_status = []
        s2_axis_status = []
        s3_axis_status = []
        spectrum_cis: list[tuple[float, float, float]] = []
        s2_cis: list[tuple[float, float, float]] = []
        phase_diag: dict = {
            "gate_a_handoff": handoff_payload,
            "grid": grid.describe(),
            "windows": [w.to_dict() for w in windows],
            "flux": {"scales_native": scales.tolist(), "mean": np.nanmean(flux, axis=0).tolist(), "status": flux_status, "slope_ci": list(map(float, flux_slope_ci)), "sign_ci": [list(map(float, ci)) for ci in flux_sign_ci]},
            "directions": {},
        }

        epsilon_r = np.nanmean(flux, axis=1)
        for axis in range(3):
            axis_name = AXIS_NAMES[axis]
            mean_spec = np.nanmean(spec_arr[axis], axis=0)
            spec_fit = fit_scaling_range(kvals[axis], mean_spec, min_points=spectrum_min_points)
            spec_slopes = _slopes_on_fit(kvals[axis], spec_arr[axis], spec_fit)
            spec_ci = bootstrap_mean_ci(spec_slopes, samples=samples, seed=seed + pidx * 100 + axis)
            spec_status = _target_status(spec_ci, -5.0 / 3.0)
            spectrum_cis.append(spec_ci)
            spectrum_axis_status.append(spec_status)

            mean_s2 = np.nanmean(sf_arr[axis][2], axis=0)
            s2_fit = fit_scaling_range(rvals[axis], mean_s2, min_points=spectrum_min_points)
            s2_slopes = _slopes_on_fit(rvals[axis], sf_arr[axis][2], s2_fit)
            s2_ci = bootstrap_mean_ci(s2_slopes, samples=samples, seed=seed + 1000 + pidx * 100 + axis)
            s2_status = _target_status(s2_ci, 2.0 / 3.0)
            s2_cis.append(s2_ci)
            s2_axis_status.append(s2_status)

            mean_s3 = np.nanmean(sf_arr[axis][3], axis=0)
            s3_fit = fit_scaling_range(rvals[axis], mean_s3, min_points=spectrum_min_points)
            s3_slopes = _slopes_on_fit(rvals[axis], sf_arr[axis][3], s3_fit)
            s3_slope_ci = bootstrap_mean_ci(s3_slopes, samples=samples, seed=seed + 2000 + pidx * 100 + axis)
            s3_slope_status = _target_status(s3_slope_ci, 1.0)
            ratios = np.full(nreal, np.nan)
            if s3_fit is not None:
                ids = slice(s3_fit.start, s3_fit.stop)
                rr = rvals[axis][ids]
                for ridx in range(nreal):
                    eps = epsilon_r[ridx]
                    s3v = sf_arr[axis][3][ridx, ids]
                    if np.isfinite(eps) and eps > 0:
                        ratio = s3v / (-(4.0 / 5.0) * eps * rr)
                        finite = ratio[np.isfinite(ratio)]
                        if finite.size:
                            ratios[ridx] = float(np.mean(finite))
            ratio_ci = bootstrap_mean_ci(ratios, samples=samples, seed=seed + 3000 + pidx * 100 + axis)
            ratio_status = _target_status(ratio_ci, 1.0)
            if s3_slope_status == "CONSISTENT" and ratio_status == "CONSISTENT":
                s3_status = "CONSISTENT"
            elif s3_slope_status == "INCONSISTENT" and ratio_status == "INCONSISTENT":
                s3_status = "INCONSISTENT"
            else:
                s3_status = "UNRESOLVED"
            s3_axis_status.append(s3_status)

            ess4 = np.asarray([ess_exponent(sf_arr[axis][3][i], sf_arr[axis][4][i]) for i in range(nreal)], dtype=float)
            ess6 = np.asarray([ess_exponent(sf_arr[axis][3][i], sf_arr[axis][6][i]) for i in range(nreal)], dtype=float)
            ess4_ci = bootstrap_mean_ci(ess4, samples=samples, seed=seed + 4000 + pidx * 100 + axis)
            ess6_ci = bootstrap_mean_ci(ess6, samples=samples, seed=seed + 5000 + pidx * 100 + axis)

            phase_diag["directions"][axis_name] = {
                "spectrum": {"k": kvals[axis].tolist(), "mean": mean_spec.tolist(), "fit": spec_fit.to_dict() if spec_fit else None, "slope_ci": list(map(float, spec_ci)), "status": spec_status},
                "S2": {"r": rvals[axis].tolist(), "mean": mean_s2.tolist(), "fit": s2_fit.to_dict() if s2_fit else None, "slope_ci": list(map(float, s2_ci)), "status": s2_status},
                "S3": {"r": rvals[axis].tolist(), "mean": mean_s3.tolist(), "fit": s3_fit.to_dict() if s3_fit else None, "slope_ci": list(map(float, s3_slope_ci)), "four_fifths_ratio_ci": list(map(float, ratio_ci)), "status": s3_status},
                "ESS": {"zeta4_ci": list(map(float, ess4_ci)), "zeta6_ci": list(map(float, ess6_ci))},
            }
            for quantity, ci, target in (("spectrum_slope", spec_ci, -5.0 / 3.0), ("S2_slope", s2_ci, 2.0 / 3.0), ("S3_slope", s3_slope_ci, 1.0), ("four_fifths_ratio", ratio_ci, 1.0), ("ESS_zeta4", ess4_ci, np.nan), ("ESS_zeta6", ess6_ci, np.nan)):
                uncertainty.append({"phase": phase, "direction": axis_name, "quantity": quantity, "lower": ci[0], "median": ci[1], "upper": ci[2], "theory_target": target, "bootstrap_axis": "realizations"})

        spectrum_status = _aggregate_directional(spectrum_axis_status)
        s2_status = _aggregate_directional(s2_axis_status)
        s3_status = _aggregate_directional(s3_axis_status)
        spec_overlap = _interval_overlap(spectrum_cis)
        s2_overlap = _interval_overlap(s2_cis)
        if spec_overlap is False or s2_overlap is False:
            local_isotropy = False
        elif spec_overlap is True and s2_overlap is True:
            local_isotropy = True
        else:
            local_isotropy = None
        core = {"spectrum": spectrum_status, "S2": s2_status, "S3_four_fifths": s3_status, "flux_plateau": flux_status}
        all_resolved = local_isotropy is True and all(v != "UNRESOLVED" for v in core.values())
        if all_resolved and all(v == "CONSISTENT" for v in core.values()):
            ko_supported: bool | None = True
        elif all_resolved and sum(v == "INCONSISTENT" for v in core.values()) >= 2:
            ko_supported = False
        else:
            ko_supported = None
        ko_resolved = bool(all_resolved)
        phase_diag.update({"local_isotropy_admissible": local_isotropy, "core_diagnostics": core, "ko_resolved": ko_resolved, "ko_supported": ko_supported, "isotropic_spectrum_secondary": {"k": iso_k.tolist() if iso_k is not None else [], "mean": np.nanmean(iso_specs_arr, axis=0).tolist() if iso_specs_arr.size else []}})
        diagnostics["phases"][phase] = phase_diag
        status = "KO_SUPPORTED" if ko_supported is True else "KO_INCONSISTENT" if ko_supported is False else "KO_UNRESOLVED"
        phase_summary[phase] = {
            "status": status,
            "ko_resolved": ko_resolved,
            "ko_supported": ko_supported,
            "core_diagnostics": core,
            "local_isotropy_admissible": local_isotropy,
            "gate_a_handoff": handoff_payload,
        }
        metrics.append({
            "phase": phase,
            "status": status,
            "ko_resolved": ko_resolved,
            "ko_supported": ko_supported,
            "local_isotropy_admissible": local_isotropy,
            "gate_a_handoff_authorized": True,
            "gate_a_handoff_source": phase_handoff.source,
            **{f"status_{k}": v for k, v in core.items()},
        })
        uncertainty.append({"phase": phase, "direction": "all", "quantity": "flux_log_slope", "lower": flux_slope_ci[0], "median": flux_slope_ci[1], "upper": flux_slope_ci[2], "theory_target": 0.0, "bootstrap_axis": "realizations"})
        append_log(run_dir, f"{phase} {status}; handoff={phase_handoff.source}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for axis in range(3):
                ax.loglog(kvals[axis], np.nanmean(spec_arr[axis], axis=0), label=AXIS_NAMES[axis])
            ax.set_xlabel("Wavenumber (1/native length)")
            ax.set_ylabel("Directional spectral energy")
            ax.legend()
            fig.tight_layout()
            fig.savefig(run_dir / "figures" / f"{phase}_directional_spectra.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots()
            mean_flux = np.nanmean(flux, axis=0)
            ax.plot(scales, mean_flux, marker="o")
            ax.axhline(0.0, linewidth=0.8)
            ax.set_xlabel("Filter scale (native length)")
            ax.set_ylabel("Mean Pi_ell (native units)")
            fig.tight_layout()
            fig.savefig(run_dir / "figures" / f"{phase}_flux.png", dpi=180)
            plt.close(fig)
        except Exception as exc:
            append_log(run_dir, f"figure warning {phase}: {exc}")

    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_json(run_dir / "diagnostics.json", diagnostics)
    summary = {"run_dir": str(run_dir), "phases": phase_summary}
    write_json(run_dir / "kolmogorov_summary.json", summary)
    finalize_run(run_dir)
    return summary
