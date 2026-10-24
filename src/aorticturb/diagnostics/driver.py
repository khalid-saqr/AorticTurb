from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..grid import infer_structured_grid, adaptive_windows
from ..kolmogorov.structure_functions import directional_structure_functions
from ..uncertainty.bootstrap import bootstrap_mean_ci
from ..reporting.run_manifest import (
    create_run_directory,
    initialize_run,
    finalize_run,
    write_json,
    write_csv,
    append_log,
)
from .admissibility import physical_dissipation_admissibility
from .gradients import (
    derivative_interior_mask,
    velocity_gradient_central,
    strain_rate,
    vorticity,
    enstrophy,
    resolved_dissipation_surrogate,
)
from .intermittency import flatness_from_moments
from .uncertainty import leave_one_out_mean_summary, subset_mean_convergence


METHODS = """# Canonical turbulence diagnostics

This stage is a consistency and characterization stage downstream of Gate A. It does not
re-open the turbulence decision or create a new primary classification.

Only Gate-A-authorized phases are analyzed. All derivative diagnostics are evaluated on
fully finite local rectangular windows selected from the native deposited grid. No NaN
filling, interpolation, remeshing, or phase pooling is performed.

Primary gradients use centered finite differences over one native grid interval. A second
centered estimate over two native grid intervals is evaluated on the same common interior
for derivative-scale sensitivity. The common interior excludes two nodes from every local
window face, preventing window-edge differencing from masquerading as a physical gradient.
The reported sensitivity is descriptive; no threshold is used to accept or reject a phase.
No claim is made that the Kolmogorov/dissipative scale is resolved.

Reported derivative diagnostics are vorticity magnitude, strain-rate magnitude, enstrophy,
the resolved surrogate `2 S'_ij S'_ij`, divergence RMS, and divergence relative to velocity-
gradient RMS. Physical dissipation `epsilon = 2 nu <S'_ij S'_ij>` is reported only if the
configuration explicitly contains verified kinematic viscosity and verified small-scale
resolution. Otherwise only the resolved surrogate is reported.

Longitudinal-increment flatness is computed directionally from the phase-conditioned
residual field as `F(r)=S4(r)/S2(r)^2`; Gaussian reference value 3 is reported only as a
reference, not as a cascade criterion. Bootstrap uncertainty is across the 16 independent
realizations. Leave-one-realization-out sensitivity and descriptive ensemble-size
convergence are also recorded. Spatial nodes are not treated as independent realizations.

Q/R velocity-gradient topology is marked inadmissible unless derivative accuracy at the
required small scales has been independently established; this run does not force a Q/R
answer. Plot-ready pointwise fields and directional increment moments are persisted for
later scientific visualization without recomputing the canonical diagnostics.
"""

DERIVATIVE_METRIC_KEYS = (
    "vorticity_rms",
    "enstrophy_mean",
    "strain_rate_rms",
    "resolved_dissipation_surrogate_mean",
    "divergence_rms",
    "gradient_rms",
    "divergence_ratio",
)


def _gradient_diagnostics(grad: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    strain = strain_rate(grad)
    omega = vorticity(grad)
    ens = enstrophy(omega)
    surrogate = resolved_dissipation_surrogate(strain)
    div = np.trace(grad, axis1=-2, axis2=-1)
    omega_mag = np.sqrt(np.sum(omega**2, axis=-1))
    strain_mag = np.sqrt(np.sum(strain**2, axis=(-2, -1)))
    grad_mag = np.sqrt(np.sum(grad**2, axis=(-2, -1)))

    vorticity_rms = float(np.sqrt(np.mean(omega_mag**2)))
    enstrophy_mean = float(np.mean(ens))
    strain_rate_rms = float(np.sqrt(np.mean(strain_mag**2)))
    surrogate_mean = float(np.mean(surrogate))
    divergence_rms = float(np.sqrt(np.mean(div**2)))
    gradient_rms = float(np.sqrt(np.mean(grad_mag**2)))
    divergence_ratio = divergence_rms / gradient_rms if gradient_rms > 0 else np.nan
    metrics = {
        "vorticity_rms": vorticity_rms,
        "enstrophy_mean": enstrophy_mean,
        "strain_rate_rms": strain_rate_rms,
        "resolved_dissipation_surrogate_mean": surrogate_mean,
        "divergence_rms": divergence_rms,
        "gradient_rms": gradient_rms,
        "divergence_ratio": float(divergence_ratio),
    }
    fields = {
        "vorticity_magnitude": omega_mag,
        "enstrophy": ens,
        "strain_rate_magnitude": strain_mag,
        "resolved_dissipation_surrogate": surrogate,
        "divergence": div,
    }
    return metrics, fields


def _pad_interior(field: np.ndarray, shape: tuple[int, int, int], trim: int) -> np.ndarray:
    out = np.full(shape, np.nan, dtype=float)
    out[trim:-trim, trim:-trim, trim:-trim] = np.asarray(field, dtype=float)
    return out


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
    regime_path = regime_dir / "regime_summary.json"
    regime = json.loads(regime_path.read_text(encoding="utf-8"))

    run_dir = create_run_directory(output_root, "05_canonical_diagnostics", run_id=run_id)
    initialize_run(
        run_dir,
        branch="05_canonical_diagnostics",
        dataset_root=dataset_root,
        config=config,
        methods=METHODS,
        extra_inputs=[decomposition_dir / "decomposition_summary.json", regime_path],
    )

    analysis = config.get("analysis", {})
    kcfg = config.get("kolmogorov", {})
    dcfg = config.get("diagnostics", {})
    samples = int(analysis.get("bootstrap_samples", 1000))
    seed = int(analysis.get("random_seed", 20261024))
    preferred = tuple(int(x) for x in kcfg.get("window_shape", [32, 32, 32]))
    fallback = tuple(int(x) for x in kcfg.get("fallback_window_sizes", [24, 16, 12, 8]))
    max_windows = int(kcfg.get("max_windows", 4))
    max_lag_cfg = int(dcfg.get("flatness_max_lag", 8))
    primary_step = int(dcfg.get("derivative_primary_step", 1))
    sensitivity_step = int(dcfg.get("derivative_sensitivity_step", 2))
    common_trim = int(dcfg.get("derivative_common_trim", max(primary_step, sensitivity_step)))
    if primary_step != 1 or sensitivity_step != 2:
        append_log(
            run_dir,
            "diagnostic note: derivative steps differ from project defaults step=1/2; values are recorded in diagnostics",
        )
    if common_trim < max(primary_step, sensitivity_step):
        raise ValueError("derivative_common_trim must cover both derivative steps.")

    convergence_sizes = tuple(int(x) for x in dcfg.get("convergence_sizes", [4, 8, 12, 16]))
    convergence_draws = int(dcfg.get("convergence_draws", 256))
    dissipation_admissibility = physical_dissipation_admissibility(config)

    metrics: list[dict] = []
    uncertainty: list[dict] = []
    derivative_sensitivity_rows: list[dict] = []
    convergence_rows: list[dict] = []
    flatness_rows: list[dict] = []
    window_metrics: list[dict] = []
    diagnostics: dict = {
        "phases": {},
        "bootstrap_axis": "realizations",
        "dissipation_admissibility": dissipation_admissibility,
        "derivative_method": {
            "primary_step_grid_intervals": primary_step,
            "sensitivity_step_grid_intervals": sensitivity_step,
            "common_trim_grid_points": common_trim,
            "scheme": "centered_finite_difference",
            "interpretation": (
                "Native-grid derivatives with a two-grid-interval sensitivity estimate on the same common interior; "
                "no claim that dissipative scales are resolved."
            ),
        },
        "q_r_topology": {
            "status": "INADMISSIBLE_DERIVATIVE_ACCURACY_NOT_ESTABLISHED",
            "reason": (
                "The project requires derivative quality to permit Q/R topology. Independent small-scale derivative "
                "accuracy is not established, so Q/R is not forced numerically."
            ),
        },
    }
    summary_phases: dict[str, dict] = {}

    for pidx, phase in enumerate(PHASES):
        if regime["phases"][phase]["turbulence_established"] is not True:
            summary_phases[phase] = {"status": "SKIPPED_GATE_A"}
            diagnostics["phases"][phase] = {"status": "SKIPPED_GATE_A"}
            metrics.append({"phase": phase, "status": "SKIPPED_GATE_A"})
            append_log(run_dir, f"{phase} SKIPPED_GATE_A")
            continue

        data = load_decomposition(decomposition_dir, phase)
        residuals = np.asarray(data["residuals"], dtype=float)
        grid = infer_structured_grid(data["coordinates"])
        support_grid = grid.reshape_nodes(np.asarray(data["support"], dtype=bool))
        windows = adaptive_windows(
            support_grid,
            preferred=preferred,
            fallback_sizes=fallback,
            overlap=float(kcfg.get("overlap", 0.5)),
            min_valid_fraction=1.0,
            require_full=True,
            max_windows=max_windows,
        )
        if not windows:
            summary_phases[phase] = {"status": "UNRESOLVED_NO_LOCAL_WINDOW"}
            diagnostics["phases"][phase] = {"status": "UNRESOLVED_NO_LOCAL_WINDOW"}
            metrics.append({"phase": phase, "status": "UNRESOLVED_NO_LOCAL_WINDOW"})
            append_log(run_dir, f"{phase} UNRESOLVED_NO_LOCAL_WINDOW")
            continue

        window_shape = windows[0].shape
        if any(window.shape != window_shape for window in windows):
            raise RuntimeError(f"{phase}: selected diagnostic windows do not share one shape.")
        if any(n <= 2 * common_trim for n in window_shape):
            raise RuntimeError(f"{phase}: local window too small for derivative common trim.")

        nreal = residuals.shape[0]
        nwin = len(windows)
        max_lag = min(max_lag_cfg, max(3, min(window_shape) // 4))
        interior_mask = derivative_interior_mask(window_shape, trim=common_trim)

        per_window_primary = {
            key: np.full((nreal, nwin), np.nan, dtype=float)
            for key in DERIVATIVE_METRIC_KEYS
        }
        per_window_sensitivity = {
            key: np.full((nreal, nwin), np.nan, dtype=float)
            for key in DERIVATIVE_METRIC_KEYS
        }
        field_store = {
            "vorticity_magnitude": np.full((nreal, nwin) + window_shape, np.nan, dtype=float),
            "strain_rate_magnitude": np.full((nreal, nwin) + window_shape, np.nan, dtype=float),
            "divergence": np.full((nreal, nwin) + window_shape, np.nan, dtype=float),
        }
        sensitivity_field_sums = {
            "vorticity_magnitude_sq": np.zeros((nwin,) + window_shape, dtype=float),
            "strain_rate_magnitude_sq": np.zeros((nwin,) + window_shape, dtype=float),
            "divergence_sq": np.zeros((nwin,) + window_shape, dtype=float),
        }

        flatness_store = np.full((nreal, nwin, 3, max_lag), np.nan, dtype=float)
        s2_store = np.full_like(flatness_store, np.nan)
        s4_store = np.full_like(flatness_store, np.nan)
        r_store = np.full((3, max_lag), np.nan, dtype=float)

        for ridx in range(nreal):
            for widx, window in enumerate(windows):
                u = grid.extract_window(residuals[ridx], window)
                grad_primary = velocity_gradient_central(
                    u,
                    grid.spacing,
                    step=primary_step,
                    trim=common_trim,
                )
                grad_sensitivity = velocity_gradient_central(
                    u,
                    grid.spacing,
                    step=sensitivity_step,
                    trim=common_trim,
                )
                primary_metrics, primary_fields = _gradient_diagnostics(grad_primary)
                sensitivity_metrics, sensitivity_fields = _gradient_diagnostics(grad_sensitivity)

                for key in DERIVATIVE_METRIC_KEYS:
                    per_window_primary[key][ridx, widx] = primary_metrics[key]
                    per_window_sensitivity[key][ridx, widx] = sensitivity_metrics[key]

                for key in ("vorticity_magnitude", "strain_rate_magnitude", "divergence"):
                    field_store[key][ridx, widx] = _pad_interior(
                        primary_fields[key], window_shape, common_trim
                    )
                sensitivity_field_sums["vorticity_magnitude_sq"][widx] += _pad_interior(
                    sensitivity_fields["vorticity_magnitude"] ** 2,
                    window_shape,
                    common_trim,
                )
                sensitivity_field_sums["strain_rate_magnitude_sq"][widx] += _pad_interior(
                    sensitivity_fields["strain_rate_magnitude"] ** 2,
                    window_shape,
                    common_trim,
                )
                sensitivity_field_sums["divergence_sq"][widx] += _pad_interior(
                    sensitivity_fields["divergence"] ** 2,
                    window_shape,
                    common_trim,
                )

                sf = directional_structure_functions(
                    u,
                    grid.spacing,
                    max_lag=max_lag,
                    orders=(2, 4),
                )
                for axis in range(3):
                    r = sf[axis]["r"]
                    flat = flatness_from_moments(sf[axis]["S2"], sf[axis]["S4"])
                    count = min(max_lag, r.size)
                    r_store[axis, :count] = r[:count]
                    s2_store[ridx, widx, axis, :count] = sf[axis]["S2"][:count]
                    s4_store[ridx, widx, axis, :count] = sf[axis]["S4"][:count]
                    flatness_store[ridx, widx, axis, :count] = flat[:count]

        per_real_primary = {
            key: np.nanmean(values, axis=1) for key, values in per_window_primary.items()
        }
        per_real_sensitivity = {
            key: np.nanmean(values, axis=1) for key, values in per_window_sensitivity.items()
        }

        phase_uncertainty = {}
        mean_metrics = {}
        for qidx, key in enumerate(DERIVATIVE_METRIC_KEYS):
            values = per_real_primary[key]
            mean_metrics[key] = float(np.nanmean(values))
            ci = bootstrap_mean_ci(values, samples=samples, seed=seed + pidx * 1000 + qidx)
            loo = leave_one_out_mean_summary(values)
            phase_uncertainty[key] = {
                "bootstrap_ci": [float(x) for x in ci],
                "leave_one_out": loo,
            }
            uncertainty.append(
                {
                    "phase": phase,
                    "quantity": key,
                    "lower": ci[0],
                    "median": ci[1],
                    "upper": ci[2],
                    "bootstrap_axis": "realizations",
                    **loo,
                }
            )
            for conv in subset_mean_convergence(
                values,
                sizes=convergence_sizes,
                draws=convergence_draws,
                seed=seed + pidx * 10000 + qidx,
            ):
                convergence_rows.append({"phase": phase, "quantity": key, **conv})

            primary = np.asarray(per_real_primary[key], dtype=float)
            sensitivity = np.asarray(per_real_sensitivity[key], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                relative = (sensitivity - primary) / np.abs(primary)
            relative[~np.isfinite(relative)] = np.nan
            rel_ci = bootstrap_mean_ci(
                relative,
                samples=samples,
                seed=seed + pidx * 1000 + 100 + qidx,
            )
            derivative_sensitivity_rows.append(
                {
                    "phase": phase,
                    "quantity": key,
                    "primary_step_grid_intervals": primary_step,
                    "sensitivity_step_grid_intervals": sensitivity_step,
                    "common_trim_grid_points": common_trim,
                    "primary_mean": float(np.nanmean(primary)),
                    "sensitivity_mean": float(np.nanmean(sensitivity)),
                    "relative_change_mean": float(np.nanmean(relative)),
                    "relative_change_lower": rel_ci[0],
                    "relative_change_median": rel_ci[1],
                    "relative_change_upper": rel_ci[2],
                    "bootstrap_axis": "realizations",
                }
            )

        physical_dissipation = None
        if dissipation_admissibility["admissible"]:
            physical_dissipation = (
                float(dissipation_admissibility["kinematic_viscosity"])
                * mean_metrics["resolved_dissipation_surrogate_mean"]
            )

        row = {
            "phase": phase,
            "status": "CHARACTERIZED",
            **mean_metrics,
            "physical_resolved_dissipation": physical_dissipation,
            "physical_dissipation_status": dissipation_admissibility["status"],
            "window_shape": str(window_shape),
            "windows": nwin,
            "spacing_native": str(grid.spacing),
            "derivative_primary_step": primary_step,
            "derivative_sensitivity_step": sensitivity_step,
            "derivative_common_trim": common_trim,
        }
        metrics.append(row)

        for widx, window in enumerate(windows):
            wrow = {
                "phase": phase,
                "window_index": widx,
                "window_start": str(window.start),
                "window_stop": str(window.stop),
                "valid_fraction": float(window.valid_fraction),
            }
            for key in DERIVATIVE_METRIC_KEYS:
                wrow[key] = float(np.nanmean(per_window_primary[key][:, widx]))
            window_metrics.append(wrow)

        flatness_mean = np.nanmean(flatness_store, axis=1)
        flat_diag = {}
        for axis in range(3):
            axis_rows = []
            for lag_idx in range(max_lag):
                r_value = r_store[axis, lag_idx]
                values = flatness_mean[:, axis, lag_idx]
                if not np.isfinite(r_value) or not np.isfinite(values).any():
                    continue
                ci = bootstrap_mean_ci(
                    values,
                    samples=samples,
                    seed=seed + pidx * 10000 + axis * 100 + lag_idx,
                )
                loo = leave_one_out_mean_summary(values)
                frow = {
                    "phase": phase,
                    "axis": axis,
                    "axis_label": "xyz"[axis],
                    "lag_index": lag_idx + 1,
                    "r_native": float(r_value),
                    "flatness_mean": float(np.nanmean(values)),
                    "gaussian_reference": 3.0,
                    "lower": ci[0],
                    "median": ci[1],
                    "upper": ci[2],
                    "bootstrap_axis": "realizations",
                    **loo,
                }
                flatness_rows.append(frow)
                axis_rows.append(frow)
                for conv in subset_mean_convergence(
                    values,
                    sizes=convergence_sizes,
                    draws=convergence_draws,
                    seed=seed + pidx * 100000 + axis * 1000 + lag_idx,
                ):
                    convergence_rows.append(
                        {
                            "phase": phase,
                            "quantity": f"flatness_{'xyz'[axis]}_lag{lag_idx + 1}",
                            **conv,
                        }
                    )
            flat_diag[str(axis)] = axis_rows

        with np.errstate(invalid="ignore"):
            vorticity_rms_map = np.sqrt(np.nanmean(field_store["vorticity_magnitude"] ** 2, axis=0))
            strain_rms_map = np.sqrt(np.nanmean(field_store["strain_rate_magnitude"] ** 2, axis=0))
            divergence_rms_map = np.sqrt(np.nanmean(field_store["divergence"] ** 2, axis=0))
            mean_enstrophy_map = 0.5 * np.nanmean(field_store["vorticity_magnitude"] ** 2, axis=0)
            mean_surrogate_map = 2.0 * np.nanmean(field_store["strain_rate_magnitude"] ** 2, axis=0)
            sensitivity_vorticity_rms_map = np.sqrt(
                sensitivity_field_sums["vorticity_magnitude_sq"] / nreal
            )
            sensitivity_strain_rms_map = np.sqrt(
                sensitivity_field_sums["strain_rate_magnitude_sq"] / nreal
            )
            sensitivity_divergence_rms_map = np.sqrt(
                sensitivity_field_sums["divergence_sq"] / nreal
            )

        processed_rel = Path("processed") / f"{phase}_canonical_diagnostics.npz"
        np.savez_compressed(
            run_dir / processed_rel,
            vorticity_magnitude=field_store["vorticity_magnitude"],
            strain_rate_magnitude=field_store["strain_rate_magnitude"],
            divergence=field_store["divergence"],
            vorticity_rms_map=vorticity_rms_map,
            strain_rate_rms_map=strain_rms_map,
            divergence_rms_map=divergence_rms_map,
            mean_enstrophy_map=mean_enstrophy_map,
            mean_resolved_dissipation_surrogate_map=mean_surrogate_map,
            sensitivity_vorticity_rms_map=sensitivity_vorticity_rms_map,
            sensitivity_strain_rate_rms_map=sensitivity_strain_rms_map,
            sensitivity_divergence_rms_map=sensitivity_divergence_rms_map,
            derivative_interior_mask=interior_mask,
            flatness=flatness_store,
            S2=s2_store,
            S4=s4_store,
            r_native=r_store,
            window_start=np.asarray([w.start for w in windows], dtype=np.int64),
            window_stop=np.asarray([w.stop for w in windows], dtype=np.int64),
            x=np.asarray(grid.x, dtype=float),
            y=np.asarray(grid.y, dtype=float),
            z=np.asarray(grid.z, dtype=float),
            primary_derivative_step=np.asarray(primary_step, dtype=np.int64),
            sensitivity_derivative_step=np.asarray(sensitivity_step, dtype=np.int64),
            derivative_common_trim=np.asarray(common_trim, dtype=np.int64),
        )

        diagnostics["phases"][phase] = {
            "status": "CHARACTERIZED",
            "metrics": row,
            "uncertainty": phase_uncertainty,
            "windows": [w.to_dict() for w in windows],
            "grid": grid.describe(),
            "flatness": flat_diag,
            "plot_data": {
                "path": str(processed_rel),
                "fields": [
                    "vorticity_magnitude",
                    "strain_rate_magnitude",
                    "divergence",
                    "vorticity_rms_map",
                    "strain_rate_rms_map",
                    "divergence_rms_map",
                    "mean_enstrophy_map",
                    "mean_resolved_dissipation_surrogate_map",
                    "sensitivity_vorticity_rms_map",
                    "sensitivity_strain_rate_rms_map",
                    "sensitivity_divergence_rms_map",
                    "derivative_interior_mask",
                    "flatness",
                    "S2",
                    "S4",
                    "r_native",
                    "window_start",
                    "window_stop",
                    "x",
                    "y",
                    "z",
                    "primary_derivative_step",
                    "sensitivity_derivative_step",
                    "derivative_common_trim",
                ],
                "coordinate_units": "native_dataset_units",
                "derivative_units": "native_velocity/native_length",
            },
            "derivative_resolution_statement": (
                f"Primary centered derivatives use one native grid interval; sensitivity uses {sensitivity_step} grid "
                f"intervals on a shared interior trimmed by {common_trim} nodes per face. The comparison quantifies "
                "derivative-scale sensitivity but does not establish resolution of dissipative scales."
            ),
            "physical_dissipation": dissipation_admissibility,
            "q_r_topology": diagnostics["q_r_topology"],
        }
        summary_phases[phase] = {
            "status": "CHARACTERIZED",
            "divergence_ratio": mean_metrics["divergence_ratio"],
            "physical_dissipation_status": dissipation_admissibility["status"],
            "plot_data": str(processed_rel),
        }
        append_log(run_dir, f"{phase} CHARACTERIZED windows={nwin} realizations={nreal}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for axis in range(3):
                rows = flat_diag[str(axis)]
                if rows:
                    ax.plot(
                        [item["r_native"] for item in rows],
                        [item["flatness_mean"] for item in rows],
                        marker="o",
                        label="xyz"[axis],
                    )
            ax.axhline(3.0, linestyle="--", linewidth=0.8, label="Gaussian 3")
            ax.set_xlabel("Separation (native length)")
            ax.set_ylabel("Longitudinal increment flatness")
            ax.legend()
            fig.tight_layout()
            fig.savefig(run_dir / "figures" / f"{phase}_flatness.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots()
            sensitivity_phase = [r for r in derivative_sensitivity_rows if r["phase"] == phase]
            x = np.arange(len(sensitivity_phase), dtype=float)
            y = np.asarray([r["relative_change_mean"] for r in sensitivity_phase], dtype=float)
            ax.bar(x, y)
            ax.axhline(0.0, linestyle="--", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([r["quantity"] for r in sensitivity_phase], rotation=45, ha="right")
            ax.set_ylabel("Mean relative change: step-2 vs step-1 derivative")
            fig.tight_layout()
            fig.savefig(run_dir / "figures" / f"{phase}_derivative_sensitivity.png", dpi=180)
            plt.close(fig)
        except Exception as exc:
            append_log(run_dir, f"figure warning {phase}: {exc}")

    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_csv(run_dir / "window_metrics.csv", window_metrics)
    write_csv(run_dir / "derivative_sensitivity.csv", derivative_sensitivity_rows)
    write_csv(run_dir / "ensemble_convergence.csv", convergence_rows)
    write_csv(run_dir / "flatness.csv", flatness_rows)
    write_json(run_dir / "diagnostics.json", diagnostics)
    summary = {
        "run_dir": str(run_dir),
        "phases": summary_phases,
        "classification_effect": "CHARACTERIZATION_INPUT_ONLY_NO_TERMINAL_CLASSIFICATION_RUN",
    }
    write_json(run_dir / "diagnostics_summary.json", summary)
    finalize_run(run_dir)
    return summary
