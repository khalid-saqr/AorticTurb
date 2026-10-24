from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..grid import WindowSpec, infer_structured_grid
from ..kolmogorov.flux import coarse_grained_flux
from ..reporting.run_manifest import (
    append_log,
    create_run_directory,
    finalize_run,
    initialize_run,
    write_csv,
    write_json,
)
from ..uncertainty.bootstrap import bootstrap_mean_ci
from .metrics import SIGNED_METRIC_KEYS, ko_interior_mask, signed_transfer_metrics


METHODS = """# Signed interscale energy-transfer / backscatter analysis

This characterization stage is downstream of Gate A and does not alter any turbulence,
Kolmogorov-Obukhov, Taylor, anisotropy, or terminal-classification decision.  It analyzes
only phases for which the primary Gate-A result has `turbulence_established=true`.

The signed local interscale transfer is the existing AorticTurb coarse-grained quantity

`Pi_ell(x) = - tau_ij^(ell)(x) * Sbar_ij^(ell)(x)`.

The existing sign convention is retained unchanged: `Pi_ell > 0` denotes local
forward/downscale transfer and `Pi_ell < 0` denotes local backscatter/upscale transfer.
No amplitude threshold is introduced; exact zero is kept as a separate category.  Negative
`Pi_ell` is interpreted only as local backscatter of the phase-conditioned residual kinetic
energy.  This stage does not by itself establish a sustained inverse cascade,
Kraichnan-Leith-Batchelor dynamics, or two-dimensional turbulence.

The completed KO run is the numerical source of truth for local windows, Gaussian filter
sigmas, and native filter scales.  The same KO edge exclusion is reused: a margin of
`ceil(3*sigma)` grid points is removed from every window face when the remaining interior
is sufficiently large.  The stored pointwise fields are NaN outside that admissible
interior.  Selected KO windows are summarized with equal weight, exactly as in the KO
stage; because those windows may overlap, their aggregate fractions are not presented as a
unique whole-AAo volume fraction.

For each realization and filter scale the stage reports the signed mean and median,
forward/backscatter/zero fractions, and the separate positive and negative contributions.
Uncertainty is bootstrapped only across independent phase-locked realizations.  Spatial
nodes are never treated as independent realizations.  Full pointwise signed-transfer
fields are retained for later spatial maps, PDFs, and backscatter-probability plots.
"""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _windows_from_ko(phase_diag: dict) -> list[WindowSpec]:
    windows = []
    for item in phase_diag.get("windows", []):
        windows.append(
            WindowSpec(
                tuple(int(x) for x in item["start"]),
                tuple(int(x) for x in item["stop"]),
                float(item["valid_fraction"]),
            )
        )
    return windows


def _backscatter_probability(pi: np.ndarray) -> np.ndarray:
    finite = np.isfinite(pi)
    count = np.sum(finite, axis=0)
    negative = np.sum(finite & (pi < 0.0), axis=0)
    out = np.full(count.shape, np.nan, dtype=float)
    np.divide(negative, count, out=out, where=count > 0)
    return out


def run_backscatter(
    dataset_root: str | Path,
    decomposition_dir: str | Path,
    regime_dir: str | Path,
    kolmogorov_dir: str | Path,
    output_root: str | Path,
    config: dict,
    *,
    run_id: str | None = None,
) -> dict:
    """Characterize pointwise signed interscale transfer using the locked KO support."""
    dataset_root = Path(dataset_root)
    decomposition_dir = Path(decomposition_dir)
    regime_dir = Path(regime_dir)
    kolmogorov_dir = Path(kolmogorov_dir)

    regime_path = regime_dir / "regime_summary.json"
    ko_manifest_path = kolmogorov_dir / "run_manifest.json"
    ko_config_path = kolmogorov_dir / "config.json"
    ko_diagnostics_path = kolmogorov_dir / "diagnostics.json"
    ko_summary_path = kolmogorov_dir / "kolmogorov_summary.json"

    regime = _read_json(regime_path)
    ko_manifest = _read_json(ko_manifest_path)
    ko_config = _read_json(ko_config_path)
    ko_diagnostics = _read_json(ko_diagnostics_path)

    if ko_manifest.get("status") != "COMPLETE":
        raise RuntimeError("Backscatter analysis requires a COMPLETE KO run.")
    if ko_manifest.get("branch") != "02_kolmogorov_obukhov":
        raise RuntimeError("Backscatter analysis requires a 02_kolmogorov_obukhov run.")

    analysis_cfg = ko_config.get("analysis", {})
    ko_cfg = ko_config.get("kolmogorov", {})
    samples = int(analysis_cfg.get("bootstrap_samples", 1000))
    seed = int(analysis_cfg.get("random_seed", 20261024))
    sigmas = np.asarray(
        ko_cfg.get("flux_sigmas", [1.0, 1.5, 2.0, 3.0, 4.0]), dtype=float
    )
    if sigmas.ndim != 1 or sigmas.size == 0 or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise RuntimeError("KO run contains invalid flux_sigmas.")

    run_dir = create_run_directory(
        output_root, "04b_signed_interscale_transfer", run_id=run_id
    )
    initialize_run(
        run_dir,
        branch="04b_signed_interscale_transfer",
        dataset_root=dataset_root,
        config=config,
        methods=METHODS,
        extra_inputs=[
            decomposition_dir / "decomposition_summary.json",
            regime_path,
            ko_manifest_path,
            ko_config_path,
            ko_diagnostics_path,
            ko_summary_path,
        ],
    )

    metrics: list[dict] = []
    uncertainty: list[dict] = []
    window_metrics: list[dict] = []
    diagnostics: dict = {
        "sign_convention": {
            "definition": "Pi_ell = -tau_ij^(ell) Sbar_ij^(ell)",
            "forward": "Pi_ell > 0",
            "backscatter": "Pi_ell < 0",
            "zero": "Pi_ell == 0",
        },
        "interpretation_limit": (
            "Negative Pi_ell denotes local backscatter only; this stage does not by itself "
            "establish a sustained inverse cascade, KLB dynamics, or two-dimensional turbulence."
        ),
        "parameter_source": str(ko_config_path),
        "window_source": str(ko_diagnostics_path),
        "bootstrap_axis": "realizations",
        "phases": {},
    }
    summary_phases: dict[str, dict] = {}

    for pidx, phase in enumerate(PHASES):
        gate = regime.get("phases", {}).get(phase, {}).get("turbulence_established")
        if gate is not True:
            summary_phases[phase] = {"status": "SKIPPED_GATE_A"}
            diagnostics["phases"][phase] = {"status": "SKIPPED_GATE_A"}
            metrics.append({"phase": phase, "status": "SKIPPED_GATE_A"})
            append_log(run_dir, f"{phase} SKIPPED_GATE_A")
            continue

        ko_phase = ko_diagnostics.get("phases", {}).get(phase, {})
        windows = _windows_from_ko(ko_phase)
        if not windows:
            summary_phases[phase] = {"status": "UNRESOLVED_NO_KO_WINDOWS"}
            diagnostics["phases"][phase] = {"status": "UNRESOLVED_NO_KO_WINDOWS"}
            metrics.append({"phase": phase, "status": "UNRESOLVED_NO_KO_WINDOWS"})
            append_log(run_dir, f"{phase} UNRESOLVED_NO_KO_WINDOWS")
            continue

        window_shape = windows[0].shape
        if any(window.shape != window_shape for window in windows):
            raise RuntimeError(f"{phase}: KO windows do not share one shape.")

        data = load_decomposition(decomposition_dir, phase)
        residuals = np.asarray(data["residuals"], dtype=float)
        grid = infer_structured_grid(data["coordinates"])
        support_grid = grid.reshape_nodes(np.asarray(data["support"], dtype=bool))

        for widx, window in enumerate(windows):
            if window.valid_fraction != 1.0 or not bool(np.all(support_grid[window.slices])):
                raise RuntimeError(
                    f"{phase}: saved KO window {widx} is not fully finite under the decomposition support."
                )

        scales_native = sigmas * float(np.mean(grid.spacing))
        saved_scales = np.asarray(
            ko_phase.get("flux", {}).get("scales_native", []), dtype=float
        )
        if saved_scales.shape != scales_native.shape or not np.allclose(
            saved_scales, scales_native, rtol=1e-12, atol=1e-12
        ):
            raise RuntimeError(
                f"{phase}: reconstructed filter scales do not match the saved KO scales."
            )

        masks = np.stack(
            [ko_interior_mask(window_shape, float(sigma)) for sigma in sigmas], axis=0
        )
        interior_counts = np.sum(masks, axis=(1, 2, 3)).astype(int)
        nreal = residuals.shape[0]
        nwin = len(windows)
        nscale = sigmas.size

        pi_store = np.full(
            (nreal, nwin, nscale) + window_shape,
            np.nan,
            dtype=float,
        )
        per_window = {
            key: np.full((nreal, nwin, nscale), np.nan, dtype=float)
            for key in SIGNED_METRIC_KEYS
        }

        for ridx in range(nreal):
            for widx, window in enumerate(windows):
                u = grid.extract_window(residuals[ridx], window)
                for sidx, sigma in enumerate(sigmas):
                    pi = coarse_grained_flux(u, grid.spacing, float(sigma))
                    mask = masks[sidx]
                    masked = np.where(mask, pi, np.nan)
                    pi_store[ridx, widx, sidx] = masked
                    values = pi[mask]
                    signed = signed_transfer_metrics(values)
                    for key in SIGNED_METRIC_KEYS:
                        per_window[key][ridx, widx, sidx] = signed[key]

        per_realization = {
            key: np.nanmean(values, axis=1) for key, values in per_window.items()
        }

        phase_scale_rows: list[dict] = []
        for sidx, (sigma, scale_native) in enumerate(zip(sigmas, scales_native)):
            row: dict = {
                "phase": phase,
                "status": "CHARACTERIZED",
                "scale_index": sidx,
                "sigma_grid": float(sigma),
                "scale_native": float(scale_native),
                "windows": nwin,
                "interior_points_per_window": int(interior_counts[sidx]),
            }
            for qidx, key in enumerate(SIGNED_METRIC_KEYS):
                values = per_realization[key][:, sidx]
                row[key] = float(np.nanmean(values))
                ci = bootstrap_mean_ci(
                    values,
                    samples=samples,
                    seed=seed + pidx * 10000 + sidx * 100 + qidx,
                )
                uncertainty.append(
                    {
                        "phase": phase,
                        "scale_index": sidx,
                        "sigma_grid": float(sigma),
                        "scale_native": float(scale_native),
                        "quantity": key,
                        "lower": ci[0],
                        "median": ci[1],
                        "upper": ci[2],
                        "bootstrap_axis": "realizations",
                    }
                )
            metrics.append(row)
            phase_scale_rows.append(row)

            for widx, window in enumerate(windows):
                wrow: dict = {
                    "phase": phase,
                    "window_index": widx,
                    "window_start": str(window.start),
                    "window_stop": str(window.stop),
                    "scale_index": sidx,
                    "sigma_grid": float(sigma),
                    "scale_native": float(scale_native),
                    "interior_points": int(interior_counts[sidx]),
                }
                for key in SIGNED_METRIC_KEYS:
                    wrow[key] = float(np.nanmean(per_window[key][:, widx, sidx]))
                window_metrics.append(wrow)

        probability = _backscatter_probability(pi_store)
        processed_rel = Path("processed") / f"{phase}_signed_transfer.npz"
        processed_path = run_dir / processed_rel
        np.savez_compressed(
            processed_path,
            pi=pi_store,
            backscatter_probability=probability,
            interior_mask=masks,
            sigmas=sigmas,
            scales_native=scales_native,
            window_start=np.asarray([w.start for w in windows], dtype=np.int64),
            window_stop=np.asarray([w.stop for w in windows], dtype=np.int64),
            x=np.asarray(grid.x, dtype=float),
            y=np.asarray(grid.y, dtype=float),
            z=np.asarray(grid.z, dtype=float),
        )

        phase_diag = {
            "status": "CHARACTERIZED",
            "realizations": int(nreal),
            "grid": grid.describe(),
            "windows": [w.to_dict() for w in windows],
            "window_shape": list(window_shape),
            "window_aggregation": (
                "Equal mean across the KO-selected local windows; overlapping windows are "
                "not deduplicated and are not interpreted as a unique whole-AAo volume fraction."
            ),
            "sigmas_grid": [float(x) for x in sigmas],
            "scales_native": [float(x) for x in scales_native],
            "interior_points_per_window": [int(x) for x in interior_counts],
            "edge_rule": (
                "Exclude ceil(3*sigma) points from each window face when all dimensions "
                "satisfy n > 2*margin + 2; otherwise retain the full window, matching KO."
            ),
            "plot_data": {
                "path": str(processed_rel),
                "fields": [
                    "pi",
                    "backscatter_probability",
                    "interior_mask",
                    "sigmas",
                    "scales_native",
                    "window_start",
                    "window_stop",
                    "x",
                    "y",
                    "z",
                ],
                "pi_shape": list(pi_store.shape),
                "coordinate_units": "native_dataset_units",
                "transfer_units": "native_velocity^3/native_length",
            },
            "scale_metrics": phase_scale_rows,
        }
        diagnostics["phases"][phase] = phase_diag
        summary_phases[phase] = {
            "status": "CHARACTERIZED",
            "plot_data": str(processed_rel),
            "scales_native": [float(x) for x in scales_native],
            "backscatter_fraction": [
                float(row["backscatter_fraction"]) for row in phase_scale_rows
            ],
            "mean_pi": [float(row["mean_pi"]) for row in phase_scale_rows],
        }
        append_log(
            run_dir,
            f"{phase} CHARACTERIZED windows={nwin} scales={nscale} realizations={nreal}",
        )

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            bs_ci = {
                int(row["scale_index"]): row
                for row in uncertainty
                if row.get("phase") == phase
                and row.get("quantity") == "backscatter_fraction"
            }
            mean_ci = {
                int(row["scale_index"]): row
                for row in uncertainty
                if row.get("phase") == phase and row.get("quantity") == "mean_pi"
            }

            y = np.asarray([row["backscatter_fraction"] for row in phase_scale_rows])
            lo = np.asarray([bs_ci[i]["lower"] for i in range(nscale)])
            hi = np.asarray([bs_ci[i]["upper"] for i in range(nscale)])
            fig, ax = plt.subplots()
            ax.errorbar(scales_native, y, yerr=np.vstack((y - lo, hi - y)), marker="o")
            ax.set_xlabel("Filter scale (native length)")
            ax.set_ylabel("Selected-window backscatter fraction")
            fig.tight_layout()
            fig.savefig(run_dir / "figures" / f"{phase}_backscatter_fraction.png", dpi=180)
            plt.close(fig)

            y = np.asarray([row["mean_pi"] for row in phase_scale_rows])
            lo = np.asarray([mean_ci[i]["lower"] for i in range(nscale)])
            hi = np.asarray([mean_ci[i]["upper"] for i in range(nscale)])
            fig, ax = plt.subplots()
            ax.errorbar(scales_native, y, yerr=np.vstack((y - lo, hi - y)), marker="o")
            ax.axhline(0.0, linestyle="--", linewidth=0.8)
            ax.set_xlabel("Filter scale (native length)")
            ax.set_ylabel("Mean signed interscale transfer (native units)")
            fig.tight_layout()
            fig.savefig(run_dir / "figures" / f"{phase}_mean_signed_transfer.png", dpi=180)
            plt.close(fig)
        except Exception as exc:
            append_log(run_dir, f"figure warning {phase}: {exc}")

    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_csv(run_dir / "window_metrics.csv", window_metrics)
    write_json(run_dir / "diagnostics.json", diagnostics)
    summary = {
        "run_dir": str(run_dir),
        "phases": summary_phases,
        "classification_effect": "NONE_CHARACTERIZATION_ONLY",
    }
    write_json(run_dir / "backscatter_summary.json", summary)
    finalize_run(run_dir, status="COMPLETE")
    return summary
