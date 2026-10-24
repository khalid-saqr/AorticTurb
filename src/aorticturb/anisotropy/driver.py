from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ..constants import PHASES
from ..decomposition.driver import load_decomposition
from ..reporting.run_manifest import (
    append_log,
    create_run_directory,
    finalize_run,
    initialize_run,
    write_csv,
    write_json,
)
from .tensor import (
    anisotropy_tensor,
    barycentric_coordinates,
    eigenvalues,
    invariants,
    realizability_mask,
)


METHODS = """# Lumley/Reynolds-stress anisotropy

The phase-conditioned Reynolds-stress tensor is converted to
`b_ij = R_ij/(2k) - delta_ij/3`. The project invariant convention is
`II_b = -0.5 b_ij b_ji` and `III_b = (1/3) b_ij b_jk b_ki`. Descending eigenvalues are
also mapped to barycentric realizability weights C1, C2, C3. No isotropy is assumed.
Realization bootstrap uncertainty is reported for global barycentric summaries; each
bootstrap draw is re-centered before its stress tensor is formed.

For each Gate-A-authorized phase, the already-computed nodewise anisotropy quantities are
also persisted in a compressed NPZ file under `processed/` so the classical Lumley
invariant map, barycentric realizability triangle, and spatial anisotropy maps can be
replotted later without recomputing the Reynolds-stress analysis. This persistence step
does not alter any anisotropy definition, admissibility rule, or scientific decision.
"""


PLOT_DATA_FIELDS = (
    "node_index",
    "coordinates",
    "k",
    "b",
    "II",
    "III",
    "eigenvalues",
    "barycentric",
    "realizable",
)


def _bootstrap_global_barycentric(
    residuals: np.ndarray,
    support: np.ndarray,
    *,
    samples: int,
    seed: int,
    node_cap: int,
) -> np.ndarray:
    ids = np.flatnonzero(support)
    rng = np.random.default_rng(seed)
    if ids.size > node_cap:
        ids = np.sort(rng.choice(ids, size=node_cap, replace=False))
    u = residuals[:, ids, :]
    n = u.shape[0]
    out = np.full((samples, 3), np.nan)
    for s in range(samples):
        rid = rng.integers(0, n, size=n)
        draw = u[rid]
        draw = draw - np.mean(draw, axis=0, keepdims=True)
        flat = draw.reshape(-1, 3)
        R = np.einsum("ni,nj->ij", flat, flat) / flat.shape[0]
        b, _ = anisotropy_tensor(R)
        out[s] = barycentric_coordinates(b)
    return out


def _write_plot_ready_anisotropy(
    run_dir: Path,
    phase: str,
    data: dict[str, np.ndarray],
    *,
    valid: np.ndarray,
    k: np.ndarray,
    b: np.ndarray,
    ii: np.ndarray,
    iii: np.ndarray,
    lam: np.ndarray,
    bary: np.ndarray,
    real: np.ndarray,
) -> Path:
    """Persist nodewise quantities already computed by the anisotropy stage."""
    ids = np.flatnonzero(valid).astype(np.int64, copy=False)
    coordinates = np.asarray(data["coordinates"])
    path = run_dir / "processed" / f"{phase}_anisotropy.npz"
    np.savez_compressed(
        path,
        node_index=ids,
        coordinates=coordinates[ids],
        k=np.asarray(k)[ids],
        b=np.asarray(b)[ids],
        II=np.asarray(ii)[ids],
        III=np.asarray(iii)[ids],
        eigenvalues=np.asarray(lam)[ids],
        barycentric=np.asarray(bary)[ids],
        realizable=np.asarray(real, dtype=bool)[ids],
    )
    return path


def run_anisotropy(
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
    regime = json.loads(
        (regime_dir / "regime_summary.json").read_text(encoding="utf-8")
    )
    run_dir = create_run_directory(
        output_root, "04_lumley_anisotropy", run_id=run_id
    )
    initialize_run(
        run_dir,
        branch="04_lumley_anisotropy",
        dataset_root=dataset_root,
        config=config,
        methods=METHODS,
        extra_inputs=[
            decomposition_dir / "decomposition_summary.json",
            regime_dir / "regime_summary.json",
        ],
    )
    analysis = config.get("analysis", {})
    samples = int(analysis.get("bootstrap_samples", 1000))
    seed = int(analysis.get("random_seed", 20261024))
    node_cap = int(analysis.get("bootstrap_node_cap", 4096))
    metrics: list[dict] = []
    uncertainty: list[dict] = []
    diagnostics: dict = {
        "invariant_convention": {
            "II": "-0.5 b_ij b_ji",
            "III": "(1/3) b_ij b_jk b_ki",
        },
        "plot_data_fields": list(PLOT_DATA_FIELDS),
        "phases": {},
    }
    summary_phases: dict = {}

    for pidx, phase in enumerate(PHASES):
        if regime["phases"][phase]["turbulence_established"] is not True:
            summary_phases[phase] = {"status": "SKIPPED_GATE_A"}
            metrics.append({"phase": phase, "status": "SKIPPED_GATE_A"})
            continue

        data = load_decomposition(decomposition_dir, phase)
        b, k = anisotropy_tensor(data["reynolds_stress"])
        ii, iii = invariants(b)
        lam = eigenvalues(b)
        bary = barycentric_coordinates(b)
        real = realizability_mask(b)
        valid = (
            data["support"].astype(bool)
            & np.isfinite(ii)
            & np.isfinite(iii)
            & np.isfinite(bary).all(axis=-1)
        )

        if not valid.any():
            summary_phases[phase] = {"status": "UNRESOLVED_NO_VALID_ANISOTROPY"}
            metrics.append(
                {"phase": phase, "status": "UNRESOLVED_NO_VALID_ANISOTROPY"}
            )
            continue

        plot_data_path = _write_plot_ready_anisotropy(
            run_dir,
            phase,
            data,
            valid=valid,
            k=k,
            b=b,
            ii=ii,
            iii=iii,
            lam=lam,
            bary=bary,
            real=real,
        )
        plot_data_rel = str(plot_data_path.relative_to(run_dir))

        boot = _bootstrap_global_barycentric(
            data["residuals"],
            data["support"].astype(bool),
            samples=samples,
            seed=seed + pidx,
            node_cap=node_cap,
        )
        bary_ci = np.quantile(boot, [0.025, 0.5, 0.975], axis=0)
        for i, name in enumerate(
            ("C1_one_component", "C2_two_component", "C3_three_component")
        ):
            uncertainty.append(
                {
                    "phase": phase,
                    "quantity": name,
                    "lower": float(bary_ci[0, i]),
                    "median": float(bary_ci[1, i]),
                    "upper": float(bary_ci[2, i]),
                    "bootstrap_axis": "realizations",
                }
            )

        row = {
            "phase": phase,
            "status": "CHARACTERIZED",
            "valid_nodes": int(valid.sum()),
            "realizability_fraction": float(real[valid].mean()),
            "II_median": float(np.nanmedian(ii[valid])),
            "III_median": float(np.nanmedian(iii[valid])),
            "C1_median": float(np.nanmedian(bary[valid, 0])),
            "C2_median": float(np.nanmedian(bary[valid, 1])),
            "C3_median": float(np.nanmedian(bary[valid, 2])),
        }
        metrics.append(row)
        diagnostics["phases"][phase] = {
            **row,
            "eigenvalue_medians": [
                float(np.nanmedian(lam[valid, i])) for i in range(3)
            ],
            "barycentric_bootstrap_ci": bary_ci.tolist(),
            "plot_data": {
                "path": plot_data_rel,
                "nodes": int(valid.sum()),
                "fields": list(PLOT_DATA_FIELDS),
                "coordinate_units": "native_dataset_units",
            },
        }
        summary_phases[phase] = {
            "status": "CHARACTERIZED",
            "realizability_fraction": row["realizability_fraction"],
            "median_barycentric": [
                row["C1_median"],
                row["C2_median"],
                row["C3_median"],
            ],
            "plot_data": plot_data_rel,
        }
        append_log(run_dir, f"{phase} plot-ready anisotropy saved: {plot_data_rel}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ids = np.flatnonzero(valid)
            rng = np.random.default_rng(seed + 100 + pidx)
            if ids.size > 5000:
                ids = rng.choice(ids, size=5000, replace=False)
            c = bary[ids]
            vertices = np.array(
                [[1.0, 0.0], [0.0, 0.0], [0.5, np.sqrt(3) / 2]]
            )
            xy = c @ vertices
            fig, ax = plt.subplots()
            tri = np.vstack([vertices, vertices[0]])
            ax.plot(tri[:, 0], tri[:, 1])
            ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.25)
            ax.set_aspect("equal")
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(
                run_dir / "figures" / f"{phase}_barycentric_map.png", dpi=180
            )
            plt.close(fig)
        except Exception as exc:
            append_log(run_dir, f"figure warning {phase}: {exc}")

    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_json(run_dir / "diagnostics.json", diagnostics)
    summary = {"run_dir": str(run_dir), "phases": summary_phases}
    write_json(run_dir / "anisotropy_summary.json", summary)
    finalize_run(run_dir)
    return summary
