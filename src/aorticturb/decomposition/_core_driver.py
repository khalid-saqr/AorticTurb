from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ..constants import PHASES, REALIZATIONS
from ..data.layout import phase_dir
from ..data.loader import load_velocity_field, split_field
from ..validation.masks import realization_masks
from ..validation.support import audit_realization_masks
from ..validation.startup_gate import run_startup_gate
from ..uncertainty.bootstrap import bootstrap_phase_energy
from ..reporting.run_manifest import (
    create_run_directory, initialize_run, finalize_run, write_json, write_csv, append_log,
)
from .core import phase_decomposition, reynolds_stress


METHODS = """# Phase-conditioned decomposition

Each phase is treated independently. Sixteen phase-locked realizations are decomposed as
`u^(n) = u_bar + u'^(n)`. The common finite-velocity intersection is used for ensemble
statistics; source NaNs are never filled. Reynolds stresses and `k = 0.5 R_ii` are
computed at every common-support node. Uncertainty is bootstrapped across realizations,
with the phase mean recomputed on each draw using a deterministic node subsample for
computational tractability. Phases are never pooled.

Measurement support is audited separately from turbulence classification. The audit records
whole-lattice occupancy, union/intersection support, realization-to-realization mask
agreement, and whether all realization masks are identical. These quantities are diagnostic
only at this stage and do not alter the Gate-A decision rule.
"""


def load_phase(root: str | Path, phase: str) -> tuple[np.ndarray, np.ndarray]:
    coordinates = None
    velocities = []
    for name in REALIZATIONS:
        field = load_velocity_field(phase_dir(root, phase) / name, mmap=True)
        xyz, uvw = split_field(field)
        if coordinates is None:
            coordinates = np.asarray(xyz)
        elif not np.array_equal(np.asarray(xyz), coordinates, equal_nan=True):
            raise ValueError(f"{phase}: coordinate grid differs across realizations")
        velocities.append(np.asarray(uvw))
    assert coordinates is not None
    return coordinates, np.stack(velocities)


def _convergence(fields: np.ndarray, support: np.ndarray, node_cap: int, seed: int) -> list[dict]:
    ids = np.flatnonzero(support)
    rng = np.random.default_rng(seed)
    if ids.size > node_cap:
        ids = np.sort(rng.choice(ids, size=node_cap, replace=False))
    subset = fields[:, ids, :]
    rows = []
    for n in range(2, subset.shape[0] + 1):
        mean = np.mean(subset[:n], axis=0)
        residual = subset[:n] - mean[None, ...]
        k = 0.5 * np.mean(np.sum(residual**2, axis=-1))
        rows.append({"n_realizations": n, "mean_k_native": float(k)})
    return rows


def run_decomposition(
    root: str | Path,
    output_root: str | Path,
    config: dict,
    *,
    run_id: str | None = None,
) -> dict:
    root = Path(root)
    gate = run_startup_gate(root, verify_checksums=config.get("dataset", {}).get("verify_checksums", True))
    if not gate.passed:
        raise RuntimeError("Startup gate failed: " + json.dumps(gate.to_dict()))

    run_dir = create_run_directory(output_root, "01_laminar_vs_unsteady", run_id=run_id)
    initialize_run(run_dir, branch="01_laminar_vs_unsteady", dataset_root=root, config=config, methods=METHODS)
    write_json(run_dir / "startup_gate.json", gate.to_dict())

    analysis = config.get("analysis", {})
    samples = int(analysis.get("bootstrap_samples", 1000))
    seed = int(analysis.get("random_seed", 20261024))
    node_cap = int(analysis.get("bootstrap_node_cap", 4096))

    metrics: list[dict] = []
    uncertainty: list[dict] = []
    diagnostics: dict = {"phases": {}}
    index: dict[str, str] = {}

    for pidx, phase in enumerate(PHASES):
        append_log(run_dir, f"loading {phase}")
        xyz, fields = load_phase(root, phase)
        masks = realization_masks(fields)
        support_audit = audit_realization_masks(masks)
        support_count = masks.sum(axis=0).astype(np.uint8)
        support_union = support_count > 0
        support = support_count == masks.shape[0]
        mean, residuals = phase_decomposition(fields, support=support)
        R, k = reynolds_stress(residuals)
        finite = support & np.isfinite(k)
        residual_energy = 0.5 * np.sum(residuals**2, axis=-1)
        per_realization_k = np.nanmean(residual_energy[:, support], axis=1)

        lower, median, upper = bootstrap_phase_energy(
            fields,
            support,
            samples=samples,
            seed=seed + pidx,
            node_cap=node_cap,
        )
        conv = _convergence(fields, support, node_cap=node_cap, seed=seed + 100 + pidx)
        npz = run_dir / "processed" / f"{phase}_decomposition.npz"
        np.savez_compressed(
            npz,
            coordinates=xyz,
            support=support,
            support_union=support_union,
            support_count=support_count,
            mean=mean,
            residuals=residuals,
            reynolds_stress=R,
            k=k,
            per_realization_k=per_realization_k,
        )
        index[phase] = str(npz.relative_to(run_dir))
        metrics.append({
            "phase": phase,
            "support_fraction": float(support.mean()),
            "common_support_nodes": int(support.sum()),
            "union_support_nodes": int(support_union.sum()),
            "intersection_over_union_fraction": support_audit.intersection_over_union_fraction,
            "masks_identical": support_audit.masks_identical,
            "mean_speed_median_native": float(np.nanmedian(np.linalg.norm(mean, axis=-1))),
            "k_mean_native": float(np.nanmean(k[finite])) if finite.any() else np.nan,
            "k_p95_native": float(np.nanpercentile(k[finite], 95)) if finite.any() else np.nan,
            "realizations": fields.shape[0],
        })
        uncertainty.append({
            "phase": phase,
            "quantity": "global_phase_conditioned_k_native",
            "lower": lower,
            "median": median,
            "upper": upper,
            "confidence": 0.95,
            "bootstrap_samples": samples,
            "bootstrap_axis": "realizations",
            "node_cap": node_cap,
        })
        diagnostics["phases"][phase] = {
            "finite_velocity_rows_per_realization": [int(v) for v in masks.sum(axis=1)],
            "common_support_nodes": int(support.sum()),
            "support_audit": support_audit.to_dict(),
            "convergence": conv,
        }
        del fields, residuals, R, k, mean

    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "uncertainty.csv", uncertainty)
    write_json(run_dir / "diagnostics.json", diagnostics)
    write_json(run_dir / "decomposition_index.json", index)

    # Convergence figure is deliberately scalar and realization-based.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for phase in PHASES:
            rows = diagnostics["phases"][phase]["convergence"]
            ax.plot([r["n_realizations"] for r in rows], [r["mean_k_native"] for r in rows], label=phase)
        ax.set_xlabel("Realizations")
        ax.set_ylabel("Mean k (native units)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / "ensemble_convergence.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        append_log(run_dir, f"figure warning: {exc}")

    summary = {"run_dir": str(run_dir), "phases": metrics, "index": index}
    write_json(run_dir / "decomposition_summary.json", summary)
    finalize_run(run_dir)
    return summary


def load_decomposition(run_dir: str | Path, phase: str) -> dict[str, np.ndarray]:
    run_dir = Path(run_dir)
    index = json.loads((run_dir / "decomposition_index.json").read_text(encoding="utf-8"))
    path = run_dir / index[phase]
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}
