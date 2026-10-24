from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .constants import PHASES, REALIZATIONS
from .data.layout import phase_dir
from .data.loader import load_velocity_field, split_field
from .validation.masks import common_support, support_fraction
from .validation.startup_gate import run_startup_gate
from .decomposition.core import phase_decomposition, reynolds_stress
from .regime.gate_a import evaluate_gate_a


def load_phase(root: str | Path, phase: str) -> tuple[np.ndarray, np.ndarray]:
    coordinates = None
    velocities = []
    for name in REALIZATIONS:
        field = load_velocity_field(phase_dir(root, phase) / name, mmap=True)
        xyz, uvw = split_field(field)
        if coordinates is None:
            coordinates = np.asarray(xyz)
        elif not np.array_equal(np.asarray(xyz), coordinates, equal_nan=True):
            raise ValueError(f"{phase}: coordinate grid differs across realizations.")
        velocities.append(np.asarray(uvw))
    return coordinates, np.stack(velocities)


def run_decomposition(root: str | Path, output: str | Path) -> dict:
    gate = run_startup_gate(root)
    if not gate.passed:
        raise RuntimeError("Startup gate failed: " + json.dumps(gate.to_dict()))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"startup_gate": gate.to_dict(), "phases": {}}

    for phase in PHASES:
        xyz, fields = load_phase(root, phase)
        support = common_support(fields, policy="intersection")
        mean, residuals = phase_decomposition(fields, support=support)
        R, k = reynolds_stress(residuals)

        np.savez_compressed(
            output / f"{phase}_decomposition.npz",
            coordinates=xyz,
            support=support,
            mean=mean,
            residuals=residuals,
            reynolds_stress=R,
            k=k,
        )

        evidence = evaluate_gate_a(
            mean,
            residuals,
            support_fraction_value=support_fraction(fields, "intersection"),
        )
        summary["phases"][phase] = {
            "support_fraction": float(support.mean()),
            "gate_a_readiness": evidence.to_dict(),
        }

    (output / "decomposition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
