from __future__ import annotations

from pathlib import Path
from .config import load_config
from .reporting.run_manifest import new_run_id, write_json
from .decomposition.driver import run_decomposition
from .regime.driver import run_regime
from .kolmogorov.driver import run_kolmogorov
from .taylor.driver import run_taylor
from .anisotropy.driver import run_anisotropy
from .diagnostics.driver import run_diagnostics
from .classification.driver import run_classification


def run_all(dataset_root: str | Path, output_root: str | Path, *, config_path: str | Path | None = None) -> dict:
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    config = load_config(config_path)
    study_id = new_run_id()

    decomposition = run_decomposition(dataset_root, output_root, config, run_id=f"{study_id}_decomposition")
    regime = run_regime(dataset_root, decomposition["run_dir"], output_root, config, run_id=f"{study_id}_gate_a")
    ko = run_kolmogorov(dataset_root, decomposition["run_dir"], regime["run_dir"], output_root, config, run_id=f"{study_id}_ko")
    taylor = run_taylor(dataset_root, decomposition["run_dir"], regime["run_dir"], output_root, config, run_id=f"{study_id}_taylor")
    anisotropy = run_anisotropy(dataset_root, decomposition["run_dir"], regime["run_dir"], output_root, config, run_id=f"{study_id}_anisotropy")
    diagnostics = run_diagnostics(dataset_root, decomposition["run_dir"], regime["run_dir"], output_root, config, run_id=f"{study_id}_diagnostics")
    classification = run_classification(dataset_root, regime["run_dir"], ko["run_dir"], output_root, config, taylor_dir=taylor["run_dir"], anisotropy_dir=anisotropy["run_dir"], diagnostics_dir=diagnostics["run_dir"], run_id=f"{study_id}_classification")

    summary = {
        "study_id": study_id,
        "dataset_root": str(dataset_root),
        "stages": {
            "decomposition": decomposition["run_dir"],
            "regime": regime["run_dir"],
            "kolmogorov_obukhov": ko["run_dir"],
            "taylor": taylor["run_dir"],
            "anisotropy": anisotropy["run_dir"],
            "diagnostics": diagnostics["run_dir"],
            "classification": classification["run_dir"],
        },
        "phase_classifications": classification["phase_classifications"],
    }
    study_dir = output_root / "study_runs" / study_id
    study_dir.mkdir(parents=True, exist_ok=False)
    write_json(study_dir / "study_summary.json", summary)
    return summary
