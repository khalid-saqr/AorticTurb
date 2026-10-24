from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import numpy as np

from ..reporting.run_manifest import finalize_run, sha256_file, write_json, append_log
from ._core_driver import run_decomposition as _run_core


def run_decomposition(
    root: str | Path,
    workspace_root: str | Path,
    config: dict,
    *,
    run_id: str | None = None,
) -> dict:
    """Run decomposition and place reusable arrays under canonical 03_processed_data."""
    workspace_root = Path(workspace_root)
    result = _run_core(root, workspace_root, config, run_id=run_id)
    run_dir = Path(result["run_dir"])
    old_processed = run_dir / "processed"
    processed_dir = workspace_root / "03_processed_data" / run_dir.name
    processed_dir.mkdir(parents=True, exist_ok=False)

    index = json.loads((run_dir / "decomposition_index.json").read_text(encoding="utf-8"))
    processed_records = []
    new_index: dict[str, str] = {}
    for phase, relpath in index.items():
        source = run_dir / relpath
        target = processed_dir / source.name
        shutil.move(str(source), str(target))
        relative = os.path.relpath(target, run_dir)
        new_index[phase] = relative
        processed_records.append({
            "phase": phase,
            "path_from_workspace": str(target.relative_to(workspace_root)),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    if old_processed.exists():
        old_processed.rmdir()

    write_json(run_dir / "decomposition_index.json", new_index)
    write_json(run_dir / "processed_outputs.json", {
        "storage_policy": "Reusable transformed arrays are stored under 03_processed_data.",
        "processed_run_directory": str(processed_dir.relative_to(workspace_root)),
        "files": processed_records,
    })
    summary_path = run_dir / "decomposition_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["index"] = new_index
    summary["processed_run_directory"] = str(processed_dir)
    write_json(summary_path, summary)
    append_log(run_dir, f"moved reusable arrays to {processed_dir}")
    finalize_run(run_dir)
    return summary


def load_decomposition(run_dir: str | Path, phase: str) -> dict[str, np.ndarray]:
    run_dir = Path(run_dir)
    index = json.loads((run_dir / "decomposition_index.json").read_text(encoding="utf-8"))
    path = (run_dir / index[phase]).resolve()
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}
