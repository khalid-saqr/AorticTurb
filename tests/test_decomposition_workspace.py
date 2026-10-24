import json
from pathlib import Path
import numpy as np

from aorticturb.decomposition import driver
from aorticturb.reporting.run_manifest import create_run_directory, initialize_run, finalize_run, write_json


def test_decomposition_wrapper_moves_reusable_arrays_to_processed_tree(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    workspace = tmp_path / "workspace"

    def fake_core(root, output_root, config, run_id=None):
        run_dir = create_run_directory(output_root, "01_laminar_vs_unsteady", run_id=run_id or "run_fake")
        initialize_run(run_dir, branch="01_laminar_vs_unsteady", dataset_root=dataset, config=config, methods="# test")
        index = {}
        for phase in ("t018", "t021", "t024"):
            path = run_dir / "processed" / f"{phase}_decomposition.npz"
            np.savez_compressed(path, coordinates=np.zeros((1, 3)), support=np.ones(1, dtype=bool))
            index[phase] = str(path.relative_to(run_dir))
        write_json(run_dir / "decomposition_index.json", index)
        summary = {"run_dir": str(run_dir), "index": index, "phases": []}
        write_json(run_dir / "decomposition_summary.json", summary)
        finalize_run(run_dir)
        return summary

    monkeypatch.setattr(driver, "_run_core", fake_core)
    result = driver.run_decomposition(dataset, workspace, {}, run_id="run_fake")
    run_dir = Path(result["run_dir"])
    processed = workspace / "03_processed_data" / "run_fake"
    assert processed.is_dir()
    assert not (run_dir / "processed").exists()
    index = json.loads((run_dir / "decomposition_index.json").read_text(encoding="utf-8"))
    for phase, rel in index.items():
        assert (run_dir / rel).resolve() == (processed / f"{phase}_decomposition.npz").resolve()
        assert (run_dir / rel).exists()
    processed_manifest = json.loads((run_dir / "processed_outputs.json").read_text(encoding="utf-8"))
    assert len(processed_manifest["files"]) == 3
