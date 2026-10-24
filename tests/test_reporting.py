import json
from pathlib import Path

from aorticturb.reporting.run_manifest import create_run_directory, initialize_run, finalize_run


def test_finalized_run_manifest_hashes_outputs(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "SHA256SUMS.txt").write_text("# test ledger\n", encoding="utf-8")
    run_dir = create_run_directory(tmp_path / "out", "branch", run_id="run_test")
    initialize_run(run_dir, branch="branch", dataset_root=dataset, config={"x": 1}, methods="# Methods")
    (run_dir / "metrics.csv").write_text("a\n1\n", encoding="utf-8")
    finalize_run(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    paths = {item["path"] for item in manifest["outputs"]}
    assert "metrics.csv" in paths
    assert "config.json" in paths
    assert "input_hashes.txt" in paths
