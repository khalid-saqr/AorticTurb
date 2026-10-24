import hashlib
import json
from pathlib import Path

from aorticturb.reporting.run_manifest import create_run_directory, initialize_run, finalize_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analysis_run_uses_canonical_workspace_tree(tmp_path: Path):
    run_dir = create_run_directory(tmp_path, "02_kolmogorov_obukhov", run_id="run_test")
    assert run_dir == tmp_path / "04_analysis" / "02_kolmogorov_obukhov" / "run_test"


def test_results_run_uses_canonical_workspace_tree(tmp_path: Path):
    run_dir = create_run_directory(tmp_path, "05_results", run_id="run_test")
    assert run_dir == tmp_path / "05_results" / "run_test"


def test_finalized_run_manifest_hashes_final_outputs(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "SHA256SUMS.txt").write_text("# test ledger\n", encoding="utf-8")
    run_dir = create_run_directory(tmp_path / "workspace", "02_kolmogorov_obukhov", run_id="run_test")
    initialize_run(run_dir, branch="02_kolmogorov_obukhov", dataset_root=dataset, config={"x": 1}, methods="# Methods")
    (run_dir / "metrics.csv").write_text("a\n1\n", encoding="utf-8")
    finalize_run(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    records = {item["path"]: item for item in manifest["outputs"]}
    for relpath, record in records.items():
        assert record["sha256"] == _sha(run_dir / relpath)
    assert "metrics.csv" in records
    assert "logs/run.log" in records
