import hashlib
import json
from pathlib import Path

from aorticturb.reporting.run_manifest import create_run_directory, initialize_run, finalize_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_finalized_run_manifest_hashes_final_outputs(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "SHA256SUMS.txt").write_text("# test ledger\n", encoding="utf-8")
    run_dir = create_run_directory(tmp_path / "out", "branch", run_id="run_test")
    initialize_run(run_dir, branch="branch", dataset_root=dataset, config={"x": 1}, methods="# Methods")
    (run_dir / "metrics.csv").write_text("a\n1\n", encoding="utf-8")
    finalize_run(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    records = {item["path"]: item for item in manifest["outputs"]}
    for relpath, record in records.items():
        assert record["sha256"] == _sha(run_dir / relpath)
    assert "metrics.csv" in records
    assert "logs/run.log" in records
