import json
from pathlib import Path
import pytest

from aorticturb.config import load_config


def test_packaged_default_config_loads():
    cfg = load_config()
    assert cfg["analysis"]["phases"] == ["t018", "t021", "t024"]
    assert cfg["analysis"]["realizations_per_phase"] == 16


def test_locked_phase_override_is_rejected(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("analysis:\n  phases: [t021]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_root_and_packaged_dataset_manifests_are_semantically_equal():
    root_manifest = json.loads(Path("dataset_manifest.json").read_text(encoding="utf-8"))
    packaged = json.loads(Path("src/aorticturb/dataset_manifest.json").read_text(encoding="utf-8"))
    assert root_manifest == packaged
