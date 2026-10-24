from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "AorticTurb_Colab.ipynb"
RUNNER = ROOT / "notebooks" / "pass3_runner.py"
PINNED_PASS3_COMMIT = "f48d6564fa29231a432543f721ef0448f719c3b9"


def _notebook_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )


def test_pass3_notebook_is_thin_and_pinned():
    source = _notebook_source()
    assert f'AORTICTURB_COMMIT = "{PINNED_PASS3_COMMIT}"' in source
    assert "pass3_runner.py" in source
    assert "runner.run_pass3" in source
    assert "run_all(" not in source
    assert "run_kolmogorov" not in source
    assert "run_taylor" not in source
    assert "run_anisotropy" not in source


def test_pass3_notebook_forces_import_from_pinned_checkout():
    source = _notebook_source()
    assert "--force-reinstall" in source
    assert 'module_name == "aorticturb"' in source
    assert 'module_name.startswith("aorticturb.")' in source
    assert 'source_root = (WORKTREE / "src").resolve()' in source
    assert "sys.path.insert(0, str(source_root))" in source
    assert "importlib.invalidate_caches()" in source
    assert "loaded_package = Path(aorticturb.__file__).resolve()" in source
    assert "AORTICTURB_IMPORT_PROVENANCE_FAILED" in source


def test_pass3_runner_is_syntax_valid_and_scope_locked():
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "run_decomposition" in source
    assert "run_regime" in source
    assert "run_support_sensitivity" in source
    assert "run_all" not in source
    assert "run_kolmogorov" not in source
    assert "run_taylor" not in source
    assert "run_anisotropy" not in source
    assert '"canonical_05_results_promoted": False' in source
    assert '"legacy_global_fraction"' in source
    assert '"strict_mask_consistency"' in source
    assert '"fully_finite_local_windows"' in source
    assert '"strict_mask_and_local_windows"' in source


def test_pass3_runner_preserves_git_provenance_for_run_manifests():
    source = RUNNER.read_text(encoding="utf-8")
    assert "observed_commit != expected_commit" in source
    assert "os.chdir(repo_root)" in source
    assert 'run_manifest.get("git_commit") != expected_commit' in source
