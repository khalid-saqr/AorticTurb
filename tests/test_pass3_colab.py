from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "AorticTurb_Colab.ipynb"
PINNED_SCIENTIFIC_COMMIT = "4be7c4ca4c2fc62e54008c65ad275dfd9b5945e5"


def _payload() -> dict:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    return payload


def _code_source() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _payload()["cells"]
        if cell.get("cell_type") == "code"
    )


def test_all_notebook_code_cells_are_syntax_valid():
    for cell in _payload()["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))


def test_rebuilt_notebook_uses_isolated_pinned_public_install():
    source = _code_source()
    assert f'AORTICTURB_COMMIT = "{PINNED_SCIENTIFIC_COMMIT}"' in source
    assert "git+https://github.com/khalid-saqr/AorticTurb.git@" in source
    assert 'Path("/content/aorticturb-pass3-venv")' in source
    assert '"-m", "venv", "--system-site-packages"' in source
    assert "direct_url.json" in source
    assert "INSTALLED_COMMIT != AORTICTURB_COMMIT" in source


def test_rebuilt_notebook_has_no_old_self_checkout_or_kernel_import_machinery():
    source = _code_source()
    forbidden = (
        "git clone",
        "WORKTREE",
        "os.chdir",
        "sys.modules",
        "pass3_runner",
        "import aorticturb",
        "run_all(",
    )
    for token in forbidden:
        assert token not in source


def test_rebuilt_notebook_scope_is_gate_decomposition_regime_sensitivity_only():
    source = _code_source()
    assert 'run_cli("gate", "--root", DATASET_ROOT)' in source
    assert '"decompose"' in source
    assert '"regime"' in source
    assert '"support-sensitivity"' in source
    assert '"kolmogorov"' not in source
    assert '"taylor"' not in source
    assert '"anisotropy"' not in source
    assert '"diagnostics"' not in source
    assert '"classify"' not in source


def test_rebuilt_notebook_validates_canonical_phases_support_and_locked_policies():
    source = _code_source()
    for phase in ("t018", "t021", "t024"):
        assert phase in source
    for column in (
        "union_support_nodes",
        "intersection_over_union_fraction",
        "masks_identical",
    ):
        assert column in source
    for policy in (
        "legacy_global_fraction",
        "strict_mask_consistency",
        "fully_finite_local_windows",
        "strict_mask_and_local_windows",
    ):
        assert policy in source
    assert 'set(decomposition_metrics["realizations"].astype(int)) != {16}' in source


def test_rebuilt_notebook_writes_analysis_workspace_not_root_results():
    source = _code_source()
    assert 'WORKSPACE_ROOT = PROJECT_ROOT / "results"' in source
    assert '"--output", WORKSPACE_ROOT' in source
    assert 'canonical_root_05_results_written_by_notebook": False' in source
