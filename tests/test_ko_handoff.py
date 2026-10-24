import hashlib
import inspect
import json
from pathlib import Path

import pytest

from aorticturb.kolmogorov.driver import run_kolmogorov
from aorticturb.regime.ko_handoff import (
    LOCKED_NONPRIMARY_HANDOFF_POLICIES,
    resolve_ko_handoff,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regime_payload(*, t018=None, t021=None, t024=None) -> dict:
    return {
        "phases": {
            "t018": {"decision": "UNRESOLVED", "turbulence_established": t018},
            "t021": {"decision": "UNRESOLVED", "turbulence_established": t021},
            "t024": {"decision": "UNRESOLVED", "turbulence_established": t024},
        }
    }


def _projection(supported: bool) -> dict:
    if supported:
        return {
            "evaluable": True,
            "decision": "STOCHASTIC_MULTISCALE_SUPPORTED",
            "turbulence_established": True,
            "reason": "supported",
        }
    return {
        "evaluable": True,
        "decision": "UNRESOLVED",
        "turbulence_established": None,
        "reason": "unresolved",
    }


def _prepare_upstream(tmp_path: Path) -> tuple[Path, Path]:
    decomposition_dir = tmp_path / "decomposition"
    regime_dir = tmp_path / "regime"
    _write_json(decomposition_dir / "decomposition_summary.json", {"kind": "decomposition"})
    _write_json(regime_dir / "regime_summary.json", _regime_payload())
    _write_json(regime_dir / "config.json", {"kind": "config"})
    _write_json(regime_dir / "diagnostics.json", {"kind": "diagnostics"})
    return decomposition_dir, regime_dir


def _sensitivity_payload(
    decomposition_dir: Path,
    regime_dir: Path,
    *,
    relative_source_paths: bool = False,
) -> dict:
    phases = {}
    for phase, supported in (("t018", False), ("t021", True), ("t024", True)):
        phases[phase] = {
            "primary_gate_a": {
                "policy": "legacy_global_fraction",
                "decision": "UNRESOLVED",
                "turbulence_established": None,
                "unchanged_by_sensitivity_run": True,
            },
            "projected_gate_a_by_policy": {
                policy: _projection(supported)
                for policy in LOCKED_NONPRIMARY_HANDOFF_POLICIES
            },
        }
    source_decomposition = (
        decomposition_dir.name if relative_source_paths else str(decomposition_dir.resolve())
    )
    source_regime = regime_dir.name if relative_source_paths else str(regime_dir.resolve())
    return {
        "source_decomposition_run": source_decomposition,
        "source_regime_run": source_regime,
        "primary_policy_unchanged": "legacy_global_fraction",
        "policies_are_sensitivity_comparators_only": True,
        "canonical_comparator_set_locked": True,
        "phases": phases,
    }


def _write_sensitivity_run(
    sensitivity_dir: Path,
    decomposition_dir: Path,
    regime_dir: Path,
    *,
    relative_source_paths: bool = False,
) -> None:
    _write_json(
        sensitivity_dir / "support_sensitivity.json",
        _sensitivity_payload(
            decomposition_dir,
            regime_dir,
            relative_source_paths=relative_source_paths,
        ),
    )
    sources = (
        decomposition_dir / "decomposition_summary.json",
        regime_dir / "config.json",
        regime_dir / "diagnostics.json",
        regime_dir / "regime_summary.json",
    )
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    (sensitivity_dir / "input_hashes.txt").write_text(
        "".join(f"{_sha256(path)}  {path}\n" for path in sources),
        encoding="utf-8",
    )


def test_primary_gate_a_behavior_is_backward_compatible(tmp_path):
    decomposition_dir = tmp_path / "decomposition"
    regime_dir = tmp_path / "regime"
    _write_json(
        regime_dir / "regime_summary.json",
        _regime_payload(t018=True, t021=None, t024=False),
    )

    handoff = resolve_ko_handoff(decomposition_dir, regime_dir)

    assert handoff["t018"].ko_authorized is True
    assert handoff["t018"].source == "primary_gate_a"
    assert handoff["t021"].ko_authorized is False
    assert handoff["t024"].ko_authorized is False


def test_locked_sensitivity_concordance_authorizes_t021_t024_only(tmp_path):
    decomposition_dir, regime_dir = _prepare_upstream(tmp_path)
    sensitivity_dir = tmp_path / "sensitivity"
    _write_sensitivity_run(sensitivity_dir, decomposition_dir, regime_dir)

    handoff = resolve_ko_handoff(decomposition_dir, regime_dir, sensitivity_dir)

    assert handoff["t018"].ko_authorized is False
    assert handoff["t021"].ko_authorized is True
    assert handoff["t024"].ko_authorized is True
    assert handoff["t021"].source == "locked_support_sensitivity_concordance"
    assert handoff["t021"].primary_turbulence_established is None
    assert tuple(handoff["t021"].sensitivity_policies) == LOCKED_NONPRIMARY_HANDOFF_POLICIES


def test_discordant_locked_comparator_fails_closed(tmp_path):
    decomposition_dir, regime_dir = _prepare_upstream(tmp_path)
    sensitivity_dir = tmp_path / "sensitivity"
    _write_sensitivity_run(sensitivity_dir, decomposition_dir, regime_dir)
    payload = json.loads((sensitivity_dir / "support_sensitivity.json").read_text())
    policy = LOCKED_NONPRIMARY_HANDOFF_POLICIES[0]
    payload["phases"]["t021"]["projected_gate_a_by_policy"][policy] = _projection(False)
    _write_json(sensitivity_dir / "support_sensitivity.json", payload)

    handoff = resolve_ko_handoff(decomposition_dir, regime_dir, sensitivity_dir)

    assert handoff["t021"].ko_authorized is False
    assert "do not concordantly establish" in handoff["t021"].reason


def test_mismatched_decomposition_is_rejected_by_input_hash(tmp_path):
    decomposition_dir, regime_dir = _prepare_upstream(tmp_path)
    sensitivity_dir = tmp_path / "sensitivity"
    _write_sensitivity_run(sensitivity_dir, decomposition_dir, regime_dir)

    other_decomposition = tmp_path / "other_decomposition"
    _write_json(
        other_decomposition / "decomposition_summary.json",
        {"kind": "different decomposition"},
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        resolve_ko_handoff(other_decomposition, regime_dir, sensitivity_dir)


def test_mismatched_regime_is_rejected_by_input_hash(tmp_path):
    decomposition_dir, regime_dir = _prepare_upstream(tmp_path)
    sensitivity_dir = tmp_path / "sensitivity"
    _write_sensitivity_run(sensitivity_dir, decomposition_dir, regime_dir)

    _write_json(regime_dir / "diagnostics.json", {"kind": "changed diagnostics"})

    with pytest.raises(ValueError, match="provenance mismatch"):
        resolve_ko_handoff(decomposition_dir, regime_dir, sensitivity_dir)


def test_relative_recorded_source_paths_do_not_depend_on_current_cwd(tmp_path, monkeypatch):
    decomposition_dir, regime_dir = _prepare_upstream(tmp_path)
    sensitivity_dir = tmp_path / "sensitivity"
    _write_sensitivity_run(
        sensitivity_dir,
        decomposition_dir,
        regime_dir,
        relative_source_paths=True,
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    handoff = resolve_ko_handoff(decomposition_dir, regime_dir, sensitivity_dir)

    assert handoff["t021"].ko_authorized is True
    assert handoff["t024"].ko_authorized is True


def test_public_ko_driver_exposes_explicit_sensitivity_handoff():
    signature = inspect.signature(run_kolmogorov)
    assert "support_sensitivity_dir" in signature.parameters
