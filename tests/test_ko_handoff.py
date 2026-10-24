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


def _sensitivity_payload(regime_dir: Path) -> dict:
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
    return {
        "source_regime_run": str(regime_dir.resolve()),
        "primary_policy_unchanged": "legacy_global_fraction",
        "policies_are_sensitivity_comparators_only": True,
        "canonical_comparator_set_locked": True,
        "phases": phases,
    }


def test_primary_gate_a_behavior_is_backward_compatible(tmp_path):
    regime_dir = tmp_path / "regime"
    _write_json(
        regime_dir / "regime_summary.json",
        _regime_payload(t018=True, t021=None, t024=False),
    )

    handoff = resolve_ko_handoff(regime_dir)

    assert handoff["t018"].ko_authorized is True
    assert handoff["t018"].source == "primary_gate_a"
    assert handoff["t021"].ko_authorized is False
    assert handoff["t024"].ko_authorized is False


def test_locked_sensitivity_concordance_authorizes_t021_t024_only(tmp_path):
    regime_dir = tmp_path / "regime"
    sensitivity_dir = tmp_path / "sensitivity"
    _write_json(regime_dir / "regime_summary.json", _regime_payload())
    _write_json(
        sensitivity_dir / "support_sensitivity.json",
        _sensitivity_payload(regime_dir),
    )

    handoff = resolve_ko_handoff(regime_dir, sensitivity_dir)

    assert handoff["t018"].ko_authorized is False
    assert handoff["t021"].ko_authorized is True
    assert handoff["t024"].ko_authorized is True
    assert handoff["t021"].source == "locked_support_sensitivity_concordance"
    assert handoff["t021"].primary_turbulence_established is None
    assert tuple(handoff["t021"].sensitivity_policies) == LOCKED_NONPRIMARY_HANDOFF_POLICIES


def test_discordant_locked_comparator_fails_closed(tmp_path):
    regime_dir = tmp_path / "regime"
    sensitivity_dir = tmp_path / "sensitivity"
    _write_json(regime_dir / "regime_summary.json", _regime_payload())
    payload = _sensitivity_payload(regime_dir)
    policy = LOCKED_NONPRIMARY_HANDOFF_POLICIES[0]
    payload["phases"]["t021"]["projected_gate_a_by_policy"][policy] = _projection(False)
    _write_json(sensitivity_dir / "support_sensitivity.json", payload)

    handoff = resolve_ko_handoff(regime_dir, sensitivity_dir)

    assert handoff["t021"].ko_authorized is False
    assert "do not concordantly establish" in handoff["t021"].reason


def test_mismatched_sensitivity_source_regime_is_rejected(tmp_path):
    regime_dir = tmp_path / "regime"
    sensitivity_dir = tmp_path / "sensitivity"
    _write_json(regime_dir / "regime_summary.json", _regime_payload())
    payload = _sensitivity_payload(regime_dir)
    payload["source_regime_run"] = str((tmp_path / "different_regime").resolve())
    _write_json(sensitivity_dir / "support_sensitivity.json", payload)

    with pytest.raises(ValueError, match="provenance mismatch"):
        resolve_ko_handoff(regime_dir, sensitivity_dir)


def test_public_ko_driver_exposes_explicit_sensitivity_handoff():
    signature = inspect.signature(run_kolmogorov)
    assert "support_sensitivity_dir" in signature.parameters
