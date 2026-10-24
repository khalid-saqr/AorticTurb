import inspect
import json
from pathlib import Path

from aorticturb.kolmogorov.driver import run_kolmogorov
from aorticturb.regime.ko_handoff import resolve_ko_handoff


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _regime_payload(*, t018=None, t021=None, t024=None) -> dict:
    return {
        "phases": {
            "t018": {"decision": "UNRESOLVED", "turbulence_established": t018},
            "t021": {
                "decision": (
                    "STOCHASTIC_MULTISCALE_SUPPORTED" if t021 is True else "UNRESOLVED"
                ),
                "turbulence_established": t021,
            },
            "t024": {"decision": "UNRESOLVED", "turbulence_established": t024},
        }
    }


def test_primary_gate_a_is_the_only_ko_authorization(tmp_path):
    regime_dir = tmp_path / "regime"
    decomposition_dir = tmp_path / "decomposition"
    _write_json(
        regime_dir / "regime_summary.json",
        _regime_payload(t018=False, t021=True, t024=None),
    )

    handoff = resolve_ko_handoff(decomposition_dir, regime_dir)

    assert handoff["t018"].ko_authorized is False
    assert handoff["t021"].ko_authorized is True
    assert handoff["t024"].ko_authorized is False
    assert all(value.source == "primary_gate_a" for value in handoff.values())


def test_sensitivity_argument_cannot_open_unresolved_gate_a(tmp_path):
    regime_dir = tmp_path / "regime"
    decomposition_dir = tmp_path / "decomposition"
    sensitivity_dir = tmp_path / "sensitivity"
    _write_json(regime_dir / "regime_summary.json", _regime_payload())
    sensitivity_dir.mkdir()

    handoff = resolve_ko_handoff(
        decomposition_dir,
        regime_dir,
        support_sensitivity_dir=sensitivity_dir,
    )

    assert all(not value.ko_authorized for value in handoff.values())
    assert all(value.source == "primary_gate_a" for value in handoff.values())


def test_public_ko_driver_keeps_backward_compatible_sensitivity_argument():
    signature = inspect.signature(run_kolmogorov)
    assert "support_sensitivity_dir" in signature.parameters
