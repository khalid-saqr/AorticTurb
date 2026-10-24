from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from ..constants import PHASES


@dataclass(frozen=True)
class GateAHandoff:
    phase: str
    ko_authorized: bool
    source: str
    primary_decision: str | None
    primary_turbulence_established: bool | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def resolve_ko_handoff(
    decomposition_dir: str | Path,
    regime_dir: str | Path,
    support_sensitivity_dir: str | Path | None = None,
) -> dict[str, GateAHandoff]:
    """Authorize KO only from the formal primary Gate-A result.

    ``decomposition_dir`` and ``support_sensitivity_dir`` are accepted for backward API
    compatibility with PR #8, but support-sensitivity outputs have no authorization effect.
    A phase enters KO only when the supplied primary Gate-A run records
    ``turbulence_established=True``.
    """
    _ = Path(decomposition_dir)
    _ = support_sensitivity_dir
    regime_dir = Path(regime_dir).resolve()
    regime_path = regime_dir / "regime_summary.json"
    if not regime_path.is_file():
        raise FileNotFoundError(regime_path)
    regime = _load_object(regime_path)
    regime_phases = regime.get("phases")
    if not isinstance(regime_phases, dict):
        raise ValueError("regime_summary.json must contain a 'phases' object")

    resolved: dict[str, GateAHandoff] = {}
    for phase in PHASES:
        primary = regime_phases.get(phase)
        if not isinstance(primary, dict):
            raise ValueError(f"Missing primary Gate-A result for {phase}")
        primary_decision = primary.get("decision")
        primary_turbulence = primary.get("turbulence_established")
        if primary_turbulence not in (True, False, None):
            raise ValueError(
                f"Invalid turbulence_established value for {phase}: {primary_turbulence!r}"
            )

        if primary_turbulence is True:
            reason = "Primary Gate A established stochastic multiscale turbulence."
            authorized = True
        elif primary_turbulence is False:
            reason = "Primary Gate A did not establish turbulence."
            authorized = False
        else:
            reason = "Primary Gate A is unresolved."
            authorized = False

        resolved[phase] = GateAHandoff(
            phase=phase,
            ko_authorized=authorized,
            source="primary_gate_a",
            primary_decision=primary_decision,
            primary_turbulence_established=primary_turbulence,
            reason=reason,
        )

    return resolved
