from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from ..constants import PHASES
from .support_sensitivity import (
    FULLY_FINITE_LOCAL_WINDOWS,
    LEGACY_GLOBAL_FRACTION,
    STRICT_MASK_AND_LOCAL_WINDOWS,
    STRICT_MASK_CONSISTENCY,
)


STOCHASTIC_MULTISCALE_SUPPORTED = "STOCHASTIC_MULTISCALE_SUPPORTED"
LOCKED_NONPRIMARY_HANDOFF_POLICIES = (
    STRICT_MASK_CONSISTENCY,
    FULLY_FINITE_LOCAL_WINDOWS,
    STRICT_MASK_AND_LOCAL_WINDOWS,
)


@dataclass(frozen=True)
class GateAHandoff:
    phase: str
    ko_authorized: bool
    source: str
    primary_decision: str | None
    primary_turbulence_established: bool | None
    sensitivity_run: str | None
    sensitivity_policies: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sensitivity_policies"] = list(self.sensitivity_policies)
        return payload


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _same_path(recorded: str, actual: Path) -> bool:
    return Path(recorded).expanduser().resolve() == actual.expanduser().resolve()


def resolve_ko_handoff(
    regime_dir: str | Path,
    support_sensitivity_dir: str | Path | None = None,
) -> dict[str, GateAHandoff]:
    """Resolve the explicit upstream authorization for entering KO analysis.

    The primary Gate-A result is never modified. A phase is admitted directly when the
    primary Gate-A run records ``turbulence_established=True``. Otherwise, a caller may
    explicitly supply the saved support-sensitivity run. That handoff is accepted only when
    all three locked non-primary support comparators are evaluable and concordantly project
    ``STOCHASTIC_MULTISCALE_SUPPORTED`` with ``turbulence_established=True``.

    This is a provenance-bearing handoff rule, not a replacement primary classification.
    Missing, partial, mismatched, or discordant sensitivity evidence fails closed.
    """
    regime_dir = Path(regime_dir).resolve()
    regime_path = regime_dir / "regime_summary.json"
    if not regime_path.is_file():
        raise FileNotFoundError(regime_path)
    regime = _load_object(regime_path)
    regime_phases = regime.get("phases")
    if not isinstance(regime_phases, dict):
        raise ValueError("regime_summary.json must contain a 'phases' object")

    sensitivity: dict[str, Any] | None = None
    sensitivity_dir: Path | None = None
    if support_sensitivity_dir is not None:
        sensitivity_dir = Path(support_sensitivity_dir).resolve()
        sensitivity_path = sensitivity_dir / "support_sensitivity.json"
        if not sensitivity_path.is_file():
            raise FileNotFoundError(sensitivity_path)
        sensitivity = _load_object(sensitivity_path)

        source_regime = sensitivity.get("source_regime_run")
        if not isinstance(source_regime, str) or not _same_path(source_regime, regime_dir):
            raise ValueError(
                "Support-sensitivity provenance mismatch: source_regime_run does not match "
                f"the supplied regime directory {regime_dir}."
            )
        if sensitivity.get("primary_policy_unchanged") != LEGACY_GLOBAL_FRACTION:
            raise ValueError("Support-sensitivity run does not preserve the legacy primary policy.")
        if sensitivity.get("policies_are_sensitivity_comparators_only") is not True:
            raise ValueError("Support-sensitivity run is missing comparator-only provenance.")
        if sensitivity.get("canonical_comparator_set_locked") is not True:
            raise ValueError("Support-sensitivity comparator set is not recorded as locked.")
        if not isinstance(sensitivity.get("phases"), dict):
            raise ValueError("support_sensitivity.json must contain a 'phases' object")

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
            resolved[phase] = GateAHandoff(
                phase=phase,
                ko_authorized=True,
                source="primary_gate_a",
                primary_decision=primary_decision,
                primary_turbulence_established=True,
                sensitivity_run=str(sensitivity_dir) if sensitivity_dir is not None else None,
                sensitivity_policies=(),
                reason="Primary Gate A established turbulence.",
            )
            continue

        # A definitive primary non-turbulent result is never overridden by sensitivity.
        if primary_turbulence is False:
            resolved[phase] = GateAHandoff(
                phase=phase,
                ko_authorized=False,
                source="primary_gate_a",
                primary_decision=primary_decision,
                primary_turbulence_established=False,
                sensitivity_run=str(sensitivity_dir) if sensitivity_dir is not None else None,
                sensitivity_policies=(),
                reason="Primary Gate A did not establish turbulence.",
            )
            continue

        if sensitivity is None or sensitivity_dir is None:
            resolved[phase] = GateAHandoff(
                phase=phase,
                ko_authorized=False,
                source="primary_gate_a",
                primary_decision=primary_decision,
                primary_turbulence_established=None,
                sensitivity_run=None,
                sensitivity_policies=(),
                reason=(
                    "Primary Gate A is unresolved and no explicit support-sensitivity run "
                    "was supplied for the KO handoff."
                ),
            )
            continue

        phase_sensitivity = sensitivity["phases"].get(phase)
        if not isinstance(phase_sensitivity, dict):
            raise ValueError(f"Missing support-sensitivity result for {phase}")
        recorded_primary = phase_sensitivity.get("primary_gate_a")
        if not isinstance(recorded_primary, dict):
            raise ValueError(f"Missing primary_gate_a provenance for {phase}")
        if (
            recorded_primary.get("decision") != primary_decision
            or recorded_primary.get("turbulence_established") != primary_turbulence
            or recorded_primary.get("unchanged_by_sensitivity_run") is not True
        ):
            raise ValueError(
                f"Support-sensitivity primary Gate-A provenance does not match {phase}."
            )

        projected = phase_sensitivity.get("projected_gate_a_by_policy")
        if not isinstance(projected, dict):
            raise ValueError(f"Missing projected_gate_a_by_policy for {phase}")

        policy_results: list[dict[str, Any]] = []
        for policy in LOCKED_NONPRIMARY_HANDOFF_POLICIES:
            result = projected.get(policy)
            if not isinstance(result, dict):
                raise ValueError(f"Missing locked sensitivity comparator {policy!r} for {phase}")
            policy_results.append(result)

        concordant = all(
            result.get("evaluable") is True
            and result.get("decision") == STOCHASTIC_MULTISCALE_SUPPORTED
            and result.get("turbulence_established") is True
            for result in policy_results
        )

        if concordant:
            resolved[phase] = GateAHandoff(
                phase=phase,
                ko_authorized=True,
                source="locked_support_sensitivity_concordance",
                primary_decision=primary_decision,
                primary_turbulence_established=None,
                sensitivity_run=str(sensitivity_dir),
                sensitivity_policies=LOCKED_NONPRIMARY_HANDOFF_POLICIES,
                reason=(
                    "All locked non-primary support comparators concordantly project "
                    "STOCHASTIC_MULTISCALE_SUPPORTED; primary Gate-A record remains unchanged."
                ),
            )
        else:
            resolved[phase] = GateAHandoff(
                phase=phase,
                ko_authorized=False,
                source="locked_support_sensitivity_concordance",
                primary_decision=primary_decision,
                primary_turbulence_established=None,
                sensitivity_run=str(sensitivity_dir),
                sensitivity_policies=LOCKED_NONPRIMARY_HANDOFF_POLICIES,
                reason=(
                    "Locked non-primary support comparators do not concordantly establish "
                    "stochastic multiscale motion."
                ),
            )

    return resolved
