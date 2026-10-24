from __future__ import annotations

from dataclasses import dataclass, asdict

from ..constants import ALLOWED_CLASSIFICATIONS


@dataclass(frozen=True)
class Classification:
    label: str
    rationale: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def classify(
    *,
    turbulence_established: bool | None,
    ko_resolved: bool | None,
    ko_supported: bool | None,
    adequate_power: bool = True,
) -> Classification:
    """Conservative terminal decision logic.

    None denotes unresolved evidence. This function intentionally cannot infer turbulence
    from KO or Taylor diagnostics.
    """
    if turbulence_established is None or not adequate_power:
        result = Classification(
            "UNRESOLVED_AT_AVAILABLE_RESOLUTION",
            ["Turbulence status or statistical power is unresolved."],
        )
    elif turbulence_established is False:
        result = Classification(
            "LAMINAR_UNSTEADY",
            ["No defensible stochastic multiscale turbulent component was established."],
        )
    elif ko_resolved is not True:
        result = Classification(
            "UNRESOLVED_AT_AVAILABLE_RESOLUTION",
            ["Turbulence is established but KO diagnostics are not sufficiently resolved."],
        )
    elif ko_supported is True:
        result = Classification(
            "KO_TURBULENCE",
            ["Turbulence is established and admissible KO diagnostics are concordant."],
        )
    else:
        result = Classification(
            "NON_KO_TURBULENCE",
            ["Turbulence is established and adequately resolved KO predictions fail systematically."],
        )

    if result.label not in ALLOWED_CLASSIFICATIONS:
        raise RuntimeError("Classification escaped the contract.")
    return result
