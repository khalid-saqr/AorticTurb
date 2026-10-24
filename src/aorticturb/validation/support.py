from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class SupportAudit:
    """Describe measurement support without conflating mask occupancy with mask agreement.

    Fractions are dimensionless. ``intersection_fraction_total`` and
    ``union_fraction_total`` use the complete deposited Cartesian lattice as denominator.
    ``intersection_over_union_fraction`` instead measures realization-to-realization mask
    agreement on the measured support and is therefore distinct from anatomical/domain
    occupancy inside the enclosing lattice.
    """

    realizations: int
    total_nodes: int
    per_realization_valid_nodes: list[int]
    intersection_nodes: int
    union_nodes: int
    disagreement_nodes: int
    intersection_fraction_total: float
    union_fraction_total: float
    intersection_over_union_fraction: float | None
    masks_identical: bool

    def to_dict(self) -> dict:
        return asdict(self)


def audit_realization_masks(masks: np.ndarray) -> SupportAudit:
    """Audit finite-velocity masks shaped ``(N_realizations, N_nodes)``.

    This routine is deliberately classification-neutral. It records the quantities needed
    to distinguish sparse anatomical/measurement occupancy from inconsistent support across
    realizations. It does not decide whether Gate A should pass.
    """

    arr = np.asarray(masks, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("masks must have shape (N_realizations, N_nodes)")
    if arr.shape[0] < 1:
        raise ValueError("at least one realization is required")
    if arr.shape[1] < 1:
        raise ValueError("at least one grid node is required")

    intersection = arr.all(axis=0)
    union = arr.any(axis=0)
    intersection_nodes = int(intersection.sum())
    union_nodes = int(union.sum())
    total_nodes = int(arr.shape[1])
    disagreement_nodes = union_nodes - intersection_nodes
    agreement = (
        float(intersection_nodes / union_nodes)
        if union_nodes > 0
        else None
    )

    return SupportAudit(
        realizations=int(arr.shape[0]),
        total_nodes=total_nodes,
        per_realization_valid_nodes=[int(v) for v in arr.sum(axis=1)],
        intersection_nodes=intersection_nodes,
        union_nodes=union_nodes,
        disagreement_nodes=disagreement_nodes,
        intersection_fraction_total=float(intersection_nodes / total_nodes),
        union_fraction_total=float(union_nodes / total_nodes),
        intersection_over_union_fraction=agreement,
        masks_identical=bool(np.all(arr == arr[0][None, :])),
    )
