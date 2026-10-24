from __future__ import annotations

import numpy as np


def realization_masks(fields: np.ndarray) -> np.ndarray:
    """Return finite U,V,W support for fields shaped (N, nodes, 3)."""
    fields = np.asarray(fields)
    if fields.ndim != 3 or fields.shape[-1] != 3:
        raise ValueError("Expected fields shaped (N_realizations, N_nodes, 3).")
    return np.isfinite(fields).all(axis=-1)


def common_support(fields: np.ndarray, policy: str = "intersection") -> np.ndarray:
    masks = realization_masks(fields)
    if policy == "intersection":
        return masks.all(axis=0)
    if policy == "union":
        return masks.any(axis=0)
    raise ValueError("support policy must be 'intersection' or 'union'")


def support_fraction(fields: np.ndarray, policy: str = "intersection") -> float:
    mask = common_support(fields, policy=policy)
    return float(mask.mean())
