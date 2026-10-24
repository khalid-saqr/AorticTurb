import numpy as np
import pytest

from aorticturb.validation.support import audit_realization_masks


def test_support_audit_distinguishes_sparse_occupancy_from_perfect_mask_agreement():
    masks = np.zeros((3, 10), dtype=bool)
    masks[:, :4] = True

    audit = audit_realization_masks(masks)

    assert audit.total_nodes == 10
    assert audit.intersection_nodes == 4
    assert audit.union_nodes == 4
    assert audit.disagreement_nodes == 0
    assert audit.intersection_fraction_total == pytest.approx(0.4)
    assert audit.union_fraction_total == pytest.approx(0.4)
    assert audit.intersection_over_union_fraction == pytest.approx(1.0)
    assert audit.masks_identical is True
    assert audit.per_realization_valid_nodes == [4, 4, 4]


def test_support_audit_detects_realization_dependent_missing_support():
    masks = np.array(
        [
            [True, True, False, False],
            [True, False, True, False],
            [True, True, True, False],
        ],
        dtype=bool,
    )

    audit = audit_realization_masks(masks)

    assert audit.intersection_nodes == 1
    assert audit.union_nodes == 3
    assert audit.disagreement_nodes == 2
    assert audit.intersection_fraction_total == pytest.approx(0.25)
    assert audit.union_fraction_total == pytest.approx(0.75)
    assert audit.intersection_over_union_fraction == pytest.approx(1.0 / 3.0)
    assert audit.masks_identical is False
    assert audit.per_realization_valid_nodes == [2, 2, 3]


def test_support_audit_rejects_non_matrix_input():
    with pytest.raises(ValueError):
        audit_realization_masks(np.ones(10, dtype=bool))
