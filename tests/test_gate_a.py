import numpy as np
import pytest

from aorticturb.regime.gate_a import (
    LEGACY_SUPPORT_FRACTION_DEFINITION,
    PRIMARY_SUPPORT_GATE_POLICY,
    evaluate_gate_a,
)


def _evaluate(**overrides):
    mean = np.ones((100, 3), dtype=float)
    residuals = np.zeros((16, 100, 3), dtype=float)
    kwargs = {
        "support_fraction_value": 0.397,
        "intersection_over_union_fraction": 1.0,
        "fully_finite_window_count": 3,
        "local_window_min_valid_fraction": 1.0,
    }
    kwargs.update(overrides)
    return evaluate_gate_a(mean, residuals, **kwargs)


def test_gate_a_readiness_is_not_a_turbulence_detector():
    evidence = _evaluate()
    assert evidence.decision == "PROCEED_TO_MULTISCALE_DIAGNOSTICS"
    assert evidence.fluctuation_ratio_median == 0.0
    assert evidence.support_fraction_definition == LEGACY_SUPPORT_FRACTION_DEFINITION
    assert evidence.support_gate_policy == PRIMARY_SUPPORT_GATE_POLICY


def test_gate_a_insufficient_realizations_is_unresolved():
    mean = np.ones((20, 3), dtype=float)
    residuals = np.zeros((4, 20, 3), dtype=float)
    evidence = evaluate_gate_a(
        mean,
        residuals,
        support_fraction_value=0.397,
        intersection_over_union_fraction=1.0,
        fully_finite_window_count=1,
        local_window_min_valid_fraction=1.0,
    )
    assert evidence.decision == "UNRESOLVED"
    assert evidence.reasons


def test_whole_lattice_occupancy_does_not_block_identical_masks_and_valid_windows():
    evidence = _evaluate(support_fraction_value=0.397)
    assert evidence.decision == "PROCEED_TO_MULTISCALE_DIAGNOSTICS"
    assert evidence.support_fraction == pytest.approx(0.397)


def test_gate_a_requires_exact_mask_consistency():
    evidence = _evaluate(intersection_over_union_fraction=0.999)
    assert evidence.decision == "UNRESOLVED"
    assert "not exactly consistent" in evidence.reasons[0]


def test_gate_a_requires_a_fully_finite_local_window():
    evidence = _evaluate(
        fully_finite_window_count=0,
        local_window_min_valid_fraction=None,
    )
    assert evidence.decision == "UNRESOLVED"
    assert "fully finite local windows" in evidence.reasons[-1]


def test_gate_a_rejects_noncanonical_support_policy():
    with pytest.raises(ValueError):
        _evaluate(support_gate_policy="legacy_global_fraction")
