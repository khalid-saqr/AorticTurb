import numpy as np
import pytest

from aorticturb.regime.gate_a import (
    LEGACY_SUPPORT_FRACTION_DEFINITION,
    evaluate_gate_a,
)


def test_gate_a_readiness_is_not_a_turbulence_detector():
    mean = np.ones((100, 3), dtype=float)
    residuals = np.zeros((16, 100, 3), dtype=float)
    evidence = evaluate_gate_a(mean, residuals, support_fraction_value=1.0)
    assert evidence.decision == "PROCEED_TO_MULTISCALE_DIAGNOSTICS"
    assert evidence.fluctuation_ratio_median == 0.0
    assert evidence.support_fraction_definition == LEGACY_SUPPORT_FRACTION_DEFINITION
    assert evidence.support_gate_policy == "legacy_global_fraction"


def test_gate_a_insufficient_realizations_is_unresolved():
    mean = np.ones((20, 3), dtype=float)
    residuals = np.zeros((4, 20, 3), dtype=float)
    evidence = evaluate_gate_a(mean, residuals, support_fraction_value=1.0)
    assert evidence.decision == "UNRESOLVED"
    assert evidence.reasons


def test_gate_a_preserves_legacy_global_fraction_decision():
    mean = np.ones((100, 3), dtype=float)
    residuals = np.zeros((16, 100, 3), dtype=float)
    evidence = evaluate_gate_a(
        mean,
        residuals,
        support_fraction_value=0.397,
        min_support_fraction=0.5,
    )
    assert evidence.decision == "UNRESOLVED"
    assert evidence.support_fraction == pytest.approx(0.397)
    assert "total_cartesian_lattice_nodes" in evidence.reasons[0]


def test_gate_a_does_not_enable_unreviewed_support_policy():
    mean = np.ones((100, 3), dtype=float)
    residuals = np.zeros((16, 100, 3), dtype=float)
    with pytest.raises(ValueError):
        evaluate_gate_a(
            mean,
            residuals,
            support_fraction_value=0.397,
            support_gate_policy="local_window",
        )
