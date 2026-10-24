import numpy as np

from aorticturb.regime.gate_a import evaluate_gate_a


def test_gate_a_readiness_is_not_a_turbulence_detector():
    mean = np.ones((100, 3), dtype=float)
    residuals = np.zeros((16, 100, 3), dtype=float)
    evidence = evaluate_gate_a(mean, residuals, support_fraction_value=1.0)
    assert evidence.decision == "PROCEED_TO_MULTISCALE_DIAGNOSTICS"
    assert evidence.fluctuation_ratio_median == 0.0


def test_gate_a_insufficient_realizations_is_unresolved():
    mean = np.ones((20, 3), dtype=float)
    residuals = np.zeros((4, 20, 3), dtype=float)
    evidence = evaluate_gate_a(mean, residuals, support_fraction_value=1.0)
    assert evidence.decision == "UNRESOLVED"
    assert evidence.reasons
