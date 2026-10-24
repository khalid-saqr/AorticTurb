import pytest

from aorticturb.regime.support_sensitivity import (
    FULLY_FINITE_LOCAL_WINDOWS,
    LEGACY_GLOBAL_FRACTION,
    STRICT_MASK_AND_LOCAL_WINDOWS,
    STRICT_MASK_CONSISTENCY,
    evaluate_support_policy_sensitivity,
    project_gate_a_decision,
)
from aorticturb.regime.sensitivity_driver import _locked_sensitivity_config


def _evaluate(**overrides):
    kwargs = {
        "realizations": 16,
        "min_realizations": 12,
        "common_intersection_fraction_total": 0.397,
        "min_common_support_fraction": 0.50,
        "intersection_over_union_fraction": 1.0,
        "fully_finite_window_count": 3,
        "local_window_min_valid_fraction": 1.0,
        "min_fully_finite_windows": 1,
        "required_local_window_valid_fraction": 1.0,
    }
    kwargs.update(overrides)
    return evaluate_support_policy_sensitivity(**kwargs)


def test_sparse_identical_support_is_policy_sensitive_without_selecting_winner():
    result = _evaluate()

    assert result.primary_policy == LEGACY_GLOBAL_FRACTION
    assert result.policies[LEGACY_GLOBAL_FRACTION].ready is False
    assert result.policies[STRICT_MASK_CONSISTENCY].ready is True
    assert result.policies[FULLY_FINITE_LOCAL_WINDOWS].ready is True
    assert result.policies[STRICT_MASK_AND_LOCAL_WINDOWS].ready is True
    assert result.readiness_policy_sensitive is True


def test_realization_dependent_mask_disagreement_fails_strict_mask_policies():
    result = _evaluate(intersection_over_union_fraction=0.92)

    assert result.policies[STRICT_MASK_CONSISTENCY].ready is False
    assert result.policies[FULLY_FINITE_LOCAL_WINDOWS].ready is True
    assert result.policies[STRICT_MASK_AND_LOCAL_WINDOWS].ready is False


def test_missing_mask_audit_is_not_silently_interpreted():
    result = _evaluate(intersection_over_union_fraction=None)

    mask = result.policies[STRICT_MASK_CONSISTENCY]
    combined = result.policies[STRICT_MASK_AND_LOCAL_WINDOWS]
    assert mask.evaluable is False
    assert mask.ready is None
    assert combined.evaluable is False
    assert combined.ready is None


def test_no_fully_finite_window_fails_local_policy():
    result = _evaluate(
        fully_finite_window_count=0,
        local_window_min_valid_fraction=None,
    )

    assert result.policies[FULLY_FINITE_LOCAL_WINDOWS].ready is False
    assert result.policies[STRICT_MASK_AND_LOCAL_WINDOWS].ready is False


def test_counterfactual_projection_reuses_same_multiscale_evidence():
    result = _evaluate()
    legacy = project_gate_a_decision(
        result.policies[LEGACY_GLOBAL_FRACTION],
        coherent_spatial_residual=True,
        correlation_decay_supported=True,
        forward_transfer_supported=True,
        laminar_evidence=False,
    )
    local = project_gate_a_decision(
        result.policies[FULLY_FINITE_LOCAL_WINDOWS],
        coherent_spatial_residual=True,
        correlation_decay_supported=True,
        forward_transfer_supported=True,
        laminar_evidence=False,
    )

    assert legacy["decision"] == "UNRESOLVED"
    assert legacy["turbulence_established"] is None
    assert local["decision"] == "STOCHASTIC_MULTISCALE_SUPPORTED"
    assert local["turbulence_established"] is True


def test_ready_support_does_not_manufacture_turbulence_without_transfer():
    result = _evaluate()
    local = project_gate_a_decision(
        result.policies[FULLY_FINITE_LOCAL_WINDOWS],
        coherent_spatial_residual=True,
        correlation_decay_supported=True,
        forward_transfer_supported=False,
        laminar_evidence=False,
    )

    assert local["decision"] == "UNRESOLVED"
    assert local["turbulence_established"] is None


def test_canonical_pass2_comparator_configuration_is_locked():
    canonical = {
        "regime": {
            "support_sensitivity": {
                "policies": [
                    "legacy_global_fraction",
                    "strict_mask_consistency",
                    "fully_finite_local_windows",
                    "strict_mask_and_local_windows",
                ],
                "min_fully_finite_windows": 1,
                "required_local_window_valid_fraction": 1.0,
            }
        }
    }
    policies, minimum, fraction = _locked_sensitivity_config(canonical)
    assert policies[0] == LEGACY_GLOBAL_FRACTION
    assert minimum == 1
    assert fraction == 1.0

    changed = {
        "regime": {
            "support_sensitivity": {
                "policies": ["fully_finite_local_windows"],
                "min_fully_finite_windows": 1,
                "required_local_window_valid_fraction": 1.0,
            }
        }
    }
    with pytest.raises(ValueError):
        _locked_sensitivity_config(changed)
