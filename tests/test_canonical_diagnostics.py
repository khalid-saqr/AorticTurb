import numpy as np

from aorticturb.diagnostics.admissibility import physical_dissipation_admissibility
from aorticturb.diagnostics.uncertainty import (
    leave_one_out_mean_summary,
    subset_mean_convergence,
)


def test_leave_one_out_summary_is_realization_level():
    result = leave_one_out_mean_summary(np.array([1.0, 2.0, 3.0, 4.0]))
    assert result["full_mean"] == 2.5
    assert result["loo_min"] == 2.0
    assert result["loo_max"] == 3.0
    assert result["loo_max_abs_deviation"] == 0.5


def test_subset_convergence_is_deterministic_and_full_n_is_exact():
    values = np.arange(1.0, 17.0)
    a = subset_mean_convergence(values, sizes=(4, 8, 12, 16), draws=32, seed=7)
    b = subset_mean_convergence(values, sizes=(4, 8, 12, 16), draws=32, seed=7)
    assert a == b
    final = a[-1]
    assert final["ensemble_size"] == 16
    assert final["lower"] == np.mean(values)
    assert final["median"] == np.mean(values)
    assert final["upper"] == np.mean(values)


def test_physical_dissipation_is_inadmissible_without_verified_inputs():
    missing = physical_dissipation_admissibility({"physical": {"kinematic_viscosity": None}})
    assert missing["admissible"] is False
    assert missing["status"] == "INADMISSIBLE_MISSING_VERIFIED_VISCOSITY"

    unverified = physical_dissipation_admissibility(
        {"physical": {"kinematic_viscosity": 1.0e-6}}
    )
    assert unverified["admissible"] is False
    assert unverified["status"] == "INADMISSIBLE_VISCOSITY_NOT_VERIFIED"

    unresolved = physical_dissipation_admissibility(
        {
            "physical": {
                "kinematic_viscosity": 1.0e-6,
                "kinematic_viscosity_verified": True,
            },
            "diagnostics": {"small_scale_resolution_verified": False},
        }
    )
    assert unresolved["admissible"] is False
    assert unresolved["status"] == "INADMISSIBLE_SMALL_SCALE_RESOLUTION_NOT_VERIFIED"
