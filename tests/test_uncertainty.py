import numpy as np

from aorticturb.uncertainty.bootstrap import bootstrap_mean_ci


def test_realization_bootstrap_is_deterministic_for_fixed_seed():
    x = np.arange(1.0, 17.0)
    a = bootstrap_mean_ci(x, samples=200, seed=123)
    b = bootstrap_mean_ci(x, samples=200, seed=123)
    np.testing.assert_allclose(a, b)
    assert a[0] <= a[1] <= a[2]
