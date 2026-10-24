import numpy as np

from aorticturb.kolmogorov.flux import coarse_grained_flux


def test_uniform_velocity_has_zero_coarse_grained_flux():
    u = np.ones((8, 8, 8, 3), dtype=float)
    pi = coarse_grained_flux(u, (1.0, 1.0, 1.0), sigma=1.0)
    np.testing.assert_allclose(pi, 0.0, atol=1e-12)
