import numpy as np

from aorticturb.anisotropy.tensor import anisotropy_tensor, invariants


def test_isotropic_stress_has_zero_anisotropy():
    R = np.eye(3)[None, ...] * 2.0
    b, k = anisotropy_tensor(R)
    np.testing.assert_allclose(k, [3.0])
    np.testing.assert_allclose(b, 0.0, atol=1e-14)
    ii, iii = invariants(b)
    np.testing.assert_allclose(ii, 0.0)
    np.testing.assert_allclose(iii, 0.0)


def test_one_component_limit_convention():
    R = np.zeros((1, 3, 3))
    R[0, 0, 0] = 2.0
    b, _ = anisotropy_tensor(R)
    expected = np.diag([2/3, -1/3, -1/3])[None, ...]
    np.testing.assert_allclose(b, expected)
