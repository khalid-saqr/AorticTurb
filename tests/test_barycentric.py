import numpy as np

from aorticturb.anisotropy.tensor import anisotropy_tensor, barycentric_coordinates, realizability_mask


def test_barycentric_limits():
    b_iso, _ = anisotropy_tensor(np.eye(3) * 2.0)
    c_iso = barycentric_coordinates(b_iso)
    np.testing.assert_allclose(c_iso, [0.0, 0.0, 1.0], atol=1e-12)
    R = np.diag([2.0, 0.0, 0.0])
    b_one, _ = anisotropy_tensor(R)
    c_one = barycentric_coordinates(b_one)
    np.testing.assert_allclose(c_one, [1.0, 0.0, 0.0], atol=1e-12)
    assert bool(realizability_mask(b_one))
