import numpy as np

from aorticturb.kolmogorov.structure_functions import longitudinal_increments


def test_longitudinal_increment():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    u = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    r, du = longitudinal_increments(x, u, np.array([[0, 1]]))
    np.testing.assert_allclose(r, [1.0])
    np.testing.assert_allclose(du, [2.0])
