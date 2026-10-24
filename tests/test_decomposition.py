import numpy as np

from aorticturb.decomposition.core import phase_decomposition, reynolds_stress


def test_phase_decomposition_zero_mean_residuals():
    fields = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        ]
    )
    mean, residuals = phase_decomposition(fields)
    np.testing.assert_allclose(mean, [[2, 3, 4], [5, 6, 7]])
    np.testing.assert_allclose(np.mean(residuals, axis=0), 0.0)


def test_reynolds_stress_and_k():
    residuals = np.array(
        [
            [[-1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ]
    )
    R, k = reynolds_stress(residuals, ddof=1)
    np.testing.assert_allclose(R[0, 0, 0], 2.0)
    np.testing.assert_allclose(k[0], 1.0)
