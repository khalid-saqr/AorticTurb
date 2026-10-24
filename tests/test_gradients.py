import numpy as np

from aorticturb.diagnostics.gradients import velocity_gradient, vorticity


def test_solid_body_rotation_vorticity():
    n = 5
    x = y = z = np.arange(n, dtype=float)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    u = np.stack([-Y, X, np.zeros_like(Z)], axis=-1)
    grad = velocity_gradient(u, (1.0, 1.0, 1.0))
    omega = vorticity(grad)
    np.testing.assert_allclose(omega[..., 2], 2.0, atol=1e-12)
    np.testing.assert_allclose(omega[..., :2], 0.0, atol=1e-12)
