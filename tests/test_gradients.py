import numpy as np

from aorticturb.diagnostics.gradients import (
    derivative_interior_mask,
    velocity_gradient,
    velocity_gradient_central,
    vorticity,
)


def test_solid_body_rotation_vorticity():
    n = 5
    x = y = z = np.arange(n, dtype=float)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    u = np.stack([-Y, X, np.zeros_like(Z)], axis=-1)
    grad = velocity_gradient(u, (1.0, 1.0, 1.0))
    omega = vorticity(grad)
    np.testing.assert_allclose(omega[..., 2], 2.0, atol=1e-12)
    np.testing.assert_allclose(omega[..., :2], 0.0, atol=1e-12)


def test_central_gradient_steps_agree_for_linear_field_on_common_interior():
    n = 9
    x = np.arange(n, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.stack([2.0 * X + Y, -3.0 * Y + Z, 4.0 * Z - X], axis=-1)
    g1 = velocity_gradient_central(u, (1.0, 1.0, 1.0), step=1, trim=2)
    g2 = velocity_gradient_central(u, (1.0, 1.0, 1.0), step=2, trim=2)
    expected = np.array([[2.0, 1.0, 0.0], [0.0, -3.0, 1.0], [-1.0, 0.0, 4.0]])
    np.testing.assert_allclose(g1, expected, atol=1e-12)
    np.testing.assert_allclose(g2, expected, atol=1e-12)


def test_derivative_interior_mask_trims_every_face():
    mask = derivative_interior_mask((8, 9, 10), trim=2)
    assert mask.shape == (8, 9, 10)
    assert int(mask.sum()) == 4 * 5 * 6
    assert not mask[:2].any()
    assert mask[2:-2, 2:-2, 2:-2].all()
