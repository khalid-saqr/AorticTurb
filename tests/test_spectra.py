import numpy as np

from aorticturb.kolmogorov.spectra import directional_spectrum, fit_scaling_range


def test_directional_spectrum_detects_known_mode():
    n = 16
    x = np.arange(n)
    signal = np.sin(2 * np.pi * 2 * x / n)
    u = np.zeros((n, 6, 5, 3), dtype=float)
    u[..., 0] = signal[:, None, None]
    k, e = directional_spectrum(u, (1.0, 1.0, 1.0), axis=0, taper=False)
    peak = k[np.argmax(e)]
    np.testing.assert_allclose(peak, 2 * np.pi * 2 / n)


def test_scaling_range_selection_is_target_independent():
    x = np.logspace(0, 1, 10)
    y = 3.0 * x ** (-5.0 / 3.0)
    fit = fit_scaling_range(x, y, min_points=4)
    assert fit is not None
    np.testing.assert_allclose(fit.slope, -5.0 / 3.0, atol=1e-10)
