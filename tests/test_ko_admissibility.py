import numpy as np

from aorticturb.grid import infer_structured_grid, WindowSpec
from aorticturb.kolmogorov.admissibility import assess_local_homogeneity


def test_uniform_statistical_energy_does_not_reject_homogeneity():
    n = 8
    X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(float)
    grid = infer_structured_grid(coords)
    # Every realization has spatially uniform magnitude, so all octants have equal energy.
    residuals = np.stack([
        np.tile(np.array([float(i + 1), 0.0, 0.0]), (coords.shape[0], 1))
        for i in range(16)
    ])
    window = WindowSpec((0, 0, 0), (8, 8, 8), 1.0)
    result = assess_local_homogeneity(residuals, grid, [window], samples=200, seed=11)
    assert result.not_rejected is True
    assert len(result.octant_deviation_ci) == 8
