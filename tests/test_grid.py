import numpy as np

from aorticturb.grid import infer_structured_grid, candidate_windows


def test_grid_inference_does_not_assume_row_order():
    x = np.arange(3.0)
    y = np.arange(4.0)
    z = np.arange(2.0)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    values = (X + 10 * Y + 100 * Z).ravel()
    rng = np.random.default_rng(4)
    p = rng.permutation(len(coords))
    grid = infer_structured_grid(coords[p])
    restored = grid.reshape_nodes(values[p])
    np.testing.assert_allclose(restored, X + 10 * Y + 100 * Z)
    assert grid.shape == (3, 4, 2)


def test_full_window_selection_rejects_holes():
    support = np.ones((8, 8, 8), dtype=bool)
    support[:4, :4, :4] = False
    windows = candidate_windows(support, shape=(4, 4, 4), overlap=0.0, require_full=True)
    assert windows
    assert all(w.valid_fraction == 1.0 for w in windows)
