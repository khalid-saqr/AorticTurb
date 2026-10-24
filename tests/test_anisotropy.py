import numpy as np

from aorticturb.anisotropy.driver import PLOT_DATA_FIELDS, _write_plot_ready_anisotropy
from aorticturb.anisotropy.tensor import (
    anisotropy_tensor,
    barycentric_coordinates,
    eigenvalues,
    invariants,
    realizability_mask,
)


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
    expected = np.diag([2 / 3, -1 / 3, -1 / 3])[None, ...]
    np.testing.assert_allclose(b, expected)


def test_eigenvalues_preserve_nonfinite_nodes_as_nan():
    b = np.array(
        [
            np.diag([2 / 3, -1 / 3, -1 / 3]),
            np.full((3, 3), np.nan),
            np.zeros((3, 3)),
        ],
        dtype=float,
    )

    lam = eigenvalues(b)

    np.testing.assert_allclose(lam[0], [2 / 3, -1 / 3, -1 / 3])
    assert np.isnan(lam[1]).all()
    np.testing.assert_allclose(lam[2], [0.0, 0.0, 0.0])

    bary = barycentric_coordinates(b)
    np.testing.assert_allclose(bary[0], [1.0, 0.0, 0.0])
    assert np.isnan(bary[1]).all()
    np.testing.assert_allclose(bary[2], [0.0, 0.0, 1.0])

    real = realizability_mask(b)
    np.testing.assert_array_equal(real, [True, False, True])


def test_plot_ready_anisotropy_persists_nodewise_quantities(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "processed").mkdir(parents=True)

    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    R = np.array(
        [
            np.eye(3) * 2.0,
            np.diag([2.0, 0.0, 0.0]),
            np.diag([1.0, 1.0, 0.0]),
        ]
    )
    b, k = anisotropy_tensor(R)
    ii, iii = invariants(b)
    lam = eigenvalues(b)
    bary = barycentric_coordinates(b)
    real = realizability_mask(b)
    valid = np.array([True, True, False])

    path = _write_plot_ready_anisotropy(
        run_dir,
        "t021",
        {"coordinates": coordinates},
        valid=valid,
        k=k,
        b=b,
        ii=ii,
        iii=iii,
        lam=lam,
        bary=bary,
        real=real,
    )

    assert path == run_dir / "processed" / "t021_anisotropy.npz"
    with np.load(path, allow_pickle=False) as saved:
        assert tuple(saved.files) == PLOT_DATA_FIELDS
        np.testing.assert_array_equal(saved["node_index"], [0, 1])
        np.testing.assert_allclose(saved["coordinates"], coordinates[:2])
        np.testing.assert_allclose(saved["k"], k[:2])
        np.testing.assert_allclose(saved["b"], b[:2])
        np.testing.assert_allclose(saved["II"], ii[:2])
        np.testing.assert_allclose(saved["III"], iii[:2])
        np.testing.assert_allclose(saved["eigenvalues"], lam[:2])
        np.testing.assert_allclose(saved["barycentric"], bary[:2])
        np.testing.assert_array_equal(saved["realizable"], real[:2])
