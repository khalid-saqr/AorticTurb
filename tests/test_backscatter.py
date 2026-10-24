import json
from pathlib import Path

import numpy as np

from aorticturb.backscatter.driver import run_backscatter
from aorticturb.backscatter.metrics import ko_interior_mask, signed_transfer_metrics


def test_signed_transfer_metrics_preserve_sign_balance():
    values = np.asarray([-2.0, -1.0, 0.0, 1.0, 3.0])
    result = signed_transfer_metrics(values)

    assert result["forward_fraction"] == 0.4
    assert result["backscatter_fraction"] == 0.4
    assert result["zero_fraction"] == 0.2
    assert result["forward_contribution"] == 0.8
    assert result["backscatter_contribution"] == -0.6
    assert result["backscatter_magnitude"] == 0.6
    np.testing.assert_allclose(
        result["mean_pi"],
        result["forward_contribution"] + result["backscatter_contribution"],
    )


def test_ko_interior_mask_matches_existing_three_sigma_crop():
    mask = ko_interior_mask((32, 32, 32), 4.0)
    assert mask.sum() == 8**3
    assert mask[12:20, 12:20, 12:20].all()
    assert not mask[:12].any()


def test_backscatter_driver_reuses_saved_ko_windows(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    decomposition_dir = tmp_path / "decomposition"
    regime_dir = tmp_path / "regime"
    ko_dir = tmp_path / "ko"
    workspace = tmp_path / "workspace"
    for path in (dataset_root, decomposition_dir, regime_dir, ko_dir, workspace):
        path.mkdir()

    regime = {
        "phases": {
            "t018": {"turbulence_established": None},
            "t021": {"turbulence_established": True},
            "t024": {"turbulence_established": None},
        }
    }
    (regime_dir / "regime_summary.json").write_text(json.dumps(regime), encoding="utf-8")

    ko_config = {
        "analysis": {"bootstrap_samples": 20, "random_seed": 7},
        "kolmogorov": {"flux_sigmas": [0.5]},
    }
    (ko_dir / "config.json").write_text(json.dumps(ko_config), encoding="utf-8")
    (ko_dir / "run_manifest.json").write_text(
        json.dumps({"status": "COMPLETE", "branch": "02_kolmogorov_obukhov"}),
        encoding="utf-8",
    )
    (ko_dir / "kolmogorov_summary.json").write_text(
        json.dumps({"phases": {"t021": {"status": "KO_UNRESOLVED"}}}),
        encoding="utf-8",
    )
    ko_diagnostics = {
        "phases": {
            "t021": {
                "windows": [
                    {"start": [0, 0, 0], "stop": [8, 8, 8], "valid_fraction": 1.0}
                ],
                "flux": {"scales_native": [0.5]},
            }
        }
    }
    (ko_dir / "diagnostics.json").write_text(json.dumps(ko_diagnostics), encoding="utf-8")

    axes = np.arange(8.0)
    xx, yy, zz = np.meshgrid(axes, axes, axes, indexing="ij")
    coordinates = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    support = np.ones(coordinates.shape[0], dtype=bool)
    residuals = np.empty((2, coordinates.shape[0], 3), dtype=float)
    residuals[0] = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    residuals[1] = -residuals[0]

    monkeypatch.setattr(
        "aorticturb.backscatter.driver.load_decomposition",
        lambda _directory, phase: {
            "coordinates": coordinates,
            "support": support,
            "residuals": residuals,
        },
    )

    result = run_backscatter(
        dataset_root,
        decomposition_dir,
        regime_dir,
        ko_dir,
        workspace,
        ko_config,
    )

    assert result["phases"]["t018"]["status"] == "SKIPPED_GATE_A"
    assert result["phases"]["t021"]["status"] == "CHARACTERIZED"
    assert result["phases"]["t024"]["status"] == "SKIPPED_GATE_A"
    assert result["classification_effect"] == "NONE_CHARACTERIZATION_ONLY"

    run_dir = Path(result["run_dir"])
    with np.load(run_dir / "processed" / "t021_signed_transfer.npz") as saved:
        assert saved["pi"].shape == (2, 1, 1, 8, 8, 8)
        assert saved["interior_mask"].shape == (1, 8, 8, 8)
        assert saved["interior_mask"].sum() == 4**3
        outside = ~saved["interior_mask"][0]
        assert np.isnan(saved["pi"][:, 0, 0][:, outside]).all()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    output_paths = {item["path"] for item in manifest["outputs"]}
    assert "processed/t021_signed_transfer.npz" in output_paths
