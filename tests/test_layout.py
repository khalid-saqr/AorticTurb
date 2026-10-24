from pathlib import Path

from aorticturb.data.layout import phase_dir, geometry_paths, provenance_manifest_paths


def test_project_layout_matches_agents_contract():
    root = Path("/dataset")
    assert phase_dir(root, "t021") == (
        root / "01_raw_data" / "M_medium_root" / "AAo" / "t021"
    )
    assert geometry_paths(root)["M"] == (
        root / "02_metadata_geometry" / "M_medium_root"
        / "original_geometry" / "M_aortic_root.STL"
    )
    assert provenance_manifest_paths(root)["pass5"] == (
        root / "00_provenance" / "transfer_manifests" / "pass5_integrity_audit.json"
    )
