from __future__ import annotations

from pathlib import Path
from ..constants import PHASES, REALIZATIONS


def velocity_root(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "01_raw_data" / "M_medium_root" / "AAo"


def phase_dir(dataset_root: str | Path, phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase {phase!r}; expected one of {PHASES}")
    return velocity_root(dataset_root) / phase


def expected_velocity_paths(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    return [
        root / "01_raw_data" / "M_medium_root" / "AAo" / phase / name
        for phase in PHASES
        for name in REALIZATIONS
    ]


def geometry_paths(dataset_root: str | Path) -> dict[str, Path]:
    root = Path(dataset_root) / "02_metadata_geometry"
    return {
        "M": root / "M_medium_root" / "original_geometry" / "M_aortic_root.STL",
        "S": root / "S_small_root" / "original_geometry" / "S_aortic_root.STL",
        "L": root / "L_large_root" / "original_geometry" / "L_aortic_root.STL",
        "NS": root / "NS_non_sinus_root" / "original_geometry" / "NS_aortic_root.STL",
    }


def expected_geometry_paths(dataset_root: str | Path) -> list[Path]:
    return list(geometry_paths(dataset_root).values())


def provenance_manifest_paths(dataset_root: str | Path) -> dict[str, Path]:
    root = Path(dataset_root) / "00_provenance" / "transfer_manifests"
    return {
        "pass5": root / "pass5_integrity_audit.json",
        "acquisition": root / "acquisition_manifest_v1.json",
        "geometry": root / "geometry_manifest.json",
    }
