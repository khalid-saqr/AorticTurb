from __future__ import annotations

from pathlib import Path
from ..constants import PHASES, REALIZATIONS


def velocity_root(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "raw" / "M_medium_root" / "AAo"


def phase_dir(dataset_root: str | Path, phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase {phase!r}; expected one of {PHASES}")
    return velocity_root(dataset_root) / phase


def expected_velocity_paths(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    return [
        root / "raw" / "M_medium_root" / "AAo" / phase / name
        for phase in PHASES
        for name in REALIZATIONS
    ]


def expected_geometry_paths(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root) / "geometry"
    return [
        root / "M_aortic_root.STL",
        root / "S_aortic_root.STL",
        root / "L_aortic_root.STL",
        root / "NS_aortic_root.STL",
    ]
