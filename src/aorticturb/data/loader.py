from __future__ import annotations

from pathlib import Path
import numpy as np

from ..constants import VELOCITY_COLUMNS


def load_velocity_field(path: str | Path, *, mmap: bool = True) -> np.ndarray:
    path = Path(path)
    array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"{path}: expected shape (N_nodes, 6), got {array.shape}")
    if array.dtype != np.float64:
        raise ValueError(f"{path}: expected float64, got {array.dtype}")
    return array


def split_field(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError("Expected array with six columns [X,Y,Z,U,V,W].")
    return array[:, :3], array[:, 3:]


def finite_velocity_mask(array: np.ndarray) -> np.ndarray:
    _, velocity = split_field(array)
    return np.isfinite(velocity).all(axis=1)


def field_summary(array: np.ndarray) -> dict:
    coordinates, velocity = split_field(array)
    mask = np.isfinite(velocity).all(axis=1)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "columns": list(VELOCITY_COLUMNS),
        "coordinate_rows_finite": int(np.isfinite(coordinates).all(axis=1).sum()),
        "velocity_rows_finite": int(mask.sum()),
        "velocity_rows_total": int(array.shape[0]),
        "source_nan_entries": int(np.isnan(velocity).sum()),
    }
