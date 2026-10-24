from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class StructuredGrid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    node_index: np.ndarray
    spacing: tuple[float, float, float]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.x), len(self.y), len(self.z))

    def describe(self) -> dict:
        return {
            "shape": list(self.shape),
            "spacing_native": list(self.spacing),
            "x_range": [float(self.x[0]), float(self.x[-1])],
            "y_range": [float(self.y[0]), float(self.y[-1])],
            "z_range": [float(self.z[0]), float(self.z[-1])],
        }

    def reshape_nodes(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[0] != self.node_index.size:
            raise ValueError("values first dimension must equal the number of grid nodes")
        trailing = values.shape[1:]
        flat = values[self.node_index.ravel()]
        return flat.reshape(self.shape + trailing)

    def extract_window(self, values: np.ndarray, window: "WindowSpec") -> np.ndarray:
        values = np.asarray(values)
        idx = self.node_index[window.slices]
        return values[idx]


@dataclass(frozen=True)
class WindowSpec:
    start: tuple[int, int, int]
    stop: tuple[int, int, int]
    valid_fraction: float

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop))  # type: ignore[return-value]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(b - a for a, b in zip(self.start, self.stop))

    def to_dict(self) -> dict:
        return asdict(self)


def _uniform_spacing(values: np.ndarray, *, rtol: float = 1e-5) -> float:
    if values.size < 2:
        raise ValueError("Structured grid axis must contain at least two coordinates")
    diffs = np.diff(values)
    spacing = float(np.median(diffs))
    if spacing <= 0 or not np.allclose(diffs, spacing, rtol=rtol, atol=abs(spacing) * rtol):
        raise ValueError("Coordinate axis is not uniformly spaced within tolerance")
    return spacing


def infer_structured_grid(coordinates: np.ndarray) -> StructuredGrid:
    """Infer the deposited Cartesian grid without assuming row ordering."""
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape (N_nodes, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError("Coordinate columns must be finite")
    axes = [np.unique(xyz[:, i]) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    if int(np.prod(shape)) != xyz.shape[0]:
        raise ValueError(f"Coordinates do not form a complete Cartesian grid: {shape}")
    indices = [np.searchsorted(axes[i], xyz[:, i]) for i in range(3)]
    linear = np.ravel_multi_index(tuple(indices), shape)
    if np.unique(linear).size != xyz.shape[0]:
        raise ValueError("Duplicate Cartesian grid locations detected")
    node_index_flat = np.empty(xyz.shape[0], dtype=np.int64)
    node_index_flat[linear] = np.arange(xyz.shape[0], dtype=np.int64)
    node_index = node_index_flat.reshape(shape)
    spacing = tuple(_uniform_spacing(a) for a in axes)
    return StructuredGrid(axes[0], axes[1], axes[2], node_index, spacing)


def _starts(n: int, width: int, overlap: float) -> list[int]:
    if width > n:
        return []
    stride = max(1, int(round(width * (1.0 - overlap))))
    starts = list(range(0, n - width + 1, stride))
    final = n - width
    if final not in starts:
        starts.append(final)
    return starts


def candidate_windows(
    support_grid: np.ndarray,
    *,
    shape: tuple[int, int, int],
    overlap: float = 0.5,
    min_valid_fraction: float = 0.90,
    require_full: bool = False,
    max_windows: int = 8,
) -> list[WindowSpec]:
    support = np.asarray(support_grid, dtype=bool)
    if support.ndim != 3:
        raise ValueError("support_grid must be 3-D")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    candidates: list[WindowSpec] = []
    for i in _starts(support.shape[0], shape[0], overlap):
        for j in _starts(support.shape[1], shape[1], overlap):
            for k in _starts(support.shape[2], shape[2], overlap):
                sub = support[i:i + shape[0], j:j + shape[1], k:k + shape[2]]
                fraction = float(sub.mean())
                if require_full and fraction < 1.0:
                    continue
                if fraction < min_valid_fraction:
                    continue
                candidates.append(WindowSpec((i, j, k), (i + shape[0], j + shape[1], k + shape[2]), fraction))
    center = 0.5 * np.asarray(support.shape, dtype=float)
    candidates.sort(key=lambda w: (-w.valid_fraction, float(np.linalg.norm(0.5 * (np.asarray(w.start) + np.asarray(w.stop)) - center)), w.start))
    return candidates[:max_windows]


def adaptive_windows(
    support_grid: np.ndarray,
    *,
    preferred: tuple[int, int, int] = (32, 32, 32),
    fallback_sizes: tuple[int, ...] = (24, 16, 12, 8),
    overlap: float = 0.5,
    min_valid_fraction: float = 0.90,
    require_full: bool = True,
    max_windows: int = 8,
) -> list[WindowSpec]:
    """Return the largest admissible local windows, independent of target exponents."""
    shapes: list[tuple[int, int, int]] = [preferred]
    for size in fallback_sizes:
        shapes.append(tuple(min(size, n) for n in np.asarray(support_grid).shape))
    seen: set[tuple[int, int, int]] = set()
    for shape in shapes:
        if shape in seen or min(shape) < 4:
            continue
        seen.add(shape)
        windows = candidate_windows(support_grid, shape=shape, overlap=overlap, min_valid_fraction=min_valid_fraction, require_full=require_full, max_windows=max_windows)
        if windows:
            return windows
    return []
