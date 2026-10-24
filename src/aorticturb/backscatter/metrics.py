from __future__ import annotations

import numpy as np

SIGNED_METRIC_KEYS = (
    "mean_pi",
    "median_pi",
    "forward_fraction",
    "backscatter_fraction",
    "zero_fraction",
    "forward_contribution",
    "backscatter_contribution",
    "backscatter_magnitude",
)


def signed_transfer_metrics(pi: np.ndarray) -> dict[str, float]:
    """Summarize a finite signed interscale-transfer field without thresholding.

    The sign convention is inherited from ``coarse_grained_flux``:
    ``Pi > 0`` is forward/downscale transfer and ``Pi < 0`` is local backscatter.
    Zero is retained as its own category; no amplitude cutoff is introduced.
    """
    values = np.asarray(pi, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in SIGNED_METRIC_KEYS}

    positive = np.where(values > 0.0, values, 0.0)
    negative = np.where(values < 0.0, values, 0.0)
    backscatter_contribution = float(np.mean(negative))

    return {
        "mean_pi": float(np.mean(values)),
        "median_pi": float(np.median(values)),
        "forward_fraction": float(np.mean(values > 0.0)),
        "backscatter_fraction": float(np.mean(values < 0.0)),
        "zero_fraction": float(np.mean(values == 0.0)),
        "forward_contribution": float(np.mean(positive)),
        "backscatter_contribution": backscatter_contribution,
        "backscatter_magnitude": -backscatter_contribution,
    }


def ko_interior_mask(shape: tuple[int, int, int], sigma: float) -> np.ndarray:
    """Return the exact interior region used by the existing KO mean-flux routine.

    The KO implementation removes a margin of ``ceil(3*sigma)`` grid points from
    every face only when the window remains sufficiently large.  Reusing this rule
    prevents the backscatter stage from changing the numerical support definition.
    """
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape must contain three positive dimensions")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    margin = int(np.ceil(3.0 * float(sigma)))
    mask = np.ones(shape, dtype=bool)
    if all(n > 2 * margin + 2 for n in shape):
        mask[...] = False
        mask[
            margin : shape[0] - margin,
            margin : shape[1] - margin,
            margin : shape[2] - margin,
        ] = True
    return mask
