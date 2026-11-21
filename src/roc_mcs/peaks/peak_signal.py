"""Shared signal construction used by peak cataloguing and diagnostics."""
from __future__ import annotations

import numpy as np


def local_neighbourhood_mean_curve(
    data_yxt: np.ndarray,
    y: int,
    x: int,
    window_size: int = 3,
    *,
    pixel_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Return the exact spatial-neighbourhood mean ROC used by peak detection.

    ``data_yxt`` is Y×X×theta. Invalid spatial pixels are excluded from every
    theta sample (never interpreted as zero). Non-finite samples are also
    omitted independently at each theta, matching the catalogue's finite
    contributor averaging. The result is NaN if a theta sample has no valid
    contributors. The function never mutates the input array.
    """
    values = np.asarray(data_yxt, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"data_yxt должен иметь форму Y×X×θ, получено {values.shape}")
    ny, nx, ntheta = values.shape
    yi, xi, width = int(y), int(x), int(window_size)
    if not (0 <= yi < ny and 0 <= xi < nx):
        raise IndexError(f"Координата пикселя вне сетки: x={xi}, y={yi}")
    if width < 1 or width % 2 == 0:
        raise ValueError("window_size должен быть положительным нечётным числом")

    valid = None
    if pixel_valid is not None:
        valid = np.asarray(pixel_valid, dtype=bool)
        if valid.shape != (ny, nx):
            raise ValueError(
                f"pixel_valid должен иметь форму {(ny, nx)}, получено {valid.shape}"
            )

    half = width // 2
    y0, y1 = max(0, yi - half), min(ny, yi + half + 1)
    x0, x1 = max(0, xi - half), min(nx, xi + half + 1)
    neighbourhood = values[y0:y1, x0:x1, :]
    finite = np.isfinite(neighbourhood)
    if valid is not None:
        local_valid = valid[y0:y1, x0:x1]
        finite &= local_valid[..., None]
    counts = np.count_nonzero(finite, axis=(0, 1))
    sums = np.sum(np.where(finite, neighbourhood, 0.0), axis=(0, 1))
    return np.divide(
        sums,
        counts,
        out=np.full(ntheta, np.nan, dtype=float),
        where=counts > 0,
    )
