"""Pure NumPy helpers for the peak-count / integrated-intensity 2D view.

The palette and alpha law deliberately match ``plot_peak_count_map`` in
``visualization/peaks.py`` so exported figures and the interactive ROI Explorer
share one visual meaning without coupling the GUI to pyplot.
"""
from __future__ import annotations

import numpy as np


PEAK_COUNT_COLORS: tuple[str, ...] = (
    "#1f77b4",  # 1
    "#d62728",  # 2
    "#2ca02c",  # 3
    "#ff7f0e",  # 4
    "#9467bd",  # 5+
)


def integrated_intensity_map(
    data_yxt,
    theta,
    *,
    valid_mask_yxt=None,
    analysis_valid_mask_yx=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate Y×X×theta data over theta and return (intensity, valid_yx).

    A pixel is marked valid only when every theta sample is finite and valid,
    and it passes the optional spatial analysis mask. Invalid observations are
    never converted into apparently physical zeros in the returned map.
    """
    data = np.asarray(data_yxt, dtype=np.float64)
    axis = np.asarray(theta, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"data_yxt должна иметь форму Y×X×θ, получено {data.shape}")
    if axis.ndim != 1 or axis.size != data.shape[-1] or not np.isfinite(axis).all():
        raise ValueError("theta должна быть конечной 1D-осью длины data_yxt.shape[-1]")
    if axis.size > 1 and not np.all(np.diff(axis) > 0):
        raise ValueError("Для интегральной интенсивности theta должна строго возрастать")

    sample_valid = np.isfinite(data)
    if valid_mask_yxt is not None:
        mask = np.asarray(valid_mask_yxt, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError(f"valid_mask_yxt имеет форму {mask.shape}, ожидалась {data.shape}")
        sample_valid &= mask
    valid_yx = np.all(sample_valid, axis=-1)
    if analysis_valid_mask_yx is not None:
        spatial = np.asarray(analysis_valid_mask_yx, dtype=bool)
        if spatial.shape != data.shape[:2]:
            raise ValueError(
                f"analysis_valid_mask_yx имеет форму {spatial.shape}, "
                f"ожидалась {data.shape[:2]}"
            )
        valid_yx &= spatial

    safe_data = np.where(sample_valid, data, 0.0)
    intensity = np.trapezoid(safe_data, x=axis, axis=-1)
    intensity = np.asarray(intensity, dtype=np.float64)
    intensity[~valid_yx] = np.nan
    return intensity, valid_yx


def peak_count_rgba(
    peak_count_map,
    intensity_map,
    *,
    theme: str = "dark",
    valid_mask_yx=None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Create the alpha-modulated discrete palette used by the 2D peak map.

    Hue identifies the number of peaks (1, 2, 3, 4, 5+); opacity increases
    monotonically with integrated intensity using the same 5th/99th percentile
    scaling and exponent 0.6 as the static ``plot_peak_count_map`` renderer.
    """
    counts = np.asarray(peak_count_map, dtype=np.float64)
    intensity = np.asarray(intensity_map, dtype=np.float64)
    if counts.ndim != 2 or intensity.ndim != 2 or counts.shape != intensity.shape:
        raise ValueError(
            f"peak_count_map и intensity_map должны иметь одинаковую 2D-форму; "
            f"получено {counts.shape} и {intensity.shape}"
        )
    normalized_theme = str(theme).lower()
    if normalized_theme not in {"dark", "light"}:
        raise ValueError("theme должен быть 'dark' или 'light'")

    valid = np.isfinite(counts) & np.isfinite(intensity)
    if valid_mask_yx is not None:
        mask = np.asarray(valid_mask_yx, dtype=bool)
        if mask.shape != counts.shape:
            raise ValueError(f"valid_mask_yx имеет форму {mask.shape}, ожидалась {counts.shape}")
        valid &= mask

    finite_intensity = intensity[valid]
    if finite_intensity.size:
        low, high = (float(v) for v in np.percentile(finite_intensity, [5.0, 99.0]))
    else:
        low, high = 0.0, 1.0
    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        alpha = np.ones(counts.shape, dtype=np.float64)
    else:
        alpha = np.clip((intensity - low) / (high - low), 0.0, 1.0) ** 0.6
    alpha = np.where(valid, alpha, 0.0)

    background = "#000000" if normalized_theme == "dark" else "#ffffff"
    palette = np.asarray(
        [
            [int(background[i:i + 2], 16) / 255.0 for i in (1, 3, 5)],
            *[
                [int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
                for color in PEAK_COUNT_COLORS
            ],
        ],
        dtype=np.float64,
    )
    safe_counts = np.where(np.isfinite(counts), counts, 0.0)
    indices = np.clip(np.rint(safe_counts).astype(np.int64), 0, 5)
    rgba = np.empty((*counts.shape, 4), dtype=np.float64)
    rgba[..., :3] = palette[indices]
    rgba[..., 3] = alpha
    return rgba, (low, high)
