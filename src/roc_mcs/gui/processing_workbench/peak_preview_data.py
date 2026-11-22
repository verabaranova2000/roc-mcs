"""Build bounded, scientifically consistent pixel input for the peak-form preview.

The preview uses the same Scan/Spike artifacts, validity rules, spatial
neighbourhood averaging and background-noise definition as ``peaks.detect``.
Only a small local tile is retained by the GUI; full detector volumes stay in
worker scope and are released when this function returns.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _nearest_axis_index(axis: Any, coordinate: float, name: str) -> int:
    values = np.asarray(axis, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Ось {name} не является конечной одномерной осью")
    if values.size > 1:
        delta = np.diff(values)
        if not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"Ось {name} должна быть строго монотонной")
        half_step = float(np.median(np.abs(delta))) / 2.0
        low = min(float(values[0]), float(values[-1])) - half_step
        high = max(float(values[0]), float(values[-1])) + half_step
        if not low <= float(coordinate) <= high:
            raise ValueError(
                f"Координата {name}={float(coordinate):.6g} вне текущей detector-grid"
            )
    elif not np.isclose(float(values[0]), float(coordinate), rtol=0.0, atol=1e-9):
        raise ValueError(f"Координата {name} не соответствует единственному пикселю")
    return int(np.argmin(np.abs(values - float(coordinate))))


def _background_sigma(data_yxt: np.ndarray, pixel_valid_yx: np.ndarray) -> float:
    """Match estimate_background_stats after build_peak_catalog masks invalid sites.

    The production path computes max(Y×X) over theta, then q30, median and MAD.
    We do the same in row blocks, avoiding a second full-size masked Y×X×theta
    copy merely for the interactive preview.
    """
    data = np.asarray(data_yxt, dtype=float)
    valid = np.asarray(pixel_valid_yx, dtype=bool)
    if data.ndim != 3 or valid.shape != data.shape[:2]:
        raise ValueError("Форма данных и pixel_valid_yx не согласованы")

    ny, nx, _ = data.shape
    max_map = np.full((ny, nx), np.nan, dtype=float)
    for y0 in range(0, ny, 32):
        y1 = min(ny, y0 + 32)
        # np.max intentionally propagates a NaN within a θ trace, matching
        # estimate_background_stats on the masked working array.
        with np.errstate(invalid="ignore"):
            block_max = np.max(data[y0:y1], axis=-1)
        block_valid = valid[y0:y1]
        max_map[y0:y1] = np.where(block_valid, block_max, np.nan)

    finite_max = max_map[np.isfinite(max_map)]
    if finite_max.size == 0:
        raise ValueError("Нет конечных валидных данных для оценки фонового шума")
    q30 = float(np.percentile(finite_max, 30))
    background_pixels = finite_max[finite_max <= q30]
    if background_pixels.size == 0:
        raise ValueError("Не удалось выделить пиксели фоновой области")
    median = float(np.median(background_pixels))
    mad = float(np.median(np.abs(background_pixels - median)))
    return float(1.4826 * mad if mad > 0.0 else 1.0)


def build_peak_pixel_preview_data(
    scan_bundle: Any,
    spikes_bundle: Any | None,
    detector_x: float,
    detector_y: float,
    *,
    max_window_size: int = 15,
) -> dict[str, Any]:
    """Prepare the exact peak-detector input around a clicked detector coordinate."""
    arrays = scan_bundle.arrays
    measured = np.asarray(arrays["data_yxt"], dtype=float)
    theta = np.asarray(arrays["theta"], dtype=float)
    x_coords = np.asarray(arrays["x_coords"], dtype=float)
    y_coords = np.asarray(arrays["y_coords"], dtype=float)
    if measured.ndim != 3:
        raise ValueError("Scan data_yxt должен иметь форму Y×X×θ")
    ny, nx, ntheta = measured.shape
    if theta.shape != (ntheta,) or not np.isfinite(theta).all():
        raise ValueError("Ось θ не согласована с Scan-входом или содержит нечисловые значения")
    if theta.size > 1 and np.any(np.diff(theta) <= 0):
        raise ValueError("Детектор пиков требует строго возрастающую ось θ")
    if x_coords.shape != (nx,) or y_coords.shape != (ny,):
        raise ValueError("Координатные оси не согласованы с Scan-входом")

    scan_valid_value = arrays.get("valid_mask_yxt")
    if scan_valid_value is None:
        pixel_valid = np.ones((ny, nx), dtype=bool)
    else:
        scan_valid = np.asarray(scan_valid_value, dtype=bool)
        if scan_valid.shape != measured.shape:
            raise ValueError("valid_mask_yxt не согласована с Scan-входом")
        # This is exactly the validity definition used by _run_peaks.
        pixel_valid = np.all(scan_valid, axis=-1)

    analysis_valid_value = arrays.get("analysis_valid_mask_yx")
    if analysis_valid_value is not None:
        analysis_valid = np.asarray(analysis_valid_value, dtype=bool)
        if analysis_valid.shape != (ny, nx):
            raise ValueError("analysis_valid_mask_yx не согласована с Scan-входом")
        pixel_valid &= analysis_valid

    ix = _nearest_axis_index(x_coords, detector_x, "X")
    iy = _nearest_axis_index(y_coords, detector_y, "Y")
    if not pixel_valid[iy, ix]:
        raise ValueError(
            f"Выбранный пиксель ({ix}, {iy}) исключён из анализа по valid-mask/coverage"
        )

    interpolated = None
    if spikes_bundle is not None:
        if str(spikes_bundle.metadata.get("grid_id", "")) != str(scan_bundle.metadata.get("grid_id", "")):
            raise ValueError("Spike artifact и Scan относятся к разным spatial-grid")
        interpolated_value = spikes_bundle.arrays.get("clean_data_yxt")
        if interpolated_value is not None:
            interpolated = np.asarray(interpolated_value, dtype=float)
            if interpolated.shape != measured.shape:
                raise ValueError("clean_data_yxt не согласована с Scan-входом")

    max_window_size = int(max_window_size)
    if max_window_size < 1 or max_window_size % 2 == 0:
        raise ValueError("max_window_size должен быть положительным нечётным числом")
    half = max_window_size // 2
    y0, y1 = max(0, iy - half), min(ny, iy + half + 1)
    x0, x1 = max(0, ix - half), min(nx, ix + half + 1)

    measured_sigma = _background_sigma(measured, pixel_valid)
    interpolated_sigma = (
        _background_sigma(interpolated, pixel_valid) if interpolated is not None else None
    )

    return {
        "measured_tile_yxt": np.array(measured[y0:y1, x0:x1, :], copy=True),
        "interpolated_tile_yxt": (
            np.array(interpolated[y0:y1, x0:x1, :], copy=True)
            if interpolated is not None else None
        ),
        "pixel_valid_yx": np.array(pixel_valid[y0:y1, x0:x1], copy=True),
        "center_y": int(iy - y0),
        "center_x": int(ix - x0),
        "theta": theta.copy(),
        "noise_sigma_measured": measured_sigma,
        "noise_sigma_interpolated": interpolated_sigma,
        "pixel_x": int(ix),
        "pixel_y": int(iy),
        "detector_x": float(x_coords[ix]),
        "detector_y": float(y_coords[iy]),
        "grid_id": str(scan_bundle.metadata.get("grid_id", "")),
        "interpolation_available": interpolated is not None,
    }
