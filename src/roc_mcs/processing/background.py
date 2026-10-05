from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BackgroundStats:
    median: float
    sigma: float
    threshold: float


def estimate_background_stats(data_3d: np.ndarray) -> BackgroundStats:
    """Оценивает фон, шум и порог отделения сигнала от фона."""
    data_3d = np.asarray(data_3d, dtype=float)
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено ndim={data_3d.ndim}")

    max_map = np.max(data_3d, axis=0)
    finite_max = max_map[np.isfinite(max_map)]
    if finite_max.size == 0:
        raise ValueError("Не найдено ни одного конечного значения для оценки фона.")

    q30 = np.percentile(finite_max, 30)
    bg_pixels = finite_max[finite_max <= q30]
    bg_median = float(np.median(bg_pixels))
    bg_mad = float(np.median(np.abs(bg_pixels - bg_median)))
    bg_sigma = 1.4826 * bg_mad if bg_mad > 0 else 1.0
    bg_threshold = bg_median + 5.0 * bg_sigma

    return BackgroundStats(
        median=bg_median,
        sigma=bg_sigma,
        threshold=float(bg_threshold),
    )