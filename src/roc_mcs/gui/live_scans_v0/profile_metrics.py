"""Простые воспроизводимые метрики сырого профиля без подгонки модели."""
from __future__ import annotations
import numpy as np


def peak_position(x, y) -> float:
    x = np.asarray(x, dtype=float).ravel(); y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3: return float("nan")
    x, y = x[mask], y[mask]
    order = np.argsort(x); x, y = x[order], y[order]
    return float(x[int(np.argmax(y))])


def fwhm_half_height(x, y) -> float:
    """Ширина главного максимума на половине высоты над минимумом профиля."""
    x = np.asarray(x, dtype=float).ravel(); y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3: return float("nan")
    x, y = x[mask], y[mask]
    order = np.argsort(x); x, y = x[order], y[order]
    peak_idx = int(np.argmax(y)); baseline = float(np.min(y)); peak = float(y[peak_idx])
    if not np.isfinite(peak) or peak <= baseline: return float("nan")
    half = baseline + 0.5 * (peak - baseline)
    left_idx = next((i for i in range(peak_idx - 1, -1, -1) if y[i] <= half and y[i + 1] >= half), None)
    right_idx = next((i for i in range(peak_idx, len(y) - 1) if y[i] >= half and y[i + 1] <= half), None)
    if left_idx is None or right_idx is None: return float("nan")
    def crossing(i):
        dy = y[i + 1] - y[i]
        if dy == 0: return float((x[i] + x[i + 1]) / 2)
        return float(x[i] + (half - y[i]) * (x[i + 1] - x[i]) / dy)
    width = crossing(right_idx) - crossing(left_idx)
    return float(width) if np.isfinite(width) and width >= 0 else float("nan")
