"""Перенос масштаба калибровки на live-ось: θ = Aк·(Uс/Uк)·s; B только для совмещения."""
from __future__ import annotations
import json, math, os, tempfile
from pathlib import Path
import numpy as np

SCHEMA_VERSION = 3


def _positive(value) -> float | None:
    try: result = float(value)
    except (TypeError, ValueError): return None
    return result if math.isfinite(result) and result > 0 else None


def validate_calibration(data: dict | None) -> dict | None:
    """Проверяет файл; старый файл без Uк читается, но не активирует θ-шкалу."""
    if not isinstance(data, dict) or data.get("schema_version") not in {1, 2, SCHEMA_VERSION}: return None
    try: a, b = float(data["A"]), float(data["B"])
    except (KeyError, TypeError, ValueError): return None
    if not math.isfinite(a) or not math.isfinite(b) or abs(a) < 1e-15: return None
    unit = str(data.get("unit", "угл. с")).strip() or "угл. с"
    branch = str(data.get("branch", "up")).strip().lower()
    if branch not in {"up", "down"}: return None
    phi = data.get("reference_phi")
    if phi is not None:
        try: phi = float(phi)
        except (TypeError, ValueError): return None
        if not math.isfinite(phi): return None
    u_cal = _positive(data.get("reference_amplitude_mVpp"))
    normalized = {**data, "A": a, "B": b, "unit": unit, "branch": branch,
                  "reference_phi": phi, "reference_amplitude_mVpp": u_cal}
    # Схема v2 ошибочно добавляла ручную частоту; новое сохранение её не переносит.
    normalized.pop("reference_frequency_hz", None)
    return normalized


def effective_scale(calibration: dict | None, experiment_amplitude_mVpp: float | None) -> float | None:
    """Aс = Aк·Uс/Uк. Требуется положительная известная амплитуда в mVpp."""
    valid = validate_calibration(calibration)
    if valid is None or valid["reference_amplitude_mVpp"] is None: return None
    u_exp = _positive(experiment_amplitude_mVpp)
    if u_exp is None: return None
    scale = valid["A"] * u_exp / valid["reference_amplitude_mVpp"]
    return scale if math.isfinite(scale) and abs(scale) >= 1e-15 else None


def load_calibration(path: str | Path) -> dict | None:
    try: return validate_calibration(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError): return None


def save_calibration(path: str | Path, data: dict) -> dict:
    payload = {**data, "schema_version": SCHEMA_VERSION}
    value = validate_calibration(payload)
    if value is None or value["reference_amplitude_mVpp"] is None:
        raise ValueError("Калибровка не сохранена: нужны конечные A/B, ветвь и положительная Uк (амплитуда генератора в mVpp).")
    value["schema_version"] = SCHEMA_VERSION
    value["method"] = "scale_by_drive_amplitude_theta_A_s"
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temp_name, target)
    except Exception:
        try: os.unlink(temp_name)
        except OSError: pass
        raise
    return value


def transform_axis(s_axis, intensity, calibration: dict | None, *, experiment_amplitude_mVpp: float | None = None):
    """Возвращает θ и интенсивность; при отсутствии Uк/Uс безопасно оставляет s."""
    s = np.asarray(s_axis, dtype=float).reshape(-1); y = np.asarray(intensity, dtype=float).reshape(-1)
    if s.size != y.size: raise ValueError(f"Размер оси ({s.size}) не совпадает с интенсивностью ({y.size}).")
    scale = effective_scale(calibration, experiment_amplitude_mVpp)
    axis = s if scale is None else scale * s
    order = np.argsort(axis)
    return axis[order], y[order]


def resample_rows_to_uniform_axis(axis, rows):
    """Пересчитывает только растровое представление профилей на равномерную ось."""
    x = np.asarray(axis, dtype=float).reshape(-1); values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != x.size: raise ValueError("Ожидаются профили формы (число_сканов, число_точек_оси).")
    valid = np.isfinite(x); x, values = x[valid], values[:, valid]
    order = np.argsort(x); x, values = x[order], values[:, order]
    x, unique_idx = np.unique(x, return_index=True); values = values[:, unique_idx]
    if x.size < 2: raise ValueError("Для построения РОК-карты нужны минимум две уникальные точки оси.")
    grid = np.linspace(float(x[0]), float(x[-1]), x.size)
    return grid, np.vstack([np.interp(grid, x, row) for row in values])
