"""Преобразование ROI-селекторов карты в единую геометрию VTK preview."""
from __future__ import annotations

import numpy as np

from roc_mcs.roi.roi_types import ROI, RotatedRectROI, LineROI, PointROI


_TYPE_ALIASES = {
    "rect": "rect", "rectangle": "rect", "rotated_rect": "rect", "rotated_rectangle": "rect",
    "line": "line", "point": "point",
}


def canonicalize_roi_preview(roi, selector_type: str | None = None) -> tuple[ROI, str, np.ndarray]:
    """Вернуть каноническую ROI и её точки в координатах детектора XY."""
    if isinstance(roi, ROI):
        canonical = roi.normalized()
    elif isinstance(roi, (RotatedRectROI, LineROI, PointROI)):
        canonical = ROI.from_selector_roi(roi).normalized()
    else:
        raise TypeError(f"Неподдерживаемый тип ROI preview: {type(roi).__name__}.")

    kind = _TYPE_ALIASES.get(str(canonical.type).strip().lower())
    if kind is None:
        raise ValueError(f"Неизвестный тип ROI preview: {canonical.type!r}.")
    if selector_type is not None:
        requested = _TYPE_ALIASES.get(str(selector_type).strip().lower())
        if requested is None or requested != kind:
            raise ValueError(f"Тип селектора {selector_type!r} не соответствует ROI типа {kind!r}.")

    values = (canonical.cx, canonical.cy, canonical.width, canonical.height, canonical.angle)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("Координаты и размеры ROI preview должны быть конечными числами.")
    if kind == "rect" and (canonical.width <= 0.0 or canonical.height <= 0.0):
        raise ValueError("Ширина и высота прямоугольной ROI должны быть больше нуля.")
    if kind == "line" and canonical.width <= 0.0:
        raise ValueError("Длина линии ROI должна быть больше нуля.")
    if kind == "point" and (canonical.width != 0.0 or canonical.height != 0.0):
        raise ValueError("Для точечной ROI ширина и высота должны быть равны нулю.")

    points = np.asarray(canonical.corners(), dtype=np.float64).reshape(-1, 2)
    expected = {"rect": (4, 2), "line": (2, 2), "point": (1, 2)}[kind]
    if points.shape != expected or not np.all(np.isfinite(points)):
        raise ValueError(f"Некорректная геометрия {kind} ROI preview: shape={points.shape}.")
    return canonical, kind, points
