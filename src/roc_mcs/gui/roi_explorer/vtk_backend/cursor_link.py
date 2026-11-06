"""Coordinate bridge between an ROI Explorer map cursor and a VTK volume.

Public contract: all incoming positions use detector-map coordinates (X, Y).
The returned coordinates use the current VTK dataset-local X/Y lattice. The
module is deliberately independent of Qt, VTK rendering and volume arrays so a
future embedded viewer can reuse exactly the same mapping path.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any

import numpy as np

from roc_mcs.gui.roi_explorer.volume_rendering_runtime import (
    map_detector_cursor_to_volume_xy,
)


@dataclass(frozen=True, slots=True)
class VolumeCursorTarget:
    """Resolved map cursor and a locally-scaled tube radius for GPU shading."""

    detector_x: float
    detector_y: float
    volume_x: float
    volume_y: float
    radius_x: float
    radius_y: float


def _axis_local_spacing(axis: Any, coordinate: float) -> float:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Ось объёма должна содержать минимум два координатных узла.")
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
        raise ValueError("Координатная ось должна быть конечной и строго возрастающей.")
    index = int(np.clip(np.searchsorted(values, float(coordinate), side="left"), 0, values.size - 1))
    steps: list[float] = []
    if index > 0:
        steps.append(float(values[index] - values[index - 1]))
    if index < values.size - 1:
        steps.append(float(values[index + 1] - values[index]))
    positive = [step for step in steps if np.isfinite(step) and step > 0.0]
    if not positive:
        raise ValueError("Не удалось определить локальный шаг координатной оси.")
    # Use the narrower adjacent cell for a conservative footprint on a
    # nonuniform grid; the line cannot spill over a locally dense region.
    return min(positive)


def map_detector_cursor_to_volume(
    payload: Any,
    detector_x: float,
    detector_y: float,
    *,
    line_width_voxels: float = 1.6,
) -> VolumeCursorTarget | None:
    """Map detector XY into local grid XY, returning None outside current ROI.

    The detector→local conversion is the same function used by the Plotly
    cursor integration. For rotated ROI data it calls ROI.contains/to_local;
    no second, competing rotation convention is introduced here.
    """
    x = float(detector_x)
    y = float(detector_y)
    width = float(line_width_voxels)
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError("Координаты курсора должны быть конечными.")
    if not (np.isfinite(width) and 0.5 <= width <= 4.0):
        raise ValueError("Ширина линии должна находиться в диапазоне [0.5, 4.0] вокселя.")

    # ROI-backed VTK payloads may use a detector-aligned display lattice even
    # when the source ROI was rotated and sampled in local (u,v).  In that
    # representation the cursor coordinates are already detector X/Y, but
    # the original ROI still defines the valid footprint.  Apply that exact
    # geometry gate before the optional UV conversion below.
    roi = getattr(payload, "roi", None)
    if roi is not None:
        contains = getattr(roi, "contains", None)
        if callable(contains) and not bool(contains(x, y)):
            return None

    local = map_detector_cursor_to_volume_xy(
        x,
        y,
        coord_system=str(getattr(payload, "coord_system", "xy")),
        overlay_full_detector=False,
        roi=roi,
    )
    if local is None:
        return None
    vx, vy = float(local[0]), float(local[1])
    x_axis = np.asarray(payload.x, dtype=np.float64)
    y_axis = np.asarray(payload.y, dtype=np.float64)
    sx = _axis_local_spacing(x_axis, vx)
    sy = _axis_local_spacing(y_axis, vy)

    # The first/last half-cell is assigned to the edge sample. Beyond that
    # support the cursor is hidden instead of snapping to a misleading edge.
    x_low, x_high = float(x_axis[0] - 0.5 * sx), float(x_axis[-1] + 0.5 * sx)
    y_low, y_high = float(y_axis[0] - 0.5 * sy), float(y_axis[-1] + 0.5 * sy)
    if vx < x_low or vx > x_high or vy < y_low or vy > y_high:
        return None
    vx = float(np.clip(vx, x_axis[0], x_axis[-1]))
    vy = float(np.clip(vy, y_axis[0], y_axis[-1]))
    return VolumeCursorTarget(
        detector_x=x,
        detector_y=y,
        volume_x=vx,
        volume_y=vy,
        radius_x=0.5 * width * sx,
        radius_y=0.5 * width * sy,
    )


def detector_cursor_bounds(payload: Any) -> tuple[float, float, float, float]:
    """Detector-frame bounds for standalone X/Y sliders.

    A rotated local ROI uses its transformed four corners. The slider domain
    is then the enclosing axis-aligned rectangle; coordinates inside that
    rectangle but outside the true ROI are rejected by the mapper above.
    """
    if str(getattr(payload, "coord_system", "xy")).lower() == "uv":
        roi = getattr(payload, "roi", None)
        if roi is None:
            raise ValueError("Для локальной ROI-сетки отсутствует геометрия ROI.")
        cx, cy = float(roi.cx), float(roi.cy)
        half_width, half_height = 0.5 * float(roi.width), 0.5 * float(roi.height)
        angle = radians(float(roi.angle))
        c, s = cos(angle), sin(angle)
        corners = [
            (cx + c * u - s * v, cy + s * u + c * v)
            for u, v in (
                (-half_width, -half_height),
                (-half_width, half_height),
                (half_width, -half_height),
                (half_width, half_height),
            )
        ]
        values = np.asarray(corners, dtype=np.float64)
        return (
            float(np.min(values[:, 0])), float(np.max(values[:, 0])),
            float(np.min(values[:, 1])), float(np.max(values[:, 1])),
        )
    x_axis = np.asarray(payload.x, dtype=np.float64)
    y_axis = np.asarray(payload.y, dtype=np.float64)
    if x_axis.ndim != 1 or y_axis.ndim != 1 or not x_axis.size or not y_axis.size:
        raise ValueError("Для настройки курсора нужны непустые оси X и Y.")
    return (
        float(np.min(x_axis)), float(np.max(x_axis)),
        float(np.min(y_axis)), float(np.max(y_axis)),
    )
