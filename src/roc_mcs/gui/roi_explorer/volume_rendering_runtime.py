"""Live display updates for an already-built Plotly volume scene.

This module deliberately contains no Qt or WebEngine code. It builds small,
JSON-safe style patches that can be applied to existing Plotly traces without
re-extracting, resampling, serializing, or transferring the scientific volume.
"""
from __future__ import annotations

from math import isfinite
from typing import Any

from roc_mcs.gui.roi_explorer.colormap_catalog import plotly_colorscale

# Changing one of these values does not change the sampled scientific data or
# its coordinate lattice. It can be applied to the resident Plotly traces.
LIVE_DISPLAY_SETTINGS = frozenset(
    {
        "font_family",
        "tick_size",
        "label_size",
        "axis_tick_length",
        "show_grid",
        "show_projection",
        "show_colorbar",
        "show_legend",
        "volume_colormap",
        "projection_colorscale",
        "volume_opacity",
        "volume_opacity_mode",
        "projection_opacity",
        "isomin_fraction",
        "isomax_fraction",
        "surface_count",
        "dragmode",
        "projection_type",
        "rotation_speed",
    }
)

# These settings affect the voxel lattice or values sent to the renderer.
# Their changes must rebuild the scene; pretending otherwise would be wrong.
DATA_RENDER_SETTINGS = frozenset(
    {"smooth_sigma", "volume_stride", "quality", "max_render_points"}
)


# Plotly Volume supports a scalar opacity transfer function.  The legacy mode
# intentionally stays the default; the soft mode fades low in-range values
# progressively instead of giving every isosurface the same opacity.
_SOFT_VOLUME_OPACITY_SCALE = (
    (0.00, 0.00),
    (0.10, 0.00),
    (0.22, 0.015),
    (0.36, 0.055),
    (0.52, 0.16),
    (0.68, 0.36),
    (0.84, 0.68),
    (1.00, 1.00),
)


def map_detector_cursor_to_volume_xy(
    x: float,
    y: float,
    *,
    coord_system: str,
    overlay_full_detector: bool,
    roi: Any = None,
) -> tuple[float, float] | None:
    """Map detector XY cursor coordinates to the current volume XY lattice.

    Rotated ROI volumes use local (u, v) coordinates unless they have been
    regridded onto the full detector. Points outside a local ROI return None.
    This helper only transforms two scalars; it never touches volume data.
    """
    xx, yy = float(x), float(y)
    if not (isfinite(xx) and isfinite(yy)):
        return None
    if str(coord_system).lower() == "uv" and not bool(overlay_full_detector):
        if roi is None or not bool(roi.contains(xx, yy)):
            return None
        u, v = roi.to_local(xx, yy)
        uu, vv = float(u), float(v)
        if not (isfinite(uu) and isfinite(vv)):
            return None
        return uu, vv
    return xx, yy


def volume_opacity_scale(mode: Any = "uniform") -> str | list[list[float]]:
    """Return a Plotly Volume opacity transfer function.

    ``uniform`` preserves the existing appearance. ``soft`` provides a
    continuous, monotonic fade-in from transparent weak values to the full
    configured surface opacity. The transfer function is normalized by
    Plotly to the trace value domain; it does not alter scientific samples.
    """
    if str(mode).strip().lower() != "soft":
        return "uniform"
    return [[float(x), float(alpha)] for x, alpha in _SOFT_VOLUME_OPACITY_SCALE]

_MISSING = object()


def _same_value(left: Any, right: Any) -> bool:
    """Compare settings conservatively without ambiguous NumPy truth values."""
    try:
        result = left == right
        shape = getattr(result, "shape", None)
        if shape not in (None, ()):
            return False
        return bool(result)
    except Exception:
        return left is right


def changed_setting_keys(previous: dict, current: dict) -> set[str]:
    """Return the setting keys whose values differ."""
    old = previous if isinstance(previous, dict) else {}
    new = current if isinstance(current, dict) else {}
    return {
        key
        for key in old.keys() | new.keys()
        if not _same_value(old.get(key, _MISSING), new.get(key, _MISSING))
    }


def can_update_live_display(previous: dict, current: dict) -> bool:
    """Return True only when all changed keys are known display-only settings.

    Unknown settings conservatively force a full rebuild. This is intentional:
    correctness takes precedence over accidentally leaving a setting unapplied.
    """
    return changed_setting_keys(previous, current).issubset(LIVE_DISPLAY_SETTINGS)


def _finite_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not isfinite(number):
        return default
    return max(low, min(high, number))


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    """Parse a bounded integer without allowing malformed settings to abort a UI update."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(low, min(high, number))


_SETTING_TO_PATCH_KEYS = {
    "font_family": {"font_family"},
    "tick_size": {"tick_size"},
    "label_size": {"label_size"},
    "axis_tick_length": {"axis_tick_length"},
    "show_grid": {"show_grid"},
    "show_projection": {"show_projection"},
    "show_colorbar": {"show_colorbar"},
    "show_legend": {"show_legend"},
    "volume_colormap": {"volume_colorscale"},
    "projection_colorscale": {"projection_colorscale"},
    "volume_opacity": {"volume_opacity"},
    "volume_opacity_mode": {"volume_opacityscale"},
    "projection_opacity": {"projection_opacity"},
    "isomin_fraction": {"isomin_fraction"},
    "isomax_fraction": {"isomax_fraction"},
    "surface_count": {"surface_count"},
    "dragmode": {"dragmode"},
    "projection_type": {"projection_type"},
    "rotation_speed": {"rotation_speed"},
}
_THEME_PATCH_KEYS = {"background", "foreground", "grid", "axisline"}


def build_live_display_patch(
    settings: dict,
    theme: str = "light",
    changed_settings: set[str] | None = None,
) -> dict:
    """Build a compact Plotly patch; optionally include only affected properties.

    ``changed_settings=None`` produces a complete patch (used after WebEngine
    reload). An explicit set limits the patch to the corresponding trace/layout
    properties so, for example, a contrast drag does not reapply every axis
    font, colorbar and colorscale.
    """
    values = settings if isinstance(settings, dict) else {}
    dark = str(theme).lower() == "dark"

    background = "#111316" if dark else "#ffffff"
    foreground = "#e1e4e8" if dark else "#111111"
    grid = "#343941" if dark else "#e0e0e0"
    axisline = "#aeb5be" if dark else "#222222"

    low = _finite_float(values.get("isomin_fraction", 0.16), 0.16, 0.0, 0.99)
    high = _finite_float(values.get("isomax_fraction", 0.95), 0.95, 0.01, 1.0)
    if high <= low:
        high = min(1.0, low + 0.005)

    volume_map = str(values.get("volume_colormap", "magma"))
    projection_map = str(values.get("projection_colorscale", "Greys"))
    if dark and projection_map.lower() in {"slate", "greys"}:
        projection_map = "Greys_r"

    patch = {
        "volume_colorscale": plotly_colorscale(volume_map),
        "isomin_fraction": low,
        "isomax_fraction": high,
        "volume_opacity": _finite_float(values.get("volume_opacity", 0.10), 0.10, 0.01, 1.0),
        "volume_opacityscale": volume_opacity_scale(values.get("volume_opacity_mode", "uniform")),
        "projection_opacity": _finite_float(values.get("projection_opacity", 0.75), 0.75, 0.0, 1.0),
        "surface_count": _bounded_int(values.get("surface_count", 25), 25, 2, 25),
        "show_projection": bool(values.get("show_projection", True)),
        "show_colorbar": bool(values.get("show_colorbar", False)),
        "show_legend": bool(values.get("show_legend", False)),
        "show_grid": bool(values.get("show_grid", True)),
        "dragmode": "turntable" if values.get("dragmode") == "turntable" else "orbit",
        "projection_type": "orthographic" if values.get("projection_type") == "orthographic" else "perspective",
        "background": background,
        "foreground": foreground,
        "grid": grid,
        "axisline": axisline,
        "font_family": str(values.get("font_family", "Times New Roman")),
        "tick_size": _bounded_int(values.get("tick_size", 12), 12, 6, 16),
        "label_size": _bounded_int(values.get("label_size", 15), 15, 6, 18),
        "axis_tick_length": float(_finite_float(values.get("axis_tick_length", 3.0), 3.0, 0.0, 10.0)),
        "rotation_speed": float(_finite_float(values.get("rotation_speed", 14.0), 14.0, 1.0, 90.0)),
    }
    # A full replay deliberately omits projection colorscale: the complete
    # Plotly figure already contains the exact projection styling. Reapplying a
    # derived light/dark colorscale here used to recolor the 2D projection when
    # only the theme or the 3D volume palette changed.
    if changed_settings is None:
        return patch

    requested = set(changed_settings)
    patch_keys: set[str] = set()
    for setting in requested:
        patch_keys.update(_SETTING_TO_PATCH_KEYS.get(setting, ()))
    if "theme" in requested:
        patch_keys.update(_THEME_PATCH_KEYS)

    # Only the dedicated 2D-projection setting may alter its palette. In
    # particular, theme changes must not silently invert Greys/Slate.
    if "projection_colorscale" in requested:
        patch["projection_colorscale"] = projection_map
    return {key: value for key, value in patch.items() if key in patch_keys}
