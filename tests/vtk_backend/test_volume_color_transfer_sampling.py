from __future__ import annotations

import vtk

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (
    _MAX_VOLUME_COLOR_TEXTURE_WIDTH,
    _set_adaptive_volume_palette,
    _palette_samples,
)


def _estimated_width(function, scalar_min: float, scalar_max: float) -> int:
    required = max(1, int(function.EstimateMinNumberOfSamples(scalar_min, scalar_max)))
    return max(1, int(vtk.vtkMath.NearestPowerOfTwo(required)))


def test_adaptive_palette_avoids_the_32768_texture_fallback_case():
    # A robust display interval occupying 0.2% of the full scalar range is a
    # representative case for the VTK warning observed in the application:
    # 64 control points estimate 32768 texels, while the driver's limit is 16384.
    function = vtk.vtkColorTransferFunction()
    original_colors = _palette_samples("inferno", 64)
    for index, color in enumerate(original_colors):
        value = 0.002 * index / 63.0
        function.AddRGBPoint(value, *map(float, color))
    assert _estimated_width(function, 0.0, 1.0) == 32768

    info = _set_adaptive_volume_palette(
        function,
        "inferno",
        0.0,
        0.002,
        0.0,
        1.0,
        vtk.vtkMath,
    )

    assert info["control_points"] == 32
    assert info["estimated_texture_width"] <= _MAX_VOLUME_COLOR_TEXTURE_WIDTH
    assert function.GetSize() == info["control_points"]
    low = function.GetColor(0.0)
    high = function.GetColor(0.002)
    assert all(0.0 <= channel <= 1.0 for channel in (*low, *high))


def test_adaptive_palette_keeps_full_fidelity_when_scalar_range_is_not_outlier_dominated():
    function = vtk.vtkColorTransferFunction()
    info = _set_adaptive_volume_palette(
        function,
        "viridis",
        -1.0,
        1.0,
        -1.0,
        1.0,
        vtk.vtkMath,
    )
    assert info["control_points"] == 64
    assert info["estimated_texture_width"] <= _MAX_VOLUME_COLOR_TEXTURE_WIDTH
    assert function.GetSize() == 64
