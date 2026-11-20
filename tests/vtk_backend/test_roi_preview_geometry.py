from __future__ import annotations

import numpy as np
import pytest

from roc_mcs.gui.roi_explorer.vtk_backend.roi_preview_geometry import canonicalize_roi_preview
from roc_mcs.roi.roi_types import ROI, RotatedRectROI, LineROI, PointROI


def test_selector_rotated_rectangle_is_converted_to_canonical_corners():
    source = RotatedRectROI(cx=10.0, cy=20.0, width=8.0, height=4.0, angle=30.0)
    canonical, kind, points = canonicalize_roi_preview(source)
    expected = ROI.from_selector_roi(source).corners()
    assert kind == "rect"
    np.testing.assert_allclose(points, expected, rtol=0, atol=1e-12)
    assert points.shape == (4, 2)
    assert canonical.type == "rect"


def test_selector_line_and_point_are_supported_without_corners_method_on_source():
    line = LineROI(1.0, 2.0, 5.0, 6.0)
    _, kind, points = canonicalize_roi_preview(line)
    assert kind == "line"
    np.testing.assert_allclose(points, [[1.0, 2.0], [5.0, 6.0]])

    point = PointROI(3.5, -2.0)
    _, kind, points = canonicalize_roi_preview(point)
    assert kind == "point"
    np.testing.assert_allclose(points, [[3.5, -2.0]])


def test_canonical_roi_is_supported_and_explicit_type_must_match():
    source = ROI(5.0, 6.0, 4.0, 2.0, angle=15.0, type="rect")
    canonical, kind, points = canonicalize_roi_preview(source, "rectangle")
    assert kind == "rect"
    assert canonical is not source
    assert points.shape == (4, 2)
    with pytest.raises(ValueError, match="не соответствует"):
        canonicalize_roi_preview(source, "line")


def test_invalid_or_unknown_selector_geometry_is_rejected():
    with pytest.raises(ValueError, match="Ширина и высота"):
        canonicalize_roi_preview(RotatedRectROI(0.0, 0.0, 0.0, 2.0))
    with pytest.raises(TypeError, match="Неподдерживаемый тип"):
        canonicalize_roi_preview(object())
