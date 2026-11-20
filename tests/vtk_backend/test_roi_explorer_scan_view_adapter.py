from __future__ import annotations

import numpy as np
import pytest

from roc_mcs.gui.roi_explorer.vtk_backend.scan_view_adapter import build_volume_payload
from roc_mcs.roi.roi_types import ROI


def test_adapter_preserves_yxt_values_geometry_and_source():
    source = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    before = source.copy()
    theta = np.linspace(-2.0, 2.0, 5)
    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = np.array([1.0, 2.0, 3.0])
    payload = build_volume_payload(source, theta, 2, x, y, scan_key="entry472")

    assert payload.volume.shape == (4, 3, 5)
    np.testing.assert_array_equal(payload.volume, source.transpose(1, 0, 2))
    np.testing.assert_array_equal(payload.projection, source.transpose(1, 0, 2).sum(axis=2))
    np.testing.assert_array_equal(payload.x, x)
    np.testing.assert_array_equal(payload.y, y)
    np.testing.assert_array_equal(payload.theta, theta)
    np.testing.assert_array_equal(source, before)
    assert not payload.volume.flags.writeable
    assert not payload.projection.flags.writeable


def test_adapter_masks_invalid_samples_before_projection():
    source = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    mask = np.ones(source.shape, dtype=bool)
    mask[1, 2, :] = False
    mask[0, 1, 1] = False
    payload = build_volume_payload(
        source, np.arange(4, dtype=float), 2, np.arange(3, dtype=float),
        np.arange(2, dtype=float), valid_mask=mask,
    )
    assert np.isnan(payload.volume[2, 1, :]).all()
    assert np.isnan(payload.projection[2, 1])
    assert np.isnan(payload.volume[1, 0, 1])
    assert payload.projection[1, 0] == pytest.approx(source[0, 1, [0, 2, 3]].sum())


def test_adapter_smoothing_is_display_copy_and_keeps_invalid_sites():
    source = np.zeros((5, 5, 4), dtype=np.float32)
    source[2, 2, :] = 100.0
    mask = np.ones(source.shape, dtype=bool)
    mask[0, 0, :] = False
    before = source.copy()
    payload = build_volume_payload(
        source, np.arange(4, dtype=float), 2, np.arange(5, dtype=float),
        np.arange(5, dtype=float), valid_mask=mask, smoothing_sigma=0.8,
    )
    assert payload.smoothing_sigma == pytest.approx(0.8)
    assert np.isnan(payload.volume[0, 0, :]).all()
    assert np.isnan(payload.projection[0, 0])
    assert payload.volume[2, 2, 1] < 100.0
    np.testing.assert_array_equal(source, before)


def test_adapter_rejects_invalid_smoothing_and_bad_axis():
    source = np.ones((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="smoothing_sigma"):
        build_volume_payload(source, np.arange(4), 2, np.arange(3), np.arange(2), smoothing_sigma=6)
    with pytest.raises(ValueError, match="ось θ"):
        build_volume_payload(source, np.arange(2), 1, np.arange(3), np.arange(2))


def test_scan_view_adapter_rejects_singleton_axes_before_vtk_setup():
    raw = np.ones((1, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="минимум 2 узла"):
        build_volume_payload(raw, np.arange(4), 2, np.arange(3), np.arange(1))


def test_scan_view_adapter_rejects_non_boolean_validity_mask():
    raw = np.ones((3, 4, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="bool dtype"):
        build_volume_payload(raw, np.arange(5), 2, np.arange(4), np.arange(3), valid_mask=np.ones(raw.shape, dtype=np.uint8))


@pytest.mark.parametrize("angle", [17.0, 31.5, 63.0, -28.0])
def test_adapter_regrids_rotated_roi_into_xy_theta_volume_without_axis_swap(angle):
    """Regression: rotated-ROI regrid must not put the 1-D X axis in volume."""
    ny, nx, nt = 44, 58, 88
    theta = np.linspace(-1.25, 1.25, nt)
    x = np.linspace(-14.0, 14.5, nx)
    y = np.linspace(-10.0, 11.5, ny)
    yy, xx, tt = np.meshgrid(y, x, theta, indexing="ij")
    # Trilinear data make the interpolation result independently checkable.
    raw = (1.7 * xx - 0.8 * yy + 2.3 * tt).astype(np.float64)
    roi = ROI(
        cx=float(x[nx // 2]), cy=float(y[ny // 2]),
        width=17.0, height=9.0, angle=angle,
        label="rotated regression", type="rect",
    )

    payload = build_volume_payload(raw, theta, 2, x, y, roi=roi)

    assert payload.volume.ndim == 3
    assert payload.volume.shape == (payload.x.size, payload.y.size, payload.theta.size)
    assert payload.theta.size == nt
    assert payload.coord_system == "xy"
    assert np.all(np.diff(payload.x) > 0)
    assert np.all(np.diff(payload.y) > 0)
    assert np.all(np.diff(payload.theta) > 0)

    grid_x, grid_y = np.meshgrid(payload.x, payload.y, indexing="ij")
    inside = roi.contains(grid_x, grid_y)
    assert np.array_equal(np.isfinite(payload.volume).any(axis=2), inside)
    # Interior samples must retain the physical detector-coordinate geometry.
    ix, iy = np.argwhere(inside)[len(np.argwhere(inside)) // 2]
    expected = 1.7 * payload.x[ix] - 0.8 * payload.y[iy] + 2.3 * theta
    np.testing.assert_allclose(payload.volume[ix, iy], expected, rtol=1e-10, atol=1e-10)


def test_adapter_rotated_roi_keeps_detector_geometry_and_reverses_axes_consistently():
    ny, nx, nt = 32, 40, 9
    theta = np.linspace(2.0, -2.0, nt)
    x = np.linspace(8.0, -8.0, nx)
    y = np.linspace(6.0, -6.0, ny)
    # Coordinate-coded values expose accidental axis flips/permutations.
    yy, xx, tt = np.meshgrid(y, x, theta, indexing="ij")
    raw = (1000.0 * xx + 10.0 * yy + tt).astype(np.float64)
    roi = ROI(cx=0.0, cy=0.0, width=8.0, height=4.0, angle=24.0)

    payload = build_volume_payload(raw, theta, 2, x, y, roi=roi)

    assert payload.volume.shape == (payload.x.size, payload.y.size, nt)
    assert payload.x[0] < payload.x[-1]
    assert payload.y[0] < payload.y[-1]
    np.testing.assert_array_equal(payload.theta, theta[::-1])
    grid_x, grid_y = np.meshgrid(payload.x, payload.y, indexing="ij")
    assert np.array_equal(np.isfinite(payload.volume).any(axis=2), roi.contains(grid_x, grid_y))


@pytest.mark.parametrize("alias", ["rectangle", "rotated_rect", "rotated_rectangle"])
def test_adapter_canonicalizes_legacy_rectangle_aliases_without_mutating_saved_roi(alias):
    raw = np.ones((24, 28, 7), dtype=np.float64)
    roi = ROI(cx=13.5, cy=11.5, width=12.0, height=6.0, angle=23.0, type=alias)

    payload = build_volume_payload(
        raw, np.linspace(-1.0, 1.0, 7), 2, np.arange(28), np.arange(24), roi=roi,
    )

    assert payload.volume.ndim == 3
    assert payload.volume.shape == (payload.x.size, payload.y.size, payload.theta.size)
    assert payload.roi.type == "rect"
    assert roi.type == alias  # session-owned ROI object is not modified


def test_adapter_rejects_degenerate_or_nonfinite_rotated_roi_geometry():
    raw = np.ones((12, 14, 5), dtype=np.float64)
    theta = np.arange(5, dtype=np.float64)
    for roi in (
        ROI(cx=4.0, cy=5.0, width=0.0, height=3.0, angle=10.0),
        ROI(cx=np.nan, cy=5.0, width=3.0, height=2.0, angle=10.0),
    ):
        with pytest.raises(ValueError, match="ROI"):
            build_volume_payload(raw, theta, 2, np.arange(14), np.arange(12), roi=roi)
