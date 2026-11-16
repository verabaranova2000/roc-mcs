from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (
    build_projection_grid, build_rectilinear_grid, roi_local_to_detector_matrix,
)
from roc_mcs.gui.roi_explorer.vtk_backend.h5_volume_source import _regrid_rotated_roi_to_detector
from roc_mcs.roi.roi_types import ROI


def _payload(volume=None, projection=None, x=None, y=None, theta=None):
    x = np.asarray([10., 12., 15.] if x is None else x, dtype=float)
    y = np.asarray([-2., 1., 5.] if y is None else y, dtype=float)
    theta = np.asarray([100., 101., 103., 106.] if theta is None else theta, dtype=float)
    volume = np.arange(x.size * y.size * theta.size, dtype=np.float32).reshape(x.size, y.size, theta.size) if volume is None else np.asarray(volume)
    projection = (np.sum(volume, axis=2) if volume.ndim == 3 and np.issubdtype(volume.dtype, np.number) else np.ones((x.size, y.size))) if projection is None else np.asarray(projection)
    return SimpleNamespace(volume=volume, projection=projection, x=x, y=y, theta=theta)


class TestRectilinearGrid:
    def test_coordinates_and_point_scalars_preserve_x_y_theta_mapping(self):
        p = _payload()
        grid, info = build_rectilinear_grid(p)
        assert grid.GetDimensions() == p.volume.shape
        np.testing.assert_allclose(vtk_to_numpy(grid.GetXCoordinates()), p.x)
        np.testing.assert_allclose(vtk_to_numpy(grid.GetYCoordinates()), p.y)
        np.testing.assert_allclose(vtk_to_numpy(grid.GetZCoordinates()), p.theta)
        scalars = vtk_to_numpy(grid.GetPointData().GetScalars())
        nx, ny, nz = p.volume.shape
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    point_id = i + nx * (j + ny * k)
                    assert scalars[point_id] == p.volume[i, j, k]
        assert info["point_count"] == p.volume.size
        assert info["finite_count"] == p.volume.size

    def test_nonfinite_scalars_get_transparent_sentinel_without_changing_payload(self):
        volume = np.ones((3, 3, 4), dtype=np.float32)
        volume[0, 1, 2] = np.nan
        volume[2, 0, 1] = np.inf
        p = _payload(volume=volume)
        before = p.volume.copy()
        grid, info = build_rectilinear_grid(p)
        scalars = vtk_to_numpy(grid.GetPointData().GetScalars())
        sentinel = info["invalid_sentinel"]
        assert info["invalid_count"] == 2
        assert np.count_nonzero(scalars == sentinel) == 2
        assert sentinel < info["scalar_min"]
        np.testing.assert_array_equal(np.isnan(p.volume), np.isnan(before))
        np.testing.assert_array_equal(np.isinf(p.volume), np.isinf(before))

    @pytest.mark.parametrize("p", [
        _payload(volume=np.ones((3, 3))),
        _payload(volume=np.ones((3, 3, 3)), x=[0., 1.]),
        _payload(volume=np.ones((1, 3, 3))),
        _payload(volume=np.ones((3, 3, 3)), x=[0., 0., 1.]),
        _payload(volume=np.ones((3, 3, 3)), theta=[0., np.nan, 2., 3.]),
        _payload(volume=np.full((3, 3, 4), np.nan)),
    ])
    def test_invalid_geometry_or_data_is_rejected(self, p):
        with pytest.raises((TypeError, ValueError)):
            build_rectilinear_grid(p)

    def test_complex_and_non_numeric_data_are_rejected(self):
        for dtype in (np.complex64, "U2"):
            p = _payload(volume=np.ones((3, 3, 4), dtype=dtype))
            with pytest.raises(TypeError):
                build_rectilinear_grid(p)


class TestProjectionGrid:
    def test_projection_scalars_are_xy_ordered_and_coplanar_with_low_theta(self):
        p = _payload()
        grid, info = build_projection_grid(p)
        assert grid.GetDimensions() == (p.x.size, p.y.size, 1)
        assert grid.GetBounds()[4:] == (p.theta[0], p.theta[0])
        scalars = vtk_to_numpy(grid.GetPointData().GetScalars())
        for j in range(p.y.size):
            for i in range(p.x.size):
                assert scalars[i + p.x.size * j] == p.projection[i, j]
        assert info["z_plane"] == p.theta[0]

    def test_invalid_projection_pixels_use_sentinel_and_source_is_unchanged(self):
        projection = np.ones((3, 3), dtype=float)
        projection[1, 2] = np.nan
        p = _payload(projection=projection)
        grid, info = build_projection_grid(p)
        scalars = vtk_to_numpy(grid.GetPointData().GetScalars())
        assert info["invalid_count"] == 1
        assert scalars[1 + p.x.size * 2] == info["invalid_sentinel"]
        assert np.isnan(p.projection[1, 2])

    def test_projection_rejects_shape_bad_axes_and_all_invalid_data(self):
        cases = [
            _payload(projection=np.ones((2, 2))),
            _payload(x=[0., 0., 1.]),
            _payload(theta=[100.]),
            _payload(projection=np.full((3, 3), np.nan)),
        ]
        for p in cases:
            with pytest.raises((TypeError, ValueError)):
                build_projection_grid(p)


class TestROIAffineGeometry:
    @pytest.mark.parametrize("angle", [0., 31., -45., 90., 179.])
    def test_roi_local_coordinates_transform_to_detector_xy_and_keep_theta(self, angle):
        roi = ROI(cx=42.5, cy=-7.25, width=12., height=8., angle=angle)
        matrix = roi_local_to_detector_matrix(roi)
        local = np.array([2.75, -1.5, 103.25, 1.])
        world = np.array([matrix.GetElement(r, c) for r in range(4) for c in range(4)]).reshape(4, 4) @ local
        expected_x, expected_y = roi.to_detector(local[0], local[1])
        np.testing.assert_allclose(world[:2], [expected_x, expected_y], atol=1e-12)
        assert world[2] == local[2]
        assert world[3] == 1.

    def test_none_roi_has_no_transform(self):
        assert roi_local_to_detector_matrix(None) is None

    def test_rotated_roi_regrid_preserves_linear_field_and_exact_footprint(self):
        roi = ROI(cx=5., cy=4., width=4., height=3., angle=31.)
        u = np.linspace(-2., 2., 9)
        v = np.linspace(-1.5, 1.5, 7)
        theta = np.array([10., 12., 15.])
        uu, vv = np.meshgrid(u, v, indexing="ij")
        source = np.stack([2. * uu + 3. * vv + 10. * k for k in range(theta.size)], axis=2)
        x, y, output = _regrid_rotated_roi_to_detector(
            SimpleNamespace(x=u, y=v, theta=theta, volume=source), roi
        )
        xx, yy = np.meshgrid(x, y, indexing="ij")
        local_u, local_v = roi.to_local(xx, yy)
        inside = roi.contains(xx, yy)
        expected = np.stack([2. * local_u + 3. * local_v + 10. * k for k in range(theta.size)], axis=2)
        assert np.array_equal(np.isfinite(output), np.broadcast_to(inside[:, :, None], output.shape))
        np.testing.assert_allclose(output[np.isfinite(output)], expected[np.isfinite(output)], atol=2e-5)
        assert np.all(np.diff(x) > 0.) and np.all(np.diff(y) > 0.)
