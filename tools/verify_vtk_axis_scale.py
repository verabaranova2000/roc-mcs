"""Regression smoke test for display-only X/Y/theta scaling in the VTK backend.

Run from the roc-mcs repository root:
    python tools/verify_vtk_axis_scale.py

Requires NumPy and VTK, but does not require PyQt6 or an HDF5 scan.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import vtk  # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (  # noqa: E402
    TransferSettings,
    VTKVolumeScene,
)


def _grid_axes(grid):
    return tuple(
        vtk_to_numpy(array).astype(np.float64, copy=True)
        for array in (grid.GetXCoordinates(), grid.GetYCoordinates(), grid.GetZCoordinates())
    )


def _scaled(source: np.ndarray, factor: float) -> np.ndarray:
    center = 0.5 * (float(source[0]) + float(source[-1]))
    return center + (source - center) * factor


def main() -> int:
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    scene = VTKVolumeScene(render_window, interactor, render_callback=lambda: None)

    x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y = np.asarray([10.0, 12.0, 14.0, 16.0], dtype=np.float64)
    theta = np.asarray([30.0, 31.0, 33.0, 36.0, 40.0, 45.0], dtype=np.float64)
    volume = np.arange(x.size * y.size * theta.size, dtype=np.float32).reshape(
        (x.size, y.size, theta.size)
    ) + 1.0
    projection = volume.mean(axis=2)
    payload = SimpleNamespace(
        x=x.copy(), y=y.copy(), theta=theta.copy(), volume=volume.copy(),
        projection=projection.copy(), coord_system="xy", roi=None,
    )
    source_axes = (x.copy(), y.copy(), theta.copy())
    source_volume = volume.copy()

    try:
        scene.set_payload(
            payload,
            TransferSettings(sample_distance_factor=1.25),
            axis_scales=(1.0, 1.0, 1.0),
        )
        original_grid = _grid_axes(scene.grid)
        original_projection = _grid_axes(scene.projection_grid)

        for factors in ((2.0, 0.5, 3.0), (0.7, 2.5, 0.4), (1.0, 1.0, 1.0), (2.0, 0.5, 3.0), (1.0, 1.0, 1.0)):
            scene.set_axis_scales(*factors, render=False)
            grid_axes = _grid_axes(scene.grid)
            projection_axes = _grid_axes(scene.projection_grid)
            expected = tuple(_scaled(axis, factor) for axis, factor in zip(source_axes, factors))
            for actual, reference in zip(grid_axes, expected):
                np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-12)
            for actual, reference in zip(projection_axes, (expected[0], expected[1], expected[2][:1])):
                np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-12)
            np.testing.assert_array_equal(payload.volume, source_volume)
            for actual, reference in zip((payload.x, payload.y, payload.theta), source_axes):
                np.testing.assert_array_equal(actual, reference)
            spacing = min(
                float(np.min(np.diff(axis))) * factor
                for axis, factor in zip(source_axes, factors)
            )
            assert abs(scene._min_spacing - spacing) < 1e-12
            assert abs(scene.mapper.GetSampleDistance() - spacing * 1.25) < 1e-12

        # Test the shader-space cursor remapping without depending on a GL context.
        class UniformSpy:
            def __init__(self):
                self.values = {}

            def SetUniformf(self, name, value):
                self.values[name] = float(value)

            def SetUniformi(self, name, value):
                self.values[name] = int(value)

        uniforms = UniformSpy()
        scene._theta_slice_uniforms = uniforms
        scene.set_cursor_line_position(0.5, 13.0, 0.2, 0.4, render=False)
        scene.set_axis_scales(2.0, 0.5, 1.5, render=False)
        center_x = 0.5 * (source_axes[0][0] + source_axes[0][-1])
        center_y = 0.5 * (source_axes[1][0] + source_axes[1][-1])
        assert abs(uniforms.values["rocCursorX"] - (center_x + (0.5 - center_x) * 2.0)) < 1e-12
        assert abs(uniforms.values["rocCursorY"] - (center_y + (13.0 - center_y) * 0.5)) < 1e-12
        assert abs(uniforms.values["rocCursorRadiusX"] - 0.4) < 1e-12
        assert abs(uniforms.values["rocCursorRadiusY"] - 0.2) < 1e-12

        # Force a renderer-stage error and verify that both grids and scale state roll back.
        prior_grid = _grid_axes(scene.grid)
        prior_projection = _grid_axes(scene.projection_grid)
        prior_scales = scene.axis_scales
        prior_sample_distance = float(scene.mapper.GetSampleDistance())

        class RendererProxy:
            def __init__(self, real_renderer):
                self._real = real_renderer

            def __getattr__(self, name):
                if name == "ResetCameraClippingRange":
                    def fail(*_args):
                        raise RuntimeError("injected rollback test")
                    return fail
                return getattr(self._real, name)

        real_renderer = scene.renderer
        scene.renderer = RendererProxy(real_renderer)
        try:
            scene.set_axis_scales(3.0, 2.0, 0.6, render=False)
        except RuntimeError as exc:
            assert "injected rollback test" in str(exc)
        else:
            raise AssertionError("Expected injected renderer failure")
        finally:
            scene.renderer = real_renderer

        for actual, reference in zip(_grid_axes(scene.grid), prior_grid):
            np.testing.assert_array_equal(actual, reference)
        for actual, reference in zip(_grid_axes(scene.projection_grid), prior_projection):
            np.testing.assert_array_equal(actual, reference)
        assert scene.axis_scales == prior_scales
        assert abs(float(scene.mapper.GetSampleDistance()) - prior_sample_distance) < 1e-12

        scene._theta_slice_uniforms = None
        scene.set_axis_scales(1.0, 1.0, 1.0, render=False)
        for actual, reference in zip(_grid_axes(scene.grid), original_grid):
            np.testing.assert_array_equal(actual, reference)
        for actual, reference in zip(_grid_axes(scene.projection_grid), original_projection):
            np.testing.assert_array_equal(actual, reference)
        print(f"PASS: VTK {vtk.vtkVersion.GetVTKVersion()} — stable X/Y/θ scale, XY projection, cursor, data immutability and rollback.")
        return 0
    finally:
        scene.clear(render=False)
        render_window.Finalize()


if __name__ == "__main__":
    raise SystemExit(main())
