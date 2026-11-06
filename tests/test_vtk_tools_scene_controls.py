from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import vtk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import TransferSettings, VTKVolumeScene


class _VisibilityProbe:
    def __init__(self):
        self.visible = None
        self.enabled = None

    def SetVisibility(self, value):
        self.visible = int(value)

    def SetEnabled(self, value):
        self.enabled = int(value)


class VTKSceneToolbarControlTests(unittest.TestCase):
    @staticmethod
    def _payload():
        from types import SimpleNamespace
        x = np.linspace(-2.0, 2.0, 5)
        y = np.linspace(-2.0, 2.0, 5)
        theta = np.linspace(-2.0, 2.0, 5)
        volume = np.ones((5, 5, 5), dtype=np.float32)
        return SimpleNamespace(
            x=x, y=y, theta=theta, volume=volume, projection=np.sum(volume, axis=2),
            coord_system="xy", roi=None, scan_key="toolbar-test", view_mode="full",
            scalar_dtype="float32", finite_count=int(volume.size), invalid_count=0,
            nbytes=int(volume.nbytes), masked_detector_pixels=(),
            smoothing_sigma=0.0,
            spike_report=SimpleNamespace(enabled=False, kappa=0.0, invalid_count=0,
                                        invalid_fraction=0.0, sigma=0.0),
        )

    @staticmethod
    def _make_scene():
        render_window = vtk.vtkRenderWindow()
        interactor = vtk.vtkGenericRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        return render_window, interactor, VTKVolumeScene(render_window, interactor)

    @staticmethod
    def _close_scene(render_window, interactor, scene):
        scene.close()
        interactor.SetRenderWindow(None)
        render_window.Finalize()

    def _camera_scene(self):
        renderer = vtk.vtkRenderer()
        cube = vtk.vtkCubeSource()
        cube.SetBounds(-4, 8, -6, 10, -2, 12)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)

        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(50, 60, 70)
        camera.SetPosition(70, 90, 100)
        camera.SetViewUp(0, 0, 1)

        scene = object.__new__(VTKVolumeScene)
        scene.payload = object()
        scene.renderer = renderer
        scene._update_geometry_axes = lambda **_kwargs: None
        scene._request_render = lambda: None
        return scene, camera

    def test_cursor_color_updates_only_existing_uniform(self):
        scene = object.__new__(VTKVolumeScene)
        scene._cursor_line_color = "#FFBD55"
        scene._cursor_line_enabled = True
        scene._theta_slice_uniforms = vtk.vtkOpenGLUniforms()
        scene._theta_slice_uniforms.SetUniform3f("rocCursorLineColor", (1.0, 0.7411765, 0.3333333))
        scene._theta_slice_uniforms.SetUniformf("unrelatedDensity", 0.95)
        scene._request_render = lambda: self.fail("render=False must not request a render")

        before_list_mtime = scene._theta_slice_uniforms.GetUniformListMTime()
        scene.set_cursor_line_color("#1234AB", render=False)

        self.assertEqual(scene.cursor_line_color, "#1234AB")
        rgb = [0.0, 0.0, 0.0]
        self.assertTrue(scene._theta_slice_uniforms.GetUniform3f("rocCursorLineColor", rgb))
        np.testing.assert_allclose(rgb, (18 / 255, 52 / 255, 171 / 255), atol=1e-7)
        unrelated = vtk.reference(0.0)
        self.assertTrue(scene._theta_slice_uniforms.GetUniformf("unrelatedDensity", unrelated))
        self.assertAlmostEqual(float(unrelated), 0.95)
        self.assertEqual(scene._theta_slice_uniforms.GetUniformListMTime(), before_list_mtime)

    def test_cursor_color_is_initialized_in_shader_and_survives_payload_updates(self):
        render_window, interactor, scene = self._make_scene()
        try:
            scene.set_cursor_line_color("#2468AC", render=False)
            scene.set_payload(self._payload(), TransferSettings())
            rgb = [0.0, 0.0, 0.0]
            self.assertTrue(scene._theta_slice_uniforms.GetUniform3f("rocCursorLineColor", rgb))
            np.testing.assert_allclose(rgb, (36 / 255, 104 / 255, 172 / 255), atol=1e-7)
            self.assertEqual(scene.cursor_line_color, "#2468AC")
        finally:
            self._close_scene(render_window, interactor, scene)

    def test_cursor_color_rejects_invalid_values_without_mutating_state(self):
        scene = object.__new__(VTKVolumeScene)
        scene._cursor_line_color = "#FFBD55"
        scene._cursor_line_enabled = False
        scene._theta_slice_uniforms = vtk.vtkOpenGLUniforms()
        scene._theta_slice_uniforms.SetUniform3f("rocCursorLineColor", (1.0, 0.7, 0.3))

        for invalid in ("red", "FF00AA", "#12GG00", "#12345", "#12345678"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    scene.set_cursor_line_color(invalid, render=False)
                self.assertEqual(scene.cursor_line_color, "#FFBD55")

    def test_fit_preserves_direction_and_parallel_projection(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                scene, camera = self._camera_scene()
                camera.SetParallelProjection(int(parallel))
                if parallel:
                    camera.SetParallelScale(500.0)

                old_direction = np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
                old_direction /= np.linalg.norm(old_direction)
                old_view_up = np.asarray(camera.GetViewUp())

                scene.reset_camera_fit_preserve_direction()

                new_direction = np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
                new_direction /= np.linalg.norm(new_direction)
                np.testing.assert_allclose(new_direction, old_direction, atol=1e-7)
                self.assertEqual(bool(camera.GetParallelProjection()), parallel)
                np.testing.assert_allclose(camera.GetFocalPoint(), (2.0, 2.0, 5.0), atol=1e-7)
                self.assertAlmostEqual(float(np.linalg.norm(camera.GetViewUp())), 1.0, places=6)
                self.assertGreater(float(np.dot(camera.GetViewUp(), old_view_up)), 0.0)
                if parallel:
                    self.assertLess(camera.GetParallelScale(), 500.0)

    def test_axis_reset_preserves_projection_mode_and_fits_canonical_direction(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                scene, camera = self._camera_scene()
                camera.SetParallelProjection(int(parallel))
                scene.reset_camera_axis("x")
                self.assertEqual(scene.perspective_enabled, not parallel)
                self.assertEqual(bool(camera.GetParallelProjection()), parallel)
                np.testing.assert_allclose(camera.GetFocalPoint(), (2.0, 2.0, 5.0), atol=1e-7)
                direction = np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
                direction /= np.linalg.norm(direction)
                np.testing.assert_allclose(direction, (1.0, 0.0, 0.0), atol=1e-7)
                np.testing.assert_allclose(camera.GetViewUp(), (0.0, 0.0, 1.0), atol=1e-7)

    def test_perspective_toggle_updates_public_toolbar_state(self):
        scene, camera = self._camera_scene()
        scene.set_perspective(False)
        self.assertFalse(scene.perspective_enabled)
        self.assertTrue(bool(camera.GetParallelProjection()))
        scene.set_perspective(True)
        self.assertTrue(scene.perspective_enabled)
        self.assertFalse(bool(camera.GetParallelProjection()))

    def test_axes_visibility_includes_orientation_marker_and_frame(self):
        scene = object.__new__(VTKVolumeScene)
        scene.payload = object()
        scene._axes_visible = True
        scene._grid_visible = True
        scene.axes_widget = _VisibilityProbe()
        scene.outline_actor = _VisibilityProbe()
        scene.axis_tick_actor = _VisibilityProbe()
        scene.grid_actor = _VisibilityProbe()
        scene._axis_label_actors = [_VisibilityProbe(), _VisibilityProbe()]
        scene._request_render = lambda: None

        scene.set_axes_visible(False)
        self.assertFalse(scene.axes_visible)
        self.assertEqual(scene.axes_widget.enabled, 0)
        self.assertEqual(scene.outline_actor.visible, 0)
        self.assertEqual(scene.axis_tick_actor.visible, 0)
        self.assertEqual(scene.grid_actor.visible, 0)
        self.assertTrue(all(actor.visible == 0 for actor in scene._axis_label_actors))

        scene.set_axes_visible(True)
        self.assertTrue(scene.axes_visible)
        self.assertEqual(scene.axes_widget.enabled, 1)
        self.assertEqual(scene.outline_actor.visible, 1)
        self.assertEqual(scene.axis_tick_actor.visible, 1)
        self.assertEqual(scene.grid_actor.visible, 1)
        self.assertTrue(all(actor.visible == 1 for actor in scene._axis_label_actors))

    def test_rgb_axes_icon_variants_match_vtk_axes_actor_rgb(self):
        axes = vtk.vtkAxesActor()
        expected = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        actual = tuple(
            getattr(axes, f"Get{axis}AxisShaftProperty")().GetColor()
            for axis in ("X", "Y", "Z")
        )
        self.assertEqual(actual, expected)

        icon_dir = REPO_ROOT / "roc_mcs/gui/resources/icons"
        for index in (1, 2, 3):
            path = icon_dir / f"vtk_axes_info_rgb_{index}.svg"
            self.assertTrue(path.is_file(), path)
            svg = path.read_text(encoding="utf-8").upper()
            self.assertIn("#FF0000", svg)
            self.assertIn("#00FF00", svg)
            self.assertIn("#0000FF", svg)
            self.assertIn(">I<", svg)

    def test_toolbar_actions_follow_requested_order_and_are_signal_driven(self):
        toolbar_path = REPO_ROOT / "roc_mcs/gui/roi_explorer/vtk_backend/vtk_tools_widget.py"
        viewer_path = REPO_ROOT / "roc_mcs/gui/roi_explorer/vtk_backend/desktop_viewer.py"
        toolbar = toolbar_path.read_text(encoding="utf-8")
        viewer = viewer_path.read_text(encoding="utf-8")

        ordered_actions = (
            '"Toggle axis (on/off)"',
            '"Perspective view"',
            '"Reset camera X"',
            '"Reset camera Y"',
            '"Reset camera Z"',
        )
        offsets = [toolbar.index(label) for label in ordered_actions]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn('reset_view_requested = pyqtSignal()', toolbar)
        self.assertIn('axes_toggled = pyqtSignal(bool)', toolbar)
        self.assertIn('perspective_toggled = pyqtSignal(bool)', toolbar)
        self.assertIn('cursor_line_color_changed = pyqtSignal(str)', toolbar)
        self.assertIn('_AXES_ICON = "vtk_axes_info_rgb_1.svg"', toolbar)
        self.assertIn('self.reset_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg"', toolbar)
        self.assertIn('self.viewer_layout.addWidget(self.tools_widget, 0)', viewer)
        self.assertNotIn('self.camera_group.setEnabled', viewer)

    def test_color_picker_is_independent_from_line_width_and_density_controls(self):
        toolbar_path = REPO_ROOT / "roc_mcs/gui/roi_explorer/vtk_backend/vtk_tools_widget.py"
        scene_path = REPO_ROOT / "roc_mcs/gui/roi_explorer/vtk_backend/vtk_scene.py"
        desktop_path = REPO_ROOT / "roc_mcs/gui/roi_explorer/vtk_backend/desktop_viewer.py"
        toolbar = toolbar_path.read_text(encoding="utf-8")
        scene = scene_path.read_text(encoding="utf-8")
        desktop = desktop_path.read_text(encoding="utf-8")

        self.assertIn('QColorDialog.getColor(', toolbar)
        self.assertIn('self.cursor_line_color_changed.emit(color)', toolbar)
        self.assertIn('self._theta_slice_uniforms.SetUniform3f("rocCursorLineColor", rgb)', scene)
        self.assertIn('self.cursor_line_width = self._double_spin', desktop)
        self.assertIn('self.cursor_line_density = self._double_spin', desktop)
        self.assertIn('self.scene.set_cursor_line_color(str(color), render=False)', desktop)
        self.assertIn('self.vtk_widget.Render()', desktop)



if __name__ == "__main__":
    unittest.main()
