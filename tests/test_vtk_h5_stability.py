from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (
    TransferSettings,
    VTKVolumeScene,
    build_projection_grid,
    compute_opacity_unit_distance,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "src/roc_mcs/gui/roi_explorer/vtk_backend/vtk_scene.py"
VIEWER_PATH = ROOT / "src/roc_mcs/gui/roi_explorer/vtk_backend/desktop_viewer.py"


class TestTransferSemantics(unittest.TestCase):
    def test_default_thresholds_match_vtk_lab_fraction_semantics(self):
        settings = TransferSettings()
        self.assertAlmostEqual(settings.lower_fraction, 0.16)
        self.assertAlmostEqual(settings.upper_fraction, 0.95)
        self.assertAlmostEqual(settings.max_opacity, 0.18)
        self.assertAlmostEqual(settings.gamma, 2.0)
        self.assertAlmostEqual(settings.opacity_unit_distance_factor, 1.0)
        self.assertIsNone(settings.lower_percentile)
        self.assertIsNone(settings.upper_percentile)

    def test_fraction_thresholds_are_relative_to_positive_p999_reference(self):
        scene = object.__new__(VTKVolumeScene)
        scene._finite_sample = np.array([0, 1, 2, 3, 4, 5, 10, 20, 100], dtype=float)
        settings = TransferSettings(lower_fraction=0.16, upper_fraction=0.95)
        low, high, reference = scene._resolve_thresholds(settings)
        positive = scene._finite_sample[scene._finite_sample > 0]
        expected_ref = max(float(np.percentile(positive, 99.9)), float(np.max(positive)) * 1e-9, 1e-12)
        self.assertAlmostEqual(reference, expected_ref)
        self.assertAlmostEqual(low, expected_ref * 0.16)
        self.assertAlmostEqual(high, expected_ref * 0.95)

    def test_legacy_diagnostic_percentile_arguments_remain_supported(self):
        scene = object.__new__(VTKVolumeScene)
        scene._finite_sample = np.arange(1, 101, dtype=float)
        settings = TransferSettings(lower_percentile=2.0, upper_percentile=99.9)
        VTKVolumeScene._validate_transfer_settings(settings)
        low, high, _ = scene._resolve_thresholds(settings)
        self.assertAlmostEqual(low, np.percentile(scene._finite_sample, 2.0))
        self.assertAlmostEqual(high, np.percentile(scene._finite_sample, 99.9))

    def test_invalid_threshold_pair_is_rejected_before_scene_replacement(self):
        settings = TransferSettings(lower_fraction=0.96, upper_fraction=0.95)
        with self.assertRaisesRegex(ValueError, "Vmin|нижняя доля"):
            VTKVolumeScene._validate_transfer_settings(settings)

    def test_opacity_unit_distance_is_independent_from_ray_sample_distance(self):
        self.assertAlmostEqual(compute_opacity_unit_distance(1.0, 1.0), 1.0)
        self.assertAlmostEqual(compute_opacity_unit_distance(2.0, 0.5), 1.0)
        self.assertAlmostEqual(compute_opacity_unit_distance(1.5, 2.0), 3.0)
        with self.assertRaises(ValueError):
            compute_opacity_unit_distance(1.0, float("nan"))
        with self.assertRaises(ValueError):
            compute_opacity_unit_distance(0.0, 1.0)


class TestResourceAndWorkerLifecycleSource(unittest.TestCase):
    def test_no_manual_duplicate_mapper_graphics_release_remains(self):
        source = SCENE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("_release_mapper_resources_if_current", source)
        self.assertNotIn("mapper.ReleaseGraphicsResources(", source)
        self.assertIn("self.renderer.RemoveVolume(self.volume)", source)
        self.assertIn("projection_property.LightingOff()", source)
        self.assertIn("def clear(self, *, render: bool = True)", source)

    def test_worker_result_is_consumed_after_thread_finished(self):
        source = VIEWER_PATH.read_text(encoding="utf-8")
        self.assertIn("self._worker.completed.connect(", source)
        self.assertIn("QtCore.Qt.ConnectionType.DirectConnection", source)
        self.assertIn("self._thread.finished.connect(self._thread_finished)", source)
        self.assertIn('if "payload" in outcome:', source)
        self.assertNotIn("self._worker.finished.connect(self._load_finished)", source)

    def test_ui_uses_lab_fraction_defaults_and_coupled_controls(self):
        source = VIEWER_PATH.read_text(encoding="utf-8")
        self.assertIn("self._double_spin(0.0, 99.0, 16.0", source)
        self.assertIn("self._double_spin(1.0, 100.0, 95.0", source)
        self.assertIn("lower_fraction=low", source)
        self.assertIn("upper_fraction=high", source)
        self.assertIn("def _on_low_threshold_changed", source)
        self.assertIn("def _on_high_threshold_changed", source)
        self.assertIn("Opacity unit distance ×:", source)
        self.assertIn("self._double_spin(0.0, 1.0, 0.18", source)
        self.assertIn("self._double_spin(0.05, 8.0, 2.0", source)

    def test_viewer_logs_win32_resource_counts_and_projection_state(self):
        source = VIEWER_PATH.read_text(encoding="utf-8")
        self.assertIn("GetGuiResources", source)
        self.assertIn('"gdi_objects"', source)
        self.assertIn('"QVTK_PREINIT_REPAINT_DONE"', source)
        self.assertIn('_trace_resources("before-qvtk-initialize")', source)
        self.assertIn('"user_objects"', source)
        self.assertIn('"SCENE_DIAGNOSTICS"', source)

    def test_qvtk_close_event_is_guarded_and_owns_finalize(self):
        source = VIEWER_PATH.read_text(encoding="utf-8")
        self.assertIn("self._finalized = False", source)
        self.assertIn("super().closeEvent(event)", source)
        self.assertIn("self.vtk_widget.close()", source)

    def test_viewer_settings_use_scroll_area_and_respect_windows_work_area(self):
        source = VIEWER_PATH.read_text(encoding="utf-8")
        self.assertIn("QtWidgets.QScrollArea()", source)
        self.assertIn("ScrollBarAsNeeded", source)
        self.assertIn("window.showMaximized()", source)
        self.assertIn('QCheckBox("Показывать сетку")', source)
        self.assertIn("def _toggle_grid", source)

    def test_viewer_uses_memory_full_info_and_collects_worker_result(self):
        source = VIEWER_PATH.read_text(encoding="utf-8")
        self.assertIn("process.memory_full_info()", source)
        self.assertIn('"uss_mb"', source)
        self.assertIn('"before-load-volume"', source)
        self.assertIn('"after-scene-rebuild"', source)
        self.assertIn("gc.collect()", source)


class TestScenePresentationAndReplacement(unittest.TestCase):
    @staticmethod
    def _payload():
        x = np.arange(5, dtype=np.float64)
        y = np.arange(6, dtype=np.float64)
        theta = np.asarray([-5.0, -3.0, -1.0, 1.0], dtype=np.float64)
        volume = np.arange(x.size * y.size * theta.size, dtype=np.float64).reshape(
            x.size, y.size, theta.size
        ) + 1.0
        projection = np.sum(volume, axis=2)
        return SimpleNamespace(
            volume=volume,
            x=x,
            y=y,
            theta=theta,
            projection=projection,
            coord_system="xy",
            roi=None,
        )

    def test_projection_is_exactly_coplanar_with_lower_theta_face(self):
        payload = self._payload()
        projection_grid, info = build_projection_grid(payload)
        self.assertEqual(tuple(projection_grid.GetBounds()[4:6]), (payload.theta[0], payload.theta[0]))
        self.assertEqual(info["z_plane"], float(payload.theta[0]))

    def test_projection_stays_on_lower_face_after_theta_scaling(self):
        import vtk

        render_window = vtk.vtkRenderWindow()
        interactor = vtk.vtkGenericRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        scene = VTKVolumeScene(render_window, interactor)
        try:
            scene.set_payload(self._payload(), TransferSettings())
            before = scene.diagnostics()
            self.assertEqual(before["projection_bounds"][4], before["volume_bounds"][4])
            scene.set_theta_scale(2.5, render=False)
            after = scene.diagnostics()
            self.assertAlmostEqual(after["projection_bounds"][4], after["volume_bounds"][4])
        finally:
            scene.close()

    def test_explicit_trackball_camera_replaces_joystick_default(self):
        import vtk

        render_window = vtk.vtkRenderWindow()
        interactor = vtk.vtkGenericRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        scene = VTKVolumeScene(render_window, interactor)
        try:
            style = interactor.GetInteractorStyle()
            self.assertEqual(style.GetClassName(), "vtkInteractorStyleTrackballCamera")
            self.assertAlmostEqual(style.GetMotionFactor(), 10.0)
            self.assertAlmostEqual(style.GetMouseWheelMotionFactor(), 1.0)
        finally:
            scene.close()

    def test_grid_toggle_and_repeated_rebuilds_keep_scene_prop_counts_stable(self):
        import vtk

        render_window = vtk.vtkRenderWindow()
        interactor = vtk.vtkGenericRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        scene = VTKVolumeScene(render_window, interactor)
        try:
            counts = []
            for index in range(5):
                scene.set_payload(self._payload(), TransferSettings())
                scene.set_grid_visible(index % 2 == 0, render=False)
                self.assertEqual(scene.grid_visible, index % 2 == 0)
                self.assertEqual(scene.cube_axes.GetDrawXGridlines(), int(index % 2 == 0))
                self.assertEqual(scene.cube_axes.GetDrawYGridlines(), int(index % 2 == 0))
                self.assertEqual(scene.cube_axes.GetDrawZGridlines(), int(index % 2 == 0))
                diag = scene.diagnostics()
                counts.append((diag["renderer_prop_count"], diag["renderer_actor_count"], diag["renderer_volume_count"]))
                self.assertFalse(diag["outline_actor_present"])
            self.assertTrue(all(item == counts[0] for item in counts[1:]))
            self.assertEqual(counts[0][2], 1)
            scene.clear(render=False)
            self.assertEqual(scene.renderer.GetVolumes().GetNumberOfItems(), 0)
        finally:
            scene.close()

    def test_axis_actor_has_no_parallel_outline_or_minor_ticks(self):
        import vtk

        render_window = vtk.vtkRenderWindow()
        interactor = vtk.vtkGenericRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        scene = VTKVolumeScene(render_window, interactor)
        try:
            self.assertEqual(scene.cube_axes.GetCornerOffset(), 0.0)
            self.assertEqual(scene.cube_axes.GetXAxisMinorTickVisibility(), 0)
            self.assertEqual(scene.cube_axes.GetYAxisMinorTickVisibility(), 0)
            self.assertEqual(scene.cube_axes.GetZAxisMinorTickVisibility(), 0)
            self.assertIsNone(scene.outline_actor)
        finally:
            scene.close()


if __name__ == "__main__":
    unittest.main()
