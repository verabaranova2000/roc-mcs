from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from roc_mcs.gui.roi_explorer.vtk_backend.cursor_link import (
    detector_cursor_bounds,
    map_detector_cursor_to_volume,
)
from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (
    TransferSettings,
    VTKVolumeScene,
    roi_local_to_detector_matrix,
)


class _RotatedROI:
    def __init__(self, cx=100.0, cy=80.0, width=12.0, height=8.0, angle=30.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = float(width)
        self.height = float(height)
        self.angle = float(angle)

    def to_local(self, x, y):
        angle = np.deg2rad(self.angle)
        dx, dy = float(x) - self.cx, float(y) - self.cy
        return (
            np.cos(angle) * dx + np.sin(angle) * dy,
            -np.sin(angle) * dx + np.cos(angle) * dy,
        )

    def contains(self, x, y):
        u, v = self.to_local(x, y)
        return abs(u) <= self.width / 2 and abs(v) <= self.height / 2


class CursorLinkTests(unittest.TestCase):
    @staticmethod
    def xy_payload():
        return SimpleNamespace(
            x=np.arange(10, dtype=np.float64),
            y=np.arange(20, dtype=np.float64),
            coord_system="xy",
            roi=None,
        )

    @staticmethod
    def uv_payload():
        return SimpleNamespace(
            x=np.linspace(-6.0, 6.0, 13),
            y=np.linspace(-4.0, 4.0, 9),
            coord_system="uv",
            roi=_RotatedROI(),
        )

    def test_axis_aligned_detector_cursor_maps_without_rotation(self):
        payload = self.xy_payload()
        target = map_detector_cursor_to_volume(payload, 4.25, 6.5)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.detector_x, 4.25)
        self.assertAlmostEqual(target.volume_x, 4.25)
        self.assertAlmostEqual(target.volume_y, 6.5)
        self.assertAlmostEqual(target.radius_x, 0.8)
        self.assertAlmostEqual(target.radius_y, 0.8)

    def test_rotated_roi_uses_shared_detector_to_local_mapping(self):
        payload = self.uv_payload()
        roi = payload.roi
        angle = np.deg2rad(roi.angle)
        u, v = 2.0, -1.0
        x = roi.cx + np.cos(angle) * u - np.sin(angle) * v
        y = roi.cy + np.sin(angle) * u + np.cos(angle) * v
        target = map_detector_cursor_to_volume(payload, x, y)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.volume_x, u, places=10)
        self.assertAlmostEqual(target.volume_y, v, places=10)
        self.assertAlmostEqual(target.radius_x, 0.8)
        self.assertAlmostEqual(target.radius_y, 0.8)
        # The exact same local (u,v) point maps back to the detector cursor,
        # and theta is invariant: the resulting cursor path is parallel to θ.
        matrix = roi_local_to_detector_matrix(roi)
        world = matrix.MultiplyPoint((target.volume_x, target.volume_y, 7.25, 1.0))
        world_next = matrix.MultiplyPoint((target.volume_x, target.volume_y, 8.25, 1.0))
        self.assertAlmostEqual(world[0] / world[3], x, places=10)
        self.assertAlmostEqual(world[1] / world[3], y, places=10)
        self.assertAlmostEqual(world_next[0] - world[0], 0.0, places=10)
        self.assertAlmostEqual(world_next[1] - world[1], 0.0, places=10)
        self.assertAlmostEqual(world_next[2] - world[2], 1.0, places=10)

    def test_position_outside_rotated_roi_is_hidden(self):
        payload = self.uv_payload()
        roi = payload.roi
        angle = np.deg2rad(roi.angle)
        # Local u is well outside the rectangle but the detector point can
        # still be inside its axis-aligned bounding rectangle.
        u, v = 7.0, 0.0
        x = roi.cx + np.cos(angle) * u - np.sin(angle) * v
        y = roi.cy + np.sin(angle) * u + np.cos(angle) * v
        self.assertIsNone(map_detector_cursor_to_volume(payload, x, y))

    def test_position_outside_data_support_is_hidden_not_clamped(self):
        payload = self.xy_payload()
        self.assertIsNone(map_detector_cursor_to_volume(payload, 12.0, 10.0))
        self.assertIsNone(map_detector_cursor_to_volume(payload, 4.0, -3.0))

    def test_nonuniform_axes_use_local_voxel_spacing(self):
        payload = SimpleNamespace(
            x=np.asarray([0.0, 1.0, 3.0, 7.0]),
            y=np.asarray([0.0, 2.0, 5.0, 9.0]),
            coord_system="xy",
            roi=None,
        )
        target = map_detector_cursor_to_volume(payload, 2.5, 3.5, line_width_voxels=2.0)
        self.assertIsNotNone(target)
        # At x=2.5 the adjacent steps are 2 and 4; conservative radius uses 2.
        self.assertAlmostEqual(target.radius_x, 2.0)
        # At y=3.5 the adjacent steps are 3 and 4; conservative radius uses 3.
        self.assertAlmostEqual(target.radius_y, 3.0)

    def test_rotated_roi_slider_bounds_are_four_corner_detector_bounds(self):
        payload = self.uv_payload()
        x0, x1, y0, y1 = detector_cursor_bounds(payload)
        angle = np.deg2rad(30.0)
        expected_x = 0.5 * (12.0 * abs(np.cos(angle)) + 8.0 * abs(np.sin(angle)))
        expected_y = 0.5 * (12.0 * abs(np.sin(angle)) + 8.0 * abs(np.cos(angle)))
        self.assertAlmostEqual(x0, 100.0 - expected_x)
        self.assertAlmostEqual(x1, 100.0 + expected_x)
        self.assertAlmostEqual(y0, 80.0 - expected_y)
        self.assertAlmostEqual(y1, 80.0 + expected_y)

    def test_invalid_line_width_and_nonfinite_coordinates_are_rejected(self):
        payload = self.xy_payload()
        for width in (0.0, 0.49, 4.01, np.nan, np.inf):
            with self.assertRaises(ValueError):
                map_detector_cursor_to_volume(payload, 4.0, 5.0, line_width_voxels=width)
        with self.assertRaises(ValueError):
            map_detector_cursor_to_volume(payload, np.nan, 5.0)


class CursorShaderTests(unittest.TestCase):
    @staticmethod
    def payload(*, rotated=False):
        x = np.linspace(-3.0, 3.0, 9)
        y = np.linspace(-2.5, 2.5, 8)
        theta = np.asarray([-4.0, -2.0, 0.0, 3.0, 7.0], dtype=np.float64)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        volume = np.stack(
            [
                4.0
                + 110.0 * np.exp(-((xx - 0.4 * k) ** 2 + (yy + 0.25 * k) ** 2) / 5.0)
                + 3.0 * k
                for k in range(theta.size)
            ],
            axis=2,
        ).astype(np.float64)
        volume[2, 3, :] = np.nan
        projection = np.nansum(volume, axis=2)
        roi = _RotatedROI(cx=120.0, cy=85.0, width=12.0, height=8.0, angle=31.0) if rotated else None
        return SimpleNamespace(
            volume=volume,
            projection=projection,
            x=x,
            y=y,
            theta=theta,
            coord_system="uv" if rotated else "xy",
            roi=roi,
            scan_key="cursor-test",
            view_mode="roi",
            scalar_dtype="float64",
            finite_count=int(np.isfinite(volume).sum()),
            invalid_count=int((~np.isfinite(volume)).sum()),
            nbytes=int(volume.nbytes),
            masked_detector_pixels=(),
            smoothing_sigma=0.0,
            spike_report=SimpleNamespace(enabled=False, kappa=0.0, invalid_count=0, invalid_fraction=0.0, sigma=0.0),
        )

    @staticmethod
    def make_scene():
        render_window = vtk.vtkRenderWindow()
        interactor = vtk.vtkGenericRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        return render_window, interactor, VTKVolumeScene(render_window, interactor)

    @staticmethod
    def close_scene(render_window, interactor, scene):
        scene.close()
        interactor.SetRenderWindow(None)
        render_window.Finalize()

    def test_cursor_shader_snippets_are_independent_of_volume_sample_alpha(self):
        position_code = VTKVolumeScene._theta_overlay_position_shader_code()
        theta_code = VTKVolumeScene._theta_slice_shader_color_code()
        cursor_code = VTKVolumeScene._cursor_line_shader_composite_code()
        self.assertIn("in_textureDatasetMatrix[0] * vec4(g_dataPos, 1.0)", position_code)
        self.assertIn("rocOverlayDataPos = rocOverlayDataPosH.xyz / rocOverlayW", position_code)
        self.assertIn("g_srcColor.a > 0.0", theta_code)
        # Cursor alpha is independent of volume scalar alpha. It contributes
        # its own premultiplied sample to VTK's front-to-back accumulator.
        self.assertIn("rocCursorLineEnabled != 0", cursor_code)
        self.assertNotIn("g_srcColor.a", cursor_code)
        self.assertIn("textureSize(in_volume[0], 0)", cursor_code)
        self.assertIn("length(g_dirStep * rocCursorTextureDimensions)", cursor_code)
        self.assertIn("rocCursorLineDensity * rocCursorStepVoxels", cursor_code)
        self.assertIn("1.0 - exp(-clamp(rocCursorOpticalDepth", cursor_code)
        self.assertIn("g_fragColor = (1.0 - g_fragColor.a) * rocCursorPremultiplied + g_fragColor", cursor_code)
        self.assertNotIn("gl_FragDepth", cursor_code)

    def test_cursor_shader_injection_uses_stable_post_replacement_anchors(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            prop = scene._theta_slice_shader_property
            self.assertEqual(prop.GetNumberOfShaderReplacements(), 3)
            anchors = []
            for index in range(prop.GetNumberOfShaderReplacements()):
                original = vtk.mutable("")
                replace_first = vtk.mutable(False)
                replacement = vtk.mutable("")
                replace_all = vtk.mutable(False)
                prop.GetNthShaderReplacement(index, original, replace_first, replacement, replace_all)
                self.assertFalse(bool(replace_first))
                anchors.append(original.get())
            self.assertIn("if (g_srcColor.a > 0.0)", anchors)
            self.assertIn("g_srcColor.rgb *= g_srcColor.a;", anchors)
            self.assertIn(
                "g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor;",
                anchors,
            )
        finally:
            self.close_scene(rw, interactor, scene)

    def test_compiled_opengl_shader_contains_full_length_cursor_after_volume_guard(self):
        # This is deliberately an actual OpenGL render regression test. A
        # registered vtkShaderProperty replacement is insufficient: VTK may
        # silently fail to match a whitespace-sensitive anchor. We inspect
        # the compiled volume fragment shader and the resulting framebuffer.
        x = np.linspace(-8.0, 8.0, 48)
        y = np.linspace(-8.0, 8.0, 48)
        theta = np.linspace(-25.0, 25.0, 64)
        xx, yy, zz = np.meshgrid(x, y, theta, indexing="ij")
        # A dense, localized cloud sits in front of the cursor along +X and
        # blocks only the middle theta section. Outside that cloud the same
        # coordinate path must remain visible through zero-alpha background.
        volume = (220.0 * np.exp(-(((xx - 2.0) / 0.75) ** 2 + (yy / 0.85) ** 2 + (zz / 8.0) ** 2) / 2.0)).astype(np.float32)
        payload = SimpleNamespace(
            volume=volume,
            projection=np.max(volume, axis=2),
            x=x, y=y, theta=theta,
            coord_system="xy", roi=None,
            scan_key="cursor-opengl-regression", view_mode="full",
            scalar_dtype="float32", finite_count=int(volume.size), invalid_count=0,
            nbytes=int(volume.nbytes), masked_detector_pixels=(), smoothing_sigma=0.0,
            spike_report=SimpleNamespace(enabled=False, kappa=0.0, invalid_count=0, invalid_fraction=0.0, sigma=0.0),
        )
        rw, interactor, scene = self.make_scene()
        try:
            rw.SetOffScreenRendering(1)
            rw.SetMultiSamples(0)
            rw.SetSize(480, 360)
            scene.set_payload(
                payload,
                TransferSettings(show_projection=False, lower_fraction=0.2, upper_fraction=0.8, max_opacity=0.7),
            )
            scene.set_axes_visible(False)
            scene.renderer.ResetCamera()
            camera = scene.renderer.GetActiveCamera()
            camera.SetFocalPoint(0.0, 0.0, 0.0)
            # Looking along +X makes theta the vertical screen direction;
            # the dense cloud at x=+2 is in front of the marker at x=0.
            camera.SetPosition(50.0, 0.0, 0.0)
            camera.SetViewUp(0.0, 0.0, 1.0)
            scene.renderer.ResetCameraClippingRange()

            def capture_rgb():
                rw.Render()
                capture = vtk.vtkWindowToImageFilter()
                capture.SetInput(rw)
                capture.SetInputBufferTypeToRGB()
                capture.ReadFrontBufferOff()
                capture.Update()
                return vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).copy()

            before = capture_rgb()
            scene.set_cursor_line_enabled(True, render=False)
            scene.set_cursor_line_position(0.0, 0.0, 1.0, 1.0, render=False)
            scene.set_cursor_line_density(0.95, render=False)
            after = capture_rgb()

            program = rw.GetShaderCache().GetLastShaderBound()
            compiled_fragment = program.GetFragmentShader().GetSource()
            self.assertGreater(len(compiled_fragment), 10000, "Expected the GPU volume ray-caster shader, not a post-process shader.")
            self.assertIn("castRay()", compiled_fragment)
            self.assertIn("rocCursorSampleAlpha", compiled_fragment)
            self.assertIn("rocCursorLineDensity", compiled_fragment)
            self.assertIn("rocThetaPalettePos", compiled_fragment)
            native_composite = compiled_fragment.index(
                "g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor;"
            )
            cursor_composite = compiled_fragment.index(
                "g_fragColor = (1.0 - g_fragColor.a) * rocCursorPremultiplied + g_fragColor;"
            )
            self.assertLess(native_composite, cursor_composite)
            self.assertIn("}\nif (rocCursorLineEnabled != 0)", compiled_fragment[native_composite:cursor_composite + 1])
            difference = np.abs(after.astype(np.int16) - before.astype(np.int16)).reshape(360, 480, 3)
            changed_mask = np.max(difference, axis=2) > 8
            row_hits = np.count_nonzero(changed_mask, axis=1)
            self.assertGreater(int(row_hits[20:90].sum()), 100, "The free cursor segment above the cloud must be visible.")
            self.assertGreater(int(row_hits[270:340].sum()), 100, "The free cursor segment below the cloud must be visible.")
            self.assertLess(
                int(row_hits[150:210].sum()),
                int(min(row_hits[20:90].sum(), row_hits[270:340].sum()) * 0.15),
                "The foreground cloud must attenuate/occlude the middle of the embedded cursor.",
            )
            self.assertEqual(scene.renderer.GetVolumes().GetNumberOfItems(), 1)
            self.assertEqual(scene.diagnostics()["cursor_line_extra_actors"], 0)
        finally:
            self.close_scene(rw, interactor, scene)

    def test_cursor_density_changes_only_uniform_value(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            mapper, grid, volume = scene.mapper, scene.grid, scene.volume
            shader_property = scene._theta_slice_shader_property
            uniforms = scene._theta_slice_uniforms
            shader_mtime = int(shader_property.GetShaderMTime())
            uniform_list_mtime = int(uniforms.GetUniformListMTime())
            mapper_mtime = int(mapper.GetMTime())
            grid_mtime = int(grid.GetMTime())
            scene.set_cursor_line_density(0.9, render=False)
            self.assertAlmostEqual(scene.diagnostics()["cursor_line_density_per_voxel"], 0.9)
            self.assertIs(scene.mapper, mapper)
            self.assertIs(scene.grid, grid)
            self.assertIs(scene.volume, volume)
            self.assertEqual(int(shader_property.GetShaderMTime()), shader_mtime)
            self.assertEqual(int(uniforms.GetUniformListMTime()), uniform_list_mtime)
            self.assertEqual(int(mapper.GetMTime()), mapper_mtime)
            self.assertEqual(int(grid.GetMTime()), grid_mtime)
            for invalid in (0.0, -1.0, 3.01, np.nan, np.inf):
                with self.assertRaises(ValueError):
                    scene.set_cursor_line_density(invalid, render=False)
        finally:
            self.close_scene(rw, interactor, scene)

    def test_cursor_uniform_updates_preserve_volume_mapper_grid_and_shader(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            mapper, grid, volume = scene.mapper, scene.grid, scene.volume
            shader_property = scene._theta_slice_shader_property
            uniforms = scene._theta_slice_uniforms
            props_count = int(scene.renderer.GetViewProps().GetNumberOfItems())
            actors_count = int(scene.renderer.GetActors().GetNumberOfItems())
            volumes_count = int(scene.renderer.GetVolumes().GetNumberOfItems())
            shader_mtime = int(shader_property.GetShaderMTime())
            uniforms_list_mtime = int(uniforms.GetUniformListMTime())
            grid_mtime = int(grid.GetMTime())
            mapper_mtime = int(mapper.GetMTime())
            color_mtime = int(scene.color_function.GetMTime())
            opacity_mtime = int(scene.opacity_function.GetMTime())

            scene.set_cursor_line_enabled(True, render=False)
            scene.set_cursor_line_position(0.25, -0.5, 0.8, 0.8, render=False)
            scene.set_cursor_line_position(1.0, 0.75, 0.8, 0.8, render=False)

            self.assertIs(scene.mapper, mapper)
            self.assertIs(scene.grid, grid)
            self.assertIs(scene.volume, volume)
            self.assertEqual(int(scene.renderer.GetViewProps().GetNumberOfItems()), props_count)
            self.assertEqual(int(scene.renderer.GetActors().GetNumberOfItems()), actors_count)
            self.assertEqual(int(scene.renderer.GetVolumes().GetNumberOfItems()), volumes_count)
            self.assertEqual(int(shader_property.GetShaderMTime()), shader_mtime)
            self.assertEqual(int(uniforms.GetUniformListMTime()), uniforms_list_mtime)
            self.assertEqual(int(grid.GetMTime()), grid_mtime)
            self.assertEqual(int(mapper.GetMTime()), mapper_mtime)
            self.assertEqual(int(scene.color_function.GetMTime()), color_mtime)
            self.assertEqual(int(scene.opacity_function.GetMTime()), opacity_mtime)
            diagnostics = scene.diagnostics()
            self.assertTrue(diagnostics["cursor_line_enabled"])
            self.assertEqual(diagnostics["cursor_line_extra_actors"], 0)
            self.assertEqual(diagnostics["cursor_line_dataset_xy"], (1.0, 0.75))
        finally:
            self.close_scene(rw, interactor, scene)

    def test_missing_map_position_hides_marker_but_preserves_requested_toggle(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            scene.set_cursor_line_enabled(True, render=False)
            scene.set_cursor_line_position(0.0, 0.0, 0.8, 0.8, render=False)
            scene.set_cursor_line_position(None, None, render=False)
            self.assertTrue(scene.cursor_line_enabled)
            self.assertIsNone(scene.cursor_line_position)
            self.assertFalse(scene.diagnostics()["cursor_line_enabled"])
        finally:
            self.close_scene(rw, interactor, scene)

    def test_reset_new_payload_and_clear_drop_cursor_state(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            scene.set_cursor_line_enabled(True, render=False)
            scene.set_cursor_line_position(0.0, 0.0, 0.8, 0.8, render=False)
            scene.set_payload(self.payload(), TransferSettings())
            self.assertFalse(scene.cursor_line_enabled)
            self.assertIsNone(scene.cursor_line_position)
            scene.clear(render=False)
            self.assertFalse(scene.cursor_line_enabled)
            self.assertIsNone(scene.cursor_line_position)
        finally:
            self.close_scene(rw, interactor, scene)

    def test_theta_selection_and_cursor_share_one_shader_replacement(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            scene.set_theta_slice_enabled(True, render=False)
            scene.set_theta_slice_index(2, render=False)
            scene.set_cursor_line_enabled(True, render=False)
            scene.set_cursor_line_position(0.0, 0.0, 0.8, 0.8, render=False)
            self.assertEqual(scene.diagnostics()["theta_slice_shader_replacement_count"], 3)
            self.assertEqual(scene.diagnostics()["cursor_line_extra_actors"], 0)
            self.assertEqual(scene.renderer.GetVolumes().GetNumberOfItems(), 1)
        finally:
            self.close_scene(rw, interactor, scene)


if __name__ == "__main__":
    unittest.main()
