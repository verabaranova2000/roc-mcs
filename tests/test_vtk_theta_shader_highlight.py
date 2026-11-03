from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import vtk

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (
    TransferSettings,
    VTKVolumeScene,
)


class ThetaShaderHighlightTests(unittest.TestCase):
    @staticmethod
    def payload(*, rotated: bool = False):
        x = np.linspace(-3.0, 3.0, 9)
        y = np.linspace(-2.5, 2.5, 8)
        # Deliberately nonuniform: the selected-cell bounds must use the
        # actual midpoints, not a guessed constant theta step.
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
        # A detector defect represented by invalid values across all angles.
        volume[2, 3, :] = np.nan
        projection = np.nansum(volume, axis=2)
        roi = SimpleNamespace(cx=120.0, cy=85.0, angle=31.0) if rotated else None
        return SimpleNamespace(
            volume=volume,
            projection=projection,
            x=x if rotated else x + 120.0,
            y=y if rotated else y + 85.0,
            theta=theta,
            coord_system="uv" if rotated else "xy",
            roi=roi,
        )

    @staticmethod
    def make_scene():
        render_window = vtk.vtkRenderWindow()
        interactor = vtk.vtkGenericRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        return render_window, interactor, VTKVolumeScene(render_window, interactor)

    def _close_scene(self, render_window, interactor, scene):
        scene.close()
        interactor.SetRenderWindow(None)
        render_window.Finalize()

    def test_highlight_is_installed_in_existing_gpu_volume_shader(self):
        rw, interactor, scene = self.make_scene()
        try:
            payload = self.payload()
            scene.set_payload(payload, TransferSettings())
            props_after_payload = int(scene.renderer.GetViewProps().GetNumberOfItems())
            actors_after_payload = int(scene.renderer.GetActors().GetNumberOfItems())

            self.assertIsNotNone(scene.volume)
            self.assertIs(scene.volume.GetMapper(), scene.mapper)
            self.assertEqual(scene.mapper.GetClassName(), "vtkOpenGLGPUVolumeRayCastMapper")
            self.assertIs(scene.volume.GetShaderProperty(), scene._theta_slice_shader_property)
            self.assertIsNotNone(scene._theta_slice_uniforms)
            self.assertTrue(scene.theta_slice_shader_installed)
            self.assertEqual(scene.diagnostics()["theta_slice_shader_replacement_count"], 1)
            self.assertEqual(scene.renderer.GetVolumes().GetNumberOfItems(), 1)
            self.assertEqual(scene.renderer.GetViewProps().GetNumberOfItems(), props_after_payload)
            self.assertEqual(scene.renderer.GetActors().GetNumberOfItems(), actors_after_payload)
            self.assertEqual(scene.diagnostics()["theta_slice_extra_actors"], 0)
            self.assertEqual(scene.diagnostics()["theta_slice_highlight_mode"], "in_volume_ray_sample_rgb")
            self.assertFalse(hasattr(scene, "theta_slice_actor"))
            self.assertFalse(hasattr(scene, "theta_slice_polydata"))
            self.assertFalse(hasattr(scene, "theta_slice_mapper"))
        finally:
            self._close_scene(rw, interactor, scene)

    def test_nonuniform_theta_cell_bounds_are_midpoint_based(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            self.assertEqual(scene._theta_slice_cell_bounds(0), (-5.0, -3.0))
            self.assertEqual(scene._theta_slice_cell_bounds(1), (-3.0, -1.0))
            self.assertEqual(scene._theta_slice_cell_bounds(2), (-1.0, 1.5))
            self.assertEqual(scene._theta_slice_cell_bounds(3), (1.5, 5.0))
            self.assertEqual(scene._theta_slice_cell_bounds(4), (5.0, 9.0))
        finally:
            self._close_scene(rw, interactor, scene)

    def test_theta_slider_palette_and_strength_update_only_uniforms(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings(palette="viridis"))
            mapper = scene.mapper
            grid = scene.grid
            volume = scene.volume
            volume_property = scene.volume_property
            color_function = scene.color_function
            opacity_function = scene.opacity_function
            shader_property = scene._theta_slice_shader_property
            uniforms = scene._theta_slice_uniforms
            shader_build_mtime = int(shader_property.GetShaderMTime())
            uniform_list_mtime = int(uniforms.GetUniformListMTime())
            grid_mtime = int(grid.GetMTime())
            mapper_mtime = int(mapper.GetMTime())
            color_mtime = int(color_function.GetMTime())
            opacity_mtime = int(opacity_function.GetMTime())

            scene.set_theta_slice_enabled(True, render=False)
            scene.set_theta_slice_index(3, render=False)
            scene.set_theta_slice_palette("inferno", render=False)
            scene.set_theta_slice_strength(0.65, render=False)

            self.assertIs(scene.mapper, mapper)
            self.assertIs(scene.grid, grid)
            self.assertIs(scene.volume, volume)
            self.assertIs(scene.volume_property, volume_property)
            self.assertIs(scene.color_function, color_function)
            self.assertIs(scene.opacity_function, opacity_function)
            self.assertIs(scene._theta_slice_shader_property, shader_property)
            self.assertIs(scene._theta_slice_uniforms, uniforms)
            self.assertEqual(int(shader_property.GetShaderMTime()), shader_build_mtime)
            self.assertEqual(int(uniforms.GetUniformListMTime()), uniform_list_mtime)
            self.assertEqual(int(grid.GetMTime()), grid_mtime)
            self.assertEqual(int(mapper.GetMTime()), mapper_mtime)
            self.assertEqual(int(color_function.GetMTime()), color_mtime)
            self.assertEqual(int(opacity_function.GetMTime()), opacity_mtime)
            self.assertTrue(scene.theta_slice_enabled)
            self.assertEqual(scene.theta_slice_index, 3)
            self.assertEqual(scene.theta_slice_palette, "inferno")
            self.assertAlmostEqual(scene.theta_slice_strength, 0.65)
            # Updating custom uniform values is expected to update the uniform
            # container, but must not cause VTK to rebuild the shader program.
            self.assertGreater(int(uniforms.GetMTime()), 0)
        finally:
            self._close_scene(rw, interactor, scene)

    def test_shader_tints_rgb_only_after_native_opacity_is_computed(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(
                self.payload(),
                TransferSettings(lower_fraction=0.2, upper_fraction=0.8, max_opacity=0.23, gamma=2.0),
            )
            shader = scene._theta_slice_shader_color_code()
            self.assertIn("g_srcColor.a > 0.0", shader)
            self.assertIn("in_scalarsRange[0][0]", shader)
            self.assertIn("in_scalarsRange[0][1]", shader)
            self.assertIn("g_srcColor.rgb = mix", shader)
            self.assertIn("vec3 rocThetaHighlightColor = rocThetaPalette15;", shader)
            self.assertTrue(shader.rstrip().endswith("g_srcColor.rgb *= g_srcColor.a;"))
            self.assertNotIn("g_srcColor.a =", shader)
            self.assertNotIn("g_srcColor.a *=", shader)
            self.assertNotIn("gl_FragDepth", shader)
            # The original transfer functions continue to own opacity.
            low = scene.diagnostics()["thresholds"]["threshold_min"]
            high = scene.diagnostics()["thresholds"]["threshold_max"]
            self.assertEqual(scene._theta_slice_transfer_range, (low, high))
            self.assertGreater(scene.opacity_function.GetValue(high), 0.0)
            self.assertEqual(scene.opacity_function.GetValue(scene.grid_info["invalid_sentinel"]), 0.0)
        finally:
            self._close_scene(rw, interactor, scene)

    def test_nan_and_masked_samples_remain_transparent_under_highlight(self):
        rw, interactor, scene = self.make_scene()
        try:
            payload = self.payload()
            scene.set_payload(payload, TransferSettings())
            scene.set_theta_slice_enabled(True, render=False)
            sentinel = float(scene.grid_info["invalid_sentinel"])
            self.assertEqual(scene.opacity_function.GetValue(sentinel), 0.0)
            shader = scene._theta_slice_shader_color_code()
            # No tint is entered for a sample removed by the original opacity
            # transfer; the source-alpha guard is deliberately the first gate.
            self.assertLess(shader.index("g_srcColor.a > 0.0"), shader.index("rocThetaDataPosH"))
            self.assertLess(shader.index("g_srcColor.a > 0.0"), shader.index("g_srcColor.rgb = mix"))
        finally:
            self._close_scene(rw, interactor, scene)

    def test_rotated_roi_and_theta_scale_keep_the_same_data_volume(self):
        rw, interactor, scene = self.make_scene()
        try:
            payload = self.payload(rotated=True)
            scene.set_payload(payload, TransferSettings())
            volume = scene.volume
            grid = scene.grid
            mapper = scene.mapper
            shader_mtime = int(scene._theta_slice_shader_property.GetShaderMTime())
            scene.set_theta_scale(2.5, render=False)
            scene.set_theta_slice_index(2, render=False)
            center = 0.5 * (payload.theta[0] + payload.theta[-1])
            expected = center + (payload.theta[2] - center) * 2.5
            self.assertAlmostEqual(float(scene._display_theta[2]), expected)
            low, high = scene._theta_slice_cell_bounds(2)
            self.assertAlmostEqual(low, center + (-1.0 - center) * 2.5)
            self.assertAlmostEqual(high, center + (1.5 - center) * 2.5)
            self.assertIs(scene.volume, volume)
            self.assertIs(scene.grid, grid)
            self.assertIs(scene.mapper, mapper)
            self.assertEqual(int(scene._theta_slice_shader_property.GetShaderMTime()), shader_mtime)
            matrix = scene.volume.GetMatrix()
            self.assertAlmostEqual(matrix.GetElement(0, 3), 120.0)
            self.assertAlmostEqual(matrix.GetElement(1, 3), 85.0)
        finally:
            self._close_scene(rw, interactor, scene)

    def test_xy_projection_is_not_hidden_when_highlight_changes(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings(show_projection=True))
            self.assertTrue(scene.projection_actor.GetVisibility())
            scene.set_theta_slice_enabled(True, render=False)
            self.assertTrue(scene.projection_actor.GetVisibility())
            scene.set_theta_slice_index(0, render=False)
            self.assertTrue(scene.projection_actor.GetVisibility())
            scene.set_theta_slice_index(4, render=False)
            self.assertTrue(scene.projection_actor.GetVisibility())
            scene.set_theta_slice_enabled(False, render=False)
            self.assertTrue(scene.projection_actor.GetVisibility())
        finally:
            self._close_scene(rw, interactor, scene)

    def test_frame_navigation_never_adds_geometry_or_extra_actors(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            initial_props = int(scene.renderer.GetViewProps().GetNumberOfItems())
            initial_actors = int(scene.renderer.GetActors().GetNumberOfItems())
            initial_volumes = int(scene.renderer.GetVolumes().GetNumberOfItems())
            for index in range(5):
                scene.set_theta_slice_index(index, render=False)
                scene.set_theta_slice_palette("turbo" if index % 2 == 0 else "magma", render=False)
                scene.set_theta_slice_enabled(index % 2 == 0, render=False)
                self.assertEqual(int(scene.renderer.GetViewProps().GetNumberOfItems()), initial_props)
                self.assertEqual(int(scene.renderer.GetActors().GetNumberOfItems()), initial_actors)
                self.assertEqual(int(scene.renderer.GetVolumes().GetNumberOfItems()), initial_volumes)
                diagnostics = scene.diagnostics()
                self.assertEqual(diagnostics["theta_slice_extra_actors"], 0)
                self.assertEqual(diagnostics["theta_slice_highlight_mode"], "in_volume_ray_sample_rgb")
        finally:
            self._close_scene(rw, interactor, scene)

    def test_invalid_strength_is_rejected(self):
        rw, interactor, scene = self.make_scene()
        try:
            scene.set_payload(self.payload(), TransferSettings())
            for bad_value in (-0.01, 1.01, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    scene.set_theta_slice_strength(bad_value, render=False)
        finally:
            self._close_scene(rw, interactor, scene)


if __name__ == "__main__":
    unittest.main()
