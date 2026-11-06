from __future__ import annotations

import ast
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "src" / "roc_mcs" / "gui"
ICONS = GUI / "resources" / "icons"
TOOLS = GUI / "roi_explorer" / "vtk_backend" / "vtk_tools_widget.py"
DESKTOP = GUI / "roi_explorer" / "vtk_backend" / "desktop_viewer.py"
SCENE = GUI / "roi_explorer" / "vtk_backend" / "vtk_scene.py"


class VTKToolsSourceContractTests(unittest.TestCase):
    def test_toolbar_uses_compact_roi_color_picker(self):
        source = TOOLS.read_text(encoding="utf-8")
        self.assertIn("CompactColorPicker", source)
        self.assertNotIn("QColorDialog", source)
        self.assertIn("color_selected.connect", source)
        self.assertIn("apply_theme(active)", source)

    def test_camera_flyout_is_icon_only_horizontal_widget_action(self):
        source = TOOLS.read_text(encoding="utf-8")
        self.assertIn("QWidgetAction", source)
        self.assertIn("row = QHBoxLayout(flyout)", source)
        self.assertIn("self.orientation_axes_button", source)
        self.assertIn("self.bounding_box_button", source)
        self.assertIn("self.axis_ticks_button", source)
        self.assertIn("self.perspective_button", source)
        self.assertIn("self.camera_x_button", source)
        self.assertIn("self.camera_y_button", source)
        self.assertIn("self.camera_z_button", source)
        self.assertIn("separator = QFrame(flyout)", source)

    def test_object_options_are_context_opened_and_left_click_toggles(self):
        source = TOOLS.read_text(encoding="utf-8")
        self.assertIn("setCheckable(True)", source)
        self.assertIn("Qt.ContextMenuPolicy.CustomContextMenu", source)
        self.assertIn("_show_object_options", source)
        self.assertIn("self.cursor_line_group = self._build_cursor_options_page()", source)
        self.assertIn("self.theta_slice_group = self._build_theta_options_page()", source)

    def test_desktop_has_no_old_theme_combo_and_routes_camera_preset(self):
        source = DESKTOP.read_text(encoding="utf-8")
        self.assertNotIn("theme_combo", source)
        self.assertIn("def apply_theme(self, theme: str | None = None)", source)
        self.assertIn("self.scene.set_perspective_view()", source)
        self.assertIn("self.tools_widget.orientation_axes_toggled.connect", source)
        self.assertIn("self.tools_widget.bounding_box_toggled.connect", source)
        self.assertIn("self.tools_widget.axis_ticks_toggled.connect", source)
        self.assertIn("self.tools_widget.theta_slice_toggled.connect", source)

    def test_new_svg_assets_are_valid_and_pair_variants_exist(self):
        expected = [
            "vtk_camera_axes.svg", "vtk_orientation_axes.svg",
            "vtk_bounding_box.svg", "vtk_axis_ticks.svg",
            "vtk_view_perspective.svg", "vtk_view_x.svg", "vtk_view_y.svg", "vtk_view_z.svg",
            "vtk_theme_sun.svg", "vtk_theme_moon.svg", "vtk_scene_objects.svg",
            *[f"vtk_object_cursor_{i}.svg" for i in (1, 2, 3)],
            *[f"vtk_object_theta_{i}.svg" for i in (1, 2, 3)],
        ]
        for name in expected:
            path = ICONS / name
            self.assertTrue(path.is_file(), name)
            ET.parse(path)


    def test_object_shelf_uses_extensible_registry_and_hides_empty_header(self):
        source = TOOLS.read_text(encoding="utf-8")
        self.assertIn("scene_object_toggled = pyqtSignal(str, bool)", source)
        self.assertIn("def register_scene_object(", source)
        self.assertIn("self.object_row_layout.insertWidget", source)
        self.assertIn("self.options_header.setVisible(visible)", source)
        self.assertIn("self.object_titles = {", source)
        self.assertIn("self.object_icons = {", source)

    def test_bounding_box_glyph_is_a_solid_theme_aware_cube(self):
        source = (ICONS / "vtk_bounding_box.svg").read_text(encoding="utf-8")
        self.assertIn('fill="currentColor"', source)
        self.assertIn("L12 22", source)
        self.assertIn('fill-rule="evenodd"', source)

    def test_isometric_preset_uses_pyvista_default_camera_vector(self):
        source = SCENE.read_text(encoding="utf-8")
        self.assertIn("direction = np.asarray([1.0, 1.0, 1.0]", source)
        self.assertIn("camera.SetParallelProjection(0)", source)

    def test_standalone_theme_falls_back_to_application_palette(self):
        source = DESKTOP.read_text(encoding="utf-8")
        self.assertIn("resolve_theme(app)", source)
        self.assertIn("app.setProperty(\"rocMcsTheme\", target)", source)

    def test_orientation_icon_uses_vtk_axis_rgb_and_theme_aware_i(self):
        source = (ICONS / "vtk_orientation_axes.svg").read_text(encoding="utf-8")
        self.assertIn("#FF0000", source)
        self.assertIn("#00FF00", source)
        self.assertIn("#0000FF", source)
        self.assertIn('fill="currentColor"', source)
        self.assertIn(">i<", source)

    def test_scene_has_independent_axis_frame_and_tick_states(self):
        source = SCENE.read_text(encoding="utf-8")
        for method in (
            "set_orientation_axes_visible",
            "set_bounding_box_visible",
            "set_axis_ticks_visible",
            "set_perspective_view",
        ):
            self.assertIn(f"def {method}", source)


class VTKSceneVisibilityTests(unittest.TestCase):
    def test_orientation_frame_and_ticks_are_independent(self):
        try:
            import vtk
            from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import VTKVolumeScene
        except Exception as exc:
            self.skipTest(f"VTK backend unavailable: {exc}")

        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        scene = VTKVolumeScene(render_window, interactor)
        try:
            self.assertTrue(scene.axes_visible)
            self.assertTrue(scene.bounding_box_visible)
            self.assertTrue(scene.axis_ticks_visible)

            scene.set_orientation_axes_visible(False, render=False)
            self.assertFalse(scene.axes_visible)
            self.assertTrue(scene.bounding_box_visible)
            self.assertTrue(scene.axis_ticks_visible)

            scene.set_bounding_box_visible(False, render=False)
            self.assertFalse(scene.axes_visible)
            self.assertFalse(scene.bounding_box_visible)
            self.assertTrue(scene.axis_ticks_visible)

            scene.set_axis_ticks_visible(False, render=False)
            self.assertFalse(scene.axes_visible)
            self.assertFalse(scene.bounding_box_visible)
            self.assertFalse(scene.axis_ticks_visible)

            scene.set_orientation_axes_visible(True, render=False)
            self.assertTrue(scene.axes_visible)
            self.assertFalse(scene.bounding_box_visible)
            self.assertFalse(scene.axis_ticks_visible)
        finally:
            scene.close()
            interactor.SetRenderWindow(None)
            render_window.Finalize()


if __name__ == "__main__":
    unittest.main()
