from __future__ import annotations

import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
GUI = ROOT / "roc_mcs" / "gui"
ICONS = GUI / "resources" / "icons"
TOOLS = GUI / "roi_explorer" / "vtk_backend" / "vtk_tools_widget.py"
DESKTOP = GUI / "roi_explorer" / "vtk_backend" / "desktop_viewer.py"
SCENE = GUI / "roi_explorer" / "vtk_backend" / "vtk_scene.py"


class VTKToolsSourceContractTests(unittest.TestCase):
    def test_toolbar_uses_compact_color_picker_and_signal(self):
        source = TOOLS.read_text(encoding="utf-8")
        self.assertIn("CompactColorPicker", source)
        self.assertNotIn("QColorDialog", source)
        self.assertIn("color_selected.connect", source)

    def test_camera_and_scene_controls_are_independent(self):
        toolbar = TOOLS.read_text(encoding="utf-8")
        scene = SCENE.read_text(encoding="utf-8")
        for signal in ("axes_toggled", "bounding_box_toggled", "axis_ticks_toggled", "grid_toggled", "perspective_toggled"):
            self.assertIn(signal, toolbar)
        for setter in ("set_axes_visible", "set_bounding_box_visible", "set_axis_ticks_visible", "set_grid_visible", "set_perspective"):
            self.assertIn(f"def {setter}", scene)
        self.assertIn("self.axes_button", toolbar)
        self.assertIn("self.bounding_box_button", toolbar)
        self.assertIn("self.axis_ticks_button", toolbar)

    def test_toolbar_has_directional_camera_buttons_and_reset(self):
        source = TOOLS.read_text(encoding="utf-8")
        for token in ("_DirectionalAxisButton", "camera_axis_buttons", "camera_axis_direction_requested", "reset_view_requested"):
            self.assertIn(token, source)
        for icon in ("vtk_camera_axis_x.svg", "vtk_camera_axis_y.svg", "vtk_camera_axis_z.svg", "zoom_out_map.svg"):
            self.assertTrue((ICONS / icon).is_file(), icon)

    def test_volume_source_selector_uses_existing_roi_geometry_in_dialog(self):
        toolbar = TOOLS.read_text(encoding="utf-8")
        desktop = DESKTOP.read_text(encoding="utf-8")
        for token in ("volume_source_mode_changed", "volume_source_geometry_requested",
                      "set_volume_source_mode", "set_volume_source_available",
                      "Полный детектор", "Прямоугольная ROI", "Геометрия ROI…"):
            self.assertIn(token, toolbar)
        self.assertIn("self.mode_combo.hide()", desktop)
        self.assertIn("roi_dialog_layout.addWidget(roi_group)", desktop)
        self.assertNotIn("self.panel_layout.addWidget(roi_group)", desktop)
        self.assertIn("self.roi_cx", desktop)
        self.assertIn("self.roi_width", desktop)
        self.assertIn("self.roi_geometry_dialog.exec()", desktop)
        self.assertIn("self.tools_widget.volume_source_mode_changed.connect", desktop)
        self.assertIn("self._refresh_controls()", desktop[desktop.index("def open_h5"):desktop.index("def _run_pending_auto_load")])
        self.assertIn("apply_popup_theme", desktop[:desktop.index("class TracedQVTKRenderWindowInteractor")])

    def test_volume_source_icon_variants_exist_and_parse(self):
        source = TOOLS.read_text(encoding="utf-8")
        self.assertIn("VOLUME_SOURCE_ICON_VARIANT", source)
        for index in range(1, 5):
            name = f"vtk_volume_source_{index}.svg"
            path = ICONS / name
            self.assertTrue(path.is_file(), name)
            self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"), name)

    def test_smoothing_is_documented_as_data_rebuild_not_render_quality(self):
        desktop = DESKTOP.read_text(encoding="utf-8")
        source = (GUI / "roi_explorer" / "vtk_backend" / "h5_volume_source.py").read_text(encoding="utf-8")
        self.assertIn("Гауссово сглаживание изменяет значения", desktop)
        self.assertIn("Требует повторной загрузки/перестроения объёма", desktop)
        self.assertIn("_nan_aware_smooth_preserving_invalid(volume, smoothing_sigma)", source)
        self.assertIn("min(0.2, sigma)", source)

    def test_viewer_connects_toolbar_signals_to_scene(self):
        source = DESKTOP.read_text(encoding="utf-8")
        for signal in ("axes_toggled", "bounding_box_toggled", "axis_ticks_toggled", "grid_toggled", "cursor_line_color_changed"):
            self.assertIn(signal, source)
        self.assertIn("self.scene.set_cursor_line_color", source)

    def test_all_referenced_vtk_svg_assets_exist_and_parse(self):
        source = TOOLS.read_text(encoding="utf-8")
        names = set(re.findall(r'"(vtk_[a-zA-Z0-9_]+\.svg)"', source))
        names.update(("vtk_axes_info.svg", "vtk_bounding_box.svg", "vtk_camera_perspective.svg"))
        self.assertTrue(names)
        for name in sorted(names):
            path = ICONS / name
            self.assertTrue(path.is_file(), name)
            root = ET.parse(path).getroot()
            self.assertTrue(root.tag.endswith("svg"), name)

    def test_scene_frame_ticks_grid_are_distinct_geometry(self):
        source = SCENE.read_text(encoding="utf-8")
        for token in ("outline_actor", "axis_tick_actor", "grid_actor", "outline_filter", "axis_tick_polydata"):
            self.assertIn(token, source)
        self.assertIn("def clear(self, *, render: bool = True)", source)
        self.assertIn("def close(self)", source)

    def test_memory_preflight_is_metadata_only_and_warns_before_large_load(self):
        source = DESKTOP.read_text(encoding="utf-8")
        start = source.index("def _estimate_load_memory")
        end = source.index("def load_volume", start)
        estimate = source[start:end]
        self.assertIn("dataset.shape", estimate)
        self.assertIn("dataset.dtype.itemsize", estimate)
        self.assertNotIn("dataset[...", estimate)
        self.assertIn("available", estimate)
        load_start = source.index("def load_volume", end)
        load_end = source.find("\n        def ", load_start + 10)
        load = source[load_start:load_end if load_end > 0 else len(source)]
        self.assertIn("estimated > available * 0.45", load)
        self.assertIn("QMessageBox.StandardButton.Cancel", load)


if __name__ == "__main__":
    unittest.main()
