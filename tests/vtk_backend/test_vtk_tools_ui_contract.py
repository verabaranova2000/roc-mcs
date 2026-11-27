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

    def test_all_3d_tool_icons_follow_the_application_icon_size(self):
        source = TOOLS.read_text(encoding="utf-8")
        start = source.index("    def _set_icon_sizes(self) -> None:")
        end = source.index("\n    def _color_swatch_icon", start)
        method = source[start:end]
        self.assertIn("size = QSize(self._icon_size(), self._icon_size())", method)
        for control in (
            "self.volume_source_btn", "self.orientation_btn", "self.camera_btn",
            "self.overlays_btn", "self.theta_range_btn", "self.theme_btn",
            "self.perspective_button", "self.camera_axis_buttons.values()",
            "self.axes_button", "self.bounding_box_button", "self.axis_ticks_button",
            "self.axis_scale_btn", "self.theta_range_swap_button",
            "self.theta_range_link_button", "self.axis_scale_reset_btn",
            "self._overlay_buttons.values()",
        ):
            self.assertIn(control, method)
        # Color swatches intentionally retain their rectangular preview geometry.
        self.assertNotIn("cursor_color_btn", method)
        self.assertNotIn("roi_preview_color_btn", method)
        # Creation-time defaults must not reintroduce fixed-size exceptions:
        # even compact nested controls follow the shared setting.
        for fixed_size in ("setIconSize(QSize(18, 18))", "setIconSize(QSize(17, 17))",
                           "setIconSize(QSize(15, 15))", "setIconSize(QSize(14, 14))"):
            self.assertNotIn(fixed_size, source)

    def test_toolbar_geometry_and_icons_follow_global_scale_together(self):
        source = TOOLS.read_text(encoding="utf-8")
        start = source.index("    def _set_icon_sizes(self) -> None:")
        end = source.index("\n    def _color_swatch_icon", start)
        method = source[start:end]
        self.assertIn("button_width = max(22, min(30, icon + 10))", method)
        self.assertIn("button_height = max(18, min(26, icon + 6))", method)
        self.assertIn("self.setFixedHeight(button_height + 4)", method)
        self.assertIn('button.setProperty("toolbarControl", True)', method)
        self.assertIn("button.setFixedSize(button_width, button_height)", method)
        self.assertIn("compact_size = QSize(max(10, icon - 2), max(10, icon - 2))", method)
        self.assertNotIn("self.setFixedHeight(34)", source)

    def test_embedded_viewer_reapplies_toolbar_policy_after_global_icon_propagation(self):
        explorer = (GUI / "roi_explorer" / "explorer_window.py").read_text(encoding="utf-8")
        embedded = (GUI / "roi_explorer" / "vtk_backend" / "embedded_widget.py").read_text(encoding="utf-8")
        self.assertIn('widget.objectName() == "EmbeddedVTKVolumeWidget"', explorer)
        self.assertIn("if callable(apply_icon_size):", explorer)
        self.assertIn("self.tools._set_icon_sizes()", embedded)

    def test_trs_host_applies_page_specific_icon_policy_after_global_baseline(self):
        host = (GUI / "trs_apps" / "app_host.py").read_text(encoding="utf-8")
        method_start = host.index("    def _apply_global_settings_to_pages")
        method_end = host.index("\n    def open_resource_monitor", method_start)
        method = host[method_start:method_end]
        self.assertLess(
            method.index("button.setIconSize"),
            method.index("method = getattr(page, \"apply_global_settings\", None)"),
        )

    def test_per_scan_view_state_is_separate_from_workflow_and_restored_after_payload(self):
        embedded = (GUI / "roi_explorer" / "vtk_backend" / "embedded_widget.py").read_text(encoding="utf-8")
        state_module = GUI / "roi_explorer" / "vtk_backend" / "scan_view_state.py"
        self.assertTrue(state_module.is_file())
        self.assertIn('project_dir / "view_state.json"', embedded)
        self.assertIn('"EndInteractionEvent"', embedded)
        self.assertIn("self._scene.restore_camera_state", embedded)
        self.assertIn("save_view_state(self._view_state_path", embedded)

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

    def test_embedded_explorer_source_icon_variants_are_available(self):
        embedded = GUI / "roi_explorer" / "vtk_backend" / "embedded_widget.py"
        source = embedded.read_text(encoding="utf-8")
        self.assertIn("SOURCE_ICON_VARIANT = 1", source)
        self.assertIn("vtk_volume_source_{variant}.svg", source)
        for index in range(1, 5):
            path = ICONS / f"vtk_volume_source_{index}.svg"
            self.assertTrue(path.is_file(), path.name)
            self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"), path.name)

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
