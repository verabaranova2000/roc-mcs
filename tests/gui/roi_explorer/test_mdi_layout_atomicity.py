from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_CANDIDATES = (ROOT / "src" / "roc_mcs", ROOT / "src")
SRC = next(path for path in SRC_CANDIDATES if (path / "gui" / "roi_explorer").is_dir())


def _method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def test_mdi_children_are_hidden_until_the_first_geometry_commit():
    path = SRC / "gui/roi_explorer/plot_workspace_widget.py"
    add_plot = ast.unparse(_method(path, "_PlotMdiArea", "add_plot"))
    initial = ast.unparse(_method(path, "_PlotMdiArea", "_layout_initial"))
    show_panel = ast.unparse(_method(path, "_PlotMdiArea", "show_panel"))
    assert "win.hide()" in add_plot
    assert "win.show()" in initial
    assert initial.index("setGeometry") < initial.index("win.show()")
    assert show_panel.index("_apply_layout") < show_panel.index("win.show()")
    assert "singleShot(100" not in path.read_text(encoding="utf-8")


def test_maximize_and_restore_repair_stacking_without_scene_rebuild():
    path = SRC / "gui/roi_explorer/plot_workspace_widget.py"
    state = ast.unparse(_method(path, "_PlotMdiArea", "_on_window_state_changed"))
    keep = ast.unparse(_method(path, "_PlotMdiArea", "_keep_panel_on_top"))
    assert "WindowMaximized" in state
    assert "_keep_panel_on_top" in state
    assert "panel_restored.emit" in state
    assert "lower" in keep and "raise_" in keep
    assert "refresh_volume" not in state
    assert "release_vtk_scene" not in state


def test_vtk_initialization_has_no_fixed_80_or_100_ms_startup_wait():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    schedule = ast.unparse(_method(path, "EmbeddedVTKVolumeWidget", "_schedule_vtk_init"))
    ensure = ast.unparse(_method(path, "EmbeddedVTKVolumeWidget", "_ensure_vtk"))
    assert "singleShot(0" in schedule
    assert "singleShot(0" in ensure
    assert "singleShot(80" not in schedule
    assert "singleShot(100" not in ensure


def test_terminal_close_quiesces_hud_and_leaves_native_child_lifetime_to_qt():
    embedded = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    release = ast.unparse(_method(embedded, "EmbeddedVTKVolumeWidget", "release_vtk_scene"))
    display = SRC / "gui/roi_explorer/vtk_backend/vtk_volume_display_widget.py"
    prepare = ast.unparse(_method(display, "VTKVolumeDisplayWidget", "prepare_for_shutdown"))
    assert "prepare_for_shutdown(final=True)" in release
    assert "_final_shutdown_pending" in release
    assert "settings_menu.deleteLater()" in prepare
    assert "destroy(" not in prepare
    assert "setParent(None)" not in prepare
    # The interactor is disabled only in _TrackedQVTK.closeEvent; scene.close()
    # runs first so it can unregister its widgets while the interactor is live.
    assert "interactor_to_disable.Disable()" not in release
    assert release.index("scene.close()") < release.index("widget.close()")


def test_native_hud_popup_uses_top_level_owner_not_native_child():
    display = SRC / "gui/roi_explorer/vtk_backend/vtk_volume_display_widget.py"
    text = display.read_text(encoding="utf-8")
    assert "self.settings_menu = QMenu(self._popup_owner)" in text
    assert "self.settings_menu.setParent(self._popup_owner, menu_flags)" in text
    assert "def _root_window_widget(widget: QWidget) -> QWidget" in text
    assert "while owner.parentWidget() is not None" in text
    assert "self.settings_menu = QMenu(self)" not in text


def test_native_hud_paint_events_check_painter_before_drawing():
    display = SRC / "gui/roi_explorer/vtk_backend/vtk_volume_display_widget.py"
    text = display.read_text(encoding="utf-8")
    assert text.count("if not painter.isActive():") >= 2


def test_embedded_theta_controls_match_standalone_defaults():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    text = path.read_text(encoding="utf-8")
    assert 'for name in ("turbo", "viridis"' in text
    assert 'findData("turbo")' in text
    assert 'self._spin(0.0, 1.0, 0.90, 2, step=0.05)' in text


def test_icon_proposals_keep_legacy_slots_and_separate_roc_from_standalone_volume():
    tools = SRC / "gui/roi_explorer/vtk_backend/vtk_tools_widget.py"
    tools_text = tools.read_text(encoding="utf-8")
    assert '"theta": "vtk_theta_plane_1.svg"' in tools_text
    assert '"projection": "vtk_xy_projection.svg"' in tools_text
    assert 'scene_icon_name = "vtk_scene_objects_1.svg"' in tools_text
    curve = SRC / "gui/roi_explorer/curve_tools_widget.py"
    volume = SRC / "gui/roi_explorer/volume_tools_widget.py"
    assert '"curve_overlay_add.svg"' in curve.read_text(encoding="utf-8")
    assert '"curve_overlay_3.svg"' in volume.read_text(encoding="utf-8")
    icons = SRC / "gui/resources/icons"
    for name in ("vtk_scene_add_A.svg", "vtk_scene_add_B.svg", "vtk_scene_add_C.svg",
                 "vtk_theta_plane_A.svg", "vtk_theta_plane_B.svg",
                 "vtk_xy_projection_A.svg", "vtk_xy_projection_B.svg",
                 "curve_overlay_add_A.svg", "curve_overlay_add_B.svg", "curve_overlay_add_C.svg"):
        assert (icons / "proposals" / name).is_file(), name
