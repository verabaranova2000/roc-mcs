from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "roc_mcs" / "gui" / "roi_explorer" / "vtk_backend"


def _source(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def _method(source: str, class_name: str, method_name: str):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name)


def test_display_hud_is_native_sibling_of_qvtk_and_rebound_on_teardown_and_reopen():
    source = _source("embedded_widget.py")
    assert "VTKVolumeDisplayWidget(self.viewport)" in source
    assert "self.display.set_viewport(widget)" not in source
    assert "self.display.set_viewport(self.viewport)" in source
    assert "self.viewport_layout = QtWidgets.QGridLayout(self.viewport)" in source
    assert "self.display, 0, 0" in source
    display = _source("vtk_volume_display_widget.py")
    assert "WA_NativeWindow" in display
    assert "WA_AlwaysStackOnTop" in display
    assert "content_height = max(190, int(self.sizeHint().height()))" in display


def test_roi_preview_slot_queues_render_instead_of_rendering_inside_selector_callback():
    source = _source("embedded_widget.py")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EmbeddedVTKVolumeWidget")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_update_roi_preview")
    calls = [node.func.attr for node in ast.walk(method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "start" in calls
    assert "_render" not in calls
    assert "_flush_roi_preview" in source


def test_tools_initial_theme_is_inherited_not_forced_to_light():
    source = _source("vtk_tools_widget.py")
    assert "self.apply_theme(resolve_theme(parent if parent is not None else self))" in source
    assert 'self.apply_theme("light")' not in source


def test_projection_palette_is_resolved_using_active_scene_theme():
    source = _source("vtk_scene.py")
    assert "resolve_projection_palette(configured, self.theme)" in source


def test_application_theme_is_set_before_vtk_tools_are_constructed():
    source = _source("embedded_widget.py")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EmbeddedVTKVolumeWidget")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    nodes = list(ast.walk(init))
    property_call = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setProperty" and any(isinstance(arg, ast.Constant) and arg.value == "rocMcsTheme" for arg in node.args))
    tools_call = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "VTKToolsWidget")
    assert property_call < tools_call


def test_display_hud_rebind_closes_popup_before_reparenting():
    source = _source("vtk_volume_display_widget.py")
    method = _method(source, "VTKVolumeDisplayWidget", "set_viewport")
    calls = [node.func.attr for node in ast.walk(method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "close" in calls
    assert "removeEventFilter" in calls
    assert "installEventFilter" in calls
    assert "setParent" in calls


def test_qvtk_paint_path_does_not_force_synchronous_repaints_or_overlay_repaints():
    embedded_path = SRC / "embedded_widget.py"
    source = embedded_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "repaint" not in calls
    assert "previous_output_window.ShowWindowOff()" in source
    assert "vtkOutputWindow.SetInstance(output_window)" in source
    assert "output_window.SetDisplayModeToAlways()" in source

    loader = _method(source, "EmbeddedVTKVolumeWidget", "_load_vtk_modules")
    tracked = next(
        node for node in ast.walk(loader)
        if isinstance(node, ast.ClassDef) and node.name == "_TrackedQVTK"
    )
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "paintEvent"
        for node in tracked.body
    )

    display_path = SRC / "vtk_volume_display_widget.py"
    display_source = display_path.read_text(encoding="utf-8")
    restack = _method(display_source, "VTKVolumeDisplayWidget", "restack_over_viewport")
    restack_calls = [
        node.func.attr
        for node in ast.walk(restack)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "raise_" in restack_calls
    assert "update" not in restack_calls
    event_filter = _method(display_source, "VTKVolumeDisplayWidget", "eventFilter")
    assert "LayoutRequest" not in ast.unparse(event_filter)
    position = _method(display_source, "VTKVolumeDisplayWidget", "_position_overlay")
    position_calls = [
        node.func.attr
        for node in ast.walk(position)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "raise_" not in position_calls
    sync = _method(display_source, "VTKVolumeDisplayWidget", "_sync_visibility")
    sync_calls = [
        node.func.attr
        for node in ast.walk(sync)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "update" not in sync_calls


def test_vtk_popup_fonts_are_explicitly_point_sized_and_valid():
    display_path = SRC / "vtk_volume_display_widget.py"
    source = display_path.read_text(encoding="utf-8")
    assert "def _make_positive_point_font" in source
    assert "font.pointSizeF() <= 0.0" in source
    assert "self.settings_menu.setFont(self._menu_font)" in source
    assert "panel.setFont(self._menu_font)" in source
    assert "self.setFont(self._menu_font)" in source
    assert "font-size: 6pt" in source
