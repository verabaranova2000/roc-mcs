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


def test_display_hud_is_child_of_qvtk_and_rebound_on_teardown_and_reopen():
    source = _source("embedded_widget.py")
    assert "VTKVolumeDisplayWidget(widget)" in source
    assert "self.display.set_viewport(widget)" in source
    assert "self.display.set_viewport(self.viewport)" in source


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
