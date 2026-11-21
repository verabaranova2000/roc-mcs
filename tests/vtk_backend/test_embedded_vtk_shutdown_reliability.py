from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "roc_mcs" / "gui" / "roi_explorer"

def _tree(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)

def _method(cls, name):
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)

def _calls(node):
    return [child for child in ast.walk(node) if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)]

def test_native_hud_popup_uses_outer_owner_and_painter_checks():
    path = SRC / "vtk_backend" / "vtk_volume_display_widget.py"
    text = path.read_text(encoding="utf-8")
    assert "self.settings_menu = QMenu(self._popup_owner)" in text
    assert "def _root_window_widget(widget: QWidget) -> QWidget" in text
    assert text.count("if not painter.isActive():") >= 2

def test_temporary_and_final_hud_shutdown_are_distinct_and_idempotent():
    path = SRC / "vtk_backend" / "vtk_volume_display_widget.py"
    cls = _class(_tree(path), "VTKVolumeDisplayWidget")
    method = ast.unparse(_method(cls, "prepare_for_shutdown"))
    assert "final: bool=False" in method
    assert "settings_menu.close()" in method
    assert "settings_menu.deleteLater()" in method
    assert "destroy(" not in method and "setParent(None)" not in method

def test_release_cleans_native_hud_even_when_scene_was_already_released():
    path = SRC / "vtk_backend" / "embedded_widget.py"
    cls = _class(_tree(path), "EmbeddedVTKVolumeWidget")
    method = ast.unparse(_method(cls, "release_vtk_scene"))
    assert "prepare_for_shutdown(final=final)" in method
    assert "_final_shutdown_pending" in method
    assert "if self._release_in_progress:" in method

def test_scene_detaches_before_qvtk_close_and_no_second_interactor_disable():
    path = SRC / "vtk_backend" / "embedded_widget.py"
    tree = _tree(path)
    cls = _class(tree, "EmbeddedVTKVolumeWidget")
    method = _method(cls, "release_vtk_scene")
    source = ast.unparse(method)
    assert source.index("scene.close()") < source.index("widget.close()")
    assert "interactor.Disable()" not in source
    tracked = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "_TrackedQVTK")
    close = _method(tracked, "closeEvent")
    names = [call.func.attr for call in _calls(close)]
    assert "stop" in names and "Disable" in names and "closeEvent" in names
    assert "Finalize" not in names

def test_embedded_theta_defaults_match_standalone():
    path = SRC / "vtk_backend" / "embedded_widget.py"
    text = path.read_text(encoding="utf-8")
    assert 'for name in ("turbo", "viridis"' in text
    assert 'findData("turbo")' in text
    assert 'self._spin(0.0, 1.0, 0.90, 2, step=0.05)' in text

def test_qvtk_init_has_no_fixed_80_100ms_wait_and_checks_native_surface():
    path = SRC / "vtk_backend" / "embedded_widget.py"
    text = path.read_text(encoding="utf-8")
    assert "singleShot(80" not in text
    assert "singleShot(100" not in text
    assert "WA_WState_Created" in text
    assert "widget.width() > 1" in text and "widget.height() > 1" in text

def test_custom_paint_events_guard_inactive_painters():
    plot_text = (SRC / "plot_workspace_widget.py").read_text(encoding="utf-8")
    embedded_text = (SRC / "vtk_backend" / "embedded_widget.py").read_text(encoding="utf-8")
    display_text = (SRC / "vtk_backend" / "vtk_volume_display_widget.py").read_text(encoding="utf-8")
    assert "if not p.isActive(): return" in plot_text
    assert "if not painter.isActive():" in embedded_text
    assert "if not marker.isActive():" in embedded_text
    assert display_text.count("if not painter.isActive():") >= 2
