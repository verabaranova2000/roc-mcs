from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "roc_mcs" / "gui" / "roi_explorer"

def _method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)

def test_initial_visible_panels_are_not_hidden_waiting_for_layout():
    path = SRC / "plot_workspace_widget.py"
    add = ast.unparse(_method(path, "_PlotMdiArea", "add_plot"))
    initial = ast.unparse(_method(path, "_PlotMdiArea", "_layout_initial"))
    assert "if initial_visible" in add and "win.show()" in add
    assert "win.hide()" in add  # only the explicitly initially-hidden panel branch
    assert "win.show()" not in initial
    assert "win.hide()" not in initial

def test_opening_panel_commits_geometry_before_first_show():
    path = SRC / "plot_workspace_widget.py"
    show = ast.unparse(_method(path, "_PlotMdiArea", "show_panel"))
    assert show.index("_apply_layout") < show.index("win.show()")
    assert "singleShot(100" not in path.read_text(encoding="utf-8")

def test_maximize_restore_restack_without_scene_rebuild():
    path = SRC / "plot_workspace_widget.py"
    state = ast.unparse(_method(path, "_PlotMdiArea", "_on_window_state_changed"))
    restack = ast.unparse(_method(path, "_PlotMdiArea", "_restack_panel"))
    assert "WindowMaximized" in state and "singleShot(0" in state
    assert "lower" in restack and "raise_" in restack
    assert "release_vtk_scene" not in state and "refresh_volume" not in state

def test_normal_geometry_saved_while_maximized_or_minimized():
    path = SRC / "plot_workspace_widget.py"
    save = ast.unparse(_method(path, "_PlotMdiArea", "save_geometry"))
    assert "normalGeometry" in save
    assert "isMaximized" in save and "isMinimized" in save

def test_roc_icon_uses_its_own_slot_in_all_themes():
    text = (SRC / "curve_tools_widget.py").read_text(encoding="utf-8")
    assert text.count("curve_overlay_add.svg") == 2
    assert "curve_overlay_3.svg" not in text
