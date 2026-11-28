from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "roc_mcs"
ROI_GUI = SRC / "gui" / "roi_explorer"


def _method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def test_footprint_renderer_restores_map_geometry_instead_of_forcing_auto_aspect():
    method = ast.unparse(_method(ROI_GUI / "map_plot_widget.py", "MapPlotWidget", "draw_branch_footprints"))
    assert "get_aspect()" in method
    assert "aspect=axes_state['aspect']" in method
    assert "self.ax.set_xlim(axes_state['xlim'])" in method
    assert "self.ax.set_ylim(axes_state['ylim'])" in method
    assert "set_autoscalex_on(axes_state['autoscalex'])" in method
    assert "set_autoscaley_on(axes_state['autoscaley'])" in method
    assert "add_collection(contours, autolim=False)" in method
    assert 'aspect="auto"' not in method


def test_preview_and_pinned_profiles_require_identical_nonempty_theta_and_curve_shapes():
    explorer = (ROI_GUI / "explorer_window.py").read_text(encoding="utf-8")
    update_preview = ast.unparse(_method(ROI_GUI / "explorer_window.py", "RockingCurveExplorerWindow", "_update_preview_curve"))
    refresh = ast.unparse(_method(ROI_GUI / "explorer_window.py", "RockingCurveExplorerWindow", "refresh_curves"))
    assert "theta_values.shape == curve_values.shape" in update_preview
    assert "pth.shape == pcurve.shape" in refresh
    assert "current_th.shape != curve.shape" in refresh
    assert "preview_th.shape == preview_curve.shape" in refresh
    assert "refresh_curves(" in update_preview
    assert "_update_preview_curve" in explorer


def test_curve_widget_rejects_malformed_pairs_before_relim():
    method = ast.unparse(_method(ROI_GUI / "curve_plot_widget.py", "CurvePlotWidget", "draw_curves"))
    assert "x_values.ndim != 1" in method
    assert "y_values.ndim != 1" in method
    assert "x_values.shape != y_values.shape" in method
    assert "np.isfinite(x_values) & np.isfinite(y_values)" in method
    assert method.index("valid_curves = []") < method.index("self.ax.relim()")
    assert "valid_preview =" in method and "valid_pixel =" in method


def test_branch_colors_are_reserved_in_picker_and_rechecked_before_persistence():
    manager = ast.unparse(_method(ROI_GUI / "branch_manager_widget.py", "BranchManagerWidget", "_choose_color"))
    available = ast.unparse(_method(ROI_GUI / "branch_manager_widget.py", "BranchManagerWidget", "is_branch_color_available"))
    handler = ast.unparse(_method(ROI_GUI / "explorer_window.py", "RockingCurveExplorerWindow", "_on_branch_color_changed"))
    assert "unavailable_colors" in manager
    assert "is_branch_color_available" in available
    assert "conflicting_branch" in handler
    assert "save_branch_display_colors" in handler
    assert handler.index("conflicting_branch") < handler.index("save_branch_display_colors")


def test_legacy_palette_normalization_resolves_collisions_deterministically():
    normalize = (SRC / "analysis" / "branch_display.py").read_text(encoding="utf-8")
    assert "candidate not in used" in normalize
    assert "_unused_default_branch_color" in normalize
