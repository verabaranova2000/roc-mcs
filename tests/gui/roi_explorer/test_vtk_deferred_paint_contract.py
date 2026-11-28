"""Regression contract: QVTK must only paint through the Qt event loop."""
import ast
from pathlib import Path


def test_desktop_viewer_does_not_call_synchronous_widget_repaint():
    source_path = (
        Path(__file__).parents[3]
        / "src" / "roc_mcs" / "gui" / "roi_explorer" / "vtk_backend" / "desktop_viewer.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    repaint_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "repaint"
    ]

    assert repaint_calls == []
    assert "QTimer.singleShot(0, self.vtk_widget.update)" in source_path.read_text(encoding="utf-8")
