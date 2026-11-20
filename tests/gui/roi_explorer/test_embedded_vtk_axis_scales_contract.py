from __future__ import annotations

import ast
from pathlib import Path


def _embedded_widget_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    candidates = (
        root / "src" / "roc_mcs" / "gui" / "roi_explorer" / "vtk_backend" / "embedded_widget.py",
        root / "src" / "gui" / "roi_explorer" / "vtk_backend" / "embedded_widget.py",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def test_axis_scales_property_is_passed_without_calling_it():
    path = _embedded_widget_path()
    assert path.is_file(), f"Не найден embedded_widget.py: {path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "axis_scales"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "tools"
    ]
    assert not bad_calls, "VTKToolsWidget.axis_scales — @property; нельзя писать self.tools.axis_scales()."
    correct = [
        kw for node in ast.walk(tree) if isinstance(node, ast.Call)
        for kw in node.keywords if kw.arg == "axis_scales"
        and isinstance(kw.value, ast.Attribute) and kw.value.attr == "axis_scales"
        and isinstance(kw.value.value, ast.Attribute) and kw.value.value.attr == "tools"
    ]
    assert correct, "VTKVolumeScene.set_payload должен получать self.tools.axis_scales как значение свойства."
