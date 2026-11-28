"""Keep non-modal Qt.Tool popups attached to genuine application windows."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _method(path: str, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return child
    raise AssertionError(f"Не найден {class_name}.{method_name} в {path}")


def _calls_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == name
        for child in ast.walk(node)
    )


def test_workbench_stage_popup_resolves_real_top_level_owner():
    method = _method(
        "src/roc_mcs/gui/processing_workbench/workbench_widget.py",
        "_StageConfigurationPopup", "__init__",
    )
    assert _calls_name(method, "top_level_window_owner")


def test_resource_monitor_resolves_real_top_level_owner():
    method = _method(
        "src/roc_mcs/gui/resource_monitor/panel.py",
        "ResourceMonitorPanel", "__init__",
    )
    assert _calls_name(method, "top_level_window_owner")
