"""Static regression checks for the Matplotlib footprint rendering path.

These checks deliberately do not import the Qt widget: they run in headless
CI environments without PyQt6 and catch unresolved globals before the UI is
opened by a user.
"""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path


_REPO_ROOT = Path(__file__).parents[3]
_MAP_WIDGET_SOURCE = (
    _REPO_ROOT
    / "src"
    / "roc_mcs"
    / "gui"
    / "roi_explorer"
    / "map_plot_widget.py"
)


def test_branch_footprint_renderer_has_no_unresolved_global_names():
    source = _MAP_WIDGET_SOURCE.read_text(encoding="utf-8")
    module_table = symtable.symtable(source, str(_MAP_WIDGET_SOURCE), "exec")
    module_names = {symbol.get_name() for symbol in module_table.get_symbols()}

    method_tables = [
        child
        for child in module_table.get_children()
        if child.get_name() == "MapPlotWidget"
    ]
    assert len(method_tables) == 1, "MapPlotWidget class was not found uniquely"
    methods = [
        child
        for child in method_tables[0].get_children()
        if child.get_name() == "draw_branch_footprints"
    ]
    assert len(methods) == 1, "draw_branch_footprints method was not found uniquely"

    unresolved = sorted({
        symbol.get_name()
        for symbol in methods[0].get_symbols()
        if symbol.is_global()
        and symbol.is_referenced()
        and symbol.get_name() not in module_names
        and not hasattr(builtins, symbol.get_name())
    })
    assert unresolved == [], (
        "draw_branch_footprints references names that are not defined in its "
        f"module or local scope: {unresolved}"
    )


def test_footprint_annotation_uses_the_map_widgets_active_theme():
    source = _MAP_WIDGET_SOURCE.read_text(encoding="utf-8")
    assert "annotation_theme = theme_colors(self._theme)" in source
    assert 'color=annotation_theme["text"]' in source
    assert '"facecolor": annotation_theme["base"]' in source
    assert '"edgecolor": annotation_theme["border"]' in source
