import os

import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from roc_mcs.gui.roi_explorer.branch_manager_widget import BranchManagerWidget


def _app():
    return QApplication.instance() or QApplication([])


def test_branch_picker_disables_colors_used_by_other_branches():
    app = _app()
    manager = BranchManagerWidget()
    manager.set_branches(
        [
            {"branch_id": 1, "event_count": 4, "occupied_pixels": 3, "analysis_included": True},
            {"branch_id": 2, "event_count": 5, "occupied_pixels": 4, "analysis_included": True},
        ],
        {1: "#FF0000", 2: "#0000FF"},
    )

    index = manager.model.index(1, 0)
    manager._choose_color(2, index)
    picker = manager._active_color_picker

    assert picker is not None
    assert "#FF0000" in picker._unavailable_colors
    red_cell = next(
        polygon for polygon, color in picker.swatches._cells
        if color.upper() == "#FF0000"
    )
    assert picker.swatches._index_at(red_cell.boundingRect().center()) == -1
    assert picker.swatches._index_at(
        next(
            polygon.boundingRect().center()
            for polygon, color in picker.swatches._cells
            if color.upper() == "#0000FF"
        )
    ) >= 0

    picker.close()
    manager.close()
    app.processEvents()


def test_branch_picker_rejects_reserved_hex_and_manager_rejects_duplicate_assignment():
    app = _app()
    manager = BranchManagerWidget()
    manager.set_branches(
        [
            {"branch_id": 1, "event_count": 4, "occupied_pixels": 3, "analysis_included": True},
            {"branch_id": 2, "event_count": 5, "occupied_pixels": 4, "analysis_included": True},
        ],
        {1: "#FF0000", 2: "#0000FF"},
    )
    manager._choose_color(2, manager.model.index(1, 0))
    picker = manager._active_color_picker

    assert picker._set_color("#ff0000") is False
    assert picker._color.name().upper() == "#0000FF"
    assert picker.apply_button.isEnabled() is False
    assert manager.is_branch_color_available(2, "#00FF00") is True
    assert manager.is_branch_color_available(2, "#FF0000") is False
    assert manager.set_branch_color(2, "#FF0000") is False
    assert manager.branch_colors() == {1: "#FF0000", 2: "#0000FF"}

    picker.close()
    manager.close()
    app.processEvents()
