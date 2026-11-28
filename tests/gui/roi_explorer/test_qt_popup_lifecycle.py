"""Qt popup ownership/font regressions (requires the application's PyQt6 runtime)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMainWindow, QMdiArea, QMdiSubWindow, QWidget

from roc_mcs.gui.roi_explorer.hex_color_picker import CompactColorPicker
from roc_mcs.gui.roi_explorer.theme import point_sized_font, top_level_window_owner


def _app():
    return QApplication.instance() or QApplication([])


def test_tool_popup_uses_real_application_window_not_mdi_subwindow():
    app = _app()
    main = QMainWindow()
    mdi = QMdiArea(main)
    main.setCentralWidget(mdi)
    subwindow = QMdiSubWindow()
    nested_manager = QWidget()
    subwindow.setWidget(nested_manager)
    mdi.addSubWindow(subwindow)

    assert nested_manager.window() is subwindow
    assert top_level_window_owner(nested_manager) is main

    picker = CompactColorPicker("#336699", nested_manager)
    assert picker.parentWidget() is main
    assert picker.font().pointSizeF() > 0
    picker.close()
    main.close()
    app.processEvents()


def test_point_font_falls_back_from_pixel_sized_application_font():
    app = _app()
    original = app.font()
    try:
        from PyQt6.QtGui import QFont
        pixel_font = QFont("Sans Serif")
        pixel_font.setPixelSize(13)
        app.setFont(pixel_font)
        assert point_sized_font().pointSizeF() > 0
    finally:
        app.setFont(original)
