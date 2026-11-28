"""Shared Qt window-parent and font helpers for native popup widgets."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QWidget


def top_level_window_owner(widget: QWidget | None) -> QWidget | None:
    """Return the real top-level QWidget suitable for a native transient parent.

    ``QWidget.isWindow()`` is also true for ``QMdiSubWindow``. Passing an MDI
    child as the owner of a ``Qt.Tool``/``QWindow`` popup can produce Qt's
    ``QWidgetWindow must be a top level window`` warning. Resolve the actual
    application window, including when the supplied widget is nested several
    levels below an MDI subwindow.
    """
    if widget is None:
        return None
    candidate = widget.window()
    while candidate is not None and candidate.windowType() == Qt.WindowType.SubWindow:
        parent = candidate.parentWidget()
        candidate = parent.window() if parent is not None else None
    if (
        candidate is not None
        and candidate.isWindow()
        and candidate.windowType() != Qt.WindowType.SubWindow
    ):
        return candidate
    return None


def point_sized_font(widget: QWidget | None = None, *, fallback: float = 9.0) -> QFont:
    """Copy a widget/application font with a valid point size.

    Qt represents pixel-sized fonts with ``pointSizeF() == -1``. Deriving a
    smaller point font from that value can emit ``QFont::setPointSize``
    warnings, so native popup roots use this normalized copy.
    """
    source = widget.font() if widget is not None else QApplication.font()
    font = QFont(source)
    if float(font.pointSizeF()) <= 0.0:
        font = QFont(QApplication.font())
    if float(font.pointSizeF()) <= 0.0:
        font = QFont("Sans Serif")
    if float(font.pointSizeF()) <= 0.0:
        font.setPointSizeF(max(6.0, float(fallback)))
    return font
