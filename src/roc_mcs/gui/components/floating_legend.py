"""Reusable, canvas-attached floating legend panel for scientific views.

The panel is a QWidget child of the view canvas (not a QDockWidget): it can be
moved and resized inside a plot/viewport without participating in the host
window's layout. Its scrollable content can be either a continuous-alpha key
or a list of categorical entries, which also suits future branch legends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PyQt6.QtCore import QEvent, QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QMouseEvent, QPainter, QPen, QPalette
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class LegendEntry:
    """One categorical legend entry; ``detail`` is optional secondary text."""

    label: str
    color: str
    detail: str = ""


class _AlphaGradientRows(QWidget):
    """Compact 2D key: category is encoded by hue, intensity by alpha."""

    def __init__(self, rows: Iterable[LegendEntry], axis_label: str, parent=None):
        super().__init__(parent)
        self.rows = tuple(rows)
        self.axis_label = str(axis_label)
        self.setMinimumWidth(188)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(142)

    def sizeHint(self) -> QSize:
        return QSize(242, max(142, 58 + len(self.rows) * 19))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        text_color = palette.color(QPalette.ColorRole.Text)
        muted_color = palette.color(QPalette.ColorRole.PlaceholderText)
        border_color = palette.color(QPalette.ColorRole.Mid)
        width = max(1, self.width())
        left_label = 34
        right_pad = 10
        strip_left = left_label + 5
        strip_right = max(strip_left + 42, width - right_pad)
        strip_width = strip_right - strip_left

        painter.setPen(text_color)
        heading_font = painter.font()
        heading_font.setBold(True)
        heading_font.setPointSizeF(max(7.0, heading_font.pointSizeF() - 0.5))
        painter.setFont(heading_font)
        painter.drawText(QRect(8, 5, width - 16, 17), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         "ИНТЕГРАЛЬНАЯ ИНТЕНСИВНОСТЬ  →")

        row_top = 27
        row_height = 13
        row_gap = 5
        painter.setFont(self.font())
        for index, entry in enumerate(self.rows):
            y = row_top + index * (row_height + row_gap)
            painter.setPen(text_color)
            painter.drawText(QRect(7, y - 1, left_label - 5, row_height + 2),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             entry.label)
            try:
                color = QColor(entry.color)
            except Exception:
                color = palette.color(QPalette.ColorRole.Highlight)
            gradient = QLinearGradient(float(strip_left), 0.0, float(strip_right), 0.0)
            gradient.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 10))
            gradient.setColorAt(0.18, QColor(color.red(), color.green(), color.blue(), 45))
            gradient.setColorAt(0.52, QColor(color.red(), color.green(), color.blue(), 138))
            gradient.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 255))
            painter.setPen(QPen(border_color, 0.65))
            painter.setBrush(gradient)
            painter.drawRoundedRect(QRect(strip_left, y, strip_width, row_height), 2.0, 2.0)

        labels_y = row_top + len(self.rows) * (row_height + row_gap) + 1
        painter.setPen(muted_color)
        small_font = painter.font()
        small_font.setPointSizeF(max(6.5, small_font.pointSizeF() - 1.0))
        painter.setFont(small_font)
        painter.drawText(QRect(strip_left, labels_y, strip_width // 2, 15),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "низкая")
        painter.drawText(QRect(strip_left + strip_width // 2, labels_y, strip_width - strip_width // 2, 15),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "высокая")
        painter.drawText(QRect(8, labels_y + 15, width - 16, 18),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.axis_label)
        painter.end()


class _LegendEntryList(QWidget):
    """Scrollable categorical list used by long peak-branch/component legends."""

    def __init__(self, entries: Iterable[LegendEntry], parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 6, 7, 6)
        layout.setSpacing(4)
        for entry in entries:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(7)
            swatch = QFrame(self)
            swatch.setFixedSize(11, 11)
            swatch.setStyleSheet(
                "QFrame { border: 1px solid palette(mid); border-radius: 3px; "
                f"background-color: {entry.color}; }}"
            )
            row.addWidget(swatch, 0, Qt.AlignmentFlag.AlignVCenter)
            text_column = QVBoxLayout()
            text_column.setContentsMargins(0, 0, 0, 0)
            text_column.setSpacing(0)
            label = QLabel(entry.label, self)
            label.setWordWrap(True)
            text_column.addWidget(label)
            if entry.detail:
                detail = QLabel(entry.detail, self)
                detail.setWordWrap(True)
                detail.setStyleSheet("color: palette(placeholderText); font-size: 9px;")
                text_column.addWidget(detail)
            row.addLayout(text_column, 1)
            layout.addLayout(row)
        layout.addStretch(1)
        self.setMinimumWidth(178)


class _DragHeader(QFrame):
    def __init__(self, panel: "FloatingLegendPanel"):
        super().__init__(panel)
        self._panel = panel
        self._last_global: QPoint | None = None
        self.setObjectName("floatingLegendHeader")
        self.setFixedHeight(27)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_global = event.globalPosition().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_global is not None and event.buttons() & Qt.MouseButton.LeftButton:
            current = event.globalPosition().toPoint()
            delta = current - self._last_global
            self._last_global = current
            self._panel._move_by(delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_global = None
        super().mouseReleaseEvent(event)


class _ResizeGrip(QWidget):
    """Small bottom-right grip that resizes this child overlay, not its window."""

    def __init__(self, panel: "FloatingLegendPanel"):
        super().__init__(panel)
        self._panel = panel
        self._start_global: QPoint | None = None
        self._start_size = QSize()
        self.setFixedSize(15, 15)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setToolTip("Изменить размер легенды")

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        color = self.palette().color(QPalette.ColorRole.PlaceholderText)
        painter.setPen(QPen(color, 1.0))
        for offset in (4, 8, 12):
            painter.drawLine(self.width() - offset, self.height() - 2,
                             self.width() - 2, self.height() - offset)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_global = event.globalPosition().toPoint()
            self._start_size = self._panel.size()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start_global is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._start_global
            self._panel.resize(
                max(self._panel.minimumWidth(), self._start_size.width() + delta.x()),
                max(self._panel.minimumHeight(), self._start_size.height() + delta.y()),
            )
            self._panel._clamp_to_parent()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_global = None
        super().mouseReleaseEvent(event)


class FloatingLegendPanel(QFrame):
    """Movable, resizable, collapsible and internally scrollable canvas legend.

    Create it as a child of the canvas/viewport it should overlay.  Content is
    intentionally supplied by the view so the same frame can host a 2D
    intensity key today and a long branch-ID key in a future 3D view.
    """

    def __init__(
        self,
        parent: QWidget,
        title: str = "ЛЕГЕНДА",
        *,
        default_size: QSize = QSize(258, 208),
        margin: int = 10,
    ):
        super().__init__(parent)
        self.setObjectName("FloatingLegendPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(188, 48)
        self.setMaximumSize(520, 620)
        self._margin = max(0, int(margin))
        self._expanded_size = QSize(default_size)
        self._user_moved = False
        self._collapsed = False
        self._content_widget: QWidget | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "QFrame#FloatingLegendPanel { background: palette(window); "
            "border: 1px solid palette(mid); border-radius: 5px; }"
            "QFrame#floatingLegendHeader { background: palette(button); "
            "border: 0; border-bottom: 1px solid palette(mid); border-radius: 4px 4px 0 0; }"
            "QLabel#floatingLegendTitle { font-size: 9px; font-weight: 700; "
            "letter-spacing: 0.5px; color: palette(window-text); background: transparent; }"
            "QToolButton#floatingLegendCollapse { border: 0; padding: 0; "
            "min-width: 20px; max-width: 20px; min-height: 20px; "
            "color: palette(window-text); background: transparent; }"
            "QToolButton#floatingLegendCollapse:hover { background: palette(midlight); border-radius: 3px; }"
            "QScrollArea { border: 0; background: transparent; }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        self.header = _DragHeader(self)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 0, 4, 0)
        header_layout.setSpacing(4)
        self.title_label = QLabel(str(title), self.header)
        self.title_label.setObjectName("floatingLegendTitle")
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header_layout.addWidget(self.title_label, 1)
        self.collapse_button = QToolButton(self.header)
        self.collapse_button.setObjectName("floatingLegendCollapse")
        self.collapse_button.setText("−")
        self.collapse_button.setToolTip("Свернуть легенду")
        self.collapse_button.setAccessibleName("Свернуть или развернуть легенду")
        self.collapse_button.clicked.connect(self.toggle_collapsed)
        header_layout.addWidget(self.collapse_button)
        outer.addWidget(self.header)

        body_row = QHBoxLayout()
        body_row.setContentsMargins(0, 0, 1, 1)
        body_row.setSpacing(0)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        body_row.addWidget(self.scroll_area, 1)
        self.resize_grip = _ResizeGrip(self)
        body_row.addWidget(self.resize_grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        outer.addLayout(body_row, 1)

        self.resize(default_size)
        parent.installEventFilter(self)
        self._place_initially()
        self.hide()

    def set_title(self, title: str) -> None:
        self.title_label.setText(str(title))

    def set_content_widget(self, widget: QWidget) -> None:
        previous = self.scroll_area.takeWidget()
        if previous is not None and previous is not widget:
            previous.deleteLater()
        self._content_widget = widget
        widget.setParent(self.scroll_area)
        self.scroll_area.setWidget(widget)

    def set_alpha_gradient_rows(
        self,
        rows: Iterable[LegendEntry],
        *,
        axis_label: str = "Интенсивность",
    ) -> None:
        self.set_content_widget(_AlphaGradientRows(rows, axis_label))

    def set_entries(self, entries: Iterable[LegendEntry]) -> None:
        """Set a scrollable categorical list (e.g. diffraction branch IDs)."""
        self.set_content_widget(_LegendEntryList(entries))

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle_collapsed(self) -> None:
        if self._collapsed:
            self._collapsed = False
            self.setMinimumHeight(48)
            self.setMaximumHeight(620)
            self.scroll_area.show()
            self.resize_grip.show()
            self.resize(self._expanded_size)
            self.collapse_button.setText("−")
            self.collapse_button.setToolTip("Свернуть легенду")
        else:
            self._expanded_size = self.size()
            self._collapsed = True
            self.scroll_area.hide()
            self.resize_grip.hide()
            self.setFixedHeight(self.header.height() + 2)
            self.collapse_button.setText("＋")
            self.collapse_button.setToolTip("Развернуть легенду")
        self._clamp_to_parent()

    def show_on_canvas(self) -> None:
        if self.parentWidget() is not None and not self._user_moved:
            self._place_initially()
        self.show()
        self.raise_()

    def hide_on_canvas(self) -> None:
        self.hide()

    def set_panel_palette(self, palette) -> None:
        self.setPalette(palette)
        self.header.setPalette(palette)
        self.title_label.setPalette(palette)
        self.collapse_button.setPalette(palette)
        if self._content_widget is not None:
            self._content_widget.setPalette(palette)
        self.update()

    def _place_initially(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        x = max(self._margin, parent.width() - self.width() - self._margin)
        y = self._margin + 2
        self.move(x, y)

    def _move_by(self, delta: QPoint) -> None:
        self._user_moved = True
        self.move(self.pos() + delta)
        self._clamp_to_parent()

    def _clamp_to_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        available_w = max(1, parent.width() - 2 * self._margin)
        available_h = max(1, parent.height() - 2 * self._margin)
        if self.width() > available_w:
            self.resize(max(1, available_w), self.height())
        if not self._collapsed and self.height() > available_h:
            self.resize(self.width(), max(self.header.height() + 2, available_h))
        max_x = max(self._margin, parent.width() - self.width() - self._margin)
        max_y = max(self._margin, parent.height() - self.height() - self._margin)
        self.move(min(max(self._margin, self.x()), max_x), min(max(self._margin, self.y()), max_y))
        self._layout_grip()

    def _layout_grip(self) -> None:
        if hasattr(self, "resize_grip"):
            self.resize_grip.move(self.width() - self.resize_grip.width() - 2,
                                  self.height() - self.resize_grip.height() - 2)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_grip()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            if not self._user_moved and not self._collapsed:
                self._place_initially()
            self._clamp_to_parent()
        return super().eventFilter(watched, event)
