from math import ceil, cos, pi, sin, sqrt
from PyQt6.QtCore import QPoint, QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygon, QPalette
from PyQt6.QtWidgets import (QColorDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QToolButton,
                             QVBoxLayout, QWidget, QDialog)
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, apply_popup_theme, resolve_theme


class _DragTitle(QLabel):
    """Заголовок для перетаскивания безрамочного окна."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent); self._drag_offset = None; self.setCursor(Qt.CursorShape.SizeAllCursor)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft(); event.accept(); return
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_offset); event.accept(); return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self._drag_offset = None; event.accept(); return
        super().mouseReleaseEvent(event)


class _HexSwatchCanvas(QWidget):
    """Цветовая мозаика из состыкованных шестиугольников."""
    color_clicked = pyqtSignal(str)
    COLORS = (
        "#171717", "#454545", "#777777", "#aaaaaa", "#d5d5d5", "#f4f4f4", "#ffffff",
        "#500000", "#800000", "#b00000", "#e02020", "#ff6565", "#ff9e9e", "#ffd0d0", "#ffe6e6",
        "#623000", "#8c4600", "#c46600", "#f28c00", "#ffb62e", "#ffda70", "#ffe9aa",
        "#514000", "#857000", "#b49a00", "#e6ca00", "#fff000", "#fff780", "#ffffc2", "#e9f3a1",
        "#244f1a", "#367d24", "#50a83a", "#7aca60", "#a5e18c", "#d5f0c8", "#164f68",
    )
    ROW_COUNTS = (7, 8, 7, 8, 7)

    def __init__(self, parent=None):
        super().__init__(parent); self._radius = 9.0; self._cells = []; self._hover = -1
        self.setFixedSize(ceil(8 * sqrt(3) * self._radius) + 1, ceil(8 * self._radius) + 1); self.setMouseTracking(True); self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_cells()

    def _build_cells(self):
        r = self._radius; w = sqrt(3) * r; index = 0
        for row, count in enumerate(self.ROW_COUNTS):
            start_x = (8 - count) * w / 2 + w / 2
            cy = r + row * 1.5 * r
            for col in range(count):
                cx = start_x + col * w
                points = [QPoint(round(cx + r * cos(pi / 2 + k * pi / 3)), round(cy + r * sin(pi / 2 + k * pi / 3))) for k in range(6)]
                self._cells.append((QPolygon(points), self.COLORS[index])); index += 1

    def _index_at(self, point):
        for index, (polygon, _) in enumerate(self._cells):
            if polygon.containsPoint(point, Qt.FillRule.OddEvenFill): return index
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._index_at(event.position().toPoint())
            if index >= 0: self.color_clicked.emit(self._cells[index][1])
            event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        index = self._index_at(event.position().toPoint())
        if index != self._hover: self._hover = index; self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover = -1; self.update(); super().leaveEvent(event)

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, (polygon, value) in enumerate(self._cells):
            color = QColor(value); p.setBrush(color); p.setPen(QPen(color, 1.4)); p.drawPolygon(polygon)
            if index == self._hover:
                p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), 1.5)); p.drawPolygon(polygon)
        p.end()


class _ColorPlane(QWidget):
    changed = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super().__init__(parent); self.setFixedSize(122, 66); self.hue = 210; self.saturation = self.value = 1.0; self.setCursor(Qt.CursorShape.CrossCursor)
    def set_hsv(self, hue: int, saturation: float, value: float):
        self.hue = int(hue) % 360; self.saturation = max(0.0, min(1.0, float(saturation))); self.value = max(0.0, min(1.0, float(value))); self.update()
    def _set_from_pos(self, pos):
        r = self.rect().adjusted(1, 1, -1, -1)
        if r.width() < 1 or r.height() < 1: return
        self.saturation = max(0.0, min(1.0, (pos.x() - r.left()) / r.width())); self.value = 1.0 - max(0.0, min(1.0, (pos.y() - r.top()) / r.height())); self.changed.emit(self.saturation, self.value); self.update()
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self._set_from_pos(event.position().toPoint())
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton: self._set_from_pos(event.position().toPoint())
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); r = self.rect().adjusted(1, 1, -1, -1)
        p.fillRect(r, QColor.fromHsv(self.hue, 255, 255)); white = QLinearGradient(r.left(), 0, r.right(), 0); white.setColorAt(0, QColor(255, 255, 255, 255)); white.setColorAt(1, QColor(255, 255, 255, 0)); p.fillRect(r, white)
        black = QLinearGradient(0, r.top(), 0, r.bottom()); black.setColorAt(0, QColor(0, 0, 0, 0)); black.setColorAt(1, QColor(0, 0, 0, 255)); p.fillRect(r, black)
        x = round(r.left() + self.saturation * r.width()); y = round(r.top() + (1 - self.value) * r.height()); p.setPen(QPen(Qt.GlobalColor.black, 3)); p.drawEllipse(QPoint(x, y), 4, 4); p.setPen(QPen(Qt.GlobalColor.white, 1.2)); p.drawEllipse(QPoint(x, y), 4, 4); p.setPen(QPen(self.palette().color(QPalette.ColorRole.Mid), 1)); p.drawRect(r); p.end()


class _HueStrip(QWidget):
    changed = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent); self.setFixedSize(10, 66); self.hue = 210; self.setCursor(Qt.CursorShape.PointingHandCursor)
    def _set_y(self, y):
        self.hue = max(0, min(359, round(y / max(1, self.height() - 1) * 359))); self.changed.emit(self.hue); self.update()
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self._set_y(event.position().y())
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton: self._set_y(event.position().y())
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); r = self.rect().adjusted(1, 1, -1, -1); grad = QLinearGradient(0, r.top(), 0, r.bottom())
        for i in range(7): grad.setColorAt(i / 6, QColor.fromHsv(min(359, i * 60), 255, 255))
        p.fillRect(r, grad); y = round(r.top() + self.hue / 359 * r.height()); p.setPen(QPen(Qt.GlobalColor.black, 3)); p.drawLine(0, y, self.width(), y); p.setPen(QPen(Qt.GlobalColor.white, 1)); p.drawLine(0, y, self.width(), y); p.end()


class CompactColorPicker(QFrame):
    """Компактный переносимый HSV/HEX-выбор цвета с бесшовной палитрой-сотами.

    Использование: picker = CompactColorPicker('#1f77b4', parent); picker.color_selected.connect(slot); picker.show().
    Сигнал color_selected отправляется при подтверждении кнопкой ✓; закрытие/✕ отменяет изменения.
    """
    color_selected = pyqtSignal(str)

    def __init__(self, color: str = "#1f77b4", parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("compactColorPicker"); apply_widget_palette(self, resolve_theme(parent))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True); self._color = QColor(color if QColor(color).isValid() else "#1f77b4")
        self.setStyleSheet("""
            QFrame#compactColorPicker { background: palette(window); color: palette(window-text); border: 1px solid palette(mid); border-radius: 5px; }
            QToolButton { border: 1px solid transparent; border-radius: 3px; padding: 0; font-size: 10px; }
            QToolButton:hover { background: palette(midlight); border-color: palette(mid); }
            QLineEdit { font-size: 10px; min-height: 19px; padding: 1px 4px; border: 1px solid palette(mid); border-radius: 3px; }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(5, 4, 5, 4)
        root.setSpacing(3)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.title = _DragTitle("Color", self)
        self.title.setStyleSheet("font-size: 10px; font-weight: 600; border: none;")
        header.addWidget(self.title, 1)
        self.close_button = QToolButton(self)
        self.close_button.setText("×")
        self.close_button.setFixedSize(18, 18)
        self.close_button.setToolTip("Close without applying")
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.close_button)
        root.addLayout(header)
        self.swatches = _HexSwatchCanvas(self)
        self.swatches.color_clicked.connect(self._set_color)
        root.addWidget(self.swatches, 0, Qt.AlignmentFlag.AlignHCenter)
        plane_row = QHBoxLayout()
        plane_row.setContentsMargins(0, 0, 0, 0)
        plane_row.setSpacing(4)
        
        self.plane = _ColorPlane(self)
        self.hue_strip = _HueStrip(self)
        plane_row.addWidget(self.plane)
        plane_row.addWidget(self.hue_strip)
        root.addLayout(plane_row)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(3)
        self.color_preview = QFrame(self)
        self.color_preview.setFixedSize(14, 14)
        self.hex_edit = QLineEdit(self._color.name().upper(), self)
        self.hex_edit.setFixedWidth(64)
        self.hex_edit.setMaxLength(7)
        self.hex_edit.setPlaceholderText("#RRGGBB")
        self.hex_edit.setToolTip("HEX color")
        self.more_button = QToolButton(self)
        self.more_button.setText("More…")
        self.more_button.setToolTip("Open full color dialog")
        self.cancel_button = QToolButton(self)
        self.cancel_button.setText("×")
        self.cancel_button.setFixedSize(19, 19)
        self.cancel_button.setToolTip("Cancel color change")
        self.apply_button = QToolButton(self)
        self.apply_button.setText("✓")
        self.apply_button.setFixedSize(19, 19)
        self.apply_button.setToolTip("Apply color")
        footer.addWidget(self.color_preview)
        footer.addWidget(self.hex_edit)
        footer.addWidget(self.more_button)
        footer.addStretch(1)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.apply_button)
        root.addLayout(footer)
        self.plane.changed.connect(self._on_plane_changed)
        self.hue_strip.changed.connect(self._on_hue_changed)
        self.hex_edit.editingFinished.connect(self._on_hex_edited)
        self.more_button.clicked.connect(self._open_advanced)
        self.cancel_button.clicked.connect(self.close)
        self.apply_button.clicked.connect(self._apply)
        h, s, v, _ = self._color.getHsv()
        self.hue_strip.hue = max(0, h)
        self.plane.set_hsv(max(0, h), max(0, s) / 255, max(0, v) / 255)
        self._refresh_controls()
        self.adjustSize()
        apply_popup_theme(self, resolve_theme(parent))

    def apply_theme(self, theme: str | None = None) -> None:
        apply_popup_theme(self, theme)
        self.color_preview.setStyleSheet(f"QFrame {{ background: {self._color.name()}; border: 1px solid palette(mid); border-radius: 2px; }}")

    def _refresh_controls(self):
        self.color_preview.setStyleSheet(f"QFrame {{ background: {self._color.name()}; border: 1px solid palette(mid); border-radius: 2px; }}")
        self.hex_edit.blockSignals(True); self.hex_edit.setText(self._color.name().upper()); self.hex_edit.blockSignals(False); self.hex_edit.setStyleSheet("")

    def _set_color(self, color):
        q = QColor(color)
        if not q.isValid(): return
        self._color = q; h, s, v, _ = q.getHsv(); h = max(0, h); self.hue_strip.hue = h; self.hue_strip.update(); self.plane.set_hsv(h, max(0, s) / 255, max(0, v) / 255); self._refresh_controls()

    def _on_plane_changed(self, saturation: float, value: float):
        self._color = QColor.fromHsv(self.hue_strip.hue, round(saturation * 255), round(value * 255)); self._refresh_controls()

    def _on_hue_changed(self, hue: int):
        _, saturation, value, _ = self._color.getHsv(); self.plane.set_hsv(hue, max(0, saturation) / 255, max(0, value) / 255); self._color = QColor.fromHsv(hue, max(0, saturation), max(0, value)); self._refresh_controls()

    def _on_hex_edited(self):
        candidate = QColor(self.hex_edit.text().strip())
        if candidate.isValid(): self._set_color(candidate)
        else: self.hex_edit.setStyleSheet("QLineEdit { border: 1px solid #c33; }")

    def _open_advanced(self):
        dialog = QColorDialog(self._color, self); dialog.setWindowTitle("More colors"); dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True); dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)
        dialog.setStyleSheet("QColorDialog { font-size: 10px; background: palette(window); color: palette(window-text); } QLabel { margin: 1px; color: palette(window-text); } QPushButton { padding: 2px 7px; min-height: 20px; background: palette(button); color: palette(button-text); border: 1px solid palette(mid); border-radius: 3px; } QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: palette(base); color: palette(text); border: 1px solid palette(mid); }")
        apply_popup_theme(dialog, resolve_theme(self))
        if dialog.exec() == QDialog.DialogCode.Accepted: self._set_color(dialog.selectedColor().name())

    def _apply(self):
        candidate = QColor(self.hex_edit.text().strip())
        if not candidate.isValid(): self.hex_edit.setStyleSheet("QLineEdit { border: 1px solid #c33; }"); self.hex_edit.setFocus(); return
        self._set_color(candidate.name()); self.color_selected.emit(self._color.name()); self.close()
