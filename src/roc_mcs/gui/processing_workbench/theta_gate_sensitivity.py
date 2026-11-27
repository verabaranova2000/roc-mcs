"""Compact, exact sensitivity plot for diffraction-branch ``theta_gate``."""
from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QSize, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QPainter, QPainterPath, QPen, QPolygon
from PyQt6.QtWidgets import QSizePolicy, QWidget


class ThetaGateSensitivityPlot(QWidget):
    """Log-x staircase of exact connected-component count versus θ-gate.

    The curve is computed from all merge-causing candidate-edge weights.  A
    highlighted interval, when present, is a range in which the full component
    partition is unchanged.  Dragging the vertical marker edits the same value
    shown by the θ-gate spin box in the parent form.
    """

    gate_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scan: dict[str, Any] | None = None
        self._gate = 60.0
        self._dragging = False
        self._level_gates = np.empty(0, dtype=np.float64)
        self._level_counts = np.empty(0, dtype=np.int64)
        self._curve_path: QPainterPath | None = None
        self._curve_path_size = QSize()
        self.setObjectName("thetaGateSensitivityPlot")
        self.setMinimumSize(300, 94)
        self.setMaximumHeight(124)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Зависимость числа дифракционных ветвей от θ-gate")
        self.setToolTip(
            "Ступенчатая кривая показывает точное число компонент связности при каждом θ-gate. "
            "Выделенная полоса — диапазон, внутри которого разбиение на ветви не меняется. "
            "Перетащите вертикальный указатель, чтобы изменить θ-gate."
        )

    def sizeHint(self) -> QSize:
        return QSize(470, 112)

    @property
    def gate(self) -> float:
        return float(self._gate)

    def set_scan(self, scan: dict[str, Any] | None) -> None:
        self._scan = dict(scan) if isinstance(scan, dict) else None
        levels = list((self._scan or {}).get("levels", ()))
        self._level_gates = np.asarray(
            [float(item.get("gate", np.inf)) for item in levels], dtype=np.float64
        )
        initial = int((self._scan or {}).get(
            "initial_branch_count", (self._scan or {}).get("n_events", 0)
        ))
        self._level_counts = np.asarray(
            [initial] + [int(item.get("branch_count", initial)) for item in levels],
            dtype=np.int64,
        )
        self._curve_path = None
        self.update()

    def set_gate(self, value: float, *, emit: bool = False) -> None:
        try:
            gate = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        if not np.isfinite(gate) or gate <= 0.0:
            return
        changed = not np.isclose(gate, self._gate, rtol=1e-12, atol=1e-12)
        self._gate = gate
        self.update()
        if emit and changed:
            self.gate_changed.emit(gate)

    def _plot_rect(self) -> tuple[float, float, float, float]:
        left = 43.0
        top = 23.0
        right = max(left + 40.0, float(self.width()) - 12.0)
        bottom = max(top + 20.0, float(self.height()) - 36.0)
        return left, top, right, bottom

    def _domain(self) -> tuple[float, float]:
        scan = self._scan or {}
        try:
            x_min = float(scan.get("minimum_gate", 0.001))
            x_max = float(scan.get("maximum_gate", max(x_min * 10.0, 0.01)))
        except (TypeError, ValueError, OverflowError):
            x_min, x_max = 0.001, 0.01
        if not np.isfinite(x_min) or x_min <= 0.0:
            x_min = 0.001
        if not np.isfinite(x_max) or x_max <= x_min:
            x_max = x_min * 10.0
        return x_min, x_max

    def _x_for_gate(self, value: float, left: float, right: float) -> float:
        x_min, x_max = self._domain()
        fraction = (np.log10(max(x_min, float(value))) - np.log10(x_min)) / (
            np.log10(x_max) - np.log10(x_min)
        )
        return left + float(np.clip(fraction, 0.0, 1.0)) * (right - left)

    def _gate_for_x(self, x: float, left: float, right: float) -> float:
        x_min, x_max = self._domain()
        fraction = float(np.clip((float(x) - left) / max(1e-9, right - left), 0.0, 1.0))
        value = float(10.0 ** (np.log10(x_min) + fraction * (np.log10(x_max) - np.log10(x_min))))
        return round(value, 4)

    def _branch_count_at_gate(self, gate: float) -> int | None:
        scan = self._scan
        if not scan:
            return None
        if self._level_counts.size == 0:
            return int(scan.get("initial_branch_count", scan.get("n_events", 0)))
        level_index = int(np.searchsorted(self._level_gates, float(gate), side="right"))
        return int(self._level_counts[min(level_index, self._level_counts.size - 1)])

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._curve_path = None
        super().resizeEvent(event)

    def _build_curve_path(
        self, left: float, top: float, right: float, bottom: float,
        y_low: float, y_high: float, initial: int,
        levels: list[dict[str, Any]],
    ) -> QPainterPath:
        path = QPainterPath()
        x_min, _x_max = self._domain()

        def y_for_count(count: int | float) -> float:
            fraction = (float(count) - y_low) / max(1e-12, y_high - y_low)
            return bottom - float(np.clip(fraction, 0.0, 1.0)) * (bottom - top)

        path.moveTo(self._x_for_gate(x_min, left, right), y_for_count(initial))
        previous_count = initial
        for item in levels:
            x = self._x_for_gate(float(item["gate"]), left, right)
            next_count = int(item["branch_count"])
            path.lineTo(x, y_for_count(previous_count))
            path.lineTo(x, y_for_count(next_count))
            previous_count = next_count
        path.lineTo(right, y_for_count(previous_count))
        return path

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        text = palette.color(palette.ColorRole.Text)
        muted = palette.color(palette.ColorRole.PlaceholderText)
        grid = palette.color(palette.ColorRole.Mid)
        accent = palette.color(palette.ColorRole.Highlight)
        left, top, right, bottom = self._plot_rect()
        scan = self._scan

        painter.setPen(text)
        title_font = QFont(painter.font())
        title_font.setPointSizeF(max(7.0, title_font.pointSizeF() * 0.82))
        title_font.setWeight(QFont.Weight.DemiBold)
        detail_font = QFont(painter.font())
        detail_font.setPointSizeF(max(6.5, detail_font.pointSizeF() * 0.88))
        detail_font.setWeight(QFont.Weight.Normal)
        painter.setFont(detail_font)
        gate_count = self._branch_count_at_gate(self._gate)
        current_text = f"θ={self._gate:.4g}″"
        if gate_count is not None:
            current_text += f"  ·  N={gate_count:,}"
        painter.setFont(title_font)
        title_width = max(
            30,
            int(right - 18 - painter.fontMetrics().horizontalAdvance(current_text) - 12),
        )
        title = painter.fontMetrics().elidedText(
            "ЧИСЛО КОМПОНЕНТ B(θ-gate)", Qt.TextElideMode.ElideRight, title_width
        )
        painter.setPen(QPen(text, 1.0))
        painter.drawText(8, 14, title)
        painter.setFont(detail_font)
        painter.setPen(QPen(muted, 1.0))
        painter.drawText(
            int(max(8.0, right - painter.fontMetrics().horizontalAdvance(current_text))),
            14, current_text,
        )

        if not scan:
            painter.setPen(QPen(muted, 1.0))
            painter.drawText(
                int(left), int((top + bottom) / 2), int(right - left), 16,
                Qt.AlignmentFlag.AlignCenter, "Расчёт скана θ-gate…",
            )
            painter.end()
            return

        x_min, x_max = self._domain()
        initial = int(scan.get("initial_branch_count", scan.get("n_events", 0)))
        levels = list(scan.get("levels", ()))
        y_values = [initial] + [int(item.get("branch_count", initial)) for item in levels]
        y_min = min(y_values, default=0)
        y_max = max(y_values, default=1)
        if y_max <= y_min:
            y_low, y_high = max(0, y_min - 1), y_max + 1
        else:
            pad = max(0.5, (y_max - y_min) * 0.06)
            y_low, y_high = max(0.0, y_min - pad), y_max + pad

        # A stability band marks an interval where no component merger occurs.
        stability = scan.get("stability_window")
        if isinstance(stability, dict) and scan.get("status") == "stable_window_found":
            band_left = self._x_for_gate(float(stability["lower_gate"]), left, right)
            band_right = self._x_for_gate(float(stability["upper_gate"]), left, right)
            band = QColor(accent)
            band.setAlpha(34)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(band))
            painter.drawRect(
                int(min(band_left, band_right)), int(top),
                max(1, int(abs(band_right - band_left))), int(bottom - top),
            )
            boundary = QColor(accent)
            boundary.setAlpha(155)
            painter.setPen(QPen(boundary, 0.8, Qt.PenStyle.DashLine))
            painter.drawLine(QPoint(int(band_left), int(top)), QPoint(int(band_left), int(bottom)))
            painter.drawLine(QPoint(int(band_right), int(top)), QPoint(int(band_right), int(bottom)))

        painter.setPen(QPen(grid, 0.65))
        for y in (top, bottom):
            painter.drawLine(int(left), int(y), int(right), int(y))

        # Cache the exact staircase. Dragging the marker repaints only the
        # selection overlay; a 30k-event scan is not traversed per mouse move.
        if self._curve_path is None or self._curve_path_size != self.size():
            self._curve_path = self._build_curve_path(
                left, top, right, bottom, y_low, y_high, initial, levels
            )
            self._curve_path_size = self.size()
        curve = QColor(accent)
        curve.setAlpha(220)
        painter.setPen(QPen(curve, 1.55, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._curve_path)

        # Sparse logarithmic decade ticks preserve legibility on compact widths.
        painter.setFont(detail_font)
        painter.setPen(QPen(muted, 1.0))
        log_min, log_max = np.log10(x_min), np.log10(x_max)
        first_power = int(np.ceil(log_min))
        last_power = int(np.floor(log_max))
        powers = list(range(first_power, last_power + 1))
        if len(powers) > 6:
            stride = int(np.ceil(len(powers) / 6.0))
            powers = powers[::stride]
        tick_values = [10.0 ** power for power in powers]
        for value in tick_values:
            x = self._x_for_gate(value, left, right)
            painter.drawLine(int(x), int(bottom), int(x), int(bottom + 3))
            painter.drawText(
                int(x - 20), int(bottom + 11), 40, 10,
                Qt.AlignmentFlag.AlignHCenter, f"{value:g}",
            )

        # Selected gate is the only draggable control; it is linked to the form.
        x_gate = self._x_for_gate(self._gate, left, right)
        visible_gate = x_min <= self._gate <= x_max
        marker = QColor(accent)
        painter.setPen(QPen(marker, 1.55, Qt.PenStyle.SolidLine))
        if visible_gate:
            painter.drawLine(int(x_gate), int(top), int(x_gate), int(bottom))
            painter.setBrush(QBrush(marker))
            painter.setPen(QPen(palette.color(palette.ColorRole.Base), 0.8))
            painter.drawPolygon(QPolygon([
                QPoint(int(x_gate), int(top)),
                QPoint(int(x_gate - 4), int(top + 6)),
                QPoint(int(x_gate + 4), int(top + 6)),
            ]))
        else:
            edge_x = int(left if self._gate < x_min else right)
            direction = -1 if self._gate < x_min else 1
            painter.setBrush(QBrush(marker))
            painter.setPen(QPen(palette.color(palette.ColorRole.Base), 0.8))
            painter.drawPolygon(QPolygon([
                QPoint(edge_x, int(top + 1)),
                QPoint(edge_x - direction * 7, int(top + 7)),
                QPoint(edge_x - direction * 7, int(top - 5)),
            ]))

        # Compact axis labels; units are stated once.
        painter.setPen(QPen(muted, 1.0))
        y_span = max(1e-12, y_high - y_low)
        y_for_data_max = bottom - ((float(y_max) - y_low) / y_span) * (bottom - top)
        if y_min == y_max:
            painter.drawText(
                1, int((top + bottom) / 2 - 5), 38, 10,
                Qt.AlignmentFlag.AlignRight, f"{y_max:,}",
            )
        else:
            y_for_data_min = bottom - ((float(y_min) - y_low) / y_span) * (bottom - top)
            painter.drawText(
                1, int(y_for_data_max - 5), 38, 10, Qt.AlignmentFlag.AlignRight,
                f"{y_max:,}",
            )
            painter.drawText(
                1, int(y_for_data_min - 5), 38, 10, Qt.AlignmentFlag.AlignRight,
                f"{y_min:,}",
            )
        painter.drawText(
            int(max(left, (left + right) / 2 - 48)), int(self.height() - 12),
            96, 10, Qt.AlignmentFlag.AlignHCenter, "θ-gate, ″ · лог. шкала",
        )

        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            left, top, right, bottom = self._plot_rect()
            position = event.position()
            if left <= position.x() <= right and top <= position.y() <= bottom and self._scan:
                self._dragging = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.set_gate(self._gate_for_x(position.x(), left, right), emit=True)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            left, _top, right, _bottom = self._plot_rect()
            self.set_gate(self._gate_for_x(event.position().x(), left, right), emit=True)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)
