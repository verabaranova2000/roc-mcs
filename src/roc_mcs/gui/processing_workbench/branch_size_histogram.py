"""Compact interactive distribution of diffraction-branch event counts."""
from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QSize, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPolygon
from PyQt6.QtWidgets import QWidget, QSizePolicy


class BranchSizeHistogram(QWidget):
    """Log-x histogram with a draggable minimum-event threshold.

    The histogram shows exact branch counts grouped into logarithmic bins;
    it does not smooth or discard any observation. The vertical line marks the
    inclusive minimum event count selected for downstream analysis.
    """

    threshold_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sizes = np.empty(0, dtype=np.int64)
        self._bin_edges = np.empty(0, dtype=np.float64)
        self._bin_counts = np.empty(0, dtype=np.int64)
        self._maximum = 1
        self._threshold = 1
        self._recommended = 1
        self._diagnostics: dict[str, Any] = {}
        self._dragging = False
        self._plot_rect = (48.0, 20.0, 0.0, 0.0)
        self.setObjectName("branchSizeHistogram")
        self.setMinimumSize(320, 104)
        self.setMaximumHeight(132)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Распределение размеров ветвей и порог включения в анализ")
        self.setToolTip(
            "Каждая ветвь учитывается один раз. Горизонтальная шкала размеров логарифмическая. "
            "Перетащите вертикальную линию, чтобы задать минимальное число событий. "
            "Ветви ниже порога сохраняются в каталоге, но не используются downstream."
        )

    def sizeHint(self) -> QSize:
        return QSize(500, 120)

    @property
    def threshold(self) -> int:
        return int(self._threshold)

    @property
    def recommended_threshold(self) -> int:
        return int(self._recommended)

    def set_distribution(
        self, sizes, recommended_min: int = 1, diagnostics: dict[str, Any] | None = None,
    ) -> None:
        raw = np.asarray(sizes).reshape(-1)
        if np.issubdtype(raw.dtype, np.integer):
            if raw.size and (np.any(raw < 1) or np.any(raw > np.iinfo(np.int64).max)):
                raise ValueError("Размеры ветвей должны быть положительными числами в диапазоне int64")
            values = np.ascontiguousarray(raw, dtype=np.int64)
        else:
            numeric = np.asarray(raw, dtype=np.float64)
            if numeric.size and (
                not np.isfinite(numeric).all() or np.any(numeric < 1)
                or np.any(numeric > np.iinfo(np.int64).max)
                or not np.equal(numeric, np.rint(numeric)).all()
            ):
                raise ValueError("Размеры ветвей должны быть конечными положительными целыми числами")
            values = np.ascontiguousarray(numeric, dtype=np.int64)
        self._sizes = values
        self._maximum = max(1, int(values.max()) if values.size else 1)
        self._recommended = max(1, int(recommended_min))
        self._diagnostics = dict(diagnostics or {})
        self._threshold = min(max(1, self._threshold), self._maximum)
        if values.size:
            n_bins = max(8, min(22, int(np.ceil(np.log2(self._maximum + 1))) + 1))
            edges = np.geomspace(1.0, max(1.01, self._maximum * 1.001), n_bins + 1)
            edges[0] = 0.5
            counts, edges = np.histogram(values, bins=edges)
            self._bin_edges = np.ascontiguousarray(edges, dtype=np.float64)
            self._bin_counts = np.ascontiguousarray(counts, dtype=np.int64)
        else:
            self._bin_edges = np.empty(0, dtype=np.float64)
            self._bin_counts = np.empty(0, dtype=np.int64)
        self.update()

    def set_threshold(self, value: int, *, emit: bool = False) -> None:
        value = min(max(1, int(value)), self._maximum)
        changed = value != self._threshold
        if not changed:
            return
        self._threshold = value
        self.update()
        if emit and changed:
            self.threshold_changed.emit(value)

    def _geometry(self):
        left, top, right, bottom = 43.0, 22.0, float(self.width()) - 12.0, float(self.height()) - 25.0
        right = max(left + 30.0, right)
        bottom = max(top + 20.0, bottom)
        return left, top, right, bottom

    def _x_for_value(self, value: float, left: float, right: float, maximum: float) -> float:
        maximum_log = max(0.0, float(np.log10(max(1.0, maximum))))
        if maximum_log <= 1e-12:
            return left
        return left + (float(np.log10(max(1.0, value))) / maximum_log) * (right - left)

    def _value_for_x(self, x: float, left: float, right: float, maximum: float) -> int:
        maximum = max(1, int(maximum))
        if right <= left or maximum <= 1:
            return 1
        fraction = float(np.clip((x - left) / (right - left), 0.0, 1.0))
        return max(1, min(maximum, int(round(10.0 ** (fraction * np.log10(maximum))))))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        text = palette.color(palette.ColorRole.Text)
        muted = palette.color(palette.ColorRole.PlaceholderText)
        grid = palette.color(palette.ColorRole.Mid)
        accent = palette.color(palette.ColorRole.Highlight)
        left, top, right, bottom = self._geometry()
        maximum = self._maximum
        plot_h = bottom - top
        plot_w = right - left

        painter.setPen(text)
        heading = QFont(painter.font())
        heading.setPointSizeF(max(7.0, heading.pointSizeF() * 0.82))
        heading.setBold(True)
        painter.setFont(heading)
        painter.drawText(8, 14, "РАЗМЕР ВЕТВИ · ЧИСЛО СОБЫТИЙ")

        painter.setPen(QPen(grid, 0.7))
        for fraction in (0.0, 0.5, 1.0):
            y = bottom - fraction * plot_h
            painter.drawLine(int(left), int(y), int(right), int(y))
        if self._sizes.size:
            # Geometric bins retain visibility of both singleton noise and very
            # large diffraction sheets without clipping or linear-axis collapse.
            edges = self._bin_edges
            counts = self._bin_counts
            ymax = max(1, int(counts.max()))
            for index, count in enumerate(counts):
                if count <= 0:
                    continue
                x0 = self._x_for_value(max(1, edges[index]), left, right, maximum)
                x1 = self._x_for_value(max(1, edges[index + 1]), left, right, maximum)
                bar_w = max(1.0, x1 - x0 - 1.0)
                bar_h = (float(count) / ymax) * (plot_h - 2.0)
                color = QColor(accent)
                color.setAlpha(145)
                painter.setPen(QPen(accent.darker(112), 0.65))
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(int(x0), int(bottom - bar_h), int(bar_w), int(bar_h), 1.6, 1.6)
            # Sparse log-decade ticks; avoid illegible dense labels.
            tick_values = [1]
            for power in range(1, int(np.floor(np.log10(maximum))) + 1):
                tick_values.append(10 ** power)
            tick_values = [v for v in tick_values if v <= maximum]
            painter.setFont(QFont(painter.font().family(), max(6, painter.font().pointSize() - 1)))
            painter.setPen(muted)
            for value in tick_values:
                x = self._x_for_value(value, left, right, maximum)
                painter.drawLine(int(x), int(bottom), int(x), int(bottom + 3))
                painter.drawText(int(x - 15), int(bottom + 15), 30, 10,
                                 Qt.AlignmentFlag.AlignHCenter, f"{value:g}")
            painter.drawText(3, int(top + 6), 34, 12, Qt.AlignmentFlag.AlignRight, str(ymax))
            painter.drawText(16, int(bottom), 20, 12, Qt.AlignmentFlag.AlignRight, "0")
            x_threshold = self._x_for_value(self._threshold, left, right, maximum)
            painter.setPen(QPen(accent, 1.8, Qt.PenStyle.SolidLine))
            painter.drawLine(int(x_threshold), int(top), int(x_threshold), int(bottom))
            handle = QColor(accent)
            painter.setBrush(QBrush(handle))
            painter.setPen(QPen(palette.color(palette.ColorRole.Base), 1.0))
            triangle = QPolygon([
                QPoint(int(x_threshold), int(top + 1)),
                QPoint(int(x_threshold - 5), int(top + 8)),
                QPoint(int(x_threshold + 5), int(top + 8)),
            ])
            painter.drawPolygon(triangle)
            painter.setPen(text)
            label = f"≥ {self._threshold}"
            painter.drawText(
                int(min(max(left, x_threshold - 24), right - 52)), int(top + 20), 52, 12,
                Qt.AlignmentFlag.AlignHCenter, label,
            )
        else:
            painter.setPen(muted)
            painter.drawText(int(left), int(top + plot_h / 2), int(plot_w), 16,
                             Qt.AlignmentFlag.AlignCenter, "Распределение загружается…")
        painter.setPen(muted)
        painter.setFont(QFont(painter.font().family(), max(6, painter.font().pointSize() - 1)))
        painter.drawText(8, int(self.height() - 2), "Событий в ветви · лог. шкала")
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._sizes.size:
            left, top, right, bottom = self._geometry()
            maximum = max(1, int(self._sizes.max()))
            x_line = self._x_for_value(self._threshold, left, right, maximum)
            if abs(event.position().x() - x_line) <= 14 and top - 4 <= event.position().y() <= bottom + 4:
                self._dragging = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return
            if left <= event.position().x() <= right and top <= event.position().y() <= bottom:
                self._dragging = True
                self.set_threshold(self._value_for_x(event.position().x(), left, right, maximum), emit=True)
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._sizes.size:
            left, _top, right, _bottom = self._geometry()
            maximum = max(1, int(self._sizes.max()))
            self.set_threshold(self._value_for_x(event.position().x(), left, right, maximum), emit=True)
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
