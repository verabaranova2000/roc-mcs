"""Compact live explanation of the theta gate used by branch linking."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QSize, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPalette
from PyQt6.QtWidgets import QWidget, QSizePolicy

from roc_mcs.peaks.branches import match_neighbor_peaks


class BranchDetectionPreview(QWidget):
    """Draw the exact one-to-one theta matcher on a deterministic two-pixel example.

    The miniature is explanatory rather than a preview of the current scan:
    it uses the same production ``match_neighbor_peaks`` function, so changing
    ``theta_gate`` immediately changes accepted links by the actual algorithm.
    """

    _PEAKS_A = (24.0, 72.0, 128.0, 181.0)
    _PEAKS_B = (31.0, 101.0, 139.0, 224.0)
    _THETA_MAX = 250.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theta_gate = 60.0
        self._pairs_all: list[tuple[int, int, float]] = []
        self._pairs_accepted: list[tuple[int, int, float]] = []
        self._unmatched_a: set[int] = set()
        self._unmatched_b: set[int] = set()
        self._recalculate()
        self.setObjectName("branchDetectionPreview")
        self.setMinimumSize(270, 112)
        self.setMaximumHeight(142)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Иллюстрация действия theta gate при поиске дифракционных ветвей")
        self.setToolTip(
            "Схематический пример для двух соседних пикселей. Каждая точка — отдельный пик. "
            "Алгоритм сначала строит one-to-one назначение с минимальной суммой |Δθ|, "
            "затем принимает связь только при |Δθ| ≤ theta_gate. Это не данные текущего скана."
        )

    def sizeHint(self) -> QSize:
        return QSize(440, 126)

    @property
    def theta_gate(self) -> float:
        return float(self._theta_gate)

    def set_theta_gate(self, value: float) -> None:
        gate = float(value)
        if not np.isfinite(gate) or gate <= 0.0:
            return
        if gate == self._theta_gate:
            return
        self._theta_gate = gate
        self._recalculate()
        self.update()

    def _recalculate(self) -> None:
        peaks_a = [
            {"peak_id": index, "theta": theta}
            for index, theta in enumerate(self._PEAKS_A)
        ]
        peaks_b = [
            {"peak_id": index + 10, "theta": theta}
            for index, theta in enumerate(self._PEAKS_B)
        ]
        # Infinite gate exposes the exact Hungarian one-to-one candidate
        # assignment; the finite-gate call is the production acceptance rule.
        self._pairs_all, _, _ = match_neighbor_peaks(
            peaks_a, peaks_b, theta_gate=float("inf")
        )
        self._pairs_accepted, unmatched_a, unmatched_b = match_neighbor_peaks(
            peaks_a, peaks_b, theta_gate=self._theta_gate
        )
        self._unmatched_a = {int(item["peak_id"]) for item in unmatched_a}
        self._unmatched_b = {int(item["peak_id"]) for item in unmatched_b}

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Base)
        surface = palette.color(QPalette.ColorRole.AlternateBase)
        text = palette.color(QPalette.ColorRole.Text)
        muted = palette.color(QPalette.ColorRole.Mid)
        accepted = (
            QColor("#2F9B78")
            if palette.color(QPalette.ColorRole.Window).value() >= 128
            else QColor("#66CDAA")
        )
        rejected = QColor("#B47D55")

        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1.0))
        painter.setBrush(QBrush(surface))
        painter.drawRoundedRect(rect, 5.0, 5.0)

        left = rect.left() + 13.0
        right = rect.right() - 13.0
        top = rect.top() + 34.0
        bottom = rect.bottom() - 21.0
        x_a = left + (right - left) * 0.30
        x_b = left + (right - left) * 0.70
        theta_y = lambda value: bottom - (float(value) / self._THETA_MAX) * (bottom - top)

        painter.setPen(QPen(text, 1.0))
        painter.setFont(QFont(self.font().family(), max(8, self.font().pointSize() - 1), QFont.Weight.DemiBold))
        painter.drawText(int(left), int(rect.top() + 17), "A · пиксель (x, y)")
        painter.drawText(int(x_b - 44), int(rect.top() + 17), "B · сосед")
        painter.setPen(QPen(muted, 1.0))
        painter.drawText(int(rect.right() - 142), int(rect.top() + 17), f"|Δθ| ≤ {self._theta_gate:.3g}″")

        # Theta reference spine and sparse labels keep the scientific axis clear.
        axis_x = rect.left() + 7.0
        painter.setPen(QPen(muted, 0.8, Qt.PenStyle.SolidLine))
        painter.drawLine(QPointF(axis_x, top), QPointF(axis_x, bottom))
        painter.setFont(QFont(self.font().family(), max(7, self.font().pointSize() - 2)))
        for value in (0, 125, 250):
            y = theta_y(value)
            painter.drawLine(QPointF(axis_x - 2, y), QPointF(axis_x + 2, y))
            painter.drawText(QPointF(axis_x + 4, y + 3), str(value))

        accepted_keys = {(int(a), int(b)) for a, b, _delta in self._pairs_accepted}
        # Draw assignment candidates behind event markers. Rejected candidates
        # are deliberately dashed rather than disappearing: a smaller gate
        # changes graph connectivity, not the underlying measured peak events.
        for peak_a, peak_b, delta in self._pairs_all:
            is_accepted = (int(peak_a), int(peak_b)) in accepted_keys
            color = accepted if is_accepted else rejected
            pen = QPen(color, 2.0 if is_accepted else 1.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if not is_accepted:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            y_a = theta_y(self._PEAKS_A[int(peak_a)])
            y_b = theta_y(self._PEAKS_B[int(peak_b) - 10])
            painter.drawLine(QPointF(x_a, y_a), QPointF(x_b, y_b))
            painter.setPen(QPen(color, 1.0))
            painter.setFont(QFont(self.font().family(), max(7, self.font().pointSize() - 2)))
            label_x = (x_a + x_b) / 2.0 + 3.0
            label_y = (y_a + y_b) / 2.0 - 3.0
            painter.drawText(QPointF(label_x, label_y), f"{delta:.0f}")

        for index, theta in enumerate(self._PEAKS_A):
            y = theta_y(theta)
            is_linked = index not in self._unmatched_a
            painter.setPen(QPen(accepted if is_linked else muted, 1.5))
            painter.setBrush(QBrush(accepted if is_linked else background))
            painter.drawEllipse(QPointF(x_a, y), 4.0, 4.0)
            painter.setPen(QPen(text, 1.0))
            painter.setFont(QFont(self.font().family(), max(7, self.font().pointSize() - 2)))
            painter.drawText(QPointF(x_a - 17, y - 6), f"{theta:g}")
        for index, theta in enumerate(self._PEAKS_B):
            peak_id = index + 10
            y = theta_y(theta)
            is_linked = peak_id not in self._unmatched_b
            painter.setPen(QPen(accepted if is_linked else muted, 1.5))
            painter.setBrush(QBrush(accepted if is_linked else background))
            painter.drawEllipse(QPointF(x_b, y), 4.0, 4.0)
            painter.setPen(QPen(text, 1.0))
            painter.setFont(QFont(self.font().family(), max(7, self.font().pointSize() - 2)))
            painter.drawText(QPointF(x_b + 7, y + 3), f"{theta:g}")

        painter.setPen(QPen(muted, 1.0))
        painter.setFont(QFont(self.font().family(), max(7, self.font().pointSize() - 2)))
        accepted_count = len(self._pairs_accepted)
        candidate_count = len(self._pairs_all)
        painter.drawText(
            int(rect.left() + 13), int(rect.bottom() - 6),
            f"Принято связей: {accepted_count}/{candidate_count}   ·   сплошная — принято, пунктир — Δθ больше gate",
        )
        painter.end()
