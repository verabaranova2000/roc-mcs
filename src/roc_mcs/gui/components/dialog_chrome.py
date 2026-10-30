"""Компактная тематическая шапка модальных окон ROI Explorer."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


class DialogTitleBar(QWidget):
    def __init__(self, dialog, title: str):
        super().__init__(dialog); self.setObjectName("dialogTitleBar"); self._dialog = dialog; self._drag_offset = None
        self.setFixedHeight(32); self.setMouseTracking(True); self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(self); row.setContentsMargins(10, 0, 4, 0); row.setSpacing(6)
        label = QLabel(title, self); label.setObjectName("dialogTitleLabel"); label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True); row.addWidget(label, 1)
        close = QToolButton(self); close.setObjectName("dialogTitleClose"); close.setText("×"); close.setToolTip("Закрыть окно"); close.setAccessibleName("Закрыть окно"); close.setFixedSize(27, 26); close.clicked.connect(dialog.reject); row.addWidget(close)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self._drag_offset = event.globalPosition().toPoint() - self._dialog.frameGeometry().topLeft(); event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton: self._dialog.move(event.globalPosition().toPoint() - self._drag_offset); event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None; super().mouseReleaseEvent(event)


def install_dialog_titlebar(dialog, layout: QVBoxLayout, title: str) -> DialogTitleBar:
    dialog.setWindowTitle(title); dialog.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    bar = DialogTitleBar(dialog, title); layout.addWidget(bar)
    dialog.setStyleSheet(dialog.styleSheet() + "\nQDialog { border:1px solid palette(mid); } QWidget#dialogTitleBar { background:palette(window); color:palette(window-text); border-bottom:1px solid palette(mid); } QLabel#dialogTitleLabel { background:transparent; color:palette(window-text); font-weight:600; } QToolButton#dialogTitleClose { background:transparent; color:palette(window-text); border:1px solid transparent; border-radius:3px; font-size:18px; padding:0; } QToolButton#dialogTitleClose:hover { background:palette(midlight); border-color:palette(mid); }")
    return bar
