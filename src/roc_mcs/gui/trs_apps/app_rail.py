"""Лаунчер приложений: здесь перечисляются приложения, а не разделы одного окна."""
from __future__ import annotations
from PyQt6 import QtCore, QtWidgets
from roc_mcs.gui.components.icon_provider import resource_svg_icon
from .app_registry import APP_SPECS

class AppLauncherRail(QtWidgets.QWidget):
    appRequested = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("appLauncherRail"); self.setFixedWidth(48)
        self._buttons = {}; layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(6,8,6,8); layout.setSpacing(5)
        group = QtWidgets.QButtonGroup(self); group.setExclusive(True); self._group=group
        for index, spec in enumerate(APP_SPECS):
            if index == 2:
                line = QtWidgets.QFrame(self); line.setObjectName("railSeparator"); line.setFrameShape(QtWidgets.QFrame.Shape.HLine); line.setFixedHeight(1); layout.addSpacing(5); layout.addWidget(line); layout.addSpacing(5)
            button = QtWidgets.QToolButton(self); button.setObjectName("appRailButton"); button.setFixedSize(36,36); button.setCheckable(True); button.setAutoRaise(True)
            button.setIcon(resource_svg_icon(spec.icon, widget=button, size=19, color="#7c8797" if not spec.available else None)); button.setIconSize(QtCore.QSize(19,19))
            button.setToolTip(spec.title + ("\n"+spec.description if spec.description else "")); button.setAccessibleName(spec.title); button.setProperty("planned", not spec.available)
            if not spec.available: button.setProperty("planned", True)
            button.clicked.connect(lambda checked=False, key=spec.key: self.appRequested.emit(key) if checked else None)
            group.addButton(button); self._buttons[spec.key]=button; layout.addWidget(button)
        layout.addStretch(1)
    def set_active(self, key: str | None):
        for name, button in self._buttons.items(): button.setChecked(name == key)
    def button(self, key: str): return self._buttons.get(key)
