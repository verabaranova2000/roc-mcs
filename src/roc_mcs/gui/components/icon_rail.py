"""Компактная вертикальная навигация по разделам панели."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets
from roc_mcs.gui.components.icon_provider import resource_svg_icon


class IconRail(QtWidgets.QWidget):
    """Узкая панель иконок; не владеет страницами и логикой эксперимента."""
    sectionSelected = QtCore.pyqtSignal(str)

    def __init__(self, items: list[tuple[str, str, str, bool]], parent=None):
        super().__init__(parent)
        self.setObjectName("navRail")
        self.setFixedWidth(44)
        self._buttons: dict[str, QtWidgets.QToolButton] = {}
        self._group = QtWidgets.QButtonGroup(self); self._group.setExclusive(True)
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(4, 6, 4, 6); layout.setSpacing(5)
        for key, icon_name, label, enabled in items:
            button = QtWidgets.QToolButton(self); button.setObjectName("railButton"); button.setFixedSize(34, 34)
            button.setCheckable(True); button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setIcon(resource_svg_icon(icon_name, widget=button, size=18)); button.setIconSize(QtCore.QSize(18, 18))
            button.setToolTip(label); button.setAccessibleName(label); button.setAccessibleDescription(label)
            button.setEnabled(bool(enabled)); button.setProperty("sectionKey", key)
            button.clicked.connect(lambda checked=False, k=key: self.sectionSelected.emit(k) if checked else None)
            self._group.addButton(button); self._buttons[key] = button; layout.addWidget(button)
        layout.addStretch(1)

    def set_current(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is not None and button.isEnabled(): button.setChecked(True)

    def button(self, key: str) -> QtWidgets.QToolButton | None:
        return self._buttons.get(key)
