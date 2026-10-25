"""Компактная секция с раскрываемым содержимым для Qt-интерфейсов."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QToolButton, QVBoxLayout, QWidget, QSizePolicy


class CollapsibleSection(QWidget):
    """Сворачиваемый блок без анимации: предсказуемая геометрия и минимум побочных эффектов."""
    expandedChanged = pyqtSignal(bool)

    def __init__(self, title: str, *, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("collapsibleSection")
        self._expanded = bool(expanded)
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(2)
        self.header = QToolButton(self); self.header.setObjectName("collapsibleSectionHeader")
        self.header.setText(title); self.header.setCheckable(True); self.header.setChecked(bool(expanded))
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.header.setAccessibleName(title); self.header.setToolTip("Свернуть/развернуть раздел")
        self.body = QWidget(self); self.body.setObjectName("collapsibleSectionBody")
        self.content_layout = QVBoxLayout(self.body); self.content_layout.setContentsMargins(5, 3, 3, 5); self.content_layout.setSpacing(5)
        root.addWidget(self.header); root.addWidget(self.body)
        self.body.setVisible(bool(expanded)); self.header.toggled.connect(self.setExpanded)

    def setExpanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        changed = self._expanded != expanded
        self._expanded = expanded
        self.header.blockSignals(True); self.header.setChecked(expanded); self.header.blockSignals(False)
        self.header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)
        if changed:
            self.expandedChanged.emit(expanded)

    def isExpanded(self) -> bool:
        return self._expanded
