"""Компактная кнопка монитора с индикатором запаса памяти системы."""
from PyQt6 import QtCore, QtGui, QtWidgets

class ResourceMonitorButton(QtWidgets.QToolButton):
    _COLORS = {"normal": "#32966f", "warning": "#d69b35", "critical": "#d34f55", "unknown": "#8792a2"}
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("chromeToolButton"); self.setFixedSize(28, 28); self.setAutoRaise(True)
        self.setAccessibleName("Системные ресурсы"); self.setToolTip("Монитор системных ресурсов · нет данных")
        self._state = "unknown"

    @QtCore.pyqtSlot(object)
    def set_snapshot(self, snapshot):
        self._state = getattr(snapshot, "state", "unknown")
        labels = {"normal": "Запас системной RAM / commit выше ориентировочного порога", "warning": "Запас системной памяти / commit снижается", "critical": "Критически малый запас системной памяти / commit", "unknown": "Данные о системных ресурсах недоступны"}
        self.setToolTip("Монитор системных ресурсов\n" + labels.get(self._state, labels["unknown"]) + "\nGPU показывается отдельно; индикатор не является гарантией допуска к тяжёлой операции.")
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self); painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = QtGui.QColor(self._COLORS.get(self._state, self._COLORS["unknown"]))
        painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.ColorRole.Window), 1.2)); painter.setBrush(color)
        painter.drawEllipse(QtCore.QPointF(self.width() - 6.5, self.height() - 6.5), 3.1, 3.1); painter.end()
