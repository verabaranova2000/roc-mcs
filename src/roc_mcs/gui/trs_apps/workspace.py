"""Вкладочная панель одного рабочего пространства."""
from __future__ import annotations
from PyQt6 import QtCore, QtWidgets

class _EmptyLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton: self.clicked.emit()
        super().mousePressEvent(event)

class AppPane(QtWidgets.QWidget):
    activated = QtCore.pyqtSignal(object)
    closeRequested = QtCore.pyqtSignal(object, int)
    def __init__(self, empty_message: str, parent=None):
        super().__init__(parent); self.setObjectName("appPane")
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        self.stack = QtWidgets.QStackedWidget(self); layout.addWidget(self.stack,1)
        self.empty = _EmptyLabel(empty_message); self.empty.setObjectName("workspaceEmpty"); self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter); self.empty.setWordWrap(True); self.empty.setCursor(QtCore.Qt.CursorShape.PointingHandCursor); self.empty.clicked.connect(lambda: self.activated.emit(self))
        self.tabs = QtWidgets.QTabWidget(self); self.tabs.setObjectName("appTabs"); self.tabs.setDocumentMode(True); self.tabs.setMovable(True); self.tabs.setTabsClosable(True); self.tabs.setUsesScrollButtons(True)
        self.tabs.tabCloseRequested.connect(lambda index: self.closeRequested.emit(self,index))
        self.tabs.tabBar().tabBarClicked.connect(lambda _index: self.activated.emit(self))
        self.tabs.currentChanged.connect(lambda _index: self.activated.emit(self))
        self.stack.addWidget(self.empty); self.stack.addWidget(self.tabs); self.stack.setCurrentWidget(self.empty)
    def count(self) -> int: return self.tabs.count()
    def currentWidget(self): return self.tabs.currentWidget()
    def indexOf(self, widget) -> int: return self.tabs.indexOf(widget)
    def pages(self): return [self.tabs.widget(i) for i in range(self.tabs.count())]
    def add_page(self, page, title: str, icon, select=True):
        page.setParent(self.tabs); page.setWindowFlags(QtCore.Qt.WindowType.Widget)
        if isinstance(page, QtWidgets.QMainWindow): page.setMinimumSize(480, 320)
        index = self.tabs.addTab(page, icon, title); self.stack.setCurrentWidget(self.tabs); self._sync_empty()
        if select: self.tabs.setCurrentIndex(index)
        page.show(); return index
    def take_page(self, index: int):
        page = self.tabs.widget(index)
        if page is None: return None
        title, icon = self.tabs.tabText(index), self.tabs.tabIcon(index); self.tabs.removeTab(index); page.setParent(None); self._sync_empty()
        return page, title, icon
    def remove_page(self, index: int):
        item = self.take_page(index)
        if item is None: return
        page, _title, _icon = item; page.deleteLater()
    def _sync_empty(self): self.stack.setCurrentWidget(self.tabs if self.tabs.count() else self.empty)
