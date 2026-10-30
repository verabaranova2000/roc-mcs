"""Компактная тёмная рамка окна с системным перетаскиванием/изменением размера."""
from __future__ import annotations
from PyQt6 import QtCore, QtWidgets
from roc_mcs.gui.components.icon_provider import resource_svg_icon

class ResizeHandle(QtWidgets.QWidget):
    def __init__(self, edges: QtCore.Qt.Edges, cursor: QtCore.Qt.CursorShape, parent=None):
        super().__init__(parent); self.edges = edges; self._start_pos = None; self._start_geo = None
        self.setCursor(cursor); self.setMouseTracking(True); self.setStyleSheet("background: transparent;")
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None and hasattr(handle, "startSystemResize") and handle.startSystemResize(self.edges):
                event.accept(); return
            self._start_pos = event.globalPosition().toPoint(); self._start_geo = self.window().geometry(); event.accept(); return
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self._start_pos is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            window = self.window(); delta = event.globalPosition().toPoint() - self._start_pos; g = self._start_geo
            x, y, w, h = g.x(), g.y(), g.width(), g.height(); edges = self.edges
            if edges & QtCore.Qt.Edge.LeftEdge: x, w = x + delta.x(), w - delta.x()
            if edges & QtCore.Qt.Edge.RightEdge: w += delta.x()
            if edges & QtCore.Qt.Edge.TopEdge: y, h = y + delta.y(), h - delta.y()
            if edges & QtCore.Qt.Edge.BottomEdge: h += delta.y()
            min_w, min_h = window.minimumWidth(), window.minimumHeight()
            if w < min_w:
                if edges & QtCore.Qt.Edge.LeftEdge: x -= min_w - w
                w = min_w
            if h < min_h:
                if edges & QtCore.Qt.Edge.TopEdge: y -= min_h - h
                h = min_h
            window.setGeometry(x, y, w, h); event.accept(); return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event): self._start_pos = None; self._start_geo = None; super().mouseReleaseEvent(event)

class ChromeSurface(QtWidgets.QWidget):
    """Оставляет тонкий край для системного resize без зависимости от сторонних пакетов."""
    EDGE = 6
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("chromeSurface")
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(self.EDGE, self.EDGE, self.EDGE, self.EDGE); layout.setSpacing(0)
        self.frame = QtWidgets.QWidget(self); self.frame.setObjectName("chromeFrame"); layout.addWidget(self.frame, 1)
        E = QtCore.Qt.Edge; C = QtCore.Qt.CursorShape
        self.handles = [
            ResizeHandle(E.LeftEdge, C.SizeHorCursor, self), ResizeHandle(E.RightEdge, C.SizeHorCursor, self),
            ResizeHandle(E.TopEdge, C.SizeVerCursor, self), ResizeHandle(E.BottomEdge, C.SizeVerCursor, self),
            ResizeHandle(E.LeftEdge | E.TopEdge, C.SizeFDiagCursor, self), ResizeHandle(E.RightEdge | E.BottomEdge, C.SizeFDiagCursor, self),
            ResizeHandle(E.RightEdge | E.TopEdge, C.SizeBDiagCursor, self), ResizeHandle(E.LeftEdge | E.BottomEdge, C.SizeBDiagCursor, self),
        ]
        self._layout_handles()
    def _layout_handles(self):
        w, h, e = self.width(), self.height(), self.EDGE
        geoms = [(0,e,e,max(0,h-2*e)), (w-e,e,e,max(0,h-2*e)), (e,0,max(0,w-2*e),e), (e,h-e,max(0,w-2*e),e),
                 (0,0,e,e),(w-e,h-e,e,e),(w-e,0,e,e),(0,h-e,e,e)]
        for handle, rect in zip(self.handles, geoms): handle.setGeometry(*rect); handle.setVisible(not self.window().isMaximized()); handle.raise_()
    def resizeEvent(self, event): self._layout_handles(); super().resizeEvent(event)
    def showEvent(self, event): self._layout_handles(); super().showEvent(event)

class WindowTitleBar(QtWidgets.QWidget):
    def __init__(self, window, split_side, split_stack, unsplit, open_settings, parent=None):
        super().__init__(parent); self._window = window; self._drag_pos = None
        self.setObjectName("trsTitleBar"); self.setFixedHeight(38)
        row = QtWidgets.QHBoxLayout(self); row.setContentsMargins(9, 0, 6, 0); row.setSpacing(5)
        brand = QtWidgets.QLabel(); brand.setPixmap(resource_svg_icon("trs_apps.svg", size=20, color="#dbe3ee").pixmap(20,20)); brand.setFixedSize(22,26); row.addWidget(brand)
        title = QtWidgets.QLabel("TRS Apps"); title.setObjectName("appBrand"); row.addWidget(title); row.addSpacing(10)
        self.btn_split_side = self._tool("Разделить рабочую область слева направо", "app_split_horizontal.svg", split_side); row.addWidget(self.btn_split_side)
        self.btn_split_stack = self._tool("Разделить рабочую область сверху вниз", "app_split_vertical.svg", split_stack); row.addWidget(self.btn_split_stack)
        self.btn_unsplit = self._tool("Убрать разделение", "app_unsplit.svg", unsplit); row.addWidget(self.btn_unsplit)
        self.btn_settings = self._tool("Глобальные настройки TRS Apps", "setting_icon.svg", open_settings); row.addWidget(self.btn_settings)
        row.addStretch(1)
        self._tool_icons = [(self.btn_split_side, "app_split_horizontal.svg"), (self.btn_split_stack, "app_split_vertical.svg"), (self.btn_unsplit, "app_unsplit.svg"), (self.btn_settings, "setting_icon.svg")]
        self.btn_min = self._control("—", "Свернуть окно", window.showMinimized); row.addWidget(self.btn_min)
        self.btn_max = self._control("□", "Развернуть окно", self._toggle_maximize); row.addWidget(self.btn_max)
        self.btn_close = self._control("×", "Закрыть приложение", window.close, danger=True); row.addWidget(self.btn_close)
    def _tool(self, tip, icon, callback):
        b = QtWidgets.QToolButton(self); b.setObjectName("chromeToolButton"); b.setFixedSize(28,28); b.setIcon(resource_svg_icon(icon, widget=b, size=16)); b.setIconSize(QtCore.QSize(16,16)); b.setToolTip(tip); b.setAccessibleName(tip); b.clicked.connect(callback); return b
    def apply_theme(self, foreground: str, icon_size: int = 16):
        size = max(12, min(20, int(icon_size)))
        for button, name in self._tool_icons:
            button.setIcon(resource_svg_icon(name, widget=button, size=size, color=foreground)); button.setIconSize(QtCore.QSize(size, size))
    def _control(self, text, tip, callback, danger=False):
        b = QtWidgets.QToolButton(self); b.setObjectName("windowControlDanger" if danger else "windowControl"); b.setText(text); b.setFixedSize(34,28); b.setToolTip(tip); b.clicked.connect(callback); return b
    def _toggle_maximize(self):
        if self._window.isMaximized(): self._window.showNormal()
        else: self._window.showMaximized()
        self.btn_max.setText("❐" if self._window.isMaximized() else "□")
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            handle = self._window.windowHandle()
            if handle is not None and hasattr(handle, "startSystemMove") and handle.startSystemMove(): event.accept(); return
            self._drag_pos = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft(); event.accept(); return
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton and not self._window.isMaximized(): self._window.move(event.globalPosition().toPoint() - self._drag_pos); event.accept(); return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event): self._drag_pos = None; super().mouseReleaseEvent(event)
    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton: self._toggle_maximize(); event.accept(); return
        super().mouseDoubleClickEvent(event)
