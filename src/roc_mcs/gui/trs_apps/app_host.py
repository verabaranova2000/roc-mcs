"""Единая оболочка TRS Apps: запуск модулей, вкладки и два split-пространства."""
from __future__ import annotations
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from roc_mcs.gui.components.icon_provider import resource_svg_icon, set_icon_theme
from .app_registry import APP_BY_KEY
from .app_rail import AppLauncherRail
from .workspace import AppPane
from .window_chrome import ChromeSurface, WindowTitleBar

class TRSAppsWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("trsAppsWindow"); self.setWindowTitle("TRS Apps"); self.setWindowIcon(resource_svg_icon("trs_apps.svg", size=32, color="#d8e0ec"))
        self.setWindowFlags(QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.FramelessWindowHint)
        app = QtWidgets.QApplication.instance()
        if app is not None: set_icon_theme(app, "#d8e0ec")
        self._active_pane = None; self._target_pane = None; self._split = False
        self._surface = ChromeSurface(self); self.setCentralWidget(self._surface)
        outer = QtWidgets.QVBoxLayout(self._surface.frame); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        self.title_bar = WindowTitleBar(self, lambda: self.set_split(QtCore.Qt.Orientation.Horizontal), lambda: self.set_split(QtCore.Qt.Orientation.Vertical), self.unsplit, self._surface.frame); outer.addWidget(self.title_bar)
        body = QtWidgets.QWidget(self._surface.frame); body.setObjectName("appBody"); row = QtWidgets.QHBoxLayout(body); row.setContentsMargins(0,0,0,0); row.setSpacing(0); outer.addWidget(body,1)
        self.rail = AppLauncherRail(body); row.addWidget(self.rail)
        self.pane_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, body); self.pane_splitter.setObjectName("workspaceSplitter"); self.pane_splitter.setChildrenCollapsible(False); self.pane_splitter.setHandleWidth(4); row.addWidget(self.pane_splitter,1)
        self.pane_a = AppPane("Выберите приложение на левой панели.\n\nLive Scans — потоковая ROC-карта.\nROI Explorer — анализ HDF5-сканов 2D-детектора.", self.pane_splitter)
        self.pane_b = AppPane("Вторая область просмотра.\nВыберите приложение на вертикальной панели.", self.pane_splitter)
        self.pane_splitter.addWidget(self.pane_a); self.pane_splitter.addWidget(self.pane_b); self.pane_b.hide(); self._active_pane = self.pane_a; self._target_pane = self.pane_a
        self.pane_splitter.setSizes([1200, 0])
        self.status_label = QtWidgets.QLabel("Готово · выберите приложение слева", self._surface.frame); self.status_label.setObjectName("appStatus"); self.status_label.setMinimumHeight(22); outer.addWidget(self.status_label)
        for pane in (self.pane_a, self.pane_b):
            pane.activated.connect(self._activate_pane); pane.closeRequested.connect(self._close_tab)
        self.rail.appRequested.connect(self.open_app)
        self._apply_theme(); self._fit_window()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed); app.installEventFilter(self)
        self.status_label.setText("Готово · выберите приложение слева")
    def _apply_theme(self):
        self.setStyleSheet("""
        QMainWindow#trsAppsWindow, QWidget#chromeSurface { background:#11151b; color:#d8e0ec; }
        QWidget#chromeFrame { background:#11151b; border:1px solid #303744; border-radius:4px; }
        QWidget#trsTitleBar { background:#171c24; border-bottom:1px solid #2b323e; }
        QLabel#appBrand { color:#e1e7f0; font-size:10pt; font-weight:600; }
        QToolButton#chromeToolButton, QToolButton#windowControl, QToolButton#windowControlDanger { background:transparent; color:#cbd5e1; border:0; border-radius:4px; }
        QToolButton#chromeToolButton:hover, QToolButton#windowControl:hover { background:#2a313d; }
        QToolButton#windowControlDanger:hover { background:#8b343b; color:white; border:0; border-radius:4px; }
        QWidget#appBody { background:#11151b; color:#d8e0ec; }
        QWidget#appLauncherRail { background:#171c24; border-right:1px solid #2b323e; }
        QToolButton#appRailButton { background:transparent; border:1px solid transparent; border-radius:5px; padding:3px; }
        QToolButton#appRailButton:hover { background:#252d39; }
        QToolButton#appRailButton:checked { background:#27394d; border:1px solid #3b536d; }
        QToolButton#appRailButton[planned="true"] { color:#7c8797; }
        QFrame#railSeparator { background:#343c48; border:0; }
        QSplitter#workspaceSplitter::handle { background:#2b323e; }
        QTabWidget#appTabs::pane { border:0; background:#11151b; top:0; }
        QTabWidget#appTabs QTabBar { background:#171c24; }
        QTabWidget#appTabs QTabBar::tab { background:#171c24; color:#9eabba; padding:8px 13px; border:0; border-right:1px solid #2b323e; min-width:85px; }
        QTabWidget#appTabs QTabBar::tab:selected { background:#11151b; color:#e5ebf4; border-top:2px solid #8da9c7; padding-top:6px; }
        QTabWidget#appTabs QTabBar::tab:hover:!selected { background:#202733; color:#d8e0ec; }
        QLabel#workspaceEmpty { background:#11151b; color:#778395; font-size:10pt; padding:24px; }
        QLabel#appStatus { background:#171c24; color:#98a5b6; border-top:1px solid #2b323e; padding:3px 9px; }
        """)
    def _fit_window(self):
        screen = QtWidgets.QApplication.primaryScreen(); area = screen.availableGeometry() if screen else QtCore.QRect(0,0,1400,900)
        width, height = min(1440, max(800, area.width()-50)), min(920, max(600, area.height()-60))
        self.resize(width,height); self.setMinimumSize(min(760,width), min(520,height))
        if screen: self.move(area.center()-self.rect().center())
    def _activate_pane(self, pane):
        if pane in (self.pane_a, self.pane_b): self._active_pane = pane; self._target_pane = pane; self._sync_active_rail()
    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            current = watched
            while current is not None:
                for pane in (self.pane_a, self.pane_b):
                    if current in pane.pages(): self._activate_pane(pane); break
                else:
                    current = current.parentWidget() if hasattr(current, "parentWidget") else None
                    continue
                break
        return super().eventFilter(watched, event)
    def _on_focus_changed(self, _old, now):
        current = now
        while current is not None:
            for pane in (self.pane_a, self.pane_b):
                if current in pane.pages(): self._activate_pane(pane); return
            current = current.parentWidget() if hasattr(current,"parentWidget") else None
    def _sync_active_rail(self):
        page = self._active_pane.currentWidget() if self._active_pane else None
        self.rail.set_active(page.property("trsAppKey") if page else None)
    def _all_pages(self):
        for pane in (self.pane_a, self.pane_b):
            for index in range(pane.count()): yield pane, index, pane.tabs.widget(index)
    def open_app(self, key: str):
        spec = APP_BY_KEY.get(key)
        if spec is None: return
        if not spec.available:
            QtWidgets.QMessageBox.information(self, spec.title, f"Раздел «{spec.title}» запланирован, но пока не подключён.\n\n{spec.description}")
            self._sync_active_rail(); return
        if spec.single_instance:
            for pane, index, page in self._all_pages():
                if page.property("trsAppKey") == key:
                    pane.tabs.setCurrentIndex(index); self._activate_pane(pane); return
        target = self._target_pane if self._split and self._target_pane is self.pane_b and self.pane_b.count() == 0 else self._active_pane
        page, title = self._create_app(key)
        if page is None: self._sync_active_rail(); return
        page.setProperty("trsAppKey", key)
        icon = resource_svg_icon(spec.icon, size=16, color="#d8e0ec")
        index = target.add_page(page, title, icon, select=True); target.tabs.setTabToolTip(index, spec.description)
        self._activate_pane(target); self.status_label.setText(f"Открыто: {title}")
    def _create_app(self, key: str):
        if key == "live_scans":
            from roc_mcs.gui.live_scans.roc_live_dashboard_v4 import LiveRocDashboard
            return LiveRocDashboard(records=None), "Live Scans"
        if key == "roi_explorer":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть HDF5 для ROI Explorer", str(Path.home()), "HDF5 files (*.h5 *.hdf5 *.hdf);;All files (*)")
            if not path: return None, ""
            from roc_mcs.gui.roi_explorer.explorer_window import RockingCurveExplorerWindow
            page = RockingCurveExplorerWindow(Path(path)); return page, f"ROI · {Path(path).name}"
        return None, ""
    def set_split(self, orientation: QtCore.Qt.Orientation):
        if not self._split:
            self._split = True; self.pane_b.show(); self.pane_splitter.setOrientation(orientation)
            self._target_pane = self.pane_b if self.pane_b.count() == 0 else self._active_pane
        else:
            self.pane_splitter.setOrientation(orientation)
        span = self.pane_splitter.width() if orientation == QtCore.Qt.Orientation.Horizontal else self.pane_splitter.height()
        self.pane_splitter.setSizes([max(1,span//2), max(1,span//2)])
        self.status_label.setText("Рабочая область разделена · откройте приложение во второй панели" if self.pane_b.count() == 0 else "Режим разделённого просмотра")
    def unsplit(self):
        if not self._split: return
        while self.pane_b.count():
            item = self.pane_b.take_page(0)
            if item:
                page,title,icon=item; index=self.pane_a.add_page(page,title,icon,select=False); self.pane_a.tabs.setTabToolTip(index, page.windowTitle() or title)
        self.pane_b.hide(); self._split=False; self.pane_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal); self.pane_splitter.setSizes([1,0]); self._activate_pane(self.pane_a); self.status_label.setText("Разделение закрыто")
    def _page_is_busy(self, page):
        worker = getattr(page, "_export_worker", None)
        if worker is not None and worker.isRunning(): return True, "Экспорт ещё выполняется. Дождитесь его завершения."
        dialog = getattr(page, "_calibration_dialog", None); worker = getattr(getattr(dialog,"widget",None),"worker",None) if dialog else None
        if worker is not None and worker.isRunning(): return True, "Калибровка ещё выполняется. Дождитесь её завершения."
        return False, ""
    def _close_tab(self, pane, index):
        page = pane.tabs.widget(index)
        if page is None: return
        busy, message = self._page_is_busy(page)
        if busy: QtWidgets.QMessageBox.information(self,"Нельзя закрыть вкладку",message); return
        if not page.close(): return
        pane.remove_page(index)
        if pane.count() == 0 and pane is self.pane_b: self.unsplit()
        self._sync_active_rail()
    def closeEvent(self, event):
        pages = [page for _pane,_index,page in self._all_pages()]
        for page in pages:
            busy,message=self._page_is_busy(page)
            if busy: QtWidgets.QMessageBox.information(self,"Нельзя закрыть TRS Apps",message); event.ignore(); return
        for page in pages:
            if not page.close(): event.ignore(); return
        app=QtWidgets.QApplication.instance()
        if app is not None:
            try: app.focusChanged.disconnect(self._on_focus_changed); app.removeEventFilter(self)
            except (TypeError, RuntimeError): pass
        event.accept()
