"""Единая оболочка TRS Apps: запуск модулей, вкладки и два split-пространства."""
from __future__ import annotations
from pathlib import Path
from PyQt6 import QtCore, QtGui, QtWidgets
from roc_mcs.gui.app_settings import load_settings, save_settings
from roc_mcs.gui.components.icon_provider import resource_svg_icon, set_icon_theme
from .app_registry import APP_BY_KEY
from .app_rail import AppLauncherRail
from .workspace import AppPane
from .window_chrome import ChromeSurface, WindowTitleBar
from .global_settings_dialog import TRSGlobalSettingsDialog

_THEME = {
    "dark": {"window": "#11151b", "surface": "#171c24", "text": "#d8e0ec", "muted": "#98a5b6", "border": "#303744", "hover": "#2a313d", "selected": "#27394d", "accent": "#8da9c7", "base": "#151c27", "tab_text": "#9eabba"},
    "light": {"window": "#f5f7fa", "surface": "#e8edf3", "text": "#263445", "muted": "#64748b", "border": "#cbd5e1", "hover": "#dfe7f0", "selected": "#d6e6f5", "accent": "#527fa8", "base": "#ffffff", "tab_text": "#526276"},
}

class TRSAppsWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("trsAppsWindow"); self.setWindowTitle("TRS Apps")
        self.setWindowFlags(QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.FramelessWindowHint)
        app = QtWidgets.QApplication.instance(); self._system_palette = QtGui.QPalette(app.palette()) if app is not None else QtGui.QPalette()
        self._settings = load_settings(); self._configure_app_from_settings(self._settings)
        self._active_pane = None; self._target_pane = None; self._split = False
        self._surface = ChromeSurface(self); self.setCentralWidget(self._surface)
        outer = QtWidgets.QVBoxLayout(self._surface.frame); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        self.title_bar = WindowTitleBar(self, lambda: self.set_split(QtCore.Qt.Orientation.Horizontal), lambda: self.set_split(QtCore.Qt.Orientation.Vertical), self.unsplit, self.open_global_settings, self._surface.frame); outer.addWidget(self.title_bar)
        body = QtWidgets.QWidget(self._surface.frame); body.setObjectName("appBody"); row = QtWidgets.QHBoxLayout(body); row.setContentsMargins(0,0,0,0); row.setSpacing(0); outer.addWidget(body,1)
        self.rail = AppLauncherRail(body); row.addWidget(self.rail)
        self.pane_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, body); self.pane_splitter.setObjectName("workspaceSplitter"); self.pane_splitter.setChildrenCollapsible(False); self.pane_splitter.setHandleWidth(4); row.addWidget(self.pane_splitter,1)
        self.pane_a = AppPane("Выберите приложение на левой панели.\n\nLive Scans — потоковая ROC-карта.\nROI Explorer — анализ HDF5-сканов 2D-детектора.", self.pane_splitter)
        self.pane_b = AppPane("Вторая область просмотра.\nВыберите приложение на вертикальной панели.", self.pane_splitter)
        self.pane_splitter.addWidget(self.pane_a); self.pane_splitter.addWidget(self.pane_b); self.pane_b.hide(); self._active_pane = self.pane_a; self._target_pane = self.pane_a
        self.pane_splitter.setSizes([1200, 0])
        self.status_label = QtWidgets.QLabel("Готово · выберите приложение слева", self._surface.frame); self.status_label.setObjectName("appStatus"); self.status_label.setMinimumHeight(22); outer.addWidget(self.status_label)
        for pane in (self.pane_a, self.pane_b): pane.activated.connect(self._activate_pane); pane.closeRequested.connect(self._close_tab)
        self.rail.appRequested.connect(self.open_app); self._apply_theme(); self._fit_window()
        if app is not None: app.focusChanged.connect(self._on_focus_changed); app.installEventFilter(self)
        self.status_label.setText("Готово · выберите приложение слева")

    def _configure_app_from_settings(self, settings: dict) -> None:
        app = QtWidgets.QApplication.instance(); mode = str(settings.get("theme", "dark")).lower()
        if mode not in ("dark", "light", "system"): mode = "dark"
        palette = QtGui.QPalette(self._system_palette)
        active = mode if mode in ("dark", "light") else ("dark" if palette.color(QtGui.QPalette.ColorRole.Window).lightnessF() < 0.5 else "light")
        size = max(12, min(20, self._safe_icon_size(settings.get("icon_size", 16))))
        self._settings = dict(settings); self._theme_mode = mode; self._active_theme = active; self._icon_size = size
        if app is None: return
        colors = _THEME[active]; app.setProperty("rocMcsThemeMode", mode); app.setProperty("rocMcsTheme", active); app.setProperty("rocMcsIconSize", size)
        roles = {QtGui.QPalette.ColorRole.Window: colors["window"], QtGui.QPalette.ColorRole.WindowText: colors["text"], QtGui.QPalette.ColorRole.Base: colors["base"], QtGui.QPalette.ColorRole.AlternateBase: colors["surface"], QtGui.QPalette.ColorRole.Text: colors["text"], QtGui.QPalette.ColorRole.Button: colors["surface"], QtGui.QPalette.ColorRole.ButtonText: colors["text"], QtGui.QPalette.ColorRole.Highlight: colors["selected"], QtGui.QPalette.ColorRole.HighlightedText: colors["text"], QtGui.QPalette.ColorRole.ToolTipBase: colors["surface"], QtGui.QPalette.ColorRole.ToolTipText: colors["text"], QtGui.QPalette.ColorRole.Mid: colors["border"]}
        for role, value in roles.items(): palette.setColor(role, QtGui.QColor(value))
        app.setPalette(palette); set_icon_theme(app, colors["text"])

    @staticmethod
    def _safe_icon_size(value) -> int:
        try: return int(value)
        except (TypeError, ValueError, OverflowError): return 16

    def _apply_theme(self):
        c = _THEME[self._active_theme]
        self.setWindowIcon(resource_svg_icon("trs_apps.svg", size=32, color=c["text"]))
        self.setStyleSheet(f"""
        QMainWindow#trsAppsWindow, QWidget#chromeSurface {{ background:{c['window']}; color:{c['text']}; }}
        QWidget#chromeFrame {{ background:{c['window']}; border:1px solid {c['border']}; border-radius:4px; }}
        QWidget#trsTitleBar {{ background:{c['surface']}; border-bottom:1px solid {c['border']}; }}
        QLabel#appBrand {{ color:{c['text']}; font-size:10pt; font-weight:600; }}
        QToolButton#chromeToolButton, QToolButton#windowControl, QToolButton#windowControlDanger {{ background:transparent; color:{c['text']}; border:0; border-radius:4px; }}
        QToolButton#chromeToolButton:hover, QToolButton#windowControl:hover {{ background:{c['hover']}; }}
        QToolButton#windowControlDanger:hover {{ background:#8b343b; color:white; border:0; border-radius:4px; }}
        QWidget#appBody {{ background:{c['window']}; color:{c['text']}; }}
        QWidget#appLauncherRail {{ background:{c['surface']}; border-right:1px solid {c['border']}; }}
        QToolButton#appRailButton {{ background:transparent; border:1px solid transparent; border-radius:5px; padding:3px; }}
        QToolButton#appRailButton:hover {{ background:{c['hover']}; }}
        QToolButton#appRailButton:checked {{ background:{c['selected']}; border:1px solid {c['accent']}; }}
        QToolButton#appRailButton[planned="true"] {{ color:{c['muted']}; }}
        QFrame#railSeparator {{ background:{c['border']}; border:0; }}
        QSplitter#workspaceSplitter::handle {{ background:{c['border']}; }}
        QTabWidget#appTabs::pane {{ border:0; background:{c['window']}; top:0; }}
        QTabWidget#appTabs QTabBar {{ background:{c['surface']}; }}
        QTabWidget#appTabs QTabBar::tab {{ background:{c['surface']}; color:{c['tab_text']}; padding:8px 13px; border:0; border-right:1px solid {c['border']}; min-width:85px; }}
        QTabWidget#appTabs QTabBar::tab:selected {{ background:{c['window']}; color:{c['text']}; border-top:2px solid {c['accent']}; padding-top:6px; }}
        QTabWidget#appTabs QTabBar::tab:hover:!selected {{ background:{c['hover']}; color:{c['text']}; }}
        QLabel#workspaceEmpty {{ background:{c['window']}; color:{c['muted']}; font-size:10pt; padding:24px; }}
        QLabel#appStatus {{ background:{c['surface']}; color:{c['muted']}; border-top:1px solid {c['border']}; padding:3px 9px; }}
        """)
        self.title_bar.apply_theme(c["text"], self._icon_size); self.rail.apply_theme(c["text"], self._icon_size)
        for pane, index, page in self._all_pages():
            spec = APP_BY_KEY.get(page.property("trsAppKey"));
            if spec: pane.tabs.setTabIcon(index, resource_svg_icon(spec.icon, size=self._icon_size, color=c["text"]))
        for button in self.findChildren(QtWidgets.QAbstractButton):
            if not button.icon().isNull(): button.setIconSize(QtCore.QSize(self._icon_size, self._icon_size))

    def _apply_global_settings_to_pages(self, settings: dict) -> None:
        colors = _THEME[self._active_theme]
        for _pane, _index, page in self._all_pages():
            set_icon_theme(page, colors["text"]); page.setProperty("rocMcsTheme", self._active_theme)
            method = getattr(page, "apply_global_settings", None)
            if callable(method): method(settings)
            for button in page.findChildren(QtWidgets.QAbstractButton):
                if not button.icon().isNull(): button.setIconSize(QtCore.QSize(self._icon_size, self._icon_size))

    def open_global_settings(self) -> None:
        dialog = TRSGlobalSettingsDialog(load_settings(), self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted: return
        merged = load_settings(); merged.update(dialog.values())
        try: save_settings(merged)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Не удалось сохранить настройки", f"settings.json не изменён.\n\n{exc}"); return
        self._configure_app_from_settings(merged); self._apply_theme(); self._apply_global_settings_to_pages(merged)
        self.status_label.setText("Глобальные настройки TRS Apps применены")
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
        page.setProperty("trsAppKey", key); page.setProperty("rocMcsTheme", self._active_theme)
        set_icon_theme(page, _THEME[self._active_theme]["text"])
        method = getattr(page, "apply_global_settings", None)
        if callable(method): method(self._settings)
        icon = resource_svg_icon(spec.icon, size=self._icon_size, color=_THEME[self._active_theme]["text"])
        index = target.add_page(page, title, icon, select=True); target.tabs.setTabToolTip(index, spec.description)
        for button in page.findChildren(QtWidgets.QAbstractButton):
            if not button.icon().isNull(): button.setIconSize(QtCore.QSize(self._icon_size, self._icon_size))
        self._activate_pane(target); self.status_label.setText(f"Открыто: {title}")
    def _create_app(self, key: str):
        if key == "live_scans":
            from roc_mcs.gui.live_scans.roc_live_dashboard_v4 import LiveRocDashboard
            return LiveRocDashboard(records=None), "Live Scans"
        if key == "roi_explorer":
            from roc_mcs.gui.roi_explorer.explorer_window import RockingCurveExplorerWindow
            page = RockingCurveExplorerWindow()
            title = f"ROI · {page.current_file.name}" if page.current_file else "ROI Explorer"
            return page, title
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
