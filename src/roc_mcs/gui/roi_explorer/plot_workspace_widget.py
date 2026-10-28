"""Рабочая область с двумя изменяемыми окнами графиков без перекрытия."""
from __future__ import annotations
from PyQt6.QtCore import QEvent, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QSplitter, QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QMdiArea, QMdiSubWindow, QSizePolicy
from roc_mcs.gui.roi_explorer.roi_manager_widget import RoiManagerWidget
from roc_mcs.gui.roi_explorer.map_plot_widget import MapPlotWidget
from roc_mcs.gui.roi_explorer.curve_plot_widget import CurvePlotWidget
from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings, save_plot_display_settings
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, theme_colors, resolve_theme

class _PlotSubWindow(QMdiSubWindow):
    """Оконная рамка графического экрана; закрытие скрывает, не уничтожая состояние."""
    def __init__(self, area, key, title):
        super().__init__(); self._area = area; self.plot_key = key; self.setWindowTitle(title)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandMove, True)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandResize, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

    def closeEvent(self, event):
        event.accept(); self.hide(); self._area.panel_visibility_changed.emit(self.plot_key, False)

class _PlotMdiArea(QMdiArea):
    """Две MDI-панели: резиновый контур при перетаскивании, запрет перекрытия после отпускания."""
    panel_visibility_changed = pyqtSignal(str, bool)
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("plotMdiArea"); self.setViewMode(QMdiArea.ViewMode.SubWindowView)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setActivationOrder(QMdiArea.WindowOrder.ActivationHistoryOrder); self.setOption(QMdiArea.AreaOption.DontMaximizeSubWindowOnActivation, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.setMinimumSize(350, 440)
        self._guard = False; self._layout_mode = None; self._last_viewport = QSize(); self._last_valid = {}; self._keys = {}
        self.panel_visibility_changed.connect(self._save_geometry_debounced); self.panel_visibility_changed.connect(self._visibility_changed)

    def add_plot(self, key: str, title: str, widget: QWidget):
        win = _PlotSubWindow(self, key, title); win.setWindowFlags(Qt.WindowType.SubWindow | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowCloseButtonHint)
        win.setWidget(widget); minimum = widget.minimumSize(); win.setMinimumSize(QSize(minimum.width()+2, minimum.height()+24).expandedTo(QSize(212, 194)))
        self.addSubWindow(win); win.installEventFilter(self); self._keys[key] = win
        win.show(); self._last_valid[win] = QRect(0, 0, 320, 300)
        return win

    def show_panel(self, key: str, visible: bool):
        win = self._keys.get(key)
        if win is None: return
        if visible:
            win.show(); win.raise_(); self.setActiveSubWindow(win)
            QTimer.singleShot(0, lambda w=win: self._enforce_geometry(w))
        else: win.hide()
        self.panel_visibility_changed.emit(key, bool(visible))

    def _bounds(self): return self.viewport().rect().adjusted(2, 2, -2, -2)

    def _layout_initial(self):
        if not self._keys: return
        saved = load_plot_display_settings().get("workspace", {})
        bounds = self._bounds(); mode = saved.get("orientation")
        self._layout_mode = mode if mode in ("horizontal", "vertical") else None
        old_size = saved.get("viewport_size")
        sx = bounds.width()/max(1, int(old_size[0])) if isinstance(old_size, (list, tuple)) and len(old_size)==2 else 1.0
        sy = bounds.height()/max(1, int(old_size[1])) if isinstance(old_size, (list, tuple)) and len(old_size)==2 else 1.0
        restored = 0
        for key, win in self._keys.items():
            data = saved.get(f"{key}_geometry")
            if isinstance(data, (list, tuple)) and len(data) == 4:
                x, y, w, h = map(int, data); r = QRect(round(x*sx), round(y*sy), max(win.minimumWidth(), round(w*sx)), max(win.minimumHeight(), round(h*sy)))
                win.setGeometry(self._clamp_rect(r, bounds)); restored += 1
        self._adapt_layout(force=self._layout_mode is None or restored < 2)
        self._last_viewport = self.viewport().size()

    def _adapt_layout(self, force=False):
        if self._guard or len(self._keys) < 2: return
        bounds = self._bounds()
        if bounds.width() < 100 or bounds.height() < 100: return
        a, b = self._keys.get("map"), self._keys.get("curve"); gap = 8; pad = 6
        min_horizontal = max(a.minimumWidth() + b.minimumWidth() + gap, 760) if a is not None and b is not None else 760
        desired = "horizontal" if bounds.width() >= min_horizontal else "vertical"
        visible = [w for w in self._keys.values() if w.isVisible()]
        if len(visible) <= 1:
            self._guard = True
            try:
                if visible: visible[0].setGeometry(bounds)
                self._layout_mode = desired; self._remember_valid()
            finally: self._guard = False
            self._last_viewport = self.viewport().size(); self._save_geometry_debounced(); return
        if force or desired != self._layout_mode:
            self._guard = True
            try:
                if a is not None and b is not None:
                    if desired == "horizontal":
                        avail_w = max(1, bounds.width() - 2*pad - gap); avail_h = max(1, bounds.height() - 2*pad)
                        ratio = max(0.25, min(0.75, float(getattr(self, "_map_ratio", 0.53))))
                        wa = min(int(avail_w*ratio), int(avail_h*0.95)); wb = min(int(avail_w*(1-ratio-0.01)), int(avail_h*0.90))
                        wa = max(a.minimumWidth(), wa); wb = max(b.minimumWidth(), wb)
                        if wa + wb > avail_w:
                            reduce_b = min(max(0, wb-b.minimumWidth()), wa+wb-avail_w); wb -= reduce_b; wa = min(wa, avail_w-wb)
                        if wa < a.minimumWidth() or wb < b.minimumWidth(): wa = max(a.minimumWidth(), avail_w-b.minimumWidth()); wb = avail_w-wa
                        ha = min(avail_h, max(a.minimumHeight(), min(int(avail_h*0.76), int(wa/1.08))))
                        hb = min(avail_h, max(b.minimumHeight(), min(int(avail_h*0.68), int(wb/1.25))))
                        row_w = wa + gap + wb; x0 = bounds.left() + pad + max(0, (avail_w-row_w)//2); y0 = bounds.top() + pad
                        a.setGeometry(QRect(x0, y0, wa, ha)); b.setGeometry(QRect(x0+wa+gap, y0, wb, hb))
                    else:
                        avail_w = max(1, bounds.width()-2*pad); avail_h = max(1, bounds.height()-2*pad-gap)
                        panel_w = min(avail_w, max(a.minimumWidth(), b.minimumWidth(), int(bounds.width()*0.94)))
                        ha = max(a.minimumHeight(), avail_h//2); hb = max(b.minimumHeight(), avail_h-ha)
                        if ha+hb > avail_h: ha = max(a.minimumHeight(), avail_h-b.minimumHeight()); hb = avail_h-ha
                        x0 = bounds.left() + (bounds.width()-panel_w)//2; y0 = bounds.top()+pad
                        a.setGeometry(QRect(x0, y0, panel_w, ha)); b.setGeometry(QRect(x0, y0+ha+gap, panel_w, hb))
                self._layout_mode = desired
                self._remember_valid()
            finally: self._guard = False
        else:
            self._scale_to_viewport()
            self._enforce_all()
        self._last_viewport = self.viewport().size()
        self._save_geometry_debounced()

    def _scale_to_viewport(self):
        old = self._last_viewport; new = self.viewport().size()
        if old.width() <= 0 or old.height() <= 0 or new.width() <= 0 or new.height() <= 0: return
        sx, sy = new.width()/old.width(), new.height()/old.height(); bounds = self._bounds()
        self._guard = True
        try:
            for win in self.subWindowList():
                if not win.isVisible() or win.isMinimized() or win.isMaximized(): continue
                r = win.geometry(); nw=max(win.minimumWidth(), round(r.width()*sx)); nh=max(win.minimumHeight(), round(r.height()*sy))
                nw=min(nw,bounds.width()); nh=min(nh,bounds.height())
                win.setGeometry(QRect(round(r.x()*sx), round(r.y()*sy), nw, nh).intersected(bounds))
        finally: self._guard = False

    @staticmethod
    def _clamp_rect(rect: QRect, bounds: QRect):
        w, h = min(rect.width(), bounds.width()), min(rect.height(), bounds.height())
        return QRect(max(bounds.left(), min(rect.x(), bounds.right()-w+1)), max(bounds.top(), min(rect.y(), bounds.bottom()-h+1)), w, h)

    def _enforce_geometry(self, active):
        if self._guard or active not in self.subWindowList() or not active.isVisible() or active.isMaximized() or active.isMinimized(): return
        others = [w for w in self.subWindowList() if w is not active and w.isVisible() and not w.isMinimized() and not w.isMaximized()]
        if not others: self._remember_valid(); self._save_geometry_debounced(); return
        bounds = self._bounds(); desired = self._clamp_rect(active.geometry(), bounds); other = others[0]; other_rect = self._clamp_rect(other.geometry(), bounds)
        needs_retile = False
        self._guard = True
        try:
            if desired != active.geometry(): active.setGeometry(desired)
            if desired.intersects(other_rect):
                w, h = desired.width(), desired.height(); r = other_rect
                candidates = [QRect(r.left()-w-4, desired.y(), w, h), QRect(r.right()+5, desired.y(), w, h),
                              QRect(desired.x(), r.top()-h-4, w, h), QRect(desired.x(), r.bottom()+5, w, h)]
                candidates = [self._clamp_rect(c, bounds) for c in candidates]
                candidates = [c for c in candidates if not c.intersects(r) and c.width() == w and c.height() == h]
                if candidates:
                    target = min(candidates, key=lambda c: abs(c.x()-desired.x()) + abs(c.y()-desired.y()))
                    active.setGeometry(target)
                else:
                    old = self._last_valid.get(active)
                    if old is not None and not old.intersects(r): active.setGeometry(self._clamp_rect(old, bounds))
                    else: needs_retile = True
            # Финальная страховка: все видимые панели помещаются в поле и не пересекаются.
            if any(active.geometry().intersects(w.geometry()) for w in others): needs_retile = True
            if not needs_retile: self._remember_valid()
        finally: self._guard = False
        if needs_retile: self._adapt_layout(force=True)
        self._save_geometry_debounced()

    def _enforce_all(self):
        wins = [w for w in self.subWindowList() if w.isVisible() and not w.isMinimized() and not w.isMaximized()]
        if len(wins) == 2 and wins[0].geometry().intersects(wins[1].geometry()): self._adapt_layout(force=True)
        bounds = self._bounds(); self._guard = True
        try:
            for win in wins: win.setGeometry(self._clamp_rect(win.geometry(), bounds))
            self._remember_valid()
        finally: self._guard = False

    def _remember_valid(self):
        wins = [w for w in self.subWindowList() if w.isVisible() and not w.isMinimized() and not w.isMaximized()]
        if len(wins) < 2 or not wins[0].geometry().intersects(wins[1].geometry()):
            for win in wins: self._last_valid[win] = QRect(win.geometry())

    def eventFilter(self, watched, event):
        if watched in self._keys.values() and event.type() in (QEvent.Type.Move, QEvent.Type.Resize) and not self._guard:
            QTimer.singleShot(0, lambda w=watched: self._enforce_geometry(w))
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._keys: QTimer.singleShot(0, self._adapt_layout)

    def _visibility_changed(self, *_):
        QTimer.singleShot(0, lambda: self._adapt_layout(force=True))

    def _save_geometry_debounced(self, *_):
        # Сохранение через один таймер не пишет settings.json на каждом пикселе drag.
        if not hasattr(self, "_save_timer"):
            from PyQt6.QtCore import QTimer as _Timer
            self._save_timer = _Timer(self); self._save_timer.setSingleShot(True); self._save_timer.timeout.connect(self.save_geometry)
        self._save_timer.start(450)

    def save_geometry(self):
        data = load_plot_display_settings(); work = data["workspace"]
        for key, win in self._keys.items():
            if win.isVisible() and not win.isMinimized() and not win.isMaximized(): work[f"{key}_geometry"] = list(win.geometry().getRect())
        work["orientation"] = self._layout_mode or "horizontal"; work["viewport_size"] = [self.viewport().width(), self.viewport().height()]
        try: save_plot_display_settings(data)
        except Exception: pass  # Ошибка записи настроек не должна ломать интерфейс.

class PlotWorkspaceWidget(QSplitter):
    """Слева инспектор ROI; справа — два независимо изменяемых научных экрана."""
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent); self.setObjectName("PlotWorkspaceWidget")
        self.roi_manager = RoiManagerWidget(); self.roi_table = self.roi_manager.table
        self.map_widget = MapPlotWidget(); self.curve_widget = CurvePlotWidget(); self._map_ratio = 0.53
        self.plot_area = QWidget(self); self.plot_area.setObjectName("PlotScreensArea"); self.plot_area.setMinimumSize(355, 500)
        layout = QVBoxLayout(self.plot_area); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(3)
        controls = QWidget(self.plot_area); controls.setObjectName("PlotScreensToolbar"); row = QHBoxLayout(controls); row.setContentsMargins(2, 0, 2, 0); row.setSpacing(4)
        self.map_window_btn = QToolButton(controls); self.map_window_btn.setText("Карта"); self.map_window_btn.setCheckable(True); self.map_window_btn.setChecked(True); self.map_window_btn.setToolTip("Показать/скрыть окно карты")
        self.curve_window_btn = QToolButton(controls); self.curve_window_btn.setText("ROC-кривые"); self.curve_window_btn.setCheckable(True); self.curve_window_btn.setChecked(True); self.curve_window_btn.setToolTip("Показать/скрыть окно рок-кривых")
        self.arrange_btn = QToolButton(controls); self.arrange_btn.setText("Выровнять"); self.arrange_btn.setToolTip("Разместить окна рядом без наложений")
        for b in (self.map_window_btn, self.curve_window_btn, self.arrange_btn): b.setAutoRaise(True); b.setFixedHeight(23)
        row.addWidget(self.map_window_btn); row.addWidget(self.curve_window_btn); row.addStretch(1); row.addWidget(self.arrange_btn)
        self.mdi_area = _PlotMdiArea(self.plot_area); self.mdi_area._map_ratio = self._map_ratio; layout.addWidget(controls); layout.addWidget(self.mdi_area, 1)
        self.map_subwindow = self.mdi_area.add_plot("map", "2D MAP · Детектор", self.map_widget)
        self.curve_subwindow = self.mdi_area.add_plot("curve", "ROC CURVES · Rocking curves", self.curve_widget)
        self.map_window_btn.toggled.connect(lambda v: self.mdi_area.show_panel("map", v))
        self.curve_window_btn.toggled.connect(lambda v: self.mdi_area.show_panel("curve", v))
        self.arrange_btn.clicked.connect(lambda: self.mdi_area._adapt_layout(force=True))
        self.mdi_area.panel_visibility_changed.connect(self._sync_panel_buttons)
        self.addWidget(self.roi_manager); self.addWidget(self.plot_area)
        self.setStretchFactor(0, 0); self.setStretchFactor(1, 1); self.setCollapsible(0, True); self.setCollapsible(1, False)
        self.setHandleWidth(4); self.setSizes([22, 800]); self.setMinimumSize(392, 500)
        self._style_toolbar(); QTimer.singleShot(0, self.mdi_area._layout_initial)

    def _sync_panel_buttons(self, key, visible):
        button = self.map_window_btn if key == "map" else self.curve_window_btn
        button.blockSignals(True); button.setChecked(bool(visible)); button.blockSignals(False)

    def _style_toolbar(self):
        c = theme_colors(resolve_theme(self))
        self.plot_area.setStyleSheet(f"QWidget#PlotScreensArea {{ background:{c['window']}; }} QWidget#PlotScreensToolbar QToolButton {{ color:{c['text']}; background:transparent; border:1px solid transparent; border-radius:3px; padding:2px 7px; font-size:10px; }} QWidget#PlotScreensToolbar QToolButton:hover {{ background:{c['hover']}; border-color:{c['border']}; }} QWidget#PlotScreensToolbar QToolButton:checked {{ background:{c['button']}; border-color:{c['border']}; }}")
        self.mdi_area.setBackground(self.palette().window())

    def apply_theme(self, theme=None):
        apply_widget_palette(self, theme); self._style_toolbar()
        for win in (self.map_subwindow, self.curve_subwindow):
            win.setPalette(self.palette()); win.setStyleSheet(f"QMdiSubWindow {{ background:{theme_colors(theme or resolve_theme(self))['window']}; border:1px solid {theme_colors(theme or resolve_theme(self))['border']}; }}")
        self.mdi_area.setBackground(self.palette().window())

    def setSizes(self, sizes):
        values = list(sizes or [])
        # Старый вызывающий код передаёт [ROI-менеджер, карта, кривые]; сохраняем этот контракт.
        if len(values) >= 3 and hasattr(self, "plot_area"):
            super().setSizes([values[0], max(1, sum(values[1:]))])
            total = max(1, sum(values[1:])); self._map_ratio = max(0.25, min(0.75, values[1]/total))
            QTimer.singleShot(0, lambda: self.mdi_area._adapt_layout(force=True))
        else: super().setSizes(values)

    def sizes(self): return super().sizes()
