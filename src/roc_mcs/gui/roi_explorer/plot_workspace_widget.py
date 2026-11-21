"""Рабочая область научных экранов: ручное изменение окон не перетасовывает соседей."""
from __future__ import annotations
import math
from PyQt6.QtCore import QEvent,QLineF,QRect,QSize,Qt,QTimer,pyqtSignal,QRectF
from PyQt6.QtWidgets import QHBoxLayout,QMdiArea,QMdiSubWindow,QSizePolicy,QSplitter,QToolButton,QVBoxLayout,QWidget,QMenu,QWidgetAction
from PyQt6.QtGui import QPainter,QPen,QColor,QPainterPath
from roc_mcs.gui.roi_explorer.roi_manager_widget import RoiManagerWidget
from roc_mcs.gui.roi_explorer.map_plot_widget import MapPlotWidget
from roc_mcs.gui.roi_explorer.curve_plot_widget import CurvePlotWidget
from roc_mcs.gui.roi_explorer.vtk_backend.embedded_widget import EmbeddedVTKVolumeWidget
from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings,save_plot_display_settings
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette,resolve_theme,theme_colors

_LAYOUTS = ("focus", "row", "column", "map3d-top")
_LAYOUT_VERSION = 8
DEFAULT_VISIBLE_PANELS = ("map", "curve")
LAYOUT_PREVIEW_STYLE = "A"  # A/B/C — варианты рисунка раскладок; см. PNG-лист.
FOCUS_LEFT_RATIO = 0.50  # Доля ширины карты в стартовой раскладке.
FOCUS_TOP_RATIO = 0.50  # Доля высоты верхнего окна ROC справа.
ROI_RAIL_COMPACT_WIDTH = 22
ROI_RAIL_EXPANDED_WIDTH = 282

class _PlotSubWindow(QMdiSubWindow):
    """Закрытие скрывает MDI-экран; финальный backend teardown принадлежит владельцу."""
    def __init__(self, area, key, title, on_close=None):
        super().__init__()
        self._area = area
        self.plot_key = key
        self._on_explicit_close = on_close
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandMove, True)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandResize, True)

    def closeEvent(self, event):
        # MDI close means "hide this panel", not "destroy its renderer".
        # The 3D scene remains reusable while the other ROI Explorer panels are
        # open. Any optional backend cleanup is deferred until this close event
        # has returned, never performed re-entrantly inside QMdiSubWindow.
        callback = self._on_explicit_close
        event.accept()
        self.hide()
        self._area._on_subwindow_closed(self.plot_key)
        if callable(callback):
            QTimer.singleShot(0, callback)


class _PlotMdiArea(QMdiArea):
    """Управляет геометрией/порядком MDI-панелей без пересоздания их содержимого."""
    panel_visibility_changed = pyqtSignal(str, bool)
    panel_restored = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("plotMdiArea")
        self.setViewMode(QMdiArea.ViewMode.SubWindowView)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setActivationOrder(QMdiArea.WindowOrder.ActivationHistoryOrder)
        self.setOption(QMdiArea.AreaOption.DontMaximizeSubWindowOnActivation, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(340, 300)
        self._keys = {}
        self._layout_choice = "focus"
        self._initial_layout_done = False
        self._initial_layout_pending = False
        self._maximized_key = None
        self._viewport_size = QSize()
        self._area_size = QSize()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self.save_geometry)
        self._viewport_timer = QTimer(self)
        self._viewport_timer.setSingleShot(True)
        self._viewport_timer.setInterval(16)
        self._viewport_timer.timeout.connect(self._fit_after_viewport_resize)
        # Visibility changes are laid out synchronously in show_panel(); do not
        # schedule a second 100 ms layout which caused the visible overlap/flicker.
        self.panel_visibility_changed.connect(self._schedule_save)

    def _queue_initial_layout(self):
        """Запланировать геометрию; видимость начальных окон от этого не зависит."""
        if self._initial_layout_done or self._initial_layout_pending:
            return
        self._initial_layout_pending = True
        QTimer.singleShot(0, self._run_initial_layout)

    def _run_initial_layout(self):
        self._initial_layout_pending = False
        self._layout_initial()

    def showEvent(self, event):
        super().showEvent(event)
        self._queue_initial_layout()

    def add_plot(self, key, title, widget, *, initial_visible=True, on_close=None):
        win = _PlotSubWindow(self, key, title, on_close=on_close)
        win.setWindowFlags(
            Qt.WindowType.SubWindow
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        win.setWidget(widget)
        minimum = widget.minimumSize()
        hint = widget.minimumSizeHint()
        win.setMinimumSize(
            QSize(
                max(250, minimum.width(), hint.width()) + 2,
                max(190, minimum.height(), hint.height()) + 24,
            )
        )
        self.addSubWindow(win)
        self._keys[key] = win
        win.installEventFilter(self)
        win.windowStateChanged.connect(
            lambda old, new, panel_key=key: self._on_window_state_changed(panel_key, old, new)
        )
        # Важно: стартовые карта и ROC остаются видимыми сразу при создании.
        # Только 3D явно скрыто до нажатия кнопки. Не повторять прежнюю ошибку,
        # когда все MDI-окна hide() зависели от успешного первого layout pass.
        if initial_visible:
            win.show()
        else:
            win.hide()
        return win

    def _current_maximized_key(self):
        for key, win in self._keys.items():
            if not win.isHidden() and not win.isMinimized() and win.isMaximized():
                return key
        return None

    def _on_window_state_changed(self, key, old_state, new_state):
        minimized = Qt.WindowState.WindowMinimized
        maximized = Qt.WindowState.WindowMaximized
        win = self._keys.get(key)
        if win is None:
            return

        if bool(new_state & maximized):
            self._maximized_key = key
            # windowStateChanged приходит во время перехода Qt. Переставляем
            # sibling windows в следующий event-loop turn, когда MDI уже
            # завершил внутреннюю обработку maximize/restore.
            QTimer.singleShot(0, lambda panel_key=key: self._restack_panel(panel_key))
        elif bool(old_state & maximized):
            if self._maximized_key == key:
                self._maximized_key = None
            QTimer.singleShot(0, lambda panel_key=key: self._restack_panel(panel_key))

        if bool(old_state & minimized) and not bool(new_state & minimized):
            QTimer.singleShot(0, lambda panel_key=key: self._notify_panel_restored(panel_key))

    def _restack_panel(self, key):
        win = self._keys.get(key)
        if win is None or win.isHidden() or win.isMinimized():
            return
        # В максимизированном состоянии все остальные MDI-панели должны
        # оставаться позади 3D. На restore активная панель также остаётся
        # сверху, но геометрия соседей не меняется.
        for other in self._keys.values():
            if other is not win and not other.isHidden() and not other.isMinimized():
                other.lower()
        win.raise_()
        self.setActiveSubWindow(win)

    def _notify_panel_restored(self, key):
        if self.is_panel_visible(key):
            self._restack_panel(key)
            self.panel_restored.emit(key)

    def _on_subwindow_closed(self, key):
        self._maximized_key = None if self._maximized_key == key else self._maximized_key
        maximized_key = self._current_maximized_key()
        if maximized_key:
            QTimer.singleShot(0, lambda panel_key=maximized_key: self._restack_panel(panel_key))
        else:
            # Убираем освободившуюся панель и синхронно перераскладываем только
            # оставшиеся окна. Нет отложенного второго layout pass.
            self._apply_layout(self._layout_choice, windows=self._visible())
        self.panel_visibility_changed.emit(key, False)

    def show_panel(self, key, visible):
        win = self._keys.get(key)
        if win is None:
            return

        if visible:
            if win.isMinimized():
                win.showNormal()
            if not self._initial_layout_done and self.isVisible():
                self._layout_initial()

            # При открытии 3D конечная геометрия назначается ДО show(). Поэтому
            # showEvent/ленивая инициализация QVTK сразу получают правильный
            # размер; не возникает первого кадра с наложенными окнами.
            target = [item for item in self._visible() if item is not win]
            target.append(win)
            maximized_key = self._current_maximized_key()
            if maximized_key is None:
                self._apply_layout(self._layout_choice, windows=target)

            win.show()
            if maximized_key is None:
                win.raise_()
                self.setActiveSubWindow(win)
            else:
                QTimer.singleShot(0, lambda panel_key=maximized_key: self._restack_panel(panel_key))
        else:
            if win.isMaximized():
                win.showNormal()
            win.hide()
            if self._maximized_key == key:
                self._maximized_key = None
            maximized_key = self._current_maximized_key()
            if maximized_key is None:
                self._apply_layout(self._layout_choice, windows=self._visible())
            else:
                QTimer.singleShot(0, lambda panel_key=maximized_key: self._restack_panel(panel_key))

        self.panel_visibility_changed.emit(key, bool(visible))

    def is_panel_visible(self, key):
        win = self._keys.get(key)
        return bool(win is not None and not win.isHidden() and not win.isMinimized())

    def _bounds(self):
        return self.viewport().rect().adjusted(2, 2, -2, -2)

    def _visible(self):
        return [win for win in self._keys.values() if not win.isHidden() and not win.isMinimized()]

    def _layout_initial(self):
        """Применить стартовую геометрию после получения реального viewport.

        Этот метод никогда не меняет видимость окон. Если первый Qt showEvent
        пришёл с временным размером viewport, resizeEvent запросит повторную
        попытку; карта и ROC к этому моменту уже показаны.
        """
        if self._initial_layout_done or not self.isVisible():
            return
        bounds = self._bounds()
        if bounds.width() < 120 or bounds.height() < 100:
            return

        self._area_size = self.size()
        saved = load_plot_display_settings().get("workspace", {})
        self._layout_choice = (
            saved.get("layout_choice", "focus")
            if saved.get("layout_choice") in _LAYOUTS
            else "focus"
        )

        if int(saved.get("layout_version", 0) or 0) < _LAYOUT_VERSION:
            self._apply_layout("focus")
            self._initial_layout_done = True
            self._viewport_size = self.viewport().size()
            self._area_size = self.size()
            return

        prev = saved.get("viewport_size") or [bounds.width(), bounds.height()]
        sx = bounds.width() / max(1, int(prev[0]))
        sy = bounds.height() / max(1, int(prev[1]))
        restored_geometries = {}
        for key in self._keys:
            geometry = saved.get(f"{key}_geometry")
            if isinstance(geometry, (list, tuple)) and len(geometry) == 4:
                try:
                    x, y, width, height = map(int, geometry)
                    restored_geometries[key] = self._clamp(
                        QRect(
                            round(x * sx), round(y * sy),
                            round(width * sx), round(height * sy),
                        ),
                        bounds,
                    )
                except (ValueError, TypeError, OverflowError):
                    pass

        # Восстанавливаем пользовательскую геометрию только когда она есть для
        # всех сейчас видимых окон. Частичный набор мог бы оставить одну панель
        # в Qt default-геометрии и визуально наложить её на соседнюю.
        visible_keys = [win.plot_key for win in self._visible()]
        if visible_keys and all(key in restored_geometries for key in visible_keys):
            for key, geometry in restored_geometries.items():
                self._keys[key].setGeometry(geometry)
        else:
            self._apply_layout(self._layout_choice)

        self._initial_layout_done = True
        self._viewport_size = self.viewport().size()
        self._area_size = self.size()

    def set_layout_choice(self, choice):
        if choice not in _LAYOUTS:
            return
        self._layout_choice = choice
        if self._current_maximized_key() is None:
            self._apply_layout(choice)
        self._schedule_save()

    def _apply_layout(self, choice=None, *, windows=None):
        if choice in _LAYOUTS:
            self._layout_choice = choice
        wins = list(self._visible() if windows is None else windows)
        wins = [win for win in wins if win is not None and not win.isMinimized()]
        if not wins or any(win.isMaximized() for win in wins):
            return
        bounds = self._bounds()
        n = len(wins)
        if bounds.width() < 120 or bounds.height() < 100:
            return
        by = {win.plot_key: win for win in wins}
        x, y, width, height = bounds.x(), bounds.y(), bounds.width(), bounds.height()
        gap = 6

        def minimum(key, axis):
            win = by[key]
            return win.minimumWidth() if axis == "w" else win.minimumHeight()

        choice = self._layout_choice
        if n == 1:
            by[wins[0].plot_key].setGeometry(bounds)
        elif n == 2:
            keys = [key for key in ("map", "curve", "volume") if key in by]
            first_key, second_key = keys
            if choice == "column" and height-gap >= minimum(first_key, "h") + minimum(second_key, "h"):
                first_size = max(
                    minimum(first_key, "h"),
                    min(height-gap-minimum(second_key, "h"), int((height-gap)*0.5)),
                )
                by[first_key].setGeometry(QRect(x, y, width, first_size))
                by[second_key].setGeometry(QRect(x, y+first_size+gap, width, height-first_size-gap))
            elif width-gap >= minimum(first_key, "w") + minimum(second_key, "w"):
                first_size = max(
                    minimum(first_key, "w"),
                    min(width-gap-minimum(second_key, "w"), int((width-gap)*0.5)),
                )
                by[first_key].setGeometry(QRect(x, y, first_size, height))
                by[second_key].setGeometry(QRect(x+first_size+gap, y, width-first_size-gap, height))
            else:
                first_size = max(1, (height-gap)//2)
                by[first_key].setGeometry(QRect(x, y, width, first_size))
                by[second_key].setGeometry(QRect(x, y+first_size+gap, width, height-first_size-gap))
        elif choice == "row" and width-2*gap >= sum(minimum(key, "w") for key in ("map", "curve", "volume")):
            sizes = self._fit_sizes(
                width-2*gap, [.34, .33, .33],
                [minimum(key, "w") for key in ("map", "curve", "volume")],
            )
            pos = x
            for key, size in zip(("map", "curve", "volume"), sizes):
                by[key].setGeometry(QRect(pos, y, size, height))
                pos += size + gap
        elif choice == "column" and height-2*gap >= sum(minimum(key, "h") for key in ("map", "curve", "volume")):
            sizes = self._fit_sizes(
                height-2*gap, [.30, .30, .40],
                [minimum(key, "h") for key in ("map", "curve", "volume")],
            )
            pos = y
            for key, size in zip(("map", "curve", "volume"), sizes):
                by[key].setGeometry(QRect(x, pos, width, size))
                pos += size + gap
        elif choice == "map3d-top" and height-gap >= max(minimum("map", "h"), minimum("volume", "h")) + minimum("curve", "h"):
            top = max(
                max(minimum("map", "h"), minimum("volume", "h")),
                min(height-gap-minimum("curve", "h"), int((height-gap)*.58)),
            )
            left = max(
                minimum("map", "w"),
                min(width-gap-minimum("volume", "w"), (width-gap)//2),
            )
            by["map"].setGeometry(QRect(x, y, left, top))
            by["volume"].setGeometry(QRect(x+left+gap, y, width-left-gap, top))
            by["curve"].setGeometry(QRect(x, y+top+gap, width, height-top-gap))
        else:
            # Стартовая схема: карта слева, ROC сверху справа, 3D снизу справа.
            left_min = max(minimum("map", "w"), 180)
            right_min = max(minimum("curve", "w"), minimum("volume", "w"), 180)
            if width-gap >= left_min+right_min and height-gap >= minimum("curve", "h")+minimum("volume", "h"):
                left = max(left_min, min(width-gap-right_min, round((width-gap)*FOCUS_LEFT_RATIO)))
                right = width-left-gap
                top = max(
                    minimum("curve", "h"),
                    min(height-gap-minimum("volume", "h"), round((height-gap)*FOCUS_TOP_RATIO)),
                )
                by["map"].setGeometry(QRect(x, y, left, height))
                by["curve"].setGeometry(QRect(x+left+gap, y, right, top))
                by["volume"].setGeometry(QRect(x+left+gap, y+top+gap, right, height-top-gap))
            else:
                sizes = self._fit_sizes(
                    max(3, height-2*gap), [.34, .33, .33],
                    [minimum(key, "h") for key in ("map", "curve", "volume")],
                )
                pos = y
                for key, size in zip(("map", "curve", "volume"), sizes):
                    by[key].setGeometry(QRect(x, pos, width, size))
                    pos += size + gap

        self._viewport_size = self.viewport().size()
        self._area_size = self.size()
        self._schedule_save()

    @staticmethod
    def _fit_sizes(total, weights, minimums):
        total = max(1, int(total))
        minimums = [max(1, int(value)) for value in minimums]
        if sum(minimums) > total:
            return [max(1, int(total*max(.01, weight)/sum(weights))) for weight in weights]
        weights = [max(.01, float(value)) for value in weights]
        weight_sum = sum(weights)
        sizes = [max(minimum, int(total*weight/weight_sum)) for weight, minimum in zip(weights, minimums)]
        while sum(sizes) > total:
            index = max(
                (i for i, value in enumerate(sizes) if value > minimums[i]),
                key=lambda i: sizes[i]-minimums[i],
                default=None,
            )
            if index is None:
                break
            sizes[index] -= 1
        while sum(sizes) < total:
            sizes[max(range(len(weights)), key=lambda i: weights[i])] += 1
        return sizes

    @staticmethod
    def _clamp(rect, bounds):
        width = max(1, min(rect.width(), bounds.width()))
        height = max(1, min(rect.height(), bounds.height()))
        return QRect(
            max(bounds.x(), min(rect.x(), bounds.right()-width+1)),
            max(bounds.y(), min(rect.y(), bounds.bottom()-height+1)),
            width,
            height,
        )

    def resizeEvent(self, event):
        old = QSize(self._area_size)
        super().resizeEvent(event)
        new = QSize(self.size())
        if not self._initial_layout_done and self.isVisible():
            bounds = self._bounds()
            if bounds.width() >= 120 and bounds.height() >= 100:
                self._queue_initial_layout()
        elif old.isValid() and old.width() > 0 and old.height() > 0 and (new.width() != old.width() or new.height() != old.height()):
            self._viewport_timer.start()
        elif not old.isValid():
            self._area_size = new
            self._viewport_size = self.viewport().size()

    def _fit_after_viewport_resize(self):
        bounds = self._bounds()
        for win in self._visible():
            if win.isMaximized():
                continue
            geometry = win.geometry()
            width = min(geometry.width(), bounds.width())
            height = min(geometry.height(), bounds.height())
            x = max(bounds.left(), min(geometry.x(), bounds.right()-width+1))
            y = max(bounds.top(), min(geometry.y(), bounds.bottom()-height+1))
            if (x, y, width, height) != (geometry.x(), geometry.y(), geometry.width(), geometry.height()):
                win.setGeometry(QRect(x, y, width, height))
        self._viewport_size = self.viewport().size()
        self._area_size = self.size()
        self._schedule_save()

    def eventFilter(self, watched, event):
        # Перетаскивание/ручной resize сохраняются, но не перестраивают соседей.
        if watched in self._keys.values() and event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
            self._schedule_save()
        return super().eventFilter(watched, event)

    def _schedule_save(self, *_args):
        self._save_timer.start()

    def save_geometry(self):
        data = load_plot_display_settings()
        workspace = data.setdefault("workspace", {})
        for key, win in self._keys.items():
            geometry = win.normalGeometry() if (win.isMaximized() or win.isMinimized()) else win.geometry()
            workspace[f"{key}_geometry"] = list(geometry.getRect()) if not win.isHidden() else None
        workspace["orientation"] = "manual"
        workspace["layout_choice"] = self._layout_choice
        workspace["layout_version"] = _LAYOUT_VERSION
        workspace["viewport_size"] = [max(1, self.viewport().width()), max(1, self.viewport().height())]
        try:
            save_plot_display_settings(data)
        except Exception:
            pass


class _LayoutPreviewButton(QToolButton):
    """Схематичная плитка-предпросмотр схемы окон."""
    def __init__(self,mode,label,parent=None):
        super().__init__(parent);self.mode=mode;self.label=label;self._selected=False;self.setFixedSize(158,54);self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly);self.setCursor(Qt.CursorShape.PointingHandCursor);self.setToolTip(label)
    def set_selected(self,value):self._selected=bool(value);self.update()
    def paintEvent(self,event):
        p=QPainter(self)
        if not p.isActive(): return
        p.setRenderHint(QPainter.RenderHint.Antialiasing,True);pal=self.palette();bg=pal.color(pal.ColorRole.Base);fg=pal.color(pal.ColorRole.Text);edge=pal.color(pal.ColorRole.Highlight if self._selected else pal.ColorRole.Mid)
        p.setPen(QPen(edge,1.5 if self._selected else 0.8));p.setBrush(bg);p.drawRoundedRect(QRectF(1,1,self.width()-2,self.height()-2),4,4)
        left,top,ww,hh,gap=8,7,82,40,3;panel=QColor(100,145,185,75)
        def tile(rx,ry,rw,rh,kind):
            p.setPen(QPen(QColor(100,145,185,180),0.75));p.setBrush(panel);p.drawRoundedRect(QRectF(rx,ry,rw,rh),2,2)
            if kind=="ROC":
                # Только гауссиана без текста «ROC»: проще распознаётся в маленькой плитке.
                x0=rx+max(2,rw*.10);x1=rx+rw-max(2,rw*.10);baseline=ry+rh*.78;mid=(x0+x1)*.5;half=max(1,(x1-x0)*(.19 if LAYOUT_PREVIEW_STYLE=="A" else .24 if LAYOUT_PREVIEW_STYLE=="B" else .15));amp=max(2,rh*(.43 if LAYOUT_PREVIEW_STYLE!="C" else .34));path=QPainterPath();pts=[]
                for j in range(31):
                    xx=x0+(x1-x0)*j/30;yy=baseline-amp*math.exp(-.5*((xx-mid)/half)**2);pts.append((xx,yy))
                path.moveTo(*pts[0]);[path.lineTo(xx,yy) for xx,yy in pts[1:]]
                if LAYOUT_PREVIEW_STYLE=="B":p.setPen(Qt.PenStyle.NoPen);p.setBrush(QColor(90,150,195,75));fill=QPainterPath(path);fill.lineTo(x1,baseline);fill.lineTo(x0,baseline);fill.closeSubpath();p.drawPath(fill)
                p.setBrush(Qt.BrushStyle.NoBrush);p.setPen(QPen(fg,1.25 if LAYOUT_PREVIEW_STYLE!="C" else 1.0));p.drawPath(path)
                if LAYOUT_PREVIEW_STYLE=="C":p.setPen(QPen(fg,0.7));p.drawLine(QLineF(x0,baseline,x1,baseline))
            else:
                p.setPen(fg);p.drawText(QRectF(rx,ry,rw,rh),int(Qt.AlignmentFlag.AlignCenter),kind)
        if self.mode=="focus":tile(left,top,ww*.5-gap/2,hh,"2D");tile(left+ww*.5+gap/2,top,ww*.5-gap/2,hh*.49,"ROC");tile(left+ww*.5+gap/2,top+hh*.51,ww*.5-gap/2,hh*.49,"3D")
        elif self.mode=="row":
            cw=(ww-2*gap)/3
            for i,kind in enumerate(("2D","ROC","3D")):tile(left+i*(cw+gap),top,cw,hh,kind)
        elif self.mode=="column":
            ch=(hh-2*gap)/3
            for i,kind in enumerate(("2D","ROC","3D")):tile(left,top+i*(ch+gap),ww,ch,kind)
        else:tile(left,top,ww*.5-gap/2,hh*.49,"2D");tile(left+ww*.5+gap/2,top,ww*.5-gap/2,hh*.49,"3D");tile(left,top+hh*.51,ww,hh*.49,"ROC")
        p.setPen(fg);p.drawText(QRectF(96,4,56,46),int(Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignLeft|Qt.TextFlag.TextWordWrap),self.label);p.end()

class PlotWorkspaceWidget(QSplitter):
    """Слева менеджер ROI; справа три независимых MDI-экрана."""
    def __init__(self,parent=None):
        super().__init__(Qt.Orientation.Horizontal,parent);self.setObjectName("PlotWorkspaceWidget");self.roi_manager=RoiManagerWidget();self.roi_table=self.roi_manager.table;self.map_widget=MapPlotWidget();self.curve_widget=CurvePlotWidget();self.volume_widget=EmbeddedVTKVolumeWidget();self._map_ratio=.53
        self.plot_area=QWidget(self);self.plot_area.setObjectName("PlotScreensArea");self.plot_area.setMinimumSize(340,300);layout=QVBoxLayout(self.plot_area);layout.setContentsMargins(0,0,0,0);layout.setSpacing(2)
        controls=QWidget(self.plot_area);controls.setObjectName("PlotScreensToolbar");row=QHBoxLayout(controls);row.setContentsMargins(2,0,2,0);row.setSpacing(4);self._buttons={}
        for key,text,tip in (("map","Карта","Показать или скрыть окно карты"),("curve","ROC-кривые","Показать или скрыть окно ROC-кривых"),("volume","3D","Показать или скрыть 3D-объём")):
            b=QToolButton(controls);b.setText(text);b.setCheckable(True);b.setChecked(key in DEFAULT_VISIBLE_PANELS);b.setToolTip(tip);b.setAutoRaise(True);b.setFixedHeight(23);row.addWidget(b);self._buttons[key]=b
        self.map_window_btn=self._buttons["map"];self.curve_window_btn=self._buttons["curve"];self.volume_window_btn=self._buttons["volume"]
        self.layout_btn=QToolButton(controls);self.layout_btn.setText("Раскладка ▦");self.layout_btn.setToolTip("Выбрать схему размещения окон");self.layout_btn.setAutoRaise(True);self.layout_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup);self.layout_menu=QMenu(self.layout_btn);self.layout_btn.setMenu(self.layout_menu);self._layout_preview_buttons={}
        self.arrange_btn=QToolButton(controls);self.arrange_btn.setText("Выровнять");self.arrange_btn.setToolTip("Применить выбранную схему к видимым окнам");self.arrange_btn.setAutoRaise(True);self.arrange_btn.setFixedHeight(23);row.addStretch(1);row.addWidget(self.layout_btn);row.addWidget(self.arrange_btn)
        self.mdi_area=_PlotMdiArea(self.plot_area);layout.addWidget(controls);layout.addWidget(self.mdi_area,1)
        self.map_subwindow=self.mdi_area.add_plot("map","2D MAP · Детектор",self.map_widget);self.curve_subwindow=self.mdi_area.add_plot("curve","ROC CURVES · Rocking curves",self.curve_widget);self.volume_subwindow=self.mdi_area.add_plot("volume","3D VOLUME · Diffraction",self.volume_widget,initial_visible=False)
        for key,b in self._buttons.items():
            b.toggled.connect(lambda visible,k=key:self.mdi_area.show_panel(k,visible))
            if key not in DEFAULT_VISIBLE_PANELS:self.mdi_area.show_panel(key,False)
        for mode,label in (("focus","Карта слева · ROC и 3D справа"),("row","Три окна в ряд"),("column","Три окна вертикально"),("map3d-top","Карта и 3D сверху · ROC снизу")):
            preview=_LayoutPreviewButton(mode,label,self.layout_menu);action=QWidgetAction(self.layout_menu);action.setDefaultWidget(preview);self.layout_menu.addAction(action);preview.clicked.connect(lambda _=False,m=mode:self._choose_layout(m));self._layout_preview_buttons[mode]=preview
        self.arrange_btn.clicked.connect(lambda:self.mdi_area.set_layout_choice(self.mdi_area._layout_choice));self.mdi_area.panel_visibility_changed.connect(self._sync_panel_buttons)
        self.addWidget(self.roi_manager);self.addWidget(self.plot_area);self.setStretchFactor(0,0);self.setStretchFactor(1,1);self.setCollapsible(0,True);self.setCollapsible(1,False);self.setHandleWidth(8);self.roi_manager.setFixedWidth(22);self.setSizes([22,900]);self.setMinimumSize(400,340);self._style_toolbar();QTimer.singleShot(0,self._force_compact_rail)
    def _choose_layout(self,mode):
        self.mdi_area.set_layout_choice(mode)
        for key,button in self._layout_preview_buttons.items():button.set_selected(key==mode)
        self.layout_menu.close()
    def _sync_panel_buttons(self,key,visible):
        b=self._buttons.get(key)
        if b:b.blockSignals(True);b.setChecked(bool(visible));b.blockSignals(False)
    def _style_toolbar(self):
        c=theme_colors(resolve_theme(self));self.plot_area.setStyleSheet(f"QWidget#PlotScreensArea{{background:{c['window']};}} QWidget#PlotScreensToolbar QToolButton{{color:{c['text']};background:transparent;border:1px solid transparent;border-radius:3px;padding:2px 7px;font-size:10px;}} QWidget#PlotScreensToolbar QToolButton:hover{{background:{c['hover']};border-color:{c['border']};}} QWidget#PlotScreensToolbar QToolButton:checked{{background:{c['button']};border-color:{c['border']};}}")
    def apply_theme(self,theme=None):
        apply_widget_palette(self,theme);self._style_toolbar();c=theme_colors(theme or resolve_theme(self))
        for win in (self.map_subwindow,self.curve_subwindow,self.volume_subwindow):win.setPalette(self.palette());win.setStyleSheet(f"QMdiSubWindow{{background:{c['window']};border:2px solid {c['border']};border-radius:2px;}} QMdiSubWindow:hover{{border-color:{c['highlight']};}}")
        self.mdi_area.setBackground(self.palette().window())
        for key,button in self._layout_preview_buttons.items():button.set_selected(key==self.mdi_area._layout_choice)
    def setSizes(self,sizes):
        values=[max(0,int(v)) for v in (sizes or [])]
        if not values:return
        if len(values)>=3:
            right_total=max(1,sum(values[1:]));self._map_ratio=max(.25,min(.75,values[1]/right_total))
        total=max(sum(values),self.width() if self.width()>0 else 0);rail=ROI_RAIL_EXPANDED_WIDTH if self.roi_manager._expanded else ROI_RAIL_COMPACT_WIDTH
        super().setSizes([rail,max(1,total-rail)])
    def _force_compact_rail(self):
        if not self.roi_manager._expanded:
            self.roi_manager.table.hide();self.roi_manager.setFixedWidth(ROI_RAIL_COMPACT_WIDTH);self.roi_manager.rail.setFixedWidth(ROI_RAIL_COMPACT_WIDTH-2)
            super().setSizes([ROI_RAIL_COMPACT_WIDTH,max(1,self.width()-ROI_RAIL_COMPACT_WIDTH-8)])
