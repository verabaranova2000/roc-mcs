"""Визуальные компоненты live ROC dashboard; без логики управления сеансом."""
from __future__ import annotations
from typing import Any
import numpy as np
import pyqtgraph as pg
from matplotlib import cm
from PyQt6 import QtCore, QtWidgets
from .highlight import RocSliceHighlight
from .axis_calibration import transform_axis, resample_rows_to_uniform_axis, effective_scale
from .profile_metrics import fwhm_half_height, peak_position
from roc_mcs.fitting.registry import MODEL_SPECS
from roc_mcs.processing.moments import compute_profile_moments

class RocMapWidget(pg.PlotWidget):
    """Interactive ROC-map view."""
    # Теперь виджет отправляет сырую X-координату мыши, а не вычисляет индексы.
    sigHoverX = QtCore.pyqtSignal(float)
    sigClickedX = QtCore.pyqtSignal(float)
    sigHoverLeft = QtCore.pyqtSignal()
    sigColorLevelsChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.14)
        self.setLabel("bottom", "Время", units="с")
        # self.setLabel("left", "Параметр s", units="отн. ед.")
        # self.setTitle("ROC map")
        # self.setMouseTracking(True)
        self.setLabel("left", "Параметр s", units="отн. ед.")
        self.getAxis("left").enableAutoSIPrefix(enable=False)
        
        self.setTitle("ROC-карта")
        self.setMouseTracking(True)
        
        self.image_item = pg.ImageItem(axisOrder="row-major")
        # Inferno: постоянная цветовая шкала с тёмным минимумом и светлым максимумом.
        self.image_item.setLookupTable((cm.get_cmap("inferno", 256)(np.linspace(0, 1, 256)) * 255).astype(np.uint8))
        self.image_item.setZValue(-10)
        self.addItem(self.image_item)
        self._color_levels = None
        self._last_roc_map = None

        # Отдельный ViewBox накладывает нагрузку/давление на правую ось.
        self.load_view = pg.ViewBox()
        self.scene().addItem(self.load_view)
        self.load_view.setZValue(100)
        # 2. Включаем уже существующую правую ось (вместо создания новой)
        self.showAxis('right')
        right_axis = self.getAxis('right')
        right_axis.linkToView(self.load_view)
        
        # 3. Синхронизируем ось X нового слоя с основной рок-картой
        self.load_view.setXLink(self.getPlotItem())
        # Для правой оси тоже отключаем авто-масштабирование (опционально, но надежно)
        right_axis.setLabel("Сила", units="кг", color="#ff6464")
        
        # 4. Линия самого графика нагрузки (красная, толстая)
        self.load_curve = pg.PlotDataItem(pen=pg.mkPen(color=(255, 100, 100, 220), width=2.5))
        self.load_view.addItem(self.load_curve)
        
        # 5. Обработчик изменения размеров окна (чтобы слои не разъезжались)
        self.getPlotItem().vb.sigResized.connect(self._update_load_view_geometry)
        # ------------------------------------------
        
        # Плавный курсор-линейка
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 230, 120, 220), width=2))
        self.cursor.hide()
        self.addItem(self.cursor)
        
        # Подсветка среза
        self.slice_highlight = RocSliceHighlight(self.plotItem.vb, color=(100, 220, 255), fade_px=24, edge_alpha=110)

        self._mouse_proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)
        self._click_proxy = pg.SignalProxy(self.scene().sigMouseClicked, rateLimit=60, slot=self._on_mouse_clicked)

        self._y_min = 0.0
        self._y_max = 1.0

    def _update_load_view_geometry(self):
        """Гарантирует, что слой нагрузки всегда совпадает по размерам с рок-картой."""
        self.load_view.setGeometry(self.getPlotItem().vb.sceneBoundingRect())
        self.load_view.linkedViewChanged(self.getPlotItem().vb, self.load_view.XAxis)

    def set_load_data(self, time_s: np.ndarray, values: np.ndarray, metric: str = "force"):
        """Отрисовывает нагрузку или давление на правой оси."""
        if metric == "pressure":
            self.getAxis("right").setLabel("Давление", units="МПа", color="#ff6464")
        else:
            self.getAxis("right").setLabel("Сила", units="кг", color="#ff6464")
        self.load_curve.setData(time_s, values)
    
    @staticmethod
    def _finite_levels(image: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(image, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0: return 0.0, 1.0
        low, high = float(finite.min()), float(finite.max())
        if high <= low: high = low + 1.0
        return low, high

    def auto_levels(self) -> None:
        """Пересчитывает пределы по накопленным сканам и затем фиксирует их."""
        if self._last_roc_map is not None:
            self._color_levels = None
            self.set_roc_map(self._last_roc_map)

    def clear_map(self) -> None:
        """Убирает старую карту перед началом нового сеанса."""
        self._last_roc_map = None
        self._color_levels = None
        self.image_item.hide()
        self.load_curve.setData([], [])
        self.cursor.hide()
        self.slice_highlight.hide()
        self._y_min, self._y_max = 0.0, 1.0
        self.setLimits(xMin=0, xMax=1, yMin=0, yMax=1)
        self.getPlotItem().vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)

    def set_roc_map(self, roc_map: dict[str, Any]) -> None:
        self._last_roc_map = roc_map
        time_s = np.asarray(roc_map["time_s"], dtype=float)
        theta = np.asarray(roc_map.get("theta_axis", roc_map["s_axis"]), dtype=float)
        axis_name = str(roc_map.get("axis_name", "Параметр s"))
        axis_unit = str(roc_map.get("axis_unit", "отн. ед."))
        self.setLabel("left", axis_name, units=axis_unit)
        image = np.asarray(roc_map["intensity"], dtype=float)
        if theta.size and image.ndim == 2 and image.shape[1] == theta.size:
            # ImageItem использует равномерную сетку. Переинтерполируем только растр по θ;
            # исходные профили в карточках и MCS-файлы остаются без изменений.
            theta, image = resample_rows_to_uniform_axis(theta, image)
        if self._color_levels is None: self._color_levels = self._finite_levels(image)
        self.image_item.setImage(image.T, autoLevels=False, levels=self._color_levels)
        self.image_item.show()
        self.sigColorLevelsChanged.emit(*self._color_levels)

        x0 = float(time_s.min())
        x1 = float(time_s.max()) if len(time_s) > 1 else x0 + 1.0
        self._y_min = float(theta.min())
        self._y_max = float(theta.max()) if len(theta) > 1 else self._y_min + 1.0
        
        self.image_item.setRect(QtCore.QRectF(x0, self._y_min, x1 - x0, self._y_max - self._y_min))
        self.setLimits(xMin=x0, xMax=x1, yMin=self._y_min, yMax=self._y_max)
        self.autoRange(padding=0.02)
        self.slice_highlight.refresh()

    def show_cursors(self, cursor_x: float, slice_left: float, slice_right: float) -> None:
        """Метод вызывается контроллером для синхронной отрисовки"""
        self.cursor.setPos(cursor_x)
        self.cursor.show()
        self.slice_highlight.show_slice(
            left=slice_left,
            right=slice_right,
            y_min=self._y_min,
            y_max=self._y_max,
        )

    def hide_cursors(self) -> None:
        self.cursor.hide()
        self.slice_highlight.hide()

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        if self.plotItem.vb.sceneBoundingRect().contains(pos):
            mouse_point = self.plotItem.vb.mapSceneToView(pos)
            self.sigHoverX.emit(float(mouse_point.x()))
        else:
            self.sigHoverLeft.emit()

    def _on_mouse_clicked(self, evt) -> None:
        ev = evt[0]
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self.sceneBoundingRect().contains(ev.scenePos()):
            mouse_point = self.plotItem.vb.mapSceneToView(ev.scenePos())
            self.sigClickedX.emit(float(mouse_point.x()))
            ev.accept()

    def leaveEvent(self, event):
        self.sigHoverLeft.emit()
        super().leaveEvent(event)

class CurveCardWidget(QtWidgets.QWidget):
    """One compact curve panel used in the filmstrip."""

    sigWheelStep = QtCore.pyqtSignal(int)
    sigClicked = QtCore.pyqtSignal()
    sigHovered = QtCore.pyqtSignal()  # добавим hover-сигналы
    sigLeft = QtCore.pyqtSignal()     # (карточка должна сообщать, что на неё навели мышь)

    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent=parent)
        self.compact = compact
        self._active = False

        card_h = 160 if not compact else 150
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setFixedSize(220, card_h)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(3)

        self._title = QtWidgets.QLabel("")
        self._title.setFixedHeight(18)
        self._title.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
        self._title.setStyleSheet(
            "color: rgba(235, 240, 248, 210); font-size: 10px; font-weight: 500;"
        )
        outer.addWidget(self._title)

        self._plot = pg.PlotWidget()
        self._plot.setMenuEnabled(False)
        self._plot.setMouseTracking(True)
        self._plot.showGrid(x=False, y=False, alpha=0.12)
        self._plot.setBackground((18, 20, 26))
        self._plot.setClipToView(True)
        self._plot.showAxis("left", False)
        self._plot.showAxis("bottom", False)
        self._plot.setStyleSheet("PlotWidget { background-color: transparent; border: none; }")
        outer.addWidget(self._plot, 1)
        # обязательно включим mouse tracking:
        self.setMouseTracking(True)
        self._title.setMouseTracking(True)
        self._plot.setMouseTracking(True)
        self._plot.viewport().setMouseTracking(True)
        
        # ловим wheel/click и на заголовке, и на графике (установим eventFilter на себя и детей)
        for obj in (self, self._title, self._plot.viewport()):
            obj.installEventFilter(self)

        self._apply_style(active=False)
        
    def eventFilter(self, obj, event):
        et = event.type()
    
        if et in (
            QtCore.QEvent.Type.Enter,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QEvent.Type.HoverEnter,
            QtCore.QEvent.Type.HoverMove,
        ):
            self.sigHovered.emit()
    
        elif et == QtCore.QEvent.Type.Leave and obj is self:
            self.sigLeft.emit()
    
        if et == QtCore.QEvent.Type.Wheel:
            dy = event.angleDelta().y()
            if dy == 0:
                dy = event.angleDelta().x()
            if dy != 0:
                step = -1 if dy > 0 else 1
                self.sigWheelStep.emit(step)
                event.accept()
                return True
    
        if et == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.sigClicked.emit()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _apply_style(self, active: bool) -> None:
        # одинаковая толщина рамки всегда, меняем только цвет/фон
        if active:
            border = "1px solid rgba(255, 215, 120, 190)"
            bg = "rgba(255,255,255,15)"
        else:
            border = "1px solid rgba(145,160,180,55)"
            bg = "rgba(255,255,255,8)"

        self.setStyleSheet(
            f"""
            QWidget {{
                border: {border};
                border-radius: 4px;
                background-color: {bg};
            }}
            """
        )

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self._apply_style(self._active)

    def _set_title_text(self, title: str) -> None:
        self._title.setToolTip(title)
        fm = self._title.fontMetrics()
        self._title.setText(fm.elidedText(title, QtCore.Qt.TextElideMode.ElideRight, self._title.width()))

    def set_curve(
        self,
        theta: np.ndarray | None,
        y_exp: np.ndarray | None,
        y_fit: np.ndarray | None = None,
        title: str = "",
        active: bool = False,
        fade: float = 1.0,
    ) -> None:
        self._plot.clear()
        self._set_title_text(title)
        self.set_active(active)

        if theta is None or y_exp is None:
            self._plot.plot([0, 1], [0, 0], pen=pg.mkPen((180, 180, 180, int(70 * fade)), width=1))
            return

        theta = np.asarray(theta, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)

        self._plot.plot(
            theta,
            y_exp,
            pen=pg.mkPen((90, 170, 255, int(225 * fade if active else 120 * fade)), width=2.2 if active else 1.4),
        )

        if y_fit is not None:
            y_fit = np.asarray(y_fit, dtype=float)
            self._plot.plot(
                theta,
                y_fit,
                pen=pg.mkPen(
                    (255, 190, 95, int(230 * fade if active else 110 * fade)),
                    width=2.0 if active else 1.2,
                    style=QtCore.Qt.PenStyle.DashLine,
                ),
            )

        # ymin = float(np.nanmin([np.nanmin(y_exp), np.nanmin(y_fit) if y_fit is not None else np.nanmin(y_exp)]))
        # ymax = float(np.nanmax([np.nanmax(y_exp), np.nanmax(y_fit) if y_fit is not None else np.nanmax(y_exp)]))
        # Безопасный расчет границ Y без предупреждений о NaN
        ymin = float(np.nanmin(y_exp))
        ymax = float(np.nanmax(y_exp))
        
        # Если y_fit передан и содержит хотя бы одно валидное число (не NaN)
        if y_fit is not None and not np.all(np.isnan(y_fit)):
            ymin = float(np.nanmin([ymin, np.nanmin(y_fit)]))
            ymax = float(np.nanmax([ymax, np.nanmax(y_fit)]))
    
        pad = max(1.0, 0.12 * (ymax - ymin))

        self._plot.setYRange(ymin - pad, ymax + pad, padding=0)
        self._plot.setXRange(float(theta.min()), float(theta.max()), padding=0.02)

class FilmstripWidget(QtWidgets.QWidget):
    """Horizontal strip with all rocking curves."""
    sigIndexHovered = QtCore.pyqtSignal(int)
    sigIndexClicked = QtCore.pyqtSignal(int)
    sigHoverLeft = QtCore.pyqtSignal()

    def __init__(self, parent=None, radius: int = 2):
        super().__init__(parent=parent)
        self._cards: list[CurveCardWidget] = []
        self._hover_index = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(False)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        card_h = 190
        scroll_h = self._scroll.horizontalScrollBar().sizeHint().height()
        self._scroll.setFixedHeight(card_h + scroll_h + 18)
        self.setMinimumHeight(card_h + scroll_h + 22)        

        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QHBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll)

        self.setStyleSheet("background: transparent;")
        self._scroll.setStyleSheet("""
            QScrollArea { background: transparent; }
            QScrollBar:horizontal { background: rgba(255,255,255,10); height: 14px; margin: 2px 18px 2px 18px; border-radius: 7px; }
            QScrollBar::handle:horizontal { background: rgba(255,255,255,75); min-width: 40px; border-radius: 7px; }
            QScrollBar::handle:horizontal:hover { background: rgba(255,255,255,105); }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; background: none; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
        """)
        
        self._scroll.viewport().setMouseTracking(True)
        self._scroll.viewport().installEventFilter(self)

    def sync_cards(self, n: int) -> None:
        while len(self._cards) < n:
            idx = len(self._cards)
            plot = CurveCardWidget(self._content)
            # Внимание: тут карточки не крутят скролл сами, они только сигналят
            plot.sigClicked.connect(lambda i=idx: self.sigIndexClicked.emit(i))
            plot.sigHovered.connect(lambda i=idx: self._on_card_hovered(i))
            plot.sigLeft.connect(lambda i=idx: self._on_card_left(i))
            plot.sigWheelStep.connect(self._on_card_wheel) # прокрутка ленты колесиком
            self._cards.append(plot)
            self._content_layout.addWidget(plot)            
        while len(self._cards) > n:
            plot = self._cards.pop()
            self._content_layout.removeWidget(plot)
            plot.setParent(None)
            plot.deleteLater()
            
    
    def _on_card_wheel(self, step: int) -> None:
        """Перехватывает прокрутку колесика над карточкой и двигает ленту."""
        bar = self._scroll.horizontalScrollBar()
        # step равен 1 (крутим колесо на себя) или -1 (от себя)
        # 150 — это шаг прокрутки в пикселях. 
        bar.setValue(bar.value() + step * 150)            

    def render_state(self, roc_map: dict[str, Any], highlight_idx: int) -> None:
        """Перерисовываем карточки согласно переданному индексу"""
        theta = np.asarray(roc_map.get("theta_axis", roc_map["s_axis"]), dtype=float)
        intensity = np.asarray(roc_map["intensity"], dtype=float)
        time_s = np.asarray(roc_map["time_s"], dtype=float)
        records = roc_map.get("records", None)

        n = len(time_s)
        for idx, plot in enumerate(self._cards):
            if 0 <= idx < n:
                y_exp = intensity[idx]
                y_fit = None
                title = f"t = {time_s[idx]:.1f} с"

                if records is not None and idx < len(records):
                    record = records[idx]
                    params = record.params
                    
                    # 1. Логика фита (остается нетронутой для отрисовки линии)
                    if "model_name" in roc_map and roc_map["model_name"] in MODEL_SPECS:
                        model = MODEL_SPECS[roc_map["model_name"]]
                        fit_p = tuple(params.get(name, np.nan) for name in model.param_names)
                        y_fit = np.asarray(model.func(theta, *fit_p), dtype=float)
                    
                    # 2. СТРОГИЙ РАСЧЕТ МОМЕНТОВ ИЗ ПЕРВОИСТОЧНИКА КАРТОЧКИ
                    try:
                        # Берем родные массивы конкретного record: theta_axis и counts
                        moments = compute_profile_moments(record.theta_axis, record.counts)
                        centroid_val = moments.centroid
                        area_val = moments.area
                    except Exception:
                        # Если данные повреждены или пустые — защищаем поток от падения
                        centroid_val = np.nan
                        area_val = np.nan

                    # 3. Формирование чистого информативного заголовка
                    title += f" | центр.={centroid_val:.3g}; S={area_val:.3g}"

                active = (idx == highlight_idx)
                fade = 1.0 if active else 0.55
                plot.setVisible(True)
                plot.set_curve(theta, y_exp, y_fit=y_fit, title=title, active=active, fade=fade)
            else:
                plot.setVisible(False)
        self._content.adjustSize()

    
    def _on_card_hovered(self, idx: int) -> None:
        if idx != self._hover_index:
            self._hover_index = idx
            self.sigIndexHovered.emit(idx)
            
    def _on_card_left(self, idx: int) -> None:
        if self._hover_index == idx:
            self._hover_index = None
            self.sigHoverLeft.emit()        

    def scroll_to_index(self, idx: int) -> None:
        if not (0 <= idx < len(self._cards)):
            return
        # Центрирование выполняется сразу, без отложенного таймера.
        self._execute_scroll(idx)        

    def _execute_scroll(self, idx: int) -> None:
        bar = self._scroll.horizontalScrollBar()
        viewport_w = self._scroll.viewport().width()
        card = self._cards[idx]
        
        # Надежная центровка без вылета за края (без дырок)
        card_center_x = card.geometry().center().x()
        target = int(round(card_center_x - viewport_w / 2))
        target = max(bar.minimum(), min(bar.maximum(), target))
        bar.setValue(target)

    def scroll_to_end(self) -> None:
        bar = self._scroll.horizontalScrollBar()
        bar.setValue(bar.maximum())

    def eventFilter(self, obj, event):
        if obj is self._scroll.viewport():
            et = event.type()
            if et in (QtCore.QEvent.Type.MouseMove, QtCore.QEvent.Type.HoverMove):
                p = self._content.mapFrom(self._scroll.viewport(), event.pos())
                if not any(card.geometry().contains(p) for card in self._cards):
                    self._hover_index = None
                    self.sigHoverLeft.emit()
            elif et == QtCore.QEvent.Type.Leave:
                self._hover_index = None
                self.sigHoverLeft.emit()
        return super().eventFilter(obj, event)

class MetricsPlotWidget(pg.GraphicsLayoutWidget):
    """Динамика параметров профиля без подмены отсутствующих данных."""
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setBackground((16, 20, 28))
        self._processed_count = 0
        self._cache_area, self._cache_centroid = [], []
        self._cache_fwhm, self._cache_peak = [], []
        self._plots, self._curves, self.cursors = [], [], []
        specs = [
            ("Ширина FWHM", "#ffbe5f"),
            ("Площадь профиля", "#96e696"),
            ("Положение пика", "#6baeff"),
            ("Центроид", "#c896ff"),
        ]
        for i, (label, color) in enumerate(specs):
            plot = self.addPlot(row=0, col=i)
            plot.setTitle(label, color="#c7d2e0", size="9pt")
            plot.setLabel("bottom", "Время", units="с")
            plot.showGrid(x=True, y=True, alpha=0.16)
            curve = plot.plot(pen=pg.mkPen(color=color, width=1.8))
            cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 220, 120, 210), width=1.5))
            cursor.hide(); plot.addItem(cursor)
            self._plots.append(plot); self._curves.append(curve); self.cursors.append(cursor)
        self.plot_fwhm, self.plot_area, self.plot_theta, self.plot_centroid = self._plots
        self.curve_fwhm, self.curve_area, self.curve_theta, self.curve_centroid = self._curves
        for plot in self._plots[1:]: plot.setXLink(self.plot_fwhm)
        self.set_axis_units("отн. ед. s", False)

    def set_axis_units(self, unit: str, calibrated: bool) -> None:
        axis_unit = unit if calibrated else "отн. ед. s"
        self.plot_fwhm.setLabel("left", "Ширина", units=axis_unit)
        self.plot_area.setLabel("left", "Интенсивность", units=f"отсчёты·{axis_unit}")
        self.plot_theta.setLabel("left", "Угол пика" if calibrated else "Положение пика", units=axis_unit)
        self.plot_centroid.setLabel("left", "Центроид", units=axis_unit)

    def reset(self) -> None:
        self._processed_count = 0
        for values in (self._cache_area, self._cache_centroid, self._cache_fwhm, self._cache_peak): values.clear()
        for curve in self._curves: curve.setData([], [])
        for cursor in self.cursors: cursor.hide()

    def update_data(self, time_s: np.ndarray, records: list) -> None:
        if len(time_s) == 0 or len(records) == 0: return
        if len(records) < self._processed_count: self.reset()
        for record in records[self._processed_count:]:
            x = getattr(record, "theta_axis", None); y = getattr(record, "counts", None)
            if x is None or y is None:
                self._cache_area.append(np.nan); self._cache_centroid.append(np.nan)
                self._cache_fwhm.append(np.nan); self._cache_peak.append(np.nan)
                continue
            try:
                moments = compute_profile_moments(x, y)
                self._cache_area.append(float(moments.area)); self._cache_centroid.append(float(moments.centroid))
            except Exception:
                self._cache_area.append(np.nan); self._cache_centroid.append(np.nan)
            self._cache_fwhm.append(fwhm_half_height(x, y))
            self._cache_peak.append(peak_position(x, y))
        self._processed_count = len(records)
        arrays = [np.asarray(v, dtype=float) for v in (self._cache_fwhm, self._cache_area, self._cache_peak, self._cache_centroid)]
        n = min(len(time_s), *(len(a) for a in arrays)); times = np.asarray(time_s[:n], dtype=float)
        for curve, values in zip(self._curves, arrays):
            values = values[:n]; mask = np.isfinite(times) & np.isfinite(values)
            curve.setData(times[mask], values[mask])
        x0 = float(np.nanmin(times)); x1 = float(np.nanmax(times)) if len(times) > 1 else x0 + 1.0
        self.plot_fwhm.setXRange(x0, x1, padding=0.02)

    def show_cursor(self, x: float) -> None:
        for cursor in self.cursors: cursor.setPos(x); cursor.show()

    def hide_cursor(self) -> None:
        for cursor in self.cursors: cursor.hide()

class LiveMcsAdapter:
    """Адаптер MCS-среза; хранит исходную s и ось θ только при валидной шкале."""
    def __init__(self, data: dict, calibration: dict | None = None, *, experiment_amplitude_mVpp=None):
        self.file_name = data.get("file_name", "")
        self.phase_phi = data.get("phase_phi")
        self.phase_source_file = data.get("phase_source_file")
        self.s_axis = np.asarray(data["s_axis"], dtype=float).copy()
        self._raw_counts = np.asarray(data["intensity"], dtype=float).copy()
        self.params = data.get("params", {})
        self.model_name = data.get("model_name", "pvoigt")
        self.header = {"file_info": {"datetime": data["datetime"]}}
        self.set_calibration(calibration, experiment_amplitude_mVpp=experiment_amplitude_mVpp)

    def set_calibration(self, calibration: dict | None, *, experiment_amplitude_mVpp=None) -> None:
        self.axis_calibration = calibration
        self.experiment_amplitude_mVpp = experiment_amplitude_mVpp
        self.scale_applied = effective_scale(calibration, experiment_amplitude_mVpp)
        self.theta_axis, self.counts = transform_axis(self.s_axis, self._raw_counts, calibration,
            experiment_amplitude_mVpp=experiment_amplitude_mVpp)
        self.axis_calibrated = self.scale_applied is not None
        self.axis_name = "Угол θ" if self.axis_calibrated else "Параметр s"
        self.axis_unit = calibration.get("unit", "угл. с") if self.axis_calibrated and calibration else "отн. ед."
        self.axis_label = f"{self.axis_name}, {self.axis_unit}"
