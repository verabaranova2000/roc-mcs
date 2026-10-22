"""Панель инструментов карты; вычислительная логика остаётся в MapPlotWidget."""
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox,
    QDoubleSpinBox, QPushButton, QLabel, QToolButton, QMenu, QWidgetAction, QSpinBox)
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QLinearGradient, QColor, QActionGroup, QPen, QPolygon
from PyQt6.QtCore import QSize, Qt, QRectF, QLineF, pyqtSignal, QPointF, QPoint


from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR, COMPACT_PLOT_TOOLBAR_STYLE




def pooling_preview_pixmap(size: int, stride: int) -> QPixmap:
    """Компактная схема окна pooling и шага сдвига."""
    size, stride = max(1, int(size)), max(1, int(stride))
    width, height, side = 88, 66, 27.0
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#fbfcfe"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor("#d7dee8"), 1))
    painter.setBrush(QColor("#fbfcfe"))
    painter.drawRoundedRect(QRectF(0.5, 0.5, width - 1, height - 1), 4, 4)
    cell = side / max(size, stride)
    window = cell * size; offset = cell * stride
    extent = window + offset
    x0, y0 = (width - extent) / 2, (height - extent) / 2
    windows = ((QRectF(x0, y0 + offset, window, window), QColor(66, 133, 190, 105), QColor("#356d9d")),
               (QRectF(x0 + offset, y0, window, window), QColor(232, 160, 65, 115), QColor("#ad7029")))
    for rect, fill, edge in windows:
        painter.setPen(QPen(edge, 1.1))
        painter.setBrush(fill)
        painter.drawRect(rect)
    divisions = min(size, 9)
    if divisions > 1:
        for rect, _, edge in windows:
            painter.setPen(QPen(edge, 0.65))
            for i in range(1, divisions):
                pos = rect.left() + rect.width() * i / divisions
                painter.drawLine(QLineF(pos, rect.top(), pos, rect.bottom()))
                pos = rect.top() + rect.height() * i / divisions
                painter.drawLine(QLineF(rect.left(), pos, rect.right(), pos))
    painter.end()
    return pixmap


class MapToolsWidget(QWidget):
    """Панель инструментов карты; сообщает наружу о действиях пользователя."""
    reset_view_requested = pyqtSignal()
    pin_roi_requested = pyqtSignal()
    cmap_selected = pyqtSignal(str)
    roi_toggled = pyqtSignal(bool)
    roi_tool_clicked = pyqtSignal(object)
    pool_size_changed = pyqtSignal(int)
    pool_stride_changed = pyqtSignal(int)
    pool_apply_requested = pyqtSignal()

    def __init__(self, roi_tools_config: dict, current_roi_type: str, smart_dropdown_cls, range_slider_cls, parent=None):
        super().__init__(parent)
        self.roi_tools_config = roi_tools_config
        self.current_roi_type = current_roi_type
        self.smart_dropdown_cls = smart_dropdown_cls
        self.range_slider_cls = range_slider_cls
        self.setObjectName("MapToolBar")
        self.setFixedHeight(26)
        self.setStyleSheet(COMPACT_PLOT_TOOLBAR_STYLE)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        self.home_btn = QToolButton(self)
        self.home_btn.setIcon(QIcon(str(ICON_DIR / "zoom_out_map.svg")))
        self.home_btn.setToolTip("Reset view")
        self.home_btn.clicked.connect(self.reset_view_requested.emit)
        layout.addWidget(self.home_btn)
        self._build_roi_tools(layout)
        self._build_pooling_tools(layout)
        self.pin_roi_btn = QToolButton(self)
        self.pin_roi_btn.setIcon(QIcon(str(ICON_DIR / "push_pin.svg")))
        self.pin_roi_btn.setIconSize(QSize(16, 16))
        self.pin_roi_btn.setToolTip("Pin preview ROI")
        self.pin_roi_btn.clicked.connect(self.pin_roi_requested.emit)
        layout.addWidget(self.pin_roi_btn)
        layout.addStretch(1)
        self.cmap_btn = QToolButton(self)
        self.cmap_btn.setIcon(self._create_cmap_icon())
        self.cmap_btn.setToolTip("Select colormap")
        self.cmap_menu = QMenu(self.cmap_btn)
        for name in ("gray", "inferno", "magma", "viridis", "cividis", "plasma"):
            self.cmap_menu.addAction(name).triggered.connect(lambda checked=False, cmap=name: self.cmap_selected.emit(cmap))
        self.cmap_btn.setMenu(self.cmap_menu)
        self.cmap_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        layout.addWidget(self.cmap_btn)
        self.auto_contrast_btn = QToolButton(self)
        self.auto_contrast_btn.setText("A")
        self.auto_contrast_btn.setToolTip("Auto Contrast (1.0% - 99.0%)")
        self.auto_contrast_btn.setStyleSheet("font-weight: bold; font-family: monospace; color: #000080;")
        layout.addWidget(self.auto_contrast_btn)
        self.low_pct = QDoubleSpinBox(self)
        self.high_pct = QDoubleSpinBox(self)
        for spin, value in ((self.low_pct, 1.0), (self.high_pct, 99.0)):
            spin.setRange(0, 100)
            spin.setDecimals(1)
            spin.setValue(value)
            spin.setSuffix("%")
            spin.setFixedWidth(46)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range_slider = range_slider_cls(low=1.0, high=99.0)
        layout.addWidget(self.low_pct)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.high_pct)
        self._wire_contrast_controls()

    def _wire_contrast_controls(self) -> None:
        def sync_values(low, high):
            self.low_pct.blockSignals(True)
            self.high_pct.blockSignals(True)
            self.low_pct.setValue(round(low, 1))
            self.high_pct.setValue(round(high, 1))
            self.low_pct.blockSignals(False)
            self.high_pct.blockSignals(False)
            self.low_pct.valueChanged.emit(self.low_pct.value())
        self.range_slider.valueChanged.connect(sync_values)
        self.low_pct.valueChanged.connect(self.range_slider.setLow)
        self.high_pct.valueChanged.connect(self.range_slider.setHigh)

    @staticmethod
    def _create_cmap_icon() -> QIcon:
        pixmap = QPixmap(16, 16)
        painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, 16, 0)
        gradient.setColorAt(0.0, QColor("#000004"))
        gradient.setColorAt(0.5, QColor("#bc3754")); gradient.setColorAt(1.0, QColor("#fcffa4"))
        painter.fillRect(0, 0, 16, 16, gradient)
        painter.end()
        return QIcon(pixmap)

    def _build_roi_tools(self, layout) -> None:
        self.roi_btn = self.smart_dropdown_cls(self)
        self.roi_btn.setIconSize(QSize(16, 16))
        self.roi_btn.setCheckable(True)
        self.roi_btn.setChecked(True)
        self.roi_btn.setToolTip("ROI drawing tools; when off, map clicks do not select pixels")
        self.roi_btn.setPopupMode(QToolButton.ToolButtonPopupMode.DelayedPopup)
        self.roi_btn.setStyleSheet("SmartDropdownButton::menu-indicator { image: none; }")
        self.roi_menu = QMenu(self.roi_btn)
        self.roi_menu.setWindowFlags(self.roi_menu.windowFlags() | Qt.WindowType.NoDropShadowWindowHint)
        flyout = QWidget()
        flyout.setStyleSheet("QWidget { background: transparent; }")
        row = QHBoxLayout(flyout)
        row.setContentsMargins(1, 1, 1, 1)
        row.setSpacing(1)
        self.roi_action_group = QActionGroup(self)
        self.roi_action_group.setExclusive(True)
        self.roi_tool_buttons = {}
        for roi_id, info in self.roi_tools_config.items():
            button = QToolButton(flyout)
            button.setIcon(QIcon(info["icon"]))
            button.setToolTip(info["name"])
            button.setAutoRaise(True)
            button.setCheckable(True)
            button.setProperty("roi_id", roi_id)
            if roi_id == self.current_roi_type:
                button.setChecked(True)
                self.roi_btn.setIcon(QIcon(info["icon"]))
                self.roi_btn.setToolTip(info["name"])
            button.clicked.connect(lambda checked=False, b=button: self.roi_tool_clicked.emit(b))
            row.addWidget(button)
            self.roi_tool_buttons[roi_id] = button
        action = QWidgetAction(self)
        action.setDefaultWidget(flyout)
        self.roi_menu.addAction(action)
        self.roi_btn.setMenu(self.roi_menu)
        self.roi_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.roi_btn.customContextMenuRequested.connect(lambda pos: self.roi_menu.exec(self.roi_btn.mapToGlobal(pos)))
        self.roi_btn.toggled.connect(self.roi_toggled.emit)
        layout.addWidget(self.roi_btn)

    def _build_pooling_tools(self, layout) -> None:
        self.pool_btn = self.smart_dropdown_cls(self)
        self.pool_btn.setIcon(QIcon(str(ICON_DIR / "pool_stride_dense.svg")))
        self.pool_btn.setIconSize(QSize(16, 16))
        self.pool_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.pool_btn.setAccessibleName("Spatial pooling")
        self.pool_btn.setToolTip("Spatial view: Raw (window 1×1, stride 1)")
        self.pool_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.pool_btn.setStyleSheet("SmartDropdownButton::menu-indicator { image: none; }")
        self.pool_menu = QMenu(self.pool_btn)
        self.pool_menu.setWindowFlags(self.pool_menu.windowFlags() | Qt.WindowType.NoDropShadowWindowHint)
        panel = QWidget(self.pool_menu)
        panel.setObjectName("PoolingPanel")
        panel.setStyleSheet("""
            QWidget#PoolingPanel { background: #fff; }
            QWidget#PoolingPanel QLabel { color: #343a40; font-size: 10px; }
            QWidget#PoolingPanel QSpinBox { min-height: 19px; max-height: 19px; padding: 0 3px; border: 1px solid #cbd2dc; border-radius: 3px; background: #fff; font-size: 10px; }
            QWidget#PoolingPanel QPushButton { min-height: 20px; max-height: 20px; padding: 0 7px; border: 1px solid #cbd2dc; border-radius: 3px; background: #f5f7fa; font-size: 10px; }
            QWidget#PoolingPanel QPushButton:hover { background: #e6f2ff; border-color: #8aaed0; }
        """)
        outer = QHBoxLayout(panel)
        outer.setContentsMargins(7, 5, 7, 5)
        outer.setSpacing(7)
        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)
        fields = QGridLayout()
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setHorizontalSpacing(5)
        fields.setVerticalSpacing(3)
        self.pool_size_spin = QSpinBox(panel)
        self.pool_size_spin.setRange(1, 101)
        self.pool_size_spin.setSingleStep(2)
        self.pool_size_spin.setValue(1)
        self.pool_size_spin.setFixedWidth(44)
        self.pool_size_spin.setFixedHeight(19)
        self.pool_size_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.pool_size_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pool_size_spin.setKeyboardTracking(False)
        self.pool_size_spin.valueChanged.connect(self.pool_size_changed.emit)
        self.pool_stride_spin = QSpinBox(panel)
        self.pool_stride_spin.setRange(1, 256)
        self.pool_stride_spin.setValue(1)
        self.pool_stride_spin.setFixedWidth(44)
        self.pool_stride_spin.setFixedHeight(19)
        self.pool_stride_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.pool_stride_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pool_stride_spin.setKeyboardTracking(False)
        self.pool_stride_spin.valueChanged.connect(self.pool_stride_changed.emit)
        fields.addWidget(QLabel("Window", panel), 0, 0)
        fields.addWidget(self.pool_size_spin, 0, 1)
        fields.addWidget(QLabel("Stride", panel), 1, 0)
        fields.addWidget(self.pool_stride_spin, 1, 1)
        controls.addLayout(fields)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(4)
        hint = QLabel("Sum counts", panel)
        hint.setToolTip("Суммируются counts в каждом окне; ось θ не меняется.")
        footer.addWidget(hint)
        footer.addStretch(1)
        self.pool_apply_btn = QPushButton("Apply", panel)
        self.pool_apply_btn.setFixedWidth(47)
        self.pool_apply_btn.clicked.connect(self.pool_apply_requested.emit)
        footer.addWidget(self.pool_apply_btn)
        controls.addLayout(footer)
        outer.addLayout(controls)
        self.pool_preview_label = QLabel(panel)
        self.pool_preview_label.setFixedSize(88, 66)
        self.pool_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pool_preview_label.setAccessibleName("Pooling window and stride preview")
        self.pool_preview_label.setToolTip("Схема двух окон: сетка показывает исходные пиксели, диагональный сдвиг — stride, смешанный цвет — пересечение. Для size > 9 сетка показана условно.")
        self.pool_preview_label.setPixmap(pooling_preview_pixmap(1, 1))
        outer.addWidget(self.pool_preview_label)
        action = QWidgetAction(self.pool_menu)
        action.setDefaultWidget(panel)
        self.pool_menu.addAction(action)
        self.pool_btn.setMenu(self.pool_menu)
        layout.addWidget(self.pool_btn)




# ========================================================
# Components (возможно, переедут в общие для всех GUI)
# ========================================================


class QRangeSlider(QWidget):
    """
   Двухсторонний слайдер (Range Slider) для контраста.
    Рисует тонкую синюю полоску с двумя круглыми маркерами 
        [ Low <===◯======◯===> High ]
    Занимает минимум места и идеально вписывается в тулбары научного ПО.
    """
    valueChanged = pyqtSignal(float, float)  # (low, high)

    def __init__(self, low=1.0, high=99.0, min_val=0.0, max_val=100.0, parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._low = low
        self._high = high
        self._active_handle = None

        self.setFixedHeight(18)
        self.setMinimumWidth(80)
        self.setMaximumWidth(140)

    def low(self) -> float: return self._low
    def high(self) -> float: return self._high

    def setLow(self, val: float) -> None:
        self._low = max(self._min, min(val, self._high - 0.5))
        self.update()

    def setHigh(self, val: float) -> None:
        self._high = min(self._max, max(val, self._low + 0.5))
        self.update()

    def _val_to_x(self, val: float) -> float:
        w = self.width() - 12
        return 6 + ((val - self._min) / (self._max - self._min)) * w

    def _x_to_val(self, x: float) -> float:
        w = self.width() - 12
        if w <= 0: return self._min
        rel = max(0.0, min(1.0, (x - 6) / w))
        return self._min + rel * (self._max - self._min)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        y = self.height() / 2.0
        w = float(self.width())

        # 1. Серый трек (подложка)
        painter.setPen(QPen(QColor("#cccccc"), 3))
        painter.drawLine(6, int(y), int(w - 6), int(y))

        # 2. Синий активный диапазон
        x1 = self._val_to_x(self._low)
        x2 = self._val_to_x(self._high)
        painter.setPen(QPen(QColor("#005fb8"), 3))  # Строгий научный синий
        painter.drawLine(int(x1), int(y), int(x2), int(y))

        # 3. Ползунки (круглые маркеры)
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QPointF(x1, y), 4.5, 4.5)
        painter.drawEllipse(QPointF(x2, y), 4.5, 4.5)

    def mousePressEvent(self, event) -> None:
        x = event.position().x()
        x1 = self._val_to_x(self._low)
        x2 = self._val_to_x(self._high)
        # Определяем, к какому маркеру ближе кликнули
        if abs(x - x1) < abs(x - x2):
            self._active_handle = 'low'
            self.setLow(self._x_to_val(x))
        else:
            self._active_handle = 'high'
            self.setHigh(self._x_to_val(x))
        self.valueChanged.emit(self._low, self._high)

    def mouseMoveEvent(self, event) -> None:
        if not self._active_handle: return
        x = event.position().x()
        val = self._x_to_val(x)
        if self._active_handle == 'low':
            self.setLow(val)
        else:
            self.setHigh(val)
        self.valueChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, event) -> None:
        self._active_handle = None






class SmartDropdownButton(QToolButton):
    """
    Кастомная кнопка с треугольным индикатором выпадающего меню.
    Этот класс просто наследует стандартную кнопку, 
    но после её отрисовки аккуратно дорисовывает чернильно-синий уголок.
    """
    def paintEvent(self, event):
        super().paintEvent(event)    # 1. Рисуем стандартную кнопку и иконку

        painter = QPainter(self)     # 2. Поверх рисуем наш векторный треугольник
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#000080"))  # Мой фирменный чернильно-синий
        
        rect = self.rect()           # 3. Рассчитываем координаты четко в правом нижнем углу
        size = 6                     # Размер уголка в пикселях (можно поменять на 5 или 7)
        
        triangle = QPolygon([
            QPoint(rect.right() - size, rect.bottom()), # Нижняя грань (левее угла)
            QPoint(rect.right(), rect.bottom() - size), # Правая грань (выше угла)
            QPoint(rect.right(), rect.bottom())         # Сам правый-нижний угол
        ])
        
        painter.drawPolygon(triangle)
        painter.end()