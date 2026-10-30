"""Панель инструментов карты; вычислительная логика остаётся в MapPlotWidget."""
from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox,
    QDoubleSpinBox, QPushButton, QLabel, QToolButton, QMenu, QWidgetAction, QSpinBox, QDialog, QSizePolicy)
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QLinearGradient, QColor, QActionGroup, QPen, QPolygon, QPalette
from PyQt6.QtCore import QSize, Qt, QRectF, QLineF, pyqtSignal, QPointF, QPoint


from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR, COMPACT_PLOT_TOOLBAR_STYLE
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, apply_popup_theme, theme_colors




def pooling_preview_pixmap(size: int, stride: int, palette: QPalette | None = None) -> QPixmap:
    """Компактная схема pooling с фоном из текущей палитры."""
    size, stride = max(1, int(size)), max(1, int(stride))
    palette = palette or QApplication.palette()
    background = palette.color(QPalette.ColorRole.Base); border = palette.color(QPalette.ColorRole.Mid)
    width, height, side = 88, 66, 27.0
    pixmap = QPixmap(width, height)
    pixmap.fill(background)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(border, 1))
    painter.setBrush(background)
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
    settings_requested = pyqtSignal()

    def __init__(self, roi_tools_config: dict, current_roi_type: str, smart_dropdown_cls, range_slider_cls, parent=None):
        super().__init__(parent)
        self.roi_tools_config = roi_tools_config
        self.current_roi_type = current_roi_type; self._theme = "light"
        self.smart_dropdown_cls = smart_dropdown_cls
        self.range_slider_cls = range_slider_cls
        self.setObjectName("MapToolBar")
        self.setFixedHeight(26)
        self.setStyleSheet(COMPACT_PLOT_TOOLBAR_STYLE + "\nQWidget#MapToolBar QToolButton::menu-indicator { image:none; } QWidget#MapToolBar QMenu::indicator { background:palette(base); border:1px solid palette(mid); } QWidget#MapToolBar QMenu::indicator:checked { background:palette(highlight); }")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        self.home_btn = QToolButton(self)
        self.home_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", widget=self.home_btn))
        self.home_btn.setToolTip("Reset view")
        self.home_btn.clicked.connect(self.reset_view_requested.emit)
        layout.addWidget(self.home_btn)
        self._build_roi_tools(layout)
        self._build_pooling_tools(layout)
        self.pin_roi_btn = QToolButton(self)
        self.pin_roi_btn.setIcon(svg_icon(ICON_DIR / "push_pin.svg", widget=self.pin_roi_btn))
        self.pin_roi_btn.setIconSize(QSize(16, 16))
        self.pin_roi_btn.setToolTip("Pin preview ROI")
        self.pin_roi_btn.clicked.connect(self.pin_roi_requested.emit)
        layout.addWidget(self.pin_roi_btn)
        layout.addStretch(1)
        # Контраст вынесен из узкой панели в отдельный немодальный диалог.
        self.contrast_dialog = QDialog(self, Qt.WindowType.Tool); self.contrast_dialog.setObjectName("mapContrastDialog")
        self.contrast_dialog.setWindowTitle("Контраст и цветовая карта"); self.contrast_dialog.setMinimumWidth(310); self.contrast_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        dialog_layout = QGridLayout(self.contrast_dialog); dialog_layout.setContentsMargins(10, 9, 10, 9); dialog_layout.setHorizontalSpacing(7); dialog_layout.setVerticalSpacing(6)
        self.low_pct = QDoubleSpinBox(self.contrast_dialog); self.high_pct = QDoubleSpinBox(self.contrast_dialog)
        for spin, value in ((self.low_pct, 1.0), (self.high_pct, 99.0)):
            spin.setRange(0, 100); spin.setDecimals(1); spin.setValue(value); spin.setSuffix("%"); spin.setFixedWidth(72); spin.setFixedHeight(24)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons); spin.setAlignment(Qt.AlignmentFlag.AlignCenter); spin.setKeyboardTracking(False)
            spin.setStyleSheet("QDoubleSpinBox{background:palette(base);color:palette(text);border:1px solid palette(mid);border-radius:3px;padding:1px 4px;selection-background-color:palette(highlight);selection-color:palette(highlighted-text);}")
        dialog_layout.addWidget(QLabel("Мин.", self.contrast_dialog), 0, 0); dialog_layout.addWidget(self.low_pct, 0, 1)
        dialog_layout.addWidget(QLabel("Макс.", self.contrast_dialog), 0, 2); dialog_layout.addWidget(self.high_pct, 0, 3)
        self.auto_contrast_btn = QToolButton(self.contrast_dialog); self.auto_contrast_btn.setText("A"); self.auto_contrast_btn.setToolTip("Автоконтраст: 1–99-й процентили")
        self.auto_contrast_btn.setStyleSheet("font-weight:bold;color:palette(highlight);min-width:28px;min-height:24px;")
        dialog_layout.addWidget(self.auto_contrast_btn, 1, 0)
        self.cmap_btn = QToolButton(self.contrast_dialog); self.cmap_btn.setIcon(self._create_cmap_icon()); self.cmap_btn.setText("Цветовая карта")
        self.cmap_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon); self.cmap_btn.setToolTip("Выбрать цветовую карту")
        self.cmap_menu = QMenu(self.cmap_btn); self._cmap_actions = {}
        for name in ("gray", "inferno", "magma", "viridis", "cividis", "plasma"):
            action = self.cmap_menu.addAction(name); action.setCheckable(True); action.setChecked(name == "inferno")
            action.triggered.connect(lambda checked=False, cmap=name: self._select_cmap(cmap)); self._cmap_actions[name] = action
        self.cmap_btn.setMenu(self.cmap_menu); self.cmap_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        dialog_layout.addWidget(self.cmap_btn, 1, 1, 1, 3)
        self.range_slider = None
        self.settings_btn = QToolButton(self); self.settings_btn.setAutoRaise(True); self.settings_btn.setIcon(svg_icon(ICON_DIR / "setting_icon.svg", widget=self.settings_btn)); self.settings_btn.setIconSize(QSize(15, 15))
        self.settings_btn.setToolTip("Настройки внешнего вида карты"); self.settings_btn.setAccessibleName("Настройки карты")
        self.settings_btn.clicked.connect(self.settings_requested.emit); self.settings_btn.hide()

    def apply_theme(self, _theme: str | None = None) -> None:
        """Согласует панель, flyout-меню и SVG-иконки с активной темой."""
        active = apply_widget_palette(self, _theme); self._theme = active; c = theme_colors(active)
        icon_color = c["text"]
        self.home_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", color=icon_color, widget=self.home_btn))
        self.pin_roi_btn.setIcon(svg_icon(ICON_DIR / "push_pin.svg", color=icon_color, widget=self.pin_roi_btn))
        self.settings_btn.setIcon(svg_icon(ICON_DIR / "setting_icon.svg", color=icon_color, widget=self.settings_btn))
        self.pool_btn.setIcon(svg_icon(ICON_DIR / "pool_stride_dense.svg", color=icon_color, widget=self.pool_btn))
        for roi_id, button in self.roi_tool_buttons.items():
            icon_path = self.roi_tools_config[roi_id]["icon"]
            button.setIcon(svg_icon(icon_path, color=icon_color, widget=button))
            if roi_id == self.current_roi_type: self.roi_btn.setIcon(svg_icon(icon_path, color=icon_color, widget=self.roi_btn))
        apply_popup_theme(self.cmap_menu, active); apply_popup_theme(self.roi_menu, active); apply_popup_theme(self.pool_menu, active); apply_popup_theme(self.contrast_dialog, active)
        apply_widget_palette(self.roi_flyout, active)
        apply_widget_palette(self.pool_panel, active)
        self.pool_panel.setAutoFillBackground(True)
        self.pool_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.pool_panel.setPalette(self.palette())
        pc = {"window": c["window"], "text": c["text"], "base": c["base"], "button": c["button"], "button_text": c["button_text"], "border": c["border"], "hover": c["hover"], "highlight": c["highlight"], "highlight_text": c["highlight_text"]}
        self.pool_panel.setStyleSheet(self._pool_panel_base_style + f"""
            QWidget#PoolingPanel {{ background:{pc['window']}; color:{pc['text']}; border:1px solid {pc['border']}; }}
            QWidget#PoolingPanel QWidget {{ background:{pc['window']}; color:{pc['text']}; }}
            QWidget#PoolingPanel QLabel {{ background:transparent; color:{pc['text']}; }}
            QWidget#PoolingPanel QSpinBox, QWidget#PoolingPanel QDoubleSpinBox, QWidget#PoolingPanel QComboBox,
            QWidget#PoolingPanel QLineEdit {{ background:{pc['base']}; color:{pc['text']}; border:1px solid {pc['border']}; selection-background-color:{pc['highlight']}; selection-color:{pc['highlight_text']}; }}
            QWidget#PoolingPanel QPushButton, QWidget#PoolingPanel QToolButton {{ background:{pc['button']}; color:{pc['button_text']}; border:1px solid {pc['border']}; }}
            QWidget#PoolingPanel QPushButton:hover, QWidget#PoolingPanel QToolButton:hover {{ background:{pc['hover']}; border-color:{pc['highlight']}; }}
        """)

    def _select_cmap(self, name: str) -> None:
        self.set_current_cmap(name); self.cmap_selected.emit(name)

    def set_current_cmap(self, name: str) -> None:
        for key, action in self._cmap_actions.items(): action.setChecked(key == name)

    def bind_range_slider(self, slider) -> None:
        """Связывает поля контраста с вертикальным ползунком на полотне карты."""
        self.range_slider = slider
        def sync_values(low, high):
            self.low_pct.blockSignals(True); self.high_pct.blockSignals(True)
            self.low_pct.setValue(round(low, 1)); self.high_pct.setValue(round(high, 1))
            self.low_pct.blockSignals(False); self.high_pct.blockSignals(False)
            self.low_pct.valueChanged.emit(self.low_pct.value())
        slider.valueChanged.connect(sync_values); self.low_pct.valueChanged.connect(slider.setLow); self.high_pct.valueChanged.connect(slider.setHigh)
    def open_contrast_dialog(self, global_pos=None) -> None:
        self.contrast_dialog.adjustSize()
        if global_pos is not None and not self.contrast_dialog.isVisible():
            screen=QApplication.screenAt(global_pos) or QApplication.primaryScreen(); pos=QPoint(global_pos)
            if screen is not None:
                area=screen.availableGeometry(); pos.setX(max(area.left(),min(pos.x(),area.right()-self.contrast_dialog.width()+1))); pos.setY(max(area.top(),min(pos.y(),area.bottom()-self.contrast_dialog.height()+1)))
            self.contrast_dialog.move(pos)
        apply_popup_theme(self.contrast_dialog, self._theme)
        self.contrast_dialog.show(); self.contrast_dialog.raise_(); self.contrast_dialog.activateWindow()

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
        self.roi_flyout = flyout
        flyout.setStyleSheet("QWidget { background: transparent; }")
        row = QHBoxLayout(flyout)
        row.setContentsMargins(1, 1, 1, 1)
        row.setSpacing(1)
        self.roi_action_group = QActionGroup(self)
        self.roi_action_group.setExclusive(True)
        self.roi_tool_buttons = {}
        for roi_id, info in self.roi_tools_config.items():
            button = QToolButton(flyout)
            button.setIcon(svg_icon(info["icon"], widget=button))
            button.setToolTip(info["name"])
            button.setAutoRaise(True)
            button.setCheckable(True)
            button.setProperty("roi_id", roi_id)
            if roi_id == self.current_roi_type:
                button.setChecked(True)
                self.roi_btn.setIcon(svg_icon(info["icon"], widget=self.roi_btn))
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
        self.pool_btn.setIcon(svg_icon(ICON_DIR / "pool_stride_dense.svg", widget=self.pool_btn))
        self.pool_btn.setIconSize(QSize(16, 16))
        self.pool_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.pool_btn.setAccessibleName("Spatial pooling")
        self.pool_btn.setToolTip("Spatial view: Raw (window 1×1, stride 1)")
        self.pool_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.pool_btn.setStyleSheet("SmartDropdownButton::menu-indicator { image: none; }")
        self.pool_menu = QMenu(self.pool_btn)
        self.pool_menu.setWindowFlags(self.pool_menu.windowFlags() | Qt.WindowType.NoDropShadowWindowHint)
        panel = QWidget(self.pool_menu)
        self.pool_panel = panel
        panel.setObjectName("PoolingPanel")
        panel.setStyleSheet("""
            QWidget#PoolingPanel { background: palette(base); color: palette(text); }
            QWidget#PoolingPanel QLabel { color: palette(text); font-size: 10px; }
            QWidget#PoolingPanel QSpinBox { min-height: 19px; max-height: 19px; padding: 0 3px; border: 1px solid palette(mid); border-radius: 3px; background: palette(base); color: palette(text); font-size: 10px; }
            QWidget#PoolingPanel QPushButton { min-height: 20px; max-height: 20px; padding: 0 7px; border: 1px solid palette(mid); border-radius: 3px; background: palette(button); color: palette(button-text); font-size: 10px; }
            QWidget#PoolingPanel QPushButton:hover { background: palette(midlight); border-color: palette(highlight); }
        """)
        self._pool_panel_base_style = panel.styleSheet()
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
        self.pool_size_spin.setStyleSheet("QSpinBox { background: palette(base); color: palette(text); border: 1px solid palette(mid); border-radius: 3px; padding: 0 2px; selection-background-color: palette(highlight); selection-color: palette(highlighted-text); }")
        self.pool_size_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.pool_size_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pool_size_spin.setKeyboardTracking(False)
        self.pool_size_spin.valueChanged.connect(self.pool_size_changed.emit)
        self.pool_stride_spin = QSpinBox(panel)
        self.pool_stride_spin.setRange(1, 256)
        self.pool_stride_spin.setValue(1)
        self.pool_stride_spin.setFixedWidth(44)
        self.pool_stride_spin.setFixedHeight(19)
        self.pool_stride_spin.setStyleSheet("QSpinBox { background: palette(base); color: palette(text); border: 1px solid palette(mid); border-radius: 3px; padding: 0 2px; selection-background-color: palette(highlight); selection-color: palette(highlighted-text); }")
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
        self.pool_preview_label.setPixmap(pooling_preview_pixmap(1, 1, self.palette()))
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
    """Компактный двухсторонний ползунок диапазона для горизонтальной или вертикальной установки."""
    valueChanged = pyqtSignal(float, float)
    def __init__(self, low=1.0, high=99.0, min_val=0.0, max_val=100.0, parent=None, orientation=Qt.Orientation.Horizontal):
        super().__init__(parent); self._min=float(min_val); self._max=float(max_val); self._low=float(low); self._high=float(high); self._active_handle=None; self._orientation=orientation
        if orientation == Qt.Orientation.Vertical: self.setFixedWidth(7); self.setMinimumHeight(80); self.setMaximumWidth(7); self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        else: self.setFixedHeight(18); self.setMinimumWidth(52); self.setMaximumWidth(92)
        self.setCursor(Qt.CursorShape.PointingHandCursor); self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def low(self)->float:return self._low
    def high(self)->float:return self._high
    def setLow(self,val:float)->None:self._low=max(self._min,min(float(val),self._high-0.5));self.update()
    def setHigh(self,val:float)->None:self._high=min(self._max,max(float(val),self._low+0.5));self.update()
    def _val_to_pos(self,val:float)->float:
        if self._orientation==Qt.Orientation.Vertical:
            h=max(1,self.height()-12);return self.height()-6-((val-self._min)/(self._max-self._min))*h
        w=max(1,self.width()-12);return 6+((val-self._min)/(self._max-self._min))*w
    def _pos_to_val(self,pos:float)->float:
        if self._orientation==Qt.Orientation.Vertical:
            h=max(1,self.height()-12);rel=max(0.0,min(1.0,(self.height()-6-pos)/h))
        else:
            w=max(1,self.width()-12);rel=max(0.0,min(1.0,(pos-6)/w))
        return self._min+rel*(self._max-self._min)
    def _coordinate(self,event)->float:return event.position().y() if self._orientation==Qt.Orientation.Vertical else event.position().x()
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing,True);base=self.palette().color(QPalette.ColorRole.Mid);accent=self.palette().color(QPalette.ColorRole.Highlight);edge=self.palette().color(QPalette.ColorRole.ButtonText);fill=self.palette().color(QPalette.ColorRole.Base)
        if self._orientation==Qt.Orientation.Vertical:
            x=self.width()/2; a=self._val_to_pos(self._high);b=self._val_to_pos(self._low);p.setPen(QPen(base,3,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap));p.drawLine(QLineF(x,6,x,self.height()-6));p.setPen(QPen(accent,3,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap));p.drawLine(QLineF(x,a,x,b))
            for y in (a,b):p.setPen(QPen(edge,0.8));p.setBrush(fill);p.drawRoundedRect(QRectF(x-3.5,y-2.0,7,4.0),1.1,1.1);p.setPen(QPen(accent,0.8));p.drawLine(QLineF(x-2.2,y,x+2.2,y))
        else:
            y=self.height()/2;x1=self._val_to_pos(self._low);x2=self._val_to_pos(self._high);p.setPen(QPen(base,3));p.drawLine(QLineF(6,y,self.width()-6,y));p.setPen(QPen(accent,3));p.drawLine(QLineF(x1,y,x2,y))
            for x in (x1,x2):p.setPen(QPen(edge,1));p.setBrush(fill);p.drawEllipse(QPointF(x,y),4.5,4.5)
        p.end()
    def mousePressEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton:return
        pos=self._coordinate(event);a=self._val_to_pos(self._low);b=self._val_to_pos(self._high)
        self._active_handle='low' if abs(pos-a)<=abs(pos-b) else 'high';self._set_position(pos);self.setFocus();event.accept()
    def mouseMoveEvent(self,event):
        if self._active_handle:self._set_position(self._coordinate(event));event.accept()
    def _set_position(self,pos):
        value=self._pos_to_val(pos);self.setLow(value) if self._active_handle=='low' else self.setHigh(value);self.valueChanged.emit(self._low,self._high)
    def mouseReleaseEvent(self,event):self._active_handle=None;event.accept()



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
        painter.setBrush(self.palette().color(QPalette.ColorRole.Highlight))
        
        rect = self.rect()           # 3. Рассчитываем координаты четко в правом нижнем углу
        size = 6                     # Размер уголка в пикселях (можно поменять на 5 или 7)
        
        triangle = QPolygon([
            QPoint(rect.right() - size, rect.bottom()), # Нижняя грань (левее угла)
            QPoint(rect.right(), rect.bottom() - size), # Правая грань (выше угла)
            QPoint(rect.right(), rect.bottom())         # Сам правый-нижний угол
        ])
        
        painter.drawPolygon(triangle)
        painter.end()