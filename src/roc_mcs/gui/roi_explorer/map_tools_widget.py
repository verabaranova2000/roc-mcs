"""Панель инструментов карты; вычислительная логика остаётся в MapPlotWidget."""
from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox,
    QDoubleSpinBox, QPushButton, QLabel, QToolButton, QMenu, QWidgetAction, QSpinBox, QDialog, QSizePolicy)
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QLinearGradient, QColor, QActionGroup, QPen, QPolygon, QPalette
from PyQt6.QtCore import QSize, Qt, QRectF, QLineF, pyqtSignal, QPointF, QPoint
from bisect import bisect_left, bisect_right
import math


from roc_mcs.gui.roi_explorer.plot_toolbar_style import (ICON_DIR, COMPACT_PLOT_TOOLBAR_STYLE, SMART_DROPDOWN_INDICATOR_RGBA, SMART_DROPDOWN_INDICATOR_SIZE)
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, apply_popup_theme, theme_colors
from roc_mcs.gui.roi_explorer.colormap_catalog import NAPARI_STYLE_COLORMAPS, colormap_icon
from roc_mcs.gui.roi_explorer.theta_range_preview import complementary_theta_range




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
    theta_display_range_changed = pyqtSignal(int, int)
    theta_range_link_toggled = pyqtSignal(bool)

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
        self._theta_axis_values: list[float] = []
        self._theta_range_indices: tuple[int, int] | None = None
        self._theta_range_available = False
        self._theta_range_link_available = False
        self._build_theta_range_tools(layout)
        self.results_btn = QToolButton(self)
        self.results_btn.setObjectName("mapAnalysisResultsButton")
        self.results_btn.setAutoRaise(True)
        self.results_btn.setIcon(svg_icon(ICON_DIR / "analysis.svg", widget=self.results_btn))
        self.results_btn.setIconSize(QSize(16, 16))
        self.results_btn.setToolTip("Выбрать результат анализа для карты")
        self.results_btn.setAccessibleName("Результат анализа на 2D-карте")
        self.results_menu = QMenu(self.results_btn)
        self.results_menu.setToolTipsVisible(True)
        self.results_btn.setMenu(self.results_menu)
        self.results_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.results_btn.setEnabled(False)
        layout.addWidget(self.results_btn)
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
        self.cmap_menu.setToolTipsVisible(True)
        for name in NAPARI_STYLE_COLORMAPS:
            action = self.cmap_menu.addAction(colormap_icon(name), name); action.setCheckable(True); action.setChecked(name == "inferno")
            action.setToolTip(f"Цветовая карта {name}")
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
        self.results_btn.setIcon(svg_icon(ICON_DIR / "analysis.svg", color=icon_color, widget=self.results_btn))
        apply_popup_theme(self.results_menu, active)
        self.settings_btn.setIcon(svg_icon(ICON_DIR / "setting_icon.svg", color=icon_color, widget=self.settings_btn))
        self.pool_btn.setIcon(svg_icon(ICON_DIR / "pool_stride_dense.svg", color=icon_color, widget=self.pool_btn))
        self.theta_range_btn.setIcon(svg_icon(
            ICON_DIR / "vtk_theta_range_1.svg", color=icon_color,
            widget=self.theta_range_btn, monochrome=False,
        ))
        self.theta_range_swap_btn.setIcon(svg_icon(
            ICON_DIR / "theta_range_swap_1.svg", color=icon_color,
            widget=self.theta_range_swap_btn, monochrome=True,
        ))
        self.theta_range_link_btn.setIcon(svg_icon(
            ICON_DIR / "vtk_theta_link_1.svg", color=icon_color,
            widget=self.theta_range_link_btn, monochrome=True,
        ))
        apply_popup_theme(self.theta_range_menu, active)
        apply_widget_palette(self.theta_range_panel, active)
        apply_widget_palette(self.theta_range_slider, active)
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

    def _build_theta_range_tools(self, toolbar_layout) -> None:
        """Create the 2D projection-only theta-range popup."""
        self.theta_range_btn = self.smart_dropdown_cls(self)
        self.theta_range_btn.setObjectName("mapThetaDisplayRangeButton")
        self.theta_range_btn.setAutoRaise(True)
        self.theta_range_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.theta_range_btn.setFixedSize(26, 22)
        self.theta_range_btn.setIconSize(QSize(17, 17))
        self.theta_range_btn.setIcon(svg_icon(
            ICON_DIR / "vtk_theta_range_1.svg", widget=self.theta_range_btn,
            monochrome=False,
        ))
        self.theta_range_btn.setToolTip(
            "Предпросмотр диапазона θ для mean/max/sum projection"
        )
        self.theta_range_btn.setAccessibleName(
            "Предпросмотр диапазона θ на 2D-карте"
        )
        self.theta_range_menu = QMenu(self.theta_range_btn)
        self.theta_range_menu.setObjectName("MapThetaDisplayRangeMenu")
        self.theta_range_menu.setToolTipsVisible(True)

        panel = QWidget(self.theta_range_menu)
        panel.setObjectName("MapThetaRangePanel")
        panel.setMinimumWidth(258)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title = QLabel("ОБЛАСТЬ ПРОЕКЦИИ ПО θ", panel)
        title.setStyleSheet("font-size:8px;font-weight:700;letter-spacing:0.4px;")
        layout.addWidget(title)

        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(5)
        self.theta_range_value_label = QLabel("θ: —", panel)
        self.theta_range_value_label.setMinimumWidth(0)
        self.theta_range_value_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.theta_range_value_label.setStyleSheet("font-size:10px;font-weight:600;")
        self.theta_range_swap_btn = QToolButton(panel)
        self.theta_range_swap_btn.setObjectName("mapThetaRangeComplementButton")
        self.theta_range_swap_btn.setAutoRaise(True)
        self.theta_range_swap_btn.setFixedSize(22, 22)
        self.theta_range_swap_btn.setIconSize(QSize(15, 15))
        self.theta_range_swap_btn.setIcon(svg_icon(
            ICON_DIR / "theta_range_swap_1.svg", widget=self.theta_range_swap_btn,
            monochrome=True,
        ))
        self.theta_range_swap_btn.setToolTip(
            "Показать противоположный участок θ, оставив общую граничную точку"
        )
        self.theta_range_swap_btn.setAccessibleName(
            "Переключить на дополнительный диапазон θ"
        )
        self.theta_range_swap_btn.setEnabled(False)
        self.theta_range_swap_btn.clicked.connect(self.swap_theta_display_range)
        value_row.addWidget(self.theta_range_value_label, 1)
        value_row.addWidget(self.theta_range_swap_btn, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(value_row)

        self.theta_range_slider = self.range_slider_cls(
            low=0.0, high=1.0, min_val=0.0, max_val=1.0,
            parent=panel, orientation=Qt.Orientation.Horizontal,
            minimum_gap=0.01,
        )
        self.theta_range_slider.setObjectName("mapThetaDisplayRangeSlider")
        self.theta_range_slider.setFixedWidth(238)
        self.theta_range_slider.setToolTip(
            "Ограничить расчёт и отображение mean/max/sum по выбранным отсчётам θ"
        )
        self.theta_range_slider.setEnabled(False)
        self.theta_range_slider.valueChanged.connect(
            self._on_theta_range_slider_changed
        )
        layout.addWidget(
            self.theta_range_slider, 0, Qt.AlignmentFlag.AlignHCenter
        )

        endpoints = QHBoxLayout()
        endpoints.setContentsMargins(0, 0, 0, 0)
        endpoints.setSpacing(4)
        self.theta_range_min_label = QLabel("min: —", panel)
        self.theta_range_max_label = QLabel("max: —", panel)
        self.theta_range_min_label.setStyleSheet("font-size:8px;color:palette(mid);")
        self.theta_range_max_label.setStyleSheet("font-size:8px;color:palette(mid);")
        endpoints.addWidget(self.theta_range_min_label)
        endpoints.addStretch(1)
        endpoints.addWidget(self.theta_range_max_label)
        layout.addLayout(endpoints)

        link_row = QHBoxLayout()
        link_row.setContentsMargins(0, 1, 0, 0)
        link_row.setSpacing(5)
        self.theta_range_link_btn = QToolButton(panel)
        self.theta_range_link_btn.setObjectName("mapThetaRangeLinkButton")
        self.theta_range_link_btn.setAutoRaise(True)
        self.theta_range_link_btn.setCheckable(True)
        self.theta_range_link_btn.setFixedSize(20, 20)
        self.theta_range_link_btn.setIconSize(QSize(14, 14))
        self.theta_range_link_btn.setIcon(svg_icon(
            ICON_DIR / "vtk_theta_link_1.svg", widget=self.theta_range_link_btn,
            monochrome=True,
        ))
        self.theta_range_link_btn.setToolTip(
            "Синхронизировать границы диапазона с 3D-окном; физические θ-оси должны совпадать"
        )
        self.theta_range_link_btn.setAccessibleName(
            "Связать диапазон θ с 3D-окном"
        )
        self.theta_range_link_btn.toggled.connect(
            self.theta_range_link_toggled.emit
        )
        self.theta_range_link_label = QLabel("Связать с 3D", panel)
        self.theta_range_link_label.setToolTip(self.theta_range_link_btn.toolTip())
        link_row.addWidget(self.theta_range_link_btn)
        link_row.addWidget(self.theta_range_link_label, 1)
        layout.addLayout(link_row)

        self.theta_range_reset_button = QPushButton("Полный диапазон", panel)
        self.theta_range_reset_button.setObjectName("MapThetaRangeReset")
        self.theta_range_reset_button.setToolTip(
            "Вернуть отображение всей θ-проекции"
        )
        self.theta_range_reset_button.clicked.connect(
            self.reset_theta_display_range
        )
        layout.addWidget(
            self.theta_range_reset_button, 0, Qt.AlignmentFlag.AlignRight
        )

        action = QWidgetAction(self.theta_range_menu)
        action.setDefaultWidget(panel)
        self.theta_range_menu.addAction(action)
        self.theta_range_panel = panel
        self.theta_range_btn.setMenu(self.theta_range_menu)
        self.theta_range_btn.hide()
        toolbar_layout.addWidget(self.theta_range_btn)

    @property
    def theta_axis_values(self) -> tuple[float, ...]:
        return tuple(self._theta_axis_values)

    @property
    def theta_range_indices(self) -> tuple[int, int] | None:
        return self._theta_range_indices

    def set_theta_range_available(self, available: bool) -> None:
        self._theta_range_available = bool(available)
        self.theta_range_btn.setVisible(self._theta_range_available)
        has_axis = len(self._theta_axis_values) > 2
        enabled = self._theta_range_available and has_axis
        self.theta_range_btn.setEnabled(enabled)
        self.theta_range_btn.setToolTip(
            "Предпросмотр диапазона θ для mean/max/sum projection"
            if enabled else
            "Предпросмотр диапазона θ доступен для mean/max/sum projection "
            "при наличии не менее трёх отсчётов θ"
        )
        self.theta_range_slider.setEnabled(enabled)
        self.theta_range_reset_button.setEnabled(enabled)
        self._update_theta_range_aux_controls()

    def set_theta_axis_values(self, values) -> None:
        """Set ascending physical θ coordinates, preserving a matching range."""
        if values is None:
            self._theta_axis_values = []
            self._theta_range_indices = None
            self.theta_range_slider.setEnabled(False)
            self.theta_range_reset_button.setEnabled(False)
            self.theta_range_value_label.setText("θ: —")
            self.theta_range_min_label.setText("min: —")
            self.theta_range_max_label.setText("max: —")
            self._update_theta_range_aux_controls()
            return

        theta_values = [float(value) for value in values]
        if len(theta_values) < 2:
            raise ValueError("Для предпросмотра диапазона θ нужны минимум два отсчёта.")
        if any(not math.isfinite(value) for value in theta_values):
            raise ValueError("Ось θ содержит нечисловые или бесконечные значения.")
        if any(right <= left for left, right in zip(theta_values, theta_values[1:])):
            raise ValueError("Ось θ должна быть строго возрастающей.")

        same_axis = (
            len(theta_values) == len(self._theta_axis_values)
            and all(
                math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
                for a, b in zip(theta_values, self._theta_axis_values)
            )
        )
        if same_axis and self._theta_range_indices is not None:
            self._update_theta_range_labels(*self._theta_range_indices)
            self.theta_range_slider.setEnabled(
                self._theta_range_available and len(theta_values) > 2
            )
            self.theta_range_reset_button.setEnabled(
                self._theta_range_available and len(theta_values) > 2
            )
            self._update_theta_range_aux_controls()
            return

        self._theta_axis_values = theta_values
        self._replace_theta_range_slider(theta_values)
        self._theta_range_indices = (0, len(theta_values) - 1)
        enabled = self._theta_range_available and len(theta_values) > 2
        self.theta_range_slider.setEnabled(enabled)
        self.theta_range_reset_button.setEnabled(enabled)
        self._update_theta_range_labels(0, len(theta_values) - 1)
        self._update_theta_range_aux_controls()

    def _replace_theta_range_slider(self, values: list[float]) -> None:
        old_slider = self.theta_range_slider
        minimum_gap = min(b - a for a, b in zip(values, values[1:]))
        new_slider = self.range_slider_cls(
            low=float(values[0]), high=float(values[-1]),
            min_val=float(values[0]), max_val=float(values[-1]),
            parent=self.theta_range_panel,
            orientation=Qt.Orientation.Horizontal,
            minimum_gap=float(minimum_gap),
        )
        new_slider.setObjectName("mapThetaDisplayRangeSlider")
        new_slider.setFixedWidth(238)
        new_slider.setToolTip(
            "Границы привязываются к существующим θ-отсчётам"
        )
        new_slider.valueChanged.connect(self._on_theta_range_slider_changed)
        self.theta_range_panel.layout().replaceWidget(old_slider, new_slider)
        old_slider.hide()
        old_slider.deleteLater()
        self.theta_range_slider = new_slider
        apply_widget_palette(self.theta_range_slider, self._theme)

    def _update_theta_range_labels(self, low_index: int, high_index: int) -> None:
        values = self._theta_axis_values
        if not values:
            return
        self.theta_range_value_label.setText(
            f"θ: {values[int(low_index)]:.7g}°  —  {values[int(high_index)]:.7g}°"
        )
        self.theta_range_min_label.setText(f"min: {values[0]:.7g}°")
        self.theta_range_max_label.setText(f"max: {values[-1]:.7g}°")

    def _update_theta_range_aux_controls(self) -> None:
        indices = self._theta_range_indices
        complement = (
            complementary_theta_range(*indices, len(self._theta_axis_values))
            if indices is not None else None
        )
        enabled = bool(
            self._theta_range_available
            and len(self._theta_axis_values) > 2
            and complement is not None
        )
        self.theta_range_swap_btn.setEnabled(enabled)
        self.theta_range_link_btn.setEnabled(
            self._theta_range_available and self._theta_range_link_available
        )

    def set_theta_range_link_available(self, available: bool) -> None:
        self._theta_range_link_available = bool(available)
        can_link = bool(
            self._theta_range_available and self._theta_range_link_available
        )
        self.theta_range_link_btn.setEnabled(can_link)
        tooltip = (
            "Диапазон синхронизирован с 3D-объёмом"
            if can_link else
            "Связь доступна только для mean/max/sum projection и при совпадающих "
            "физических θ-осях 2D- и 3D-окон"
        )
        self.theta_range_link_btn.setToolTip(tooltip)
        self.theta_range_link_label.setToolTip(tooltip)
        if not self._theta_range_link_available:
            self.set_theta_range_link_checked(False)

    def set_theta_range_link_checked(self, checked: bool) -> None:
        self.theta_range_link_btn.blockSignals(True)
        self.theta_range_link_btn.setChecked(bool(checked))
        self.theta_range_link_btn.blockSignals(False)

    def _on_theta_range_slider_changed(
        self, low_value: float, high_value: float
    ) -> None:
        values = self._theta_axis_values
        if len(values) < 2:
            return
        low_index = max(
            0, min(len(values) - 2, bisect_left(values, float(low_value)))
        )
        high_index = max(
            low_index + 1,
            min(len(values) - 1, bisect_right(values, float(high_value)) - 1),
        )
        self.set_theta_display_range_indices(low_index, high_index, emit=True)

    def set_theta_display_range_indices(
        self, low_index: int, high_index: int, *, emit: bool = False
    ) -> None:
        values = self._theta_axis_values
        if len(values) < 2:
            return
        low, high = int(low_index), int(high_index)
        if not (0 <= low < high < len(values)):
            raise ValueError(
                "Индексы предпросмотра θ должны удовлетворять 0 ≤ low < high < N."
            )
        self._theta_range_indices = (low, high)
        self.theta_range_slider.blockSignals(True)
        self.theta_range_slider.setValues(values[low], values[high])
        self.theta_range_slider.blockSignals(False)
        self._update_theta_range_labels(low, high)
        self._update_theta_range_aux_controls()
        if emit:
            self.theta_display_range_changed.emit(low, high)

    def reset_theta_display_range(self) -> None:
        if len(self._theta_axis_values) < 2:
            return
        self.set_theta_display_range_indices(
            0, len(self._theta_axis_values) - 1, emit=True
        )

    def swap_theta_display_range(self) -> None:
        indices = self._theta_range_indices
        if indices is None:
            return
        target = complementary_theta_range(*indices, len(self._theta_axis_values))
        if target is not None:
            self.set_theta_display_range_indices(*target, emit=True)

    def _select_cmap(self, name: str) -> None:
        self.set_current_cmap(name); self.cmap_selected.emit(name)

    def set_current_cmap(self, name: str) -> None:
        for key, action in self._cmap_actions.items(): action.setChecked(key == name)
        self.cmap_btn.setIcon(colormap_icon(name, 24, 14))

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
    def __init__(self, low=1.0, high=99.0, min_val=0.0, max_val=100.0, parent=None, orientation=Qt.Orientation.Horizontal, minimum_gap=0.5):
        super().__init__(parent); self._min=float(min_val); self._max=float(max_val); self._minimum_gap=min(max(0.0,float(minimum_gap)),max(0.0,self._max-self._min)); self._low=float(low); self._high=float(high); self._active_handle=None; self._orientation=orientation
        self._low=max(self._min,min(self._low,self._max-self._minimum_gap)); self._high=max(self._low+self._minimum_gap,min(self._high,self._max))
        if orientation == Qt.Orientation.Vertical: self.setFixedWidth(7); self.setMinimumHeight(80); self.setMaximumWidth(7); self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        else: self.setFixedHeight(18); self.setMinimumWidth(52); self.setMaximumWidth(92)
        self.setCursor(Qt.CursorShape.PointingHandCursor); self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def low(self)->float:return self._low
    def high(self)->float:return self._high
    @property
    def minimum_gap(self)->float:return self._minimum_gap
    def setValues(self,low:float,high:float)->None:
        low=max(self._min,min(float(low),self._max-self._minimum_gap))
        high=max(low+self._minimum_gap,min(float(high),self._max))
        self._low=low;self._high=high;self.update()
    def setLow(self,val:float)->None:self._low=max(self._min,min(float(val),self._high-self._minimum_gap));self.update()
    def setHigh(self,val:float)->None:self._high=min(self._max,max(float(val),self._low+self._minimum_gap));self.update()
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



class ColorbarOptionsButton(QToolButton):
    """Shared compact ellipsis control used below the map and volume colorbars."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._visual_width = 5
        self.setMinimumSize(0, 0)
        self.setText("")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAutoRaise(True)
        self.setMouseTracking(True)

    def set_visual_width(self, width: int) -> None:
        self._visual_width = max(4, int(width))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        hovered = self.underMouse()
        visual_width = min(self._visual_width, max(1, self.width()))
        visual_height = min(8, max(1, self.height()))
        x = (self.width() - visual_width) / 2
        y = (self.height() - visual_height) / 2
        rect = QRectF(x, y, visual_width, visual_height)
        painter.setPen(QPen(
            palette.color(QPalette.ColorRole.Highlight if hovered else QPalette.ColorRole.Mid),
            0.8,
        ))
        painter.setBrush(palette.color(
            QPalette.ColorRole.Midlight if hovered else QPalette.ColorRole.Button
        ))
        painter.drawRoundedRect(rect, 1.5, 1.5)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.ButtonText), 1.2))
        if visual_width >= 9:
            for dx in (-2, 0, 2):
                painter.drawPoint(QPointF(x + visual_width / 2 + dx, y + visual_height / 2))
        else:
            for dy in (-2, 0, 2):
                painter.drawPoint(QPointF(x + visual_width / 2, y + visual_height / 2 + dy))
        painter.end()



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
        painter.setBrush(QColor(*SMART_DROPDOWN_INDICATOR_RGBA))
        
        rect = self.rect()           # 3. Рассчитываем координаты в правом нижнем углу
        size = SMART_DROPDOWN_INDICATOR_SIZE
        
        triangle = QPolygon([
            QPoint(rect.right() - size, rect.bottom()), # Нижняя грань (левее угла)
            QPoint(rect.right(), rect.bottom() - size), # Правая грань (выше угла)
            QPoint(rect.right(), rect.bottom())         # Сам правый-нижний угол
        ])
        
        painter.drawPolygon(triangle)
        painter.end()