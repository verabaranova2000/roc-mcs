"""Компактная панель управления интерактивным 3D-экраном."""
from __future__ import annotations
from PyQt6.QtCore import QSize, Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPalette, QActionGroup
from PyQt6.QtWidgets import QApplication, QGridLayout, QMenu, QToolButton, QWidget, QWidgetAction, QHBoxLayout, QDialog, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QFormLayout, QLabel, QPushButton
from roc_mcs.gui.roi_explorer.theme import (
    top_level_window_owner, point_sized_font,
    apply_popup_theme, apply_widget_palette, theme_colors,
)
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR, COMPACT_PLOT_TOOLBAR_STYLE
from roc_mcs.gui.roi_explorer.colormap_catalog import NAPARI_STYLE_COLORMAPS, colormap_icon
from roc_mcs.gui.roi_explorer.map_tools_widget import SmartDropdownButton
from roc_mcs.gui.roi_explorer.hex_color_picker import CompactColorPicker, _place_popup

class VolumeToolsWidget(QWidget):
    roi_source_changed = pyqtSignal(object)
    roi_overlay_changed = pyqtSignal(bool)
    reset_view_requested = pyqtSignal()
    camera_requested = pyqtSignal()
    autorotate_toggled = pyqtSignal(bool)
    settings_requested = pyqtSignal()
    info_requested = pyqtSignal()
    align_xy_requested = pyqtSignal()
    view_axis_requested = pyqtSignal(str)
    rotation_lock_toggled = pyqtSignal(bool)
    colormap_changed = pyqtSignal(str)
    opacity_mode_changed = pyqtSignal(bool)
    contrast_changed = pyqtSignal(float, float)
    contrast_auto_requested = pyqtSignal()
    cursor_line_toggled = pyqtSignal(bool)
    peak_display_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent); self.setFont(point_sized_font(parent, fallback=9.0)); self.setObjectName("VolumeToolBar"); self.setFixedHeight(max(30, self._icon_size()+10)); self._theme = "light"; self._source_key = "full"; self._choices = []; self._rotation_lock = False; self._show_roi_on_full_detector = False
        self.setStyleSheet(COMPACT_PLOT_TOOLBAR_STYLE + "\nQWidget#VolumeToolBar QToolButton { max-height:26px; min-height:20px; border:1px solid transparent; border-radius:2px; background:transparent; padding:0 2px; } QWidget#VolumeToolBar QToolButton:hover { background:palette(midlight); border-color:palette(mid); } QWidget#VolumeToolBar QToolButton:checked { background:palette(highlight); color:palette(highlighted-text); }\nQWidget#VolumeToolBar SmartDropdownButton::menu-indicator { image:none; }")
        row = QHBoxLayout(self); row.setContentsMargins(4, 2, 4, 2); row.setSpacing(4)
        self.roi_btn = SmartDropdownButton(self); self.roi_btn.setToolTip("Источник объёма: весь детектор или выбранная ROI"); self.roi_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup); self.roi_menu = QMenu(self.roi_btn); self.roi_btn.setMenu(self.roi_menu); row.addWidget(self.roi_btn)
        self.reset_btn = QToolButton(self); self.reset_btn.setToolTip("Сбросить параллельный перенос и вписать объём в окно, сохранив направление камеры"); self.reset_btn.clicked.connect(self.reset_view_requested.emit); row.addWidget(self.reset_btn)
        self.camera_btn = QToolButton(self); self.camera_btn.setToolTip("Настроить ракурс камеры"); self.camera_btn.clicked.connect(self.camera_requested.emit); row.addWidget(self.camera_btn)
        self.orientation_btn=SmartDropdownButton(self); self.orientation_btn.setToolTip("Ориентация: щелчок — горизонтальная плоскость XY; стрелка — вид вдоль осей и ограничение вращения"); self.orientation_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup); self.orientation_menu=QMenu(self.orientation_btn); self.orientation_btn.setMenu(self.orientation_menu); self.orientation_btn.clicked.connect(self.align_xy_requested.emit); self._axis_actions={}
        for axis,label in (("x","Вдоль оси X"),("y","Вдоль оси Y"),("z","Вдоль оси Z")):
            action=self.orientation_menu.addAction(label); action.triggered.connect(lambda _=False,a=axis:self.view_axis_requested.emit(a)); self._axis_actions[axis]=action
        self.orientation_menu.addSeparator(); self.rotation_lock_action=self.orientation_menu.addAction("Сохранять вертикаль при вращении"); self.rotation_lock_action.setCheckable(True); self.rotation_lock_action.setChecked(False); self.rotation_lock_action.toggled.connect(self._on_rotation_lock); row.addWidget(self.orientation_btn)
        self.rotate_btn = QToolButton(self); self.rotate_btn.setCheckable(True); self.rotate_btn.setToolTip("Включить или остановить автоматическое вращение"); self.rotate_btn.toggled.connect(self.autorotate_toggled.emit); row.addWidget(self.rotate_btn)
        self.overlays_btn = SmartDropdownButton(self); self.overlays_btn.setAutoRaise(True); self.overlays_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.overlays_btn.setIcon(svg_icon(ICON_DIR / "curve_overlay_3.svg", widget=self.overlays_btn)); self.overlays_btn.setIconSize(QSize(15, 15))
        self.overlays_btn.setToolTip("Дополнительные объекты на 3D-сцене")
        self.overlays_menu = QMenu(self.overlays_btn)
        self.cursor_line_action = self.overlays_menu.addAction("Линия положения курсора 2D ↔ 3D")
        self.cursor_line_action.setCheckable(True)
        self.cursor_line_action.setToolTip("Показывать вертикальную линию по координатам курсора на карте или при наведении на 3D-объём")
        self.cursor_line_action.toggled.connect(self.cursor_line_toggled.emit)
        self.overlays_btn.setMenu(self.overlays_menu); row.addWidget(self.overlays_btn)
        self.peaks_btn = QToolButton(self)
        self.peaks_btn.setAutoRaise(True)
        self.peaks_btn.setToolTip("Настройки отображения центров пиков в 3D")
        self.peaks_btn.clicked.connect(self._open_peak_display_dialog)
        row.addWidget(self.peaks_btn)
        self._peak_display = {
            "show": True, "size": 3.0, "opacity": 1.0,
            "color_mode": "branch", "color": "#ffb347", "branch_colors": {},
        }
        self._peak_branch_ids = []
        self._peak_color_picker = None
        self._peak_display_dialog = self._build_peak_display_dialog()
        self.color_btn = SmartDropdownButton(self); self.color_btn.setToolTip("Контраст и цветовая карта 3D"); self.color_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup); self.color_menu=QMenu(self.color_btn); self.color_btn.setMenu(self.color_menu); row.addWidget(self.color_btn)
        self._cmap_actions={}
        self._cmap_action_group=QActionGroup(self.color_menu); self._cmap_action_group.setExclusive(True)
        for cmap in NAPARI_STYLE_COLORMAPS:
            action=self.color_menu.addAction(colormap_icon(cmap),cmap); action.setCheckable(True); self._cmap_action_group.addAction(action); action.triggered.connect(lambda _=False,n=cmap:self._choose_colormap(n)); self._cmap_actions[cmap]=action
        self.color_menu.addSeparator()
        self.soft_opacity_action = self.color_menu.addAction("Мягкое затухание слабых сигналов")
        self.soft_opacity_action.setCheckable(True)
        self.soft_opacity_action.setChecked(False)
        self.soft_opacity_action.setToolTip(
            "Плавно уменьшает прозрачность слабых значений внутри текущего диапазона контраста. "
            "Не меняет массив данных и не добавляет новую жёсткую отсечку; "
            "действующие нижний и верхний пороги контраста сохраняются. "
            "Отключение возвращает прежнюю равномерную прозрачность."
        )
        self.soft_opacity_action.toggled.connect(self.opacity_mode_changed.emit)
        self.color_menu.addSeparator()
        self.contrast_action=self.color_menu.addAction("Контраст…"); self.contrast_action.triggered.connect(self._open_contrast_dialog)
        self._contrast_dialog=QDialog(top_level_window_owner(self),Qt.WindowType.Tool); self._contrast_dialog.setFont(point_sized_font(self, fallback=9.0)); self._contrast_dialog.setWindowTitle("Контраст 3D"); self.destroyed.connect(self._contrast_dialog.deleteLater); grid=QGridLayout(self._contrast_dialog); grid.setContentsMargins(9,8,9,8); grid.setSpacing(5)
        self._contrast_low=QDoubleSpinBox(self._contrast_dialog); self._contrast_high=QDoubleSpinBox(self._contrast_dialog)
        for spin,value in ((self._contrast_low,16.0),(self._contrast_high,95.0)):
            spin.setRange(0.0,100.0); spin.setDecimals(1); spin.setSuffix(" %"); spin.setKeyboardTracking(False); spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons); spin.setFixedWidth(76)
        grid.addWidget(QLabel("Мин.",self._contrast_dialog),0,0);grid.addWidget(self._contrast_low,0,1);grid.addWidget(QLabel("Макс.",self._contrast_dialog),0,2);grid.addWidget(self._contrast_high,0,3)
        auto=QPushButton("Авто",self._contrast_dialog); auto.clicked.connect(self._contrast_auto); grid.addWidget(auto,1,0,1,2)
        apply_btn=QPushButton("Применить",self._contrast_dialog); apply_btn.clicked.connect(self._apply_contrast); grid.addWidget(apply_btn,1,2,1,2)
        row.addStretch(1)
        self.info_btn = QToolButton(self); self.info_btn.setToolTip("Справка по управлению 3D"); self.info_btn.clicked.connect(self.info_requested.emit); row.addWidget(self.info_btn)
        self.settings_btn = QToolButton(self); self.settings_btn.setToolTip("Настройки всех экранов ROI Explorer"); self.settings_btn.clicked.connect(self.settings_requested.emit); self.settings_btn.hide()
        self.set_roi_choices([]); self.apply_theme("light")

    def _build_peak_display_dialog(self):
        dialog = QDialog(top_level_window_owner(self), Qt.WindowType.Tool)
        dialog.setFont(point_sized_font(self, fallback=9.0))
        self.destroyed.connect(dialog.deleteLater)
        dialog.setWindowTitle("Центры пиков в 3D")
        dialog.setMinimumWidth(250)
        form = QFormLayout(dialog)
        form.setContentsMargins(9, 8, 9, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        show = QCheckBox("Показывать центры пиков", dialog)
        show.setChecked(bool(self._peak_display["show"]))
        size = QDoubleSpinBox(dialog)
        size.setRange(0.5, 12.0); size.setDecimals(1); size.setSingleStep(0.5); size.setSuffix(" px")
        size.setValue(float(self._peak_display["size"])); size.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        opacity = QDoubleSpinBox(dialog)
        opacity.setRange(0.05, 1.0); opacity.setDecimals(2); opacity.setSingleStep(0.05)
        opacity.setValue(float(self._peak_display["opacity"])); opacity.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        mode = QComboBox(dialog)
        mode.addItem("Один цвет", "single")
        mode.addItem("Цвет по branch_id", "branch")
        mode.setCurrentIndex(mode.findData(self._peak_display["color_mode"]))
        branch = QComboBox(dialog)
        branch_color = QToolButton(dialog)
        branch_color.setFixedSize(29, 23)
        branch_color.setToolTip("Выбрать цвет для выбранной ветви")
        global_color = QToolButton(dialog)
        global_color.setFixedSize(29, 23)
        global_color.setToolTip("Выбрать общий цвет центров пиков")
        form.addRow("", show)
        form.addRow("Размер маркера:", size)
        form.addRow("Непрозрачность:", opacity)
        form.addRow("Окраска:", mode)
        form.addRow("Ветвь:", branch)
        form.addRow("Цвет ветви:", branch_color)
        form.addRow("Общий цвет:", global_color)
        hint = QLabel(
            "Цвета ветвей назначаются независимо и сохраняются в настройках текущего окна 3D. "
            "Наведение на маркер показывает peak_id и branch_id."
        , dialog)
        hint.setWordWrap(True)
        form.addRow(hint)
        close = QPushButton("Закрыть", dialog)
        close.clicked.connect(dialog.hide)
        form.addRow(close)
        self._peak_show_check = show
        self._peak_size_spin = size
        self._peak_opacity_spin = opacity
        self._peak_mode_combo = mode
        self._peak_branch_combo = branch
        self._peak_branch_color_button = branch_color
        self._peak_global_color_button = global_color
        show.toggled.connect(lambda value: self._update_peak_display(show=value))
        size.valueChanged.connect(lambda value: self._update_peak_display(size=float(value)))
        opacity.valueChanged.connect(lambda value: self._update_peak_display(opacity=float(value)))
        mode.currentIndexChanged.connect(lambda *_: self._update_peak_display(color_mode=mode.currentData()))
        branch.currentIndexChanged.connect(self._update_peak_branch_swatch)
        branch_color.clicked.connect(self._open_branch_color_picker)
        global_color.clicked.connect(self._open_global_peak_color_picker)
        self._update_peak_branch_swatch()
        self._update_peak_color_buttons()
        return dialog

    @property
    def peak_display_settings(self):
        return {
            **self._peak_display,
            "branch_colors": dict(self._peak_display.get("branch_colors", {})),
        }

    def set_peak_branch_ids(self, branch_ids):
        normalized = set()
        for value in branch_ids or []:
            try:
                import math
                numeric = float(value)
                if math.isfinite(numeric) and numeric.is_integer():
                    normalized.add(int(numeric))
            except (TypeError, ValueError, OverflowError):
                continue
        ids = sorted(normalized)
        current = self._peak_branch_combo.currentData() if hasattr(self, "_peak_branch_combo") else None
        self._peak_branch_ids = ids
        if not hasattr(self, "_peak_branch_combo"):
            return
        self._peak_branch_combo.blockSignals(True)
        self._peak_branch_combo.clear()
        for branch_id in ids:
            self._peak_branch_combo.addItem(f"Ветвь {branch_id}", str(branch_id))
        if current is not None:
            idx = self._peak_branch_combo.findData(str(current))
            if idx >= 0:
                self._peak_branch_combo.setCurrentIndex(idx)
        self._peak_branch_combo.blockSignals(False)
        has_branches = bool(ids)
        self._peak_mode_combo.setEnabled(has_branches)
        self._peak_branch_combo.setEnabled(has_branches and self._peak_display.get("color_mode") == "branch")
        self._peak_branch_color_button.setEnabled(has_branches and self._peak_display.get("color_mode") == "branch")
        self._update_peak_branch_swatch()

    def _update_peak_display(self, **updates):
        self._peak_display.update(updates)
        if not self._peak_branch_ids and self._peak_display.get("color_mode") == "branch":
            self._peak_display["color_mode"] = "single"
            self._peak_mode_combo.blockSignals(True)
            self._peak_mode_combo.setCurrentIndex(self._peak_mode_combo.findData("single"))
            self._peak_mode_combo.blockSignals(False)
        branch_mode = bool(self._peak_branch_ids) and self._peak_display.get("color_mode") == "branch"
        self._peak_branch_combo.setEnabled(branch_mode)
        self._peak_branch_color_button.setEnabled(branch_mode)
        self._update_peak_color_buttons()
        self.peak_display_changed.emit(self.peak_display_settings)

    def _update_peak_color_buttons(self):
        color = str(self._peak_display.get("color", "#ffb347"))
        self._peak_global_color_button.setStyleSheet(
            f"QToolButton {{ background:{color}; border:1px solid palette(mid); border-radius:3px; }}"
        )
        self._update_peak_branch_swatch()

    def _selected_peak_branch(self):
        value = self._peak_branch_combo.currentData()
        return None if value is None else str(value)

    def _update_peak_branch_swatch(self, *_args):
        if not hasattr(self, "_peak_branch_color_button"):
            return
        branch_id = self._selected_peak_branch()
        color = self._peak_display.get("branch_colors", {}).get(branch_id, self._default_branch_color(branch_id))
        self._peak_branch_color_button.setStyleSheet(
            f"QToolButton {{ background:{color}; border:1px solid palette(mid); border-radius:3px; }}"
        )

    @staticmethod
    def _default_branch_color(branch_id):
        if branch_id is None:
            return "#ffb347"
        # Stable default; the user may override any branch independently.
        import colorsys
        hue = (int(branch_id) * 0.6180339887498949) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.68, 0.92)
        return "#%02x%02x%02x" % tuple(round(channel * 255) for channel in rgb)

    def _open_peak_color_picker(self, initial_color, anchor, callback):
        if self._peak_color_picker is not None:
            self._peak_color_picker.close()
        picker = CompactColorPicker(initial_color, self)
        self._peak_color_picker = picker
        picker.destroyed.connect(lambda *_args, current=picker: self._clear_peak_color_picker(current))
        picker.color_selected.connect(callback)
        pos = anchor.mapToGlobal(QPoint(0, anchor.height() + 3))
        _place_popup(picker, pos)
        picker.show(); picker.raise_(); picker.activateWindow()

    def _clear_peak_color_picker(self, picker) -> None:
        # A closing older popup must not clear the reference to a newer picker.
        if self._peak_color_picker is picker:
            self._peak_color_picker = None

    def _open_branch_color_picker(self):
        branch_id = self._selected_peak_branch()
        if branch_id is None:
            return
        color = self._peak_display.get("branch_colors", {}).get(branch_id, self._default_branch_color(branch_id))
        self._open_peak_color_picker(color, self._peak_branch_color_button,
                                     lambda selected, bid=branch_id: self._set_branch_color(bid, selected))

    def _open_global_peak_color_picker(self):
        self._open_peak_color_picker(
            self._peak_display.get("color", "#ffb347"), self._peak_global_color_button,
            lambda selected: self._update_peak_display(color=selected),
        )

    def _set_branch_color(self, branch_id, color):
        colors = dict(self._peak_display.get("branch_colors", {}))
        colors[str(branch_id)] = str(color)
        self._update_peak_display(branch_colors=colors)
        self._update_peak_branch_swatch()

    def _open_peak_display_dialog(self):
        apply_popup_theme(self._peak_display_dialog, self._theme)
        self._peak_display_dialog.adjustSize()
        self._peak_display_dialog.show()
        self._peak_display_dialog.raise_()
        self._peak_display_dialog.activateWindow()

    @staticmethod
    def _icon_size():
        app = QApplication.instance(); value = app.property("rocMcsIconSize") if app is not None else None
        try: return max(12, min(20, int(value)))
        except (TypeError, ValueError, OverflowError): return 16

    def _swatch_icon(self, color, widget=None):
        size = 20; pixmap = QPixmap(size, size); pixmap.fill(Qt.GlobalColor.transparent); painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = (widget or self).palette(); painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1)); painter.setBrush(QColor(color)); painter.drawRoundedRect(3, 3, size-6, size-6, 2, 2); painter.end(); return QIcon(pixmap)

    def set_roi_choices(self, choices, selected=None):
        self._choices = list(choices or []); valid = {"full", *(c[0] for c in self._choices)}
        if selected in valid: self._source_key = selected
        elif self._source_key not in valid: self._source_key = "full"
        self._rebuild_roi_menu(); self._update_roi_button()

    def current_source(self): return self._source_key

    def _rebuild_roi_menu(self):
        self.roi_menu.clear(); panel = QWidget(self.roi_menu); panel.setObjectName("VolumeRoiPalette"); grid = QGridLayout(panel); grid.setContentsMargins(4, 4, 4, 4); grid.setSpacing(2)
        entries = [("full", "Full detector", None)] + self._choices
        for i, (key, label, color) in enumerate(entries):
            button = QToolButton(panel); button.setFixedSize(max(23, self._icon_size()+4), max(23, self._icon_size()+4)); button.setToolTip("Весь детектор" if key == "full" else str(label)); button.setCheckable(True); button.setChecked(key == self._source_key); button.setCursor(Qt.CursorShape.PointingHandCursor)
            icon_color = theme_colors(self._theme)["text"] if key == "full" else (QColor(color).name() if QColor(str(color)).isValid() else theme_colors(self._theme)["text"])
            button.setIcon(svg_icon(ICON_DIR / "volume_source_2.svg", color=icon_color, widget=button) if key == "full" else self._swatch_icon(icon_color, button)); button.setIconSize(QSize(self._icon_size(), self._icon_size())); button.clicked.connect(lambda _=False, k=key: self._select_source(k)); grid.addWidget(button, i // 8, i % 8)
        action = QWidgetAction(self.roi_menu); action.setDefaultWidget(panel); self.roi_menu.addAction(action)
        self.roi_menu.addSeparator()
        self.overlay_action=self.roi_menu.addAction("ROI-объём на полной XY-проекции")
        self.overlay_action.setCheckable(True); self.overlay_action.setChecked(self._show_roi_on_full_detector)
        self.overlay_action.setEnabled(bool(self._choices)); self.overlay_action.setToolTip("Показывать выбранную ROI как 3D-облако на полной XY-проекции детектора")
        self.overlay_action.toggled.connect(self._on_overlay_toggled)
        apply_popup_theme(self.roi_menu, self._theme); apply_widget_palette(panel, self._theme)

    def _on_overlay_toggled(self, enabled):
        self._show_roi_on_full_detector=bool(enabled); self.roi_overlay_changed.emit(self._show_roi_on_full_detector)

    def show_roi_on_full_detector(self): return self._show_roi_on_full_detector

    def _select_source(self, key):
        if key == self._source_key: self.roi_menu.close(); return
        self._source_key = key; self.roi_menu.close(); self._rebuild_roi_menu(); self._update_roi_button(); self.roi_source_changed.emit(key)

    def _update_roi_button(self):
        item = next((x for x in self._choices if x[0] == self._source_key), None)
        if item is None:
            self.roi_btn.setIcon(svg_icon(ICON_DIR / "volume_source_2.svg", color=theme_colors(self._theme)["text"], widget=self.roi_btn)); self.roi_btn.setToolTip("Источник объёма: весь детектор")
        else:
            color = QColor(str(item[2])); icon_color = color.name() if color.isValid() else theme_colors(self._theme)["text"]
            self.roi_btn.setIcon(self._swatch_icon(icon_color, self.roi_btn)); self.roi_btn.setToolTip(f"Источник объёма: {item[1]}")
        self.roi_btn.setIconSize(QSize(self._icon_size(), self._icon_size()))

    def _choose_colormap(self,name):
        self.set_colormap(name); self.color_menu.close(); self.colormap_changed.emit(str(name))

    def set_colormap(self,name):
        name=str(name)
        if name not in self._cmap_actions:
            name=next(iter(self._cmap_actions), "magma")
        for key,action in self._cmap_actions.items(): action.setChecked(key==name)
        self.color_btn.setIcon(colormap_icon(name,24,14)); self.color_btn.setIconSize(QSize(24,14))
        self.color_btn.setToolTip(f"Контраст и цветовая карта 3D · текущая палитра: {name}")
        self.color_btn.setAccessibleName(f"Цветовая карта 3D: {name}")

    def set_soft_opacity(self, enabled):
        enabled = bool(enabled)
        self.soft_opacity_action.blockSignals(True)
        self.soft_opacity_action.setChecked(enabled)
        self.soft_opacity_action.blockSignals(False)

    def set_contrast(self,low,high):
        low=max(0.0,min(99.5,float(low))); high=max(low+0.5,min(100.0,float(high)))
        for spin,value in ((self._contrast_low,low),(self._contrast_high,high)):
            spin.blockSignals(True);spin.setValue(value);spin.blockSignals(False)

    def _open_contrast_dialog(self):
        apply_popup_theme(self._contrast_dialog,self._theme);self._contrast_dialog.adjustSize();self._contrast_dialog.show();self._contrast_dialog.raise_();self._contrast_dialog.activateWindow()

    def _apply_contrast(self):
        low=float(self._contrast_low.value());high=float(self._contrast_high.value())
        if high<=low:high=min(100.0,low+0.5);self._contrast_high.setValue(high)
        self.contrast_changed.emit(low,high)

    def _contrast_auto(self):
        self.set_contrast(16.0,95.0);self.contrast_auto_requested.emit()

    def apply_theme(self, theme=None):
        self._theme = apply_widget_palette(self, theme); self.setFixedHeight(max(30, self._icon_size()+10)); c = theme_colors(self._theme); color = c["text"]
        self.reset_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", color=color, widget=self.reset_btn)); self.camera_btn.setIcon(svg_icon(ICON_DIR / "camera_1.svg", color=color, widget=self.camera_btn)); self.rotate_btn.setIcon(svg_icon(ICON_DIR / "refresh_variant_4_minimal.svg", color=color, widget=self.rotate_btn)); self.info_btn.setIcon(svg_icon(ICON_DIR / "info_circle_icon.svg", color=color, widget=self.info_btn)); self.settings_btn.setIcon(svg_icon(ICON_DIR / "setting_icon.svg", color=color, widget=self.settings_btn))
        self.peaks_btn.setIcon(svg_icon(ICON_DIR / "vtk_scene_objects.svg", color=color, widget=self.peaks_btn))
        for button in (self.reset_btn, self.camera_btn, self.orientation_btn, self.rotate_btn, self.overlays_btn, self.peaks_btn, self.info_btn, self.settings_btn): button.setIconSize(QSize(self._icon_size(), self._icon_size()))
        for axis,action in self._axis_actions.items(): action.setIcon(svg_icon(ICON_DIR / f"view_axis_{axis}_1.svg", color=color, widget=self.orientation_btn))
        self.orientation_btn.setIcon(svg_icon(ICON_DIR / "view_orientation.svg", color=color, widget=self.orientation_btn))
        self.overlays_btn.setIcon(svg_icon(ICON_DIR / "curve_overlay_3.svg", color=color, widget=self.overlays_btn))
        apply_popup_theme(self.orientation_menu,self._theme); apply_popup_theme(self.color_menu,self._theme); apply_popup_theme(self.overlays_menu,self._theme); apply_popup_theme(self._contrast_dialog,self._theme); apply_popup_theme(self._peak_display_dialog,self._theme)
        self._rebuild_roi_menu(); self._update_roi_button()

    def _on_rotation_lock(self, enabled):
        self._rotation_lock=bool(enabled); self.rotation_lock_toggled.emit(self._rotation_lock)

    def set_rotation_lock(self, enabled):
        enabled=bool(enabled); self._rotation_lock=enabled
        self.rotation_lock_action.blockSignals(True); self.rotation_lock_action.setChecked(enabled); self.rotation_lock_action.blockSignals(False)
