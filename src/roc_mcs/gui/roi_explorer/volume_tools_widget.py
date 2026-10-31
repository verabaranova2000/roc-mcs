"""Компактная панель управления интерактивным 3D-экраном."""
from __future__ import annotations
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPalette
from PyQt6.QtWidgets import QApplication, QGridLayout, QMenu, QToolButton, QWidget, QWidgetAction, QHBoxLayout, QDialog, QDoubleSpinBox, QLabel, QPushButton
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR, COMPACT_PLOT_TOOLBAR_STYLE
from roc_mcs.gui.roi_explorer.theme import apply_popup_theme, apply_widget_palette, theme_colors
from roc_mcs.gui.roi_explorer.colormap_catalog import NAPARI_STYLE_COLORMAPS, colormap_icon

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
    contrast_changed = pyqtSignal(float, float)
    contrast_auto_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("VolumeToolBar"); self.setFixedHeight(max(30, self._icon_size()+10)); self._theme = "light"; self._source_key = "full"; self._choices = []; self._rotation_lock = False; self._show_roi_on_full_detector = False
        self.setStyleSheet(COMPACT_PLOT_TOOLBAR_STYLE + "\nQWidget#VolumeToolBar QToolButton { max-height:26px; min-height:20px; border:1px solid transparent; border-radius:2px; background:transparent; padding:0 2px; } QWidget#VolumeToolBar QToolButton:hover { background:palette(midlight); border-color:palette(mid); } QWidget#VolumeToolBar QToolButton:checked { background:palette(highlight); color:palette(highlighted-text); }")
        row = QHBoxLayout(self); row.setContentsMargins(4, 2, 4, 2); row.setSpacing(4)
        self.roi_btn = QToolButton(self); self.roi_btn.setToolTip("Источник объёма: весь детектор или выбранная ROI"); self.roi_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup); self.roi_menu = QMenu(self.roi_btn); self.roi_btn.setMenu(self.roi_menu); row.addWidget(self.roi_btn)
        self.reset_btn = QToolButton(self); self.reset_btn.setToolTip("Вписать объём в окно, сохранив направление камеры"); self.reset_btn.clicked.connect(self.reset_view_requested.emit); row.addWidget(self.reset_btn)
        self.camera_btn = QToolButton(self); self.camera_btn.setToolTip("Настроить ракурс камеры"); self.camera_btn.clicked.connect(self.camera_requested.emit); row.addWidget(self.camera_btn)
        self.orientation_btn=QToolButton(self); self.orientation_btn.setToolTip("Ориентация: щелчок — горизонтальная плоскость XY; стрелка — вид вдоль осей и ограничение вращения"); self.orientation_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup); self.orientation_menu=QMenu(self.orientation_btn); self.orientation_btn.setMenu(self.orientation_menu); self.orientation_btn.clicked.connect(self.align_xy_requested.emit); self._axis_actions={}
        for axis,label in (("x","Вдоль оси X"),("y","Вдоль оси Y"),("z","Вдоль оси Z")):
            action=self.orientation_menu.addAction(label); action.triggered.connect(lambda _=False,a=axis:self.view_axis_requested.emit(a)); self._axis_actions[axis]=action
        self.orientation_menu.addSeparator(); self.rotation_lock_action=self.orientation_menu.addAction("Сохранять вертикаль при вращении"); self.rotation_lock_action.setCheckable(True); self.rotation_lock_action.setChecked(False); self.rotation_lock_action.toggled.connect(self._on_rotation_lock); row.addWidget(self.orientation_btn)
        self.rotate_btn = QToolButton(self); self.rotate_btn.setCheckable(True); self.rotate_btn.setToolTip("Включить или остановить автоматическое вращение"); self.rotate_btn.toggled.connect(self.autorotate_toggled.emit); row.addWidget(self.rotate_btn)
        self.color_btn = QToolButton(self); self.color_btn.setToolTip("Контраст и цветовая карта 3D"); self.color_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup); self.color_menu=QMenu(self.color_btn); self.color_btn.setMenu(self.color_menu); row.addWidget(self.color_btn)
        self._cmap_actions={}
        for cmap in NAPARI_STYLE_COLORMAPS:
            action=self.color_menu.addAction(colormap_icon(cmap),cmap); action.setCheckable(True); action.triggered.connect(lambda _=False,n=cmap:self._choose_colormap(n)); self._cmap_actions[cmap]=action
        self.color_menu.addSeparator(); self.contrast_action=self.color_menu.addAction("Контраст…"); self.contrast_action.triggered.connect(self._open_contrast_dialog)
        self._contrast_dialog=QDialog(self,Qt.WindowType.Tool); self._contrast_dialog.setWindowTitle("Контраст 3D"); grid=QGridLayout(self._contrast_dialog); grid.setContentsMargins(9,8,9,8); grid.setSpacing(5)
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
        name=str(name);
        for key,action in self._cmap_actions.items(): action.setChecked(key==name)
        self.color_btn.setIcon(colormap_icon(name,24,14)); self.color_btn.setIconSize(QSize(24,14))

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
        for button in (self.reset_btn, self.camera_btn, self.orientation_btn, self.rotate_btn, self.info_btn, self.settings_btn): button.setIconSize(QSize(self._icon_size(), self._icon_size()))
        for axis,action in self._axis_actions.items(): action.setIcon(svg_icon(ICON_DIR / f"view_axis_{axis}_1.svg", color=color, widget=self.orientation_btn))
        self.orientation_btn.setIcon(svg_icon(ICON_DIR / "view_orientation.svg", color=color, widget=self.orientation_btn))
        apply_popup_theme(self.orientation_menu,self._theme); apply_popup_theme(self.color_menu,self._theme); apply_popup_theme(self._contrast_dialog,self._theme)
        self._rebuild_roi_menu(); self._update_roi_button()

    def _on_rotation_lock(self, enabled):
        self._rotation_lock=bool(enabled); self.rotation_lock_toggled.emit(self._rotation_lock)

    def set_rotation_lock(self, enabled):
        enabled=bool(enabled); self._rotation_lock=enabled
        self.rotation_lock_action.blockSignals(True); self.rotation_lock_action.setChecked(enabled); self.rotation_lock_action.blockSignals(False)
