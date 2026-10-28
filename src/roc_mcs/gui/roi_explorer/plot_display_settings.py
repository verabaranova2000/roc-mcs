"""Настройки отображения двух экранов ROI Explorer."""
from __future__ import annotations
from copy import deepcopy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QFontComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)
from roc_mcs.gui.app_settings import load_settings, save_settings

DEFAULT_PLOT_DISPLAY_SETTINGS = {
    "map": {"font_family": "Segoe UI", "tick_size": 8, "label_size": 9, "colorbar_tick_size": 8,
            "show_colorbar": True, "show_scalebar": False, "scalebar_um": 2000.0, "pixel_size_um": 55.0,
            "scalebar_font_size": 9},
    "curve": {"font_family": "Segoe UI", "tick_size": 8, "label_size": 9, "show_grid": True},
    "workspace": {"map_geometry": None, "curve_geometry": None, "orientation": "horizontal", "viewport_size": None},
}
_SETTINGS_KEY = "roi_explorer_display"

def _safe_int(value, default, low, high):
    try: return max(low, min(high, int(value)))
    except (TypeError, ValueError, OverflowError): return default

def _safe_float(value, default, low, high):
    try:
        number = float(value)
        return max(low, min(high, number)) if number == number and abs(number) != float("inf") else default
    except (TypeError, ValueError, OverflowError): return default

def _safe_bool(value, default):
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)) and value in (0, 1): return bool(value)
    if isinstance(value, str):
        if value.strip().lower() in ("true", "1", "yes", "on"): return True
        if value.strip().lower() in ("false", "0", "no", "off"): return False
    return default

def load_plot_display_settings() -> dict:
    raw = load_settings().get(_SETTINGS_KEY, {})
    result = deepcopy(DEFAULT_PLOT_DISPLAY_SETTINGS)
    if isinstance(raw, dict):
        for section in result:
            values = raw.get(section, {})
            if isinstance(values, dict): result[section].update({k: v for k, v in values.items() if k in result[section]})
    m, c, w = result["map"], result["curve"], result["workspace"]
    for values in (m, c):
        if not isinstance(values.get("font_family"), str) or not values["font_family"].strip(): values["font_family"] = "Segoe UI"
        values["tick_size"] = _safe_int(values.get("tick_size"), 8, 6, 14)
        values["label_size"] = _safe_int(values.get("label_size"), 9, 6, 16)
    m["colorbar_tick_size"] = _safe_int(m.get("colorbar_tick_size"), 8, 6, 14)
    m["scalebar_font_size"] = _safe_int(m.get("scalebar_font_size"), 9, 6, 16)
    m["scalebar_um"] = _safe_float(m.get("scalebar_um"), 2000.0, 10.0, 100000.0)
    m["pixel_size_um"] = _safe_float(m.get("pixel_size_um"), 55.0, 0.001, 100000.0)
    m["show_colorbar"] = _safe_bool(m.get("show_colorbar"), True); m["show_scalebar"] = _safe_bool(m.get("show_scalebar"), False); c["show_grid"] = _safe_bool(c.get("show_grid"), True)
    if w.get("orientation") not in ("horizontal", "vertical"): w["orientation"] = "horizontal"
    for key in ("map_geometry", "curve_geometry"):
        v = w.get(key)
        if not isinstance(v, (list, tuple)) or len(v) != 4:
            w[key] = None
        else:
            try:
                x, y, width, height = map(int, v)
                w[key] = [x, y, max(210, width), max(170, height)]
            except (TypeError, ValueError, OverflowError): w[key] = None
    v = w.get("viewport_size")
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        w["viewport_size"] = None
    else:
        try: w["viewport_size"] = [max(1, int(v[0])), max(1, int(v[1]))]
        except (TypeError, ValueError, OverflowError): w["viewport_size"] = None
    return result

def save_plot_display_settings(values: dict) -> None:
    app = load_settings(); current = load_plot_display_settings()
    for section in current:
        if isinstance(values.get(section), dict): current[section].update(values[section])
    app[_SETTINGS_KEY] = current
    save_settings(app)

def _spin(value, low, high, decimals=0, suffix=""):
    widget = QDoubleSpinBox() if decimals else QSpinBox()
    widget.setRange(low, high); widget.setValue(value); widget.setSuffix(suffix); widget.setKeyboardTracking(False)
    if decimals: widget.setDecimals(decimals); widget.setSingleStep(5.0 if suffix else 0.5)
    widget.setFixedWidth(132 if decimals else 82)
    return widget

class PlotDisplaySettingsDialog(QDialog):
    """Компактный диалог стиля карты и рок-кривых."""
    def __init__(self, settings: dict | None = None, parent=None):
        super().__init__(parent); self.setWindowTitle("Настройки экранов ROI Explorer"); self.setObjectName("plotDisplaySettings")
        self.setMinimumWidth(405); self._defaults = deepcopy(DEFAULT_PLOT_DISPLAY_SETTINGS)
        self._values = deepcopy(settings or load_plot_display_settings())
        root = QVBoxLayout(self); root.setContentsMargins(12, 10, 12, 10); root.setSpacing(8)
        tabs = QTabWidget(self); root.addWidget(tabs)
        self._controls = {"map": {}, "curve": {}}
        for key, title in (("map", "Карта"), ("curve", "ROC-кривые")):
            page = QWidget(tabs); form = QFormLayout(page); form.setContentsMargins(9, 10, 9, 8); form.setHorizontalSpacing(12); form.setVerticalSpacing(8)
            values = self._values[key]; controls = self._controls[key]
            font = QFontComboBox(page); font.setCurrentFont(QFont(str(values["font_family"]))); controls["font_family"] = font; form.addRow("Шрифт", font)
            tick = _spin(int(values["tick_size"]), 6, 14); controls["tick_size"] = tick; form.addRow("Числа осей, pt", tick)
            label = _spin(int(values["label_size"]), 6, 16); controls["label_size"] = label; form.addRow("Подписи осей, pt", label)
            if key == "map":
                cb_tick = _spin(int(values["colorbar_tick_size"]), 6, 14); controls["colorbar_tick_size"] = cb_tick; form.addRow("Числа цветовой шкалы, pt", cb_tick)
                show_cb = QCheckBox("Показывать цветовую шкалу"); show_cb.setChecked(bool(values["show_colorbar"])); controls["show_colorbar"] = show_cb; form.addRow("Цветовая шкала", show_cb)
                show_sb = QCheckBox("Показывать масштабный отрезок"); show_sb.setChecked(bool(values["show_scalebar"])); controls["show_scalebar"] = show_sb; form.addRow("Масштаб", show_sb)
                length = _spin(float(values["scalebar_um"]), 10.0, 100000.0, 1, " µm"); controls["scalebar_um"] = length; form.addRow("Длина отрезка", length)
                pixel = _spin(float(values["pixel_size_um"]), 0.001, 100000.0, 3, " µm/пиксель"); controls["pixel_size_um"] = pixel; form.addRow("Размер пикселя детектора", pixel)
                sb_font = _spin(int(values["scalebar_font_size"]), 6, 16); controls["scalebar_font_size"] = sb_font; form.addRow("Подпись масштаба, pt", sb_font)
                note = QLabel("Координаты карты — в пикселях исходного детектора. Параметры pooling не домножают линейку.", page); note.setWordWrap(True); note.setStyleSheet("color: palette(mid); font-size: 10px;"); form.addRow(note)
            else:
                grid = QCheckBox("Ненавязчивая координатная сетка"); grid.setChecked(bool(values["show_grid"])); controls["show_grid"] = grid; form.addRow("Сетка", grid)
            tabs.addTab(page, title)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, parent=self)
        reset = buttons.addButton("Заводские", QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); reset.clicked.connect(self.reset_defaults)
        root.addWidget(buttons)
        self.setStyleSheet("QDialog#plotDisplaySettings QSpinBox, QDialog#plotDisplaySettings QDoubleSpinBox, QDialog#plotDisplaySettings QComboBox { background: palette(base); color: palette(text); border: 1px solid palette(mid); border-radius: 3px; padding: 3px; } QDialog#plotDisplaySettings QTabWidget::pane { border: 1px solid palette(mid); }")

    def reset_defaults(self):
        self._set_controls(deepcopy(self._defaults))

    def _set_controls(self, values):
        for section, controls in self._controls.items():
            for name, widget in controls.items():
                value = values[section][name]
                if name == "font_family": widget.setCurrentFont(QFont(str(value)))
                elif isinstance(widget, QCheckBox): widget.setChecked(bool(value))
                else: widget.setValue(value)

    def values(self) -> dict:
        result = deepcopy(self._values)
        for section, controls in self._controls.items():
            for name, widget in controls.items():
                result[section][name] = widget.currentFont().family() if name == "font_family" else widget.isChecked() if isinstance(widget, QCheckBox) else widget.value()
        return result
