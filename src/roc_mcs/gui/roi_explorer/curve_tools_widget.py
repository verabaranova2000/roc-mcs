"""Компактная панель инструментов рок-кривых."""
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QToolButton
from PyQt6.QtCore import QSize, pyqtSignal
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR, COMPACT_PLOT_TOOLBAR_STYLE
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, theme_colors

class CurveToolsWidget(QWidget):
    reset_view_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("CurveToolBar"); self.setFixedHeight(26); self.setStyleSheet(COMPACT_PLOT_TOOLBAR_STYLE)
        layout = QHBoxLayout(self); layout.setContentsMargins(4, 2, 4, 2); layout.setSpacing(5)
        self.home_btn = QToolButton(self); self.home_btn.setAutoRaise(True); self.home_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", widget=self.home_btn)); self.home_btn.setIconSize(QSize(15, 15)); self.home_btn.setToolTip("Сбросить масштаб"); self.home_btn.setAccessibleName("Сбросить масштаб ROC-кривых")
        self.home_btn.clicked.connect(self.reset_view_requested.emit); layout.addWidget(self.home_btn); layout.addStretch(1)
        self.settings_btn = QToolButton(self); self.settings_btn.setAutoRaise(True); self.settings_btn.setIcon(svg_icon(ICON_DIR / "setting_icon.svg", widget=self.settings_btn)); self.settings_btn.setIconSize(QSize(15, 15)); self.settings_btn.setToolTip("Настройки внешнего вида ROC-кривых"); self.settings_btn.setAccessibleName("Настройки ROC-кривых")
        self.settings_btn.clicked.connect(self.settings_requested.emit); layout.addWidget(self.settings_btn)
    def apply_theme(self, _theme: str | None = None) -> None:
        active = apply_widget_palette(self, _theme); color = theme_colors(active)["text"]
        self.home_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", color=color, widget=self.home_btn)); self.settings_btn.setIcon(svg_icon(ICON_DIR / "setting_icon.svg", color=color, widget=self.settings_btn))
