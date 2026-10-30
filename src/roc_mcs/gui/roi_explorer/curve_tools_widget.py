"""Компактная панель инструментов рок-кривых."""
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QToolButton, QMenu
from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QAction
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR, COMPACT_PLOT_TOOLBAR_STYLE
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, theme_colors

class CurveToolsWidget(QWidget):
    reset_view_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    theta_guide_toggled = pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("CurveToolBar"); self.setFixedHeight(26); self.setStyleSheet(COMPACT_PLOT_TOOLBAR_STYLE)
        layout = QHBoxLayout(self); layout.setContentsMargins(4, 2, 4, 2); layout.setSpacing(5)
        self.home_btn = QToolButton(self); self.home_btn.setAutoRaise(True); self.home_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", widget=self.home_btn)); self.home_btn.setIconSize(QSize(15, 15)); self.home_btn.setToolTip("Сбросить масштаб"); self.home_btn.setAccessibleName("Сбросить масштаб ROC-кривых")
        self.home_btn.clicked.connect(self.reset_view_requested.emit); layout.addWidget(self.home_btn)
        self.overlays_btn=QToolButton(self); self.overlays_btn.setAutoRaise(True); self.overlays_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup); self.overlays_btn.setIcon(svg_icon(ICON_DIR/"curve_overlay_3.svg",widget=self.overlays_btn)); self.overlays_btn.setIconSize(QSize(15,15)); self.overlays_btn.setToolTip("Дополнительные объекты на ROC-графике"); self.overlays_btn.setAccessibleName("Дополнительные объекты ROC-графика")
        self.overlays_menu=QMenu(self.overlays_btn); self.theta_guide_action=QAction("Вертикальная линия текущего угла θ",self.overlays_menu); self.theta_guide_action.setCheckable(True); self.theta_guide_action.toggled.connect(self.theta_guide_toggled.emit); self.overlays_menu.addAction(self.theta_guide_action); self.overlays_btn.setMenu(self.overlays_menu); layout.addWidget(self.overlays_btn); layout.addStretch(1)
        self.settings_btn = QToolButton(self); self.settings_btn.setAutoRaise(True); self.settings_btn.setIcon(svg_icon(ICON_DIR / "setting_icon.svg", widget=self.settings_btn)); self.settings_btn.setIconSize(QSize(15, 15)); self.settings_btn.setToolTip("Настройки внешнего вида ROC-кривых"); self.settings_btn.setAccessibleName("Настройки ROC-кривых")
        self.settings_btn.clicked.connect(self.settings_requested.emit); self.settings_btn.hide()
    def apply_theme(self, _theme: str | None = None) -> None:
        active = apply_widget_palette(self, _theme); color = theme_colors(active)["text"]
        from roc_mcs.gui.roi_explorer.theme import apply_popup_theme
        apply_popup_theme(self.overlays_menu,active)
        self.home_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", color=color, widget=self.home_btn)); self.overlays_btn.setIcon(svg_icon(ICON_DIR / "curve_overlay_3.svg", color=color, widget=self.overlays_btn)); self.settings_btn.setIcon(svg_icon(ICON_DIR / "setting_icon.svg", color=color, widget=self.settings_btn))
