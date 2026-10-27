"""Панель инструментов графика кривых."""
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QToolButton
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR, COMPACT_PLOT_TOOLBAR_STYLE
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, theme_colors


class CurveToolsWidget(QWidget):
    reset_view_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CurveToolBar")
        self.setFixedHeight(26)
        self.setStyleSheet(COMPACT_PLOT_TOOLBAR_STYLE)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        self.home_btn = QToolButton(self)
        self.home_btn.setAutoRaise(True)
        self.home_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", widget=self.home_btn))
        self.home_btn.setIconSize(QSize(16, 16))
        self.home_btn.setToolTip("Reset view")
        self.home_btn.setAccessibleName("Reset curve view")
        self.home_btn.clicked.connect(self.reset_view_requested.emit)
        layout.addWidget(self.home_btn)
        layout.addStretch(1)

    def apply_theme(self, _theme: str | None = None) -> None:
        active = apply_widget_palette(self, _theme)
        self.home_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", color=theme_colors(active)["text"], widget=self.home_btn))
