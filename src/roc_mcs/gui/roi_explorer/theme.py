"""Единая палитра и оформление ROI Explorer для светлой и тёмной тем."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QAbstractScrollArea, QDialog, QMenu, QWidget

from roc_mcs.gui.components.icon_provider import set_icon_theme

_PALETTES = {
    "dark": {"window": "#11151b", "text": "#d8e0ec", "base": "#151c27", "alternate": "#1b2430", "button": "#202a38", "button_text": "#d8e0ec", "border": "#384454", "hover": "#2a3646", "highlight": "#3e77a8", "highlight_text": "#ffffff", "muted": "#9eafc3", "grid": "#344255", "canvas": "#11151b", "axes": "#151c27", "roi": "#ffd54a", "roi_hover": "#7ad7ff"},
    "light": {"window": "#f5f7fa", "text": "#263445", "base": "#ffffff", "alternate": "#f3f6fa", "button": "#edf1f5", "button_text": "#263445", "border": "#cbd5e1", "hover": "#e3eaf2", "highlight": "#cfe5fa", "highlight_text": "#163c5e", "muted": "#64748b", "grid": "#9aa4b2", "canvas": "#f5f7fa", "axes": "#ffffff", "roi": "#174a78", "roi_hover": "#087e8b"},
}

# Эти поверхности должны явно закрашиваться: одной смены цвета текста недостаточно
# для viewport у QScrollArea и всплывающих окон Qt.
_SURFACE_NAMES = {"roiExplorerCentral", "leftSidebarSurface", "leftSidebarContent", "leftSidebarViewport", "leftSidebarRail", "collapsibleSection", "collapsibleSectionBody", "compactRoiManager", "roiPaletteRail", "roiPaletteViewport", "roiPaletteBody", "MapPlotWidget", "CurvePlotWidget", "PoolingPanel", "MapHeader", "roiEditorPopup", "compactColorPicker"}


def theme_colors(theme: str | None) -> dict[str, str]:
    return _PALETTES["dark" if str(theme).lower() == "dark" else "light"].copy()


def resolve_theme(widget=None) -> str:
    app = QApplication.instance()
    current = widget
    while current is not None:
        value = current.property("rocMcsTheme")
        if str(value).lower() in _PALETTES:
            return str(value).lower()
        current = current.parentWidget() if hasattr(current, "parentWidget") else None
    if app is not None:
        value = app.property("rocMcsTheme")
        if str(value).lower() in _PALETTES:
            return str(value).lower()
        palette = app.palette()
    else:
        palette = QPalette()
    return "dark" if palette.color(QPalette.ColorRole.Window).lightnessF() < 0.5 else "light"


def _make_palette(theme: str, source: QPalette | None = None) -> QPalette:
    c = theme_colors(theme)
    palette = QPalette(source if source is not None else (QApplication.palette() if QApplication.instance() else QPalette()))
    roles = {
        QPalette.ColorRole.Window: c["window"], QPalette.ColorRole.WindowText: c["text"],
        QPalette.ColorRole.Base: c["base"], QPalette.ColorRole.AlternateBase: c["alternate"],
        QPalette.ColorRole.Text: c["text"], QPalette.ColorRole.Button: c["button"],
        QPalette.ColorRole.ButtonText: c["button_text"], QPalette.ColorRole.Highlight: c["highlight"],
        QPalette.ColorRole.HighlightedText: c["highlight_text"], QPalette.ColorRole.ToolTipBase: c["base"],
        QPalette.ColorRole.ToolTipText: c["text"], QPalette.ColorRole.Mid: c["border"],
        QPalette.ColorRole.Midlight: c["hover"], QPalette.ColorRole.Dark: c["window"],
        QPalette.ColorRole.Light: c["alternate"], QPalette.ColorRole.Shadow: c["window"],
        QPalette.ColorRole.BrightText: c["highlight_text"], QPalette.ColorRole.Link: c["highlight"],
        QPalette.ColorRole.LinkVisited: c["highlight"], QPalette.ColorRole.PlaceholderText: c["muted"],
    }
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
        for role, color in roles.items():
            palette.setColor(group, role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.ToolTipText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(c["muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(c["window"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(c["window"]))
    return palette


def apply_widget_palette(widget: QWidget, theme: str | None = None) -> str:
    """Назначает палитру каждому виджету дерева, включая отдельные popup-окна."""
    active_theme = str(theme or resolve_theme(widget)).lower()
    if active_theme not in _PALETTES:
        active_theme = "light"
    c = theme_colors(active_theme)
    widget.setProperty("rocMcsTheme", active_theme)
    palette = _make_palette(active_theme, widget.palette())
    targets = [widget, *widget.findChildren(QWidget)]
    for target in targets:
        target.setPalette(palette)
        # Палитра иконки задаётся локально: это надёжно и для popup/flyout с отдельным Qt-окном.
        set_icon_theme(target, c["text"])
        if target.objectName() in _SURFACE_NAMES or isinstance(target, (QMenu, QDialog)):
            target.setAutoFillBackground(True)
            target.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        if isinstance(target, QAbstractScrollArea):
            viewport = target.viewport()
            viewport.setPalette(palette)
            viewport.setAutoFillBackground(True)
            viewport.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return active_theme


def _set_theme_stylesheet(widget: QWidget, themed_css: str) -> None:
    """Сохраняет локальный QSS и переопределяет только цветовые правила темы."""
    base = widget.property("rocMcsBaseThemeStyleSheet")
    if base is None:
        base = widget.styleSheet()
        widget.setProperty("rocMcsBaseThemeStyleSheet", base)
    widget.setStyleSheet(f"{base}\n{themed_css}".strip())


def apply_popup_theme(widget: QWidget, theme: str | None = None) -> str:
    """Явно оформляет отдельные Qt-окна и меню: они не всегда наследуют QSS родителя."""
    active = apply_widget_palette(widget, theme)
    c = theme_colors(active)
    if isinstance(widget, QMenu):
        _set_theme_stylesheet(widget, f"""
            QMenu {{ background:{c['base']}; color:{c['text']}; border:1px solid {c['border']}; }}
            QMenu::item {{ background:transparent; color:{c['text']}; padding:4px 18px; }}
            QMenu::item:selected {{ background:{c['highlight']}; color:{c['highlight_text']}; }}
            QMenu::separator {{ height:1px; background:{c['border']}; margin:3px 5px; }}
        """)
    elif widget.objectName() in ("roiEditorPopup", "compactColorPicker"):
        name = widget.objectName()
        _set_theme_stylesheet(widget, f"""
            QFrame#{name} {{ background:{c['window']}; color:{c['text']}; border-color:{c['border']}; }}
            QFrame#{name} QWidget {{ background:{c['window']}; color:{c['text']}; }}
            QFrame#{name} QLabel {{ color:{c['text']}; background:transparent; }}
            QFrame#{name} QLineEdit, QFrame#{name} QTableWidget, QFrame#{name} QPlainTextEdit,
            QFrame#{name} QTextEdit, QFrame#{name} QSpinBox, QFrame#{name} QDoubleSpinBox,
            QFrame#{name} QComboBox, QFrame#{name} QAbstractScrollArea,
            QFrame#{name} QAbstractScrollArea QWidget {{ background:{c['base']}; color:{c['text']}; border-color:{c['border']}; selection-background-color:{c['highlight']}; selection-color:{c['highlight_text']}; }}
            QFrame#{name} QHeaderView::section {{ background:{c['button']}; color:{c['text']}; border-color:{c['border']}; }}
            QFrame#{name} QToolButton {{ color:{c['text']}; border-color:{c['border']}; }}
            QFrame#{name} QToolButton:hover {{ background:{c['hover']}; }}
            QFrame#{name} QTableWidget::item:selected {{ background:{c['highlight']}; color:{c['highlight_text']}; }}
        """)
    elif isinstance(widget, QDialog):
        _set_theme_stylesheet(widget, f"""
            QDialog, QColorDialog {{ background:{c['window']}; color:{c['text']}; }}
            QDialog > QWidget, QColorDialog > QWidget,
            QDialog QWidget, QColorDialog QWidget {{ background:{c['window']}; color:{c['text']}; }}
            QDialog QLabel, QColorDialog QLabel {{ color:{c['text']}; background:transparent; }}
            QDialog QFrame, QDialog QGroupBox, QDialog QTabWidget::pane,
            QColorDialog QFrame, QColorDialog QGroupBox, QColorDialog QTabWidget::pane {{ background:{c['window']}; color:{c['text']}; border-color:{c['border']}; }}
            QDialog QLineEdit, QDialog QPlainTextEdit, QDialog QTextEdit, QDialog QSpinBox,
            QDialog QDoubleSpinBox, QDialog QComboBox, QDialog QAbstractItemView,
            QDialog QAbstractScrollArea, QDialog QAbstractScrollArea QWidget, QColorDialog QAbstractScrollArea QWidget,
            QColorDialog QLineEdit, QColorDialog QPlainTextEdit,
            QColorDialog QTextEdit, QColorDialog QSpinBox, QColorDialog QDoubleSpinBox,
            QColorDialog QComboBox, QColorDialog QAbstractItemView, QColorDialog QAbstractScrollArea {{
                background:{c['base']}; color:{c['text']}; border:1px solid {c['border']};
                selection-background-color:{c['highlight']}; selection-color:{c['highlight_text']};
            }}
            QDialog QHeaderView::section, QColorDialog QHeaderView::section {{ background:{c['button']}; color:{c['text']}; border-color:{c['border']}; }}
            QDialog QPushButton, QColorDialog QPushButton {{ background:{c['button']}; color:{c['button_text']}; border:1px solid {c['border']}; border-radius:3px; padding:2px 7px; }}
            QDialog QPushButton:hover, QColorDialog QPushButton:hover {{ background:{c['hover']}; }}
            QDialog QScrollBar, QColorDialog QScrollBar {{ background:{c['window']}; }}
        """)
    return active


def apply_theme(widget: QWidget, theme: str | None = None) -> str:
    """Применяет общие Qt-стили и палитру ко всем поверхностям ROI Explorer."""
    active_theme = apply_widget_palette(widget, theme)
    c = theme_colors(active_theme)
    widget.setStyleSheet(f"""
        QMainWindow, QWidget#roiExplorerCentral, QWidget#leftSidebarSurface,
        QWidget#leftSidebarContent, QWidget#leftSidebarViewport, QWidget#leftSidebarRail,
        QWidget#collapsibleSection, QWidget#collapsibleSectionBody, QWidget#compactRoiManager,
        QWidget#roiPaletteRail, QWidget#roiPaletteViewport, QWidget#roiPaletteBody,
        QWidget#MapPlotWidget, QWidget#CurvePlotWidget, QWidget#PoolingPanel {{
            background:{c['window']}; color:{c['text']};
        }}
        QWidget {{ color:{c['text']}; }}
        QScrollArea, QAbstractScrollArea {{ background:{c['window']}; border:0; }}
        QScrollArea#leftSidebarScroll QWidget#leftSidebarContent {{ background:{c['window']}; color:{c['text']}; }}
        QScrollArea#roiPaletteScroll, QWidget#roiPaletteViewport {{ background:{c['window']}; }}
        QGroupBox {{ border:1px solid {c['border']}; border-radius:4px; margin-top:7px; padding-top:5px; }}
        QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 3px; }}
        QLineEdit, QComboBox, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QTableWidget {{ background:{c['base']}; color:{c['text']}; border:1px solid {c['border']}; border-radius:3px; selection-background-color:{c['highlight']}; selection-color:{c['highlight_text']}; }}
        QHeaderView::section {{ background:{c['button']}; color:{c['text']}; border:0; border-right:1px solid {c['border']}; border-bottom:1px solid {c['border']}; padding:2px; }}
        QAbstractItemView {{ background:{c['base']}; color:{c['text']}; selection-background-color:{c['highlight']}; selection-color:{c['highlight_text']}; outline:0; }}
        QComboBox QAbstractItemView, QMenu {{ background:{c['base']}; color:{c['text']}; border:1px solid {c['border']}; selection-background-color:{c['highlight']}; selection-color:{c['highlight_text']}; }}
        QMenu::item {{ padding:4px 18px; }} QMenu::item:selected {{ background:{c['highlight']}; color:{c['highlight_text']}; }}
        QMenu::separator {{ height:1px; background:{c['border']}; margin:3px 5px; }}
        QPushButton, QToolButton {{ background:{c['button']}; color:{c['button_text']}; border:1px solid {c['border']}; border-radius:4px; padding:3px 5px; }}
        QPushButton:hover, QToolButton:hover {{ background:{c['hover']}; }}
        QPushButton:disabled, QToolButton:disabled {{ color:{c['muted']}; background:{c['window']}; border-color:{c['border']}; }}
        QToolButton#leftSidebarToggle, QToolButton#roiExpandButton, QToolButton#databaseOpenButton, QToolButton#refreshScansButton, QWidget#MapToolBar QToolButton, QWidget#CurveToolBar QToolButton {{ background:transparent; border-color:transparent; }}
        QToolButton[poolActive="true"] {{ background:{c['highlight']}; color:{c['highlight_text']}; border-color:{c['highlight']}; font-weight:bold; }}
        QToolButton#collapsibleSectionHeader {{ text-align:left; padding:5px 4px; font-size:10px; font-weight:700; border:0; border-bottom:1px solid {c['border']}; border-radius:3px; background:transparent; }}
        QToolButton#collapsibleSectionHeader:hover {{ background:{c['hover']}; }}
        QToolTip {{ background:{c['base']}; color:{c['text']}; border:1px solid {c['border']}; padding:3px; }}
        QSplitter::handle {{ background:{c['border']}; }} QSplitter::handle:hover {{ background:{c['highlight']}; }}
        QScrollBar:horizontal, QScrollBar:vertical {{ background:{c['window']}; border:0; width:10px; height:10px; }}
        QScrollBar::handle {{ background:{c['border']}; border-radius:4px; min-width:24px; min-height:24px; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
        QLabel#MapCoordLabel {{ color:{c['muted']}; }}
    """)
    # Локальные QSS-стили с palette(...) используют правильную палитру;
    # независимые QMenu/QDialog оформляются отдельным явным стилем.
    apply_widget_palette(widget, active_theme)
    for target in widget.findChildren(QWidget):
        if isinstance(target, (QMenu, QDialog)) or target.objectName() in ("roiEditorPopup", "compactColorPicker"):
            apply_popup_theme(target, active_theme)
    return active_theme
