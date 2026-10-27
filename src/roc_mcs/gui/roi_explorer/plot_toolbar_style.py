"""Общий стиль компактных панелей инструментов графиков."""
from pathlib import Path

ICON_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"
COMPACT_PLOT_TOOLBAR_STYLE = """
    QWidget#MapToolBar QToolButton, QWidget#CurveToolBar QToolButton {
        max-height: 20px; min-height: 20px; padding: 0px; margin: 0px;
        border: 1px solid transparent; border-radius: 2px; background: transparent; font-size: 13px;
        color: palette(button-text);
    }
    QWidget#MapToolBar QToolButton:hover, QWidget#CurveToolBar QToolButton:hover {
        background: palette(midlight); border: 1px solid palette(mid);
    }
    QWidget#MapToolBar QToolButton:checked, QWidget#CurveToolBar QToolButton:checked {
        background: palette(highlight); color: palette(highlighted-text); border: 1px solid palette(highlight); font-weight: bold;
    }
    QWidget#MapToolBar QDoubleSpinBox {
        max-height: 18px; min-height: 18px; font-size: 11px; padding: 0px;
        border: 1px solid palette(mid); border-radius: 2px; background: palette(base); color: palette(text);
    }
"""
