"""Общий стиль компактных панелей инструментов графиков."""
from pathlib import Path

ICON_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"
COMPACT_PLOT_TOOLBAR_STYLE = """
    QWidget#MapToolBar QToolButton, QWidget#CurveToolBar QToolButton {
        max-height: 20px; min-height: 20px; padding: 0px; margin: 0px;
        border: 1px solid transparent; border-radius: 2px; background: transparent; font-size: 13px;
    }
    QWidget#MapToolBar QToolButton:hover, QWidget#CurveToolBar QToolButton:hover {
        background: #e8e8e8; border: 1px solid #cccccc;
    }
    QWidget#MapToolBar QToolButton:checked, QWidget#CurveToolBar QToolButton:checked {
        background: #e6f2ff; border: 1px solid #000080; font-weight: bold;
    }
    QWidget#MapToolBar QDoubleSpinBox {
        max-height: 18px; min-height: 18px; font-size: 11px; padding: 0px;
        border: 1px solid #cccccc; border-radius: 2px; background: #ffffff;
    }
"""
