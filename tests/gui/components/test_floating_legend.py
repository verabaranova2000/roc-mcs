import os

import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QWidget

from roc_mcs.gui.components.floating_legend import FloatingLegendPanel, LegendEntry


def test_floating_legend_switches_between_gradient_and_scrollable_entries():
    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(700, 500)
    panel = FloatingLegendPanel(parent, "2D-ЛЕГЕНДА", default_size=QSize(250, 210))
    panel.set_alpha_gradient_rows([LegendEntry("1", "#1f77b4")], axis_label="Интенсивность")
    panel.show_on_canvas()
    assert panel.isVisible()
    assert not panel.is_collapsed

    panel.toggle_collapsed()
    assert panel.is_collapsed
    panel.toggle_collapsed()
    assert not panel.is_collapsed

    panel.set_entries([
        LegendEntry("Ветвь 1", "#1f77b4", "Kα₁ · домен A"),
        LegendEntry("Ветвь 2", "#d62728", "Kα₂ · домен A"),
    ])
    assert panel.scroll_area.widget() is not None
    panel.close()
    parent.close()
    app.processEvents()
