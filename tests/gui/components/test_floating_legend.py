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
    initial_scale = panel.legend_font_scale
    panel.font_decrease_button.click()
    assert panel.legend_font_scale < initial_scale
    panel.font_increase_button.click()
    assert panel.legend_font_scale == initial_scale

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


def test_floating_legend_excluded_group_checkbox_emits_a_single_action():
    from PyQt6.QtTest import QSignalSpy
    from PyQt6.QtWidgets import QCheckBox

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(700, 500)
    panel = FloatingLegendPanel(parent, default_size=QSize(250, 210))
    panel.set_entries([
        LegendEntry(
            "Исключённые · 74 ветви", "#858B95",
            "Опционально показать нейтральным слоем.",
            action_key="show_excluded", checked=False,
        ),
    ])
    spy = QSignalSpy(panel.entry_toggled)
    checkbox = panel.scroll_area.widget().findChild(QCheckBox)
    assert checkbox is not None
    checkbox.click()
    assert spy.count() == 1
    assert spy.at(0) == ["show_excluded", True]
    panel.close()
    parent.close()
    app.processEvents()


def test_floating_legend_drag_uses_one_absolute_global_origin():
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(700, 500)
    panel = FloatingLegendPanel(parent, default_size=QSize(250, 210))
    panel.show_on_canvas(position=QPoint(40, 40))
    app.processEvents()
    start_panel = panel.pos()
    header = panel.header
    start_local = QPointF(60.0, 13.0)
    start_global = QPointF(header.mapToGlobal(start_local.toPoint()))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress, start_local, start_global,
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(header, press)

    delta = QPoint(73, 41)
    target_global = start_global + QPointF(delta)
    target_local = QPointF(header.mapFromGlobal(target_global.toPoint()))
    move = QMouseEvent(
        QEvent.Type.MouseMove, target_local, target_global,
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(header, move)
    assert panel.pos() == start_panel + delta

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease, target_local, target_global,
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(header, release)
    panel.close()
    parent.close()
    app.processEvents()
