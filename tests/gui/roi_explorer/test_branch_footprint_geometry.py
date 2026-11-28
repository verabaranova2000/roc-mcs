import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from roc_mcs.gui.roi_explorer.map_plot_widget import MapPlotWidget


def _app():
    return QApplication.instance() or QApplication([])


def test_footprint_overlay_preserves_map_aspect_and_view_limits():
    app = _app()
    widget = MapPlotWidget()
    image = np.arange(48, dtype=float).reshape(6, 8)
    widget.draw_map(
        image, "detector map", "viridis", 0.0, 47.0,
        x_coords=np.arange(8, dtype=float),
        y_coords=np.arange(6, dtype=float),
    )
    widget.ax.set_xlim(1.0, 6.0)
    widget.ax.set_ylim(0.5, 4.5)
    before = {
        "xlim": widget.ax.get_xlim(),
        "ylim": widget.ax.get_ylim(),
        "aspect": widget.ax.get_aspect(),
        "adjustable": widget.ax.get_adjustable(),
        "anchor": widget.ax.get_anchor(),
        "autoscalex": widget.ax.get_autoscalex_on(),
        "autoscaley": widget.ax.get_autoscaley_on(),
    }
    mask = np.zeros((6, 8), dtype=float)
    mask[1:5, 2:7] = 1.0

    assert widget.draw_branch_footprints([(3, mask, 1.0)], {3: "#FF0000"}) == 1
    assert widget.ax.get_xlim() == before["xlim"]
    assert widget.ax.get_ylim() == before["ylim"]
    assert widget.ax.get_aspect() == before["aspect"]
    assert widget.ax.get_adjustable() == before["adjustable"]
    assert widget.ax.get_anchor() == before["anchor"]
    assert widget.ax.get_autoscalex_on() == before["autoscalex"]
    assert widget.ax.get_autoscaley_on() == before["autoscaley"]

    widget.close()
    app.processEvents()
