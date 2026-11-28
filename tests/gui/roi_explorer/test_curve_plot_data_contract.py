import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from roc_mcs.gui.roi_explorer.curve_plot_widget import CurvePlotWidget


def _app():
    return QApplication.instance() or QApplication([])


def test_mismatched_and_empty_curve_pairs_are_ignored_before_matplotlib_relim():
    app = _app()
    widget = CurvePlotWidget()

    lines = widget.draw_curves(
        [
            {
                "id": 1,
                "th": np.arange(126, dtype=float),
                "curve": np.array([], dtype=float),
                "color": "#FF0000",
                "is_selected": False,
                "label": "invalid pinned profile",
            }
        ],
        {"th": np.arange(126, dtype=float), "curve": np.array([], dtype=float)},
        pixel_data={"th": np.arange(126), "curve": np.arange(125), "label": "invalid pixel"},
    )

    assert lines == {}
    assert not widget._curve_artists
    assert widget._preview_artist is None
    assert widget._pixel_artist is None

    # Shape-compatible but wholly non-finite data is not a renderable curve.
    lines = widget.draw_curves(
        [{
            "id": 3,
            "th": np.arange(126, dtype=float),
            "curve": np.full(126, np.nan),
            "color": "#FF0000",
            "is_selected": False,
            "label": "all-invalid profile",
        }],
    )
    assert lines == {}
    assert not widget._curve_artists

    # A subsequent valid update must still render normally after invalid input.
    valid = np.linspace(1.0, 3.0, 126)
    lines = widget.draw_curves(
        [{
            "id": 2,
            "th": np.arange(126, dtype=float),
            "curve": valid,
            "color": "#00AA00",
            "is_selected": True,
            "label": "valid profile",
        }],
    )
    assert set(lines) == {2}
    widget.canvas.draw()

    widget.close()
    app.processEvents()
