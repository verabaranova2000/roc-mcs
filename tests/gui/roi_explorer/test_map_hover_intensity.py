import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from roc_mcs.gui.roi_explorer.map_plot_widget import MapPlotWidget


def test_hover_intensity_formats_scalar_samples_and_rejects_rgba_values():
    assert MapPlotWidget._format_hover_intensity(np.float64(12.345)) == "12.3"
    assert MapPlotWidget._format_hover_intensity(np.asarray(7.0)) == "7.0"
    assert MapPlotWidget._format_hover_intensity(np.asarray([0.1, 0.2, 0.3, 1.0])) == "N/A"
    assert MapPlotWidget._format_hover_intensity(np.ma.masked) == "N/A"
    assert MapPlotWidget._format_hover_intensity(float("nan")) == "N/A"
    assert MapPlotWidget._format_hover_intensity(None) == "N/A"
