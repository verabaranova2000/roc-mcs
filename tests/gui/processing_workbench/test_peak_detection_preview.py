import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from roc_mcs.gui.processing_workbench.peak_detection_preview import PeakDetectionPreview


def test_peak_preview_recomputes_local_neighbourhood_and_detector_live():
    app = QApplication.instance() or QApplication([])
    preview = PeakDetectionPreview()
    before = preview._raw.copy()
    initial_count = int(np.count_nonzero(preview._valid))

    preview.set_parameters(window_size=15, min_distance=20, min_relative_prominence=0.4)

    assert not np.allclose(before, preview._raw)
    assert preview._error == ""
    assert preview._noise_sigma > 0
    assert preview._raw.shape == (preview._N_SAMPLES,)
    assert int(np.count_nonzero(preview._valid)) <= initial_count
    preview.close()
    app.processEvents()
