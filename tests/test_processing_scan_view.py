from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from roc_mcs.analysis.artifacts import ArtifactBundle
from roc_mcs.analysis.scan_view import build_scan_view_state


def _node(operation_id: str):
    return SimpleNamespace(id="node-1", operation_id=operation_id, label=operation_id)


def _scan_bundle():
    data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    valid = np.ones(data.shape, dtype=bool)
    valid[1, 2, :] = False
    data[1, 2, :] = 999.0
    return ArtifactBundle(
        arrays={
            "data_yxt": data,
            "theta": np.linspace(1.0, 5.0, 5),
            "x_coords": np.arange(4, dtype=float) + 0.5,
            "y_coords": np.arange(3, dtype=float) + 1.5,
            "valid_mask_yxt": valid,
        },
        metadata={"grid_id": "grid-A", "scan_key": "entry472"},
    )


def test_raw_or_qc_state_keeps_yxt_and_masks_invalid_samples_in_map():
    scan = _scan_bundle()
    state = build_scan_view_state(_node("preprocess.detector_bad_pixels"), scan)
    assert state.data_yxt.shape == (3, 4, 5)
    assert state.map_data.shape == (3, 4)
    assert np.isnan(state.map_data[1, 2])
    np.testing.assert_array_equal(state.x_coords, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(state.theta, [1., 2., 3., 4., 5.])


def test_spike_state_displays_interpolated_data_but_uses_spike_count_map():
    scan = _scan_bundle()
    clean = np.asarray(scan.arrays["data_yxt"], dtype=float).copy()
    clean[1, 2, :] = np.nan
    detector_valid = np.asarray(scan.arrays["valid_mask_yxt"], dtype=bool)
    spike_count = np.zeros((3, 4), dtype=np.int32)
    spike_count[0, 1] = 2
    spikes = ArtifactBundle(
        arrays={
            "clean_data_yxt": clean,
            "detector_valid_mask_yxt": detector_valid,
            "spike_count_map": spike_count,
        },
        metadata={"grid_id": "grid-A", "n_spikes": 2},
    )
    state = build_scan_view_state(_node("preprocess.spikes"), scan, spikes)
    assert state.map_title.startswith("Число обнаруженных")
    assert state.map_data[0, 1] == 2
    assert state.valid_mask_yxt is detector_valid
    assert np.isnan(state.data_yxt[1, 2]).all()


def test_spike_state_rejects_artifact_from_another_grid():
    scan = _scan_bundle()
    spikes = ArtifactBundle(
        arrays={
            "clean_data_yxt": scan.arrays["data_yxt"],
            "spike_count_map": np.zeros((3, 4)),
        },
        metadata={"grid_id": "grid-B"},
    )
    with pytest.raises(ValueError, match="другой координатной сетке"):
        build_scan_view_state(_node("preprocess.spikes"), scan, spikes)


def test_masked_detector_projection_respects_validity_and_nan_for_axis_2():
    from roc_mcs.gui.roi_explorer.volume_rendering import masked_detector_projection

    data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    valid = np.ones(data.shape, dtype=bool)
    valid[1, 2, :] = False
    data[0, 0, 0] = np.nan
    expected = np.sum(np.where(np.isfinite(data) & valid, data, 0.0), axis=2, dtype=np.float64).T
    expected[0, 0] = np.sum(data[0, 0, 1:])
    expected[2, 1] = np.nan
    actual = masked_detector_projection(data, 2, valid, chunk_rows=1)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_masked_detector_projection_supports_theta_first_layout():
    from roc_mcs.gui.roi_explorer.volume_rendering import masked_detector_projection

    yxt = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    tyx = np.moveaxis(yxt, 2, 0)
    mask = np.ones(tyx.shape, dtype=bool)
    mask[:, 1, 2] = False
    expected = np.sum(np.where(np.isfinite(tyx) & mask, tyx, 0.0), axis=0, dtype=np.float64).T
    expected[2, 1] = np.nan
    actual = masked_detector_projection(tyx, 0, mask, chunk_rows=1)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_masked_detector_projection_rejects_incompatible_mask():
    from roc_mcs.gui.roi_explorer.volume_rendering import masked_detector_projection

    with pytest.raises(ValueError, match="совпадать по форме"):
        masked_detector_projection(np.ones((2, 3, 4)), 2, np.ones((2, 3), dtype=bool))
