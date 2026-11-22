from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from roc_mcs.gui.processing_workbench.peak_preview_data import (
    _background_sigma,
    build_peak_pixel_preview_data,
)
from roc_mcs.processing.background import estimate_background_stats
from roc_mcs.peaks.peak_signal import local_neighbourhood_mean_curve


def _bundle(data_yxt: np.ndarray, *, valid=None, analysis_valid=None, grid_id="grid-a"):
    theta = np.linspace(10.0, 20.0, data_yxt.shape[-1])
    arrays = {
        "data_yxt": data_yxt,
        "theta": theta,
        "x_coords": np.linspace(100.0, 100.0 + data_yxt.shape[1] - 1, data_yxt.shape[1]),
        "y_coords": np.linspace(200.0, 200.0 + data_yxt.shape[0] - 1, data_yxt.shape[0]),
    }
    if valid is not None:
        arrays["valid_mask_yxt"] = valid
    if analysis_valid is not None:
        arrays["analysis_valid_mask_yx"] = analysis_valid
    return SimpleNamespace(arrays=arrays, metadata={"grid_id": grid_id})


def test_background_sigma_matches_production_estimator_with_invalid_pixels():
    rng = np.random.default_rng(42)
    data = rng.normal(20.0, 1.5, size=(7, 8, 32))
    valid = np.ones((7, 8), dtype=bool)
    valid[1, 3] = False
    valid[5, 6] = False

    masked = data.copy()
    masked[~valid] = np.nan
    expected = estimate_background_stats(np.moveaxis(masked, -1, 0)).sigma

    assert _background_sigma(data, valid) == pytest.approx(expected)


def test_build_pixel_preview_returns_exact_local_input_and_coordinates():
    rng = np.random.default_rng(7)
    data = rng.normal(10.0, 0.5, size=(9, 11, 48))
    valid_yxt = np.ones_like(data, dtype=bool)
    valid_yxt[3, 4, :] = False
    analysis_valid = np.ones(data.shape[:2], dtype=bool)
    analysis_valid[4, 6] = False
    scan = _bundle(data, valid=valid_yxt, analysis_valid=analysis_valid)

    clean = data + 0.25
    spikes = SimpleNamespace(
        arrays={"clean_data_yxt": clean},
        metadata={"grid_id": "grid-a"},
    )

    payload = build_peak_pixel_preview_data(scan, spikes, detector_x=105.0, detector_y=204.0)

    assert payload["pixel_x"] == 5
    assert payload["pixel_y"] == 4
    assert payload["detector_x"] == pytest.approx(105.0)
    assert payload["detector_y"] == pytest.approx(204.0)
    assert payload["measured_tile_yxt"].shape == (9, 11, 48)
    assert payload["interpolated_tile_yxt"].shape == (9, 11, 48)
    assert payload["center_y"] == 4
    assert payload["center_x"] == 5
    assert payload["interpolation_available"] is True
    assert payload["pixel_valid_yx"].shape == (9, 11)
    assert not bool(payload["pixel_valid_yx"][3, 4])
    assert not bool(payload["pixel_valid_yx"][4, 6])

    # The bounded tile must reproduce the same N×N ROC as the production
    # full-grid helper, including invalid-neighbour exclusion.
    expected = local_neighbourhood_mean_curve(
        data, 4, 5, 3, pixel_valid=np.all(valid_yxt, axis=-1) & analysis_valid
    )
    actual = local_neighbourhood_mean_curve(
        payload["measured_tile_yxt"], payload["center_y"], payload["center_x"], 3,
        pixel_valid=payload["pixel_valid_yx"],
    )
    np.testing.assert_allclose(actual, expected)


def test_build_pixel_preview_rejects_invalid_or_out_of_grid_pixel():
    data = np.ones((5, 6, 20), dtype=float)
    valid_yxt = np.ones_like(data, dtype=bool)
    valid_yxt[2, 3, :] = False
    scan = _bundle(data, valid=valid_yxt)

    with pytest.raises(ValueError, match="исключён из анализа"):
        build_peak_pixel_preview_data(scan, None, detector_x=103.0, detector_y=202.0)

    with pytest.raises(ValueError, match="вне текущей detector-grid"):
        build_peak_pixel_preview_data(scan, None, detector_x=999.0, detector_y=202.0)


def test_build_pixel_preview_rejects_spike_artifact_from_another_grid():
    data = np.ones((5, 6, 20), dtype=float)
    scan = _bundle(data, grid_id="grid-a")
    spikes = SimpleNamespace(
        arrays={"clean_data_yxt": data.copy()},
        metadata={"grid_id": "grid-b"},
    )
    with pytest.raises(ValueError, match="разным spatial-grid"):
        build_peak_pixel_preview_data(scan, spikes, detector_x=103.0, detector_y=202.0)
