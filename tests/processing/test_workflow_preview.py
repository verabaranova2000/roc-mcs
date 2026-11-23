from types import SimpleNamespace

import numpy as np
import pytest

from roc_mcs.analysis.workflow_preview import build_workflow_preview_state


def _scan():
    raw = np.array([[[1.0, 100.0, 3.0]]])
    return SimpleNamespace(
        arrays={
            "data_yxt": raw,
            "theta": np.array([0.0, 1.0, 2.0]),
            "x_coords": np.array([0.0]),
            "y_coords": np.array([0.0]),
            "valid_mask_yxt": np.ones(raw.shape, dtype=bool),
        },
        metadata={},
    )


def _spikes():
    clean = np.array([[[1.0, 2.0, 3.0]]])
    return SimpleNamespace(
        arrays={
            "clean_data_yxt": clean,
            # The ordinary analysis mask excludes the replaced spike sample.
            "valid_mask_yxt": np.array([[[True, False, True]]]),
            # The detector mask says the interpolated sample is display-valid.
            "detector_valid_mask_yxt": np.ones(clean.shape, dtype=bool),
            "spike_count_map": np.array([[1.0]]),
        },
        metadata={},
    )


def _node(operation_id):
    return SimpleNamespace(
        id="node-1", operation_id=operation_id, label="Test node", summary=""
    )


def test_spike_node_displays_cleaned_map_and_roc_by_default():
    scan = _scan()
    spikes = _spikes()
    state = build_workflow_preview_state(
        _node("preprocess.spikes"), {"scan": scan, "spikes": spikes}
    )

    np.testing.assert_array_equal(state.data_yxt, spikes.arrays["clean_data_yxt"])
    assert state.map_layer("scan.intensity_sum").data_yx[0, 0] == 6.0
    assert state.map_layer("scan.valid_fraction").data_yx[0, 0] == 1.0
    assert "очищенного" in state.map_layer("scan.intensity_sum").label
    assert state.default_map_key == "spikes.count"
    assert state.default_curve_key == "spikes.interpolated"

    # The measured source remains immutable/raw, while the cleaned signal
    # includes the interpolated sample using detector_valid_mask_yxt.
    measured = state.curve_source("scan.measured")
    interpolated = state.curve_source("spikes.interpolated")
    assert measured.data_yxt[0, 0, 1] == 100.0
    assert interpolated.data_yxt[0, 0, 1] == 2.0
    assert interpolated.valid_mask_yxt[0, 0, 1]


def test_peak_node_uses_interpolated_primary_signal_only_when_declared():
    scan = _scan()
    spikes = _spikes()
    peaks = SimpleNamespace(metadata={"used_interpolated_data": True}, arrays={})
    state = build_workflow_preview_state(
        _node("peaks.detect"), {"scan": scan, "spikes": spikes, "peaks": peaks}
    )

    np.testing.assert_array_equal(state.data_yxt, spikes.arrays["clean_data_yxt"])
    assert state.default_curve_key == "spikes.interpolated"


def test_fit_node_exposes_only_registered_derived_maps_and_compatible_curves():
    import pandas as pd

    scan = _scan()
    fit_cube = np.array([[[0.5, 1.5, 2.5]]])
    residual_cube = np.array([[[0.5, 0.5, 0.5]]])
    pixel_catalog = pd.DataFrame([{
        "x": 0, "y": 0, "n_detected": 1, "n_fitted": 1,
        "rmse": 0.5, "redchi": 0.25, "chisqr": 0.75,
        "p95_abs_residual": 0.5, "max_abs_residual": 0.5,
        "rmse_over_noise": 0.5, "aic": 1.0, "bic": 1.1,
    }])
    fitted_catalog = pd.DataFrame([{
        "x": 0, "y": 0, "component_id": 0,
        "S": 10.0, "theta0": 1.0, "H": 0.3, "eta": 0.4,
        "delta_theta": 0.1,
    }])
    fit = SimpleNamespace(
        arrays={
            "theta": np.array([0.0, 1.0, 2.0]),
            "fit_cube": fit_cube,
            "residual_cube": residual_cube,
            "valid_mask": np.ones(fit_cube.shape, dtype=bool),
        },
        tables={"pixel_fit_catalog": pixel_catalog, "fitted_peak_catalog": fitted_catalog},
        metadata={"model_id": "pvoigt", "component_param_names": ["S", "theta0", "H", "eta"]},
    )
    state = build_workflow_preview_state(_node("fit.scan"), {"scan": scan, "fit": fit})
    map_keys = {layer.key for layer in state.map_layers}
    curve_keys = {source.key for source in state.curve_sources}

    assert "fit.rmse" in map_keys
    assert "fit.parameter.H.component0" in map_keys
    assert "fit.derived.FWHM.component0" in map_keys
    assert "fit.residual_mean" in map_keys
    assert "fit.residual_rmse_cube" in map_keys
    assert "fit.model" in curve_keys and "fit.residual" in curve_keys
    assert state.default_map_key == "fit.rmse"
    assert state.default_curve_key == "fit.model"
    assert state.map_layer("fit.derived.FWHM.component0").data_yx[0, 0] == 0.3
    assert state.map_layer("fit.residual_mean").data_yx[0, 0] == 0.5


def test_analysis_spatial_validity_mask_is_preserved_in_preview_state():
    scan = _scan()
    scan.arrays["analysis_valid_mask_yx"] = np.array([[True]])
    state = build_workflow_preview_state(_node("peaks.detect"), {"scan": scan})
    np.testing.assert_array_equal(state.analysis_valid_mask_yx, [[True]])


def test_analysis_spatial_validity_mask_shape_is_validated():
    scan = _scan()
    scan.arrays["analysis_valid_mask_yx"] = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="analysis_valid_mask_yx"):
        build_workflow_preview_state(_node("peaks.detect"), {"scan": scan})


def test_spike_input_offers_measured_intensity_map_for_peak_configuration():
    scan = _scan()
    spikes = _spikes()
    state = build_workflow_preview_state(
        _node("preprocess.spikes"), {"scan": scan, "spikes": spikes}
    )

    measured_layer = state.map_layer("scan.measured_intensity_sum")
    assert measured_layer is not None
    # This layer intentionally shows the original measured counts, including
    # the sample later marked as a spike. Pixel eligibility is checked against
    # the same combined validity mask as peaks.detect when a point is selected.
    assert measured_layer.data_yx[0, 0] == 104.0
    assert state.map_layer("scan.intensity_sum").data_yx[0, 0] == 6.0


def test_branch_map_uses_zero_for_valid_pixels_without_branch_and_preserves_edges():
    import pandas as pd

    raw = np.ones((2, 2, 3), dtype=float)
    scan = SimpleNamespace(arrays={
        "data_yxt": raw,
        "theta": np.array([0.0, 1.0, 2.0]),
        "x_coords": np.array([0.0, 1.0]),
        "y_coords": np.array([0.0, 1.0]),
        "valid_mask_yxt": np.ones(raw.shape, dtype=bool),
        "analysis_valid_mask_yx": np.array([[True, True], [True, False]]),
    }, metadata={})
    catalog = pd.DataFrame([
        {"peak_id": 10, "x": 0, "y": 0, "theta": 0.5, "branch_id": 0},
        {"peak_id": 11, "x": 1, "y": 0, "theta": 0.6, "branch_id": 0},
    ])
    edges = pd.DataFrame([{"peak_id_a": 10, "peak_id_b": 11, "delta_theta": 0.1}])
    branches = SimpleNamespace(tables={"branch_catalog": catalog, "branch_edges": edges}, metadata={})

    state = build_workflow_preview_state(
        _node("peaks.branches"), {"scan": scan, "branches": branches}
    )
    np.testing.assert_allclose(
        state.map_layer("branches.count").data_yx,
        [[1.0, 1.0], [0.0, np.nan]],
        equal_nan=True,
    )
    assert state.branch_edges is edges
