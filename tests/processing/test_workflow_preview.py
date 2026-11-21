from types import SimpleNamespace

import numpy as np

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
