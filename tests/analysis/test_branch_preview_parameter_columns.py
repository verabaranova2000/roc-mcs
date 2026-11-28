import pandas as pd

from roc_mcs.analysis.preview_context import _branch_catalog_preview_columns
from roc_mcs.analysis.branch_display import fitted_branch_parameter_columns
from roc_mcs.fitting.registry import MODEL_SPECS
from roc_mcs.peaks.branches import fitted_peak_positions


def test_branch_preview_schema_retains_registry_fit_parameters():
    preview_columns = set(_branch_catalog_preview_columns())
    fitted_columns = set(fitted_branch_parameter_columns())

    assert fitted_columns.issubset(preview_columns)
    assert {"branch_id", "analysis_included", "branch_position_kind"}.issubset(preview_columns)
    for spec in MODEL_SPECS.values():
        assert set(spec.param_names).issubset(preview_columns)


def test_fitted_peak_identity_merge_preserves_all_model_values():
    class Bundle:
        def __init__(self, tables):
            self.tables = tables

    bundle = Bundle({
        "peak_catalog": pd.DataFrame([
            {"peak_id": 41, "x": 2, "y": 3, "theta": 12.25, "component_id": 0},
        ]),
        "fitted_peak_catalog": pd.DataFrame([
            {"x": 2, "y": 3, "component_id": 0, "model": "pvoigt",
             "fit_success": True, "S": 150.0, "theta0": 12.4, "H": 0.75,
             "eta": 0.32},
        ]),
    })

    result = fitted_peak_positions(bundle)

    assert result.loc[0, "peak_id"] == 41
    assert result.loc[0, "theta"] == 12.4
    assert result.loc[0, "theta_detected"] == 12.25
    assert result.loc[0, "model"] == "pvoigt"
    assert result.loc[0, "S"] == 150.0
    assert result.loc[0, "H"] == 0.75
    assert result.loc[0, "eta"] == 0.32
