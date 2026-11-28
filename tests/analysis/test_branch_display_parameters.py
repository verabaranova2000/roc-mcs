import numpy as np
import pandas as pd

from roc_mcs.analysis.branch_display import (
    available_branch_parameters,
    build_branch_parameter_map,
    fitted_branch_parameter_columns,
    _fwhm_values,
)
from roc_mcs.fitting.registry import MODEL_SPECS


def _field_keys(frame):
    return [item["key"] for item in available_branch_parameters(frame)]


def test_fitted_parameter_menu_orders_rci_values_before_model_parameters():
    catalog = pd.DataFrame([
        {
            "branch_id": 0, "x": 1, "y": 2, "peak_id": 10,
            "theta0": 12.5, "theta": 12.5, "S": 100.0,
            "model": "pvoigt", "H": 2.4, "eta": 0.35,
        },
    ])

    fields = available_branch_parameters(catalog)

    assert [item["key"] for item in fields[:3]] == ["theta0", "FWHM", "S"]
    assert [item["key"] for item in fields] == ["theta0", "FWHM", "S", "eta"]
    assert fields[0]["label"] == "Положение пика θ₀"
    assert fields[1]["label"].endswith("FWHM")
    assert fields[2]["label"].startswith("Интегральная интенсивность")
    # Pseudo-Voigt H is exactly its FWHM, so it must not be offered twice.
    assert "H" not in _field_keys(catalog)


def test_model_registry_columns_cover_every_registered_fit_parameter():
    required = set(fitted_branch_parameter_columns())
    for spec in MODEL_SPECS.values():
        assert set(spec.param_names).issubset(required)
    assert {"model", "fit_success", "peak_id", "x", "y"}.issubset(required)


def test_detection_only_catalog_does_not_invent_fit_quantities():
    catalog = pd.DataFrame({
        "branch_id": [0, 0], "x": [0, 1], "y": [0, 0],
        "theta": [1.0, 1.5], "prominence": [4.0, 5.0],
    })

    keys = _field_keys(catalog)

    assert keys == ["theta", "prominence"]
    assert "FWHM" not in keys
    assert "S" not in keys


def test_analytic_fwhm_is_computed_per_fitted_event_and_averaged_per_pixel():
    catalog = pd.DataFrame([
        {"branch_id": 0, "x": 1, "y": 0, "model": "gauss", "S": 10.0,
         "theta0": 1.0, "sigma": 1.0},
        {"branch_id": 0, "x": 1, "y": 0, "model": "lorentz", "S": 20.0,
         "theta0": 1.1, "gamma": 2.0},
        {"branch_id": 1, "x": 0, "y": 0, "model": "pvoigt", "S": 5.0,
         "theta0": 1.2, "H": 3.0, "eta": 0.5},
    ])

    values = _fwhm_values(catalog)
    image = build_branch_parameter_map(catalog, 0, "FWHM", (1, 2))

    expected = (2.0 * np.sqrt(2.0 * np.log(2.0)) + 4.0) / 2.0
    assert np.allclose(values, [2.0 * np.sqrt(2.0 * np.log(2.0)), 4.0, 3.0])
    assert np.isclose(image[0, 1], expected)
    assert np.isnan(image[0, 0])


def test_registered_models_without_closed_form_return_finite_numerical_fwhm():
    # Split-Voigt, EMG and ultrasound-broadened Gaussian have no closed-form
    # FWHM in DERIVED_SPECS, so their registered profiles are sampled using a
    # bounded per-event grid.
    catalog = pd.DataFrame([
        {"branch_id": 0, "x": 0, "y": 0, "model": "emg", "S": 10.0,
         "theta0": 0.0, "sigma": 1.0, "gamma": 0.3, "lam": 0.2},
        {"branch_id": 0, "x": 1, "y": 0, "model": "split_voigt", "S": 10.0,
         "theta0": 0.0, "beta_Gl": 1.0, "beta_Cl": 1.0,
         "beta_Gr": 1.5, "beta_Cr": 0.6},
        {"branch_id": 0, "x": 2, "y": 0, "model": "gauss_us", "S": 10.0,
         "theta0": 0.0, "sigma": 1.0, "Delta": 0.4},
    ])

    widths = _fwhm_values(catalog)

    assert widths.shape == (3,)
    assert np.isfinite(widths).all()
    assert (widths > 0).all()


def test_parameter_menu_is_branch_specific_when_branches_use_different_models():
    catalog = pd.DataFrame([
        {"branch_id": 0, "x": 0, "y": 0, "model": "gauss", "S": 10.0,
         "theta0": 1.0, "sigma": 0.8},
        {"branch_id": 1, "x": 1, "y": 0, "model": "pvoigt", "S": 12.0,
         "theta0": 1.5, "H": 2.2, "eta": 0.4},
    ])

    gaussian = _field_keys_for_branch(catalog, 0)
    pseudo_voigt = _field_keys_for_branch(catalog, 1)

    assert gaussian[:3] == ["theta0", "FWHM", "S"]
    assert "sigma" in gaussian and "eta" not in gaussian
    assert pseudo_voigt[:3] == ["theta0", "FWHM", "S"]
    assert "eta" in pseudo_voigt and "sigma" not in pseudo_voigt


def _field_keys_for_branch(catalog, branch_id):
    return [
        item["key"]
        for item in available_branch_parameters(catalog, branch_id=branch_id)
    ]


def test_branch_parameter_choice_cache_is_per_branch_and_preview_scoped():
    from roc_mcs.analysis.branch_display import BranchParameterChoices

    choices = BranchParameterChoices()
    choices.bind_context(("fit-node-1", "fingerprint-a"))
    choices.remember(0, "theta0")
    choices.remember(1, "eta")

    assert choices.preferred(0, ["theta0", "eta"], fallback="theta0") == "theta0"
    assert choices.preferred(1, ["theta0", "eta"], fallback="theta0") == "eta"
    # A branch with a different model cannot inherit an unavailable parameter.
    assert choices.preferred(2, ["theta0", "sigma"], fallback="theta0") == "theta0"

    # Rebinding the same immutable preview preserves choices; a new preview
    # invalidates all branch IDs/parameter selections from the old catalogue.
    choices.bind_context(("fit-node-1", "fingerprint-a"))
    assert choices.preferred(1, ["theta0", "eta"], fallback="theta0") == "eta"
    choices.bind_context(("fit-node-2", "fingerprint-b"))
    assert choices.preferred(1, ["theta0", "eta"], fallback="theta0") == "theta0"
