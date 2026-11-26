from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from roc_mcs.analysis.workflow_preview import _hkl_alignment_map_layers


def _preview():
    measurements = pd.DataFrame([
        {
            "pair_id": "1↔2", "source_branch_id": 1, "partner_branch_id": 2,
            "source_peak_id": 11, "matched_partner_peak_id": 21,
            "delta_theta_signed_arcsec": 0.40,
        },
        {
            "pair_id": "1↔2", "source_branch_id": 1, "partner_branch_id": 2,
            "source_peak_id": 12, "matched_partner_peak_id": 22,
            "delta_theta_signed_arcsec": -0.10,
        },
    ])
    statistics = pd.DataFrame([
        {"pair_id": "1↔2", "theta_shift_final_arcsec": -0.25},
    ])
    return SimpleNamespace(measurements=measurements, pair_statistics=statistics)


def _catalog():
    # The partner peaks deliberately share one detector pixel. Their residuals
    # must be averaged there, while their source-grid positions remain distinct.
    return pd.DataFrame([
        {"peak_id": 11, "branch_id": 1, "x": 1, "y": 1},
        {"peak_id": 12, "branch_id": 1, "x": 2, "y": 1},
        {"peak_id": 21, "branch_id": 2, "x": 3, "y": 2},
        {"peak_id": 22, "branch_id": 2, "x": 3, "y": 2},
    ])


def test_hkl_residual_maps_apply_final_registration_shift_on_both_branch_grids():
    layers, summary, warning = _hkl_alignment_map_layers(_preview(), _catalog(), (4, 5))

    assert warning is None
    assert len(layers) == 2
    by_key = {layer.key: layer for layer in layers}
    source = by_key["hkl.delta_theta.pair_1_2.source"]
    partner = by_key["hkl.delta_theta.pair_1_2.partner"]

    # residual = raw signed (theta_partner - theta_source) + final shift
    # = [+0.40, -0.10] + (-0.25) = [+0.15, -0.35] arcsec.
    np.testing.assert_allclose(
        [source.data_yx[1, 1], source.data_yx[1, 2]], [0.15, -0.35], atol=1e-12,
    )
    # Two distinct partner peak IDs land on the same partner pixel. The
    # scientific event values are preserved as their arithmetic mean there.
    assert partner.data_yx[2, 3] == pytest.approx(-0.10, abs=1e-12)
    assert np.isnan(partner.data_yx[1, 3])
    assert source.unit == partner.unit == "″"
    assert "среднее" in partner.description
    assert "таких пикселей: 1" in partner.description
    assert summary["1↔2"]["matched_events"] == 2
    assert summary["1↔2"]["partner_occupied_pixels"] == 1


def test_hkl_map_refuses_ambiguous_peak_id_catalogue():
    catalog = pd.concat([_catalog(), _catalog().iloc[[0]]], ignore_index=True)
    layers, summary, warning = _hkl_alignment_map_layers(_preview(), catalog, (4, 5))

    assert layers == []
    assert summary == {}
    assert warning is not None
    assert "неуникальные peak_id" in warning


def test_hkl_map_skips_noninteger_or_out_of_grid_coordinates_without_guessing():
    catalog = _catalog().astype({"x": float, "y": float})
    catalog.loc[catalog["peak_id"] == 11, "x"] = 1.5
    catalog.loc[catalog["peak_id"] == 12, "x"] = 99
    catalog.loc[catalog["peak_id"] == 21, "y"] = 99
    catalog.loc[catalog["peak_id"] == 22, "y"] = 99

    layers, summary, warning = _hkl_alignment_map_layers(_preview(), catalog, (4, 5))

    assert layers == []
    assert summary == {}
    assert warning is not None
    assert "не осталось валидных точек" in warning


def test_hkl_diverging_colormap_is_blue_white_red_with_zero_at_midpoint():
    from roc_mcs.gui.roi_explorer.colormap_catalog import mpl_colormap

    cmap = mpl_colormap("RdBu_r")
    negative = np.asarray(cmap(0.0)[:3])
    zero = np.asarray(cmap(0.5)[:3])
    positive = np.asarray(cmap(1.0)[:3])

    assert negative[2] > negative[0]  # blue for negative residuals
    assert positive[0] > positive[2]  # red for positive residuals
    assert np.min(zero) > 0.9        # neutral near-white at zero


def test_workflow_preview_publishes_hkl_maps_as_default_result_layers():
    from types import SimpleNamespace
    from roc_mcs.analysis.workflow_preview import build_workflow_preview_state

    scan_data = np.ones((4, 5, 3), dtype=float)
    scan = SimpleNamespace(
        arrays={
            "data_yxt": scan_data,
            "theta": np.array([0.0, 1.0, 2.0]),
            "x_coords": np.arange(5, dtype=float),
            "y_coords": np.arange(4, dtype=float),
            "valid_mask_yxt": np.ones(scan_data.shape, dtype=bool),
        },
        metadata={},
    )
    branch_bundle = SimpleNamespace(tables={"branch_catalog": _catalog()}, metadata={})
    hkl_tables = {
        "hkl_theoretical_reflections": pd.DataFrame([
            {"hkl": "111", "d_A": 2.0, "theta_ka1_deg": 20.0, "delta_theta_arcsec": 0.2},
        ]),
        "hkl_bragg_rejected": pd.DataFrame(columns=["hkl", "d_A", "reason"]),
        "hkl_pair_measurements": _preview().measurements.assign(delta_theta_abs_arcsec=[0.40, 0.10]),
        "hkl_pair_statistics": _preview().pair_statistics.assign(
            matched_assignment_count=[2], delta_theta_abs_median_arcsec=[0.25],
            identification_status=["ready"],
        ),
        "hkl_pair_ranking": pd.DataFrame(columns=[
            "pair_id", "rank", "hkl", "delta_theta_arcsec", "difference_from_exp_median_arcsec",
        ]),
    }
    hkl_bundle = SimpleNamespace(tables=hkl_tables, metadata={"theory": {"crystal_name": "Test"}})
    node = SimpleNamespace(
        id="hkl-node", operation_id="theory.identify_hkl", label="HKL", summary="",
    )

    state = build_workflow_preview_state(
        node, {"scan": scan, "branches": branch_bundle, "hkl_analysis": hkl_bundle},
    )

    assert state.default_map_key == "hkl.delta_theta.pair_1_2.source"
    assert {layer.key for layer in state.map_layers} >= {
        "hkl.delta_theta.pair_1_2.source", "hkl.delta_theta.pair_1_2.partner",
    }
    assert state.metadata["hkl_alignment_maps"]["unit"] == "arcsec"
    assert state.metadata["hkl_alignment_maps"]["aggregation"] == "mean_per_reference_pixel"
    assert state.hkl_identification_preview is not None
