from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from roc_mcs.analysis.artifacts import ArtifactBundle
from roc_mcs.analysis.operations import default_registry
from roc_mcs.peaks.branches import (
    build_peak_branch_graph,
    fitted_peak_positions,
    label_peak_branches,
    summarize_peak_spacing,
)


def test_isolated_peak_events_are_individual_branches():
    catalog = pd.DataFrame(
        [
            {"peak_id": 10, "x": 0, "y": 0, "theta": 100.0},
            {"peak_id": 20, "x": 2, "y": 0, "theta": 300.0},
            {"peak_id": 30, "x": 4, "y": 4, "theta": 500.0},
        ]
    )

    edges, _ = build_peak_branch_graph(catalog, theta_gate=10.0)
    n_branches, labels = label_peak_branches(catalog, edges)

    assert edges == []
    assert n_branches == 3
    assert len(np.unique(labels)) == 3


def test_branch_graph_uses_four_connected_neighbors_and_theta_gate():
    catalog = pd.DataFrame(
        [
            {"peak_id": 0, "x": 0, "y": 0, "theta": 100.0},
            {"peak_id": 1, "x": 0, "y": 0, "theta": 400.0},
            {"peak_id": 2, "x": 1, "y": 0, "theta": 105.0},
            {"peak_id": 3, "x": 1, "y": 0, "theta": 405.0},
            # Diagonal neighbours are deliberately not connected.
            {"peak_id": 4, "x": 2, "y": 1, "theta": 105.0},
        ]
    )

    edges, deltas = build_peak_branch_graph(catalog, theta_gate=10.0)
    assert set(edges) == {(0, 2), (1, 3)}
    assert np.allclose(np.sort(deltas), [5.0, 5.0])

    n_branches, labels = label_peak_branches(catalog, edges)
    assert n_branches == 3
    by_id = dict(zip(catalog["peak_id"], labels))
    assert by_id[0] == by_id[2]
    assert by_id[1] == by_id[3]
    assert by_id[4] != by_id[0]


def test_spacing_diagnostics_use_pixel_medians_and_recommend_gate_between_limits():
    catalog = pd.DataFrame(
        [
            {"peak_id": 0, "x": 0, "y": 0, "theta": 100.0},
            {"peak_id": 1, "x": 0, "y": 0, "theta": 400.0},
            {"peak_id": 2, "x": 1, "y": 0, "theta": 105.0},
            {"peak_id": 3, "x": 1, "y": 0, "theta": 405.0},
        ]
    )

    stats = summarize_peak_spacing(catalog)
    row = stats["internal_gap_by_n"][0]
    assert row["n_peaks"] == 2
    assert row["pixel_count"] == 2
    assert row["median_gap"] == pytest.approx(300.0)
    assert row["q25_gap"] == pytest.approx(300.0)
    assert row["q75_gap"] == pytest.approx(300.0)
    assert row["min_pixel_median_gap"] == pytest.approx(300.0)
    assert stats["q90_neighbor_delta"] == pytest.approx(5.0)
    assert stats["recommended_theta_gate"] == pytest.approx(78.75)
    assert stats["theta_unit"] == "arcsec"


def test_q90_uses_reference_gate_edges_like_notebook():
    catalog = pd.DataFrame(
        [
            {"peak_id": 0, "x": 0, "y": 0, "theta": 100.0},
            {"peak_id": 1, "x": 0, "y": 0, "theta": 200.0},
            {"peak_id": 2, "x": 1, "y": 0, "theta": 105.0},
            {"peak_id": 3, "x": 1, "y": 0, "theta": 900.0},
        ]
    )

    stats = summarize_peak_spacing(catalog, reference_theta_gate=60.0)
    # The 700-arcsec unmatched assignment must not enter Q90: the notebook's
    # edge_deltas included only links accepted at theta_gate=60 arcsec.
    assert stats["q90_neighbor_delta"] == pytest.approx(5.0)
    assert stats["neighbor_edge_count_reference_gate"] == 1
    assert stats["reference_theta_gate"] == pytest.approx(60.0)


def test_notebook_reference_example_recommends_sixty_arcseconds():
    catalog = pd.DataFrame(
        [
            {"peak_id": 0, "x": 0, "y": 0, "theta": 100.0},
            {"peak_id": 1, "x": 0, "y": 0, "theta": 205.0},
            {"peak_id": 2, "x": 1, "y": 0, "theta": 145.0},
        ]
    )

    stats = summarize_peak_spacing(catalog, reference_theta_gate=60.0)
    assert stats["q90_neighbor_delta"] == pytest.approx(45.0)
    assert stats["min_internal_gap"] == pytest.approx(105.0)
    assert stats["recommended_theta_gate"] == pytest.approx(60.0)


def test_zero_internal_gap_is_not_discarded_from_diagnostics():
    catalog = pd.DataFrame(
        [
            {"peak_id": 0, "x": 0, "y": 0, "theta": 100.0},
            {"peak_id": 1, "x": 0, "y": 0, "theta": 100.0},
        ]
    )
    stats = summarize_peak_spacing(catalog)
    assert stats["min_internal_gap"] == pytest.approx(0.0)


def test_fitted_positions_keep_peak_identity_after_shuffled_fit_rows():
    detected = pd.DataFrame(
        [
            {"peak_id": 11, "x": 0, "y": 0, "theta_index": 3, "theta": 100.0, "component_id": 0},
            {"peak_id": 12, "x": 0, "y": 0, "theta_index": 14, "theta": 400.0, "component_id": 1},
        ]
    )
    fitted = pd.DataFrame(
        [
            {"x": 0, "y": 0, "component_id": 1, "theta0": 402.5},
            {"x": 0, "y": 0, "component_id": 0, "theta0": 98.5},
        ]
    )
    bundle = ArtifactBundle(tables={"peak_catalog": detected, "fitted_peak_catalog": fitted})

    result = fitted_peak_positions(bundle)
    by_id = result.set_index("peak_id")
    assert by_id.loc[11, "theta"] == pytest.approx(98.5)
    assert by_id.loc[12, "theta"] == pytest.approx(402.5)
    assert by_id.loc[11, "theta_detected"] == pytest.approx(100.0)
    assert by_id.loc[11, "theta_index"] == 3
    assert by_id.loc[12, "theta_index"] == 14


def test_fitted_catalog_existing_peak_ids_are_validated_and_preserved():
    detected = pd.DataFrame(
        [{"peak_id": 4, "x": 1, "y": 2, "theta": 10.0, "component_id": 0}]
    )
    fitted = pd.DataFrame(
        [{"peak_id": 4, "x": 1, "y": 2, "component_id": 0, "theta0": 12.0}]
    )
    result = fitted_peak_positions(
        ArtifactBundle(tables={"peak_catalog": detected, "fitted_peak_catalog": fitted})
    )
    assert result.loc[0, "peak_id"] == 4
    assert result.loc[0, "theta"] == pytest.approx(12.0)

    fitted.loc[0, "peak_id"] = 99
    with pytest.raises(ValueError, match="не совпадает"):
        fitted_peak_positions(
            ArtifactBundle(tables={"peak_catalog": detected, "fitted_peak_catalog": fitted})
        )


def test_fit_mapping_fails_closed_when_component_identity_is_ambiguous():
    detected = pd.DataFrame(
        [
            {"peak_id": 11, "x": 0, "y": 0, "theta": 100.0, "component_id": 0},
            {"peak_id": 12, "x": 0, "y": 0, "theta": 110.0, "component_id": 0},
        ]
    )
    fitted = pd.DataFrame(
        [{"x": 0, "y": 0, "component_id": 0, "theta0": 105.0}]
    )
    with pytest.raises(ValueError, match="неоднозначно"):
        fitted_peak_positions(
            ArtifactBundle(tables={"peak_catalog": detected, "fitted_peak_catalog": fitted})
        )


def test_branch_operation_persists_branch_catalog_and_provenance():
    catalog = pd.DataFrame(
        [
            {"peak_id": 0, "x": 0, "y": 0, "theta": 100.0},
            {"peak_id": 1, "x": 1, "y": 0, "theta": 105.0},
            {"peak_id": 2, "x": 4, "y": 4, "theta": 900.0},
        ]
    )
    positions = ArtifactBundle(
        tables={"peak_catalog": catalog},
        metadata={"grid_id": "grid-a", "position_kind": "detected"},
    )
    result = default_registry().execute(
        "peaks.branches", {"positions": positions}, {"theta_gate": 10.0}
    )

    assert result.output_ports == ("branches",)
    branch_catalog = result.bundle.tables["branch_catalog"]
    assert "branch_id" in branch_catalog
    assert result.bundle.metadata["n_branches"] == 2
    assert result.bundle.metadata["position_kind"] == "detected"
    assert result.bundle.metadata["theta_unit"] == "arcsec"


def test_branch_size_recommendation_uses_clear_multiplicative_gap_conservatively():
    from roc_mcs.peaks.branches import recommend_min_branch_events

    threshold, info = recommend_min_branch_events([1, 1, 5])
    assert threshold == 3
    assert info["status"] == "clear_gap"
    assert info["ratio"] == pytest.approx(5.0)

    threshold, info = recommend_min_branch_events([1, 2, 3, 4, 5])
    assert threshold == 1
    assert info["status"] == "no_clear_gap"

    # A single tiny component is not enough evidence for an automatic cut.
    threshold, info = recommend_min_branch_events([1, 30])
    assert threshold == 1
    assert info["status"] == "insufficient_components"


def test_branch_analysis_mask_preserves_legacy_catalogue_compatibility():
    from roc_mcs.peaks.branches import branch_analysis_mask

    legacy = pd.DataFrame({"branch_id": [0, 1, 2]})
    assert branch_analysis_mask(legacy).tolist() == [True, True, True]

    selected = pd.DataFrame({
        "branch_id": [0, 0, 1, 2],
        "analysis_included": [True, True, False, None],
    })
    assert branch_analysis_mask(selected).tolist() == [True, True, False, False]


def test_branch_operation_persists_non_destructive_automatic_size_selection():
    # One five-event spatial/angular component and two isolated singletons.
    # The latter remain in the catalogue but are not consumed downstream.
    catalog = pd.DataFrame([
        {"peak_id": i, "x": i, "y": 0, "theta": 100.0} for i in range(5)
    ] + [
        {"peak_id": 5, "x": 10, "y": 3, "theta": 700.0},
        {"peak_id": 6, "x": 20, "y": 7, "theta": 900.0},
    ])
    positions = ArtifactBundle(
        tables={"peak_catalog": catalog},
        metadata={"grid_id": "grid-size-selection", "position_kind": "detected"},
    )
    result = default_registry().execute(
        "peaks.branches", {"positions": positions}, {"theta_gate": 1.0}
    )
    branch_catalog = result.bundle.tables["branch_catalog"]
    branch_sizes = result.bundle.tables["branch_sizes"].sort_values("event_count")

    assert len(branch_catalog) == len(catalog)
    assert set(branch_catalog["peak_id"]) == set(catalog["peak_id"])
    assert branch_catalog["branch_id"].nunique() == 3
    assert branch_catalog.groupby("branch_id")["analysis_included"].nunique().max() == 1
    assert branch_catalog.loc[branch_catalog["analysis_included"], "branch_id"].nunique() == 1
    assert branch_catalog.loc[~branch_catalog["analysis_included"], "branch_id"].nunique() == 2
    assert branch_sizes["event_count"].tolist() == [1, 1, 5]
    assert result.bundle.metadata["n_branches"] == 3
    assert result.bundle.metadata["n_analysis_branches"] == 1
    assert result.bundle.metadata["n_excluded_branches"] == 2
    assert result.bundle.metadata["branch_selection"]["min_branch_events"] == 3


def test_manual_branch_size_threshold_overrides_automatic_selection():
    catalog = pd.DataFrame([
        {"peak_id": i, "x": i, "y": 0, "theta": 100.0} for i in range(5)
    ] + [
        {"peak_id": 5, "x": 10, "y": 3, "theta": 700.0},
        {"peak_id": 6, "x": 20, "y": 7, "theta": 900.0},
    ])
    positions = ArtifactBundle(tables={"peak_catalog": catalog}, metadata={"grid_id": "grid-manual"})
    result = default_registry().execute(
        "peaks.branches", {"positions": positions},
        {"theta_gate": 1.0, "min_branch_events": 1},
    )
    assert result.bundle.tables["branch_catalog"]["analysis_included"].all()
    assert result.bundle.metadata["branch_selection"]["mode"] == "manual"
    assert result.bundle.metadata["n_analysis_branches"] == 3


def test_branch_inclusion_flags_roundtrip_in_immutable_artifact(tmp_path):
    from roc_mcs.analysis.artifacts import ArtifactStore

    catalog = pd.DataFrame([
        {"peak_id": 0, "x": 0, "y": 0, "theta": 100.0},
        {"peak_id": 1, "x": 1, "y": 0, "theta": 105.0},
        {"peak_id": 2, "x": 10, "y": 3, "theta": 900.0},
    ])
    result = default_registry().execute(
        "peaks.branches",
        {"positions": ArtifactBundle(tables={"peak_catalog": catalog}, metadata={"grid_id": "roundtrip"})},
        {"theta_gate": 10.0},
    )
    store = ArtifactStore(tmp_path / "artifacts")
    artifact_id = store.write(result.bundle)
    restored = store.read(artifact_id)
    restored_catalog = restored.tables["branch_catalog"]

    assert restored_catalog["analysis_included"].dtype == bool
    assert restored_catalog["peak_id"].tolist() == [0, 1, 2]
    assert restored_catalog["branch_id"].nunique() == 2
    assert restored_catalog.groupby("branch_id")["analysis_included"].nunique().max() == 1
    assert restored.metadata["branch_selection"]["preserves_all_catalog_rows"] is True
