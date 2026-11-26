from __future__ import annotations

import json
import numpy as np
import pandas as pd

from roc_mcs.analysis.artifacts import ArtifactBundle, ArtifactStore
from roc_mcs.analysis.engine import WorkflowExecutor
from roc_mcs.analysis.operations import default_registry
from roc_mcs.analysis.preview_context import load_node_preview_context
from roc_mcs.analysis.workflow_history import NodeStatus, ReviewStatus, WorkflowHistory
from roc_mcs.analysis.workflow_preview import build_workflow_preview_state


def _make_branch_correlation_workflow(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    history = WorkflowHistory()

    scan_data = np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]], dtype=np.float32)
    scan_bundle = ArtifactBundle(
        arrays={
            "data_yxt": scan_data,
            "theta": np.array([10.0, 11.0, 12.0]),
            "x_coords": np.array([100.0, 101.0]),
            "y_coords": np.array([200.0]),
            "valid_mask_yxt": np.ones(scan_data.shape, dtype=bool),
        },
        metadata={"grid_id": "grid-test", "scan_key": "scan-test"},
    )
    scan_artifact = store.write(scan_bundle)
    scan_node = history.add_node(
        "preprocess.pool", "Scan input", output_refs={"scan": scan_artifact},
        status=NodeStatus.SUCCEEDED, review_status=ReviewStatus.ACCEPTED,
    )

    peak_catalog = pd.DataFrame([
        {"peak_id": 10, "x": 0, "y": 0, "theta": 10.5, "theta_index": 0},
        {"peak_id": 11, "x": 1, "y": 0, "theta": 11.5, "theta_index": 1},
    ])
    peaks_artifact = store.write(ArtifactBundle(
        tables={"peak_catalog": peak_catalog}, metadata={"position_kind": "detected"}
    ))
    peaks_node = history.add_node(
        "peaks.detect", "Detected peaks", parent_ids=[scan_node.id],
        input_refs={"scan": {"node_id": scan_node.id, "output": "scan"}},
        output_refs={"peaks": peaks_artifact}, status=NodeStatus.SUCCEEDED,
        review_status=ReviewStatus.ACCEPTED,
    )

    branch_catalog = peak_catalog.copy()
    branch_catalog["branch_id"] = np.array([3, 7], dtype=np.int64)
    branch_edges = pd.DataFrame([
        {"peak_id_a": 10, "peak_id_b": 11, "delta_theta": 1.0}
    ])
    branches_artifact = store.write(ArtifactBundle(
        tables={"branch_catalog": branch_catalog, "branch_edges": branch_edges},
        metadata={"position_kind": "detected", "n_branches": 2},
    ))
    branches_node = history.add_node(
        "peaks.branches", "Branches", parent_ids=[peaks_node.id],
        input_refs={"positions": {"node_id": peaks_node.id, "output": "peaks"}},
        output_refs={"branches": branches_artifact}, status=NodeStatus.SUCCEEDED,
        review_status=ReviewStatus.ACCEPTED,
    )

    correlation_bundle = ArtifactBundle(
        tables={
            "branch_correlation_pairs": pd.DataFrame([{
                "source_branch_id": 3, "partner_branch_id": 7,
                "source_event_count": 1, "partner_event_count": 1,
                "refined_shift_y_px": 0, "refined_shift_x_px": 0,
                "theta_shift_final_arcsec": -1.0,
                "assigned_source_count": 1,
                "refinement_status": "refined_on_exclusive_core",
            }]),
            "branch_correlation_assignments": pd.DataFrame([{
                "source_peak_id": 10, "assignment_class": "partner",
                "assigned_partner_branch_id": 7,
            }]),
            "branch_correlation_candidates": pd.DataFrame(),
        },
        metadata={"branch_correlation": {
            "source_branch_id": 3, "partner_branch_ids": [7],
            "assigned_event_count": 1, "ambiguous_event_count": 0,
            "residual_event_count": 0,
        }},
    )
    correlation_artifact = store.write(correlation_bundle)
    correlation_node = history.add_node(
        "peaks.branch_correlation", "Correlation", parent_ids=[branches_node.id],
        input_refs={"branches": {"node_id": branches_node.id, "output": "branches"}},
        output_refs={"branch_correlation": correlation_artifact},
        status=NodeStatus.SUCCEEDED, review_status=ReviewStatus.CANDIDATE,
    )
    executor = WorkflowExecutor(history, store, default_registry())
    return history, store, executor, branches_node, correlation_node, branch_catalog


def test_branch_correlation_preview_resolves_scan_via_branch_ancestry(tmp_path):
    history, _store, executor, _branches_node, correlation_node, expected_catalog = (
        _make_branch_correlation_workflow(tmp_path)
    )

    context = load_node_preview_context(correlation_node, history, executor)

    assert context["scan"].arrays["data_yxt"].shape == (1, 2, 3)
    assert context["peaks"] is not None
    assert context["fit"] is None
    assert context["branches"].tables["branch_catalog"].equals(expected_catalog)
    assert context["branch_correlation"].tables["branch_correlation_assignments"].shape[0] == 1


def test_branch_correlation_preview_keeps_original_points_edges_and_branch_map(tmp_path):
    history, _store, executor, _branches_node, correlation_node, _catalog = (
        _make_branch_correlation_workflow(tmp_path)
    )
    context = load_node_preview_context(correlation_node, history, executor)

    state = build_workflow_preview_state(correlation_node, context)

    assert state.default_map_key == "branches.count"
    assert set(state.peak_points["branch_id"].tolist()) == {3, 7}
    assert len(state.branch_edges) == 1
    np.testing.assert_array_equal(state.map_layer("branches.count").data_yx, [[1.0, 1.0]])
    assert state.metadata["branch_correlation"]["source_branch_id"] == 3


def test_branch_detection_preview_still_resolves_exact_scan_and_branches(tmp_path):
    history, _store, executor, branches_node, _correlation_node, _catalog = (
        _make_branch_correlation_workflow(tmp_path)
    )

    context = load_node_preview_context(branches_node, history, executor)

    assert context["scan"].arrays["data_yxt"].shape == (1, 2, 3)
    assert context["branches"].tables["branch_catalog"].shape[0] == 2
    assert context["peaks"] is not None
    assert context["fit"] is None


def test_branch_correlation_preview_rejects_broken_lineage_with_specific_error(tmp_path):
    history, _store, executor, _branches_node, correlation_node, _catalog = (
        _make_branch_correlation_workflow(tmp_path)
    )
    # The selected branch input and correlation output remain real; break only
    # the scan ancestry to exercise a clear lineage diagnostic, not a generic
    # ``ScanArtifact could not be resolved`` error.
    from dataclasses import replace

    branch_node = history.get(correlation_node.input_refs["branches"]["node_id"])
    position_node = history.get(branch_node.input_refs["positions"]["node_id"])
    detached_position = replace(
        position_node, parent_ids=(), input_refs_json="{}"
    )
    detached_branch = replace(
        branch_node, parent_ids=(detached_position.id,),
        input_refs_json=json.dumps({
            "positions": {"node_id": detached_position.id, "output": "peaks"}
        }),
    )
    # Keep the graph valid while removing only the Scan-bearing ancestry.
    detached_history = WorkflowHistory([detached_position, detached_branch, correlation_node])
    try:
        load_node_preview_context(
            detached_history.get(correlation_node.id), detached_history, executor
        )
    except Exception as exc:
        assert "предшествующий выход 'scan'" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("Ожидалась диагностируемая ошибка lineage")


def test_hkl_preview_resolves_original_branch_catalog_and_correlation_lineage(tmp_path):
    from roc_mcs.analysis.engine import build_input_refs
    from roc_mcs.analysis.hkl_preview import build_hkl_identification_preview
    from roc_mcs.analysis.workflow_history import ReviewStatus

    history, store, executor, branches_node, correlation_node, _catalog = (
        _make_branch_correlation_workflow(tmp_path)
    )
    history.set_review_status(correlation_node.id, ReviewStatus.ACCEPTED)
    refs = build_input_refs(history, correlation_node.id, "theory.identify_hkl", {})
    assert refs == {
        "branch_correlation": {"node_id": correlation_node.id, "output": "branch_correlation"},
        "branches": {"node_id": branches_node.id, "output": "branches"},
    }

    hkl_bundle = ArtifactBundle(
        tables={
            "hkl_theoretical_reflections": pd.DataFrame([{
                "hkl": "(111)", "d_A": 3.154, "theta_ka1_deg": 6.45,
                "delta_theta_arcsec": 141.0,
            }]),
            "hkl_bragg_rejected": pd.DataFrame(columns=["hkl", "d_A", "reason"]),
            "hkl_pair_measurements": pd.DataFrame([{
                "pair_id": "3↔7", "delta_theta_abs_arcsec": 140.0,
                "source_peak_id": 10, "matched_partner_peak_id": 11,
            }]),
            "hkl_pair_statistics": pd.DataFrame([{
                "pair_id": "3↔7", "matched_assignment_count": 1,
                "delta_theta_abs_median_arcsec": 140.0,
                "identification_status": "insufficient_matches",
            }]),
            "hkl_pair_ranking": pd.DataFrame(columns=[
                "pair_id", "rank", "hkl", "delta_theta_arcsec",
                "difference_from_exp_median_arcsec",
            ]),
        },
        metadata={"theory": {
            "crystal_name": "CaF₂ · флюорит", "lattice_a_A": 5.4635,
            "lambda_ka1_A": 0.7093000, "lambda_ka2_A": 0.7135900,
            "histogram_bin_width_arcsec": 7.0,
        }},
    )
    hkl_artifact = store.write(hkl_bundle)
    # The dedicated result path must be able to open the HKL artifact directly;
    # it must not need a Scan/branch lineage or generic ROI preview construction.
    from roc_mcs.analysis.hkl_preview import load_hkl_identification_preview
    direct_view = load_hkl_identification_preview(store, hkl_artifact)
    assert direct_view.pair_statistics.shape[0] == 1
    assert direct_view.pair_statistics.iloc[0]["pair_id"] == "3↔7"

    hkl_node = history.add_node(
        "theory.identify_hkl", "HKL · CaF₂", parent_ids=[correlation_node.id],
        input_refs=refs, output_refs={"hkl_analysis": hkl_artifact},
        status=NodeStatus.SUCCEEDED, review_status=ReviewStatus.ACCEPTED,
    )

    context = load_node_preview_context(hkl_node, history, executor)
    assert context["scan"].arrays["data_yxt"].shape == (1, 2, 3)
    assert context["branches"].tables["branch_catalog"].shape[0] == 2
    assert context["branch_correlation"].tables["branch_correlation_pairs"].shape[0] == 1
    assert context["hkl_analysis"].tables["hkl_pair_statistics"].shape[0] == 1

    state = build_workflow_preview_state(hkl_node, context)
    assert state.hkl_identification_preview is not None
    assert state.hkl_identification_preview.pair_statistics.iloc[0]["pair_id"] == "3↔7"
    assert state.branch_correlation_preview is not None
    assert state.branch_correlation_preview.pairs
    assert state.default_map_key == "branches.count"
    assert set(state.peak_points["branch_id"].tolist()) == {3, 7}


def test_hkl_preview_model_detaches_tables_from_artifact_bundle():
    from roc_mcs.analysis.hkl_preview import build_hkl_identification_preview

    bundle = ArtifactBundle(
        tables={
            "hkl_theoretical_reflections": pd.DataFrame([{
                "hkl": "(111)", "d_A": 3.0, "theta_ka1_deg": 6.0,
                "delta_theta_arcsec": 140.0,
            }]),
            "hkl_bragg_rejected": pd.DataFrame(columns=["hkl", "d_A", "reason"]),
            "hkl_pair_measurements": pd.DataFrame(columns=[
                "pair_id", "delta_theta_abs_arcsec", "source_peak_id", "matched_partner_peak_id",
            ]),
            "hkl_pair_statistics": pd.DataFrame(columns=[
                "pair_id", "matched_assignment_count", "delta_theta_abs_median_arcsec",
                "identification_status",
            ]),
            "hkl_pair_ranking": pd.DataFrame(columns=[
                "pair_id", "rank", "hkl", "delta_theta_arcsec",
                "difference_from_exp_median_arcsec",
            ]),
        },
        metadata={"theory": {"crystal_name": "CaF₂"}},
    )
    view = build_hkl_identification_preview(bundle)
    view.theoretical_reflections.loc[0, "hkl"] = "MUTATED"
    assert bundle.tables["hkl_theoretical_reflections"].loc[0, "hkl"] == "(111)"
