from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from roc_mcs.analysis.branch_correlation_preview import build_branch_correlation_preview


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peak_id": [101, 102, 201, 202, 301],
            "branch_id": [10, 10, 20, 20, 30],
            "x": [1, 2, 1, 2, 0],
            "y": [0, 0, 0, 0, 1],
            "theta": [0.0, 1.0, 0.5, 1.5, 0.2],
        }
    )


def _bundle(*, legacy: bool = False):
    pairs = pd.DataFrame(
        {
            "source_branch_id": [10],
            "partner_branch_id": [20],
            "source_event_count": [2],
            "partner_event_count": [2],
            "refined_shift_x_px": [-1],
            "refined_shift_y_px": [2],
            "theta_shift_final_arcsec": [-7.5],
            "theta_residual_mad_arcsec": [1.25],
            "final_supported_count": [2],
            "refinement_status": ["refined"],
        }
    )
    assignment_data = {
        "source_peak_id": [101, 102],
        "source_branch_id": [10, 10],
        "assignment_class": ["partner", "residual"],
        "assigned_partner_branch_id": pd.array([20, None], dtype="Int64"),
    }
    if not legacy:
        assignment_data["matched_partner_peak_id"] = pd.array([201, None], dtype="Int64")
    assignments = pd.DataFrame(assignment_data)
    candidates = pd.DataFrame(
        {
            "source_peak_id": [101, 102],
            "source_branch_id": [10, 10],
            "partner_branch_id": [20, 20],
            "matched_partner_peak_id": [201, 202],
            "supported": [True, True],
            "assigned": [True, False],
        }
    )
    return SimpleNamespace(
        tables={
            "branch_correlation_pairs": pairs,
            "branch_correlation_assignments": assignments,
            "branch_correlation_candidates": candidates,
        },
        metadata={"branch_correlation": {"source_branch_id": 10}},
    )


def test_preview_uses_only_stored_peak_id_assignments_and_physical_axes():
    preview = build_branch_correlation_preview(
        _catalog(), _bundle(), np.asarray([100.0, 110.0, 120.0]), np.asarray([200.0, 210.0])
    )
    assert preview.source_branch_id == 10
    assert len(preview.pairs) == 1
    pair = preview.pairs[0]
    assert pair.matched_total_count == 1
    assert pair.candidate_total_count == 1
    np.testing.assert_allclose(
        pair.matched_segments,
        np.asarray([[[110.0, 200.0, 0.0], [110.0, 200.0, 0.5]]]),
    )
    np.testing.assert_allclose(
        pair.candidate_segments,
        np.asarray([[[120.0, 200.0, 1.0], [120.0, 200.0, 1.5]]]),
    )
    assert pair.matched_peak_ids.tolist() == [101, 201]
    assert pair.summary["shift_x_px"] == -1
    assert pair.summary["shift_y_px"] == 2
    assert pair.summary["shift_theta_arcsec"] == -7.5
    assert "партнёру" in pair.summary["shift_direction"]


def test_preview_does_not_infer_missing_legacy_target_peak_ids():
    preview = build_branch_correlation_preview(
        _catalog(), _bundle(legacy=True), np.arange(3.0), np.arange(2.0)
    )
    assert preview.total_matched == 0
    assert preview.pairs[0].matched_segments.shape == (0, 2, 3)
    assert any("matched_partner_peak_id" in warning for warning in preview.warnings)


def test_preview_keeps_exact_match_counts_when_display_geometry_is_capped():
    catalog = _catalog()
    bundle = _bundle()
    # Duplicate accepted assignment rows as independent source events, while
    # retaining unique catalogue IDs and targets for an exact geometry cap.
    more = pd.DataFrame(
        {
            "peak_id": [103, 104, 203, 204],
            "branch_id": [10, 10, 20, 20],
            "x": [0, 1, 0, 1],
            "y": [1, 1, 1, 1],
            "theta": [0.0, 1.0, 0.5, 1.5],
        }
    )
    catalog = pd.concat([catalog, more], ignore_index=True)
    assignment = bundle.tables["branch_correlation_assignments"]
    bundle.tables["branch_correlation_assignments"] = pd.concat(
        [assignment, pd.DataFrame({
            "source_peak_id": [103, 104],
            "source_branch_id": [10, 10],
            "assignment_class": ["partner", "partner"],
            "assigned_partner_branch_id": pd.array([20, 20], dtype="Int64"),
            "matched_partner_peak_id": pd.array([203, 204], dtype="Int64"),
        })], ignore_index=True
    )
    preview = build_branch_correlation_preview(
        catalog, bundle, np.arange(3.0), np.arange(2.0), max_matched_links=2
    )
    pair = preview.pairs[0]
    assert pair.matched_total_count == 3
    assert pair.matched_segments.shape[0] == 2
    assert pair.matched_truncated is True
    # Core/residual status remains exact even when line geometry is thinned.
    assert pair.matched_peak_ids.tolist() == [101, 103, 104, 201, 203, 204]


def test_preview_warns_instead_of_fabricating_links_for_missing_peak_ids():
    bundle = _bundle()
    bundle.tables["branch_correlation_assignments"].loc[0, "matched_partner_peak_id"] = 999_999
    preview = build_branch_correlation_preview(
        _catalog(), bundle, np.arange(3.0), np.arange(2.0)
    )
    assert preview.total_matched == 0
    assert preview.pairs[0].summary["assignment_row_count"] == 1
    assert preview.pairs[0].summary["invalid_assignment_count"] == 1
    assert any("не удалось связать" in warning for warning in preview.warnings)


def test_preview_arrays_are_read_only_and_source_tables_are_not_mutated():
    catalog = _catalog()
    bundle = _bundle()
    catalog_before = catalog.copy(deep=True)
    assignments_before = bundle.tables["branch_correlation_assignments"].copy(deep=True)
    preview = build_branch_correlation_preview(
        catalog, bundle, np.arange(3.0), np.arange(2.0)
    )
    assert not preview.pairs[0].matched_segments.flags.writeable
    assert not preview.pairs[0].matched_peak_ids.flags.writeable
    pd.testing.assert_frame_equal(catalog, catalog_before)
    pd.testing.assert_frame_equal(
        bundle.tables["branch_correlation_assignments"], assignments_before
    )



def test_preview_does_not_display_missing_registration_shift_as_zero():
    bundle = _bundle()
    bundle.tables["branch_correlation_pairs"].loc[0, "refined_shift_x_px"] = np.nan
    preview = build_branch_correlation_preview(
        _catalog(), bundle, np.arange(3.0), np.arange(2.0)
    )
    assert preview.pairs[0].summary["shift_x_px"] is None
