from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roc_mcs.processing.branch_correlation import (
    BranchCorrelationConfig,
    _assign_candidates,
    correlate_branch_catalog,
    directed_support,
    points_to_volume,
    register_branches_3d_coarse,
)


def _catalog_for_pair(theta_residual: float = 5.0) -> pd.DataFrame:
    source_grid = np.asarray(
        [[0, 0, 0], [1, 0, 1], [2, 1, 1], [3, 1, 2]], dtype=np.int64
    )
    partner_grid = source_grid + np.asarray([20, 1, 2], dtype=np.int64)
    source = pd.DataFrame({
        "peak_id": np.arange(100, 104),
        "branch_id": 10,
        "theta_index": source_grid[:, 0],
        "theta": source_grid[:, 0] * 15.0,
        "x": source_grid[:, 2],
        "y": source_grid[:, 1],
    })
    partner = pd.DataFrame({
        "peak_id": np.arange(200, 204),
        "branch_id": 20,
        "theta_index": partner_grid[:, 0],
        "theta": partner_grid[:, 0] * 15.0 + theta_residual,
        "x": partner_grid[:, 2],
        "y": partner_grid[:, 1],
    })
    return pd.concat([source, partner], ignore_index=True)


def test_points_to_volume_keeps_origin_and_collapses_duplicate_voxels():
    points = np.asarray([[4, 5, 6], [4, 5, 6], [5, 6, 7]], dtype=np.int64)
    volume, origin = points_to_volume(points)
    assert tuple(origin) == (4, 5, 6)
    assert volume.shape == (2, 2, 2)
    assert float(volume.sum()) == 2.0


def test_coarse_registration_returns_translation_to_apply_to_partner():
    source = np.asarray([[0, 0, 0], [1, 0, 1], [2, 1, 1], [3, 1, 2]])
    offset = np.asarray([20, 1, 2])
    partner = source + offset
    shift, overlap = register_branches_3d_coarse(source, partner)
    np.testing.assert_array_equal(shift, -offset)
    assert overlap == pytest.approx(4.0)



def test_sparse_exact_registration_matches_dense_integer_translation_without_large_bbox():
    # A huge, mostly empty local bounding box must not allocate a dense 3D grid.
    # The sparse vote is exact for binary occupancy and recovers the same shift.
    source = np.asarray([[0, 0, 0], [0, 0, 10_000_000], [2, 3, 4]], dtype=np.int64)
    offset = np.asarray([7, 11, 13], dtype=np.int64)
    partner = source + offset
    shift, overlap = register_branches_3d_coarse(source, partner)
    np.testing.assert_array_equal(shift, -offset)
    assert overlap == pytest.approx(3.0)


def test_sparse_translation_fallback_preserves_duplicate_voxel_semantics():
    source = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 5]], dtype=np.int64)
    partner = source + np.asarray([4, 2, 3], dtype=np.int64)
    shift, overlap = register_branches_3d_coarse(source, partner)
    np.testing.assert_array_equal(shift, [-4, -2, -3])
    assert overlap == pytest.approx(2.0)


def test_dense_fft_and_exact_sparse_fallback_choose_identical_peak_on_ties():
    rng = np.random.default_rng(8)
    for _ in range(100):
        source = rng.integers(0, 6, size=(12, 3), dtype=np.int64)
        partner = rng.integers(0, 6, size=(10, 3), dtype=np.int64)
        dense_shift, dense_score = register_branches_3d_coarse(source, partner)
        sparse_shift, sparse_score = register_branches_3d_coarse(
            source, partner, max_fft_memory_bytes=1
        )
        np.testing.assert_array_equal(dense_shift, sparse_shift)
        assert dense_score == pytest.approx(sparse_score, abs=0.0)

def test_directed_support_retains_target_minus_source_delta():
    source = np.asarray([[0.0, 10.0, 20.0], [30.0, 11.0, 21.0]])
    target = source + np.asarray([4.0, 0.5, -0.5])
    result = directed_support(
        source, target, xy_tolerance_px=1.5, theta_tolerance_arcsec=10.0
    )
    assert result["supported"].all()
    np.testing.assert_allclose(result["delta"], [[4.0, 0.5, -0.5]] * 2)


def test_assignment_marks_ambiguous_when_scores_are_not_separated():
    first = {
        "supported": np.asarray([True, True, False]),
        "score": np.asarray([0.10, 0.20, 1.0]),
    }
    second = {
        "supported": np.asarray([True, True, False]),
        "score": np.asarray([0.40, 0.22, 1.0]),
    }
    selected, ambiguous, residual, n_supported = _assign_candidates([first, second], 0.20)
    np.testing.assert_array_equal(selected, [0, -1, -1])
    np.testing.assert_array_equal(ambiguous, [False, True, False])
    np.testing.assert_array_equal(residual, [False, False, True])
    np.testing.assert_array_equal(n_supported, [2, 2, 0])


def test_continuous_theta_refinement_uses_opposite_sign_of_residual():
    catalog = _catalog_for_pair(theta_residual=5.0)
    result = correlate_branch_catalog(
        catalog,
        source_branch_id=10,
        partner_branch_ids=[20],
        config=BranchCorrelationConfig(
            theta_step_arcsec=15.0,
            xy_tolerance_px=1.5,
            theta_tolerance_arcsec=20.0,
            assignment_margin=0.20,
        ),
    )
    pair = result.pairs.iloc[0]
    assert pair["coarse_shift_theta_index"] == -20
    assert pair["theta_correction_arcsec"] == pytest.approx(-5.0)
    assert pair["theta_shift_final_arcsec"] == pytest.approx(-305.0)
    assert pair["theta_residual_mad_arcsec"] == pytest.approx(0.0)
    assert pair["assigned_source_count"] == 4
    assert result.metadata["assigned_event_count"] == 4
    assert set(result.assignments["assignment_class"]) == {"partner"}
    assert result.assignments["matched_partner_peak_id"].notna().all()


def test_nullable_catalog_values_are_rejected_with_clear_error():
    catalog = _catalog_for_pair()
    catalog.loc[0, "branch_id"] = pd.NA
    with pytest.raises(ValueError, match="branch_id.*NaN/Inf"):
        correlate_branch_catalog(
            catalog, source_branch_id=10, partner_branch_ids=[20],
            config=BranchCorrelationConfig(theta_step_arcsec=15.0),
        )


def test_config_validation_rejects_zero_step_and_invalid_margin():
    with pytest.raises(ValueError, match="не может быть равен нулю"):
        BranchCorrelationConfig(theta_step_arcsec=0).validated()
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        BranchCorrelationConfig(theta_step_arcsec=15.0, assignment_margin=1).validated()


def test_end_to_end_mixed_source_branch_is_split_between_two_distinct_partners():
    # The source branch is the union of two different 3D motifs. Each partner
    # is one motif translated independently, so the two coarse registrations
    # should recover different offsets and event-level support should separate
    # the source without forcing ambiguous points into either partner.
    motif_a = np.asarray(
        [[0, 0, 0], [1, 0, 1], [2, 1, 0], [3, 1, 2]], dtype=np.int64
    )
    motif_b = np.asarray(
        [[10, 5, 10], [11, 5, 11], [12, 6, 11], [13, 7, 12], [14, 7, 10]],
        dtype=np.int64,
    )
    source_grid = np.vstack([motif_a, motif_b])
    partner_a_grid = motif_a + np.asarray([20, 1, 2], dtype=np.int64)
    partner_b_grid = motif_b + np.asarray([-7, -1, 3], dtype=np.int64)

    source = pd.DataFrame({
        "peak_id": np.arange(100, 109),
        "branch_id": 10,
        "theta_index": source_grid[:, 0],
        "theta": source_grid[:, 0] * 15.0,
        "x": source_grid[:, 2],
        "y": source_grid[:, 1],
    })
    partner_a = pd.DataFrame({
        "peak_id": np.arange(200, 204),
        "branch_id": 20,
        "theta_index": partner_a_grid[:, 0],
        "theta": partner_a_grid[:, 0] * 15.0 + 3.0,
        "x": partner_a_grid[:, 2],
        "y": partner_a_grid[:, 1],
    })
    partner_b = pd.DataFrame({
        "peak_id": np.arange(300, 305),
        "branch_id": 30,
        "theta_index": partner_b_grid[:, 0],
        "theta": partner_b_grid[:, 0] * 15.0 - 4.0,
        "x": partner_b_grid[:, 2],
        "y": partner_b_grid[:, 1],
    })
    catalog = pd.concat([source, partner_a, partner_b], ignore_index=True)

    result = correlate_branch_catalog(
        catalog,
        source_branch_id=10,
        partner_branch_ids=[20, 30],
        config=BranchCorrelationConfig(
            theta_step_arcsec=15.0,
            xy_tolerance_px=1.5,
            theta_tolerance_arcsec=20.0,
            assignment_margin=0.20,
        ),
    )

    assigned = result.assignments.set_index("source_peak_id")["assigned_partner_branch_id"]
    assert assigned.loc[100:103].astype(int).tolist() == [20] * 4
    assert assigned.loc[104:108].astype(int).tolist() == [30] * 5
    assert set(result.assignments["assignment_class"]) == {"partner"}
    assert result.metadata["assigned_event_count"] == 9
    pairs = result.pairs.set_index("partner_branch_id")
    assert pairs.loc[20, "theta_correction_arcsec"] == pytest.approx(-3.0)
    assert pairs.loc[30, "theta_correction_arcsec"] == pytest.approx(4.0)



def _large_shifted_catalog(n_points: int = 35_000) -> pd.DataFrame:
    """Two 35k-event branches; dimensions match the reported 63x512x324 correlation."""
    rng = np.random.default_rng(20261124)
    shape = (32, 256, 162)
    flat = rng.choice(int(np.prod(shape)), size=n_points - 2, replace=False)
    coords = np.column_stack(np.unravel_index(flat, shape)).astype(np.int64)
    source_grid = np.vstack((coords, [[0, 0, 0], [31, 255, 161]]))
    source_grid = np.unique(source_grid, axis=0)
    shift = np.asarray([20, 4, 5], dtype=np.int64)
    partner_grid = np.vstack((
        source_grid + shift,
        np.asarray([source_grid[:, 0].min() + shift[0], 256 + shift[1], 162 + shift[2]]),
    ))

    source = pd.DataFrame({
        "peak_id": np.arange(len(source_grid), dtype=np.int64),
        "branch_id": np.zeros(len(source_grid), dtype=np.int64),
        "theta_index": source_grid[:, 0],
        "theta": source_grid[:, 0].astype(np.float64) * 15.0,
        "x": source_grid[:, 2],
        "y": source_grid[:, 1],
    })
    partner = pd.DataFrame({
        "peak_id": np.arange(1_000_000, 1_000_000 + len(partner_grid), dtype=np.int64),
        "branch_id": np.ones(len(partner_grid), dtype=np.int64),
        "theta_index": partner_grid[:, 0],
        "theta": partner_grid[:, 0].astype(np.float64) * 15.0 + 5.0,
        "x": partner_grid[:, 2],
        "y": partner_grid[:, 1],
    })
    return pd.concat([source, partner], ignore_index=True)


def test_large_35k_branches_use_full_float64_fft_and_recover_exact_shift():
    from roc_mcs.processing.branch_correlation import _register_branches_3d_coarse

    catalog = _large_shifted_catalog()
    source = catalog.loc[catalog["branch_id"] == 0, ["theta_index", "y", "x"]].to_numpy()
    partner = catalog.loc[catalog["branch_id"] == 1, ["theta_index", "y", "x"]].to_numpy()
    shift, peak, diagnostics = _register_branches_3d_coarse(source, partner)

    np.testing.assert_array_equal(shift, [-20, -4, -5])
    assert peak == len(source)
    assert diagnostics["method"] == "fft_float64"
    assert diagnostics["correlation_shape_theta_y_x"] == [63, 512, 324]
    assert diagnostics["fft_shape_theta_y_x"] == [64, 512, 324]
    assert diagnostics["estimated_peak_memory_bytes"] < 1024**3


def test_large_35k_zero_based_pair_end_to_end_keeps_every_peak():
    catalog = _large_shifted_catalog()
    result = correlate_branch_catalog(
        catalog,
        source_branch_id=0,
        partner_branch_ids=[1],
        config=BranchCorrelationConfig(
            theta_step_arcsec=15.0,
            xy_tolerance_px=1.5,
            theta_tolerance_arcsec=20.0,
            assignment_margin=0.20,
        ),
    )

    pair = result.pairs.iloc[0]
    assert pair["source_branch_id"] == 0
    assert pair["partner_branch_id"] == 1
    assert pair["coarse_registration_method"] == "fft_float64"
    assert pair["coarse_overlap_fraction"] == pytest.approx(1.0)
    assert pair["assigned_source_count"] == len(catalog[catalog["branch_id"] == 0])
    assert pair["theta_shift_final_arcsec"] == pytest.approx(-305.0)
    assert result.metadata["assigned_event_count"] == len(catalog[catalog["branch_id"] == 0])
    assert set(result.assignments["assignment_class"]) == {"partner"}


def test_fft_memory_estimate_and_explicit_budget_fallback_are_deterministic():
    source = np.asarray([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=np.int64)
    partner = source + np.asarray([7, 8, 9])
    dense_shift, dense_peak = register_branches_3d_coarse(source, partner)
    sparse_shift, sparse_peak = register_branches_3d_coarse(
        source, partner, max_fft_memory_bytes=1
    )
    np.testing.assert_array_equal(dense_shift, sparse_shift)
    assert dense_peak == sparse_peak == 3.0
