from __future__ import annotations

import numpy as np
import pytest

from roc_mcs.gui.roi_explorer.hkl_plot_utils import (
    build_histogram_edges,
    histogram_view_bounds,
    zoomed_view_bounds,
)


def test_histogram_edges_cover_all_finite_values_and_keep_requested_width():
    values = np.asarray([1.2, 2.1, 8.9, np.nan, np.inf, 14.0])
    edges = build_histogram_edges(values, 2.0, max_bins=100)

    assert edges[0] <= 1.2
    assert edges[-1] >= 14.0
    assert np.all(np.diff(edges) > 0)
    counts, _ = np.histogram(values[np.isfinite(values)], bins=edges)
    assert counts.sum() == 4
    assert np.allclose(np.diff(edges), 2.0)


def test_capped_histogram_expands_width_without_dropping_upper_endpoint():
    values = np.asarray([0.3, 100_000.0])
    edges = build_histogram_edges(values, 0.1, max_bins=20)

    assert len(edges) == 21
    assert edges[0] <= values.min()
    assert edges[-1] >= values.max()
    counts, _ = np.histogram(values, bins=edges)
    assert counts.sum() == values.size
    assert np.all(np.diff(edges) > 0)


def test_constant_sample_gets_one_finite_bin_and_valid_view_bounds():
    values = np.asarray([12.5, 12.5, 12.5])
    edges = build_histogram_edges(values, 0.5, max_bins=100)

    assert len(edges) == 2
    assert edges[0] < values[0] < edges[1]
    counts, _ = np.histogram(values, bins=edges)
    assert counts.tolist() == [3]

    low, high = histogram_view_bounds(values, edges)
    assert 0.0 <= low < values[0] < high


def test_view_bounds_include_complete_bin_range_and_effective_bin_width():
    values = np.asarray([10.2, 10.3])
    edges = build_histogram_edges(values, 100.0, max_bins=10)

    low, high = histogram_view_bounds(values, edges, padding_fraction=0.1)
    effective_width = edges[-1] - edges[-2]
    expected_padding = effective_width * 0.1
    assert low == pytest.approx(max(0.0, edges[0] - expected_padding))
    assert high == pytest.approx(edges[-1] + expected_padding)


def test_invalid_histogram_parameters_are_rejected():
    with pytest.raises(ValueError, match="requested_width"):
        build_histogram_edges(np.asarray([1.0, 2.0]), 0.0, max_bins=10)
    with pytest.raises(ValueError, match="max_bins"):
        build_histogram_edges(np.asarray([1.0, 2.0]), 1.0, max_bins=0)


def test_view_bounds_do_not_clip_histogram_bins_or_allow_negative_absolute_separation():
    values = np.asarray([0.01, 0.02])
    edges = build_histogram_edges(values, 1.0, max_bins=10)

    low, high = histogram_view_bounds(values, edges)
    assert low == 0.0
    assert high >= edges[-1]


def test_zoomed_view_bounds_pan_without_changing_full_range():
    full = (0.0, 100.0)
    assert zoomed_view_bounds(full, 1.0, 0.5) == full
    assert zoomed_view_bounds(full, 4.0, 0.0) == pytest.approx((0.0, 25.0))
    assert zoomed_view_bounds(full, 4.0, 0.5) == pytest.approx((37.5, 62.5))
    assert zoomed_view_bounds(full, 4.0, 1.0) == pytest.approx((75.0, 100.0))


def test_zoomed_view_bounds_validates_arguments_and_clamps_scroll_fraction():
    assert zoomed_view_bounds((2.0, 6.0), 2.0, -4.0) == pytest.approx((2.0, 4.0))
    assert zoomed_view_bounds((2.0, 6.0), 2.0, 4.0) == pytest.approx((4.0, 6.0))
    with pytest.raises(ValueError, match="full_bounds"):
        zoomed_view_bounds((1.0, 1.0), 2.0, 0.5)
    with pytest.raises(ValueError, match="zoom_factor"):
        zoomed_view_bounds((0.0, 1.0), 0.5, 0.5)


def test_suggested_histogram_width_is_adaptive_and_robust_to_extreme_outliers():
    from roc_mcs.gui.roi_explorer.hkl_plot_utils import suggest_histogram_bin_width

    clustered = np.linspace(0.10, 0.30, 200)
    width = suggest_histogram_bin_width(clustered)
    width_with_outlier = suggest_histogram_bin_width(
        np.concatenate((clustered, [1_000_000.0]))
    )
    assert 0.01 <= width <= 100.0
    assert width < 1.0
    assert width_with_outlier < 1.0


def test_suggested_histogram_width_handles_constant_and_empty_samples():
    from roc_mcs.gui.roi_explorer.hkl_plot_utils import suggest_histogram_bin_width

    assert suggest_histogram_bin_width(np.asarray([2.0, 2.0, np.nan])) == pytest.approx(0.25)
    assert suggest_histogram_bin_width(np.asarray([])) == pytest.approx(0.25)


def test_comparison_axis_uses_symlog_only_for_wide_dynamic_range():
    from roc_mcs.gui.roi_explorer.hkl_plot_utils import comparison_axis_parameters

    assert comparison_axis_parameters(np.asarray([0.0, 1.0, 3.0, 8.0])) == ("linear", None)
    mode, linthresh = comparison_axis_parameters(np.asarray([0.0, 0.01, 0.1, 1.0, 1000.0]))
    assert mode == "symlog"
    assert linthresh is not None and linthresh > 0.0


def test_comparison_axis_does_not_ignore_one_distant_theoretical_candidate():
    from roc_mcs.gui.roi_explorer.hkl_plot_utils import comparison_axis_parameters

    # A single remote candidate among many ordinary values must not compress
    # all measured Q10–Q90 intervals into an almost invisible segment.
    values = np.concatenate((np.linspace(0.9, 1.1, 100), [100_000.0]))
    mode, linthresh = comparison_axis_parameters(values)

    assert mode == "symlog"
    assert linthresh is not None and 0.9 <= linthresh <= 1.1
