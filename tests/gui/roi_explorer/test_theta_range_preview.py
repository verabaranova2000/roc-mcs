import numpy as np
import pytest

from roc_mcs.gui.roi_explorer.theta_range_preview import (
    complementary_theta_range,
    infer_theta_projection_kind,
    normalize_theta_axis,
    reduce_theta_stack,
)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("mean", "mean"),
        ("maximum", "max"),
        ("sum", "sum"),
        ("mean projection", "mean"),
        ("Projection_Max", "max"),
        ("scan.intensity_sum", "sum"),
        ("scan.measured_intensity_sum", "sum"),
    ],
)
def test_projection_kind_recognizes_only_explicit_reductions(identifier, expected):
    assert infer_theta_projection_kind(identifier) == expected


@pytest.mark.parametrize(
    "identifier",
    [
        "peaks.count",
        "spikes.count",
        "branches.count",
        "fit.n_fitted_map",
        "peak mean projection",
        "residual_sum_projection",
        "unknown intensity image",
    ],
)
def test_processing_results_are_not_classified_as_theta_projections(identifier):
    assert infer_theta_projection_kind(identifier) is None


def test_conflicting_identifiers_are_ambiguous():
    assert infer_theta_projection_kind("mean", "max projection") is None


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ((45, 200), (0, 45)),
        ((0, 45), (45, 200)),
        ((0, 200), None),
        ((40, 150), None),
        ((0, 1), (1, 200)),
        ((199, 200), (0, 199)),
    ],
)
def test_complementary_range_shares_cut_boundary(current, expected):
    assert complementary_theta_range(*current, 201) == expected


def test_complementary_range_rejects_invalid_indices():
    assert complementary_theta_range(-1, 4, 10) is None
    assert complementary_theta_range(4, 4, 10) is None
    assert complementary_theta_range(2, 1, 10) is None


def test_normalize_theta_axis_accepts_both_monotonic_directions():
    ascending, order_a = normalize_theta_axis([-2, -1, 0, 1])
    descending, order_d = normalize_theta_axis([1, 0, -1, -2])
    np.testing.assert_array_equal(ascending, [-2, -1, 0, 1])
    np.testing.assert_array_equal(descending, ascending)
    np.testing.assert_array_equal(order_a, [0, 1, 2, 3])
    np.testing.assert_array_equal(order_d, [3, 2, 1, 0])


@pytest.mark.parametrize(
    "axis",
    [
        [0],
        [0, 1, 1],
        [0, 2, 1],
        [0, np.nan, 2],
    ],
)
def test_normalize_theta_axis_rejects_short_nonmonotonic_or_invalid_axes(axis):
    with pytest.raises(ValueError):
        normalize_theta_axis(axis)


def test_projection_reductions_respect_nan_and_validity_mask():
    stack = np.array(
        [
            [[1.0, np.nan], [5.0, 3.0]],
            [[3.0, 2.0], [np.nan, 9.0]],
            [[5.0, 4.0], [8.0, 12.0]],
        ]
    )
    mask = np.ones(stack.shape, dtype=bool)
    mask[2, 0, 0] = False
    mask[:, 1, 1] = False

    np.testing.assert_allclose(
        reduce_theta_stack(stack, "sum", mask),
        [[4.0, 6.0], [13.0, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        reduce_theta_stack(stack, "mean", mask),
        [[2.0, 3.0], [6.5, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        reduce_theta_stack(stack, "max", mask),
        [[3.0, 4.0], [8.0, np.nan]],
        equal_nan=True,
    )


def test_projection_reducer_rejects_mismatched_masks_and_invalid_modes():
    stack = np.ones((2, 3, 4), dtype=float)
    with pytest.raises(ValueError):
        reduce_theta_stack(stack, "sum", np.ones((2, 3, 3), dtype=bool))
    with pytest.raises(ValueError):
        reduce_theta_stack(stack, "median")
