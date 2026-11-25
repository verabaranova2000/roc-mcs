import numpy as np
import pytest

from roc_mcs.gui.roi_explorer.theta_range_preview import (
    ThetaProjectionAccumulator,
    reduce_theta_stack,
)


@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("kind", ["mean", "sum", "max"])
def test_incremental_projection_matches_reference(axis, descending, kind):
    rng = np.random.default_rng(240519)
    shape = (19, 13, 11) if axis == 0 else (13, 11, 19)
    data = rng.normal(size=shape).astype(np.float32)
    data.ravel()[::31] = np.nan
    mask = rng.random(shape) > 0.17
    count = shape[axis]
    order = np.arange(count, dtype=np.int64)
    if descending:
        order = order[::-1].copy()

    accumulator = ThetaProjectionAccumulator()
    # Includes one-point edge shifts, expansion, contraction and a jump to
    # the opposite one-sided interval.
    ranges = [(2, count - 2), (3, count - 2), (3, count - 3),
              (1, count - 3), (0, count - 3), (0, 4), (4, count - 1)]
    for low, high in ranges:
        actual = accumulator.project(
            data, order, axis, kind, low, high, mask,
            source_token=("synthetic", axis, descending),
        )
        source_indices = order[low:high + 1]
        if axis == 0:
            reference_stack = np.take(data, source_indices, axis=0)
            reference_mask = np.take(mask, source_indices, axis=0)
        else:
            reference_stack = np.take(data, source_indices, axis=2).transpose(2, 0, 1)
            reference_mask = np.take(mask, source_indices, axis=2).transpose(2, 0, 1)
        expected = reduce_theta_stack(reference_stack, kind, reference_mask)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_mean_and_sum_reuse_output_buffer_for_adjacent_drag_steps():
    rng = np.random.default_rng(90210)
    data = rng.normal(size=(32, 48, 72)).astype(np.float32)
    order = np.arange(data.shape[-1], dtype=np.int64)
    accumulator = ThetaProjectionAccumulator()

    first = accumulator.project(data, order, 2, "mean", 8, 60)
    second = accumulator.project(data, order, 2, "mean", 9, 60)
    assert first is second

    third = accumulator.project(data, order, 2, "sum", 9, 60)
    # The accumulator intentionally reuses its 2D output buffer across modes.
    assert third is second
    expected = reduce_theta_stack(data[:, :, 9:61].transpose(2, 0, 1), "sum")
    np.testing.assert_allclose(third, expected, rtol=1e-12, atol=1e-12)


def test_max_and_integer_data_keep_nan_mask_semantics():
    data = np.arange(4 * 5 * 7, dtype=np.int16).reshape(4, 5, 7)
    mask = np.ones(data.shape, dtype=bool)
    mask[2, 3, :] = False
    result = ThetaProjectionAccumulator().project(
        data, np.arange(7), 2, "max", 0, 6, mask
    )
    assert np.isnan(result[2, 3])
    assert result[0, 0] == float(np.max(data[0, 0, :]))


def test_source_change_invalidates_accumulated_state():
    first = np.ones((4, 5, 9), dtype=np.float64)
    second = first * 3.0
    order = np.arange(9)
    accumulator = ThetaProjectionAccumulator()
    # Results share the accumulator's reusable output array, so snapshot the
    # first result before asking it to project a different source.
    got_first = accumulator.project(first, order, 2, "sum", 1, 7).copy()
    got_second = accumulator.project(second, order, 2, "sum", 1, 7).copy()
    np.testing.assert_allclose(got_first, np.full((4, 5), 7.0))
    np.testing.assert_allclose(got_second, np.full((4, 5), 21.0))


@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parametrize("descending", [False, True])
def test_max_incremental_repairs_match_reference_for_random_edge_motion(axis, descending):
    rng = np.random.default_rng(77123 + axis + int(descending))
    shape = (23, 17, 15) if axis == 0 else (17, 15, 23)
    # Quantization deliberately creates repeated maxima so the max-count path
    # (not just the sparse unique-maximum repair) is exercised.
    data = rng.integers(-7, 9, size=shape, dtype=np.int16).astype(np.float32)
    data.ravel()[::29] = np.nan
    mask = rng.random(shape) > 0.12
    count = shape[axis]
    order = np.arange(count, dtype=np.int64)
    if descending:
        order = order[::-1].copy()

    accumulator = ThetaProjectionAccumulator()
    low, high = 2, count - 3
    for step in range(60):
        if step:
            low = int(np.clip(low + rng.integers(-2, 3), 0, count - 3))
            high = int(np.clip(high + rng.integers(-2, 3), low + 1, count - 1))
        actual = accumulator.project(
            data, order, axis, "max", low, high, mask,
            source_token=("random-max", axis, descending),
        )
        selected = order[low:high + 1]
        if axis == 0:
            slab = np.take(data, selected, axis=0)
            selected_mask = np.take(mask, selected, axis=0)
        else:
            slab = np.take(data, selected, axis=2).transpose(2, 0, 1)
            selected_mask = np.take(mask, selected, axis=2).transpose(2, 0, 1)
        expected = reduce_theta_stack(slab, "max", selected_mask)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_accumulator_rejects_non_monotonic_theta_index_order():
    data = np.ones((5, 4, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="непрерывный возрастающий"):
        ThetaProjectionAccumulator().project(
            data,
            np.array([0, 2, 1, 3, 4, 5, 6]),
            2,
            "sum",
            0,
            6,
        )
