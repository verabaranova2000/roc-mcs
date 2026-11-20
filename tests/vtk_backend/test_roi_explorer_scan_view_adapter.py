from __future__ import annotations

import numpy as np
import pytest

from roc_mcs.gui.roi_explorer.vtk_backend.scan_view_adapter import build_volume_payload


def test_adapter_preserves_yxt_values_geometry_and_source():
    source = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    before = source.copy()
    theta = np.linspace(-2.0, 2.0, 5)
    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = np.array([1.0, 2.0, 3.0])
    payload = build_volume_payload(source, theta, 2, x, y, scan_key="entry472")

    assert payload.volume.shape == (4, 3, 5)
    np.testing.assert_array_equal(payload.volume, source.transpose(1, 0, 2))
    np.testing.assert_array_equal(payload.projection, source.transpose(1, 0, 2).sum(axis=2))
    np.testing.assert_array_equal(payload.x, x)
    np.testing.assert_array_equal(payload.y, y)
    np.testing.assert_array_equal(payload.theta, theta)
    np.testing.assert_array_equal(source, before)
    assert not payload.volume.flags.writeable
    assert not payload.projection.flags.writeable


def test_adapter_masks_invalid_samples_before_projection():
    source = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    mask = np.ones(source.shape, dtype=bool)
    mask[1, 2, :] = False
    mask[0, 1, 1] = False
    payload = build_volume_payload(
        source, np.arange(4, dtype=float), 2, np.arange(3, dtype=float),
        np.arange(2, dtype=float), valid_mask=mask,
    )
    assert np.isnan(payload.volume[2, 1, :]).all()
    assert np.isnan(payload.projection[2, 1])
    assert np.isnan(payload.volume[1, 0, 1])
    assert payload.projection[1, 0] == pytest.approx(source[0, 1, [0, 2, 3]].sum())


def test_adapter_smoothing_is_display_copy_and_keeps_invalid_sites():
    source = np.zeros((5, 5, 4), dtype=np.float32)
    source[2, 2, :] = 100.0
    mask = np.ones(source.shape, dtype=bool)
    mask[0, 0, :] = False
    before = source.copy()
    payload = build_volume_payload(
        source, np.arange(4, dtype=float), 2, np.arange(5, dtype=float),
        np.arange(5, dtype=float), valid_mask=mask, smoothing_sigma=0.8,
    )
    assert payload.smoothing_sigma == pytest.approx(0.8)
    assert np.isnan(payload.volume[0, 0, :]).all()
    assert np.isnan(payload.projection[0, 0])
    assert payload.volume[2, 2, 1] < 100.0
    np.testing.assert_array_equal(source, before)


def test_adapter_rejects_invalid_smoothing_and_bad_axis():
    source = np.ones((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="smoothing_sigma"):
        build_volume_payload(source, np.arange(4), 2, np.arange(3), np.arange(2), smoothing_sigma=6)
    with pytest.raises(ValueError, match="ось θ"):
        build_volume_payload(source, np.arange(2), 1, np.arange(3), np.arange(2))


def test_scan_view_adapter_rejects_singleton_axes_before_vtk_setup():
    raw = np.ones((1, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="минимум 2 узла"):
        build_volume_payload(raw, np.arange(4), 2, np.arange(3), np.arange(1))


def test_scan_view_adapter_rejects_non_boolean_validity_mask():
    raw = np.ones((3, 4, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="bool dtype"):
        build_volume_payload(raw, np.arange(5), 2, np.arange(4), np.arange(3), valid_mask=np.ones(raw.shape, dtype=np.uint8))
