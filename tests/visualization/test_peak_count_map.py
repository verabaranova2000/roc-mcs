import numpy as np
import pytest

from roc_mcs.visualization.peak_count_map import (
    PEAK_COUNT_COLORS,
    integrated_intensity_map,
    peak_count_rgba,
)


def test_integrated_intensity_map_uses_theta_axis_and_combines_validity_masks():
    data = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]],
            [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
        ]
    )
    theta = np.array([0.0, 1.0, 3.0])
    valid = np.ones(data.shape, dtype=bool)
    valid[0, 1, 1] = False
    spatial = np.array([[True, True], [True, False]])

    intensity, valid_yx = integrated_intensity_map(
        data,
        theta,
        valid_mask_yxt=valid,
        analysis_valid_mask_yx=spatial,
    )

    assert intensity.shape == (2, 2)
    assert valid_yx.tolist() == [[True, False], [True, False]]
    assert intensity[0, 0] == pytest.approx(6.5)
    assert np.isnan(intensity[0, 1])
    assert np.isnan(intensity[1, 1])


def test_integrated_intensity_rejects_non_monotonic_theta():
    with pytest.raises(ValueError, match="строго возрастать"):
        integrated_intensity_map(np.ones((2, 2, 3)), [0.0, 2.0, 1.0])


def test_peak_count_rgba_encodes_peak_count_by_hue_and_intensity_by_alpha():
    counts = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 8]], dtype=float)
    intensity = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6]], dtype=float)
    dark, _ = peak_count_rgba(counts, intensity, theme="dark")
    light, _ = peak_count_rgba(counts, intensity, theme="light")

    np.testing.assert_allclose(dark[0, 0, :3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(light[0, 0, :3], [1.0, 1.0, 1.0])
    for index, color in enumerate(PEAK_COUNT_COLORS, start=1):
        rgb = np.array([int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
        np.testing.assert_allclose(dark[0, index, :3], rgb)
    assert dark[1, 5, 3] == pytest.approx(1.0)
    assert dark[0, 1, 3] < dark[0, 2, 3]
    assert dark[0, 1, 3] <= dark[0, 2, 3]


def test_peak_count_rgba_masks_invalid_pixels_as_transparent():
    counts = np.array([[1.0, 2.0]])
    intensity = np.array([[1.0, 2.0]])
    rgba, _ = peak_count_rgba(
        counts, intensity, valid_mask_yx=np.array([[True, False]])
    )
    assert rgba[0, 0, 3] > 0.0
    assert rgba[0, 1, 3] == 0.0


def test_peak_count_rgba_rejects_unknown_theme():
    with pytest.raises(ValueError, match="theme"):
        peak_count_rgba([[1]], [[1]], theme="blue")
