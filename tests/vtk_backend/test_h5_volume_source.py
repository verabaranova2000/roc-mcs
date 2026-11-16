from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from roc_mcs.gui.roi_explorer.vtk_backend.h5_volume_source import (
    H5VolumeLoader, SpikeReport, _display_dtype, _nan_aware_smooth_preserving_invalid,
    _sum_projection_chunked, orient_to_ascending,
)


def _write_scan(path: Path, raw: np.ndarray, theta: np.ndarray) -> Path:
    with h5py.File(path, "w") as h5:
        measurement = h5.create_group("entry001/measurement")
        measurement.create_dataset("Frame", data=raw)
        measurement.create_dataset("th", data=theta)
    return path


def _loader(path: Path, monkeypatch) -> H5VolumeLoader:
    settings = path.with_name("settings.json")
    settings.write_text(json.dumps({"detector_bad_pixels": []}), encoding="utf-8")
    monkeypatch.setenv("ROC_MCS_SETTINGS", str(settings))
    return H5VolumeLoader(path)


class TestH5VolumeLoader:
    def test_theta_y_x_layout_maps_to_x_y_theta_without_permutation(self, tmp_path, monkeypatch):
        raw = np.arange(4 * 3 * 5, dtype=np.int16).reshape(4, 3, 5)
        path = _write_scan(tmp_path / "scan.h5", raw, np.array([0., 1., 2., 3.]))
        payload = _loader(path, monkeypatch).load("entry001")
        expected = raw.transpose(2, 1, 0).astype(np.float32)
        np.testing.assert_array_equal(payload.volume, expected)
        np.testing.assert_array_equal(payload.projection, expected.sum(axis=2))
        np.testing.assert_array_equal(payload.x, np.arange(5))
        np.testing.assert_array_equal(payload.y, np.arange(3))
        assert payload.shape == (5, 3, 4)
        assert payload.coord_system == "xy"

    def test_y_x_theta_layout_maps_to_x_y_theta(self, tmp_path, monkeypatch):
        raw = np.arange(3 * 5 * 4, dtype=np.float64).reshape(3, 5, 4)
        path = _write_scan(tmp_path / "scan.h5", raw, np.arange(4, dtype=float))
        payload = _loader(path, monkeypatch).load("entry001")
        np.testing.assert_array_equal(payload.volume, raw.transpose(1, 0, 2))

    def test_axis_aligned_roi_extracts_exact_detector_samples_in_xy_theta_order(self, tmp_path, monkeypatch):
        from roc_mcs.roi.roi_types import ROI
        raw = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        loader = _loader(_write_scan(tmp_path / "scan.h5", raw, np.arange(3.)), monkeypatch)
        roi = ROI(cx=2., cy=1., width=2., height=2., angle=0.)
        payload = loader.load("entry001", mode="roi", roi=roi)
        expected = raw[:, 0:3, 1:4].transpose(2, 1, 0)
        np.testing.assert_array_equal(payload.volume, expected)
        np.testing.assert_array_equal(payload.x, [1., 2., 3.])
        np.testing.assert_array_equal(payload.y, [0., 1., 2.])
        np.testing.assert_array_equal(payload.projection, expected.sum(axis=2))
        assert payload.coord_system == "xy"

    def test_descending_theta_reverses_samples_and_axis_together(self, tmp_path, monkeypatch):
        raw = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
        theta = np.array([3., 2., 1., 0.])
        payload = _loader(_write_scan(tmp_path / "scan.h5", raw, theta), monkeypatch).load("entry001")
        np.testing.assert_array_equal(payload.theta, [0., 1., 2., 3.])
        np.testing.assert_array_equal(payload.volume, raw[::-1].transpose(2, 1, 0))

    def test_theta_range_is_inclusive_and_projection_uses_only_selected_frames(self, tmp_path, monkeypatch):
        raw = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
        payload = _loader(_write_scan(tmp_path / "scan.h5", raw, np.arange(4.)), monkeypatch).load(
            "entry001", theta_range=(1., 2.)
        )
        expected = raw[1:3].transpose(2, 1, 0)
        np.testing.assert_array_equal(payload.theta, [1., 2.])
        np.testing.assert_array_equal(payload.volume, expected)
        np.testing.assert_array_equal(payload.projection, expected.sum(axis=2))

    def test_theta_range_with_fewer_than_two_samples_is_rejected(self, tmp_path, monkeypatch):
        raw = np.ones((4, 3, 5), dtype=np.float32)
        loader = _loader(_write_scan(tmp_path / "scan.h5", raw, np.arange(4.)), monkeypatch)
        with pytest.raises(ValueError, match="минимум два"):
            loader.load("entry001", theta_range=(1.5, 2.5))
        with pytest.raises(ValueError, match="min < max"):
            loader.load("entry001", theta_range=(2., 1.))

    def test_nonfinite_samples_are_excluded_from_sum_and_all_invalid_pixels_stay_nan(self, tmp_path, monkeypatch):
        raw = np.ones((3, 2, 4), dtype=np.float32)
        raw[:, 0, 0] = [1., np.nan, 3.]
        raw[:, 0, 1] = [np.nan, np.inf, -np.inf]
        payload = _loader(_write_scan(tmp_path / "scan.h5", raw, np.arange(3.)), monkeypatch).load("entry001")
        assert payload.projection[0, 0] == 4.
        assert np.isnan(payload.projection[1, 0])
        assert not np.isfinite(payload.volume[1, 0, :]).any()
        assert payload.finite_count == 3 * 2 * 4 - 4
        assert payload.invalid_count == 4

    def test_output_arrays_are_read_only_and_nbytes_is_consistent(self, tmp_path, monkeypatch):
        raw = np.ones((3, 2, 4), dtype=np.int16)
        payload = _loader(_write_scan(tmp_path / "scan.h5", raw, np.arange(3.)), monkeypatch).load("entry001")
        for array in (payload.volume, payload.projection, payload.x, payload.y, payload.theta):
            assert not array.flags.writeable
        assert payload.nbytes == sum(a.nbytes for a in (payload.volume, payload.projection, payload.x, payload.y, payload.theta))
        with pytest.raises(ValueError):
            payload.volume[0, 0, 0] = 9

    def test_empty_scan_and_unknown_key_fail_closed(self, tmp_path, monkeypatch):
        raw = np.full((3, 2, 4), np.nan, dtype=np.float32)
        loader = _loader(_write_scan(tmp_path / "scan.h5", raw, np.arange(3.)), monkeypatch)
        with pytest.raises(ValueError, match="не содержит конечных"):
            loader.load("entry001")
        with pytest.raises(KeyError):
            loader.load("entry999")

    def test_spike_validity_mask_is_applied_before_theta_crop_and_projection(self, tmp_path, monkeypatch):
        raw = np.full((4, 2, 5), 3., dtype=np.float32)
        loader = _loader(_write_scan(tmp_path / "scan.h5", raw, np.arange(4.)), monkeypatch)
        valid_yxt = np.ones((2, 5, 4), dtype=bool)
        valid_yxt[1, 2, 1] = False
        report = SpikeReport(True, kappa=3., sigma=1., invalid_count=1, invalid_fraction=1 / raw.size)
        monkeypatch.setattr(loader, "_make_spike_valid_mask", lambda _repo, _key: (valid_yxt, report))
        payload = loader.load("entry001", remove_spikes=True, theta_range=(0., 2.))
        assert payload.spike_report is report
        assert np.isnan(payload.volume[2, 1, 1])
        np.testing.assert_array_equal(payload.volume[2, 1, [0, 2]], [3., 3.])
        assert payload.projection[2, 1] == 6.
        assert payload.invalid_count == 1

    def test_malformed_hdf5_theta_layout_is_rejected_as_ambiguous(self, tmp_path, monkeypatch):
        raw = np.ones((3, 2, 3), dtype=np.float32)
        loader = _loader(_write_scan(tmp_path / "ambiguous.h5", raw, np.arange(3.)), monkeypatch)
        with pytest.raises(ValueError, match="неоднозначна"):
            loader.load("entry001")

    def test_missing_hdf5_path_is_rejected_before_repository_creation(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            H5VolumeLoader(tmp_path / "missing.h5")

    def test_bad_inputs_fail_closed_and_source_file_is_not_modified(self, tmp_path, monkeypatch):
        raw = np.arange(3 * 2 * 4, dtype=np.int16).reshape(3, 2, 4)
        path = _write_scan(tmp_path / "scan.h5", raw, np.arange(3.))
        before = path.read_bytes()
        loader = _loader(path, monkeypatch)
        with pytest.raises(ValueError, match="mode"):
            loader.load("entry001", mode="other")
        with pytest.raises(ValueError, match="sigma"):
            loader.load("entry001", smoothing_sigma=-1.)
        with pytest.raises(ValueError, match="ROI"):
            loader.load("entry001", mode="roi")
        assert path.read_bytes() == before


class TestArrayNumerics:
    def test_orient_to_ascending_reverses_every_descending_dimension(self):
        a = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        result, x, y, theta = orient_to_ascending(a, [2, 1, 0], [0, 1, 2, 3], [4, 3, 2, 1, 0])
        np.testing.assert_array_equal(x, [0, 1, 2])
        np.testing.assert_array_equal(y, [0, 1, 2, 3])
        np.testing.assert_array_equal(theta, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result, a[::-1, :, ::-1])

    @pytest.mark.parametrize("axis", [[0, 0, 1], [0, 2, 1], [0, np.nan, 2], [0, np.inf, 2]])
    def test_bad_axes_are_rejected(self, axis):
        with pytest.raises(ValueError):
            orient_to_ascending(np.zeros((3, 2, 4)), axis, [0, 1], [0, 1, 2, 3])

    def test_axis_length_mismatch_and_non_3d_volume_are_rejected(self):
        with pytest.raises(ValueError):
            orient_to_ascending(np.zeros((3, 2, 4)), [0, 1], [0, 1], [0, 1, 2, 3])
        with pytest.raises(ValueError):
            orient_to_ascending(np.zeros((3, 2)), [0, 1, 2], [0, 1], [0, 1])

    def test_dtype_policy_preserves_float32_and_exactly_represents_small_integers(self):
        assert _display_dtype(np.array([1], dtype=np.float32)) == np.dtype(np.float32)
        assert _display_dtype(np.array([1], dtype=np.int16)) == np.dtype(np.float32)
        assert _display_dtype(np.array([1], dtype=np.int32)) == np.dtype(np.float64)
        assert _display_dtype(np.array([1], dtype=np.float64)) == np.dtype(np.float64)

    def test_chunked_sum_matches_finite_reference_and_preserves_all_invalid_nan(self):
        a = np.arange(9 * 4 * 5, dtype=np.float32).reshape(9, 4, 5)
        a[0, 0, :] = np.nan
        a[1, 1, 0] = np.inf
        expected = np.sum(np.where(np.isfinite(a), a, 0.), axis=2, dtype=np.float64)
        expected[~np.isfinite(a).any(axis=2)] = np.nan
        for chunk in (1, 3, 32, 100):
            np.testing.assert_allclose(_sum_projection_chunked(a, chunk), expected, equal_nan=True)

    def test_masked_smoothing_preserves_invalid_sites_and_constant_field(self):
        volume = np.full((9, 7, 5), 4., dtype=np.float32)
        volume[4, 3, 2] = np.nan
        volume[0, 0, 0] = np.inf
        _nan_aware_smooth_preserving_invalid(volume, 1.2)
        assert np.isnan(volume[4, 3, 2]) and not np.isfinite(volume[0, 0, 0])
        np.testing.assert_allclose(volume[np.isfinite(volume)], 4., atol=1e-6)
