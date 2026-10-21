import h5py
import numpy as np

from roc_mcs.roi.extraction import extract_from_roi, roi_pixel_mask
from roc_mcs.roi.h5_scan_repository import H5ScanRepository
from roc_mcs.roi.roi_types import ROI
from roc_mcs.roi.spatial_pooling import PoolSpec


def _make_scan(path, axis=0):
    th = np.array([-2., -1., 0., 1., 2.])
    yy, xx = np.mgrid[:12, :14]
    tyx = np.stack([t + 10 * yy + xx for t in range(len(th))]).astype(float)
    frame = tyx if axis == 0 else np.moveaxis(tyx, 0, -1)
    with h5py.File(path, "w") as f:
        measurement = f.create_group("entry1/measurement")
        measurement.create_dataset("Frame", data=frame)
        measurement.create_dataset("th", data=th)
    return frame, th


def test_source_roi_is_stable_across_raw_and_pooled_views(tmp_path):
    _, th = _make_scan(tmp_path / "scan.h5")
    repo = H5ScanRepository(tmp_path / "scan.h5")
    roi = ROI(cx=5., cy=5., width=6., height=6., type="rect")
    original_roi = roi.to_dict().copy()

    t_raw, curve_raw, n_raw = repo.curve_from_roi("entry1", roi, "mean")
    rv_raw = repo.volume_from_roi("entry1", roi)
    assert curve_raw.shape == (len(th),) and n_raw > 0
    assert rv_raw.coord_system == "xy"

    repo.set_pool(PoolSpec(3, 2))
    view = repo.load_scan("entry1")
    x, y = repo.centers_xy("entry1")
    assert view.shape_yx == (5, 6)
    np.testing.assert_array_equal(x, [1., 3., 5., 7., 9., 11.])
    np.testing.assert_array_equal(y, [1., 3., 5., 7., 9.])

    t_pool, curve_pool, n_pool = repo.curve_from_roi("entry1", roi, "mean")
    rv_pool = repo.volume_from_roi("entry1", roi)
    assert np.array_equal(t_raw, t_pool)
    assert curve_pool.shape == (len(th),) and n_pool > 0
    assert rv_pool.coord_system == "xy"
    assert rv_pool.volume.shape == (len(rv_pool.x), len(rv_pool.y), len(th))
    assert set(rv_pool.x).issubset(set(x)) and set(rv_pool.y).issubset(set(y))
    assert roi.to_dict() == original_roi

    sx, sy = repo.view_indices_to_source_xy("entry1", [0, 2], [1, 3])
    np.testing.assert_array_equal(sx, [1., 5.])
    np.testing.assert_array_equal(sy, [3., 7.])


def test_rotated_roi_volume_uses_current_view_spacing(tmp_path):
    _make_scan(tmp_path / "scan.h5")
    repo = H5ScanRepository(tmp_path / "scan.h5", pool=PoolSpec(3, 2))
    roi = ROI(cx=5., cy=5., width=7., height=5., angle=30., type="rect")
    rv = repo.volume_from_roi("entry1", roi)
    assert rv.coord_system == "uv"
    assert rv.volume.shape == (len(rv.x), len(rv.y), len(rv.theta))
    assert len(rv.x) <= int(np.ceil(7 / 2)) + 1
    assert np.isfinite(rv.volume).any()


def test_roi_mask_uses_detector_coordinates_of_view_centers():
    roi = ROI(cx=4., cy=4., width=2., height=2., type="rect")
    centers = np.array([1., 3., 5., 7.])
    mask = roi_pixel_mask(roi, 4, 4, x_coords=centers, y_coords=centers)
    assert mask.shape == (4, 4) and mask.sum() == 4
    assert mask[1, 1] and mask[1, 2] and mask[2, 1] and mask[2, 2]


def test_curve_extraction_respects_current_view_valid_mask():
    th = np.array([0., 1., 2.])
    raw = np.empty((3, 4, 4), dtype=float)
    raw[0], raw[1], raw[2] = 1., 10., 100.
    roi = ROI(cx=1.5, cy=1.5, width=3., height=3., type="rect")
    valid = np.ones((4, 4, 3), dtype=bool)
    valid[1, 1, 1] = False
    _, curve, count = extract_from_roi(raw, th, 0, roi, 4, 4, "mean", x_coords=np.arange(4), y_coords=np.arange(4), valid_mask=valid)
    assert count == 16
    np.testing.assert_allclose(curve, [1., 10., 100.])
    valid[:, :, 1] = False
    _, curve, _ = extract_from_roi(raw, th, 0, roi, 4, 4, "sum", x_coords=np.arange(4), y_coords=np.arange(4), valid_mask=valid)
    assert np.isnan(curve[1])


def test_point_and_line_roi_select_current_view_samples(tmp_path):
    _make_scan(tmp_path / "scan.h5")
    repo = H5ScanRepository(tmp_path / "scan.h5", pool=PoolSpec(3, 2))
    _, point_curve, point_count = repo.curve_from_roi("entry1", ROI(cx=4, cy=4, width=0, height=0, type="point"), "mean")
    _, line_curve, line_count = repo.curve_from_roi("entry1", ROI(cx=5, cy=5, width=8, height=0, angle=0, type="line"), "mean")
    assert point_count == 1 and point_curve.ndim == 1
    assert line_count > 1 and line_curve.ndim == 1


def test_native_axis_2_is_supported(tmp_path):
    _, th = _make_scan(tmp_path / "scan.h5", axis=2)
    repo = H5ScanRepository(tmp_path / "scan.h5", pool=PoolSpec(3, 2))
    roi = ROI(cx=5, cy=5, width=6, height=6, type="rect")
    _, curve, count = repo.curve_from_roi("entry1", roi, "mean")
    volume = repo.volume_from_roi("entry1", roi)
    assert curve.shape == th.shape and count > 0
    assert volume.volume.shape == (len(volume.x), len(volume.y), len(th))


def test_legacy_extract_call_remains_compatible():
    th = np.array([0., 1.])
    raw = np.arange(2 * 4 * 5.).reshape(2, 4, 5)
    roi = ROI(cx=2., cy=1., width=2., height=2., type="rect")
    _, curves, count = extract_from_roi(raw, th, 0, roi, 5, 4, None)
    assert curves.ndim == 2 and curves.shape == (count, 2) and count > 0
