from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from roc_mcs.gui.app_settings import (
    load_detector_bad_pixels,
    normalize_detector_bad_pixels,
    save_detector_bad_pixels,
)
from roc_mcs.gui.roi_explorer.vtk_backend.h5_volume_source import H5VolumeLoader
from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (
    VTKVolumeScene,
    build_projection_grid,
)
from roc_mcs.roi.roi_types import ROI


class TestDetectorPixelSettings(unittest.TestCase):
    def test_normalization_accepts_editor_text_and_deduplicates(self):
        self.assertEqual(
            normalize_detector_bad_pixels("241,26; 10,5; 241,26"),
            [{"x": 241, "y": 26}, {"x": 10, "y": 5}],
        )

    def test_normalization_rejects_fractional_or_negative_indices(self):
        for value in ("1.2,3", "-1,3", "1,foo"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_detector_bad_pixels(value)

    def test_settings_round_trip_preserves_global_pixel_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            settings_path = Path(temp) / "settings.json"
            with patch.dict(os.environ, {"ROC_MCS_SETTINGS": str(settings_path)}):
                save_detector_bad_pixels([(241, 26), (4, 9)])
                self.assertEqual(load_detector_bad_pixels(), [(241, 26), (4, 9)])
                saved = json.loads(settings_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["detector_bad_pixels"], [
                    {"x": 241, "y": 26}, {"x": 4, "y": 9}
                ])


class TestBadPixelAndThetaPipeline(unittest.TestCase):
    def _make_h5(self, path: Path):
        # Native layout is theta,y,x. The one deliberately hot pixel is at
        # source-grid x=4,y=3; all other voxels have intensity 2.
        data = np.full((5, 8, 10), 2.0, dtype=np.float32)
        data[:, 3, 4] = 50000.0
        with h5py.File(path, "w") as h5:
            for scan_key in ("entry001", "entry002"):
                measurement = h5.create_group(f"{scan_key}/measurement")
                measurement.create_dataset("Frame", data=data)
                measurement.create_dataset("th", data=np.asarray([0., 2., 4., 6., 8.]))
        return data

    def test_bad_pixel_mask_is_applied_before_full_projection_and_theta_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            h5_path = root / "test.h5"
            source = self._make_h5(h5_path)
            settings_path = root / "settings.json"
            settings_path.write_text(
                json.dumps({"detector_bad_pixels": [{"x": 4, "y": 3}]}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ROC_MCS_SETTINGS": str(settings_path)}):
                loader = H5VolumeLoader(h5_path)
                self.assertEqual(loader.theta_bounds("entry001"), (0.0, 8.0))
                payload = loader.load("entry001", mode="full", theta_range=(2.0, 6.0))

            self.assertEqual(payload.shape, (10, 8, 3))
            self.assertEqual(payload.theta.tolist(), [2.0, 4.0, 6.0])
            self.assertEqual(payload.masked_detector_pixels, ((4, 3),))
            self.assertTrue(np.isnan(payload.volume[4, 3, :]).all())
            self.assertTrue(np.isnan(payload.projection[4, 3]))
            self.assertTrue(np.all(payload.volume[3, 3, :] == 2.0))
            self.assertTrue(np.all(payload.projection[3, 3] == 6.0))
            self.assertEqual(int(np.count_nonzero(np.isfinite(payload.volume))), 10 * 8 * 3 - 3)
            second_scan = loader.load("entry002", mode="full")
            self.assertEqual(second_scan.masked_detector_pixels, ((4, 3),))
            self.assertTrue(np.isnan(second_scan.volume[4, 3, :]).all())
            with h5py.File(h5_path, "r") as h5:
                self.assertTrue(np.all(h5["entry001/measurement/Frame"][:, 3, 4] == 50000.0))
            self.assertEqual(source.shape, (5, 8, 10))  # source/HDF5 data is untouched

    def test_bad_pixel_mask_is_used_by_axis_aligned_roi_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            h5_path = root / "test.h5"
            self._make_h5(h5_path)
            settings_path = root / "settings.json"
            settings_path.write_text(
                json.dumps({"detector_bad_pixels": [{"x": 4, "y": 3}]}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ROC_MCS_SETTINGS": str(settings_path)}):
                loader = H5VolumeLoader(h5_path)
                roi = ROI(cx=4.5, cy=3.5, width=6.0, height=6.0, angle=0.0, type="rect")
                payload = loader.load("entry001", mode="roi", roi=roi)

            # ROI coordinates retain detector-grid position; a masked source
            # pixel remains invalid rather than contaminating interpolation.
            xi = int(np.flatnonzero(payload.x == 4.0)[0])
            yi = int(np.flatnonzero(payload.y == 3.0)[0])
            self.assertTrue(np.isnan(payload.volume[xi, yi, :]).all())
            self.assertTrue(np.isnan(payload.projection[xi, yi]))
            self.assertEqual(payload.masked_detector_pixels, ((4, 3),))

    def test_rotated_roi_interpolation_never_reintroduces_masked_source_pixel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            h5_path = root / "rotated.h5"
            self._make_h5(h5_path)
            settings_path = root / "settings.json"
            settings_path.write_text(
                json.dumps({"detector_bad_pixels": [{"x": 4, "y": 3}]}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ROC_MCS_SETTINGS": str(settings_path)}):
                loader = H5VolumeLoader(h5_path)
                roi = ROI(cx=4.0, cy=3.0, width=5.0, height=5.0, angle=31.0, type="rect")
                payload = loader.load("entry001", mode="roi", roi=roi)
            finite = payload.volume[np.isfinite(payload.volume)]
            self.assertGreater(finite.size, 0)
            self.assertLessEqual(float(np.max(finite)), 2.0 + 1e-6)
            self.assertEqual(payload.coord_system, "uv")
            self.assertEqual(payload.masked_detector_pixels, ((4, 3),))

    def test_default_bad_pixel_outside_a_smaller_detector_is_ignored_safely(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            h5_path = root / "small.h5"
            self._make_h5(h5_path)
            settings_path = root / "settings.json"
            settings_path.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"ROC_MCS_SETTINGS": str(settings_path)}):
                payload = H5VolumeLoader(h5_path).load("entry001", mode="full")
            self.assertEqual(payload.masked_detector_pixels, ())
            self.assertTrue(np.isfinite(payload.volume).all())

    def test_projection_display_range_is_robust_to_single_extreme_value(self):
        projection = np.ones((100, 100), dtype=np.float64)
        projection[24, 71] = 86184.0
        payload = SimpleNamespace(
            projection=projection,
            x=np.arange(100, dtype=float),
            y=np.arange(100, dtype=float),
            theta=np.arange(5, dtype=float),
        )
        _grid, info = build_projection_grid(payload)
        self.assertEqual(info["scalar_max"], 86184.0)
        self.assertEqual(info["display_scalar_min"], 0.0)
        self.assertEqual(info["display_scalar_max"], 1.0)

    def test_displayed_theta_spacing_tracks_visual_scale_only(self):
        payload = SimpleNamespace(
            x=np.arange(5, dtype=float),
            y=np.arange(5, dtype=float),
            theta=np.arange(0.0, 10.0, 2.0),
        )
        self.assertEqual(VTKVolumeScene._estimate_min_spacing(payload, 1.0), 1.0)
        self.assertAlmostEqual(VTKVolumeScene._estimate_min_spacing(payload, 0.25), 0.5)
        self.assertAlmostEqual(VTKVolumeScene._estimate_min_spacing(payload, 3.0), 1.0)


class TestRoiGeometryInitializationRegression(unittest.TestCase):
    def test_metadata_seed_marks_roi_initialized_and_fallback_matches_35_percent(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src/roc_mcs/gui/roi_explorer/vtk_backend/desktop_viewer.py"
        ).read_text(encoding="utf-8")
        seed = source.split("def _seed_roi_from_h5_metadata", 1)[1].split("def _current_roi", 1)[0]
        self.assertIn("self._roi_geometry_initialized = True", seed)
        fallback = source.split("def _seed_roi_from_payload", 1)[1].split("def _seed_theta_controls", 1)[0]
        self.assertIn("* 0.35", fallback)
        self.assertNotIn("* 0.7", fallback)
        self.assertIn("not self._roi_geometry_initialized and payload.view_mode == \"full\"", source)
        self.assertIn("self._roi_by_scan[previous_scan] = self._current_roi_geometry()", source)
        self.assertIn("self._set_roi_geometry(saved_roi)", source)
        self.assertIn("self._roi_by_scan.clear()", source)
        self.assertIn("Масштаб θ (только отображение)", source)
        self.assertIn("Ограничить диапазон θ при загрузке", source)


if __name__ == "__main__":
    unittest.main()
