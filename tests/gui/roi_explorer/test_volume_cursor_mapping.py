"""Регрессионные проверки перевода 2D detector cursor в координаты 3D ROI."""
import math
import unittest

from roc_mcs.roi.roi_types import ROI
from roc_mcs.gui.roi_explorer.volume_rendering_runtime import map_detector_cursor_to_volume_xy


class VolumeCursorMappingTests(unittest.TestCase):
    def test_xy_volume_keeps_detector_coordinates(self):
        self.assertEqual(
            map_detector_cursor_to_volume_xy(
                12.5, 25.5, coord_system="xy", overlay_full_detector=False
            ),
            (12.5, 25.5),
        )

    def test_full_detector_overlay_keeps_detector_coordinates(self):
        roi = ROI(cx=10, cy=20, width=10, height=8, angle=90, type="rect")
        self.assertEqual(
            map_detector_cursor_to_volume_xy(
                12, 20, coord_system="uv", overlay_full_detector=True, roi=roi
            ),
            (12.0, 20.0),
        )

    def test_rotated_roi_is_mapped_to_local_uv(self):
        roi = ROI(cx=10, cy=20, width=10, height=8, angle=90, type="rect")
        mapped = map_detector_cursor_to_volume_xy(
            12, 20, coord_system="uv", overlay_full_detector=False, roi=roi
        )
        self.assertIsNotNone(mapped)
        self.assertTrue(math.isclose(mapped[0], 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(mapped[1], -2.0, abs_tol=1e-12))

    def test_cursor_outside_selected_rotated_roi_is_hidden(self):
        roi = ROI(cx=10, cy=20, width=10, height=8, angle=90, type="rect")
        self.assertIsNone(
            map_detector_cursor_to_volume_xy(
                30, 20, coord_system="uv", overlay_full_detector=False, roi=roi
            )
        )

    def test_non_finite_cursor_is_rejected(self):
        self.assertIsNone(
            map_detector_cursor_to_volume_xy(
                float("nan"), 2, coord_system="xy", overlay_full_detector=False
            )
        )


if __name__ == "__main__":
    unittest.main()
