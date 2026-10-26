from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "src" / "roc_mcs" / "gui" / "live_scans"
CAL = ROOT / "src" / "roc_mcs" / "gui" / "calibration"

class TestSingleCalibrationSource(unittest.TestCase):
    def test_dashboard_imports_shared_dialog(self):
        source = (LIVE / "roc_live_dashboard_v4.py").read_text(encoding="utf-8")
        self.assertIn("from .calibration_dialog import CalibrationDialog", source)
        self.assertNotIn(".calibration_ui", source)

    def test_dialog_wraps_canonical_widget(self):
        source = (LIVE / "calibration_dialog.py").read_text(encoding="utf-8")
        self.assertIn("from roc_mcs.gui.calibration.calibration_widget import CalibrationWidget", source)
        self.assertNotIn("core_calibration", source)
        self.assertNotIn("from .calibration_widget import", source)

    def test_canonical_widget_exposes_integration_api(self):
        source = (CAL / "calibration_widget.py").read_text(encoding="utf-8")
        self.assertIn("calibration_result_ready = pyqtSignal(dict)", source)
        self.assertIn("def set_h5_path(self, path)", source)
        self.assertIn("self._calibration_context.update(getattr(self, \"_extra_calibration_context\", {}))", source)

    def test_duplicate_calibration_package_not_needed_by_live_scans(self):
        for path in LIVE.glob("*.py"):
            self.assertNotIn("calibration_ui", path.read_text(encoding="utf-8"), path.name)

if __name__ == "__main__":
    unittest.main()
