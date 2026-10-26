"""Окно калибровки угловой оси, встроенное в live-дашборд."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import math
from PyQt6 import QtCore, QtWidgets
from .calibration_widget import CalibrationWidget


class DashboardCalibrationWidget(CalibrationWidget):
    calibration_ready = QtCore.pyqtSignal(dict)

    def start_calibration(self):
        dialog = self.parentWidget()
        u_cal = float(dialog.spin_reference_amplitude.value())
        if not math.isfinite(u_cal) or u_cal <= 0:
            QtWidgets.QMessageBox.warning(dialog, "Не задана амплитуда калибровки",
                "Введите фактическую амплитуду генератора адаптивного элемента Uк (mVpp), при которой снята калибровочная пара.")
            return
        # Базовый виджет объединяет этот контекст с путями, Entry, ветвью и автоматически найденной φк.
        self._extra_calibration_context = {"reference_amplitude_mVpp": u_cal}
        super().start_calibration()

    def on_calibration_finished(self, result, fig):
        super().on_calibration_finished(result, fig)
        try:
            a, b = float(result.A_best), float(result.B_best)
            if not all(math.isfinite(x) for x in (a, b)) or abs(a) < 1e-15:
                raise ValueError("Расчёт вернул недопустимые коэффициенты A/B.")
            ctx = getattr(self, "_calibration_context", {})
            u_cal = float(ctx["reference_amplitude_mVpp"])
            if not math.isfinite(u_cal) or u_cal <= 0: raise ValueError("Амплитуда Uк должна быть положительной.")
            payload = {"schema_version": 3, "method": "scale_by_drive_amplitude_theta_A_s", "A": a, "B": b,
                "unit": "угл. с", "source_mcs": ctx["source_mcs"], "reference_hdf5": ctx["reference_hdf5"],
                "entry_id": str(ctx["entry_id"]), "branch": ctx["branch"],
                "reference_phi": float(ctx["reference_phi"]), "reference_amplitude_mVpp": u_cal,
                "calibrated_at": datetime.now(timezone.utc).isoformat()}
            score = float(result.best_row.get("score_best", float("nan")))
            if math.isfinite(score): payload["score_best"] = score
            self.log(f"[КАЛИБРОВКА] Aк={a:.8g}; Bк={b:.8g} угл. с (только для совмещения); Uк={u_cal:.4g} mVpp. φк найдена автоматически.")
            self.calibration_ready.emit(payload)
        except Exception as exc:
            self.log(f"[ОШИБКА] Не удалось передать коэффициенты дашборду: {exc}")


class CalibrationDialog(QtWidgets.QDialog):
    calibration_saved = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None, initial_calibration=None):
        super().__init__(parent)
        self.setWindowTitle("Калибровка угловой оси MCS"); self.resize(1120, 760); self.setMinimumSize(900, 640)
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(5)
        note = QtWidgets.QLabel("Сопоставьте моторный скан HDF5 и MCS-скан того же объекта. Внутри калибровки подбираются Aк и Bк для совмещения; live-ось использует только масштаб. Фаза φк находится автоматически по калибровочному MCS. Для серии фаза будет найдена заново по её первому MCS. Частота вручную не вводится.")
        note.setWordWrap(True); note.setObjectName("calibrationNote"); layout.addWidget(note)
        row = QtWidgets.QHBoxLayout(); row.setContentsMargins(2, 0, 2, 2); row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel("Амплитуда генератора Uк, mVpp"))
        self.spin_reference_amplitude = QtWidgets.QDoubleSpinBox(); self.spin_reference_amplitude.setRange(0, 10000); self.spin_reference_amplitude.setDecimals(2); self.spin_reference_amplitude.setSingleStep(50); self.spin_reference_amplitude.setKeyboardTracking(False); self.spin_reference_amplitude.setToolTip("Фактическая амплитуда сигнала генератора адаптивного элемента peak-to-peak. Не напряжение трубки и не напряжение Korad/пресса.")
        row.addWidget(self.spin_reference_amplitude); row.addStretch(1); layout.addLayout(row)
        self.widget = DashboardCalibrationWidget(parent=self); layout.addWidget(self.widget, 1)
        self.widget.calibration_ready.connect(self._publish); self._last_payload = None
        if isinstance(initial_calibration, dict):
            self.spin_reference_amplitude.setValue(float(initial_calibration.get("reference_amplitude_mVpp") or 0))
            mcs_path = initial_calibration.get("source_mcs"); h5_path = initial_calibration.get("reference_hdf5")
            if mcs_path and Path(mcs_path).is_file(): self.widget.input_mcs_path.setText(str(mcs_path))
            if h5_path and Path(h5_path).is_file():
                try:
                    self.widget.set_h5_path(h5_path)
                    entry_text = str(initial_calibration.get("entry_id", "")); entry_index = self.widget.combo_entry_id.findText(entry_text)
                    if entry_index >= 0: self.widget.combo_entry_id.setCurrentIndex(entry_index)
                except Exception as exc: self.widget.log(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось восстановить HDF5/Entry ID: {exc}")
            idx = self.widget.combo_branch.findData(str(initial_calibration.get("branch", "up")))
            if idx >= 0: self.widget.combo_branch.setCurrentIndex(idx)
        self.setStyleSheet("QDialog { background:#10141c; color:#e2e8f0; } QLabel#calibrationNote { color:#aab8ca; padding:4px 6px; border-bottom:1px solid #303b4a; } QWidget { color:#e2e8f0; } QLineEdit, QComboBox, QTextEdit, QDoubleSpinBox { background:#151c27; color:#e2e8f0; border:1px solid #384454; border-radius:4px; padding:3px; } QPushButton, QToolButton { background:#202a38; color:#e2e8f0; border:1px solid #3a485b; border-radius:4px; padding:4px 7px; } QPushButton:hover, QToolButton:hover { background:#2a394c; } QGroupBox { border:1px solid #384454; margin-top:6px; padding-top:5px; } QGroupBox::title { subcontrol-origin:margin; left:8px; color:#8fc5ff; } QProgressBar { background:#202a38; border:1px solid #384454; } QProgressBar::chunk { background:#448bd4; }")

    def _publish(self, payload: dict):
        self._last_payload = payload
        self.calibration_saved.emit(payload)
