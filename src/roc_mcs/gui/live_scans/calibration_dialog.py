"""Диалог калибровки live-оси на основе единственного CalibrationWidget из gui.calibration."""
from __future__ import annotations
from datetime import datetime, timezone
import math
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from roc_mcs.gui.calibration.calibration_widget import CalibrationWidget


class _EmbeddedCalibrationWidget(CalibrationWidget):
    """Тонкий адаптер контекста; интерфейс и расчёт унаследованы из общей утилиты."""
    def __init__(self, dialog, parent=None):
        self._dashboard_dialog = dialog
        super().__init__(parent=parent, theme="dark")

    def start_calibration(self):
        u_cal = float(self._dashboard_dialog.spin_reference_amplitude.value())
        if not math.isfinite(u_cal) or u_cal <= 0:
            QtWidgets.QMessageBox.warning(self._dashboard_dialog, "Не задана амплитуда калибровки", "Введите фактическую амплитуду генератора адаптивного элемента Uк (mVpp), при которой снята калибровочная пара.")
            return
        self._extra_calibration_context = {"reference_amplitude_mVpp": u_cal}
        super().start_calibration()


class CalibrationDialog(QtWidgets.QDialog):
    """Обёртка для встраивания общей утилиты калибровки в live-дашборд."""
    calibration_saved = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None, initial_calibration=None):
        super().__init__(parent)
        self.setWindowTitle("Калибровка угловой оси MCS")
        self.resize(1120, 760)
        self.setMinimumSize(900, 640)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        note = QtWidgets.QLabel("Сопоставьте моторный скан HDF5 и MCS-скан того же объекта. Внутри калибровки подбираются Aк и Bк для совмещения; live-ось использует только масштаб. Фаза φк находится автоматически по калибровочному MCS. Для каждой live-серии фаза определяется заново по её первому MCS. Частота вручную не вводится.")
        note.setWordWrap(True)
        note.setObjectName("calibrationNote")
        layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 2)
        row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel("Амплитуда генератора Uк, mVpp"))
        self.spin_reference_amplitude = QtWidgets.QDoubleSpinBox()
        self.spin_reference_amplitude.setRange(0, 10000)
        self.spin_reference_amplitude.setDecimals(2)
        self.spin_reference_amplitude.setSingleStep(50)
        self.spin_reference_amplitude.setKeyboardTracking(False)
        self.spin_reference_amplitude.setToolTip("Фактическая амплитуда генератора адаптивного элемента peak-to-peak. Не напряжение трубки и не напряжение Korad/пресса.")
        row.addWidget(self.spin_reference_amplitude)
        row.addStretch(1)
        layout.addLayout(row)
        self.widget = _EmbeddedCalibrationWidget(self, parent=self)
        self.widget.calibration_result_ready.connect(self._publish)
        layout.addWidget(self.widget, 1)
        self._last_payload = None
        self.setStyleSheet("QDialog{background:#10141c;color:#e2e8f0;} QLabel#calibrationNote{color:#aab8ca;padding:4px 6px;border-bottom:1px solid #303b4a;} QWidget{color:#e2e8f0;} QLineEdit,QComboBox,QTextEdit,QDoubleSpinBox{background:#151c27;color:#e2e8f0;border:1px solid #384454;border-radius:4px;padding:3px;} QPushButton,QToolButton{background:#202a38;color:#e2e8f0;border:1px solid #3a485b;border-radius:4px;padding:4px 7px;} QPushButton:hover,QToolButton:hover{background:#2a394c;} QGroupBox{border:1px solid #384454;margin-top:6px;padding-top:5px;} QGroupBox::title{subcontrol-origin:margin;left:8px;color:#8fc5ff;} QProgressBar{background:#202a38;border:1px solid #384454;} QProgressBar::chunk{background:#448bd4;}")
        self._restore_initial_calibration(initial_calibration)

    def _restore_initial_calibration(self, calibration):
        if not isinstance(calibration, dict):
            return
        self.spin_reference_amplitude.setValue(float(calibration.get("reference_amplitude_mVpp") or 0))
        mcs_path = calibration.get("source_mcs")
        h5_path = calibration.get("reference_hdf5")
        if mcs_path and Path(mcs_path).is_file():
            self.widget.input_mcs_path.setText(str(mcs_path))
        if h5_path and Path(h5_path).is_file():
            try:
                self.widget.set_h5_path(h5_path)
                entry_text = str(calibration.get("entry_id", ""))
                index = self.widget.combo_entry_id.findText(entry_text)
                if index >= 0:
                    self.widget.combo_entry_id.setCurrentIndex(index)
            except Exception as exc:
                self.widget.log(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось восстановить HDF5/Entry ID: {exc}")
        branch_index = self.widget.combo_branch.findData(str(calibration.get("branch", "up")))
        if branch_index >= 0:
            self.widget.combo_branch.setCurrentIndex(branch_index)

    @QtCore.pyqtSlot(dict)
    def _publish(self, result: dict):
        try:
            a, b = float(result["A"]), float(result["B"])
            u_cal = float(result["reference_amplitude_mVpp"])
            if not all(math.isfinite(v) for v in (a, b, u_cal)) or abs(a) < 1e-15 or u_cal <= 0:
                raise ValueError("Расчёт вернул недопустимые A/B или амплитуду Uк.")
            payload = {**result, "schema_version": 3, "method": "scale_by_drive_amplitude_theta_A_s", "unit": "угл. с", "reference_amplitude_mVpp": u_cal, "calibrated_at": datetime.now(timezone.utc).isoformat()}
            self._last_payload = payload
            self.widget.log(f"[КАЛИБРОВКА] Aк={a:.8g}; Bк={b:.8g} угл. с (B только для совмещения); Uк={u_cal:.4g} mVpp. φк найдена автоматически.")
            self.calibration_saved.emit(payload)
        except Exception as exc:
            self.widget.log(f"[ОШИБКА] Не удалось передать коэффициенты дашборду: {exc}")
