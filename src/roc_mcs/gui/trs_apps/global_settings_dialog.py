"""Глобальные настройки всей оболочки TRS Apps."""
from __future__ import annotations
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from roc_mcs.gui.components.dialog_chrome import install_dialog_titlebar
from roc_mcs.gui.roi_explorer.theme import apply_popup_theme, resolve_theme


class TRSGlobalSettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings: dict | None = None, parent=None):
        super().__init__(parent); self.setObjectName("trsGlobalSettings"); self._settings = dict(settings or {})
        self.setMinimumSize(500, 310); self.resize(570, 350)
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(8, 7, 8, 8); root.setSpacing(8); install_dialog_titlebar(self, root, "Глобальные настройки TRS Apps")
        form_host = QtWidgets.QWidget(self); form = QtWidgets.QFormLayout(form_host); form.setContentsMargins(0, 0, 0, 0); form.setHorizontalSpacing(12); form.setVerticalSpacing(8); form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow); root.addWidget(form_host)
        self.theme = QtWidgets.QComboBox(self); self.theme.addItem("Тёмная", "dark"); self.theme.addItem("Светлая", "light"); self.theme.addItem("Системная", "system")
        theme = str(self._settings.get("theme", "dark")).lower(); self.theme.setCurrentIndex(max(0, self.theme.findData(theme))); form.addRow("Тема интерфейса", self.theme)
        self.icon_size = QtWidgets.QSpinBox(self); self.icon_size.setRange(12, 20); self.icon_size.setValue(self._safe_int(self._settings.get("icon_size", 16), 16, 12, 20)); self.icon_size.setFixedWidth(82); self.icon_size.setKeyboardTracking(False); form.addRow("Размер значков, px", self.icon_size)
        self.h5_path = self._path_row(form, "Основной файл HDF5", str(self._settings.get("last_hdf5_database", "") or ""), is_file=True, filters="HDF5 (*.h5 *.hdf5 *.hdf);;Все файлы (*)")
        self.experiments_dir = self._path_row(form, "Папка экспериментов", str(self._settings.get("experiments_dir", "") or ""), is_file=False)
        hint = QtWidgets.QLabel("Эти параметры принадлежат TRS Apps и общие для встроенных приложений. HDF5 используется как база по умолчанию при открытии ROI Explorer, папка экспериментов — в Live Scans. Уже открытая база ROI Explorer не заменяется автоматически.", self); hint.setWordWrap(True); hint.setObjectName("settingsHint"); root.addWidget(hint)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel, parent=self); buttons.accepted.connect(self._validate_and_accept); buttons.rejected.connect(self.reject); root.addWidget(buttons)
        self.setStyleSheet(self.styleSheet() + """
            QDialog#trsGlobalSettings { background:palette(window); color:palette(windowText); }
            QDialog#trsGlobalSettings QLineEdit, QComboBox, QSpinBox { background:palette(base); color:palette(text); border:1px solid palette(mid); border-radius:4px; padding:4px 6px; min-height:22px; }
            QDialog#trsGlobalSettings QAbstractSpinBox::up-button, QDialog#trsGlobalSettings QAbstractSpinBox::down-button { width:0; height:0; border:0; }
            QDialog#trsGlobalSettings QPushButton { padding:5px 10px; min-height:23px; }
            QDialog#trsGlobalSettings QLabel#settingsHint { color:palette(mid); background:transparent; font-size:10pt; }
        """)
        apply_popup_theme(self, resolve_theme(parent))

    @staticmethod
    def _safe_int(value, default, low, high):
        try: return min(high, max(low, int(value)))
        except (TypeError, ValueError, OverflowError): return default

    def _path_row(self, form, label, value, *, is_file, filters=""):
        host = QtWidgets.QWidget(self); row = QtWidgets.QHBoxLayout(host); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(5)
        edit = QtWidgets.QLineEdit(value, host); edit.setClearButtonEnabled(True); edit.setPlaceholderText("Не задано")
        button = QtWidgets.QToolButton(host); button.setText("…"); button.setFixedWidth(30); button.setToolTip("Выбрать файл" if is_file else "Выбрать папку")
        def browse():
            initial = edit.text().strip() or str(Path.home())
            if is_file: path, _ = QtWidgets.QFileDialog.getOpenFileName(self, label, initial, filters)
            else: path = QtWidgets.QFileDialog.getExistingDirectory(self, label, initial)
            if path: edit.setText(path)
        button.clicked.connect(browse); row.addWidget(edit, 1); row.addWidget(button); form.addRow(label, host); return edit

    def _validate_and_accept(self):
        h5 = self.h5_path.text().strip(); folder = self.experiments_dir.text().strip()
        if h5 and not Path(h5).expanduser().is_file():
            QtWidgets.QMessageBox.warning(self, "Путь HDF5", "Указанный файл HDF5 не найден. Выберите существующий файл или очистите поле."); return
        if folder and not Path(folder).expanduser().is_dir():
            QtWidgets.QMessageBox.warning(self, "Папка экспериментов", "Указанная папка не найдена. Выберите существующую папку или очистите поле."); return
        self.accept()

    def values(self) -> dict:
        return {"theme": str(self.theme.currentData() or "dark"), "icon_size": self.icon_size.value(), "last_hdf5_database": self.h5_path.text().strip(), "experiments_dir": self.experiments_dir.text().strip()}
