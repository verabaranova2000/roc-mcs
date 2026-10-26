"""Настройки экспорта по существующему pipeline/build.py."""
from pathlib import Path
from PyQt6 import QtCore, QtWidgets

class ExportOptionsDialog(QtWidgets.QDialog):
    def __init__(self, experiment_folder: str | Path, parent=None):
        super().__init__(parent); self.experiment_folder = Path(experiment_folder).resolve()
        self.setWindowTitle("Экспорт результатов эксперимента"); self.setMinimumWidth(460)
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(8)
        note = QtWidgets.QLabel("Обработка выполняется существующим pipeline/build.py по завершённым MCS-файлам. Перед экспортом остановите наблюдение.")
        note.setWordWrap(True); root.addWidget(note)
        form = QtWidgets.QFormLayout(); form.setHorizontalSpacing(6); form.setVerticalSpacing(6)
        row = QtWidgets.QWidget(); rr = QtWidgets.QHBoxLayout(row); rr.setContentsMargins(0,0,0,0); rr.setSpacing(5)
        self.output_folder = QtWidgets.QLineEdit(str(self.experiment_folder / "results")); self.output_folder.setToolTip("Папка для файлов экспорта; по умолчанию results внутри опыта.")
        browse = QtWidgets.QToolButton(); browse.setText("…"); browse.setToolTip("Выбрать папку экспорта"); browse.setFixedWidth(30); browse.clicked.connect(self._browse)
        rr.addWidget(self.output_folder,1); rr.addWidget(browse); form.addRow("Каталог", row); root.addLayout(form)
        group = QtWidgets.QGroupBox("Что сохранить"); grid = QtWidgets.QVBoxLayout(group); grid.setSpacing(5)
        self.chk_excel = QtWidgets.QCheckBox("Книга Excel (.xlsx): РОК, метаданные, калибровка, нагрузка/давление"); self.chk_excel.setChecked(True)
        self.chk_figures = QtWidgets.QCheckBox("Графики PNG: ROC-карта и динамика моментов"); self.chk_figures.setChecked(True)
        self.chk_moments = QtWidgets.QCheckBox("Рассчитать профильные моменты для таблиц и графиков"); self.chk_moments.setChecked(True)
        self.chk_diagnostics = QtWidgets.QCheckBox("Диагностические графики QC (дополнительные) ")
        self.chk_artifact = QtWidgets.QCheckBox("Полный объект анализа experiment_artifact.pkl (может быть большим)")
        for item in (self.chk_excel, self.chk_figures, self.chk_moments, self.chk_diagnostics, self.chk_artifact): grid.addWidget(item)
        root.addWidget(group)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Экспортировать"); buttons.accepted.connect(self._accept); buttons.rejected.connect(self.reject); root.addWidget(buttons)

    def _browse(self):
        initial = self.output_folder.text().strip() or str(self.experiment_folder / "results")
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Каталог результатов", initial)
        if selected: self.output_folder.setText(selected)

    def _accept(self):
        if not self.output_folder.text().strip():
            QtWidgets.QMessageBox.warning(self, "Не задан каталог", "Укажите каталог для результатов."); return
        if not any(x.isChecked() for x in (self.chk_excel, self.chk_figures, self.chk_diagnostics, self.chk_artifact)):
            QtWidgets.QMessageBox.warning(self, "Нет результатов", "Выберите хотя бы один тип файлов для сохранения."); return
        self.accept()

    def collect_options(self) -> dict:
        return {"output_folder": self.output_folder.text().strip(), "save_excel_flag": self.chk_excel.isChecked(),
                "save_figure_flag": self.chk_figures.isChecked(), "compute_moments": self.chk_moments.isChecked(),
                "save_diagnostics": self.chk_diagnostics.isChecked(), "save_artifact": self.chk_artifact.isChecked()}
