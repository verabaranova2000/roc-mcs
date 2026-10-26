from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QLineEdit, QToolButton, QFileDialog,
    QFormLayout, QHBoxLayout, QDialogButtonBox, QMessageBox
)


class H5SettingsDialog(QDialog):
    """ Диалог настроек пути к H5 """
    def __init__(self, current_path="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки HDF5")
        self.setModal(True)
        self.setMinimumWidth(520)

        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("Путь к .h5 файлу")
        self.path_edit.setText(str(current_path or ""))

        self.btn_browse = QToolButton(self)
        self.btn_browse.setText("...")
        self.btn_browse.clicked.connect(self.browse_file)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.btn_browse)

        form = QFormLayout(self)
        form.addRow("HDF5 файл:", path_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Применить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите HDF5 файл",
            str(Path.home()),
            "Файлы HDF5 (*.h5 *.hdf5);;Все файлы (*)",
        )
        if file_path:
            self.path_edit.setText(file_path)

    def get_path(self):
        return self.path_edit.text().strip()

    def accept(self):
        raw_path = self.get_path()
        if not raw_path:
            QMessageBox.warning(self, "Ошибка", "Укажите путь к HDF5 файлу.")
            return
        path = Path(raw_path)
        if not path.is_file():
            QMessageBox.warning(self, "Ошибка", "Файл не найден.")
            return
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            QMessageBox.warning(self, "Ошибка", "Нужен файл .h5 или .hdf5.")
            return
        super().accept()