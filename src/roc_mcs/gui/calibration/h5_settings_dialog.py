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
        self.setWindowTitle("HDF5 settings")
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
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите HDF5 файл",
            str(Path.home()),
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if file_path:
            self.path_edit.setText(file_path)

    def get_path(self):
        return self.path_edit.text().strip()

    def accept(self):
        path = Path(self.get_path())
        if not path:
            QMessageBox.warning(self, "Ошибка", "Укажите путь к HDF5 файлу.")
            return
        if not path.exists():
            QMessageBox.warning(self, "Ошибка", "Файл не найден.")
            return
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            QMessageBox.warning(self, "Ошибка", "Нужен файл .h5 или .hdf5.")
            return
        super().accept()