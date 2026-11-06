from __future__ import annotations

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QMessageBox

from roc_mcs.analysis.project import WorkflowProject
from .workbench_widget import ProcessingWorkbenchWidget


class ProcessingWorkbenchWindow(QMainWindow):
    def __init__(self, project: WorkflowProject, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROC-MCS · Processing Workbench")
        self.resize(1420, 880)
        self.workbench = ProcessingWorkbenchWidget(project, self)
        self.setCentralWidget(self.workbench)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.workbench.is_busy:
            QMessageBox.information(
                self,
                "Операция выполняется",
                "Перед закрытием дождитесь завершения вычисления или загрузки предпросмотра.",
            )
            event.ignore()
            return
        self.workbench._restore_preview()
        try:
            self.workbench.project.save()
        except Exception as exc:
            QMessageBox.critical(
                self, "Не удалось сохранить проект",
                f"История обработки не сохранена; окно оставлено открытым.\n\n{exc}",
            )
            event.ignore()
            return
        super().closeEvent(event)
