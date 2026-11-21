"""ROI Explorer adapter for the canonical Processing Workbench graph.

This module owns per-scan project lifecycle only. All graph editing, node
execution, review/acceptance, provenance and artifact loading stay in the
existing ProcessingWorkbenchWidget so the embedded and standalone workflows
share one implementation.
"""
from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QStandardPaths, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from roc_mcs.analysis.project import WorkflowProject
from .workbench_widget import ProcessingWorkbenchWidget


class ProcessingStepsPanel(QWidget):
    """Compact, full-featured workflow host used in ROI Explorer's sidebar."""

    preview_ready = pyqtSignal(object)
    restore_requested = pyqtSignal()
    busy_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: WorkflowProject | None = None
        self.workbench: ProcessingWorkbenchWidget | None = None
        self._source_key: tuple[str, str] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(4)
        self.project_label = QLabel("Проект обработки не открыт", self)
        self.project_label.setWordWrap(True)
        self.project_label.setStyleSheet("color: palette(placeholderText); font-size: 9pt;")
        root.addWidget(self.project_label)
        self._workbench_host = QWidget(self)
        self._workbench_layout = QVBoxLayout(self._workbench_host)
        self._workbench_layout.setContentsMargins(0, 0, 0, 0)
        self._workbench_layout.setSpacing(0)
        root.addWidget(self._workbench_host, 1)
        self.status = QLabel("Выберите HDF5 и скан", self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: palette(placeholderText); font-size: 8pt;")
        root.addWidget(self.status)

    @property
    def is_busy(self) -> bool:
        return bool(self.workbench is not None and self.workbench.is_busy)

    def apply_theme(self, theme: str) -> None:
        """Keep Workbench forms and contextual popups in the host theme."""
        if self.workbench is not None:
            self.workbench._apply_theme(theme)

    def _set_status(self, message: str) -> None:
        self.status.setText(str(message))
        self.status_changed.emit(str(message))

    def _drop_workbench(self) -> None:
        old = self.workbench
        if old is None:
            return
        if old.is_busy:
            raise RuntimeError("Нельзя сменить скан во время выполнения workflow")
        # A preview is temporary: restore the host's HDF5 view before the old
        # workbench is detached, then persist its workflow history atomically.
        old._restore_preview()
        old.project.save()
        self._workbench_layout.removeWidget(old)
        old.setParent(None)
        old.deleteLater()
        self.workbench = None

    def set_source(self, h5_path: str | Path, scan_key: str) -> None:
        """Open or create the standard per-file/per-scan analysis project."""
        path = Path(h5_path).expanduser().resolve()
        key = (str(path), str(scan_key))
        if self.is_busy:
            self._set_status("Переключение скана недоступно: workflow ещё выполняется")
            return
        if key == self._source_key and self.project is not None and self.workbench is not None:
            return

        try:
            if self.workbench is not None:
                self._drop_workbench()

            safe_scan = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scan_key)).strip("._") or "scan"
            documents = Path(
                QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
                or (Path.home() / "Documents")
            )
            project_dir = documents / "ROC-MCS Analyses" / f"{path.stem}_{safe_scan}.roi-analysis"
            manifest = project_dir / WorkflowProject.MANIFEST_NAME
            container_revision_changed = False
            if manifest.is_file():
                project = WorkflowProject.open(manifest)
                metadata = project.source_metadata or {}
                recorded_path = Path(str(metadata.get("source_file", ""))).expanduser().resolve()
                recorded_scan_key = str(metadata.get("scan_key", ""))
                if recorded_path != path or recorded_scan_key != str(scan_key):
                    raise RuntimeError(
                        "Существующий Analysis Project принадлежит другому HDF5-файлу "
                        "или другому scan_key. Существующая история не будет перезаписана."
                    )
                try:
                    recorded_size = int(metadata.get("source_size_bytes", -1))
                    recorded_mtime = int(metadata.get("source_mtime_ns", -1))
                    current_stat = path.stat()
                    container_revision_changed = (
                        recorded_size != int(current_stat.st_size)
                        or recorded_mtime != int(current_stat.st_mtime_ns)
                    )
                except (OSError, TypeError, ValueError):
                    container_revision_changed = False
            else:
                project = WorkflowProject.create(project_dir, path, str(scan_key))

            if project.source_node_id is None:
                raise RuntimeError("В Analysis Project отсутствует узел исходного скана")

            workbench = ProcessingWorkbenchWidget(project, self._workbench_host, embedded=True)
            workbench.preview_state_ready.connect(self.preview_ready.emit)
            workbench.preview_restore_requested.connect(self.restore_requested.emit)
            workbench.busy_changed.connect(self.busy_changed.emit)
            self._workbench_layout.addWidget(workbench)
            self.project = project
            self.workbench = workbench
            self._source_key = key
            self.project_label.setText(project.directory.name)
            self.project_label.setToolTip(str(project.directory))
            self._set_status(
                "Открыта сохранённая сессия; контейнер HDF5 изменился, но идентификатор скана совпадает."
                if container_revision_changed else
                "История workflow готова. Выберите узел; глаз открывает результат, + добавляет этап."
            )
            self.busy_changed.emit(self.is_busy)
        except Exception as exc:
            self.project = None
            self._source_key = None
            self.project_label.setText("Проект обработки не открыт")
            self._set_status(f"Проект не открыт: {exc}")
            self.busy_changed.emit(False)
