"""ROI Explorer adapter for the canonical Processing Workbench graph.

This module owns per-scan project lifecycle only. All graph editing, node
execution, review/acceptance, provenance and artifact loading stay in the
existing ProcessingWorkbenchWidget so the embedded and standalone workflows
share one implementation.
"""
from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QStandardPaths, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPalette
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from roc_mcs.analysis.project import WorkflowProject
from .workbench_widget import ProcessingWorkbenchWidget


class _VerticalResizeGrip(QWidget):
    """Small splitter-like handle that grows the workflow section in the sidebar."""

    def __init__(self, owner: "ProcessingStepsPanel"):
        super().__init__(owner)
        self._owner = owner
        self._drag_origin_y: int | None = None
        self._start_height = 0
        self.setObjectName("processingGraphHeightGrip")
        # A narrow, in-flow splitter handle. It is inserted directly below
        # the graph, not at the bottom of the processing controls.
        self.setFixedHeight(7)
        self.setMinimumWidth(44)
        self.setCursor(Qt.CursorShape.SplitVCursor)
        self.setToolTip(
            "Перетащите вверх или вниз, чтобы изменить высоту области дерева обработки"
        )
        self.setAccessibleName("Изменить высоту дерева обработки")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = self.palette().color(
            QPalette.ColorRole.Highlight if self.underMouse() else QPalette.ColorRole.Mid
        )
        color.setAlpha(175 if self.underMouse() else 115)
        painter.setPen(QPen(color, 1.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        painter.drawLine(round(cx - 15), round(cy), round(cx + 15), round(cy))
        painter.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for offset in (-3, 0, 3):
            painter.drawPoint(round(cx + offset), round(cy))
        painter.end()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin_y = int(event.globalPosition().toPoint().y())
            self._start_height = int(self._owner._workbench_host.height())
            if self._start_height <= 0:
                self._start_height = int(self._owner._workbench_host.minimumHeight())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin_y is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = int(event.globalPosition().toPoint().y()) - self._drag_origin_y
            self._owner._set_workbench_host_height(
                self._start_height + delta, manual=True
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_origin_y is not None:
            self._drag_origin_y = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ProcessingStepsPanel(QWidget):
    """Compact, full-featured workflow host used in ROI Explorer's sidebar."""

    preview_ready = pyqtSignal(object)
    restore_requested = pyqtSignal()
    busy_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)
    pixel_pick_mode_requested = pyqtSignal(bool, str, bool)

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
        # No hard-coded 320 px: the empty host has zero height.  Once a
        # Workbench is attached, its own layout computes a safe compact height
        # that keeps the graph viewport and every action/status control visible.
        self._workbench_host.setFixedHeight(0)
        root.addWidget(self._workbench_host, 1)
        # The handle is moved into the Workbench layout immediately below the
        # graph when a scan is opened. Keeping it owned here lets its drag update
        # the host height while placing it next to the area it resizes.
        self._height_grip = _VerticalResizeGrip(self)
        self._height_grip.hide()
        self.status = QLabel("Выберите HDF5 и скан", self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: palette(placeholderText); font-size: 8pt;")
        root.addWidget(self.status)

    def _set_workbench_host_height(self, requested: int, *, manual: bool = False) -> None:
        """Apply a safe, exact height to the embedded Workbench host.

        The lower bound comes from the current Workbench layout size hint, not
        an arbitrary constant.  It includes the graph header, graph viewport,
        resize grip, Run/Restore/Accept/Reject controls and status indicators.
        Thus the graph can never consume the space reserved for its own controls.
        """
        workbench = self.workbench
        safe_minimum = (
            workbench.embedded_graph_compact_host_height()
            if workbench is not None else 0
        )
        height = max(safe_minimum, min(100_000, int(requested)))
        if (
            self._workbench_host.minimumHeight() == height
            and self._workbench_host.maximumHeight() == height
            and self._workbench_host.height() == height
        ):
            if manual and workbench is not None:
                workbench._on_manual_graph_height_changed()
            return
        self._workbench_host.setFixedHeight(height)
        self._workbench_host.updateGeometry()
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()
        if workbench is not None:
            if manual:
                workbench._on_manual_graph_height_changed()
            else:
                QTimer.singleShot(0, workbench._sync_graph_scroll_button)

    def _initialize_workbench_height(self, workbench: ProcessingWorkbenchWidget) -> None:
        """Set the initial compact size after Qt has activated the child layouts."""
        if workbench is not self.workbench:
            return
        workbench._auto_fit_graph_height = False
        self._set_workbench_host_height(workbench.embedded_graph_compact_host_height())
        QTimer.singleShot(0, workbench._sync_graph_scroll_button)

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
        if old._active_stage_popup is not None:
            old._active_stage_popup.close()
        # A preview is temporary: restore the host's HDF5 view before the old
        # workbench is detached, then persist its workflow history atomically.
        old._restore_preview()
        old.project.save()
        # The grip is owned by the embedded graph area while a Workbench is
        # alive. Remove it from the old layout explicitly before reparenting so
        # a scan swap cannot leave a stale layout item behind.
        old._left_graph_layout.removeWidget(self._height_grip)
        self._height_grip.setParent(self)
        self._height_grip.hide()
        self._workbench_host.setFixedHeight(0)
        self._workbench_host.updateGeometry()
        self._workbench_layout.removeWidget(old)
        old.setParent(None)
        old.deleteLater()
        self.workbench = None

    def set_peak_preview_pixel(self, detector_x: float, detector_y: float) -> None:
        """Forward a detector-map pick to the active Workbench form."""
        if self.workbench is not None:
            self.workbench.set_peak_preview_pixel(float(detector_x), float(detector_y))

    def cancel_peak_pixel_pick_mode(self) -> None:
        """Synchronize the embedded point-picker toggle after a host-side cancel."""
        if self.workbench is not None:
            self.workbench.cancel_peak_pixel_pick_mode()

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
            workbench.pixel_pick_mode_requested.connect(self.pixel_pick_mode_requested.emit)
            workbench.busy_changed.connect(self.busy_changed.emit)
            workbench.graph_full_height_requested.connect(
                self._set_workbench_host_height
            )
            workbench.graph_compact_height_requested.connect(
                self._set_workbench_host_height
            )
            self._workbench_layout.addWidget(workbench)
            # Keep the resize affordance visually and semantically attached to
            # the graph: it sits between the viewport and action controls.
            graph_layout = workbench._left_graph_layout
            graph_index = graph_layout.indexOf(workbench.graph)
            self._height_grip.setParent(workbench.graph.parentWidget())
            graph_layout.insertWidget(
                graph_index + 1, self._height_grip, 0, Qt.AlignmentFlag.AlignHCenter
            )
            self._height_grip.show()
            self.project = project
            self.workbench = workbench
            self._source_key = key
            QTimer.singleShot(0, lambda wb=workbench: self._initialize_workbench_height(wb))
            self.project_label.setText(project.directory.name)
            self.project_label.setToolTip(str(project.directory))
            self._set_status(
                "Сохранённая сессия открыта. Контейнер HDF5 пополнился или изменился; "
                "выбранный scan_key проверяется отдельно и не инвалидируется добавлением других сканов."
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
