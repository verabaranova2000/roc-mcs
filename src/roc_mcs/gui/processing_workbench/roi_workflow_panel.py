"""Компактная панель первых этапов Workbench внутри ROI Explorer."""
from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QObject, QRunnable, QStandardPaths, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLineEdit,
    QSpinBox, QDoubleSpinBox, QPushButton, QLabel, QFormLayout,
)

from roc_mcs.analysis.artifacts import ArtifactStore
from roc_mcs.analysis.engine import (
    WorkflowExecutor, application_version, build_input_refs,
    make_operation_label, source_fingerprint,
)
from roc_mcs.gui.resource_monitor.activity import activity
from roc_mcs.analysis.operations import default_registry
from roc_mcs.analysis.project import WorkflowProject
from roc_mcs.analysis.scan_view import ScanViewState, build_scan_view_state
from roc_mcs.analysis.workflow_history import (
    NodeStatus, ReviewStatus, WorkflowHistory,
)


class _TaskSignals(QObject):
    succeeded = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)


class _RunTask(QRunnable):
    def __init__(self, node_id: str, history: WorkflowHistory, artifact_dir: str):
        super().__init__()
        self.node_id = node_id
        self.history = WorkflowHistory(list(history.nodes))
        self.artifact_dir = artifact_dir
        self.signals = _TaskSignals()

    def run(self) -> None:
        try:
            executor = WorkflowExecutor(
                self.history, ArtifactStore(self.artifact_dir), default_registry()
            )
            node = self.history.get(self.node_id)
            with activity(f"Analysis · {node.label}"):
                outcome = executor.execute_node(self.node_id)
            self.signals.succeeded.emit(self.node_id, outcome)
        except Exception:
            self.signals.failed.emit(self.node_id, traceback.format_exc())


class _PreviewTask(QRunnable):
    def __init__(self, node_id: str, history: WorkflowHistory, artifact_dir: str):
        super().__init__()
        self.node_id = node_id
        self.history = WorkflowHistory(list(history.nodes))
        self.artifact_dir = artifact_dir
        self.signals = _TaskSignals()

    def run(self) -> None:
        try:
            executor = WorkflowExecutor(
                self.history, ArtifactStore(self.artifact_dir), default_registry()
            )
            node = self.history.get(self.node_id)
            source_node = next(n for n in self.history.nodes if n.operation_id == "source.scan")
            source_meta = source_node.metadata
            current_source = source_fingerprint(source_meta["source_file"])
            expected_source = {
                "path": source_meta["source_file"],
                "size_bytes": int(source_meta["source_size_bytes"]),
                "mtime_ns": int(source_meta["source_mtime_ns"]),
            }
            if current_source != expected_source:
                raise RuntimeError("Исходный HDF5 изменился после создания Analysis Project")
            if node.operation_id == "preprocess.spikes":
                inputs = executor.resolve_node_inputs(node)
                scan = inputs["scan"]
                spikes = executor.load_output(node, "spikes")
            else:
                scan = executor.load_output(node, "scan")
                spikes = None
            with activity(f"Analysis preview · {node.label}"):
                state = build_scan_view_state(node, scan, spikes)
            self.signals.succeeded.emit(self.node_id, state)
        except Exception:
            self.signals.failed.emit(self.node_id, traceback.format_exc())


class ProcessingStepsPanel(QWidget):
    """Embedded source → detector QC → spikes workflow; all results remain reviewable."""

    preview_ready = pyqtSignal(object)
    restore_requested = pyqtSignal()
    busy_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: WorkflowProject | None = None
        self._source_key: tuple[str, str] | None = None
        self._active_task: QRunnable | None = None
        self._active_preview: QRunnable | None = None
        self._selected_preview_node: str | None = None
        self._thread_pool = QThreadPool.globalInstance()
        self._build_ui()
        self._set_enabled(False)

    @property
    def is_busy(self) -> bool:
        return self._active_task is not None or self._active_preview is not None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(5)
        self.project_label = QLabel("Проект обработки не открыт")
        self.project_label.setWordWrap(True)
        self.project_label.setStyleSheet("color: palette(placeholderText);")
        root.addWidget(self.project_label)
        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumHeight(105)
        self.tree.setMaximumHeight(190)
        self.tree.setIndentation(14)
        self.tree.currentItemChanged.connect(self._update_controls)
        root.addWidget(self.tree)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.pixels_edit = QLineEdit(self)
        self.pixels_edit.setPlaceholderText("x,y; x,y · пусто = явно нет дефектов")
        try:
            from roc_mcs.gui.app_settings import load_detector_bad_pixels
            self.pixels_edit.setText("; ".join(f"{x},{y}" for x, y in load_detector_bad_pixels()))
        except Exception:
            pass
        self.pixels_edit.setToolTip("Индексы исходной detector-grid с нуля: x,y; x,y")
        form.addRow("Дефекты (x,y):", self.pixels_edit)
        root.addLayout(form)

        buttons = QHBoxLayout()
        self.qc_button = QPushButton("Подтвердить дефекты")
        self.qc_button.setToolTip("Фиксирует список дефектов или явное подтверждение пустого списка")
        self.qc_button.clicked.connect(self._create_qc)
        self.spikes_button = QPushButton("Интерполировать спайки")
        self.spikes_button.clicked.connect(self._create_spikes)
        buttons.addWidget(self.qc_button)
        buttons.addWidget(self.spikes_button)
        root.addLayout(buttons)

        params = QHBoxLayout()
        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 100); self.max_iter.setValue(5)
        self.tolerance = QDoubleSpinBox(); self.tolerance.setRange(1e-6, 0.1); self.tolerance.setDecimals(6); self.tolerance.setSingleStep(0.001); self.tolerance.setValue(0.001)
        params.addWidget(QLabel("Итерации")); params.addWidget(self.max_iter)
        params.addWidget(QLabel("tol")); params.addWidget(self.tolerance)
        root.addLayout(params)

        actions = QHBoxLayout()
        self.preview_button = QPushButton("Показать")
        self.preview_button.clicked.connect(self._preview_selected)
        self.accept_button = QPushButton("Принять")
        self.accept_button.clicked.connect(self._accept_selected)
        self.reject_button = QPushButton("Отклонить")
        self.reject_button.clicked.connect(self._reject_selected)
        self.restore_button = QPushButton("Исходный вид")
        self.restore_button.clicked.connect(self._restore)
        for button in (self.preview_button, self.accept_button, self.reject_button, self.restore_button):
            actions.addWidget(button)
        root.addLayout(actions)
        self.status = QLabel("Выберите HDF5 и скан")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: palette(placeholderText);")
        root.addWidget(self.status)

    def _set_enabled(self, enabled: bool) -> None:
        self.tree.setEnabled(enabled)
        self.pixels_edit.setEnabled(enabled)
        self.max_iter.setEnabled(enabled)
        self.tolerance.setEnabled(enabled)
        self._update_controls()

    def set_source(self, h5_path: str | Path, scan_key: str) -> None:
        """Open/create the standard Documents project without loading the volume."""
        path = Path(h5_path).expanduser().resolve()
        key = (str(path), str(scan_key))
        if self.is_busy:
            self._set_status("Нельзя переключить скан: этап обработки ещё выполняется")
            return
        if key == self._source_key and self.project is not None:
            return
        self.project = None
        self._source_key = None
        self.tree.clear()
        try:
            safe_scan = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scan_key)).strip("._") or "scan"
            documents = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation) or (Path.home() / "Documents"))
            project_dir = documents / "ROC-MCS Analyses" / f"{path.stem}_{safe_scan}.roi-analysis"
            manifest = project_dir / WorkflowProject.MANIFEST_NAME
            if manifest.is_file():
                project = WorkflowProject.open(manifest)
                metadata = project.source_metadata or {}
                expected = source_fingerprint(path)
                actual = {
                    "path": str(metadata.get("source_file", "")),
                    "size_bytes": int(metadata.get("source_size_bytes", -1)),
                    "mtime_ns": int(metadata.get("source_mtime_ns", -1)),
                }
                if actual != expected or str(metadata.get("scan_key", "")) != str(scan_key):
                    raise RuntimeError(
                        "Существующий Analysis Project относится к другой версии исходного HDF5/скану. "
                        "Он не будет перезаписан; выберите/создайте отдельную сессию анализа."
                    )
            else:
                project = WorkflowProject.create(project_dir, path, str(scan_key))
            if project.source_node_id is None:
                raise RuntimeError("В Analysis Project отсутствует узел исходного скана")
            self.project = project
            self._source_key = key
            self.project_label.setText(f"{project.directory.name}")
            self.project_label.setToolTip(str(project.directory))
            self._refresh_tree(project.source_node_id)
            self._set_enabled(True)
            self._set_status("История обработки готова. Исходный HDF5 не изменяется.")
        except Exception as exc:
            self.project = None
            self._source_key = None
            self.project_label.setText("Проект обработки не открыт")
            self._set_enabled(False)
            self._set_status(f"Проект не открыт: {exc}")

    def _source_node_id(self) -> str | None:
        return None if self.project is None else self.project.source_node_id

    def _selected_node_id(self) -> str | None:
        item = self.tree.currentItem()
        return None if item is None else item.data(0, Qt.ItemDataRole.UserRole)

    @staticmethod
    def _status_text(node) -> str:
        status = {
            NodeStatus.PENDING: "ожидает запуска", NodeStatus.RUNNING: "выполняется",
            NodeStatus.SUCCEEDED: "выполнен", NodeStatus.FAILED: "ошибка",
            NodeStatus.CANCELLED: "отменён",
        }.get(node.status, node.status)
        review = {
            ReviewStatus.CANDIDATE: " · на проверке",
            ReviewStatus.ACCEPTED: " · принят",
            ReviewStatus.REJECTED: " · отклонён",
        }.get(node.review_status, "")
        return f"{status}{review}"

    def _refresh_tree(self, select_id: str | None = None) -> None:
        if self.project is None:
            self.tree.clear(); return
        self.tree.clear()
        items: dict[str, QTreeWidgetItem] = {}
        nodes = self.project.history.nodes
        source = next((n for n in nodes if n.operation_id == "source.scan"), None)
        if source is None:
            return
        root = QTreeWidgetItem([f"{source.label} · исходный HDF5"])
        root.setData(0, Qt.ItemDataRole.UserRole, source.id)
        items[source.id] = root
        self.tree.addTopLevelItem(root)
        for node in nodes:
            if node.id == source.id:
                continue
            parent_item = next((items[parent] for parent in node.parent_ids if parent in items), root)
            text = f"{node.label}  [{self._status_text(node)}]"
            item = QTreeWidgetItem([text])
            item.setData(0, Qt.ItemDataRole.UserRole, node.id)
            parent_item.addChild(item)
            items[node.id] = item
        root.setExpanded(True)
        for item in items.values():
            if item.childCount(): item.setExpanded(True)
        target = items.get(select_id or (nodes[-1].id if nodes else source.id), root)
        self.tree.setCurrentItem(target)
        self._update_controls()

    def _update_controls(self, *_args) -> None:
        busy = self.is_busy
        node = None
        if self.project is not None and self._selected_node_id() is not None:
            try: node = self.project.history.get(self._selected_node_id())
            except KeyError: pass
        selected_success = bool(node and node.status == NodeStatus.SUCCEEDED)
        candidate = bool(selected_success and node.review_status == ReviewStatus.CANDIDATE)
        selected_qc_accepted = bool(
            node and node.operation_id == "preprocess.detector_bad_pixels"
            and node.status == NodeStatus.SUCCEEDED
            and node.review_status == ReviewStatus.ACCEPTED
        )
        self.qc_button.setEnabled(not busy and self.project is not None)
        self.spikes_button.setEnabled(not busy and selected_qc_accepted)
        self.preview_button.setEnabled(not busy and selected_success)
        self.accept_button.setEnabled(not busy and candidate)
        self.reject_button.setEnabled(not busy and candidate and not self.project.history.children(node.id))
        self.restore_button.setEnabled(not busy and self._selected_preview_node is not None)

    def _set_status(self, value: str) -> None:
        self.status.setText(value)
        self.status_changed.emit(value)

    def _create_qc(self) -> None:
        if self.project is None or self.is_busy:
            return
        try:
            source_id = self._source_node_id()
            params = {"pixels": self.pixels_edit.text()}
            self._create_candidate(source_id, "preprocess.detector_bad_pixels", params)
        except Exception as exc:
            self._set_status(f"Не удалось создать этап дефектов: {exc}")

    def _create_spikes(self) -> None:
        if self.project is None or self.is_busy:
            return
        try:
            selected_id = self._selected_node_id()
            selected = self.project.history.get(selected_id) if selected_id else None
            if not (selected and selected.operation_id == "preprocess.detector_bad_pixels"
                    and selected.status == NodeStatus.SUCCEEDED
                    and selected.review_status == ReviewStatus.ACCEPTED):
                raise ValueError("Выберите в дереве и примите конкретный результат подтверждения дефектных пикселей")
            params = {"max_iter": self.max_iter.value(), "tol": self.tolerance.value(), "collect_kappa_fits": True}
            self._create_candidate(selected.id, "preprocess.spikes", params)
        except Exception as exc:
            self._set_status(f"Не удалось создать этап спайков: {exc}")

    def _create_candidate(self, parent_id: str, operation_id: str, raw_params: dict[str, Any]) -> None:
        assert self.project is not None
        spec = default_registry().get(operation_id)
        params = spec.validate(raw_params)
        refs = build_input_refs(self.project.history, parent_id, operation_id, params)
        node = self.project.history.add_node(
            operation_id, make_operation_label(operation_id, params),
            parent_ids=[parent_id], input_refs=refs, parameters=params,
            metadata={"created_in": "roi_explorer", "project_schema": 1},
            software_version=application_version(), parameter_schema_version=spec.parameter_schema_version,
            review_status=ReviewStatus.CANDIDATE,
        )
        self.project.save()
        self._refresh_tree(node.id)
        self._run_node(node.id)

    def _run_node(self, node_id: str) -> None:
        assert self.project is not None
        self.project.history.transition(node_id, NodeStatus.RUNNING)
        self.project.save()
        task = _RunTask(node_id, self.project.history, str(self.project.artifact_store.directory))
        task.signals.succeeded.connect(self._run_succeeded)
        task.signals.failed.connect(self._run_failed)
        self._active_task = task
        self._update_controls(); self.busy_changed.emit(True)
        self._set_status(f"Выполняется: {self.project.history.get(node_id).label}")
        self._thread_pool.start(task)

    def _run_succeeded(self, node_id: str, outcome: Any) -> None:
        try:
            assert self.project is not None
            self.project.history.transition(
                node_id, NodeStatus.SUCCEEDED,
                output_refs={port: outcome.artifact_id for port in outcome.output_ports},
                summary=outcome.summary,
            )
            self.project.save()
            summary = outcome.summary
        except Exception:
            self._set_status("Артефакт вычислен, но сохранить историю не удалось: " + traceback.format_exc())
            self._finish_task(node_id)
            return
        self._finish_task(node_id)
        self._set_status(f"Готово: {summary} · проверьте результат и примите или отклоните")
        self._preview_node(node_id)

    def _run_failed(self, node_id: str, error: str) -> None:
        try:
            assert self.project is not None
            self.project.history.transition(node_id, NodeStatus.FAILED, error=error)
            self.project.save()
        except Exception:
            error += "\nНе удалось сохранить состояние ошибки:\n" + traceback.format_exc()
        self._finish_task(node_id)
        self._set_status("Этап не выполнен:\n" + error[-2500:])

    def _finish_task(self, node_id: str) -> None:
        self._active_task = None
        self._refresh_tree(node_id)
        self._update_controls(); self.busy_changed.emit(False)

    def _preview_selected(self) -> None:
        node_id = self._selected_node_id()
        if node_id is not None:
            self._preview_node(node_id)

    def _preview_node(self, node_id: str) -> None:
        if self.project is None or self.is_busy:
            return
        node = self.project.history.get(node_id)
        if node.status != NodeStatus.SUCCEEDED:
            self._set_status("Предпросмотр доступен только после успешного выполнения этапа")
            return
        task = _PreviewTask(node_id, self.project.history, str(self.project.artifact_store.directory))
        task.signals.succeeded.connect(self._preview_succeeded)
        task.signals.failed.connect(self._preview_failed)
        self._active_preview = task
        self._update_controls(); self.busy_changed.emit(True)
        self._set_status(f"Подготовка представления: {node.label}")
        self._thread_pool.start(task)

    def _preview_succeeded(self, node_id: str, state: ScanViewState) -> None:
        self._active_preview = None
        self._selected_preview_node = node_id
        self.preview_ready.emit(state)
        self._refresh_tree(node_id)
        self._update_controls(); self.busy_changed.emit(False)
        self._set_status(f"Показан результат: {state.label}")

    def _preview_failed(self, node_id: str, error: str) -> None:
        self._active_preview = None
        self._update_controls(); self.busy_changed.emit(False)
        self._set_status("Не удалось подготовить предпросмотр:\n" + error[-2500:])

    def _accept_selected(self) -> None:
        if self.project is None or self.is_busy:
            return
        node_id = self._selected_node_id()
        if node_id is None:
            return
        try:
            self.project.history.set_review_status(node_id, ReviewStatus.ACCEPTED)
            self.project.save()
            self._refresh_tree(node_id)
            self._set_status("Результат принят и разрешён как вход следующего этапа")
        except Exception as exc:
            self._set_status(f"Не удалось принять результат: {exc}")

    def _reject_selected(self) -> None:
        if self.project is None or self.is_busy:
            return
        node_id = self._selected_node_id()
        if node_id is None:
            return
        try:
            node = self.project.history.get(node_id)
            if self.project.history.children(node_id):
                raise ValueError("Сначала отклоните дочерние этапы")
            self.project.history.set_review_status(node_id, ReviewStatus.REJECTED)
            self.project.save()
            if self._selected_preview_node == node_id:
                self._restore()
            self._refresh_tree(node.parent_ids[0] if node.parent_ids else None)
            self._set_status("Результат отклонён; артефакт сохранён в истории")
        except Exception as exc:
            self._set_status(f"Не удалось отклонить результат: {exc}")

    def _restore(self) -> None:
        self._selected_preview_node = None
        self.restore_requested.emit()
        self._update_controls()
        self._set_status("Возвращено исходное представление текущего скана")
