"""Branchable processing workbench, embeddable as a QWidget or standalone window."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QFormLayout, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QLabel, QPlainTextEdit, QTabWidget, QStackedWidget,
    QProgressBar, QMessageBox, QSizePolicy, QFrame,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from roc_mcs.analysis.artifacts import ArtifactStore
from roc_mcs.analysis.engine import (
    WorkflowExecutor, application_version, build_input_refs, make_operation_label,
)
from roc_mcs.analysis.operations import default_registry, fit_backend_availability
from roc_mcs.analysis.project import WorkflowProject
from roc_mcs.analysis.workflow_history import NodeStatus, WorkflowHistory
from .presenter import ArtifactPreviewRequest


_STATUS_MARK = {
    NodeStatus.PENDING: "○",
    NodeStatus.RUNNING: "◌",
    NodeStatus.SUCCEEDED: "✓",
    NodeStatus.FAILED: "!",
    NodeStatus.CANCELLED: "×",
}


class _WorkerSignals(QObject):
    succeeded = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)


class _RunNodeTask(QRunnable):
    def __init__(self, node_id: str, history: WorkflowHistory, artifact_directory: str):
        super().__init__()
        self.node_id = node_id
        self.history_snapshot = WorkflowHistory(list(history.nodes))
        self.artifact_directory = artifact_directory
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            executor = WorkflowExecutor(
                self.history_snapshot,
                ArtifactStore(self.artifact_directory),
                default_registry(),
            )
            outcome = executor.execute_node(self.node_id)
            self.signals.succeeded.emit(self.node_id, outcome)
        except Exception:
            self.signals.failed.emit(self.node_id, traceback.format_exc())


class _PreviewNodeTask(QRunnable):
    """Loads potentially large preview artifacts outside the GUI thread."""

    def __init__(self, node_id: str, history: WorkflowHistory, artifact_directory: str):
        super().__init__()
        self.node_id = node_id
        self.history_snapshot = WorkflowHistory(list(history.nodes))
        self.artifact_directory = artifact_directory
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            executor = WorkflowExecutor(
                self.history_snapshot,
                ArtifactStore(self.artifact_directory),
                default_registry(),
            )
            node = self.history_snapshot.get(self.node_id)
            if node.operation_id == "source.scan":
                context = {
                    "node": node,
                    "scan": executor.load_output(node, "scan"),
                    "spikes": None,
                    "peaks": None,
                    "fit": None,
                }
            elif node.operation_id == "preprocess.pool":
                # Pool output is already a complete ScanArtifact; don't read the raw
                # source volume a second time just to preview the pooled result.
                context = {
                    "node": node,
                    "scan": executor.load_output(node, "scan"),
                    "spikes": None,
                    "peaks": None,
                    "fit": None,
                }
            else:
                inputs = executor.resolve_node_inputs(node)
                stage_output = executor.load_output(node, next(iter(node.output_refs)))
                context = {
                    "node": node,
                    "scan": inputs.get("scan"),
                    "spikes": stage_output if node.operation_id == "preprocess.spikes" else inputs.get("spikes"),
                    "peaks": stage_output if node.operation_id == "peaks.detect" else inputs.get("peaks"),
                    "fit": stage_output if node.operation_id == "fit.scan" else None,
                }
            if context["scan"] is None:
                raise RuntimeError("Для предпросмотра не удалось разрешить ScanArtifact")
            self.signals.succeeded.emit(self.node_id, context)
        except Exception:
            self.signals.failed.emit(self.node_id, traceback.format_exc())


class ProcessingWorkbenchWidget(QWidget):
    """The workbench itself; host applications can embed it in a dock/dialog.

    Signals:
      preview_requested(request): temporarily show selected node in host views.
      preview_restore_requested(): restore the host views as they were before preview.
    """

    preview_requested = pyqtSignal(object)
    preview_restore_requested = pyqtSignal()

    def __init__(self, project: WorkflowProject, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.registry = default_registry()
        self.thread_pool = QThreadPool.globalInstance()
        self._active_task: _RunNodeTask | None = None
        self._active_preview_task: _PreviewNodeTask | None = None
        self._preview_context: dict[str, Any] | None = None
        self._map_click_connection = None
        self._build_ui()
        self._rebuild_tree()
        if self.project.history.nodes:
            self.tree.setCurrentItem(self._items_by_id[self.project.history.nodes[-1].id])
        self._update_controls()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(7)

        header = QHBoxLayout()
        source = self.project.source_metadata or {}
        self.source_label = QLabel(
            f"<b>Скан:</b> {source.get('scan_key', 'не выбран')} "
            f"<span style='color:gray'>· {Path(source.get('source_file', '')).name}</span>"
        )
        self.source_label.setTextFormat(Qt.TextFormat.RichText)
        header.addWidget(self.source_label, 1)
        self.save_button = QPushButton("Сохранить историю")
        self.save_button.clicked.connect(self._save_project)
        header.addWidget(self.save_button)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.addWidget(QLabel("<b>Дерево обработки</b>"))
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(270)
        self.tree.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.tree, 1)
        self.branch_hint = QLabel(
            "Выберите узел, от которого хотите продолжить ветвь."
        )
        self.branch_hint.setWordWrap(True)
        self.branch_hint.setStyleSheet("color: palette(placeholderText);")
        left_layout.addWidget(self.branch_hint)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs, 1)

        # Operation creation tab.
        operation_page = QWidget()
        operation_layout = QVBoxLayout(operation_page)
        operation_layout.setContentsMargins(8, 8, 8, 8)
        selection_row = QHBoxLayout()
        selection_row.addWidget(QLabel("Новый этап:"))
        self.operation_combo = QComboBox()
        self._operation_ids = [
            "preprocess.pool", "preprocess.spikes", "peaks.detect", "fit.scan"
        ]
        for operation_id in self._operation_ids:
            self.operation_combo.addItem(self.registry.get(operation_id).label, operation_id)
        self.operation_combo.currentIndexChanged.connect(self._show_operation_form)
        selection_row.addWidget(self.operation_combo, 1)
        operation_layout.addLayout(selection_row)
        self.operation_dependency_hint = QLabel()
        self.operation_dependency_hint.setWordWrap(True)
        self.operation_dependency_hint.setStyleSheet("color: palette(placeholderText);")
        operation_layout.addWidget(self.operation_dependency_hint)
        self._unavailable_operations: dict[str, str] = {}
        self._refresh_operation_availability()

        self.form_stack = QStackedWidget()
        self._forms: dict[str, dict[str, Any]] = {}
        self._build_pool_form()
        self._build_spike_form()
        self._build_peak_form()
        self._build_fit_form()
        operation_layout.addWidget(self.form_stack)

        actions = QHBoxLayout()
        self.add_button = QPushButton("Добавить этап")
        self.add_button.clicked.connect(self._add_stage)
        self.run_button = QPushButton("Запустить выбранный")
        self.run_button.clicked.connect(self._run_selected)
        self.run_button.setDefault(True)
        actions.addWidget(self.add_button)
        actions.addWidget(self.run_button)
        operation_layout.addLayout(actions)

        self.node_details = QPlainTextEdit()
        self.node_details.setReadOnly(True)
        self.node_details.setPlaceholderText("Параметры и происхождение выбранного узла")
        self.node_details.setMaximumBlockCount(3000)
        operation_layout.addWidget(self.node_details, 1)
        self.tabs.addTab(operation_page, "Операции и параметры")

        # Preview tab.
        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        preview_toolbar = QHBoxLayout()
        self.preview_button = QPushButton("Показать результат")
        self.preview_button.clicked.connect(self._preview_selected)
        self.restore_button = QPushButton("Вернуть исходный вид")
        self.restore_button.clicked.connect(self._restore_preview)
        preview_toolbar.addWidget(self.preview_button)
        preview_toolbar.addWidget(self.restore_button)
        preview_toolbar.addStretch(1)
        self.pixel_label = QLabel("Пиксель: —")
        preview_toolbar.addWidget(self.pixel_label)
        preview_layout.addLayout(preview_toolbar)

        self.preview_views = QTabWidget()
        map_curve_page = QWidget()
        map_curve_layout = QVBoxLayout(map_curve_page)
        map_curve_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax_map = self.figure.add_subplot(1, 2, 1)
        self.ax_curve = self.figure.add_subplot(1, 2, 2)
        self._map_click_connection = self.canvas.mpl_connect("button_press_event", self._on_map_click)
        map_curve_layout.addWidget(self.canvas, 1)
        self.preview_views.addTab(map_curve_page, "Карта и ROC")

        diagnostics_page = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_page)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        self.diagnostic_figure = Figure(figsize=(7, 4), tight_layout=True)
        self.diagnostic_canvas = FigureCanvasQTAgg(self.diagnostic_figure)
        diagnostics_layout.addWidget(self.diagnostic_canvas, 1)
        self.preview_views.addTab(diagnostics_page, "Диагностика этапа")
        preview_layout.addWidget(self.preview_views, 1)
        self.preview_summary = QLabel("Выберите завершённый узел и нажмите «Показать результат».")
        self.preview_summary.setWordWrap(True)
        preview_layout.addWidget(self.preview_summary)
        self.tabs.addTab(preview_page, "Просмотр результата")

        # Execution log / status tab.
        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        self.status_label = QLabel("Готово к работе")
        log_layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        log_layout.addWidget(self.progress)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text, 1)
        self.tabs.addTab(log_page, "Выполнение")

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([310, 850])

        self._show_operation_form()

    def _add_form_page(self, operation_id: str, title: str) -> QFormLayout:
        page = QWidget()
        layout = QFormLayout(page)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.setVerticalSpacing(9)
        self.form_stack.addWidget(page)
        self._forms[operation_id] = {"page": page, "layout": layout}
        return layout

    @staticmethod
    def _int_spin(minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSingleStep(step)
        return widget

    @staticmethod
    def _float_spin(minimum: float, maximum: float, value: float, decimals: int = 4) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        widget.setSingleStep(0.01)
        return widget

    def _build_pool_form(self) -> None:
        layout = self._add_form_page("preprocess.pool", "Pooling")
        size = self._int_spin(1, 101, 5, 2)
        stride = self._int_spin(1, 101, 1, 1)
        layout.addRow("Размер окна (px):", size)
        layout.addRow("Stride (px):", stride)
        hint = QLabel(
            "Count-sum по X/Y; θ не агрегируется. Размер окна — нечётный: 1, 3, 5…"
        )
        hint.setWordWrap(True)
        layout.addRow("", hint)
        self._forms["preprocess.pool"].update(size=size, stride=stride)

    def _build_spike_form(self) -> None:
        layout = self._add_form_page("preprocess.spikes", "Spikes")
        max_iter = self._int_spin(1, 20, 5)
        tolerance = self._float_spin(0.000001, 0.1, 0.001, 6)
        collect = QCheckBox("Сохранять диагностику κ-fit")
        collect.setChecked(True)
        layout.addRow("Максимум итераций:", max_iter)
        layout.addRow("Критерий сходимости:", tolerance)
        layout.addRow("", collect)
        hint = QLabel(
            "Используется существующий refine_spike_mask. "
            "Маска и интерполированное представление сохраняются раздельно."
        )
        hint.setWordWrap(True)
        layout.addRow("", hint)
        self._forms["preprocess.spikes"].update(
            max_iter=max_iter, tol=tolerance, collect=collect
        )

    def _build_peak_form(self) -> None:
        layout = self._add_form_page("peaks.detect", "Peak detection")
        window = self._int_spin(1, 15, 3, 2)
        smooth = self._int_spin(3, 101, 11, 2)
        poly = self._int_spin(0, 10, 2)
        distance = self._int_spin(1, 1000, 6)
        mode = QComboBox()
        mode.addItem("Порог в σ шума", "sigma")
        mode.addItem("Доля размаха ROC", "fraction")
        prominence = self._float_spin(0.0001, 1000.0, 3.0, 4)
        relative = self._float_spin(0.0001, 1.0, 0.07, 4)
        interpolated = QCheckBox("Использовать интерполированные данные после spike detection")
        interpolated.setChecked(False)
        interpolated.setToolTip(
            "По умолчанию детектор работает с исходными counts. "
            "Включайте, если выше по выбранной ветви уже выполнен Spike detection."
        )
        layout.addRow("Окрестность каталога:", window)
        layout.addRow("Окно Savitzky–Golay:", smooth)
        layout.addRow("Степень полинома:", poly)
        layout.addRow("Минимальная дистанция:", distance)
        layout.addRow("Порог prominence:", mode)
        layout.addRow("Значение prominence:", prominence)
        layout.addRow("Относительный prominence:", relative)
        layout.addRow("", interpolated)
        self._forms["peaks.detect"].update(
            window_size=window, smooth_window=smooth, smooth_polyorder=poly,
            min_distance=distance, prominence_mode=mode, prominence_value=prominence,
            min_relative_prominence=relative, use_interpolated_data=interpolated,
        )

    def _build_fit_form(self) -> None:
        layout = self._add_form_page("fit.scan", "Fit")
        model = QComboBox()
        model.addItems(["pvoigt", "voigt", "gauss", "lorentz", "emg", "split_voigt"])
        backend = QComboBox()
        backend.addItem("LMFit (reference)", "lmfit")
        backend.addItem("JAX", "jax")
        backend_availability = fit_backend_availability()
        self._fit_has_lmfit = bool(backend_availability["lmfit"])
        self._fit_has_jax = bool(backend_availability["jax"])
        self._fit_model_has_jax = dict(backend_availability["jax_models"])
        model_items = model.model()
        for index in range(model.count()):
            model_name = model.itemText(index)
            supports_jax = bool(self._fit_model_has_jax.get(model_name, False))
            item = model_items.item(index) if hasattr(model_items, "item") else None
            if item is not None:
                item.setEnabled(self._fit_has_lmfit or (self._fit_has_jax and supports_jax))
        backend_items = backend.model()
        lmfit_item = backend_items.item(backend.findData("lmfit"))
        jax_item = backend_items.item(backend.findData("jax"))
        if lmfit_item is not None:
            lmfit_item.setEnabled(self._fit_has_lmfit)
        if jax_item is not None:
            jax_item.setEnabled(self._fit_has_jax)
        if not self._fit_has_lmfit and self._fit_has_jax and self._fit_model_has_jax.get("pvoigt", False):
            backend.setCurrentIndex(backend.findData("jax"))
        elif self._fit_has_lmfit:
            backend.setCurrentIndex(backend.findData("lmfit"))
        else:
            backend.setCurrentIndex(-1)
        center = self._float_spin(0.01, 10000.0, 40.0, 2)
        batch = self._int_spin(1, 4096, 128)
        max_iter = self._int_spin(10, 10000, 200)
        background = QCheckBox("Добавлять фоновую компоненту")
        background.setChecked(True)
        use_spikes = QCheckBox("Исключать точки из upstream spike mask")
        use_spikes.setChecked(True)
        use_spikes.setToolTip(
            "Если в выбранной ветви есть успешный этап Spike detection, "
            "исключает помеченные измерения из fit через valid_mask. "
            "Исходные counts при этом не интерполируются и не изменяются."
        )
        shift = QCheckBox("Разрешить смещение центра")
        shift.setChecked(True)
        layout.addRow("Модель:", model)
        layout.addRow("Backend:", backend)
        layout.addRow("Окно центра θ:", center)
        layout.addRow("Batch size (JAX):", batch)
        layout.addRow("Максимум итераций (JAX):", max_iter)
        layout.addRow("", background)
        layout.addRow("", use_spikes)
        layout.addRow("", shift)
        self._forms["fit.scan"].update(
            model=model, backend=backend, center_window=center, batch_size=batch,
            solver_max_iter=max_iter, add_background=background,
            use_spike_mask=use_spikes, allow_center_shift=shift,
        )
        model.currentTextChanged.connect(self._update_fit_backend_options)
        backend.currentIndexChanged.connect(self._update_fit_backend_controls)
        self._fit_model_combo = model
        self._fit_backend_combo = backend
        self._fit_batch_spin = batch
        self._fit_iter_spin = max_iter
        self._update_fit_backend_options()
        self._update_fit_backend_controls()
        # The operation list was first built before optional fit backends were
        # inspected. Re-evaluate it now that the actual availability is known.
        self._refresh_operation_availability()

    def _update_fit_backend_options(self, *_args) -> None:
        """Expose only installed backends implemented for the selected model."""
        model_name = self._fit_model_combo.currentText()
        supports_jax = self._fit_model_has_jax.get(model_name, False)
        backend_model = self._fit_backend_combo.model()
        lmfit_index = self._fit_backend_combo.findData("lmfit")
        jax_index = self._fit_backend_combo.findData("jax")
        lmfit_item = backend_model.item(lmfit_index) if hasattr(backend_model, "item") else None
        jax_item = backend_model.item(jax_index) if hasattr(backend_model, "item") else None
        if lmfit_item is not None:
            lmfit_item.setEnabled(self._fit_has_lmfit)
            lmfit_item.setToolTip("" if self._fit_has_lmfit else "Пакет lmfit не установлен")
        jax_available = self._fit_has_jax and supports_jax
        if jax_item is not None:
            jax_item.setEnabled(jax_available)
            jax_item.setToolTip(
                "" if jax_available else
                ("Для этой модели JAX-backend не реализован" if not supports_jax else "Пакет jax/jaxlib не установлен")
            )
        current = self._fit_backend_combo.currentData()
        current_available = (
            (current == "lmfit" and self._fit_has_lmfit)
            or (current == "jax" and jax_available)
        )
        if not current_available:
            if self._fit_has_lmfit:
                self._fit_backend_combo.setCurrentIndex(lmfit_index)
            elif jax_available:
                self._fit_backend_combo.setCurrentIndex(jax_index)
            else:
                self._fit_backend_combo.setCurrentIndex(-1)
        self._update_fit_backend_controls()

    def _update_fit_backend_controls(self, *_args) -> None:
        use_jax = self._fit_backend_combo.currentData() == "jax"
        self._fit_batch_spin.setEnabled(use_jax)
        self._fit_iter_spin.setEnabled(use_jax)
        tooltip = "Используется только backend JAX; LMFit применяет свои настройки solver."
        self._fit_batch_spin.setToolTip(tooltip)
        self._fit_iter_spin.setToolTip(tooltip)

    def _refresh_operation_availability(self) -> None:
        """Disable adapters whose canonical scientific dependency is absent."""
        try:
            from roc_mcs.utils import to_yxt as _canonical_to_yxt  # noqa: F401
            peaks_fit_error = None
        except Exception as exc:
            peaks_fit_error = (
                "Детектор пиков и Fit отключены: существующий модуль "
                "roc_mcs.peaks.catalog импортирует roc_mcs.utils.to_yxt, "
                "но этот helper недоступен в текущем исходном дереве. "
                "Восстановите канонический utils.py из рабочей версии проекта; "
                "Workbench не подменяет научное преобразование осей собственной реализацией. "
                f"Причина: {type(exc).__name__}: {exc}"
            )

        self._unavailable_operations = {}
        if peaks_fit_error:
            self._unavailable_operations.update({
                "peaks.detect": peaks_fit_error,
                "fit.scan": peaks_fit_error,
            })
        elif hasattr(self, "_fit_has_lmfit") and not (
            self._fit_has_lmfit or self._fit_has_jax
        ):
            self._unavailable_operations["fit.scan"] = (
                "Fit недоступен: не установлен ни lmfit, ни jax/jaxlib. "
                "Установите хотя бы один поддерживаемый backend."
            )

        model = self.operation_combo.model()
        for index, operation_id in enumerate(self._operation_ids):
            item = model.item(index) if hasattr(model, "item") else None
            if item is None:
                continue
            reason = self._unavailable_operations.get(operation_id)
            item.setEnabled(reason is None)
            item.setToolTip(reason or self.registry.get(operation_id).description)

        if self._unavailable_operations:
            self.operation_dependency_hint.setText(
                "Часть операций недоступна из-за отсутствующей научной зависимости. "
                "Подробность наведением на соответствующий пункт списка."
            )
            self.operation_dependency_hint.show()
        else:
            self.operation_dependency_hint.hide()

    def _show_operation_form(self, *_args) -> None:
        operation_id = self.operation_combo.currentData()
        page = self._forms.get(operation_id)
        if page:
            self.form_stack.setCurrentWidget(page["page"])

    def _form_parameters(self, operation_id: str) -> dict[str, Any]:
        form = self._forms[operation_id]
        if operation_id == "preprocess.pool":
            return {"size": form["size"].value(), "stride": form["stride"].value()}
        if operation_id == "preprocess.spikes":
            return {
                "max_iter": form["max_iter"].value(),
                "tol": form["tol"].value(),
                "collect_kappa_fits": form["collect"].isChecked(),
            }
        if operation_id == "peaks.detect":
            return {
                "window_size": form["window_size"].value(),
                "smooth_window": form["smooth_window"].value(),
                "smooth_polyorder": form["smooth_polyorder"].value(),
                "min_distance": form["min_distance"].value(),
                "prominence_mode": form["prominence_mode"].currentData(),
                "prominence_value": form["prominence_value"].value(),
                "min_relative_prominence": form["min_relative_prominence"].value(),
                "use_interpolated_data": form["use_interpolated_data"].isChecked(),
            }
        if operation_id == "fit.scan":
            return {
                "model": form["model"].currentText(),
                "backend": form["backend"].currentData(),
                "center_window": form["center_window"].value(),
                "batch_size": form["batch_size"].value(),
                "solver_max_iter": form["solver_max_iter"].value(),
                "add_background": form["add_background"].isChecked(),
                "use_spike_mask": form["use_spike_mask"].isChecked(),
                "allow_center_shift": form["allow_center_shift"].isChecked(),
            }
        raise KeyError(operation_id)

    def _selected_node_id(self) -> str | None:
        item = self.tree.currentItem()
        return None if item is None else item.data(0, Qt.ItemDataRole.UserRole)

    def _add_stage(self) -> None:
        if self._active_task is not None or self._active_preview_task is not None:
            QMessageBox.information(self, "Выполнение", "Дождитесь завершения текущей операции.")
            return
        selected_id = self._selected_node_id()
        if not selected_id:
            return
        selected_node = self.project.history.get(selected_id)
        if selected_node.status != NodeStatus.SUCCEEDED:
            QMessageBox.information(
                self, "Выбор родителя", "Новый этап можно создавать только от успешно завершённого узла."
            )
            return
        operation_id = self.operation_combo.currentData()
        try:
            unavailable_reason = self._unavailable_operations.get(operation_id)
            if unavailable_reason:
                raise RuntimeError(unavailable_reason)
            parameters = self.registry.get(operation_id).validate(
                self._form_parameters(operation_id)
            )
            input_refs = build_input_refs(
                self.project.history, selected_id, operation_id, parameters
            )
            parent_ids = [selected_id]
            label = make_operation_label(operation_id, parameters)
            node = self.project.history.add_node(
                operation_id,
                label,
                parent_ids=parent_ids,
                input_refs=input_refs,
                parameters=parameters,
                metadata={"created_in": "processing_workbench"},
                software_version=application_version(),
                parameter_schema_version=self.registry.get(operation_id).parameter_schema_version,
            )
            self.project.save()
            self._rebuild_tree(select_id=node.id)
            self._log(f"Создан узел {node.label} · {node.id[:8]}")
            self._update_controls()
        except Exception as exc:
            QMessageBox.warning(self, "Нельзя добавить этап", str(exc))

    def _run_selected(self) -> None:
        if self._active_task is not None or self._active_preview_task is not None:
            return
        node_id = self._selected_node_id()
        if not node_id:
            return
        node = self.project.history.get(node_id)
        if node.operation_id == "source.scan":
            return
        if node.status != NodeStatus.PENDING:
            QMessageBox.information(
                self, "Запуск", "Узел уже запускался или находится в терминальном состоянии. "
                "Для новых параметров создайте новую ветвь от входного узла."
            )
            return
        try:
            self.project.history.transition(node_id, NodeStatus.RUNNING)
            self.project.save()
            snapshot = WorkflowHistory(list(self.project.history.nodes))
            task = _RunNodeTask(
                node_id,
                snapshot,
                str(self.project.artifact_store.directory),
            )
            task.signals.succeeded.connect(self._on_run_succeeded)
            task.signals.failed.connect(self._on_run_failed)
            self._active_task = task
            self.progress.setRange(0, 0)
            self.status_label.setText(f"Выполняется: {node.label}")
            self.tabs.setCurrentIndex(2)
            self._rebuild_tree(select_id=node_id)
            self.thread_pool.start(task)
        except Exception as exc:
            if self.project.history.get(node_id).status == NodeStatus.RUNNING:
                self.project.history.transition(node_id, NodeStatus.FAILED, error=str(exc))
                self.project.save()
            self._active_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            QMessageBox.critical(self, "Ошибка запуска", str(exc))
            self._rebuild_tree(select_id=node_id)

    def _on_run_succeeded(self, node_id: str, outcome: Any) -> None:
        persistence_error = None
        try:
            node = self.project.history.get(node_id)
            output_refs = {port: outcome.artifact_id for port in outcome.output_ports}
            self.project.history.transition(
                node_id, NodeStatus.SUCCEEDED,
                output_refs=output_refs,
                summary=outcome.summary,
            )
            self.project.save()
            self._log(f"✓ {node.label}: {outcome.summary}")
            self.status_label.setText("Этап успешно завершён")
        except Exception:
            persistence_error = traceback.format_exc()
            self._log(persistence_error)
            self.status_label.setText("Результат вычислен, но историю не удалось сохранить")
        finally:
            self._active_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if persistence_error is None else 0)
            self._rebuild_tree(select_id=node_id)
            self._update_controls()
        if persistence_error is not None:
            QMessageBox.critical(
                self, "Не удалось сохранить проект",
                "Артефакт вычислен, но запись workflow.json завершилась ошибкой. "
                "Не закрывайте приложение до успешного сохранения.\n\n" + persistence_error,
            )

    def _on_run_failed(self, node_id: str, error: str) -> None:
        persistence_error = None
        try:
            self.project.history.transition(node_id, NodeStatus.FAILED, error=error)
            self.project.save()
        except Exception:
            persistence_error = traceback.format_exc()
            self._log(persistence_error)
        finally:
            self._active_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("Этап завершился с ошибкой")
            self._log(error)
            self._rebuild_tree(select_id=node_id)
            self._update_controls()
        if persistence_error is not None:
            QMessageBox.critical(self, "Ошибка сохранения истории", persistence_error)

    def _rebuild_tree(self, select_id: str | None = None) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        self._items_by_id: dict[str, QTreeWidgetItem] = {}
        nodes = list(self.project.history.nodes)
        for node in nodes:
            text = f"{_STATUS_MARK[node.status]}  {node.label}  ·  {node.status}"
            item = QTreeWidgetItem([text])
            item.setData(0, Qt.ItemDataRole.UserRole, node.id)
            item.setToolTip(
                0,
                f"ID: {node.id}\nОперация: {node.operation_id}\n"
                f"Входы: {json.dumps(node.input_refs, ensure_ascii=False)}\n"
                f"Родители: {', '.join(node.parent_ids) or '—'}",
            )
            self._items_by_id[node.id] = item
            parent_id = node.parent_ids[0] if node.parent_ids else None
            if parent_id and parent_id in self._items_by_id:
                self._items_by_id[parent_id].addChild(item)
            else:
                self.tree.addTopLevelItem(item)
        for item in self._items_by_id.values():
            item.setExpanded(True)
        if select_id and select_id in self._items_by_id:
            self.tree.setCurrentItem(self._items_by_id[select_id])
        self.tree.blockSignals(False)
        self._on_selection_changed(self.tree.currentItem(), None)

    def _on_selection_changed(self, current: QTreeWidgetItem | None, _previous) -> None:
        if current is None:
            self.node_details.clear()
            self._update_controls()
            return
        node_id = current.data(0, Qt.ItemDataRole.UserRole)
        node = self.project.history.get(node_id)
        detail = {
            "operation": node.operation_id,
            "label": node.label,
            "status": node.status,
            "parameters": node.parameters,
            "inputs": node.input_refs,
            "outputs": node.output_refs,
            "metadata": node.metadata,
            "summary": node.summary,
            "error": node.error,
            "created_at": node.created_at,
            "started_at": node.started_at,
            "finished_at": node.finished_at,
        }
        self.node_details.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2, default=str))
        self._update_controls()

    def _update_controls(self) -> None:
        busy = self._active_task is not None or self._active_preview_task is not None
        node_id = self._selected_node_id()
        runnable = False
        previewable = False
        addable = False
        if node_id:
            node = self.project.history.get(node_id)
            addable = node.status == NodeStatus.SUCCEEDED
            runnable = node.operation_id != "source.scan" and node.status == NodeStatus.PENDING
            previewable = node.status == NodeStatus.SUCCEEDED
        self.add_button.setEnabled(not busy and addable)
        self.run_button.setEnabled(not busy and runnable)
        self.preview_button.setEnabled(not busy and previewable)
        self.restore_button.setEnabled(not busy and self._preview_context is not None)

    def _save_project(self) -> None:
        try:
            self.project.save()
            self.status_label.setText(f"История сохранена: {self.project.manifest_path}")
            self._log(f"История сохранена: {self.project.manifest_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def _log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    def _preview_selected(self) -> None:
        if self._active_task is not None or self._active_preview_task is not None:
            return
        node_id = self._selected_node_id()
        if not node_id:
            return
        node = self.project.history.get(node_id)
        if node.status != NodeStatus.SUCCEEDED:
            QMessageBox.information(self, "Предпросмотр", "Сначала успешно выполните выбранный этап.")
            return
        try:
            task = _PreviewNodeTask(
                node_id,
                WorkflowHistory(list(self.project.history.nodes)),
                str(self.project.artifact_store.directory),
            )
            task.signals.succeeded.connect(self._on_preview_ready)
            task.signals.failed.connect(self._on_preview_failed)
            self._active_preview_task = task
            self.status_label.setText(f"Загрузка результата: {node.label}")
            self.progress.setRange(0, 0)
            self.tabs.setCurrentIndex(1)
            self._update_controls()
            self.thread_pool.start(task)
        except Exception as exc:
            self._active_preview_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            QMessageBox.warning(self, "Предпросмотр недоступен", str(exc))
            self._update_controls()

    def _on_preview_ready(self, node_id: str, context: dict[str, Any]) -> None:
        previous_context = self._preview_context
        try:
            node = self.project.history.get(node_id)
            self._preview_context = context
            self._draw_preview()
            source = self.project.source_metadata or {}
            scan_bundle = context["scan"]
            request = ArtifactPreviewRequest(
                project_directory=str(self.project.directory),
                node_id=node.id,
                operation_id=node.operation_id,
                source_file=str(source.get("source_file", "")),
                scan_key=str(source.get("scan_key", "")),
                grid_id=str(scan_bundle.metadata.get("grid_id", "unknown")),
                input_refs=node.input_refs,
                output_refs=node.output_refs,
                preferred_view=self._preferred_view(node.operation_id),
                summary=node.summary,
                metadata=scan_bundle.metadata,
            )
            self.preview_requested.emit(request)
            self.preview_summary.setText(
                f"Предпросмотр: {node.label}. Внутренняя карта и ROC-кривая обновлены; "
                "основные окна приложения могут подключаться через preview_requested."
            )
            self.status_label.setText(f"Предпросмотр загружен: {node.label}")
            self._log(f"Предпросмотр загружен: {node.label}")
        except Exception:
            error_text = traceback.format_exc()
            # A failed preview must not replace a previously visible preview.
            # Restore the last successfully displayed local context atomically.
            self._preview_context = previous_context
            if previous_context is not None:
                try:
                    self._draw_preview()
                except Exception:
                    self._log("Не удалось восстановить предыдущий предпросмотр:\n" + traceback.format_exc())
            else:
                self.figure.clear()
                empty_map = self.figure.add_subplot(1, 2, 1)
                empty_curve = self.figure.add_subplot(1, 2, 2)
                empty_map.text(0.5, 0.5, "Предпросмотр недоступен", ha="center", va="center")
                empty_map.set_axis_off()
                empty_curve.set_axis_off()
                self.canvas.draw_idle()
            self._log(error_text)
            QMessageBox.warning(self, "Ошибка отображения", error_text)
        finally:
            self._active_preview_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self._update_controls()

    def _on_preview_failed(self, node_id: str, error: str) -> None:
        self._active_preview_task = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("Не удалось загрузить предпросмотр")
        self._log(error)
        QMessageBox.warning(self, "Предпросмотр недоступен", error)
        self._update_controls()

    @staticmethod
    def _preferred_view(operation_id: str) -> str:
        return {
            "source.scan": "intensity_map",
            "preprocess.pool": "intensity_map",
            "preprocess.spikes": "spike_count_map",
            "peaks.detect": "peak_count_map",
            "fit.scan": "fit_quality_map",
        }.get(operation_id, "intensity_map")

    def _draw_preview(self) -> None:
        ctx = self._preview_context
        if not ctx:
            return
        scan = ctx["scan"]
        data = np.asarray(scan.arrays["data_yxt"], dtype=float)
        theta = np.asarray(scan.arrays["theta"], dtype=float)
        node = ctx["node"]
        spikes = ctx["spikes"]
        peaks = ctx["peaks"]
        fit = ctx["fit"]

        if node.operation_id == "preprocess.spikes" and spikes is not None:
            map_data = np.asarray(spikes.arrays["spike_count_map"])
            map_title = "Число spike-точек"
        elif node.operation_id == "peaks.detect" and peaks is not None:
            map_data = np.asarray(peaks.arrays["peak_count_map"])
            map_title = "Число найденных пиков"
        elif node.operation_id == "fit.scan" and fit is not None and "n_fitted_map" in fit.arrays:
            map_data = np.asarray(fit.arrays["n_fitted_map"])
            map_title = "Число fitted-компонент"
        else:
            map_data = np.nansum(data, axis=-1)
            map_title = "Суммарная интенсивность"

        ny, nx = map_data.shape
        x = nx // 2
        y = ny // 2
        self._draw_map_and_curve(map_data, map_title, data, theta, x, y)
        self._draw_diagnostics()

    def _draw_diagnostics(self) -> None:
        """Draw operation-specific QC without re-running scientific algorithms."""
        self.diagnostic_figure.clear()
        ax = self.diagnostic_figure.add_subplot(1, 1, 1)
        ctx = self._preview_context or {}
        node = ctx.get("node")
        spikes = ctx.get("spikes")
        peaks = ctx.get("peaks")
        fit = ctx.get("fit")

        if node is None:
            ax.text(0.5, 0.5, "Нет выбранного результата", ha="center", va="center")
            ax.set_axis_off()
        elif node.operation_id == "preprocess.spikes" and spikes is not None:
            table = spikes.tables.get("kappa_thresholds")
            if table is None or table.empty:
                ax.text(0.5, 0.5, "Диагностика κ-fit не сохранена", ha="center", va="center")
                ax.set_axis_off()
            else:
                for iteration, group in table.groupby("iteration", sort=True):
                    ax.plot(
                        group["tested_kappa"].to_numpy(float),
                        group["n_spikes"].to_numpy(float),
                        marker="o", markersize=3, linewidth=1.1,
                        label=f"Итерация {int(iteration) + 1}",
                    )
                selected_kappa = float(spikes.metadata.get("kappa", np.nan))
                if np.isfinite(selected_kappa):
                    ax.axvline(selected_kappa, linestyle="--", linewidth=1.0, label=f"κ = {selected_kappa:.3g}")
                ax.set_title("Автоматический выбор порога κ")
                ax.set_xlabel("κ")
                ax.set_ylabel("Число обнаруженных точек")
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, alpha=0.2)
        elif node.operation_id == "peaks.detect" and peaks is not None:
            catalog = peaks.tables.get("peak_catalog")
            if catalog is None or catalog.empty or "prominence" not in catalog:
                ax.text(0.5, 0.5, "Пики не обнаружены", ha="center", va="center")
                ax.set_axis_off()
            else:
                values = catalog["prominence"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    ax.hist(values, bins=min(40, max(5, int(np.sqrt(values.size)))), edgecolor="black", linewidth=0.4)
                    ax.set_title(f"Распределение prominence · {values.size} пиков")
                    ax.set_xlabel("Prominence")
                    ax.set_ylabel("Число пиков")
                else:
                    ax.text(0.5, 0.5, "Нет конечных prominence", ha="center", va="center")
                    ax.set_axis_off()
        elif node.operation_id == "fit.scan" and fit is not None:
            table = fit.tables.get("pixel_fit_catalog")
            if table is None or table.empty or "fit_status" not in table:
                ax.text(0.5, 0.5, "Нет таблицы статусов fit", ha="center", va="center")
                ax.set_axis_off()
            else:
                counts = table["fit_status"].fillna("не задан").astype(str).value_counts()
                ax.bar(counts.index.astype(str), counts.to_numpy())
                ax.set_title("Статусы подгонки по пикселям")
                ax.set_xlabel("Статус")
                ax.set_ylabel("Число пикселей")
                ax.tick_params(axis="x", labelrotation=20)
        elif node.operation_id == "preprocess.pool":
            scan = ctx.get("scan")
            pool = scan.metadata.get("pool", {}) if scan is not None else {}
            shape = np.asarray(scan.arrays["data_yxt"]).shape if scan is not None else ()
            ax.text(
                0.05, 0.95,
                "Пространственный pooling\n\n"
                f"Размер окна: {pool.get('size', '—')} × {pool.get('size', '—')}\n"
                f"Stride: {pool.get('stride', '—')}\n"
                f"Выходная форма Y×X×θ: {shape}\n"
                f"Grid ID: {scan.metadata.get('grid_id', '—') if scan is not None else '—'}",
                ha="left", va="top", transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "Для этого этапа отдельная QC-диагностика не предусмотрена", ha="center", va="center")
            ax.set_axis_off()

        self.diagnostic_figure.tight_layout()
        self.diagnostic_canvas.draw_idle()

    def _draw_map_and_curve(
        self, map_data: np.ndarray, map_title: str, data_yxt: np.ndarray,
        theta: np.ndarray, x: int, y: int,
    ) -> None:
        self.figure.clear()
        self.ax_map = self.figure.add_subplot(1, 2, 1)
        self.ax_curve = self.figure.add_subplot(1, 2, 2)
        ctx = self._preview_context or {}
        spikes = ctx.get("spikes")
        peaks = ctx.get("peaks")
        fit = ctx.get("fit")

        overlay_titles = {
            "Число spike-точек", "Число найденных пиков", "Число fitted-компонент"
        }
        if map_title in overlay_titles:
            intensity = np.nansum(np.asarray(data_yxt, dtype=float), axis=-1)
            finite = intensity[np.isfinite(intensity)]
            if finite.size:
                low, high = np.nanpercentile(finite, [5, 95])
                if not np.isfinite(high) or high <= low:
                    alpha_map = np.full(intensity.shape, 0.6, dtype=float)
                else:
                    alpha_map = np.clip((intensity - low) / (high - low), 0.0, 1.0)
                    alpha_map = 0.20 + 0.70 * alpha_map
                    alpha_map = np.nan_to_num(alpha_map, nan=0.20, posinf=0.90, neginf=0.20)
                self.ax_map.imshow(
                    intensity, origin="lower", interpolation="nearest", aspect="auto",
                    cmap="gray", alpha=0.78,
                )
                overlay_values = np.asarray(map_data, dtype=float)
                overlay = np.ma.masked_where(~np.isfinite(overlay_values) | (overlay_values <= 0), overlay_values)
                from matplotlib import colormaps
                overlay_cmap = colormaps["magma" if map_title == "Число spike-точек" else "viridis"].copy()
                overlay_cmap.set_bad(alpha=0.0)
                image = self.ax_map.imshow(
                    overlay, origin="lower", interpolation="nearest", aspect="auto",
                    cmap=overlay_cmap, alpha=alpha_map,
                    vmin=0, vmax=max(1.0, float(np.nanmax(map_data)) if np.size(map_data) else 1.0),
                )
                self.figure.colorbar(image, ax=self.ax_map, fraction=0.046, pad=0.04)
            else:
                self.ax_map.imshow(map_data, origin="lower", interpolation="nearest", aspect="auto")
        else:
            self.ax_map.imshow(map_data, origin="lower", interpolation="nearest", aspect="auto")
        self.ax_map.set_title(map_title)
        self.ax_map.set_xlabel("X index")
        self.ax_map.set_ylabel("Y index")
        self.ax_map.plot([x], [y], marker="+", markersize=10, markeredgewidth=1.5)

        curve = np.asarray(data_yxt[y, x, :], dtype=float)
        self.ax_curve.plot(theta, curve, linewidth=1.1, alpha=0.8, label="Исходная ROC")

        # Show the exact interpolated trace as a separate representation. The raw
        # counts remain visible and are never silently replaced in the preview.
        if spikes is not None and "clean_data_yxt" in spikes.arrays:
            clean = np.asarray(spikes.arrays["clean_data_yxt"], dtype=float)
            if clean.shape == data_yxt.shape:
                self.ax_curve.plot(
                    theta, clean[y, x, :], linewidth=1.35,
                    label="После интерполяции",
                )
        if spikes is not None and "spike_mask_tyx" in spikes.arrays:
            mask = np.asarray(spikes.arrays["spike_mask_tyx"], dtype=bool)[:, y, x]
            if mask.shape == theta.shape and np.any(mask):
                self.ax_curve.scatter(
                    theta[mask], curve[mask], marker="x", s=32,
                    label="Обнаруженные спайки", zorder=5,
                )
        if peaks is not None and "peak_catalog" in peaks.tables:
            catalog = peaks.tables["peak_catalog"]
            # Older v1 artifacts may contain an empty DataFrame without column
            # names. New adapters always preserve the schema, but preview must
            # also remain safe when opening those already-saved legacy results.
            if catalog is not None and not catalog.empty and {"x", "y", "theta"}.issubset(catalog.columns):
                selected = catalog[(catalog["x"] == x) & (catalog["y"] == y)]
                if not selected.empty:
                    peak_theta = selected["theta"].to_numpy(float)
                    marker_curve = curve
                    if bool(peaks.metadata.get("used_interpolated_data", False)) and spikes is not None:
                        clean = np.asarray(spikes.arrays.get("clean_data_yxt", data_yxt), dtype=float)
                        if clean.shape == data_yxt.shape:
                            marker_curve = clean[y, x, :]
                    self.ax_curve.scatter(
                        peak_theta,
                        np.interp(peak_theta, theta, marker_curve),
                        marker="v", s=38, label="Найденные пики", zorder=6,
                    )
        if fit is not None and "fit_cube" in fit.arrays:
            fitted = np.asarray(fit.arrays["fit_cube"])[y, x, :]
            if fitted.shape == theta.shape:
                self.ax_curve.plot(theta, fitted, linewidth=1.35, label="Fit")
        self.ax_curve.set_title(f"ROC-кривая · x={x}, y={y}")
        self.ax_curve.set_xlabel("θ")
        self.ax_curve.set_ylabel("Интенсивность")
        self.ax_curve.legend(loc="best", fontsize=8)
        self.pixel_label.setText(f"Пиксель: X={x}, Y={y}")
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._current_map_data = np.asarray(map_data)
        self._current_curve_data = np.asarray(data_yxt)
        self._current_theta = np.asarray(theta)

    def _on_map_click(self, event) -> None:
        if event.inaxes is not self.ax_map or event.xdata is None or event.ydata is None:
            return
        if self._preview_context is None:
            return
        ny, nx = self._current_map_data.shape
        x = int(np.clip(round(event.xdata), 0, nx - 1))
        y = int(np.clip(round(event.ydata), 0, ny - 1))
        map_title = self.ax_map.get_title()
        self._draw_map_and_curve(
            self._current_map_data, map_title,
            self._current_curve_data, self._current_theta, x, y,
        )

    @property
    def is_busy(self) -> bool:
        return self._active_task is not None or self._active_preview_task is not None

    def _restore_preview(self) -> None:
        had_preview = self._preview_context is not None
        self._preview_context = None
        if had_preview:
            self.preview_restore_requested.emit()
        self.figure.clear()
        self.ax_map = self.figure.add_subplot(1, 2, 1)
        self.ax_curve = self.figure.add_subplot(1, 2, 2)
        self.ax_map.text(0.5, 0.5, "Предпросмотр снят", ha="center", va="center")
        self.ax_map.set_axis_off()
        self.ax_curve.set_axis_off()
        self.canvas.draw_idle()
        self.diagnostic_figure.clear()
        diagnostic_ax = self.diagnostic_figure.add_subplot(1, 1, 1)
        diagnostic_ax.text(0.5, 0.5, "Предпросмотр снят", ha="center", va="center")
        diagnostic_ax.set_axis_off()
        self.diagnostic_canvas.draw_idle()
        self.preview_summary.setText("Предпросмотр снят; внешние окна восстановили предыдущее представление.")
        self.pixel_label.setText("Пиксель: —")
        self._update_controls()
