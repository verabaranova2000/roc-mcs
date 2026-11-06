"""Branchable processing workbench, embeddable as a QWidget or standalone window."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFormLayout, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QPushButton, QLabel,
    QPlainTextEdit, QTabWidget, QStackedWidget, QProgressBar, QMessageBox,
    QSizePolicy, QFrame, QMenu, QDialog, QDialogButtonBox, QScrollArea,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from roc_mcs.analysis.artifacts import ArtifactBundle, ArtifactStore
from roc_mcs.analysis.engine import (
    WorkflowExecutor, application_version, build_input_refs, make_operation_label,
)
from roc_mcs.analysis.operations import default_registry, fit_backend_availability
from roc_mcs.analysis.project import WorkflowProject
from roc_mcs.analysis.workflow_history import NodeStatus, ReviewStatus, WorkflowHistory
from .presenter import ArtifactPreviewRequest
from .graph_canvas import ProcessingGraphView


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
            elif node.operation_id in {"preprocess.pool", "preprocess.detector_bad_pixels"}:
                # These operations return a complete ScanArtifact. Preview their
                # exact output, not the unmodified upstream input scan.
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
        self._branch_parent_id: str | None = None
        self._build_ui()
        self._rebuild_graph()
        if self.project.history.nodes:
            self.graph.select_node(self.project.history.nodes[-1].id)
            self._on_graph_node_selected(self.project.history.nodes[-1].id)
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
        graph_header = QHBoxLayout()
        graph_header.addWidget(QLabel("<b>Пайплайн обработки</b>"), 1)
        self.show_rejected_check = QCheckBox("Отклонённые")
        self.show_rejected_check.setToolTip(
            "Показывать отклонённые результаты в архивном режиме. "
            "Они остаются в истории, но недоступны как входы новых этапов."
        )
        self.show_rejected_check.toggled.connect(self._rebuild_graph)
        graph_header.addWidget(self.show_rejected_check)
        self.fit_graph_button = QPushButton("Вписать")
        self.fit_graph_button.setToolTip("Показать всю схему пайплайна")
        self.fit_graph_button.clicked.connect(lambda: self.graph.fit_graph())
        graph_header.addWidget(self.fit_graph_button)
        left_layout.addLayout(graph_header)

        self.graph = ProcessingGraphView(left)
        self.graph.node_selected.connect(self._on_graph_node_selected)
        self.graph.add_stage_requested.connect(self._on_graph_add_stage_requested)
        self.graph.preview_requested.connect(self._on_graph_preview_requested)
        left_layout.addWidget(self.graph, 1)
        self.branch_hint = QLabel(
            "Щелчок по карточке — инспектор · глаз — просмотр результата · "
            "+ — новая ветвь от принятого узла. Перетаскивание — панорама; Ctrl+колесо — масштаб."
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
        # Keep one canonical set of parameter widgets, but do not expose a second
        # creation panel: stages are created contextually from a graph node (+).
        # The same form pages are temporarily reparented into the compact dialog.
        operation_page = QWidget(self)
        self._operation_page = operation_page
        operation_layout = QVBoxLayout(operation_page)
        operation_layout.setContentsMargins(8, 8, 8, 8)
        selection_row = QHBoxLayout()
        selection_row.addWidget(QLabel("Новый этап:"))
        self.operation_combo = QComboBox()
        self._operation_ids = [
            "preprocess.detector_bad_pixels",
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
        self._build_detector_bad_pixel_form()
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

        operation_page.hide()

        # Preview tab. The graph is the single entry point for creating stages.

        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        preview_toolbar = QHBoxLayout()
        self.preview_button = QPushButton("Показать результат")
        self.preview_button.clicked.connect(self._preview_selected)
        self.restore_button = QPushButton("Вернуть исходный вид")
        self.restore_button.clicked.connect(self._restore_preview)
        self.accept_button = QPushButton("✓ Принять результат")
        self.accept_button.clicked.connect(self._accept_selected)
        self.reject_button = QPushButton("Отклонить ветвь")
        self.reject_button.clicked.connect(self._reject_selected)
        self.accept_button.setToolTip(
            "Зафиксировать результат как проверенный и разрешить использовать его дальше."
        )
        self.reject_button.setToolTip(
            "Отклонить результат. Он исчезнет из активного графа, но останется в истории."
        )
        preview_toolbar.addWidget(self.preview_button)
        preview_toolbar.addWidget(self.restore_button)
        preview_toolbar.addWidget(self.accept_button)
        preview_toolbar.addWidget(self.reject_button)
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
        self.tabs.addTab(preview_page, "Результат")

        # Provenance/metadata inspector is kept separate from node creation.
        inspector_page = QWidget()
        inspector_layout = QVBoxLayout(inspector_page)
        inspector_layout.setContentsMargins(8, 8, 8, 8)
        inspector_layout.addWidget(QLabel("<b>Инспектор узла</b>"))
        self.node_details = QPlainTextEdit()
        self.node_details.setReadOnly(True)
        self.node_details.setPlaceholderText(
            "Выберите узел, чтобы увидеть параметры, входы, выходы, metadata и историю запуска."
        )
        self.node_details.setMaximumBlockCount(5000)
        inspector_layout.addWidget(self.node_details, 1)
        self.tabs.addTab(inspector_page, "Инспектор")

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

    def _build_detector_bad_pixel_form(self) -> None:
        layout = self._add_form_page(
            "preprocess.detector_bad_pixels", "Исключение битых пикселей"
        )
        pixels = QLineEdit()
        try:
            from roc_mcs.gui.app_settings import load_detector_bad_pixels
            configured = load_detector_bad_pixels()
        except Exception:
            configured = []
        pixels.setText("; ".join(f"{x},{y}" for x, y in configured))
        pixels.setPlaceholderText("например: 241,26; 12,45")
        pixels.setToolTip(
            "Координаты исходного detector-grid, нумерация с нуля: x,y; x,y. "
            "Список первоначально загружается из настроек приложения."
        )
        layout.addRow("Битые пиксели (x,y):", pixels)
        hint = QLabel(
            "Исходные counts и ROC сохраняются без изменения; заданные пиксели "
            "получают valid=False и отображаются как ‘нет данных’, а не как нулевая линия. "
            "Исключённые значения не участвуют в pooling-вкладе, spike-background, "
            "соседних peak-оценках и Fit. Этап обязателен первым в каждой ветви "
            "и выполняется на исходной detector-grid "
            "до pooling. Если список пуст, это явно фиксирует, что для данного скана "
            "исключения не применяются."
        )
        hint.setWordWrap(True)
        layout.addRow("", hint)
        self._forms["preprocess.detector_bad_pixels"].update(pixels=pixels)

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
        if operation_id == "preprocess.detector_bad_pixels":
            from roc_mcs.gui.app_settings import normalize_detector_bad_pixels
            return {"pixels": normalize_detector_bad_pixels(form["pixels"].text())}
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
        return self.graph.selected_node_id

    def _available_operations_for_parent(self, parent_id: str) -> list[str]:
        """Return operation ids that can consume outputs of the selected node."""
        node = self.project.history.get(parent_id)
        if node.status != NodeStatus.SUCCEEDED or node.review_status != ReviewStatus.ACCEPTED:
            return []
        if node.operation_id == "source.scan":
            # Make the detector-QC decision explicit and auditable before any
            # scientific operation can consume this scan. An empty list is a
            # recorded confirmation that no detector sites are excluded.
            return ["preprocess.detector_bad_pixels"]
        if node.operation_id in {"preprocess.detector_bad_pixels", "preprocess.pool"}:
            return ["preprocess.pool", "preprocess.spikes", "peaks.detect"]
        if node.operation_id == "preprocess.spikes":
            return ["peaks.detect"]
        if node.operation_id == "peaks.detect":
            return ["fit.scan"]
        return []

    def _on_graph_add_stage_requested(self, parent_id: str) -> None:
        if self.is_busy:
            return
        try:
            parent = self.project.history.get(parent_id)
            operation_ids = self._available_operations_for_parent(parent_id)
            operation_ids = [
                operation_id for operation_id in operation_ids
                if operation_id not in self._unavailable_operations
            ]
            if not operation_ids:
                QMessageBox.information(
                    self, "Новая ветвь",
                    "Для выбранного узла нет доступных следующих операций. "
                    "Проверьте статус результата и его выходные данные.",
                )
                return
            menu = QMenu(self)
            menu.setTitle("Добавить этап от выбранного результата")
            for operation_id in operation_ids:
                spec = self.registry.get(operation_id)
                action = menu.addAction(spec.label)
                action.setToolTip(spec.description)
                action.triggered.connect(
                    lambda _checked=False, op=operation_id, pid=parent_id:
                    self._open_stage_configuration(pid, op)
                )
            menu.exec(QCursor.pos())
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось добавить ветвь", str(exc))

    def _open_stage_configuration(self, parent_id: str, operation_id: str) -> None:
        """Show the selected node's existing parameter form in a compact dialog.

        Reparenting the single existing form page avoids maintaining a second
        set of parameter widgets with potentially divergent validation/defaults.
        Closing the dialog does not create a history node; only the explicit
        "create and run" action persists a candidate.
        """
        if self.is_busy:
            return
        parent = self.project.history.get(parent_id)
        operation_ids = self._available_operations_for_parent(parent_id)
        if operation_id not in operation_ids:
            QMessageBox.warning(
                self, "Недоступный этап",
                "Выбранную операцию нельзя создать от этого узла. "
                "Проверьте состояние и принятые результаты ветви.",
            )
            return
        self._branch_parent_id = parent_id
        combo_index = self.operation_combo.findData(operation_id)
        if combo_index < 0:
            return
        self.operation_combo.setCurrentIndex(combo_index)
        form_info = self._forms.get(operation_id)
        if form_info is None:
            return
        page = form_info["page"]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Новый этап · {self.registry.get(operation_id).label}")
        dialog.setModal(True)
        dialog.setMinimumWidth(430)
        dialog.resize(470, 390)

        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(14, 12, 14, 12)
        dialog_layout.setSpacing(9)
        parent_label = QLabel(
            f"<b>Входная ветвь:</b> {parent.label}<br>"
            f"<span style='color:gray'>{self.registry.get(operation_id).description}</span>"
        )
        parent_label.setTextFormat(Qt.TextFormat.RichText)
        parent_label.setWordWrap(True)
        dialog_layout.addWidget(parent_label)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Remove from the hidden right-hand form stack while the dialog owns it.
        self.form_stack.removeWidget(page)
        page.setParent(None)
        scroll.setWidget(page)
        dialog_layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(dialog)
        cancel_button = buttons.addButton(
            "Отмена", QDialogButtonBox.ButtonRole.RejectRole
        )
        run_button = buttons.addButton(
            "Создать и запустить", QDialogButtonBox.ButtonRole.AcceptRole
        )
        run_button.setDefault(True)
        cancel_button.clicked.connect(dialog.reject)

        def _validate_and_accept_dialog() -> None:
            try:
                # Validate before closing. Malformed detector coordinates or any
                # future domain constraint leave the form open for correction.
                self.registry.get(operation_id).validate(
                    self._form_parameters(operation_id)
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Проверьте параметры этапа", str(exc))
                return
            dialog.accept()

        run_button.clicked.connect(_validate_and_accept_dialog)
        dialog_layout.addWidget(buttons)

        accepted = False
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            if accepted:
                parameters = self._form_parameters(operation_id)
                node_id = self._create_stage_candidate(
                    parent_id, operation_id, parameters, run_immediately=True
                )
                if node_id:
                    self.graph.select_node(node_id)
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось создать этап", str(exc))
        finally:
            # Detach the page before the dialog is destroyed; otherwise Qt would
            # delete widgets still referenced by self._forms.
            if scroll.widget() is page:
                scroll.takeWidget()
            page.setParent(self.form_stack)
            self.form_stack.addWidget(page)
            self._show_operation_form()

    def _on_graph_preview_requested(self, node_id: str) -> None:
        self.graph.select_node(node_id)
        self._on_graph_node_selected(node_id)
        self._preview_selected()

    def _add_stage(self) -> None:
        if self.is_busy:
            QMessageBox.information(self, "Выполнение", "Дождитесь завершения текущей операции.")
            return
        selected_id = self._selected_node_id()
        if not selected_id:
            return
        try:
            selected_node = self.project.history.get(selected_id)
            if (
                selected_node.status != NodeStatus.SUCCEEDED
                or selected_node.review_status != ReviewStatus.ACCEPTED
            ):
                raise ValueError(
                    "Новый этап можно создавать только от успешно выполненного "
                    "и принятого результата."
                )
            operation_id = self.operation_combo.currentData()
            unavailable_reason = self._unavailable_operations.get(operation_id)
            if unavailable_reason:
                raise RuntimeError(unavailable_reason)
            self._create_stage_candidate(
                selected_id, operation_id, self._form_parameters(operation_id),
                run_immediately=False,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Нельзя добавить этап", str(exc))

    def _create_stage_candidate(
        self,
        parent_id: str,
        operation_id: str,
        raw_parameters: dict[str, Any],
        *,
        run_immediately: bool,
    ) -> str | None:
        """Validate, persist, render and optionally execute a candidate node."""
        parent = self.project.history.get(parent_id)
        if parent.status != NodeStatus.SUCCEEDED or parent.review_status != ReviewStatus.ACCEPTED:
            raise ValueError("Родительский результат должен быть успешно выполнен и принят.")
        if operation_id in self._unavailable_operations:
            raise RuntimeError(self._unavailable_operations[operation_id])
        spec = self.registry.get(operation_id)
        parameters = spec.validate(raw_parameters)
        input_refs = build_input_refs(
            self.project.history, parent_id, operation_id, parameters
        )
        node = self.project.history.add_node(
            operation_id,
            make_operation_label(operation_id, parameters),
            parent_ids=[parent_id],
            input_refs=input_refs,
            parameters=parameters,
            metadata={"created_in": "processing_workbench"},
            software_version=application_version(),
            parameter_schema_version=spec.parameter_schema_version,
            review_status=ReviewStatus.CANDIDATE,
        )
        self.project.save()
        self._branch_parent_id = parent_id
        self._rebuild_graph(select_id=node.id)
        self._log(
            f"Создан кандидат {node.label} · {node.id[:8]}. "
            "Результат будет показан для проверки; следующий этап доступен только после принятия."
        )
        if run_immediately:
            self._run_selected()
        else:
            self.tabs.setCurrentIndex(0)
            self._update_controls()
        return node.id

    def _run_selected(self) -> None:
        if self.is_busy:
            return
        node_id = self._selected_node_id()
        if not node_id:
            return
        node = self.project.history.get(node_id)
        if node.operation_id == "source.scan":
            return
        if node.status != NodeStatus.PENDING:
            QMessageBox.information(
                self, "Запуск",
                "Узел уже запускался или находится в терминальном состоянии. "
                "Для новых параметров создайте новую ветвь от входного узла.",
            )
            return
        try:
            self.project.history.transition(node_id, NodeStatus.RUNNING)
            self.project.save()
            snapshot = WorkflowHistory(list(self.project.history.nodes))
            task = _RunNodeTask(
                node_id, snapshot, str(self.project.artifact_store.directory)
            )
            task.signals.succeeded.connect(self._on_run_succeeded)
            task.signals.failed.connect(self._on_run_failed)
            self._active_task = task
            self.progress.setRange(0, 0)
            self.status_label.setText(f"Выполняется: {node.label}")
            self.tabs.setCurrentIndex(2)
            self._rebuild_graph(select_id=node_id)
            self.thread_pool.start(task)
        except Exception as exc:
            if self.project.history.get(node_id).status == NodeStatus.RUNNING:
                self.project.history.transition(node_id, NodeStatus.FAILED, error=str(exc))
                self.project.save()
            self._active_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            QMessageBox.critical(self, "Ошибка запуска", str(exc))
            self._rebuild_graph(select_id=node_id)

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
            self.status_label.setText("Этап выполнен. Проверьте результат и примите или отклоните его.")
        except Exception:
            persistence_error = traceback.format_exc()
            self._log(persistence_error)
            self.status_label.setText("Результат вычислен, но историю не удалось сохранить")
        finally:
            self._active_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if persistence_error is None else 0)
            self._rebuild_graph(select_id=node_id)
            self._update_controls()
        if persistence_error is not None:
            QMessageBox.critical(
                self, "Не удалось сохранить проект",
                "Артефакт вычислен, но запись workflow.json завершилась ошибкой. "
                "Не закрывайте приложение до успешного сохранения.\n\n" + persistence_error,
            )
            return
        # A new run is a candidate, not an accepted data revision. Show its
        # result immediately so the user can inspect it before making a decision.
        self._preview_selected()

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
            self._rebuild_graph(select_id=node_id)
            self._update_controls()
        if persistence_error is not None:
            QMessageBox.critical(self, "Ошибка сохранения истории", persistence_error)

    def _rebuild_graph(self, *_args, select_id: str | None = None) -> None:
        if not hasattr(self, "graph"):
            return
        previous = select_id or self._selected_node_id()
        self.graph.set_history(
            self.project.history,
            selected_id=previous,
            show_rejected=self.show_rejected_check.isChecked(),
        )
        selected_id = self.graph.selected_node_id
        if selected_id:
            self._on_graph_node_selected(selected_id)
        else:
            self.node_details.clear()
        self._update_controls()

    def _on_graph_node_selected(self, node_id: str) -> None:
        try:
            node = self.project.history.get(node_id)
        except KeyError:
            self.node_details.clear()
            self._update_controls()
            return
        self._branch_parent_id = (
            node_id
            if node.status == NodeStatus.SUCCEEDED
            and node.review_status == ReviewStatus.ACCEPTED
            else None
        )
        if self._branch_parent_id:
            available = [
                operation_id for operation_id in self._available_operations_for_parent(node_id)
                if operation_id not in self._unavailable_operations
            ]
            if available and self.operation_combo.currentData() not in available:
                self.operation_combo.setCurrentIndex(
                    self.operation_combo.findData(available[0])
                )
        detail = {
            "node_id": node.id,
            "operation": node.operation_id,
            "label": node.label,
            "execution_status": node.status,
            "review_status": node.review_status,
            "parameters": node.parameters,
            "inputs": node.input_refs,
            "outputs": node.output_refs,
            "metadata": node.metadata,
            "summary": node.summary,
            "error": node.error,
            "created_at": node.created_at,
            "started_at": node.started_at,
            "finished_at": node.finished_at,
            "software_version": node.software_version,
            "parameter_schema_version": node.parameter_schema_version,
            "parents": list(node.parent_ids),
            "children": [child.id for child in self.project.history.children(node_id)],
        }
        self.node_details.setPlainText(
            json.dumps(detail, ensure_ascii=False, indent=2, default=str)
        )
        self._update_controls()

    def _accept_selected(self) -> None:
        node_id = self._selected_node_id()
        if not node_id or self.is_busy:
            return
        try:
            node = self.project.history.get(node_id)
            self.project.history.set_review_status(node_id, ReviewStatus.ACCEPTED)
            self.project.save()
            self._log(f"Результат принят: {node.label}")
            self.status_label.setText("Результат принят и доступен для следующих этапов.")
            self._rebuild_graph(select_id=node_id)
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось принять результат", str(exc))

    def _reject_selected(self) -> None:
        node_id = self._selected_node_id()
        if not node_id or self.is_busy:
            return
        node = self.project.history.get(node_id)
        if self.project.history.children(node_id):
            QMessageBox.warning(
                self, "Нельзя отклонить ветвь",
                "У результата уже есть дочерние этапы. Сначала примите или отклоните "
                "дочерние результаты, затем вернитесь к этому узлу.",
            )
            return
        try:
            self.project.history.set_review_status(node_id, ReviewStatus.REJECTED)
            self.project.save()
            preview_node = self._preview_context.get("node") if self._preview_context else None
            if preview_node is not None and preview_node.id == node_id:
                self._restore_preview()
            parent_id = node.parent_ids[0] if node.parent_ids else None
            self._log(
                f"Результат отклонён: {node.label}. Артефакт и provenance сохранены в истории."
            )
            self.status_label.setText(
                "Ветка отклонена и скрыта из активного графа; включите «Отклонённые» для аудита."
            )
            self._rebuild_graph(select_id=parent_id)
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось отклонить результат", str(exc))

    def _update_controls(self) -> None:
        busy = self.is_busy
        node_id = self._selected_node_id()
        runnable = previewable = addable = accept_enabled = reject_enabled = False
        if node_id:
            node = self.project.history.get(node_id)
            accepted = node.review_status == ReviewStatus.ACCEPTED
            candidate = node.review_status == ReviewStatus.CANDIDATE
            allowed_operations = [
                operation_id for operation_id in self._available_operations_for_parent(node_id)
                if operation_id not in self._unavailable_operations
            ]
            addable = (
                node.status == NodeStatus.SUCCEEDED
                and accepted
                and self.operation_combo.currentData() in allowed_operations
            )
            runnable = (
                node.operation_id != "source.scan"
                and node.status == NodeStatus.PENDING
                and candidate
            )
            previewable = node.status == NodeStatus.SUCCEEDED
            accept_enabled = node.status == NodeStatus.SUCCEEDED and candidate
            reject_enabled = (
                accept_enabled and not self.project.history.children(node_id)
            )
        self.add_button.setEnabled(not busy and addable)
        self.run_button.setEnabled(not busy and runnable)
        self.preview_button.setEnabled(not busy and previewable)
        self.restore_button.setEnabled(not busy and self._preview_context is not None)
        self.accept_button.setVisible(accept_enabled)
        self.reject_button.setVisible(accept_enabled)
        self.accept_button.setEnabled(not busy and accept_enabled)
        self.reject_button.setEnabled(not busy and reject_enabled)
        self.fit_graph_button.setEnabled(not busy)
        self.show_rejected_check.setEnabled(not busy)

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
            self.tabs.setCurrentIndex(0)
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
            "preprocess.detector_bad_pixels": "intensity_map",
            "preprocess.spikes": "spike_count_map",
            "peaks.detect": "peak_count_map",
            "fit.scan": "fit_quality_map",
        }.get(operation_id, "intensity_map")

    @staticmethod
    def _initial_valid_pixel(scan: ArtifactBundle, shape_yx: tuple[int, int]) -> tuple[int, int]:
        """Choose a central fully valid pixel, falling back to nearest valid site."""
        ny, nx = map(int, shape_yx)
        if ny <= 0 or nx <= 0:
            return 0, 0
        valid = scan.arrays.get("analysis_valid_mask_yx")
        if valid is None:
            valid_yxt = scan.arrays.get("valid_mask_yxt")
            if valid_yxt is not None:
                valid_yxt = np.asarray(valid_yxt, dtype=bool)
                if valid_yxt.ndim == 3 and valid_yxt.shape[:2] == (ny, nx):
                    valid = np.all(valid_yxt, axis=-1)
        if valid is None:
            return nx // 2, ny // 2
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (ny, nx):
            raise ValueError("Spatial validity mask не согласована с preview grid")
        candidates = np.argwhere(valid)
        if candidates.size == 0:
            return nx // 2, ny // 2
        cy, cx = ny // 2, nx // 2
        distances = (candidates[:, 0] - cy) ** 2 + (candidates[:, 1] - cx) ** 2
        selected_y, selected_x = candidates[int(np.argmin(distances))]
        return int(selected_x), int(selected_y)

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
            # A detector exclusion is missing data, never a physical zero.
            # Partial pooling bins remain visible for QC, but bins with no
            # valid contribution are rendered as no-data (NaN).
            coverage = scan.arrays.get("valid_fraction_yx")
            if coverage is not None:
                coverage = np.asarray(coverage, dtype=float)
                if coverage.shape != map_data.shape:
                    raise ValueError("valid_fraction_yx не согласована с intensity map")
                map_data = np.where(coverage > 0.0, map_data, np.nan)
            else:
                valid_yxt = scan.arrays.get("valid_mask_yxt")
                if valid_yxt is not None:
                    valid_yxt = np.asarray(valid_yxt, dtype=bool)
                    if valid_yxt.shape != data.shape:
                        raise ValueError("valid_mask_yxt не согласована с intensity map")
                    map_data = np.where(np.any(valid_yxt, axis=-1), map_data, np.nan)
            bad_count = int(scan.metadata.get("detector_bad_pixel_count", 0))
            if bad_count:
                map_title = f"Суммарная интенсивность · исключено пикселей: {bad_count}"
            elif scan.metadata.get("detector_bad_pixel_stage_applied"):
                map_title = "Суммарная интенсивность · контроль дефектов подтверждён"
            else:
                map_title = "Суммарная интенсивность"

        ny, nx = map_data.shape
        # Select a valid pixel by default; never open an excluded high-count ROC
        # merely because it happens to be at the geometric image centre.
        x, y = self._initial_valid_pixel(scan, data.shape[:2])
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
        elif node.operation_id == "preprocess.detector_bad_pixels":
            scan = ctx.get("scan")
            bad = np.asarray(
                scan.arrays.get("detector_bad_pixel_mask_yx", np.zeros((0, 0), dtype=bool)),
                dtype=bool,
            ) if scan is not None else np.zeros((0, 0), dtype=bool)
            coords = scan.metadata.get("detector_bad_pixel_exclusions", []) if scan is not None else []
            count = int(np.count_nonzero(bad))
            mode_text = (
                f"Исходные counts сохранены без изменения на {len(scan.arrays['theta'])} θ-отсчётах; "
                "пиксели исключены только validity mask."
                if count else "Подтверждено: для этого скана исключаемые пиксели не заданы."
            )
            ax.text(
                0.05, 0.95,
                "Контроль дефектных пикселей детектора\n\n"
                f"Исключено пикселей: {count}\n"
                f"Координаты (x,y), с нуля: {coords}\n"
                f"{mode_text}\n"
                f"valid_mask=False: {'да' if count else 'нет'}\n"
                "Исходный HDF5 и counts артефакта не изменены\n\n"
                "Исключённые пиксели не участвуют в оценке фона,\n"
                "peak-neighborhood и Fit.",
                ha="left", va="top", transform=ax.transAxes,
            )
            ax.set_axis_off()
        elif node.operation_id == "preprocess.pool":
            scan = ctx.get("scan")
            pool = scan.metadata.get("pool", {}) if scan is not None else {}
            shape = np.asarray(scan.arrays["data_yxt"]).shape if scan is not None else ()
            coverage = scan.arrays.get("valid_fraction_yx") if scan is not None else None
            partial_bins = int(scan.metadata.get("pool_partial_coverage_bins", 0)) if scan is not None else 0
            coverage_mean = float(np.mean(coverage)) if coverage is not None and np.size(coverage) else 1.0
            ax.text(
                0.05, 0.95,
                "Пространственный pooling\n\n"
                f"Размер окна: {pool.get('size', '—')} × {pool.get('size', '—')}\n"
                f"Stride: {pool.get('stride', '—')}\n"
                f"Выходная форма Y×X×θ: {shape}\n"
                f"Частично покрытых окон: {partial_bins}\n"
                f"Среднее покрытие: {coverage_mean:.1%}\n"
                "Неполные sums сохранены для карты; в анализ не включены.\n"
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
            scan_for_mask = ctx.get("scan")
            if scan_for_mask is not None:
                coverage = scan_for_mask.arrays.get("valid_fraction_yx")
                if coverage is not None:
                    coverage = np.asarray(coverage, dtype=float)
                    if coverage.shape != intensity.shape:
                        raise ValueError("valid_fraction_yx не согласована с preview map")
                    intensity = np.where(coverage > 0.0, intensity, np.nan)
                else:
                    valid_yxt = scan_for_mask.arrays.get("valid_mask_yxt")
                    if valid_yxt is not None:
                        valid_yxt = np.asarray(valid_yxt, dtype=bool)
                        if valid_yxt.shape != data_yxt.shape:
                            raise ValueError("valid_mask_yxt не согласована с preview map")
                        intensity = np.where(np.any(valid_yxt, axis=-1), intensity, np.nan)
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

        bad_mask = None
        if ctx.get("scan") is not None:
            scan_arrays = ctx["scan"].arrays
            raw_bad = scan_arrays.get("detector_bad_pixel_mask_yx")
            if raw_bad is not None:
                bad_mask = np.asarray(raw_bad, dtype=bool)
                if bad_mask.shape == map_data.shape and np.any(bad_mask):
                    valid_counts = scan_arrays.get("valid_pixel_count_yx")
                    represented_counts = scan_arrays.get("represented_detector_pixel_count_yx")
                    if valid_counts is not None and represented_counts is not None:
                        valid_counts = np.asarray(valid_counts, dtype=np.int64)
                        represented_counts = np.asarray(represented_counts, dtype=np.int64)
                        if valid_counts.shape == map_data.shape and represented_counts.shape == map_data.shape:
                            fully_excluded = bad_mask & (valid_counts == 0)
                            partially_covered = bad_mask & (valid_counts > 0) & (valid_counts < represented_counts)
                        else:
                            fully_excluded = bad_mask
                            partially_covered = np.zeros_like(bad_mask)
                    else:
                        fully_excluded = bad_mask
                        partially_covered = np.zeros_like(bad_mask)

                    if np.any(fully_excluded):
                        bad_y, bad_x = np.nonzero(fully_excluded)
                        self.ax_map.scatter(
                            bad_x, bad_y, marker="x", s=36, linewidths=1.25,
                            color="#D65F68", label="Исключённый пиксель / bin без вклада",
                            zorder=5,
                        )
                    if np.any(partially_covered):
                        partial_y, partial_x = np.nonzero(partially_covered)
                        self.ax_map.scatter(
                            partial_x, partial_y, marker="s", s=34, linewidths=1.0,
                            facecolors="none", edgecolors="#D59A43",
                            label="Частичное покрытие pooling-окна",
                            zorder=5,
                        )
                    if np.any(fully_excluded) or np.any(partially_covered):
                        self.ax_map.legend(loc="best", fontsize=7)
        self.ax_map.plot([x], [y], marker="+", markersize=10, markeredgewidth=1.5)

        curve = np.asarray(data_yxt[y, x, :], dtype=float)
        is_excluded = False
        partial_coverage = False
        if ctx.get("scan") is not None:
            valid = ctx["scan"].arrays.get("valid_mask_yxt")
            if valid is not None:
                valid = np.asarray(valid, dtype=bool)
                if valid.shape != data_yxt.shape:
                    raise ValueError("valid_mask_yxt не согласована с выбранной ROC")
                is_excluded = not bool(np.any(valid[y, x, :]))
            analysis_valid = ctx["scan"].arrays.get("analysis_valid_mask_yx")
            if analysis_valid is not None:
                analysis_valid = np.asarray(analysis_valid, dtype=bool)
                if analysis_valid.shape != map_data.shape:
                    raise ValueError("analysis_valid_mask_yx не согласована с выбранной ROC")
                partial_coverage = not bool(analysis_valid[y, x]) and not is_excluded

        # Excluded detector pixels have no plotted ROC: the original high counts
        # remain in the artifact for provenance, but are deliberately not shown
        # as a physical zero line or as a measured curve.
        if is_excluded:
            self.ax_curve.text(
                0.5, 0.5,
                "НЕТ ДАННЫХ\nПиксель исключён validity mask\nИзмеренные counts сохранены без изменений",
                ha="center", va="center", transform=self.ax_curve.transAxes,
            )
            self.ax_curve.set_axis_off()
        else:
            curve_label = "Исходная ROC"
            if partial_coverage:
                curve_label = "ROC · частичное покрытие (только QC)"
            self.ax_curve.plot(theta, curve, linewidth=1.1, alpha=0.8, label=curve_label)

            # A partial pooling bin is intentionally excluded from scientific
            # analysis. Show only its observed valid-count sum; do not overlay
            # spike/peak/Fit outputs produced on fully covered sites.
            if not partial_coverage:
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
                    # Older v1 artifacts may contain an empty DataFrame without
                    # column names. Keep the preview safe for legacy results.
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
            self.ax_curve.set_xlabel("θ")
            self.ax_curve.set_ylabel("Интенсивность")
            handles, labels = self.ax_curve.get_legend_handles_labels()
            if handles:
                self.ax_curve.legend(loc="best", fontsize=8)
        if is_excluded:
            suffix = " · НЕДЕЙСТВИТЕЛЕН (counts сохранены; исключён маской)"
        elif partial_coverage:
            counts = ctx["scan"].arrays.get("valid_pixel_count_yx") if ctx.get("scan") is not None else None
            represented = ctx["scan"].arrays.get("represented_detector_pixel_count_yx") if ctx.get("scan") is not None else None
            if counts is not None and represented is not None:
                count = int(np.asarray(counts)[y, x])
                area = int(np.asarray(represented)[y, x])
                fraction = count / area if area > 0 else 0.0
                suffix = f" · ЧАСТИЧНОЕ ПОКРЫТИЕ {count}/{area} ({fraction:.1%}) · не для анализа"
            else:
                suffix = " · ЧАСТИЧНОЕ ПОКРЫТИЕ · не для анализа"
        else:
            suffix = ""
        self.ax_curve.set_title(f"ROC-кривая · x={x}, y={y}")
        self.pixel_label.setText(f"Пиксель: X={x}, Y={y}{suffix}")
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
