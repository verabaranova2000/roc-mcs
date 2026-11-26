"""Branchable processing workbench, embeddable as a QWidget or standalone window."""
from __future__ import annotations

import json
import traceback
from html import escape as html_escape
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal, QEvent, QSize, QTimer
from PyQt6.QtGui import QCursor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFormLayout, QComboBox,
    QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QCheckBox, QLineEdit, QPushButton, QLabel,
    QPlainTextEdit, QTabWidget, QStackedWidget, QProgressBar, QMessageBox,
    QSizePolicy, QFrame, QMenu, QScrollArea, QToolButton, QWidgetAction,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
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
from roc_mcs.analysis.workflow_preview import build_workflow_preview_state
from roc_mcs.analysis.hkl_preview import load_hkl_identification_preview
from roc_mcs.analysis.preview_context import load_node_preview_context
from .graph_canvas import ProcessingGraphView
from .peak_detection_preview import PeakDetectionPreview
from .branch_detection_preview import BranchDetectionPreview
from roc_mcs.visualization.peak_count_map import integrated_intensity_map, peak_count_rgba
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR
from roc_mcs.gui.roi_explorer.map_tools_widget import pooling_preview_pixmap
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, apply_popup_theme, resolve_theme, theme_colors
from roc_mcs.gui.roi_explorer.hkl_results_panel import HklResultsPanel


# Keep the initial values and the in-form reset action in one place. These are
# the established pixel peak-detector defaults used by the current workflow.
_PEAK_FORM_DEFAULTS = {
    "window_size": 3,
    "smooth_window": 11,
    "smooth_polyorder": 2,
    "min_distance": 6,
    "prominence_mode": "sigma",
    "prominence_value": 3.0,
    "min_relative_prominence": 0.07,
    "use_interpolated_data": False,
}
_PEAK_PROMINENCE_MODE_DEFAULTS = {"sigma": 3.0, "fraction": 0.07}


class _PopupDragTitle(QLabel):
    """Small drag handle for non-modal tool popups."""

    def __init__(self, text: str, parent: QWidget):
        super().__init__(text, parent)
        self._drag_offset = None
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
        super().mouseReleaseEvent(event)


class _StageConfigurationPopup(QFrame):
    """Movable, non-modal stage editor; stays open while the map is clicked."""

    about_to_close = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("processingStagePopup")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def closeEvent(self, event) -> None:
        self.about_to_close.emit()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)


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

    def __init__(self, node_id: str, history: WorkflowHistory, artifact_directory: str, source_metadata: dict[str, Any] | None = None):
        super().__init__()
        self.node_id = node_id
        self.history_snapshot = WorkflowHistory(list(history.nodes))
        self.artifact_directory = artifact_directory
        self.source_metadata = dict(source_metadata or {})
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            executor = WorkflowExecutor(
                self.history_snapshot,
                ArtifactStore(self.artifact_directory),
                default_registry(),
            )
            node = self.history_snapshot.get(self.node_id)
            context = load_node_preview_context(node, self.history_snapshot, executor)
            if context["scan"] is None:
                raise RuntimeError("Для предпросмотра не удалось разрешить ScanArtifact")
            # Build the typed host view in the worker: ROI Explorer receives only
            # grid-validated maps and θ-compatible signal cubes.
            context["view_state"] = build_workflow_preview_state(
                node, context, self.source_metadata
            )
            self.signals.succeeded.emit(self.node_id, context)
        except Exception:
            self.signals.failed.emit(self.node_id, traceback.format_exc())


class _HklResultPreviewTask(QRunnable):
    """Load only the immutable HKL result artifact; never load the 3D Scan.

    The HKL stage is analysis-only, so its dedicated result window needs the
    five tabular outputs but not the generic map/ROC preview lineage.
    """

    def __init__(self, node_id: str, history: WorkflowHistory, artifact_directory: str):
        super().__init__()
        self.node_id = str(node_id)
        self.history_snapshot = WorkflowHistory(list(history.nodes))
        self.artifact_directory = str(artifact_directory)
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            node = self.history_snapshot.get(self.node_id)
            if node.operation_id != "theory.identify_hkl":
                raise ValueError(f"Узел {node.id!r} не является HKL-этапом")
            if node.status != NodeStatus.SUCCEEDED:
                raise RuntimeError("HKL-результат ещё не завершён успешно")
            artifact_id = node.output_refs.get("hkl_analysis")
            if not artifact_id:
                raise RuntimeError("В успешно завершённом HKL-узле отсутствует выход hkl_analysis")
            preview = load_hkl_identification_preview(
                ArtifactStore(self.artifact_directory), str(artifact_id)
            )
            self.signals.succeeded.emit(self.node_id, preview)
        except Exception:
            self.signals.failed.emit(self.node_id, traceback.format_exc())


class _PeakPixelInputTask(QRunnable):
    """Loads the exact Scan/Spike inputs for one selected detector pixel."""

    def __init__(self, parent_id: str, history: WorkflowHistory, artifact_directory: str,
                 detector_x: float, detector_y: float):
        super().__init__()
        self.parent_id = str(parent_id)
        self.history_snapshot = WorkflowHistory(list(history.nodes))
        self.artifact_directory = str(artifact_directory)
        self.detector_x = float(detector_x)
        self.detector_y = float(detector_y)
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            from .peak_preview_data import build_peak_pixel_preview_data

            history = self.history_snapshot
            executor = WorkflowExecutor(
                history, ArtifactStore(self.artifact_directory), default_registry()
            )
            scan_node = history.find_ancestor_output(self.parent_id, "scan")
            if scan_node is None or scan_node.status != NodeStatus.SUCCEEDED:
                raise RuntimeError("В выбранной ветви не найден успешно завершённый Scan-вход")
            scan_bundle = executor.load_output(scan_node, "scan")

            spike_bundle = None
            spike_node = history.find_ancestor_output(self.parent_id, "spikes")
            if spike_node is not None and spike_node.status == NodeStatus.SUCCEEDED:
                spike_bundle = executor.load_output(spike_node, "spikes")

            payload = build_peak_pixel_preview_data(
                scan_bundle, spike_bundle, self.detector_x, self.detector_y
            )
            self.signals.succeeded.emit(self.parent_id, payload)
        except Exception:
            self.signals.failed.emit(self.parent_id, traceback.format_exc())


class _BranchStatsTask(QRunnable):
    """Compute per-scan gate diagnostics from a completed peak/fit artifact."""

    def __init__(self, parent_id: str, parent_operation: str, artifact_id: str, artifact_directory: str):
        super().__init__()
        self.parent_id = str(parent_id)
        self.parent_operation = str(parent_operation)
        self.artifact_id = str(artifact_id)
        self.artifact_directory = str(artifact_directory)
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            from roc_mcs.peaks.branches import summarize_peak_spacing, fitted_peak_positions
            from .branch_preview_data import select_preview_patch

            bundle = ArtifactStore(self.artifact_directory).read(self.artifact_id)
            if self.parent_operation == "fit.scan":
                catalog = fitted_peak_positions(bundle)
                position_label = "позиции после Fit"
            else:
                catalog = bundle.tables.get("peak_catalog")
                if catalog is None:
                    raise ValueError("В артефакте поиска пиков отсутствует peak_catalog")
                if getattr(catalog, "empty", False):
                    raise ValueError("Каталог пиков пуст: для диагностики ветвей нет peak-events")
                position_label = "обнаруженные пики"
            stats = summarize_peak_spacing(catalog, reference_theta_gate=60.0)
            preview_catalog, preview_info = select_preview_patch(catalog)
            preview_rows = [
                (int(row.peak_id), int(row.x), int(row.y), float(row.theta))
                for row in preview_catalog.itertuples(index=False)
            ]
            self.signals.succeeded.emit(self.parent_id, {
                "stats": stats,
                "preview_rows": preview_rows,
                "preview_info": preview_info,
                "position_label": position_label,
            })
        except Exception:
            self.signals.failed.emit(self.parent_id, traceback.format_exc())


class ProcessingWorkbenchWidget(QWidget):
    """The workbench itself; host applications can embed it in a dock/dialog.

    Signals:
      preview_requested(request): temporarily show selected node in host views.
      preview_restore_requested(): restore the host views as they were before preview.
    """

    preview_requested = pyqtSignal(object)
    preview_state_ready = pyqtSignal(object)
    preview_restore_requested = pyqtSignal()
    pixel_pick_mode_requested = pyqtSignal(bool, str, bool)
    busy_changed = pyqtSignal(bool)

    def __init__(self, project: WorkflowProject, parent: QWidget | None = None, *, embedded: bool = False):
        super().__init__(parent)
        self.project = project
        self.embedded = bool(embedded)
        self._last_busy_state = False
        self.registry = default_registry()
        self.thread_pool = QThreadPool.globalInstance()
        self._active_task: _RunNodeTask | None = None
        self._active_preview_task: _PreviewNodeTask | None = None
        self._hkl_preview_tasks: dict[str, _HklResultPreviewTask] = {}
        self._hkl_preview_cache: dict[str, Any] = {}
        self._active_peak_input_task: _PeakPixelInputTask | None = None
        self._active_branch_stats_task: _BranchStatsTask | None = None
        self._branch_gate_values: dict[str, float] = {}
        # Branch outputs are immutable artifacts. Cache only their small metadata
        # (not the peak catalog) so graph rebuilds never reload large tables.
        self._branch_count_cache: dict[str, tuple[int | None, str | None]] = {}
        self._active_branch_stats_parent_id: str | None = None
        self._pending_peak_pixel_coordinates: tuple[float, float] | None = None
        self._discard_active_peak_input = False
        self._preview_context: dict[str, Any] | None = None
        self._map_click_connection = None
        self._branch_parent_id: str | None = None
        self._active_stage_popup: _StageConfigurationPopup | None = None
        self._active_stage_operation: str | None = None
        self._pending_peak_map_parent_id: str | None = None
        self._theme = resolve_theme(parent if parent is not None else self)
        self._applying_theme = False
        self._build_ui()
        self._apply_theme(self._theme)
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
        self.theme_button = QToolButton(self)
        self.theme_button.setObjectName("processingWorkbenchThemeToggle")
        self.theme_button.setAutoRaise(True)
        self.theme_button.setIconSize(QSize(16, 16))
        self.theme_button.setFixedSize(28, 28)
        self.theme_button.setToolTip("Переключить тему Workbench")
        self.theme_button.setAccessibleName("Переключить тему Workbench")
        self.theme_button.clicked.connect(self._toggle_theme)
        header.addWidget(self.theme_button)
        self.save_button = QPushButton("Сохранить историю")
        self.save_button.clicked.connect(self._save_project)
        header.addWidget(self.save_button)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
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
        self.fit_graph_button = QToolButton(self)
        self.fit_graph_button.setObjectName("processingGraphFitButton")
        self.fit_graph_button.setAutoRaise(True)
        self.fit_graph_button.setToolTip("Вписать дерево обработки в область просмотра")
        self.fit_graph_button.setAccessibleName("Вписать дерево обработки")
        graph_header.addWidget(self.fit_graph_button)
        self.graph_scroll_button = QToolButton(self)
        self.graph_scroll_button.setObjectName("processingGraphScrollModeButton")
        self.graph_scroll_button.setAutoRaise(True)
        self.graph_scroll_button.setCheckable(True)
        self.graph_scroll_button.setChecked(True)
        self.graph_scroll_button.setToolTip(
            "Вертикальная прокрутка включена. Выключите, чтобы целиком вписать дерево по высоте."
        )
        self.graph_scroll_button.setAccessibleName("Переключить вертикальную прокрутку дерева")
        graph_header.addWidget(self.graph_scroll_button)
        left_layout.addLayout(graph_header)
        self._left_graph_layout = left_layout

        self.graph = ProcessingGraphView(left)
        self.fit_graph_button.clicked.connect(self.graph.fit_graph)
        self.graph.set_vertical_scroll_enabled(self.graph_scroll_button.isChecked())
        self.graph_scroll_button.toggled.connect(self._on_graph_scroll_mode_changed)
        self.graph.node_selected.connect(self._on_graph_node_selected)
        self.graph.add_stage_requested.connect(self._on_graph_add_stage_requested)
        self.graph.preview_requested.connect(self._on_graph_preview_requested)
        left_layout.addWidget(self.graph, 1)
        self.branch_hint = QLabel(
            "Щелчок — инспектор · глаз — результат · + — новая ветвь. "
            "Сплошная стрелка — вычислительный вход; пунктир — только происхождение ветви. "
            "Перетаскивание — панорама; Ctrl+колесо — масштаб."
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
            "preprocess.pool", "preprocess.spikes", "peaks.detect", "fit.scan",
            "peaks.branches", "peaks.branch_correlation", "theory.identify_hkl",
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
        self._build_branch_form()
        self._build_branch_correlation_form()
        self._build_hkl_identification_form()
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

        if self.embedded:
            # Embed the existing graph/runner, not a second workflow UI. The
            # operation forms remain alive for the graph's contextual + popup.
            self.theme_button.hide()
            self.save_button.hide()
            self.show_rejected_check.setText("Архив")
            self.show_rejected_check.setToolTip("Показывать отклонённые ветви истории")
            self.branch_hint.hide()
            right.hide()
            self.tabs.hide()
            self.graph.setMinimumSize(280, 240)
            self.graph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            actions_top = QHBoxLayout()
            actions_top.setContentsMargins(0, 2, 0, 0)
            actions_top.setSpacing(4)
            self.embedded_run_button = QPushButton("Запустить")
            self.embedded_run_button.setToolTip("Выполнить выбранный ожидающий этап")
            self.embedded_run_button.clicked.connect(self._run_selected)
            self.embedded_restore_button = QPushButton("Исходный вид")
            self.embedded_restore_button.setToolTip("Снять предпросмотр workflow и вернуть представление HDF5")
            self.embedded_restore_button.clicked.connect(self._restore_preview)
            actions_top.addWidget(self.embedded_run_button, 1)
            actions_top.addWidget(self.embedded_restore_button, 1)
            self._left_graph_layout.addLayout(actions_top)

            actions_bottom = QHBoxLayout()
            actions_bottom.setContentsMargins(0, 0, 0, 0)
            actions_bottom.setSpacing(4)
            self.embedded_accept_button = QPushButton("Принять")
            self.embedded_accept_button.setToolTip("Зафиксировать проверенный результат для следующих этапов")
            self.embedded_accept_button.clicked.connect(self._accept_selected)
            self.embedded_reject_button = QPushButton("Отклонить")
            self.embedded_reject_button.setToolTip("Отклонить результат; история и provenance сохраняются")
            self.embedded_reject_button.clicked.connect(self._reject_selected)
            actions_bottom.addWidget(self.embedded_accept_button, 1)
            actions_bottom.addWidget(self.embedded_reject_button, 1)
            self._left_graph_layout.addLayout(actions_bottom)

            # These are the same status/progress objects used by the standalone
            # workbench; reparenting avoids duplicated or divergent indicators.
            self.status_label.setWordWrap(True)
            self.status_label.setMaximumHeight(44)
            self.progress.setMaximumHeight(5)
            self._left_graph_layout.addWidget(self.status_label)
            self._left_graph_layout.addWidget(self.progress)

        self._show_operation_form()

    def _on_graph_scroll_mode_changed(self, enabled: bool) -> None:
        self.graph.set_vertical_scroll_enabled(bool(enabled))
        if enabled:
            self.graph_scroll_button.setToolTip(
                "Вертикальная прокрутка включена. Выключите, чтобы целиком вписать дерево по высоте."
            )
        else:
            self.graph_scroll_button.setToolTip(
                "Режим полного дерева включён: всё дерево вписано по высоте и ширине. "
                "Включите, чтобы вернуть вертикальную прокрутку и более крупные узлы."
            )

    @staticmethod
    def _global_icon_size() -> int:
        app = QApplication.instance()
        raw = app.property("rocMcsIconSize") if app is not None else None
        try:
            return max(12, min(20, int(raw)))
        except (TypeError, ValueError, OverflowError):
            return 16

    def _toggle_theme(self) -> None:
        self._apply_theme("dark" if self._theme == "light" else "light")

    def _apply_theme(self, theme: str | None = None) -> None:
        """Apply the ROI Explorer palette and the same moon/sun control as VTK."""
        active = str(theme or self._theme or "light").lower()
        if active not in {"light", "dark"}:
            active = "light"
        self._applying_theme = True
        try:
            self._theme = apply_widget_palette(self, active)
            app = QApplication.instance()
            if app is not None:
                app.setProperty("rocMcsTheme", self._theme)
            colors = theme_colors(self._theme)
            icon_name = "theme_moon.svg" if self._theme == "light" else "theme_sun.svg"
            self.theme_button.setIcon(
                svg_icon(ICON_DIR / icon_name, color=colors["text"], widget=self.theme_button)
            )
            target = "тёмную" if self._theme == "light" else "светлую"
            self.theme_button.setToolTip(f"Переключить Workbench на {target} тему")
            self.theme_button.setAccessibleName(f"Переключить Workbench на {target} тему")
            icon_size = self._global_icon_size()
            for button, name in (
                (self.fit_graph_button, "zoom_out_map.svg"),
                (self.graph_scroll_button, "vertical_scroll.svg"),
            ):
                button.setIcon(svg_icon(ICON_DIR / name, color=colors["text"], widget=button))
                button.setIconSize(QSize(icon_size, icon_size))
                button.setFixedSize(icon_size + 8, icon_size + 8)
            self.graph.setBackgroundBrush(self.graph.palette().color(QPalette.ColorRole.Base))
            self._style_matplotlib_figure(self.figure)
            self._style_matplotlib_figure(self.diagnostic_figure)
            stage_popup = self._active_stage_popup
            if stage_popup is not None:
                apply_widget_palette(stage_popup, self._theme)
                stage_panel = getattr(stage_popup, "_stage_hkl_panel", None)
                if stage_panel is not None:
                    stage_panel.apply_theme(self._theme)
            pool_form = self._forms.get("preprocess.pool", {})
            preview = pool_form.get("preview")
            if preview is not None:
                preview.setPixmap(
                    pooling_preview_pixmap(
                        pool_form["size"].value(), pool_form["stride"].value(), self.palette()
                    )
                )
        finally:
            self._applying_theme = False

    def _style_matplotlib_figure(self, figure: Figure) -> None:
        colors = theme_colors(self._theme)
        figure.set_facecolor(colors["window"])
        for axes in figure.axes:
            axes.set_facecolor(colors["axes"])
            axes.tick_params(axis="both", colors=colors["muted"])
            axes.xaxis.label.set_color(colors["text"])
            axes.yaxis.label.set_color(colors["text"])
            axes.title.set_color(colors["text"])
            for spine in axes.spines.values():
                spine.set_color(colors["border"])
            for text in axes.texts:
                text.set_color(colors["text"])
            legend = axes.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(colors["base"])
                legend.get_frame().set_edgecolor(colors["border"])
                for label in legend.get_texts():
                    label.set_color(colors["text"])
        # The colourbar is its own Axes and is handled by the same loop above.

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if self._applying_theme or event.type() not in (
            QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange
        ):
            return
        app = QApplication.instance()
        requested = app.property("rocMcsTheme") if app is not None else None
        if str(requested).lower() in {"light", "dark"} and str(requested).lower() != self._theme:
            self._apply_theme(str(requested).lower())

    def _add_form_page(self, operation_id: str, title: str) -> QFormLayout:
        page = QWidget()
        layout = QFormLayout(page)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
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
        hint = QLabel("Сумма counts по X/Y; ось θ сохраняется. Окно: нечётный размер.")
        hint.setWordWrap(True)
        hint.setToolTip(
            "В каждом пространственном окне counts суммируются по X/Y. "
            "Отсчёты θ не усредняются и не прореживаются. Окна с частичным "
            "покрытием отображаются для QC, но исключаются из научного анализа."
        )
        preview = QLabel()
        preview.setFixedSize(88, 66)
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setAccessibleName("Геометрия окна pooling и шага")
        preview.setToolTip(
            "Схема двух окон: сетка — исходные пиксели, смещение — stride, "
            "перекрытие — общие пиксели соседних окон."
        )
        preview.setPixmap(pooling_preview_pixmap(size.value(), stride.value(), self.palette()))
        detail = QWidget()
        detail_row = QHBoxLayout(detail)
        detail_row.setContentsMargins(0, 0, 0, 0)
        detail_row.setSpacing(7)
        detail_text = QLabel("Геометрия окна\nи шага смещения")
        detail_text.setWordWrap(True)
        detail_row.addWidget(detail_text, 1)
        detail_row.addWidget(preview)
        layout.addRow("Схема:", detail)
        refresh_preview = lambda *_: preview.setPixmap(
            pooling_preview_pixmap(size.value(), stride.value(), self.palette())
        )
        size.valueChanged.connect(refresh_preview)
        stride.valueChanged.connect(refresh_preview)
        self._forms["preprocess.pool"].update(size=size, stride=stride, preview=preview)

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
        layout = self._add_form_page("peaks.detect", "Поиск дифракционных пиков")

        window = self._int_spin(1, 15, _PEAK_FORM_DEFAULTS["window_size"], 2)
        smooth = self._int_spin(3, 101, _PEAK_FORM_DEFAULTS["smooth_window"], 2)
        poly = self._int_spin(0, 10, _PEAK_FORM_DEFAULTS["smooth_polyorder"])
        distance = self._int_spin(1, 1000, _PEAK_FORM_DEFAULTS["min_distance"])
        mode = QComboBox()
        mode.addItem("Порог по шуму (σ)", "sigma")
        mode.addItem("Доля размаха ROC", "fraction")
        mode.setCurrentIndex(mode.findData(_PEAK_FORM_DEFAULTS["prominence_mode"]))
        prominence = self._float_spin(
            0.0001, 1000.0, _PEAK_FORM_DEFAULTS["prominence_value"], 4
        )
        prominence.setSuffix(" σ")
        prominence.setSingleStep(0.1)
        relative = self._float_spin(
            0.0001, 1.0, _PEAK_FORM_DEFAULTS["min_relative_prominence"], 4
        )
        interpolated = QCheckBox("Использовать интерполированную ROC")
        interpolated.setChecked(_PEAK_FORM_DEFAULTS["use_interpolated_data"])
        interpolated.setToolTip(
            "ВЫКЛ.: поиск выполняется по измеренным counts Scan-входа. "
            "ВКЛ.: используется clean_data_yxt из предшествующего этапа Spike detection, "
            "где отмеченные точки заменены интерполированными значениями. "
            "Это отдельное решение о входном сигнале. Включение требует Spike detection "
            "в выбранной ветви. Исходные counts не перезаписываются."
        )

        window.setToolTip(
            "Размер пространственной окрестности detector-пикселей. Для каждого пикселя "
            "соседние ROC усредняются по окну N×N до поиска пиков. Это не окно сглаживания "
            "вдоль θ. Допустимы нечётные размеры."
        )
        smooth.setToolTip(
            "Ширина Savitzky–Golay-фильтра вдоль угловой оси θ, в числе отсчётов. "
            "Нечётное окно: большее значение подавляет мелкие быстрые флуктуации, "
            "но может сгладить близкие/узкие компоненты. Не может превышать длину ROC."
        )
        poly.setToolTip(
            "Степень локального полинома, которым Savitzky–Golay-фильтр аппроксимирует "
            "сигнал внутри окна. Должна быть строго меньше ширины окна."
        )
        distance.setToolTip(
            "Минимальный горизонтальный интервал между соседними максимумами, "
            "в отсчётах θ. При конфликте find_peaks удаляет более слабый максимум. "
            "Это не ширина пика и не расстояние между пикселями детектора."
        )
        mode.setToolTip(
            "Способ задать нижнюю границу prominence — насколько вершина выступает "
            "над локальным контуром (основанием), а не над нулём."
        )
        prominence.setToolTip(
            "Для режима σ: минимум prominence = значение × оценка σ фонового шума. "
            "Для режима размаха: минимум = доля × (max сглаженной ROC − min сглаженной ROC)."
        )
        relative.setToolTip(
            "Второй фильтр после абсолютного prominence: сохраняются кандидаты, "
            "чья prominence не меньше указанной доли от самого выраженного кандидата "
            "в этой же ROC-кривой. Например, 0.07 означает 7%."
        )

        layout.addRow("Пространственная окрестность, px:", window)
        layout.addRow("Окно сглаживания SG, θ-точек:", smooth)
        layout.addRow("Степень полинома SG:", poly)
        layout.addRow("Минимальная дистанция, θ-точек:", distance)
        layout.addRow("Критерий prominence:", mode)
        layout.addRow("Минимальная prominence:", prominence)
        layout.addRow("Относительный порог от max:", relative)

        preview = PeakDetectionPreview()
        preview_row = QWidget()
        preview_row_layout = QHBoxLayout(preview_row)
        preview_row_layout.setContentsMargins(0, 2, 0, 2)
        preview_row_layout.setSpacing(0)
        preview.setMinimumWidth(270)
        preview_row_layout.addWidget(preview, 1)
        pick_button = QToolButton(preview)
        pick_button.setObjectName("peakPreviewPixelPickButton")
        pick_button.setCheckable(True)
        pick_button.setAutoRaise(True)
        pick_button.setFixedSize(19, 19)
        pick_button.setIcon(svg_icon(ICON_DIR / "roi_point.svg", widget=pick_button))
        pick_button.setIconSize(QSize(13, 13))
        pick_button.setStyleSheet(
            "QToolButton#peakPreviewPixelPickButton { border:1px solid palette(mid); "
            "border-radius:3px; padding:0; background:palette(window); } "
            "QToolButton#peakPreviewPixelPickButton:hover { border-color:palette(highlight); } "
            "QToolButton#peakPreviewPixelPickButton:checked { "
            "background:palette(highlight); border-color:palette(highlight); }"
        )
        pick_button.setToolTip(
            "ВЫКЛ.: синтетическая ROC. ВКЛ.: режим выбора пикселя — можно щёлкать "
            "по 2D-карте сколько угодно раз, чтобы менять реальную ROC. "
            "Выключите кнопку, чтобы вернуться к синтетической схеме."
        )
        pick_button.setAccessibleName("Выбор реальной ROC по пикселю 2D-карты")
        pick_button.setEnabled(self.embedded)
        if not self.embedded:
            pick_button.setToolTip(
                "Выбор пикселя доступен при открытии Workbench внутри ROI Explorer."
            )
        preview.set_pixel_pick_button(pick_button)

        def refresh_preview(*_args) -> None:
            preview.set_parameters(
                window_size=window.value(),
                smooth_window=smooth.value(),
                smooth_polyorder=poly.value(),
                min_distance=distance.value(),
                prominence_mode=str(mode.currentData()),
                prominence_value=prominence.value(),
                min_relative_prominence=relative.value(),
                use_interpolated_data=interpolated.isChecked(),
            )

        # A Savitzky–Golay polynomial must have lower order than its window.
        def adapt_polynomial_limit(window_value: int) -> None:
            poly.setMaximum(max(0, min(10, int(window_value) - 1)))
            if poly.value() > poly.maximum():
                poly.setValue(poly.maximum())

        mode_values = dict(_PEAK_PROMINENCE_MODE_DEFAULTS)

        def remember_prominence(value: float) -> None:
            mode_values[str(mode.currentData())] = float(value)
            refresh_preview()

        def change_prominence_mode(_index: int) -> None:
            key = str(mode.currentData())
            prominence.blockSignals(True)
            if key == "fraction":
                prominence.setRange(0.0001, 1.0)
                prominence.setDecimals(4)
                prominence.setSingleStep(0.01)
                prominence.setSuffix(" × размаха")
            else:
                prominence.setRange(0.0001, 1000.0)
                prominence.setDecimals(4)
                prominence.setSingleStep(0.1)
                prominence.setSuffix(" σ")
            prominence.setValue(float(mode_values[key]))
            prominence.blockSignals(False)
            refresh_preview()

        def reset_peak_parameters() -> None:
            # Block individual signals so the preview is recalculated once, from
            # the complete restored configuration rather than intermediate states.
            controls = (window, smooth, poly, distance, mode, prominence, relative, interpolated)
            previous_block_states = [control.blockSignals(True) for control in controls]
            try:
                mode_values.update(_PEAK_PROMINENCE_MODE_DEFAULTS)
                window.setValue(_PEAK_FORM_DEFAULTS["window_size"])
                smooth.setValue(_PEAK_FORM_DEFAULTS["smooth_window"])
                poly.setMaximum(max(0, min(10, smooth.value() - 1)))
                poly.setValue(_PEAK_FORM_DEFAULTS["smooth_polyorder"])
                distance.setValue(_PEAK_FORM_DEFAULTS["min_distance"])
                mode.setCurrentIndex(mode.findData(_PEAK_FORM_DEFAULTS["prominence_mode"]))
                prominence.setRange(0.0001, 1000.0)
                prominence.setDecimals(4)
                prominence.setSingleStep(0.1)
                prominence.setSuffix(" σ")
                prominence.setValue(_PEAK_FORM_DEFAULTS["prominence_value"])
                relative.setValue(_PEAK_FORM_DEFAULTS["min_relative_prominence"])
                interpolated.setChecked(_PEAK_FORM_DEFAULTS["use_interpolated_data"])
            finally:
                for control, was_blocked in zip(controls, previous_block_states):
                    control.blockSignals(was_blocked)
            refresh_preview()

        def input_source_changed(_checked: bool) -> None:
            refresh_preview()
            if pick_button.isChecked():
                self._request_peak_pixel_pick(True)

        smooth.valueChanged.connect(adapt_polynomial_limit)
        for control in (window, smooth, poly, distance, relative):
            control.valueChanged.connect(refresh_preview)
        prominence.valueChanged.connect(remember_prominence)
        mode.currentIndexChanged.connect(change_prominence_mode)
        interpolated.toggled.connect(input_source_changed)

        def on_pixel_pick_toggled(checked: bool) -> None:
            pick_button.setToolTip(
                "Режим выбора активен: щёлкайте по 2D-карте, чтобы выбрать или сменить "
                "пиксель. Выключите кнопку, чтобы вернуться к синтетической ROC."
                if checked else
                "Сейчас показана синтетическая ROC. Нажмите, чтобы включить выбор "
                "реального пикселя на 2D-карте."
            )
            self._request_peak_pixel_pick(checked)

        pick_button.toggled.connect(on_pixel_pick_toggled)
        layout.addRow(preview_row)
        layout.addRow("", interpolated)
        refresh_preview()
        self._forms["peaks.detect"].update(
            window_size=window, smooth_window=smooth, smooth_polyorder=poly,
            min_distance=distance, prominence_mode=mode, prominence_value=prominence,
            min_relative_prominence=relative, use_interpolated_data=interpolated,
            preview=preview, reset_parameters=reset_peak_parameters,
            pick_button=pick_button,
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
        use_spikes = QCheckBox("Исключать помеченные спайки")
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

    def _build_branch_form(self) -> None:
        layout = self._add_form_page("peaks.branches", "Определение дифракционных ветвей")
        # Keep the parameter row compact, but let all explanatory content span
        # the full form width. This avoids the old narrow label column and text
        # clipping in the stage popup.
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        theta_gate = self._float_spin(0.001, 1_000_000.0, 60.0, 2)
        theta_gate.setSuffix(" ″")
        theta_gate.setSingleStep(1.0)
        theta_gate.setToolTip(
            "Максимальное |Δθ| для сопоставления пиков в 4-связных соседних пикселях. "
            "Единицы совпадают с единицами оси θ; для текущих исходных данных это угловые секунды."
        )
        layout.addRow("θ-gate:", theta_gate)

        preview = BranchDetectionPreview()
        preview.set_theta_gate(float(theta_gate.value()))
        theta_gate.valueChanged.connect(preview.set_theta_gate)
        layout.addRow(preview)

        diagnostics = QLabel(
            "Точки — реальные пики выбранного участка скана; X/Y — соседство пикселей, "
            "высота — θ. Цветные ленты показывают принятые связи, а оранжевый пунктир — "
            "минимальные взаимно-однозначные назначения, которые порог отверг. "
            "При увеличении θ-gate часть пунктирных связей станет сплошной, а отдельные "
            "компоненты могут объединиться. Цвет показывает локальную компоненту, "
            "а площадка появляется только для подтверждённой ячейки 2×2. Пересечение "
            "проекций линий не означает общий пик. Данные не прореживаются."
        )
        diagnostics.setWordWrap(True)
        diagnostics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        diagnostics.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        diagnostics.setToolTip(
            "Миниатюра вычисляется на реальных peak-events небольшого непрерывного "
            "фрагмента выбранного результата. Для каждой пары соседних пикселей "
            "используется тот же Hungarian one-to-one matcher, что и в production-алгоритме. "
            "Показанные компоненты локальны для миниатюры и не являются итоговыми "
            "глобальными branch_id всего скана."
        )
        layout.addRow(diagnostics)

        stats_label = QLabel("Статистика выбранного скана загружается при открытии формы…")
        stats_label.setWordWrap(True)
        stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        stats_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addRow(stats_label)

        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["N", "Пикс.", "Мед.", "Q25", "Q75", "Мин."])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setAlternatingRowColors(True)
        table.setMaximumHeight(142)
        table.setMinimumHeight(92)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        table.setToolTip(
            "Для каждого N: распределение медианного внутреннего промежутка между "
            "соседними пиками внутри одного пикселя. Единицы — угловые секунды."
        )
        header_tooltips = (
            "N: число пиков в одном пространственном пикселе.",
            "Пикс.: число пикселей с ровно N пиками и конечной внутренней медианой.",
            "Мед.: медиана распределения пиксельных медианных внутренних промежутков.",
            "Q25: 25-й процентиль (нижний квартиль) распределения пиксельных медиан.",
            "Q75: 75-й процентиль (верхний квартиль) распределения пиксельных медиан.",
            "Мин.: минимальная пиксельная медиана внутреннего промежутка среди пикселей с N пиками."
        )
        for column, tooltip in enumerate(header_tooltips):
            table.horizontalHeaderItem(column).setToolTip(tooltip)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet(
            "QTableWidget { font-size: 9px; } "
            "QHeaderView::section { padding: 3px 4px; font-size: 9px; }"
        )
        layout.addRow(table)

        hint = QLabel(
            "4-связность по X/Y · one-to-one assignment по минимуму |Δθ| · "
            "каждый пик остаётся отдельным узлом графа · несвязанные события образуют "
            "одиночные ветви. Диагностические Q90 и внутренние интервалы — ориентиры "
            "для sensitivity analysis, а не физические константы."
        )
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addRow(hint)
        self._forms["peaks.branches"].update(
            theta_gate=theta_gate, preview=preview,
            stats_label=stats_label, stats_table=table,
        )

    def _build_branch_correlation_form(self) -> None:
        layout = self._add_form_page(
            "peaks.branch_correlation", "3D-кросс-корреляция дифракционных ветвей"
        )
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        source = QLineEdit()
        source.setPlaceholderText("например, 34")
        source.setToolTip(
            "Одна опорная ветвь branch_id из принятого результата «Определение ветвей». "
            "Наведите курсор на точку в 3D-облаке: подсказка покажет branch_id."
        )
        partners = QLineEdit()
        partners.setPlaceholderText("например, 72, 32")
        partners.setToolTip(
            "Одна или несколько ветвей, которые будут независимо зарегистрированы "
            "относительно опорной. Разделители — запятая или точка с запятой; максимум 8. "
            "Так можно разложить смешанную ветвь на несколько структурных партнёров."
        )
        layout.addRow("Опорная branch_id:", source)
        layout.addRow("Ветви-партнёры:", partners)

        theta_step_info = QLabel(
            "Шаг θ на theta_index берётся автоматически из фактической оси исходного "
            "скана и проверяется на равномерность. Вводить условные 15″/index вручную "
            "не требуется."
        )
        theta_step_info.setWordWrap(True)
        theta_step_info.setToolTip(
            "Integer registration is converted with the signed step stored by peak detection/Fit. "
            "Irregular theta sampling is rejected rather than approximated."
        )
        theta_step_info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addRow("Шаг θ:", theta_step_info)
        xy_tolerance = self._float_spin(0.05, 100.0, 1.5, 3)
        xy_tolerance.setSuffix(" px")
        theta_tolerance = self._float_spin(0.05, 100_000.0, 45.0, 3)
        theta_tolerance.setSuffix(" ″")
        margin = self._float_spin(0.0, 0.95, 0.20, 3)
        margin.setSingleStep(0.05)
        margin.setToolTip(
            "Относительное преимущество лучшего нормированного расстояния перед вторым. "
            "Если преимущество меньше порога, точка остаётся ambiguous, а не назначается силой."
        )
        layout.addRow("Допуск X/Y:", xy_tolerance)
        layout.addRow("Допуск θ:", theta_tolerance)
        layout.addRow("Порог однозначности:", margin)

        explanation = QLabel(
            "Сначала для каждой пары выполняется независимая бинарная 3D-кросс-корреляция "
            "в координатах (theta_index, y, x), затем — направленный поиск ближайших "
            "пиков и уточнение по однозначно назначенному ядру. Одна исходная ветвь может "
            "разделиться между несколькими партнёрами. Результат содержит отдельные классы "
            "partner / ambiguous / residual и таблицу всех кандидатов. Исходный каталог "
            "пиков и результат определения ветвей не изменяются."
        )
        explanation.setWordWrap(True)
        explanation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        explanation.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addRow(explanation)

        self._forms["peaks.branch_correlation"].update(
            source_branch_id=source,
            partner_branch_ids=partners,
            xy_tolerance_px=xy_tolerance,
            theta_tolerance_arcsec=theta_tolerance,
            assignment_margin=margin,
        )

    def _build_hkl_identification_form(self) -> None:
        layout = self._add_form_page(
            "theory.identify_hkl", "Теоретическая идентификация HKL по Kα-дублету"
        )
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        from roc_mcs.theory.crystals import (
            available_crystal_models, get_crystal_model, xraylib_available,
        )
        crystal_selector = QComboBox()
        try:
            crystal_models = available_crystal_models()
            xraylib_ready = bool(xraylib_available())
        except Exception:
            # A broken optional XrayLib installation must not remove the
            # established CaF2 analysis path from the workbench.
            crystal_models = (("caf2_fluorite", "CaF₂ · флюорит"),)
            xraylib_ready = False
        for model_id, display_name in crystal_models:
            crystal_selector.addItem(display_name, model_id)
        default_index = crystal_selector.findData("caf2_fluorite")
        crystal_selector.setCurrentIndex(max(0, default_index))
        crystal_selector.setToolTip(
            "CaF₂ использует встроенную модель Fm-3m. Параметры остальных материалов "
            "читаются из выбранной записи базы XrayLib и не редактируются здесь."
        )
        layout.addRow("Кристаллическая модель:", crystal_selector)

        # The active page makes the provenance of the unit-cell constants
        # explicit: CaF2's cubic a is editable; XrayLib constants are database-owned.
        unit_cell_stack = QStackedWidget()
        unit_cell_stack.setObjectName("hklUnitCellEditor")
        builtin_cell_page = QWidget(unit_cell_stack)
        builtin_cell_row = QHBoxLayout(builtin_cell_page)
        builtin_cell_row.setContentsMargins(0, 0, 0, 0)
        builtin_cell_row.setSpacing(5)
        lattice = self._float_spin(0.1, 100.0, 5.4635, 6)
        lattice.setSuffix(" Å")
        lattice.setToolTip(
            "Параметр a кубической решётки встроенной модели CaF₂; "
            "для неё a=b=c и α=β=γ=90°."
        )
        lattice_caption = QLabel("a =")
        lattice_caption.setObjectName("hklInlineFieldCaption")
        builtin_cell_row.addWidget(lattice_caption)
        builtin_cell_row.addWidget(lattice)
        builtin_cell_row.addStretch(1)
        database_cell_page = QWidget(unit_cell_stack)
        database_cell_layout = QHBoxLayout(database_cell_page)
        database_cell_layout.setContentsMargins(0, 0, 0, 0)
        database_cell_layout.setSpacing(0)
        database_cell_label = QLabel("Параметры ячейки загружаются из XrayLib")
        database_cell_label.setObjectName("hklDatabaseCell")
        database_cell_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        database_cell_label.setWordWrap(False)
        database_cell_layout.addWidget(database_cell_label)
        unit_cell_stack.addWidget(builtin_cell_page)
        unit_cell_stack.addWidget(database_cell_page)
        layout.addRow("Параметры ячейки:", unit_cell_stack)

        def update_crystal_controls(_index: int = -1) -> None:
            model_id = str(crystal_selector.currentData() or "caf2_fluorite")
            if model_id == "caf2_fluorite":
                unit_cell_stack.setCurrentWidget(builtin_cell_page)
                lattice.setEnabled(True)
                return
            try:
                model = get_crystal_model(model_id)
                cell = model.unit_cell_parameters_A
                angles = model.unit_cell_angles_deg
                database_cell_label.setText(
                    f"a={float(cell['a']):.6g}, b={float(cell['b']):.6g}, "
                    f"c={float(cell['c']):.6g} Å   ·   "
                    f"α={float(angles['alpha']):.5g}°, β={float(angles['beta']):.5g}°, "
                    f"γ={float(angles['gamma']):.5g}°   ·   XrayLib"
                )
                # Do not overwrite the independent editable CaF2 value when
                # switching to a database crystal. The validator takes the
                # database cell as authoritative for this model.
            except Exception as exc:
                database_cell_label.setText(f"Не удалось прочитать параметры ячейки: {exc}")
            unit_cell_stack.setCurrentWidget(database_cell_page)
            lattice.setEnabled(False)

        crystal_selector.currentIndexChanged.connect(update_crystal_controls)
        update_crystal_controls()

        lambda_1 = self._float_spin(0.1, 10.0, 0.7093000, 7)
        lambda_1.setSuffix(" Å")
        lambda_2 = self._float_spin(0.1, 10.0, 0.7135900, 7)
        lambda_2.setSuffix(" Å")
        wavelength_row = QWidget()
        wavelength_layout = QHBoxLayout(wavelength_row)
        wavelength_layout.setContentsMargins(0, 0, 0, 0)
        wavelength_layout.setSpacing(8)
        for caption, control in (("Kα₁", lambda_1), ("Kα₂", lambda_2)):
            field_group = QWidget(wavelength_row)
            field_layout = QHBoxLayout(field_group)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(3)
            field_label = QLabel(caption, field_group)
            field_label.setObjectName("hklInlineFieldCaption")
            field_layout.addWidget(field_label)
            field_layout.addWidget(control)
            wavelength_layout.addWidget(field_group)
        wavelength_layout.addStretch(1)
        layout.addRow("Длины волн:", wavelength_row)

        max_index = self._int_spin(1, 30, 10)
        max_index.setToolTip(
            "Максимум |h|, |k|, |l|. Для XrayLib перебирается полный знаковый "
            "куб индексов; систематические погасания проверяются по F_H."
        )
        layout.addRow("Максимальный индекс:", max_index)

        include_crystalpy = QCheckBox("Рассчитать структурные факторы и Darwin widths")
        try:
            from roc_mcs.theory.crystalpy_caf2 import crystalpy_available
            crystalpy_ready = bool(crystalpy_available())
        except Exception:
            crystalpy_ready = False
        include_crystalpy.setChecked(crystalpy_ready)
        include_crystalpy.setEnabled(crystalpy_ready)
        thickness = self._float_spin(1e-8, 1e-2, 1e-5, 8)
        thickness.setSuffix(" m")
        thickness.setEnabled(crystalpy_ready)
        thickness.setToolTip(
            "Толщина кристалла для дополнительного расчёта динамической дифракции CrystalPy."
        )
        dynamic_row = QWidget()
        dynamic_layout = QHBoxLayout(dynamic_row)
        dynamic_layout.setContentsMargins(0, 0, 0, 0)
        dynamic_layout.setSpacing(7)
        dynamic_layout.addWidget(include_crystalpy)
        dynamic_layout.addWidget(QLabel("Толщина:", dynamic_row))
        dynamic_layout.addWidget(thickness)
        dynamic_layout.addStretch(1)
        layout.addRow("Динамическая теория:", dynamic_row)
        include_crystalpy.setToolTip(
            "Дополнительный расчёт F0/FH/FH̄, ψH/ψH̄ и Darwin widths через CrystalPy/XrayLib. "
            "Для CaF₂ используется явная basis-модель; для остальных материалов — "
            "атомная ячейка выбранной записи XrayLib."
        )
        if crystalpy_ready:
            dependency_message = (
                "XrayLib и CrystalPy доступны. Для других кристаллов используются "
                "параметры ячейки из XrayLib; дополнительные σ/π Darwin widths доступны."
            )
        elif xraylib_ready:
            dependency_message = (
                "XrayLib доступен для базы материалов и F_H; CrystalPy не обнаружен, "
                "поэтому дополнительные σ/π Darwin widths отключены."
            )
        else:
            dependency_message = (
                "Без XrayLib доступна встроенная геометрическая модель CaF₂. "
                "Для каталога других кристаллов установите xraylib; для Darwin widths "
                "дополнительно требуется crystalpy."
            )
        dependency_hint = QLabel(dependency_message)
        dependency_hint.setWordWrap(True)
        dependency_hint.setObjectName("hklDependencyHint")
        layout.addRow(dependency_hint)

        explanation = QLabel(
            "Измеренное |Δθ| восстанавливается из исходных θ по точным peak_id в "
            "branch_catalog; остатки после совмещения не используются. Теоретические "
            "Δθ ранжируются по абсолютному отклонению от медианы измерений. Ранг — "
            "геометрическое совпадение, не вероятность и не оценка интенсивности."
        )
        explanation.setWordWrap(True)
        explanation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        explanation.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        explanation.setObjectName("hklFormExplanation")
        layout.addRow(explanation)
        self._forms["theory.identify_hkl"].update(
            crystal_model_id=crystal_selector, lattice_a_A=lattice,
            lambda_ka1_A=lambda_1, lambda_ka2_A=lambda_2,
            max_miller_index=max_index,
            include_crystalpy=include_crystalpy, crystal_thickness_m=thickness,
            unit_cell_stack=unit_cell_stack,
        )

    def _request_branch_stats(self, parent_id: str) -> None:
        node = self.project.history.get(parent_id)
        form = self._forms.get("peaks.branches", {})
        label = form.get("stats_label")
        table = form.get("stats_table")
        preview = form.get("preview")
        if preview is not None:
            preview.clear_catalog("Загрузка реального фрагмента каталога пиков…")
        if table is not None:
            # Never leave another scan's diagnostic rows visible while this
            # parent's statistics are being computed or if loading fails.
            table.setRowCount(0)
        output_name = "fit" if node.operation_id == "fit.scan" else "peaks"
        artifact_id = node.output_refs.get(output_name)
        if not artifact_id:
            if label is not None:
                label.setText("У выбранного родителя нет сохранённого каталога пиков.")
            if preview is not None:
                preview.clear_catalog("У выбранного родителя нет каталога пиков для иллюстрации.")
            return
        task = _BranchStatsTask(
            parent_id, node.operation_id, artifact_id,
            str(self.project.artifact_store.directory),
        )
        task.signals.succeeded.connect(
            lambda pid, stats, source_task=task:
            self._on_branch_stats_ready(pid, stats, source_task)
        )
        task.signals.failed.connect(
            lambda pid, error, source_task=task:
            self._on_branch_stats_failed(pid, error, source_task)
        )
        self._active_branch_stats_task = task
        self._active_branch_stats_parent_id = str(parent_id)
        if label is not None:
            label.setText("Вычисляется статистика θ для выбранной ветви…")
        self.thread_pool.start(task)

    def _on_branch_stats_ready(
        self, parent_id: str, stats: dict[str, Any], source_task: _BranchStatsTask | None = None
    ) -> None:
        # Ignore a late result from a previous request (including a reopened
        # popup for the same parent). A slow read must never overwrite current UI.
        if (
            str(parent_id) != self._active_branch_stats_parent_id
            or (source_task is not None and source_task is not self._active_branch_stats_task)
        ):
            return
        self._active_branch_stats_task = None
        form = self._forms.get("peaks.branches", {})
        label = form.get("stats_label")
        table = form.get("stats_table")
        if label is None or table is None:
            return
        result = stats if isinstance(stats, dict) else {}
        preview = form.get("preview")
        if "stats" in result:
            try:
                import pandas as pd
                preview_rows = result.get("preview_rows", ())
                preview_catalog = pd.DataFrame(
                    preview_rows, columns=("peak_id", "x", "y", "theta")
                )
                source_label = str(result.get("position_label", "пики"))
                preview_info = dict(result.get("preview_info", {}))
                preview_info["position_label"] = source_label
                if preview is not None:
                    preview.set_catalog(preview_catalog, preview_info)
            except Exception as exc:
                if preview is not None:
                    preview.clear_catalog(
                        f"Не удалось подготовить иллюстрацию: {type(exc).__name__}: {exc}"
                    )
            stats = result["stats"]
        q90 = stats.get("q90_neighbor_delta")
        minimum = stats.get("min_internal_gap")
        suggested = float(stats.get("recommended_theta_gate", 60.0))
        q90_text = "н/д" if q90 is None else f"{q90:.2f}″"
        min_text = "н/д" if minimum is None else f"{minimum:.2f}″"
        edge_count = int(stats.get("neighbor_edge_count_reference_gate", 0))
        reference_gate = float(stats.get("reference_theta_gate", 60.0))
        label.setText(
            f"Q90 Δθ: {q90_text} ({edge_count:,} связей при {reference_gate:g}″)  ·  "
            f"min internal: {min_text}  ·  старт: {suggested:.2f}″"
        )
        label.setToolTip(str(stats.get("recommendation_status", "")))
        rows = list(stats.get("internal_gap_by_n", ()))
        table.setRowCount(len(rows))
        column_names = ("n_peaks", "pixel_count", "median_gap", "q25_gap", "q75_gap", "min_pixel_median_gap")
        unit_hint = "Угловые секунды (единицы θ текущего скана)."
        for row_index, row in enumerate(rows):
            values = (
                str(row["n_peaks"]), str(row["pixel_count"]),
                f"{row['median_gap']:.1f}", f"{row['q25_gap']:.1f}",
                f"{row['q75_gap']:.1f}", f"{row['min_pixel_median_gap']:.1f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip(f"{value}; {unit_hint} {table.horizontalHeaderItem(column).toolTip()}")
                table.setItem(row_index, column, item)
        table.resizeColumnsToContents()
        # Keep a separate working gate for each parent branch. A suggestion for
        # scan B must not inherit the manually tuned value from scan A.
        spin = form.get("theta_gate")
        if spin is not None:
            saved = self._branch_gate_values.get(str(parent_id))
            if saved is None:
                current = float(spin.value())
                saved = suggested if abs(current - 60.0) < 1e-9 else current
                self._branch_gate_values[str(parent_id)] = float(saved)
            spin.setValue(min(spin.maximum(), max(spin.minimum(), float(saved))))

    def _on_branch_stats_failed(
        self, parent_id: str, error: str, source_task: _BranchStatsTask | None = None
    ) -> None:
        if (
            str(parent_id) != self._active_branch_stats_parent_id
            or (source_task is not None and source_task is not self._active_branch_stats_task)
        ):
            return
        self._active_branch_stats_task = None
        form = self._forms.get("peaks.branches", {})
        label = form.get("stats_label")
        table = form.get("stats_table")
        preview = form.get("preview")
        if table is not None:
            table.setRowCount(0)
        if preview is not None:
            preview.clear_catalog("Не удалось загрузить каталог из выбранной ветви workflow.")
        if label is not None:
            label.setText("Не удалось вычислить статистику θ; доступно исходное значение 60″.")
            label.setToolTip(error)

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
        if operation_id == "peaks.branches":
            return {"theta_gate": form["theta_gate"].value()}
        if operation_id == "peaks.branch_correlation":
            return {
                "source_branch_id": form["source_branch_id"].text().strip(),
                "partner_branch_ids": form["partner_branch_ids"].text().strip(),
                "xy_tolerance_px": form["xy_tolerance_px"].value(),
                "theta_tolerance_arcsec": form["theta_tolerance_arcsec"].value(),
                "assignment_margin": form["assignment_margin"].value(),
            }
        if operation_id == "theory.identify_hkl":
            return {
                "crystal_model_id": str(form["crystal_model_id"].currentData()),
                "lattice_a_A": form["lattice_a_A"].value(),
                "lambda_ka1_A": form["lambda_ka1_A"].value(),
                "lambda_ka2_A": form["lambda_ka2_A"].value(),
                "max_miller_index": form["max_miller_index"].value(),
                "include_crystalpy": form["include_crystalpy"].isChecked(),
                "crystal_thickness_m": form["crystal_thickness_m"].value(),
            }
        raise KeyError(operation_id)

    def _selected_node_id(self) -> str | None:
        return self.graph.selected_node_id

    def _branch_count_for_node(self, node) -> tuple[int | None, str | None]:
        """Read the branch count from immutable artifact metadata without loading peak rows."""
        artifact_id = node.output_refs.get("branches")
        if not artifact_id:
            return None, "У результата определения ветвей отсутствует выходной артефакт branches."
        cached = self._branch_count_cache.get(str(artifact_id))
        if cached is not None:
            return cached
        try:
            metadata = self.project.artifact_store.read_metadata(str(artifact_id))
            raw_count = metadata.get("n_branches")
            if isinstance(raw_count, bool):
                raise ValueError("n_branches содержит bool вместо целого числа")
            count_float = float(raw_count)
            if not np.isfinite(count_float) or not count_float.is_integer() or count_float < 0:
                raise ValueError(f"Некорректное значение n_branches: {raw_count!r}")
            result = (int(count_float), None)
        except Exception as exc:
            result = (
                None,
                "Не удалось проверить число ветвей в артефакте. "
                "Повторно выполните этап определения ветвей в актуальной версии. "
                f"Причина: {type(exc).__name__}: {exc}",
            )
        self._branch_count_cache[str(artifact_id)] = result
        return result

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
            return ["fit.scan", "peaks.branches"]
        if node.operation_id == "fit.scan":
            return ["peaks.branches"]
        if node.operation_id == "peaks.branches":
            branch_count, _reason = self._branch_count_for_node(node)
            return ["peaks.branch_correlation"] if branch_count is not None and branch_count >= 2 else []
        if node.operation_id == "peaks.branch_correlation":
            return ["theory.identify_hkl"] if "branch_correlation" in node.output_refs else []
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
                if parent.operation_id == "peaks.branches":
                    branch_count, reason = self._branch_count_for_node(parent)
                    if branch_count is not None and branch_count < 2:
                        message = (
                            f"В результате обнаружена только {branch_count} дифракционная ветвь. "
                            "3D-кросс-корреляция требует как минимум двух различных branch_id: "
                            "одну опорную ветвь и хотя бы одну ветвь-партнёр."
                        )
                    else:
                        message = reason or (
                            "3D-кросс-корреляция недоступна для этого результата. "
                            "Проверьте число ветвей и метаданные артефакта."
                        )
                else:
                    message = (
                        "Для выбранного узла нет доступных следующих операций. "
                        "Проверьте статус результата и его выходные данные."
                    )
                QMessageBox.information(self, "Новая ветвь", message)
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
            apply_popup_theme(menu, self._theme)
            menu.exec(QCursor.pos())
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось добавить ветвь", str(exc))

    def _open_stage_configuration(
        self, parent_id: str, operation_id: str, *, existing_node_id: str | None = None
    ) -> None:
        """Open the canonical stage form, optionally as a read-only result view."""
        if self.is_busy:
            return
        parent = self.project.history.get(parent_id)
        operation_ids = self._available_operations_for_parent(parent_id)
        if existing_node_id is None and operation_id not in operation_ids:
            QMessageBox.warning(
                self, "Недоступный этап",
                "Выбранную операцию нельзя создать от этого узла. "
                "Проверьте состояние и принятые результаты ветви.",
            )
            return
        if self._active_stage_popup is not None:
            self._active_stage_popup.close()

        self._branch_parent_id = parent_id
        if operation_id == "peaks.branches":
            gate_control = self._forms.get("peaks.branches", {}).get("theta_gate")
            if gate_control is not None:
                gate_control.setValue(float(self._branch_gate_values.get(str(parent_id), 60.0)))
        combo_index = self.operation_combo.findData(operation_id)
        if combo_index < 0:
            return
        self.operation_combo.setCurrentIndex(combo_index)
        form_info = self._forms.get(operation_id)
        if form_info is None:
            return

        spec = self.registry.get(operation_id)
        page = form_info["page"]
        popup = _StageConfigurationPopup(self)
        popup.setObjectName("processingStagePopup")
        root = QVBoxLayout(popup)
        root.setContentsMargins(8, 5, 8, 6)
        root.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(1, 0, 1, 0)
        header.setSpacing(6)
        title = _PopupDragTitle(
            f"<b>{html_escape(spec.label)}</b>"
            f"  ·  вход: {html_escape(parent.label)}",
            popup,
        )
        title.setTextFormat(Qt.TextFormat.RichText)
        if existing_node_id is not None:
            saved_node = self.project.history.get(existing_node_id)
            title.setText(
                f"<b>{html_escape(saved_node.label)}</b>"
                "  ·  просмотр сохранённого результата"
            )
        title.setWordWrap(False)
        title.setToolTip("Перетащите за эту строку, чтобы переместить окно")
        header.addWidget(title, 1)
        close_button = QToolButton(popup)
        close_button.setObjectName("processingStageCloseButton")
        close_button.setText("×")
        close_button.setFixedSize(20, 20)
        close_button.setToolTip("Закрыть окно параметров")
        close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_button.clicked.connect(popup.close)
        header.addWidget(close_button, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)

        scroll = QScrollArea(popup)
        scroll.setObjectName("processingStageFormScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(0)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Re-use the canonical parameter widgets: no parallel form, no copied
        # defaults, and no second parameter-to-operation mapping.
        original_page_minimum_width = page.minimumWidth()
        self.form_stack.removeWidget(page)
        # QStackedWidget hides the page on removal. It is reparented directly
        # into the destination container below, without an intermediate
        # parentless/top-level state. Explicit show() is repeated after popup
        # activation to make the Qt visibility transition deterministic.
        page.setMinimumWidth(0)
        page.setEnabled(existing_node_id is None)
        is_hkl_stage = operation_id == "theory.identify_hkl"
        original_control_metrics = []
        original_layout_metrics = None
        if is_hkl_stage:
            for control_type in (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit):
                for control in page.findChildren(control_type):
                    symbols = (
                        control.buttonSymbols()
                        if isinstance(control, QAbstractSpinBox) else None
                    )
                    original_control_metrics.append((
                        control, control.minimumHeight(), control.maximumHeight(),
                        control.minimumWidth(), symbols,
                    ))
            original_layout = form_info.get("layout")
            if original_layout is not None:
                margins = original_layout.contentsMargins()
                original_layout_metrics = (
                    margins.left(), margins.top(), margins.right(), margins.bottom(),
                    original_layout.verticalSpacing(), original_layout.horizontalSpacing(),
                )
        stage_content = None
        hkl_panel = None
        hkl_status = None
        if is_hkl_stage:
            stage_content = QWidget(scroll)
            stage_content.setObjectName("hklStageContent")
            stage_content_layout = QVBoxLayout(stage_content)
            stage_content_layout.setContentsMargins(2, 2, 2, 2)
            stage_content_layout.setSpacing(4)
            page.setParent(stage_content)
            stage_content_layout.addWidget(page)
            page.setVisible(True)
            hkl_status = QLabel(
                "Результат появится здесь после выполнения этапа.", stage_content
            )
            hkl_status.setObjectName("hklStageStatus")
            hkl_status.setWordWrap(True)
            stage_content_layout.addWidget(hkl_status)
            hkl_panel = HklResultsPanel(stage_content)
            hkl_panel.apply_theme(self._theme)
            hkl_panel.hide()
            stage_content_layout.addWidget(hkl_panel)
            scroll.setWidget(stage_content)
            stage_content.show()
        else:
            scroll.setWidget(page)
            page.show()

        for widget_type in (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit):
            for field in page.findChildren(widget_type):
                compact_peak_control = (
                    operation_id == "peaks.detect"
                    and isinstance(field, QAbstractSpinBox)
                )
                compact_hkl_control = is_hkl_stage
                field.setMinimumHeight(18 if compact_peak_control or compact_hkl_control else 21)
                field.setMaximumHeight(21 if compact_peak_control or compact_hkl_control else 25)
                if (
                    (compact_peak_control or compact_hkl_control)
                    and isinstance(field, QAbstractSpinBox)
                ):
                    field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
                if isinstance(field, QDoubleSpinBox):
                    field.setMinimumWidth(82 if is_hkl_stage else 78)
                elif isinstance(field, QSpinBox):
                    field.setMinimumWidth(68)
                elif isinstance(field, QComboBox):
                    field.setMinimumWidth(112 if is_hkl_stage else 112)
        page_layout = form_info.get("layout")
        if page_layout is not None:
            compact_peak_form = operation_id == "peaks.detect"
            compact_hkl_form = is_hkl_stage
            margin = 2 if (compact_peak_form or compact_hkl_form) else 4
            page_layout.setContentsMargins(margin, margin, margin, margin)
            page_layout.setVerticalSpacing(2 if compact_hkl_form else (3 if compact_peak_form else 5))
            page_layout.setHorizontalSpacing(6 if compact_hkl_form else (7 if compact_peak_form else 8))
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 1, 0, 0)
        footer.setSpacing(6)
        footer.addStretch(1)
        reset_button = None
        if operation_id == "peaks.detect":
            reset_button = QToolButton(popup)
            reset_button.setText("↺")
            reset_button.setAutoRaise(True)
            reset_button.setFixedSize(25, 24)
            reset_button.setToolTip(
                "Восстановить заводские значения параметров поиска пиков. "
                "Уже добавленные этапы workflow не изменяются."
            )
            reset_button.setAccessibleName("Сбросить параметры поиска пиков")
            reset_button.clicked.connect(form_info["reset_parameters"])
            footer.addWidget(reset_button)
        cancel_button = QPushButton("Отмена", popup)
        run_button = QPushButton("Создать и запустить", popup)
        popup._stage_run_button = run_button
        popup._stage_cancel_button = cancel_button
        run_button.setDefault(True)
        for button in (cancel_button, run_button):
            button.setMinimumHeight(24)
        footer.addWidget(cancel_button)
        footer.addWidget(run_button)
        root.addLayout(footer)

        apply_popup_theme(popup, self._theme)
        apply_widget_palette(popup, self._theme)
        compact_form = operation_id in {"peaks.detect", "theory.identify_hkl"}
        spin_min_height = 18 if compact_form else 21
        spin_max_height = 21 if compact_form else 25
        popup_colors = theme_colors(self._theme)
        popup.setStyleSheet(
            "QFrame#processingStagePopup { background:palette(window); color:palette(window-text); "
            "border:1px solid palette(mid); border-radius:5px; }"
            "QFrame#processingStagePopup QLabel { background:transparent; color:palette(window-text); font-size:9px; }"
            f"QFrame#processingStagePopup QLabel#hklStageStatus {{ color:{popup_colors['muted']}; font-size:9px; padding:3px 2px; }}"
            "QFrame#processingStagePopup QSpinBox, QFrame#processingStagePopup QDoubleSpinBox { "
            f"font-size:10px; min-height:{spin_min_height}px; max-height:{spin_max_height}px; padding:0px 3px; "
            "background:palette(base); color:palette(text); border:1px solid palette(mid); border-radius:3px; }"
            "QFrame#processingStagePopup QComboBox, QFrame#processingStagePopup QLineEdit { "
            f"font-size:9px; min-height:{18 if compact_form else 21}px; max-height:{21 if compact_form else 25}px; padding:1px 3px; "
            "background:palette(base); color:palette(text); border:1px solid palette(mid); border-radius:3px; }"
            "QFrame#processingStagePopup QCheckBox { font-size:9px; spacing:4px; }"
            "QFrame#processingStagePopup QPushButton { font-size:9px; padding:3px 8px; "
            "background:palette(button); color:palette(button-text); border:1px solid palette(mid); border-radius:3px; }"
            "QFrame#processingStagePopup QPushButton:hover, QFrame#processingStagePopup QToolButton:hover { "
            "background:palette(midlight); }"
            "QToolButton#processingStageCloseButton { border:0; font-size:17px; padding:0; }"
        )

        # A bounded, adaptive height keeps all controls reachable on smaller
        # displays. The form itself scrolls vertically rather than being clipped.
        cursor_pos = QCursor.pos()
        screen = QApplication.screenAt(cursor_pos) or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else self.frameGeometry()
        if is_hkl_stage:
            popup_width = min(640, max(320, available.width() - 16))
        else:
            popup_width = max(340, min(440, available.width() - 16))
        popup.setFixedWidth(popup_width)
        content_height = max(140, (stage_content.sizeHint().height() if stage_content is not None else page.sizeHint().height()) + 8)
        max_popup_height = max(190, int(available.height() * (0.89 if is_hkl_stage else 0.82)))
        desired_height = min(content_height + 78, max_popup_height)
        if is_hkl_stage:
            # Keep the existing 640 px width cap intact; only give the result
            # panel a materially taller vertical viewport.
            desired_height = max(min(390, max_popup_height), desired_height)
        popup.setMaximumHeight(max_popup_height)
        scroll.setMinimumHeight(min(max(115, content_height), max(115, desired_height - 82)))
        popup.resize(popup_width, desired_height)
        popup.adjustSize()
        x = min(max(cursor_pos.x(), available.left() + 4), available.right() - popup.width() + 1)
        y = min(max(cursor_pos.y(), available.top() + 4), available.bottom() - popup.height() + 1)
        popup.move(x, y)

        self._active_stage_popup = popup
        self._active_stage_operation = operation_id
        popup._stage_form_page = page
        popup._stage_form_scroll = scroll
        popup._stage_content = stage_content
        popup._stage_original_page_minimum_width = original_page_minimum_width
        popup._stage_original_control_metrics = original_control_metrics
        popup._stage_original_layout_metrics = original_layout_metrics
        popup._stage_hkl_panel = hkl_panel
        popup._stage_hkl_status = hkl_status
        popup._stage_read_only = existing_node_id is not None
        popup._stage_temporary_model_item_index = None
        popup._stage_node_id = existing_node_id
        popup._stage_parent_id = parent_id

        def restore_page() -> None:
            if getattr(popup, "_stage_form_page_restored", False):
                return
            popup._stage_form_page_restored = True
            if operation_id == "peaks.branches":
                gate_control = form_info.get("theta_gate")
                if gate_control is not None:
                    self._branch_gate_values[str(parent_id)] = float(gate_control.value())
            if scroll.widget() is page or (stage_content is not None and scroll.widget() is stage_content):
                scroll.takeWidget()
            page.setParent(self.form_stack)
            page.setEnabled(True)
            if is_hkl_stage:
                temporary_model_index = getattr(popup, "_stage_temporary_model_item_index", None)
                if temporary_model_index is not None:
                    form_info["crystal_model_id"].removeItem(temporary_model_index)
                    popup._stage_temporary_model_item_index = None
                page.setMinimumWidth(original_page_minimum_width)
                for control, minimum_height, maximum_height, minimum_width, symbols in original_control_metrics:
                    control.setMinimumHeight(minimum_height)
                    control.setMaximumHeight(maximum_height)
                    control.setMinimumWidth(minimum_width)
                    if symbols is not None:
                        control.setButtonSymbols(symbols)
                if page_layout is not None and original_layout_metrics is not None:
                    left, top, right, bottom, vertical, horizontal = original_layout_metrics
                    page_layout.setContentsMargins(left, top, right, bottom)
                    page_layout.setVerticalSpacing(vertical)
                    page_layout.setHorizontalSpacing(horizontal)
            if stage_content is not None:
                stage_content.deleteLater()
            if self.form_stack.indexOf(page) < 0:
                self.form_stack.addWidget(page)
            page.show()
            if self._active_stage_popup is popup:
                self._active_stage_popup = None
                self._active_stage_operation = None
            if operation_id == "peaks.detect":
                self._pending_peak_map_parent_id = None
                self._pending_peak_pixel_coordinates = None
                if self._active_peak_input_task is not None:
                    self._discard_active_peak_input = True
                pick_button = form_info.get("pick_button")
                if pick_button is not None:
                    was_blocked = pick_button.blockSignals(True)
                    pick_button.setChecked(False)
                    pick_button.blockSignals(was_blocked)
                self.pixel_pick_mode_requested.emit(False, parent_id, False)
            self._show_operation_form()

        if existing_node_id is not None:
            saved_node = self.project.history.get(existing_node_id)
            saved_parameters = dict(saved_node.parameters or {})
            crystal_combo = form_info["crystal_model_id"]
            saved_model_id = str(saved_parameters.get("crystal_model_id", "caf2_fluorite"))
            saved_index = crystal_combo.findData(saved_model_id)
            if saved_index < 0:
                saved_label = str(saved_parameters.get("crystal_model_label", saved_model_id))
                crystal_combo.addItem(f"{saved_label} · модель из сохранённого узла", saved_model_id)
                popup._stage_temporary_model_item_index = crystal_combo.count() - 1
                saved_index = crystal_combo.findData(saved_model_id)
            crystal_combo.setCurrentIndex(saved_index)
            if saved_model_id == "caf2_fluorite":
                form_info["lattice_a_A"].setValue(
                    float(saved_parameters.get("lattice_a_A", 5.4635))
                )
            form_info["lambda_ka1_A"].setValue(float(saved_parameters.get("lambda_ka1_A", 0.7093000)))
            form_info["lambda_ka2_A"].setValue(float(saved_parameters.get("lambda_ka2_A", 0.7135900)))
            form_info["max_miller_index"].setValue(int(saved_parameters.get("max_miller_index", 10)))
            form_info["include_crystalpy"].setChecked(bool(saved_parameters.get("include_crystalpy", False)))
            form_info["crystal_thickness_m"].setValue(float(saved_parameters.get("crystal_thickness_m", 1e-5)))
            page.setEnabled(False)
            run_button.hide()
            cancel_button.setText("Закрыть")
            if hkl_status is not None:
                hkl_status.setText("Загрузка сохранённого HKL-артефакта…")

        popup.about_to_close.connect(restore_page)
        cancel_button.clicked.connect(popup.close)

        def validate_create_and_run() -> None:
            try:
                parameters = self._form_parameters(operation_id)
                spec.validate(parameters)
            except Exception as exc:
                QMessageBox.warning(popup, "Проверьте параметры этапа", str(exc))
                return
            try:
                if is_hkl_stage:
                    if hkl_status is not None:
                        hkl_status.setText("Выполняется теоретический расчёт HKL…")
                    if page is not None:
                        page.setEnabled(False)
                    run_button.setEnabled(False)
                    run_button.setText("Выполняется…")
                    cancel_button.setText("Закрыть")
                    node_id = self._create_stage_candidate(
                        parent_id, operation_id, parameters, run_immediately=True
                    )
                    popup._stage_node_id = node_id
                    if node_id:
                        self.graph.select_node(node_id)
                    else:
                        raise RuntimeError("Не удалось создать узел HKL-этапа")
                else:
                    popup.close()
                    node_id = self._create_stage_candidate(
                        parent_id, operation_id, parameters, run_immediately=True
                    )
                    if node_id:
                        self.graph.select_node(node_id)
            except Exception as exc:
                if is_hkl_stage:
                    if page is not None:
                        page.setEnabled(True)
                    run_button.setEnabled(True)
                    run_button.setText("Создать и запустить")
                    if hkl_status is not None:
                        hkl_status.setText(f"Не удалось запустить этап: {exc}")
                else:
                    QMessageBox.warning(self, "Не удалось создать этап", str(exc))

        run_button.clicked.connect(validate_create_and_run)
        popup.show()
        if is_hkl_stage:
            # QStackedWidget::removeWidget hides pages. Repair parent/layout if
            # necessary and re-activate visibility after Qt completes the
            # popup's first layout pass. This avoids a blank HKL form on reopen.
            def ensure_hkl_form_visible() -> None:
                if (
                    stage_content is None
                    or self._active_stage_popup is not popup
                    or not popup.isVisible()
                ):
                    return
                if page.parentWidget() is not stage_content:
                    page.setParent(stage_content)
                    stage_content_layout.insertWidget(0, page)
                page.setVisible(True)
                page.updateGeometry()
                stage_content_layout.activate()
                stage_content.updateGeometry()
                scroll.updateGeometry()

            QTimer.singleShot(0, ensure_hkl_form_visible)
            if page.layout() is None or page.layout().count() == 0:
                self._log("HKL form page has no layout rows after reparenting")
        popup.raise_()
        popup.activateWindow()

        if operation_id == "peaks.detect":
            preview = form_info.get("preview")
            if preview is not None:
                preview.clear_real_input()
            self._ensure_peak_parent_preview(parent_id)
        elif operation_id == "peaks.branches":
            self._request_branch_stats(parent_id)

    def _ensure_peak_parent_preview(self, parent_id: str) -> None:
        """Ensure the host 2D map displays the exact selected branch input."""
        form_info = self._forms.get("peaks.detect", {})
        pick_button = form_info.get("pick_button")
        if pick_button is None or not self.embedded:
            return
        pick_button.setEnabled(False)
        pick_button.setToolTip("Подготавливается 2D-карта входной ветви…")
        current_node = None
        if self._preview_context is not None:
            current_node = self._preview_context.get("node")
        if current_node is not None and current_node.id == parent_id:
            self._pending_peak_map_parent_id = None
            pick_button.setEnabled(True)
            pick_button.setToolTip(
                "Режим выбора пикселя: щёлкайте по 2D-карте, чтобы менять реальную ROC. "
                "Выключите кнопку, чтобы вернуться к синтетической схеме."
            )
            return
        self._pending_peak_map_parent_id = parent_id
        if self._active_preview_task is None:
            self._start_preview_for_node(parent_id, switch_tab=False, show_error_dialog=False)

    def _request_peak_pixel_pick(self, enabled: bool) -> None:
        parent_id = self._branch_parent_id or ""
        if enabled:
            if not self.embedded or self._active_stage_operation != "peaks.detect":
                form = self._forms.get("peaks.detect", {})
                button = form.get("pick_button")
                if button is not None:
                    button.setChecked(False)
                return
            if self._pending_peak_map_parent_id is not None:
                self.status_label.setText("Подождите, пока обновится входная 2D-карта ветви")
                button = self._forms["peaks.detect"].get("pick_button")
                if button is not None:
                    button.setChecked(False)
                return
        use_interpolated = bool(
            self._forms.get("peaks.detect", {}).get("use_interpolated_data")
            and self._forms["peaks.detect"]["use_interpolated_data"].isChecked()
        )
        if not enabled:
            self._pending_peak_pixel_coordinates = None
            if self._active_peak_input_task is not None:
                self._discard_active_peak_input = True
            preview = self._forms.get("peaks.detect", {}).get("preview")
            if preview is not None:
                preview.clear_real_input()
        self.pixel_pick_mode_requested.emit(bool(enabled), parent_id, use_interpolated)

    def cancel_peak_pixel_pick_mode(self) -> None:
        """Uncheck the map-picker control after a host-side cancellation."""
        button = self._forms.get("peaks.detect", {}).get("pick_button")
        if button is not None and button.isChecked():
            button.setChecked(False)

    def set_peak_preview_pixel(self, detector_x: float, detector_y: float) -> None:
        """Queue the latest selected pixel from the exact input of this branch."""
        if self._active_stage_popup is None or self._active_stage_operation != "peaks.detect":
            return
        pick_button = self._forms.get("peaks.detect", {}).get("pick_button")
        if pick_button is None or not pick_button.isChecked():
            return
        self._pending_peak_pixel_coordinates = (float(detector_x), float(detector_y))
        if self._active_peak_input_task is not None:
            self.status_label.setText(
                f"Пиксель ({float(detector_x):.4g}, {float(detector_y):.4g}) выбран; "
                "после текущей загрузки будет показана последняя выбранная точка…"
            )
            return
        self._start_pending_peak_pixel_input()

    def _start_pending_peak_pixel_input(self) -> None:
        """Start loading the most recently requested real-pixel preview."""
        if self._active_peak_input_task is not None:
            return
        coordinates = self._pending_peak_pixel_coordinates
        if coordinates is None:
            return
        if self._active_stage_popup is None or self._active_stage_operation != "peaks.detect":
            self._pending_peak_pixel_coordinates = None
            return
        pick_button = self._forms.get("peaks.detect", {}).get("pick_button")
        if pick_button is None or not pick_button.isChecked():
            self._pending_peak_pixel_coordinates = None
            return
        parent_id = self._branch_parent_id
        if not parent_id:
            self._pending_peak_pixel_coordinates = None
            return
        self._pending_peak_pixel_coordinates = None
        detector_x, detector_y = coordinates
        try:
            task = _PeakPixelInputTask(
                parent_id,
                WorkflowHistory(list(self.project.history.nodes)),
                str(self.project.artifact_store.directory),
                detector_x, detector_y,
            )
            task.signals.succeeded.connect(self._on_peak_pixel_input_ready)
            task.signals.failed.connect(self._on_peak_pixel_input_failed)
            self._active_peak_input_task = task
            self._discard_active_peak_input = False
            self.status_label.setText(
                f"Загрузка ROC пикселя ({detector_x:.4g}, {detector_y:.4g})…"
            )
            self._log(
                f"Выбран detector-пиксель X={detector_x:.6g}, Y={detector_y:.6g}; "
                "загружаются точные Scan/Spike-входы ветви."
            )
            self.thread_pool.start(task)
        except Exception as exc:
            self._active_peak_input_task = None
            self.status_label.setText(f"Не удалось подготовить пиксель: {exc}")
            self._log(traceback.format_exc())

    def _on_peak_pixel_input_ready(self, parent_id: str, payload: dict[str, Any]) -> None:
        self._active_peak_input_task = None
        discard_result = self._discard_active_peak_input
        self._discard_active_peak_input = False
        if (
            self._active_stage_popup is None
            or self._active_stage_operation != "peaks.detect"
            or parent_id != self._branch_parent_id
        ):
            self._pending_peak_pixel_coordinates = None
            return
        # If the user selected another pixel while this payload was loading,
        # display the newest request directly rather than flashing stale data.
        if self._pending_peak_pixel_coordinates is not None:
            self._start_pending_peak_pixel_input()
            return
        if discard_result:
            return
        form_info = self._forms.get("peaks.detect", {})
        pick_button = form_info.get("pick_button")
        if pick_button is None or not pick_button.isChecked():
            return
        preview = form_info.get("preview")
        if preview is None:
            return
        preview.set_real_input(payload)
        if bool(payload.get("interpolation_available")):
            source_hint = "измеренная / интерполированная по переключателю формы"
        else:
            source_hint = "измеренная; Spike-вход в этой ветви отсутствует"
        self.status_label.setText(
            f"Реальная ROC: пиксель ({payload['pixel_x']}, {payload['pixel_y']}); {source_hint}"
        )
        self._log(
            f"Реальное превью обновлено для пикселя ({payload['pixel_x']}, {payload['pixel_y']}); "
            f"grid={payload.get('grid_id', '')}."
        )

    def _on_peak_pixel_input_failed(self, parent_id: str, error: str) -> None:
        self._active_peak_input_task = None
        self._discard_active_peak_input = False
        if parent_id != self._branch_parent_id:
            self._pending_peak_pixel_coordinates = None
            return
        if self._pending_peak_pixel_coordinates is not None:
            self._start_pending_peak_pixel_input()
            return
        pick_button = self._forms.get("peaks.detect", {}).get("pick_button")
        if pick_button is None or not pick_button.isChecked():
            return
        self.status_label.setText("Не удалось загрузить реальную ROC выбранного пикселя")
        self._log(error)

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
        # HKL has two independent result views: the compact table/fit panel
        # reads only its small artifact, while spatial Δθ layers are assembled
        # through the established full WorkflowPreviewState path. Keep both
        # available and do not switch away from the currently open stage popup.
        if node.operation_id == "theory.identify_hkl":
            popup = self._active_stage_popup
            if popup is not None and getattr(popup, "_stage_node_id", None) == node_id:
                self._start_hkl_result_preview(node_id)
            self._start_preview_for_node(node_id, switch_tab=False, show_error_dialog=False)
            return

        # Other operations retain the established generic Scan/ROI preview.
        self._start_preview_for_node(node_id, switch_tab=True, show_error_dialog=True)

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
            self.status_label.setText("Этап завершился с ошибкой — подробности сохранены")
            self._log(error)
            self._rebuild_graph(select_id=node_id)
            self._update_controls()
        if persistence_error is not None:
            QMessageBox.critical(self, "Ошибка сохранения истории", persistence_error)
            return

        # A failed workflow node must expose its real exception immediately.
        # The full traceback remains in node.error and in the execution log;
        # QMessageBox's details section makes it accessible without opening JSON.
        stage_popup = self._active_stage_popup
        if (
            stage_popup is not None
            and self._active_stage_operation == "theory.identify_hkl"
            and getattr(stage_popup, "_stage_node_id", None) == node_id
        ):
            stage_page = getattr(stage_popup, "_stage_form_page", None)
            stage_status = getattr(stage_popup, "_stage_hkl_status", None)
            stage_run_button = getattr(stage_popup, "_stage_run_button", None)
            stage_cancel_button = getattr(stage_popup, "_stage_cancel_button", None)
            if stage_page is not None:
                stage_page.setEnabled(True)
            if stage_status is not None:
                stage_status.setText(
                    "Расчёт не выполнен. Исправьте параметры и запустите новую попытку. "
                    "Полная диагностика сохранена в журнале workflow."
                )
            if stage_run_button is not None:
                stage_run_button.setText("Создать новую попытку")
                stage_run_button.setEnabled(True)
            if stage_cancel_button is not None:
                stage_cancel_button.setText("Закрыть")
        node = self.project.history.get(node_id)
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        cause = lines[-1] if lines else "Причина не была записана; откройте журнал выполнения."
        details = QMessageBox(self)
        details.setIcon(QMessageBox.Icon.Critical)
        details.setWindowTitle("Ошибка выполнения этапа")
        details.setText(f"Этап «{node.label}» не выполнен.")
        details.setInformativeText(
            f"{cause}\n\nПараметры запуска: {node.parameters}\n\n"
            "Ни исходный скан, ни принятый результат родительского этапа не изменены. "
            "Полный traceback находится в разделе «Подробности» и сохранён в истории workflow."
        )
        details.setDetailedText(str(error))
        copy_button = details.addButton(
            "Копировать traceback", QMessageBox.ButtonRole.ActionRole
        )
        details.addButton(QMessageBox.StandardButton.Ok)
        details.exec()
        if details.clickedButton() is copy_button:
            QApplication.clipboard().setText(str(error))

    def _rebuild_graph(self, *_args, select_id: str | None = None) -> None:
        if not hasattr(self, "graph"):
            return
        previous = select_id or self._selected_node_id()
        addable_node_ids: set[str] = set()
        for candidate in self.project.history.nodes:
            if (
                candidate.status == NodeStatus.SUCCEEDED
                and candidate.review_status == ReviewStatus.ACCEPTED
                and self._available_operations_for_parent(candidate.id)
            ):
                addable_node_ids.add(candidate.id)
        self.graph.set_history(
            self.project.history,
            selected_id=previous,
            show_rejected=self.show_rejected_check.isChecked(),
            addable_node_ids=addable_node_ids,
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
        if node.operation_id == "peaks.branches" and node.status == NodeStatus.SUCCEEDED:
            branch_count, reason = self._branch_count_for_node(node)
            if branch_count is not None and branch_count >= 2:
                if node.review_status == ReviewStatus.ACCEPTED:
                    self.branch_hint.setText(
                        f"Число дифракционных ветвей: {branch_count}. Нажмите «+» в правом нижнем "
                        "углу этой карточки, чтобы добавить 3D-кросс-корреляцию; выберите одну "
                        "опорную ветвь и одну или несколько ветвей-партнёров."
                    )
                else:
                    self.branch_hint.setText(
                        f"Число дифракционных ветвей: {branch_count}. Сначала проверьте результат "
                        "и нажмите «Принять результат». После принятия на карточке появится «+» "
                        "для добавления 3D-кросс-корреляции."
                    )
            elif branch_count == 1:
                self.branch_hint.setText(
                    "Обнаружена одна дифракционная ветвь. 3D-кросс-корреляция доступна только "
                    "при наличии не менее двух различных branch_id."
                )
            elif branch_count == 0:
                self.branch_hint.setText(
                    "Каталог ветвей пуст. 3D-кросс-корреляция недоступна."
                )
            elif reason:
                self.branch_hint.setText(reason)
        else:
            self.branch_hint.setText(
                "Щелчок — инспектор · глаз — результат · + — новая ветвь. "
                "Сплошная стрелка — вычислительный вход; пунктир — только происхождение ветви. "
                "Перетаскивание — панорама; Ctrl+колесо — масштаб."
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
        self.graph_scroll_button.setEnabled(not busy)
        self.show_rejected_check.setEnabled(not busy)
        if self.embedded:
            self.embedded_run_button.setEnabled(not busy and runnable)
            self.embedded_restore_button.setEnabled(not busy and self._preview_context is not None)
            self.embedded_accept_button.setVisible(accept_enabled)
            self.embedded_reject_button.setVisible(accept_enabled)
            self.embedded_accept_button.setEnabled(not busy and accept_enabled)
            self.embedded_reject_button.setEnabled(not busy and reject_enabled)
        if busy != self._last_busy_state:
            self._last_busy_state = busy
            self.busy_changed.emit(busy)

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
        node_id = self._selected_node_id()
        if not node_id:
            return
        node = self.project.history.get(node_id)
        if node.operation_id == "theory.identify_hkl":
            if node.status != NodeStatus.SUCCEEDED:
                QMessageBox.information(
                    self, "Предпросмотр HKL",
                    "Сначала успешно выполните выбранный HKL-этап.",
                )
                return
            popup_matches_node = (
                self._active_stage_popup is not None
                and getattr(self._active_stage_popup, "_stage_node_id", None) == node_id
            )
            if not popup_matches_node:
                if not node.parent_ids:
                    QMessageBox.warning(
                        self, "Предпросмотр HKL",
                        "У сохранённого HKL-узла отсутствует родительская ветвь.",
                    )
                    return
                self._open_stage_configuration(
                    node.parent_ids[0], "theory.identify_hkl", existing_node_id=node_id
                )
            if (
                self._active_stage_popup is not None
                and getattr(self._active_stage_popup, "_stage_node_id", None) == node_id
            ):
                self._start_hkl_result_preview(node_id)
                # The eye controls the spatial result independently of the
                # read-only HKL dialog. Reload the selected node through the
                # normal preview path so its Δθ result layers are selectable
                # in the host 2D map while the dialog remains open.
                self._start_preview_for_node(node_id, switch_tab=False, show_error_dialog=True)
            return
        self._start_preview_for_node(node_id, switch_tab=True, show_error_dialog=True)

    def _start_hkl_result_preview(self, node_id: str) -> bool:
        """Attach cached data or start a lightweight artifact-only HKL read."""
        popup = self._active_stage_popup
        if popup is None or getattr(popup, "_stage_node_id", None) != node_id:
            return False
        cached = self._hkl_preview_cache.get(node_id)
        if cached is not None:
            self._apply_hkl_result_preview(node_id, cached)
            return True
        if node_id in self._hkl_preview_tasks:
            return True
        try:
            task = _HklResultPreviewTask(
                node_id, WorkflowHistory(list(self.project.history.nodes)),
                str(self.project.artifact_store.directory),
            )
            task.signals.succeeded.connect(self._on_hkl_result_preview_ready)
            task.signals.failed.connect(self._on_hkl_result_preview_failed)
            self._hkl_preview_tasks[node_id] = task
            status = getattr(popup, "_stage_hkl_status", None)
            if status is not None:
                status.setText("Загрузка таблиц HKL из сохранённого артефакта…")
                status.show()
            self.thread_pool.start(task)
            return True
        except Exception:
            self._on_hkl_result_preview_failed(node_id, traceback.format_exc())
            return False

    def _on_hkl_result_preview_ready(self, node_id: str, preview: Any) -> None:
        self._hkl_preview_tasks.pop(node_id, None)
        self._hkl_preview_cache[node_id] = preview
        # Keep only a small number of immutable result views in memory.
        while len(self._hkl_preview_cache) > 3:
            oldest_id = next(iter(self._hkl_preview_cache))
            if oldest_id == node_id and len(self._hkl_preview_cache) > 1:
                oldest_id = next(key for key in self._hkl_preview_cache if key != node_id)
            self._hkl_preview_cache.pop(oldest_id, None)
        self._apply_hkl_result_preview(node_id, preview)

    def _apply_hkl_result_preview(self, node_id: str, preview: Any) -> None:
        popup = self._active_stage_popup
        if popup is None or getattr(popup, "_stage_node_id", None) != node_id:
            return
        panel = getattr(popup, "_stage_hkl_panel", None)
        status = getattr(popup, "_stage_hkl_status", None)
        if panel is None:
            return
        # The panel received its current theme when the popup was created and
        # is updated by _apply_theme(). set_result() already performs the first
        # complete draw; re-applying the theme here would render every chart a
        # second time during the latency-sensitive preview path.
        panel.set_result(preview)
        panel.show()
        if status is not None:
            status.hide()
        run_button = getattr(popup, "_stage_run_button", None)
        if run_button is not None:
            run_button.setText("Расчёт завершён")
            run_button.setEnabled(False)
            run_button.hide()
        cancel_button = getattr(popup, "_stage_cancel_button", None)
        if cancel_button is not None:
            cancel_button.setText("Закрыть")
        popup.adjustSize()
        screen = QApplication.screenAt(popup.frameGeometry().center()) or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            max_height = max(260, int(available.height() * 0.89))
            desired_height = min(max(560, popup.sizeHint().height()), max_height)
            # Width is intentionally not increased: retain the width already
            # chosen by _open_stage_configuration (maximum 640 px for HKL).
            desired_width = min(popup.width(), max(320, available.width() - 16))
            scroll = getattr(popup, "_stage_form_scroll", None)
            if scroll is not None:
                scroll.setMinimumHeight(max(180, desired_height - 82))
                scroll.setMaximumHeight(max(180, max_height - 82))
            popup.setMaximumHeight(max_height)
            popup.resize(desired_width, desired_height)
        self.status_label.setText("Результат HKL загружен из сохранённого артефакта.")
        self._log(f"HKL-предпросмотр загружен: {node_id}")

    def _on_hkl_result_preview_failed(self, node_id: str, error: str) -> None:
        self._hkl_preview_tasks.pop(node_id, None)
        popup = self._active_stage_popup
        if popup is not None and getattr(popup, "_stage_node_id", None) == node_id:
            status = getattr(popup, "_stage_hkl_status", None)
            if status is not None:
                status.setText(
                    "HKL-расчёт сохранён, но таблицы результата не удалось открыть. "
                    "Подробности приведены в журнале workflow."
                )
                status.show()
        self._log("Не удалось загрузить HKL-артефакт:\n" + error)

    def _start_preview_for_node(
        self, node_id: str, *, switch_tab: bool, show_error_dialog: bool
    ) -> bool:
        if self._active_task is not None or self._active_preview_task is not None:
            return False
        try:
            node = self.project.history.get(node_id)
            if node.status != NodeStatus.SUCCEEDED:
                message = "Сначала успешно выполните выбранный этап."
                if show_error_dialog:
                    QMessageBox.information(self, "Предпросмотр", message)
                else:
                    self.status_label.setText(message)
                return False
            task = _PreviewNodeTask(
                node_id,
                WorkflowHistory(list(self.project.history.nodes)),
                str(self.project.artifact_store.directory),
                self.project.source_metadata or {},
            )
            task.signals.succeeded.connect(self._on_preview_ready)
            task.signals.failed.connect(self._on_preview_failed)
            self._active_preview_task = task
            self.status_label.setText(f"Загрузка результата: {node.label}")
            self.progress.setRange(0, 0)
            if switch_tab:
                self.tabs.setCurrentIndex(0)
            self._update_controls()
            self.thread_pool.start(task)
            return True
        except Exception as exc:
            self._active_preview_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            if show_error_dialog:
                QMessageBox.warning(self, "Предпросмотр недоступен", str(exc))
            else:
                self.status_label.setText(f"Предпросмотр входной карты недоступен: {exc}")
            self._update_controls()
            return False

    def _complete_pending_peak_map_preview(self, completed_node_id: str) -> None:
        pending_id = self._pending_peak_map_parent_id
        if pending_id is None:
            return
        if self._active_stage_popup is None or self._active_stage_operation != "peaks.detect":
            self._pending_peak_map_parent_id = None
            return
        context_node = (
            self._preview_context.get("node") if self._preview_context is not None else None
        )
        if pending_id == completed_node_id and context_node is not None and context_node.id == pending_id:
            self._pending_peak_map_parent_id = None
            button = self._forms.get("peaks.detect", {}).get("pick_button")
            if button is not None and self.embedded:
                button.setEnabled(True)
                button.setToolTip(
                    "Выбрать пиксель на 2D-карте и показать реальную локальную ROC из входа этой ветви."
                )
            return
        if pending_id == completed_node_id:
            self._pending_peak_map_parent_id = None
            button = self._forms.get("peaks.detect", {}).get("pick_button")
            if button is not None:
                button.setEnabled(False)
                button.setToolTip("Не удалось подготовить предпросмотр входной карты этой ветви.")
            return
        # A different preview finished while the user was configuring peaks.
        # Load the requested branch now rather than allowing a stale-map pick.
        self._start_preview_for_node(
            pending_id, switch_tab=False, show_error_dialog=False
        )

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
            view_state = context.get("view_state")
            if view_state is not None:
                self.preview_state_ready.emit(view_state)
            stage_popup = self._active_stage_popup
            if (
                node.operation_id == "theory.identify_hkl"
                and stage_popup is not None
                and getattr(stage_popup, "_stage_node_id", None) == node_id
            ):
                hkl_preview = getattr(view_state, "hkl_identification_preview", None)
                stage_panel = getattr(stage_popup, "_stage_hkl_panel", None)
                stage_status = getattr(stage_popup, "_stage_hkl_status", None)
                if hkl_preview is not None and stage_panel is not None:
                    if not getattr(stage_panel, "has_result", False):
                        self._apply_hkl_result_preview(node_id, hkl_preview)
                elif stage_status is not None:
                    warning = (getattr(view_state, "metadata", {}) or {}).get(
                        "hkl_identification_preview_warning"
                    ) if view_state is not None else None
                    stage_status.setText(
                        "Расчёт завершён и сохранён, но подготовка таблиц предпросмотра не удалась. "
                        + (str(warning) if warning else "Подробности доступны в журнале workflow.")
                    )
                    stage_status.show()
                    stage_run_button = getattr(stage_popup, "_stage_run_button", None)
                    if stage_run_button is not None:
                        stage_run_button.setText("Расчёт завершён")
                        stage_run_button.setEnabled(False)
            if node.operation_id == "theory.identify_hkl" and context.get("hkl_analysis") is not None:
                hkl_bundle = context["hkl_analysis"]
                theory_meta = (hkl_bundle.metadata or {}).get("theory", {})
                pair_stats = hkl_bundle.tables.get("hkl_pair_statistics")
                ready_rows = [] if pair_stats is None else pair_stats.loc[
                    pair_stats["identification_status"].astype(str) == "ready"
                ]
                best = [
                    f"{row.pair_id}: {row.best_hkl} (Δ={row.best_abs_difference_arcsec:.2f}″)"
                    for row in ready_rows.itertuples(index=False)
                    if getattr(row, "best_hkl", None) is not None
                ]
                self.preview_summary.setText(
                    f"Теоретическая идентификация {theory_meta.get('crystal_name', 'HKL')}: "
                    f"рассчитано рефлексов={theory_meta.get('reflection_count', 0)}; "
                    f"пар с данными={theory_meta.get('pair_count', 0)}. "
                    + ("Ближайшие геометрические совпадения: " + "; ".join(best[:4])
                       if best else "Недостаточно принятых соответствий для ранжирования.")
                    + " Вероятности не вычислялись; подробные таблицы и гистограммы расположены в окне HKL-этапа."
                )
            elif node.operation_id == "peaks.branch_correlation" and context.get("branch_correlation") is not None:
                correlation_bundle = context["branch_correlation"]
                correlation_meta = (correlation_bundle.metadata or {}).get("branch_correlation", {})
                self.preview_summary.setText(
                    f"3D-кросс-корреляция: source branch_id={correlation_meta.get('source_branch_id', '—')}; "
                    f"партнёры={correlation_meta.get('partner_branch_ids', [])}. "
                    f"Назначено пиков: {int(correlation_meta.get('assigned_event_count', 0)):,}; "
                    f"неоднозначно: {int(correlation_meta.get('ambiguous_event_count', 0)):,}; "
                    f"без соответствия: {int(correlation_meta.get('residual_event_count', 0)):,}. "
                    "В 3D сохранены исходные координаты и branch_id; таблицы подробных назначений "
                    "и сдвигов доступны в диагностике результата."
                )
            elif node.operation_id == "peaks.detect" and context.get("peaks") is not None:
                peak_meta = context["peaks"].metadata
                peak_cfg = peak_meta.get("peak_config", {})
                signal_name = (
                    "интерполированный сигнал после Spike detection"
                    if peak_meta.get("used_interpolated_data")
                    else "измеренные counts из Scan-входа"
                )
                window = int(peak_cfg.get("window_size", 3))
                summary = (
                    f"Поиск пиков: {signal_name}; детектор получает среднюю ROC-кривую "
                    f"пространственной окрестности {window}×{window}. "
                    "На графике отдельно показаны ROC выбранного пикселя и фактический "
                    "сигнал поиска; маркеры относятся к последнему."
                )
                diagnostic_warning = context.get("spike_diagnostic_warning")
                if diagnostic_warning:
                    summary += " QC-overlay спайков недоступен: " + diagnostic_warning
                self.preview_summary.setText(summary)
            else:
                self.preview_summary.setText(
                    f"Предпросмотр: {node.label}. Внутренняя карта и ROC-кривая обновлены; "
                    "основные окна приложения могут подключаться через preview_requested."
                )
            self.status_label.setText(f"Предпросмотр загружен: {node.label}")
            self._log(f"Предпросмотр загружен: {node.label}")
        except Exception:
            error_text = traceback.format_exc()
            self._preview_context = previous_context
            if previous_context is not None:
                try:
                    self._draw_preview()
                except Exception:
                    self._log(
                        "Не удалось восстановить предыдущий предпросмотр:\n"
                        + traceback.format_exc()
                    )
            else:
                self.figure.clear()
                empty_map = self.figure.add_subplot(1, 2, 1)
                empty_curve = self.figure.add_subplot(1, 2, 2)
                empty_map.text(0.5, 0.5, "Предпросмотр недоступен", ha="center", va="center")
                empty_map.set_axis_off()
                empty_curve.set_axis_off()
                self.canvas.draw_idle()
            self._log(error_text)
            if self._pending_peak_map_parent_id is None:
                QMessageBox.warning(self, "Ошибка отображения", error_text)
            else:
                self.status_label.setText("Не удалось подготовить входную 2D-карту для выбора пикселя")
        finally:
            self._active_preview_task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self._update_controls()
            self._complete_pending_peak_map_preview(node_id)

    def _on_preview_failed(self, node_id: str, error: str) -> None:
        self._active_preview_task = None
        stage_popup = self._active_stage_popup
        if (
            stage_popup is not None
            and self._active_stage_operation == "theory.identify_hkl"
            and getattr(stage_popup, "_stage_node_id", None) == node_id
        ):
            stage_status = getattr(stage_popup, "_stage_hkl_status", None)
            stage_run_button = getattr(stage_popup, "_stage_run_button", None)
            if stage_status is not None:
                stage_status.setText(
                    "HKL-расчёт завершён, но предпросмотр результата не загрузился. "
                    "Артефакт расчёта сохранён; подробности — в журнале workflow."
                )
                stage_status.show()
            if stage_run_button is not None:
                stage_run_button.setText("Расчёт сохранён")
                stage_run_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("Не удалось загрузить предпросмотр")
        self._log(error)
        if node_id == self._pending_peak_map_parent_id:
            self._pending_peak_map_parent_id = None
            button = self._forms.get("peaks.detect", {}).get("pick_button")
            if button is not None:
                button.setEnabled(False)
                button.setToolTip(
                    "Не удалось загрузить входную 2D-карту этой ветви. Закройте и откройте форму заново."
                )
        elif self._active_stage_popup is None:
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
            "peaks.branches": "branch_count_map",
            "peaks.branch_correlation": "branch_count_map",
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
            map_title = "Число пиков в локальной ROC-окрестности"
        elif node.operation_id == "fit.scan" and fit is not None and "n_fitted_map" in fit.arrays:
            map_data = np.asarray(fit.arrays["n_fitted_map"])
            map_title = "Число fitted-компонент"
        elif node.operation_id in {"peaks.branches", "peaks.branch_correlation"}:
            view_state = ctx.get("view_state")
            branch_state = next(
                (layer for layer in getattr(view_state, "map_layers", ())
                 if layer.key == "branches.count"),
                None,
            )
            if branch_state is not None:
                map_data = np.asarray(branch_state.data_yx, dtype=float)
                map_title = "Число ветвей в пикселе"
            else:
                map_data = np.nansum(data, axis=-1)
                map_title = "Суммарная интенсивность"
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
        elif node.operation_id == "peaks.branch_correlation":
            result = ctx.get("branch_correlation")
            tables = getattr(result, "tables", {}) if result is not None else {}
            assignments = tables.get("branch_correlation_assignments") if isinstance(tables, dict) else None
            pairs = tables.get("branch_correlation_pairs") if isinstance(tables, dict) else None
            metadata = (getattr(result, "metadata", {}) or {}).get("branch_correlation", {}) if result is not None else {}
            if result is None or assignments is None or assignments.empty:
                ax.text(0.5, 0.5, "Таблица назначений кросс-корреляции отсутствует или пуста", ha="center", va="center")
                ax.set_axis_off()
            else:
                classes = assignments["assignment_class"].astype(str).value_counts() if "assignment_class" in assignments else {}
                order = ["partner", "ambiguous", "residual"]
                labels = ["Назначено партнёру", "Неоднозначно", "Без соответствия"]
                values = [int(classes.get(key, 0)) for key in order]
                bars = ax.bar(labels, values)
                ax.set_title(
                    f"Результат 3D-кросс-корреляции · source branch_id={metadata.get('source_branch_id', '—')}"
                )
                ax.set_ylabel("Число пиков исходной ветви")
                ax.tick_params(axis="x", labelrotation=12)
                for bar, value in zip(bars, values):
                    ax.annotate(
                        f"{value:,}",
                        (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                        ha="center", va="bottom", fontsize=8,
                    )
                details = []
                if pairs is not None and not pairs.empty:
                    for row in pairs.itertuples(index=False):
                        details.append(
                            f"branch {int(row.partner_branch_id)}: "
                            f"Δθ={float(row.theta_shift_final_arcsec):+.3g}″, "
                            f"ΔY={int(row.refined_shift_y_px):+d}, "
                            f"ΔX={int(row.refined_shift_x_px):+d}; {row.refinement_status}"
                        )
                if details:
                    ax.text(
                        0.01, 0.98, "\n".join(details),
                        transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
                        bbox={
                            "boxstyle": "round,pad=0.35",
                            "facecolor": ax.get_facecolor(),
                            "alpha": 0.92,
                            "edgecolor": "0.55",
                        },
                    )
                ax.grid(axis="y", alpha=0.2)
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
        self._style_matplotlib_figure(self.diagnostic_figure)
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
            "Число spike-точек", "Число пиков в локальной ROC-окрестности",
            "Число fitted-компонент",
        }
        peak_map_display = map_title == "Число пиков в локальной ROC-окрестности"
        if peak_map_display:
            # Keep the embedded Workbench QC map semantically identical to the
            # main ROI Explorer map: hue = peak count, alpha = integrated I(θ).
            scan_for_mask = ctx.get("scan")
            valid_yxt = None
            analysis_valid_yx = None
            peak_input = np.asarray(data_yxt, dtype=float)
            if scan_for_mask is not None:
                valid_yxt = scan_for_mask.arrays.get("valid_mask_yxt")
                analysis_valid_yx = scan_for_mask.arrays.get("analysis_valid_mask_yx")
            if peaks is not None and bool(peaks.metadata.get("used_interpolated_data", False)):
                if spikes is None or "clean_data_yxt" not in spikes.arrays:
                    raise ValueError(
                        "Поиск пиков выполнен на spike-interpolation, "
                        "но QC карта не получила clean_data_yxt"
                    )
                peak_input = np.asarray(spikes.arrays["clean_data_yxt"], dtype=float)
                valid_yxt = spikes.arrays.get(
                    "detector_valid_mask_yxt", valid_yxt
                )
            intensity, valid_yx = integrated_intensity_map(
                peak_input, theta, valid_mask_yxt=valid_yxt,
                analysis_valid_mask_yx=analysis_valid_yx,
            )
            rgba, _ = peak_count_rgba(
                map_data, intensity, theme=self._theme, valid_mask_yx=valid_yx
            )
            self.ax_map.imshow(
                rgba, origin="lower", interpolation="nearest", aspect="auto"
            )
            self.ax_map.text(
                0.02, 0.02, "цвет: число пиков · прозрачность: ∫I dθ",
                transform=self.ax_map.transAxes, fontsize=6.5,
                va="bottom", ha="left", alpha=0.82,
            )
        elif map_title in overlay_titles:
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
            scan_meta = ctx["scan"].metadata if ctx.get("scan") is not None else {}
            pool_meta = scan_meta.get("pool", {}) if isinstance(scan_meta, dict) else {}
            if pool_meta:
                pool_label = (
                    f"pooling {pool_meta.get('size', '?')}×{pool_meta.get('size', '?')}, "
                    f"stride {pool_meta.get('stride', '?')}"
                )
            else:
                pool_label = "без pooling"
            curve_label = f"ROC выбранного пикселя · {pool_label}"
            if partial_coverage:
                curve_label += " · частичное покрытие (только QC)"
            self.ax_curve.plot(theta, curve, linewidth=1.1, alpha=0.82, label=curve_label)

            # A partial pooling bin is intentionally excluded from scientific
            # analysis. Show only its observed valid-count sum; do not overlay
            # spike/peak/Fit outputs produced on fully covered sites.
            if not partial_coverage:
                if spikes is not None:
                    spike_grid = str(spikes.metadata.get("grid_id", ""))
                    scan_grid = str(ctx["scan"].metadata.get("grid_id", "")) if ctx.get("scan") is not None else ""
                    if spike_grid and scan_grid and spike_grid != scan_grid:
                        raise ValueError("Spike QC-артефакт относится к другой пространственной сетке")
                    clean_values = spikes.arrays.get("clean_data_yxt")
                    if clean_values is not None:
                        clean = np.asarray(clean_values, dtype=float)
                        if clean.shape != data_yxt.shape:
                            raise ValueError("clean_data_yxt не согласована с Scan-входом")
                        self.ax_curve.plot(
                            theta, clean[y, x, :], linewidth=1.35,
                            label="После spike-interpolation",
                        )
                    spike_values = spikes.arrays.get("spike_mask_tyx")
                    if spike_values is not None:
                        spike_mask = np.asarray(spike_values, dtype=bool)
                        expected_spike_shape = (theta.size, *data_yxt.shape[:2])
                        if spike_mask.shape != expected_spike_shape:
                            raise ValueError(
                                f"spike_mask_tyx имеет форму {spike_mask.shape}; "
                                f"ожидалась {expected_spike_shape}"
                            )
                        mask = spike_mask[:, y, x]
                        if np.any(mask):
                            self.ax_curve.scatter(
                                theta[mask], curve[mask], marker="x", s=32,
                                label="Обнаруженные спайки", zorder=5,
                            )
                if peaks is not None and "peak_catalog" in peaks.tables:
                    # Plot the exact signal used by the detector even when this
                    # pixel has zero peaks; absence of markers must not hide the
                    # search input and create a misleading QC view.
                    from roc_mcs.peaks.peak_signal import local_neighbourhood_mean_curve

                    peak_input = data_yxt
                    input_source = "измеренные counts"
                    if bool(peaks.metadata.get("used_interpolated_data", False)):
                        if spikes is None or "clean_data_yxt" not in spikes.arrays:
                            raise ValueError(
                                "Поиск пиков выполнен на spike-interpolation, "
                                "но QC предпросмотр не получил clean_data_yxt"
                            )
                        clean = np.asarray(spikes.arrays["clean_data_yxt"], dtype=float)
                        if clean.shape != data_yxt.shape:
                            raise ValueError("clean_data_yxt не согласована с Scan-входом")
                        peak_input = clean
                        input_source = "после spike-interpolation"

                    peak_valid = np.ones(data_yxt.shape[:2], dtype=bool)
                    if ctx.get("scan") is not None:
                        scan_valid = ctx["scan"].arrays.get("valid_mask_yxt")
                        if scan_valid is not None:
                            scan_valid = np.asarray(scan_valid, dtype=bool)
                            if scan_valid.shape != data_yxt.shape:
                                raise ValueError("valid_mask_yxt не согласована с peak input")
                            peak_valid &= np.all(scan_valid, axis=-1)
                        analysis_valid = ctx["scan"].arrays.get("analysis_valid_mask_yx")
                        if analysis_valid is not None:
                            analysis_valid = np.asarray(analysis_valid, dtype=bool)
                            if analysis_valid.shape != peak_valid.shape:
                                raise ValueError("analysis_valid_mask_yx не согласована с peak input")
                            peak_valid &= analysis_valid

                    peak_cfg = peaks.metadata.get("peak_config", {})
                    window_size = int(peak_cfg.get("window_size", 3))
                    search_curve = local_neighbourhood_mean_curve(
                        peak_input, y, x, window_size, pixel_valid=peak_valid
                    )
                    self.ax_curve.plot(
                        theta, search_curve, linewidth=1.55, linestyle="--",
                        label=f"Вход поиска · среднее {window_size}×{window_size} ({input_source})",
                        zorder=4,
                    )
                    catalog = peaks.tables["peak_catalog"]
                    if catalog is not None and not catalog.empty and {"x", "y", "theta"}.issubset(catalog.columns):
                        selected = catalog[(catalog["x"] == x) & (catalog["y"] == y)]
                        # build_peak_catalog rejects incomplete θ traces, so markers
                        # are plotted only when the complete detector input exists.
                        if not selected.empty and np.isfinite(search_curve).all():
                            peak_theta = selected["theta"].to_numpy(float)
                            peak_y = np.interp(peak_theta, theta, search_curve)
                            self.ax_curve.scatter(
                                peak_theta, peak_y, marker="v", s=40,
                                label="Найденные пики · фактический вход", zorder=7,
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
        self.ax_curve.set_title(f"ROC пикселя и вход поиска · x={x}, y={y}")
        self.pixel_label.setText(f"Пиксель: X={x}, Y={y}{suffix}")
        self.figure.tight_layout()
        self._style_matplotlib_figure(self.figure)
        if peak_map_display:
            self.ax_map.set_facecolor("#000000" if self._theme == "dark" else "#ffffff")
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
