import sys
import logging

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise ImportError("PyQt6 or PySide6 is required to run this desktop application.") from exc

import matplotlib
matplotlib.use("QtAgg", force=True)


from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)

Qt = QtCore.Qt
QTimer = QtCore.QTimer
QAction = QtGui.QAction
QActionGroup = QtGui.QActionGroup
QKeySequence = QtGui.QKeySequence
QShortcut = QtGui.QShortcut
QApplication = QtWidgets.QApplication
QComboBox = QtWidgets.QComboBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QGridLayout = QtWidgets.QGridLayout
QGroupBox = QtWidgets.QGroupBox
QHBoxLayout = QtWidgets.QHBoxLayout
QInputDialog = QtWidgets.QInputDialog
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QListWidget = QtWidgets.QListWidget
QListWidgetItem = QtWidgets.QListWidgetItem
QMainWindow = QtWidgets.QMainWindow
QMessageBox = QtWidgets.QMessageBox
QAbstractButton = QtWidgets.QAbstractButton
QPushButton = QtWidgets.QPushButton
QPlainTextEdit = QtWidgets.QPlainTextEdit
QScrollArea = QtWidgets.QScrollArea
QSplitter = QtWidgets.QSplitter
QSpinBox = QtWidgets.QSpinBox
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget


import re
import json
from PyQt6.QtGui import QCursor, QColor, QKeySequence, QShortcut, QPalette
from PyQt6.QtWidgets import QApplication, QFileDialog, QMenu, QMessageBox, QSplitter, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QToolButton, QTableWidgetItem
from PyQt6.QtCore import Qt, QSize


from roc_mcs.roi.h5_scan_repository import H5ScanRepository
from roc_mcs.roi.spatial_pooling import PoolSpec, pooled_shape
from roc_mcs.gui.roi_explorer.roi_manager_widget import _pane_toggle_icon
from roc_mcs.gui.components.collapsible_section import CollapsibleSection
from roc_mcs.gui.components.icon_provider import resource_svg_icon, set_icon_theme
from roc_mcs.gui.app_settings import (
    load_settings, save_settings, resolve_initial_database, remember_hdf5_database,
    settings_path, resolve_analysis_projects_directory,
)
from roc_mcs.analysis.project import project_directory_for_scan
from roc_mcs.analysis.branch_display import (
    BranchParameterChoices, available_branch_parameters, branch_catalog_fingerprint,
    branch_footprint_layers, build_branch_parameter_map,
    load_branch_display_colors, normalize_branch_colors,
    save_branch_display_colors, summarize_branch_catalog,
)
from roc_mcs.gui.roi_explorer.theme import apply_theme as apply_roi_theme, apply_popup_theme, resolve_theme, theme_colors
from roc_mcs.gui.roi_explorer.plot_workspace_widget import PlotWorkspaceWidget
from roc_mcs.gui.roi_explorer.plot_display_settings import PlotDisplaySettingsDialog, load_plot_display_settings, save_plot_display_settings
from roc_mcs.roi.roi_types import ROI, SavedROI
from roc_mcs.roi.extraction import extract_from_roi
from roc_mcs.gui.processing_workbench.roi_workflow_panel import ProcessingStepsPanel
from roc_mcs.visualization.peak_count_map import integrated_intensity_map
from roc_mcs.gui.roi_explorer.theta_range_preview import (
    infer_theta_projection_kind,
    normalize_theta_axis,
    ThetaProjectionAccumulator,
    THETA_RANGE_PREVIEW_INTERVAL_MS,
)


class RockingCurveExplorerWindow(QMainWindow):
    def __init__(self, h5_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.setWindowTitle("Rocking Curve Explorer")
        self.resize(1100, 700)
        self._global_settings = load_settings()
        app = QApplication.instance()
        requested_theme = str(self._global_settings.get("theme", "system")).lower()
        active_theme = requested_theme if requested_theme in ("dark", "light") else resolve_theme(app)
        if app is not None:
            app.setProperty("rocMcsTheme", active_theme); app.setProperty("rocMcsThemeMode", requested_theme)
            try: app.setProperty("rocMcsIconSize", min(20, max(12, int(self._global_settings.get("icon_size", 16)))))
            except (TypeError, ValueError, OverflowError): app.setProperty("rocMcsIconSize", 16)
            set_icon_theme(app, QColor(theme_colors(active_theme)["text"]))
        self._preferred_theme = active_theme

        self.repo: Optional[H5ScanRepository] = None
        self.session_scans: Dict[str, List[SavedROI]] = {}
        self.next_roi_id = 1
        self.selected_scan: Optional[str] = None
        self.selected_map: Optional[str] = None
        self.selected_roi_id: Optional[int] = None
        self.preview_roi: Optional[ROI] = None
        self.preview_curve: Optional[np.ndarray] = None
        self.preview_th: Optional[np.ndarray] = None
        self.preview_count: int = 0
        self._preview_fit_pending = False
        self.selected_pixel: Optional[tuple[int, int]] = None  # индексы (x, y) текущего view
        self.reduction_mode = "mean"
        self.current_file: Optional[Path] = None
        self._theta_slice_mode = False; self._theta_index = 0; self._theta_scan_cache_key = None; self._theta_scan_cache = None; self._theta_levels_cache = {}; self._show_theta_guide = False
        self._theta_projection_accumulator = ThetaProjectionAccumulator()
        self._theta_projection_full_image: np.ndarray | None = None
        self._theta_projection_full_title = ""
        self._theta_range_finalize_pending = False
        self._theta_range_linked = False
        self._theta_range_link_available = False
        # Last discrete sample interval accepted from either linked slider.
        # Range sliders emit many sub-sample mouse events; ignore repeats so
        # they cannot keep postponing the coalesced 2D refresh.
        self._theta_range_last_requested_indices: tuple[int, int] | None = None
        self._analysis_view = None
        # The analysis preview may temporarily choose a semantic palette (e.g.
        # RdBu_r for signed Δθ). Keep the source-map palette for the whole
        # preview session and restore it on every exit path.
        self._analysis_previous_cmap: str | None = None
        self._volume_refresh_pending = True
        self._analysis_previous_pool = None
        self._analysis_curve_cache = {}
        self._analysis_map_key = None
        self._analysis_curve_key = None
        self._analysis_display_identity = None
        self._analysis_display_preferences = {}
        self._analysis_layer_cmap_preferences = {}
        self._analysis_quantitative_contrast = False
        self._analysis_contrast_snapshot: tuple[float, float] | None = None
        self._branch_catalog = None
        self._branch_fingerprint: str | None = None
        self._branch_display_colors: dict[int, str] = {}
        self._branch_display_state_path: Path | None = None
        self._branch_parameter_layer = None
        self._branch_parameter_branch_id: int | None = None
        # Parameter-map display is a transient view mode. Keep only the
        # selected parameter per branch (never raster arrays) so switching
        # branches is deterministic without retaining large image caches.
        self._branch_parameter_choices = BranchParameterChoices()
        self._branch_parameter_view_enabled = False
        self._branch_parameter_return_map_key: str | None = None
        self._branch_parameter_current_parameter: str | None = None

        self._theta_render_timer = QTimer(self); self._theta_render_timer.setSingleShot(True); self._theta_render_timer.setInterval(35); self._theta_render_timer.timeout.connect(self.refresh_map)
        self._theta_range_preview_timer = QTimer(self)
        self._theta_range_preview_timer.setSingleShot(True)
        self._theta_range_preview_timer.setInterval(THETA_RANGE_PREVIEW_INTERVAL_MS)
        self._theta_range_preview_timer.timeout.connect(self._refresh_theta_range_preview)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview_curve)
        self._volume_refresh_timer = QTimer(self)
        self._volume_refresh_timer.setSingleShot(True)
        self._volume_refresh_timer.timeout.connect(self.refresh_volume)

        self._build_ui()
        self._build_actions()
        self._connect_signals()
        self._apply_theme(self._preferred_theme)
        self.splitter.setHandleWidth(8)
        if h5_path is not None:
            self.load_hdf5(h5_path)
        else:
            initial_database = resolve_initial_database()
            if initial_database is not None: self.load_hdf5(initial_database)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self.central = QWidget(self)
        self.central.setObjectName("roiExplorerCentral")
        self.setCentralWidget(self.central)

        outer = QHBoxLayout(self.central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self.central)
        outer.addWidget(self.splitter)

        # Левая панель: отдельная узкая рейка остаётся видимой в свёрнутом состоянии.
        self.left_sidebar = QWidget(self.splitter); self.left_sidebar.setObjectName("leftSidebarSurface"); self.left_sidebar.setMinimumWidth(340)
        self.left_sidebar_layout = QHBoxLayout(self.left_sidebar); self.left_sidebar_layout.setContentsMargins(0, 0, 0, 0); self.left_sidebar_layout.setSpacing(2)
        self.left_rail = QWidget(self.left_sidebar); self.left_rail.setObjectName("leftSidebarRail"); self.left_rail.setFixedWidth(20)
        left_rail_layout = QVBoxLayout(self.left_rail); left_rail_layout.setContentsMargins(0, 0, 0, 0); left_rail_layout.setSpacing(0)
        self.left_sidebar_toggle = QToolButton(self.left_rail); self.left_sidebar_toggle.setObjectName("leftSidebarToggle"); self.left_sidebar_toggle.setIcon(_pane_toggle_icon(True, QColor(theme_colors(resolve_theme(self))["text"]))); self.left_sidebar_toggle.setIconSize(QSize(14, 14)); self.left_sidebar_toggle.setFixedSize(20, 20); self.left_sidebar_toggle.setAutoRaise(True); self.left_sidebar_toggle.setCursor(Qt.CursorShape.PointingHandCursor); self.left_sidebar_toggle.setToolTip("Collapse sidebar")
        self.left_sidebar_toggle.setStyleSheet("QToolButton { background: transparent; border: none; border-radius: 5px; padding: 1px; } QToolButton:hover { background: palette(midlight); }")
        left_rail_layout.addWidget(self.left_sidebar_toggle, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop); left_rail_layout.addStretch(1)
        self.left_scroll = QScrollArea(self.left_sidebar); self.left_scroll.setObjectName("leftSidebarScroll")
        self.left_scroll.viewport().setObjectName("leftSidebarViewport")
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_sidebar_layout.addWidget(self.left_rail); self.left_sidebar_layout.addWidget(self.left_scroll, 1)
        self._left_sidebar_collapsed = False; self._left_sidebar_width = 340
        self.left_sidebar_toggle.clicked.connect(self._toggle_left_sidebar)

        left = QWidget()
        left.setObjectName("leftSidebarContent")
        self.left_scroll.setWidget(left)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # База данных: выбор файла и разовое обновление списка без фонового опроса.
        file_box = QGroupBox("База данных HDF5")
        file_form = QFormLayout(file_box); file_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.file_path_edit = QLineEdit(); self.file_path_edit.setObjectName("h5PathField"); self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("База не выбрана")
        self.open_h5_btn = QToolButton(file_box); self.open_h5_btn.setObjectName("databaseOpenButton"); self.open_h5_btn.setFixedSize(32, 28); self.open_h5_btn.setIconSize(QSize(18, 18))
        self.open_h5_btn.setIcon(resource_svg_icon("folder_open.svg", widget=self.open_h5_btn, size=18)); self.open_h5_btn.setToolTip("Выбрать базу HDF5; путь сохранится для следующего запуска")
        self.open_h5_btn.setAccessibleName("Выбрать базу HDF5"); self.open_h5_btn.clicked.connect(self.open_hdf5_dialog)
        self.refresh_scans_btn = QToolButton(file_box); self.refresh_scans_btn.setObjectName("refreshScansButton"); self.refresh_scans_btn.setFixedSize(32, 28); self.refresh_scans_btn.setIconSize(QSize(18, 18))
        self.refresh_scans_btn.setIcon(resource_svg_icon("refresh.svg", widget=self.refresh_scans_btn, size=18)); self.refresh_scans_btn.setToolTip("Обновить список сканов из базы (без фонового опроса)")
        self.refresh_scans_btn.setAccessibleName("Обновить список сканов"); self.refresh_scans_btn.setEnabled(False); self.refresh_scans_btn.clicked.connect(self.refresh_scans)
        file_row = QWidget(file_box); file_row_layout = QHBoxLayout(file_row); file_row_layout.setContentsMargins(0, 0, 0, 0); file_row_layout.setSpacing(4)
        file_row_layout.addWidget(self.file_path_edit, 1); file_row_layout.addWidget(self.open_h5_btn); file_row_layout.addWidget(self.refresh_scans_btn)
        file_form.addRow(file_row); left_layout.addWidget(file_box)

        # ROI Manager перенесён в компактную рейку слева от карты.
        # Сохраняем совместимость старой логики предварительной ROI.
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("ROI label")

        # Первые этапы воспроизводимого Workbench-пайплайна.
        self.processing_section = CollapsibleSection("Обработка", expanded=True)
        self.processing_panel = ProcessingStepsPanel(self.processing_section.body)
        self.processing_section.content_layout.addWidget(self.processing_panel)
        left_layout.addWidget(self.processing_section)

        # Inspector закреплён сверху; свободное место остаётся после секций,
        # как в боковой панели Live Scans.
        self.inspector_section = CollapsibleSection("Inspector", expanded=True)
        self.stats_text = QPlainTextEdit(self.inspector_section.body); self.stats_text.setReadOnly(True); self.stats_text.setMaximumBlockCount(1000)
        self.inspector_section.content_layout.addWidget(self.stats_text); left_layout.addWidget(self.inspector_section)
        left_layout.addStretch(1)

       
        # Правая рабочая область: ROI Manager, карта и кривая.
        self.plot_workspace = PlotWorkspaceWidget(self.splitter)
        self.plot_splitter = self.plot_workspace                  # Совместимость обработчиков размеров
        self.roi_manager = self.plot_workspace.roi_manager
        self.roi_table = self.plot_workspace.roi_table
        self.map_widget = self.plot_workspace.map_widget
        self.curve_widget = self.plot_workspace.curve_widget
        self.volume_widget = self.plot_workspace.volume_widget
        self.branch_manager = self.plot_workspace.branch_manager

        
        self.preview_curve_line = None
        self.preview_marker_line = None
        self._map_patches: Dict[int, Rectangle] = {}
        self._curve_lines: Dict[int, Any] = {}

        self.status = self.statusBar()
        self.status.showMessage(f"Выберите базу HDF5 кнопкой с папкой. Настройки: {settings_path()}")

        # Собираем ГЛАВНЫЙ сплиттер: слева меню, справа - сплиттер с графиками
        self.splitter.addWidget(self.left_sidebar)
        self.splitter.addWidget(self.plot_splitter)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.left_scroll.setMinimumWidth(316)

        # --- Подключение сигналов ---
        # 1. Правый клик (Контекстное меню)
        self.map_widget.roi_right_click_callback = self._show_roi_context_menu
        self.curve_widget.roi_right_click_callback = self._show_roi_context_menu
        # 2. Горячие клавиши
        self._delete_shortcut = QShortcut(QKeySequence("Delete"), self)          # Привязываем клавишу Delete к методу удаления выделенного ROI
        self._delete_shortcut.activated.connect(self.delete_selected_roi)
        # 3. Обработчик левого клика по объектам
        self._map_pick_cid = self.map_widget.canvas.mpl_connect("pick_event", self._on_pick_event)            # Переменная self._*_pick_cid сохраняет так называемый Connection ID (идентификатор подключения) от Matplotlib. 
        self._curve_pick_cid = self.curve_widget.canvas.mpl_connect("pick_event", self._on_pick_event)        # Он нужен на тот случай, если дальше в коде я решу временно отключить выбор элементов мышкой (например, вызовом canvas.mpl_disconnect(self._pick_cid)).
        

    def _apply_theme(self, theme: str | None = None) -> None:
        if getattr(self, "_applying_theme", False): return
        self._applying_theme = True
        try:
            active_theme = apply_roi_theme(self, theme)
            icon_color = QColor(theme_colors(active_theme)["text"])
            self.left_sidebar_toggle.setIcon(_pane_toggle_icon(not self._left_sidebar_collapsed, icon_color))
            self.map_widget.apply_theme(active_theme); self.curve_widget.apply_theme(active_theme); self.volume_widget.apply_theme(active_theme); self.roi_manager.apply_theme(active_theme)
            if hasattr(self, "processing_panel"):
                self.processing_panel.apply_theme(active_theme)
            display_settings = load_plot_display_settings(); self.map_widget.apply_display_settings(display_settings["map"]); self.curve_widget.apply_display_settings(display_settings["curve"]); self.volume_widget.apply_display_settings(display_settings["volume"])
            self.plot_workspace.apply_theme(active_theme)
            for popup in self.findChildren(QWidget):
                if popup.objectName() in ("roiEditorPopup", "compactColorPicker"):
                    update_popup_theme = getattr(popup, "apply_theme", None)
                    if callable(update_popup_theme): update_popup_theme(active_theme)
            self.open_h5_btn.setIcon(resource_svg_icon("folder_open.svg", widget=self.open_h5_btn, size=self._global_icon_size()))
            self.refresh_scans_btn.setIcon(resource_svg_icon("refresh.svg", widget=self.refresh_scans_btn, size=self._global_icon_size()))
            if hasattr(self, "display_settings_action"):
                self.display_settings_action.setIcon(resource_svg_icon("setting_icon.svg", size=self._global_icon_size()))
            self._apply_global_icon_size()
        finally:
            self._applying_theme = False

    def _global_icon_size(self) -> int:
        app = QApplication.instance()
        try: return min(20, max(12, int(app.property("rocMcsIconSize") if app is not None else 16)))
        except (TypeError, ValueError, OverflowError): return 16

    def _apply_global_icon_size(self) -> None:
        size = QSize(self._global_icon_size(), self._global_icon_size())
        for button in self.findChildren(QAbstractButton):
            if not button.icon().isNull(): button.setIconSize(size)
        # Branch and ROI rails use these same global icon metrics for their
        # square swatches and circular branch markers (including row pitch).
        for manager in (getattr(self, "roi_manager", None), getattr(self, "branch_manager", None)):
            apply_icon_size = getattr(manager, "apply_global_icon_size", None)
            if callable(apply_icon_size):
                apply_icon_size()
        # Embedded tools that own icon-to-control geometry reapply that policy
        # after this generic icon-size pass. Keep the explicit VTK object-name
        # check as a guard for the native toolbar's geometry contract.
        for widget in self.findChildren(QWidget):
            is_vtk_widget = widget.objectName() == "EmbeddedVTKVolumeWidget"
            is_workbench_widget = widget.objectName() == "processingWorkbenchWidget"
            if is_vtk_widget or is_workbench_widget:
                apply_icon_size = getattr(widget, "apply_global_icon_size", None)
                if callable(apply_icon_size):
                    apply_icon_size()
        if hasattr(self, "display_settings_action"):
            self.display_settings_action.setIcon(resource_svg_icon("setting_icon.svg", size=self._global_icon_size()))
            if hasattr(self,"display_settings_btn"):self.display_settings_btn.setIconSize(size)

    def apply_global_settings(self, settings: dict | None = None) -> None:
        """Получает общую тему и размер значков от оболочки TRS Apps."""
        if settings is not None: self._global_settings = dict(settings)
        app = QApplication.instance(); active = str(app.property("rocMcsTheme") if app is not None else resolve_theme(self)).lower()
        if active not in ("dark", "light"): active = resolve_theme(self)
        self._preferred_theme = active; self._apply_theme(active); self._apply_global_icon_size()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.PaletteChange, QtCore.QEvent.Type.ApplicationPaletteChange) and hasattr(self, "map_widget"):
            app = QApplication.instance(); requested = app.property("rocMcsTheme") if app is not None else None
            target = str(requested).lower() if str(requested).lower() in ("dark", "light") else resolve_theme(app)
            if not getattr(self, "_applying_theme", False) and target != self.property("rocMcsTheme"): self._apply_theme(target)

    def _toggle_left_sidebar(self) -> None:
        """Сворачивает Data file и Inspector, оставляя кнопку на месте."""
        sizes = self.splitter.sizes()
        if not self._left_sidebar_collapsed:
            if sizes and sizes[0] > 22: self._left_sidebar_width = sizes[0]
            self._left_sidebar_collapsed = True; self.left_scroll.hide(); self.left_sidebar.setMinimumWidth(22); self.left_sidebar.setMaximumWidth(22)
            self.left_sidebar_toggle.setIcon(_pane_toggle_icon(False, self.palette().color(QPalette.ColorRole.WindowText))); self.left_sidebar_toggle.setToolTip("Expand sidebar")
            self.splitter.setSizes([22, max(1, sum(sizes) - 22)])
        else:
            self._left_sidebar_collapsed = False; self.left_sidebar.setMaximumWidth(16777215); self.left_sidebar.setMinimumWidth(340); self.left_scroll.show()
            self.left_sidebar_toggle.setIcon(_pane_toggle_icon(True, self.palette().color(QPalette.ColorRole.WindowText))); self.left_sidebar_toggle.setToolTip("Collapse sidebar")
            total = max(sum(sizes), self.width()); left_width = max(340, self._left_sidebar_width); self.splitter.setSizes([left_width, max(1, total - left_width)])

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for attr, text, slot in (
            ("act_open_h5", "Open HDF5…", self.open_hdf5_dialog),
            ("act_load_session", "Load session JSON…", self.load_session_dialog),
            ("act_save_session", "Save session JSON…", self.save_session_dialog),
            ("act_export_current", "Export current scan JSON…", self.export_current_scan_dialog),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            setattr(self, attr, action)
            file_menu.addAction(action)
        file_menu.addSeparator()
        self.act_quit = QAction("Quit", self)
        self.act_quit.triggered.connect(lambda *_: self.close())
        file_menu.addAction(self.act_quit)
        self.display_settings_action = QAction(resource_svg_icon("setting_icon.svg", size=self._global_icon_size()), "Настройки экранов", self); self.display_settings_action.setObjectName("roiDisplaySettingsAction"); self.display_settings_action.setToolTip("Настройки отображения окон «Карта», «ROC-кривые» и «3D»"); self.display_settings_action.setStatusTip(self.display_settings_action.toolTip()); self.display_settings_action.triggered.connect(self.open_plot_display_settings)
        # Кнопка ставится сразу после File, а не в дальний угол строки меню.
        self.display_settings_btn=QToolButton(self.menuBar()); self.display_settings_btn.setObjectName("roiDisplaySettingsButton"); self.display_settings_btn.setDefaultAction(self.display_settings_action); self.display_settings_btn.setIconSize(QtCore.QSize(self._global_icon_size(),self._global_icon_size())); self.display_settings_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly); self.display_settings_btn.setAutoRaise(True); self.display_settings_btn.setFixedSize(28,24); self.display_settings_btn.setToolTip(self.display_settings_action.toolTip()); self.display_settings_btn.setStyleSheet("QToolButton#roiDisplaySettingsButton{border:0;padding:1px;background:transparent;} QToolButton#roiDisplaySettingsButton:hover{background:palette(midlight);border-radius:3px;}")
        self.menuBar().installEventFilter(self); QTimer.singleShot(0,self._position_display_settings_button)

    def _position_display_settings_button(self) -> None:
        bar=self.menuBar()
        if not hasattr(self,"display_settings_btn") or bar.width()<80:return
        menus=[a for a in bar.actions() if a.menu()]
        anchor=bar.actionGeometry(menus[0]) if menus else None
        x=(anchor.right()+3) if anchor is not None and anchor.isValid() else 54
        x=max(0,min(x,bar.width()-self.display_settings_btn.width()))
        self.display_settings_btn.move(x,max(0,(bar.height()-self.display_settings_btn.height())//2)); self.display_settings_btn.raise_(); self.display_settings_btn.show()

    def eventFilter(self,watched,event):
        if watched is self.menuBar() and event.type()==QtCore.QEvent.Type.Resize:QTimer.singleShot(0,self._position_display_settings_button)
        return super().eventFilter(watched,event)

    def _connect_signals(self) -> None:
        """ 'Провода', которые соединяют кнопки с функциями """
        self.map_widget.scan_combo.currentIndexChanged.connect(self._on_scan_changed)
        self.map_widget.map_combo.currentIndexChanged.connect(self._on_map_changed)
        self.map_widget.reduction_combo.currentIndexChanged.connect(self._on_reduction_changed)
        self.map_widget.pool_changed.connect(self._on_pool_changed)
        self.map_widget.theta_index_changed.connect(self._on_theta_index_changed)
        self.map_widget.pixel_clicked.connect(self._on_pixel_clicked)
        self.map_widget.pixel_pick_selected.connect(self.processing_panel.set_peak_preview_pixel)
        self.map_widget.cursor_position_changed.connect(self.volume_widget.set_map_cursor_position)
        self.map_widget.preview_roi_changed.connect(self.volume_widget.set_roi_preview_geometry)
        self.map_widget.preview_roi_finished.connect(self.volume_widget.set_roi_preview_geometry)
        self.volume_widget.data_rebuild_requested.connect(self._request_volume_refresh)
        self.volume_widget.theta_slice_value_changed.connect(self._on_vtk_theta_slice_changed)
        # Range previews use inclusive sample indices on a normalized ascending
        # physical θ axis. Programmatic mirroring blocks the source signals.
        map_tools = self.map_widget.tools
        volume_tools = self.volume_widget.tools
        map_tools.theta_display_range_changed.connect(
            self._on_map_theta_display_range_changed
        )
        map_tools.theta_range_link_toggled.connect(
            lambda checked: self._on_theta_range_link_toggled(checked, "map")
        )
        volume_tools.theta_display_range_changed.connect(
            self._on_volume_theta_display_range_changed
        )
        volume_tools.theta_range_link_toggled.connect(
            lambda checked: self._on_theta_range_link_toggled(checked, "volume")
        )
        map_tools.theta_display_range_interaction_started.connect(
            self._on_theta_range_interaction_started
        )
        volume_tools.theta_display_range_interaction_started.connect(
            self._on_theta_range_interaction_started
        )
        map_tools.theta_display_range_interaction_finished.connect(
            self._on_theta_range_interaction_finished
        )
        volume_tools.theta_display_range_interaction_finished.connect(
            self._on_theta_range_interaction_finished
        )
        self.volume_widget.theta_range_axis_changed.connect(
            lambda _axis: self._sync_theta_range_context()
        )
        self.plot_workspace.mdi_area.panel_visibility_changed.connect(self._on_plot_panel_visibility_changed)
        self.plot_workspace.mdi_area.panel_restored.connect(self._on_plot_panel_restored)
        self.curve_widget.theta_guide_toggled.connect(self._on_theta_guide_toggled)
        self.processing_panel.preview_ready.connect(self.show_analysis_preview)
        self.processing_panel.pixel_pick_mode_requested.connect(self._set_peak_pixel_pick_mode)
        self.processing_panel.restore_requested.connect(self.restore_analysis_preview)
        self.processing_panel.busy_changed.connect(self._set_processing_busy)
        self.processing_panel.status_changed.connect(lambda text: self.status.showMessage(text, 8000))

        # --- Сохранение ROI ---
        self.map_widget.pin_roi_requested.connect(self.pin_preview_roi)    # От кнопки над картой (слушаем сигнал из MapPlotWidget)
        # --- Сигналы компактного ROI Manager ---
        self.roi_table.roi_edited.connect(self._on_roi_table_edited)
        self.roi_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.roi_manager.edit_requested.connect(self._on_roi_manager_edit_requested)
        self.roi_manager.color_changed.connect(self._on_roi_manager_color_changed)
        self.roi_manager.delete_requested.connect(self._on_roi_manager_delete_requested)
        self.roi_manager.export_requested.connect(self._save_roi_curve)
        self.roi_manager.copy_requested.connect(self._copy_roi_content)
        # PlotWorkspaceWidget owns the two adjacent object-manager rails and
        # their splitter geometry. The legacy 3-pane callback is intentionally
        # not connected because the workspace now has four conceptual regions.
        self.branch_manager.selection_changed.connect(self._on_branch_selection_changed)
        self.branch_manager.color_changed.connect(self._on_branch_color_changed)
        self.branch_manager.footprints_toggled.connect(self._on_branch_footprints_toggled)
        self.branch_manager.focus_3d_toggled.connect(self._on_branch_focus_3d_toggled)
        self.branch_manager.parameter_map_requested.connect(self._on_branch_parameter_map_requested)

        # --- Сигналы тулбара над картой ---
        self.map_widget.low_pct.valueChanged.connect(self.refresh_map)
        self.map_widget.high_pct.valueChanged.connect(self.refresh_map)
        self.map_widget.auto_contrast_btn.clicked.connect(self.reset_contrast_defaults)

        # Сигнал для включения/выключения рисования ROI
        self.map_widget.roi_btn.toggled.connect(self._toggle_roi_selector_mode)
        # Когда селектор на карте изменяется (тянут за ручку, рисуют линию) -> пересчитываем рокинги и обновляем графики
        self.map_widget.preview_roi_changed.connect(self._on_selector_change)
        self.map_widget.preview_roi_finished.connect(self._on_selector_finished)

    def open_plot_display_settings(self) -> None:
        """Настраивает стиль Map, ROC и 3D и сохраняет его в settings.json."""
        dialog = PlotDisplaySettingsDialog(load_plot_display_settings(), self)
        apply_popup_theme(dialog, resolve_theme(self))
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            values = dialog.values()
            try: save_plot_display_settings(values)
            except Exception as exc:
                warning = QMessageBox(self); warning.setWindowTitle("Настройки отображения"); warning.setIcon(QMessageBox.Icon.Warning); warning.setText(f"Изменения применены, но settings.json не сохранён:\n{exc}"); apply_popup_theme(warning, resolve_theme(self)); warning.exec()
            self.map_widget.apply_display_settings(values["map"]); self.curve_widget.apply_display_settings(values["curve"]); self.volume_widget.apply_display_settings(values["volume"])

    # ---------- HDF5/session ----------
    def load_hdf5(self, h5_path: Union[str, Path]) -> None:
        path = Path(h5_path).expanduser().resolve()
        repo = None
        try:
            repo = H5ScanRepository(path); scan_keys = list(repo.scan_keys())
        except Exception as exc:
            closer = getattr(repo, "close", None)
            if callable(closer):
                try: closer()
                except Exception: pass
            QMessageBox.critical(self, "Open HDF5", str(exc)); return

        # The selected workflow view belongs to the previous HDF5 repository.
        # Drop it before swapping repositories; its saved PoolSpec must not be
        # applied to the newly opened source.
        self._analysis_view = None
        self._clear_branch_context()
        self._restore_analysis_source_colormap()
        self._analysis_previous_pool = None
        self._analysis_curve_cache.clear()
        self._theta_projection_accumulator.clear()
        self._set_theta_range_linked(False)
        self._analysis_map_key = None; self._analysis_curve_key = None; self._analysis_display_identity = None
        self._sync_analysis_result_menus(None)
        old_repo = self.repo; self.repo = repo
        self.map_widget.set_pool_spec(repo.pool); self.map_widget.set_pool_enabled(True)
        self.current_file = path; self.file_path_edit.setText(str(path))
        self.session_scans = {key: [] for key in scan_keys}
        self.selected_scan = self.selected_map = self.selected_roi_id = None
        self.preview_roi = self.preview_curve = self.preview_th = None
        self.preview_count = 0; self._preview_fit_pending = False; self.selected_pixel = None
        self._rebuild_scan_combo(preferred_scan=scan_keys[-1] if scan_keys else None)
        self.refresh_scans_btn.setEnabled(True)
        if scan_keys: self._on_scan_changed()
        closer = getattr(old_repo, "close", None)
        if old_repo is not None and old_repo is not repo and callable(closer):
            try: closer()
            except Exception: pass
        try:
            remember_hdf5_database(path); config_warning = ""
        except (OSError, ValueError, TypeError) as exc:
            config_warning = f" Не удалось сохранить путь в настройках: {exc}"
        scan_message = f"Открыт последний скан: {scan_keys[-1]}." if scan_keys else "В базе пока нет сканов."
        self.status.showMessage(f"Загружена {path.name}; сканов: {len(scan_keys)}. {scan_message}{config_warning}", 8000)

    def open_hdf5_dialog(self, *args) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать базу HDF5",
            str(self.current_file.parent if self.current_file else Path.home()),
            "HDF5 files (*.h5 *.hdf5 *.hdf);;All files (*)",
        )
        if path:
            self.load_hdf5(path)


    def _rebuild_scan_combo(self, preferred_scan: str | None = None) -> None:
        combo = self.map_widget.scan_combo; current = preferred_scan or combo.currentText()
        combo.blockSignals(True)
        try:
            combo.clear()
            if self.repo is not None:
                for scan_key in self.repo.scan_keys():
                    combo.addItem(scan_key); combo.setItemData(combo.count() - 1, scan_key, Qt.ItemDataRole.ToolTipRole)
            index = combo.findText(current) if current else -1
            if index < 0 and combo.count(): index = combo.count() - 1
            if index >= 0: combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(False)

    def refresh_scans(self, *args) -> None:
        """Однократно переоткрывает HDF5, сохраняя выбранный скан и ROI."""
        if self.current_file is None or self.repo is None:
            self.status.showMessage("Сначала выберите базу HDF5.", 5000); return
        old_repo = self.repo; old_scan = self.selected_scan; old_map = self.selected_map
        old_sessions = self.session_scans; old_keys = set(old_repo.scan_keys())
        new_repo = None
        try:
            new_repo = H5ScanRepository(self.current_file)
            if hasattr(new_repo, "set_pool"): new_repo.set_pool(old_repo.pool)
            new_keys = list(new_repo.scan_keys())
        except Exception as exc:
            closer = getattr(new_repo, "close", None)
            if callable(closer):
                try: closer()
                except Exception: pass
            self.status.showMessage(f"Не удалось обновить HDF5; текущие данные сохранены: {exc}", 8000); return
        if not new_keys:
            closer = getattr(new_repo, "close", None)
            if callable(closer):
                try: closer()
                except Exception: pass
            self.status.showMessage("В базе пока не обнаружены сканы; прежнее состояние сохранено.", 8000); return
        self.repo = new_repo; self.session_scans = {key: old_sessions.get(key, []) for key in new_keys}
        self._rebuild_scan_combo(preferred_scan=old_scan)
        selected = self.map_widget.scan_combo.currentText()
        if selected != old_scan or old_scan not in new_keys:
            self._on_scan_changed()
        else:
            self.selected_scan = selected; self.selected_map = old_map
            self._populate_map_combo(); self.refresh_curves(); self.refresh_roi_table(); self._update_inspector()
        closer = getattr(old_repo, "close", None)
        if callable(closer):
            try: closer()
            except Exception: pass
        added = len(set(new_keys) - old_keys)
        self.status.showMessage(f"Список сканов обновлён: +{added}, всего {len(new_keys)}.", 6000)
        
    def save_session_dialog(self, *args) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save session JSON",
            self.default_session_path().as_posix() if self.default_session_path() else str(Path.home() / "roc_session.json"),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            try:
                self.save_session(path)
            except Exception as exc:
                QMessageBox.critical(self, "Save session", str(exc))

    def load_session_dialog(self, *args) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load session JSON",
            str(self.current_file.parent if self.current_file else Path.home()),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.load_session(path)

    def export_current_scan_dialog(self, *args) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current scan JSON",
            self.default_export_path().as_posix() if self.default_export_path() else str(Path.home() / "roc_current_scan.json"),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            try:
                self.export_current_scan(path)
            except Exception as exc:
                QMessageBox.critical(self, "Export JSON", str(exc))

    def default_session_path(self) -> Optional[Path]:
        if self.current_file is None:
            return None
        return self.current_file.with_name(f"{self.current_file.stem}_roc_session.json")

    def default_export_path(self) -> Optional[Path]:
        if self.current_file is None or self.selected_scan is None:
            return None
        safe_scan = self.selected_scan.replace("/", "_")
        return self.current_file.with_name(f"{self.current_file.stem}_{safe_scan}_rois.json")

    def save_session(self, path: Union[str, Path]) -> Path:
        if self.repo is None:
            raise RuntimeError("No HDF5 file is loaded.")
        payload = self._session_payload(all_scans=True)
        path = Path(path).expanduser().resolve()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status.showMessage(f"Saved session JSON: {path}")
        return path

    def export_current_scan(self, path: Union[str, Path]) -> Path:
        if self.repo is None or self.selected_scan is None:
            raise RuntimeError("No scan is active.")
        payload = self._session_payload(all_scans=False)
        path = Path(path).expanduser().resolve()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status.showMessage(f"Exported current scan JSON: {path}")
        return path

    def load_session(self, path: Union[str, Path]) -> None:
        if self.repo is None:
            QMessageBox.warning(self, "Load session", "Open the matching HDF5 file first.")
            return

        path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Load session", f"Failed to read JSON:\n{exc}")
            return

        session_h5 = payload.get("h5_path")
        if session_h5 and Path(session_h5).expanduser().resolve() != self.repo.h5_path:
            QMessageBox.warning(
                self,
                "Load session",
                "This session belongs to a different HDF5 file.\n"
                "Open that file first if you want to restore the ROIs exactly.",
            )
            return

        saved_pool = payload.get("pool")
        if saved_pool is not None:
            try:
                pool = PoolSpec(**saved_pool)
                target_scan = payload.get("selected_scan")
                if target_scan in self.repo.scans:
                    pooled_shape(self.repo.load_source(target_scan).source_shape_yx, pool)
                self.repo.set_pool(pool)
            except Exception as exc:
                self.repo.set_pool(PoolSpec())
                QMessageBox.warning(self, "Load session", f"Сохранённый spatial view недоступен; выбран Raw.\n{exc}")
        self.map_widget.set_pool_spec(self.repo.pool)

        scans = payload.get("scans", {})
        restored = 0
        self.session_scans = {key: [] for key in self.repo.scan_keys()}
        for scan_key, scan_payload in scans.items():
            if scan_key not in self.session_scans:
                continue
            rois = []
            for item in scan_payload.get("rois", []):
                try:
                    saved = SavedROI.from_dict(item)
                except Exception:
                    continue
                if saved.id <= 0:
                    saved.id = self._next_available_roi_id()
                # Кривые вычисляются заново: в старом JSON они могли относиться к другому pooling-view.
                saved.curve = None
                saved.count = 0
                self.next_roi_id = max(self.next_roi_id, saved.id + 1)
                rois.append(saved)
                restored += 1
            self.session_scans[scan_key] = rois

        self.reduction_mode = str(payload.get("reduction_mode", self.reduction_mode)).lower()
        self.map_widget.reduction_combo.blockSignals(True)
        idx = self.map_widget.reduction_combo.findText(self.reduction_mode)
        if idx >= 0:
            self.map_widget.reduction_combo.setCurrentIndex(idx)
        self.map_widget.reduction_combo.blockSignals(False)

        contrast = payload.get("contrast", {})
        if isinstance(contrast, dict):
            try:
                # self.low_pct.setValue(float(contrast.get("low", self.low_pct.value())))
                # self.high_pct.setValue(float(contrast.get("high", self.high_pct.value())))
                self.map_widget.low_pct.setValue(float(contrast.get("low", self.map_widget.low_pct.value())))
                self.map_widget.high_pct.setValue(float(contrast.get("high", self.map_widget.high_pct.value())))                 
            except Exception:
                pass

        selected_scan = payload.get("selected_scan")
        self.selected_pixel = None
        if selected_scan in self.session_scans:
            selected_map = payload.get("selected_map")
            self.map_widget.scan_combo.blockSignals(True)
            self.map_widget.scan_combo.setCurrentText(selected_scan)
            self.map_widget.scan_combo.blockSignals(False)
            self._on_scan_changed()
            if selected_map and self.map_widget.map_combo.findText(selected_map) >= 0:
                self.map_widget.map_combo.setCurrentText(selected_map)
                self.selected_map = selected_map
                self.refresh_map()
                self.refresh_curves(update_volume=False)
        else:
            self.refresh_all()
        self.status.showMessage(f"Loaded session JSON: {path} ({restored} ROI(s) restored)")

    def _session_payload(self, all_scans: bool) -> dict:
        if self.repo is None:
            raise RuntimeError("No HDF5 file is loaded.")
        if all_scans:
            scans_payload = {}
            for key, rois in self.session_scans.items():
                scans_payload[key] = {"rois": [r.to_dict() for r in rois]}
        else:
            scans_payload = {
                self.selected_scan: {"rois": [r.to_dict() for r in self.current_rois()]}
            }
        return {
            "version": 1,
            "h5_path": str(self.repo.h5_path),
            "selected_scan": self.selected_scan,
            "selected_map": self.selected_map,
            "pool": self.repo.pool.to_dict(),
            "reduction_mode": self.reduction_mode,
            "contrast": {
                "low": float(self.map_widget.low_pct.value()),
                "high": float(self.map_widget.high_pct.value()),                
            },
            "scans": scans_payload,
        }

    # ---------- Selection / current state ----------
    def current_rois(self) -> List[SavedROI]:
        if self.selected_scan is None:
            return []
        return self.session_scans.setdefault(self.selected_scan, [])

    def _next_available_roi_id(self) -> int:
        roi_id = self.next_roi_id
        self.next_roi_id += 1
        return roi_id


    def _generate_unique_label(self, base_name: str = "ROI") -> str:
        """
        Вместо _auto_label()
        Ищет первое свободное имя в формате 'ROI X' или 'ROI X (N)'.
        Никаких счетчиков, только честный поиск по существующим именам.
        """
        existing_labels = {r.roi.label for r in self.current_rois()}
        # 1. Сначала пробуем просто "ROI 1", "ROI 2" и т.д.
        counter = 1
        while True:
            candidate = f"{base_name} {counter}"
            if candidate not in existing_labels:
                return candidate
            counter += 1
            
    def _selected_pinned_roi(self) -> Optional[SavedROI]:
        if self.selected_roi_id is None:
            return None
        for pinned in self.current_rois():
            if pinned.id == self.selected_roi_id:
                return pinned
        return None

    def _set_selected_roi_id(self, roi_id: Optional[int], *, update_volume: bool = False) -> None:
        """Refresh selection visuals; optionally refresh 3D after a real data edit."""
        self.selected_roi_id = roi_id
        self._sync_label_box()
        self.refresh_map()
        self.refresh_curves(update_volume=False)
        self.refresh_roi_table()
        self._update_inspector()
        if update_volume:
            self.refresh_volume()

    def _sync_label_box(self) -> None:
        saved = self._selected_pinned_roi()
        if saved is not None:
            self.label_edit.blockSignals(True)
            self.label_edit.setText(saved.roi.label)
            self.label_edit.blockSignals(False)
        elif self.preview_roi is not None:
            self.label_edit.blockSignals(True)
            self.label_edit.setText(self.preview_roi.label)
            self.label_edit.blockSignals(False)
        else:
            self.label_edit.blockSignals(True)
            self.label_edit.clear()
            self.label_edit.blockSignals(False)

    # ---------- Scan / map ----------
    def _on_scan_changed(self, *args) -> None:
        if self.repo is None or self.map_widget.scan_combo.currentText() == "":
            return
        if self._analysis_view is not None or self._analysis_previous_cmap is not None:
            self.restore_analysis_preview()
        self.selected_scan = self.map_widget.scan_combo.currentText()
        self._theta_scan_cache_key = None; self._theta_scan_cache = None; self._theta_levels_cache.clear()
        self._theta_projection_accumulator.clear()
        source = self.repo.load_source(self.selected_scan)
        pool = self.repo.pool
        pool_warning = None
        try:
            pooled_shape(source.source_shape_yx, pool)
            self.repo.load_scan(self.selected_scan)  # проверяем, что выбранный view реально строится
        except ValueError as exc:
            if not pool.active:
                self.status.showMessage(f"Cannot load scan {self.selected_scan}: {exc}")
                QMessageBox.warning(self, "Select scan", str(exc))
                return
            self.repo.set_pool(PoolSpec())
            self._invalidate_curve_cache()  # Все ROI-кривые могли быть рассчитаны на прежней pooled-сетке.
            self.map_widget.set_pool_spec(self.repo.pool, source_shape_yx=source.source_shape_yx)
            pool_warning = f"{pool.tag} не подходит для {self.selected_scan}; view переключён на Raw."
        else:
            self.map_widget.set_pool_spec(self.repo.pool, source_shape_yx=source.source_shape_yx)

        if self.selected_scan not in self.session_scans:
            self.session_scans[self.selected_scan] = []
        self.selected_map = None
        self.preview_roi = None
        self.preview_curve = None
        self.preview_th = None
        self.preview_count = 0
        self._preview_fit_pending = False
        self.selected_pixel = None
        self._preview_timer.stop()
        self.map_widget.reset_preview_selector()
        self._populate_map_combo()
        self._set_selected_roi_id(None)
        try:
            self.processing_panel.set_source(self.current_file, self.selected_scan)
        except Exception as exc:
            self.status.showMessage(f"Processing project unavailable: {exc}", 8000)
        self.status.showMessage(pool_warning or f"Selected scan: {self.selected_scan} · {self.repo.view}")
        self._request_volume_refresh()


    def _populate_map_combo(self) -> None:
        if self.repo is None or self.selected_scan is None: return
        combo = self.map_widget.map_combo; combo.blockSignals(True); combo.clear()
        maps = list(self.repo.available_maps(self.selected_scan))
        for map_name in maps:
            combo.addItem(str(map_name), str(map_name)); combo.setItemData(combo.count()-1, str(map_name), Qt.ItemDataRole.ToolTipRole)
        combo.addItem("Срез по θ", "__theta_slice__")
        combo.setItemData(combo.count()-1, "Показать интенсивность для одного угла θ; ползунок управляет углом", Qt.ItemDataRole.ToolTipRole)
        if self.selected_map not in maps: self.selected_map = maps[0] if maps else None
        idx = combo.findData("__theta_slice__") if self._theta_slice_mode else combo.findData(self.selected_map)
        combo.setCurrentIndex(max(0, idx) if idx >= 0 else 0); combo.blockSignals(False)
        self._set_theta_axis_for_current_scan()
        self.map_widget.set_theta_slice_mode(self._theta_slice_mode)
        self._sync_theta_range_context()
        self.refresh_map(); self.refresh_curves()

    @staticmethod
    def _scan_field(scan, key, default=None):
        if isinstance(scan, dict): return scan.get(key, default)
        try:
            value = scan[key]
            if value is not None: return value
        except (KeyError, TypeError, IndexError, AttributeError): pass
        return getattr(scan, key, default)

    def _get_theta_scan(self):
        if self.repo is None or self.selected_scan is None: return None
        cache_key = (str(getattr(self.repo, "h5_path", self.current_file or "")), str(self.selected_scan), repr(getattr(self.repo, "pool", None)))
        if self._theta_scan_cache_key == cache_key and self._theta_scan_cache is not None: return self._theta_scan_cache
        scan = self.repo.load_scan(self.selected_scan)
        data = self._scan_field(scan, "data", self._scan_field(scan, "raw", None))
        theta = self._scan_field(scan, "th", self._scan_field(scan, "theta", None))
        axis = int(self._scan_field(scan, "axis", 0))
        data = np.asarray(data) if data is not None else None
        theta = np.asarray(theta, dtype=float).reshape(-1) if theta is not None else np.empty(0, dtype=float)
        if data is None or data.ndim != 3: raise ValueError("В выбранном скане не найден трёхмерный массив интенсивности.")
        ntheta = data.shape[0] if axis == 0 else data.shape[2] if axis == 2 else 0
        if axis not in (0, 2): raise ValueError(f"Не поддерживается расположение θ-оси: axis={axis}. Ожидается 0 или 2.")
        if theta.size != ntheta: raise ValueError(f"Число углов θ ({theta.size}) не совпадает с размером массива ({ntheta}).")
        if not theta.size: raise ValueError("В выбранном скане нет углов θ.")
        self._theta_scan_cache_key = cache_key; self._theta_scan_cache = (data, theta, axis)
        return self._theta_scan_cache

    def _current_map_theta_axis(self) -> np.ndarray:
        """Return the physical θ-axis used by the currently displayed 2D view."""
        try:
            if self._analysis_view is not None:
                axis = np.asarray(self._analysis_view.theta, dtype=np.float64)
            else:
                scan = self._get_theta_scan()
                axis = np.asarray(scan[1], dtype=np.float64) if scan is not None else np.empty(0)
            if axis.ndim != 1 or not np.isfinite(axis).all():
                return np.empty(0, dtype=np.float64)
            if axis.size > 1:
                delta = np.diff(axis)
                if not (np.all(delta > 0.0) or np.all(delta < 0.0)):
                    return np.empty(0, dtype=np.float64)
            return axis
        except (AttributeError, TypeError, ValueError, KeyError, IndexError):
            return np.empty(0, dtype=np.float64)

    def _current_theta_projection_kind(self) -> str | None:
        """Return a reduction only when the selected 2D layer is a projection."""
        if self._theta_slice_mode:
            return None
        identifiers: list[object] = []
        if self._analysis_view is not None:
            state = self._analysis_view
            key = self._analysis_map_key
            identifiers.append(key)
            if hasattr(state, "map_layer"):
                try:
                    layer = state.map_layer(key)
                except (AttributeError, KeyError, TypeError, ValueError):
                    layer = None
                if layer is not None:
                    identifiers.extend((
                        getattr(layer, "key", None),
                        getattr(layer, "label", None),
                        getattr(layer, "title", None),
                    ))
            else:
                identifiers.extend((
                    getattr(state, "map_title", None),
                    getattr(state, "label", None),
                ))
        else:
            identifiers.extend((
                self.selected_map,
                self.map_widget.map_combo.currentData(),
                self.map_widget.map_combo.currentText(),
            ))
        return infer_theta_projection_kind(*identifiers)

    def _sync_theta_range_context(self) -> None:
        """Refresh 2D/3D range availability and validate their physical axes."""
        map_tools = self.map_widget.tools
        volume_tools = self.volume_widget.tools
        kind = self._current_theta_projection_kind()

        if kind is None:
            self._theta_range_preview_timer.stop()
            self._theta_range_finalize_pending = False
            self._theta_range_last_requested_indices = None
            self._theta_projection_accumulator.clear()
            self._theta_projection_full_image = None
            self._theta_projection_full_title = ""
            map_tools.set_theta_range_available(False)
            map_tools.set_theta_range_link_available(False)
            volume_tools.set_theta_range_link_available(False)
            self._theta_range_link_available = False
            if self._theta_range_linked:
                self._set_theta_range_linked(False)
            else:
                map_tools.set_theta_range_link_checked(False)
                volume_tools.set_theta_range_link_checked(False)
            return

        try:
            raw_axis = self._current_map_theta_axis()
            map_axis, _order = normalize_theta_axis(raw_axis)
        except (TypeError, ValueError, AttributeError):
            self._theta_range_preview_timer.stop()
            self._theta_range_finalize_pending = False
            self._theta_range_last_requested_indices = None
            self._theta_projection_accumulator.clear()
            self._theta_projection_full_image = None
            self._theta_projection_full_title = ""
            map_tools.set_theta_range_available(False)
            map_tools.set_theta_range_link_available(False)
            volume_tools.set_theta_range_link_available(False)
            self._theta_range_link_available = False
            if self._theta_range_linked:
                self._set_theta_range_linked(False)
            return

        if map_axis.size < 3:
            map_tools.set_theta_axis_values(map_axis)
            map_tools.set_theta_range_available(False)
        else:
            map_tools.set_theta_axis_values(map_axis)
            map_tools.set_theta_range_available(True)

        current_indices = map_tools.theta_range_indices
        self._theta_range_last_requested_indices = (
            tuple(map(int, current_indices)) if current_indices is not None else None
        )

        volume_axis = np.asarray(volume_tools.theta_axis_values, dtype=np.float64)
        compatible = bool(
            map_axis.size > 2
            and volume_axis.ndim == 1
            and volume_axis.size == map_axis.size
            and np.all(np.isfinite(volume_axis))
            and np.allclose(
                map_axis, volume_axis, rtol=1e-10, atol=1e-12
            )
        )
        self._theta_range_link_available = compatible
        map_tools.set_theta_range_link_available(compatible)
        volume_tools.set_theta_range_link_available(compatible)
        if not compatible and self._theta_range_linked:
            self._set_theta_range_linked(False)
        else:
            map_tools.set_theta_range_link_checked(self._theta_range_linked)
            volume_tools.set_theta_range_link_checked(self._theta_range_linked)

    def _set_theta_range_linked(self, linked: bool) -> None:
        active = bool(linked and self._theta_range_link_available)
        self._theta_range_linked = active
        if not active:
            self.volume_widget.finish_theta_range_interaction()
        self.map_widget.tools.set_theta_range_link_checked(active)
        self.volume_widget.tools.set_theta_range_link_checked(active)

    def _on_theta_range_link_toggled(self, checked: bool, source: str) -> None:
        """Enable/disable a two-way link; the initiating window is authoritative."""
        if not checked:
            self._set_theta_range_linked(False)
            return
        if not self._theta_range_link_available:
            self._set_theta_range_linked(False)
            return

        map_tools = self.map_widget.tools
        volume_tools = self.volume_widget.tools
        source_tools = map_tools if source == "map" else volume_tools
        indices = source_tools.theta_range_indices
        if indices is None:
            indices = (0, len(map_tools.theta_axis_values) - 1)
        indices = (int(indices[0]), int(indices[1]))
        self._set_theta_range_linked(True)
        self._theta_range_last_requested_indices = indices
        if source == "map":
            self.volume_widget.set_theta_display_range_indices(*indices)
        else:
            map_tools.set_theta_display_range_indices(*indices)
            self._schedule_theta_range_preview(immediate=True)

    def _schedule_theta_range_preview(self, *, immediate: bool = False) -> None:
        """Throttle range redraws without starving them during a continuous drag."""
        if immediate:
            self._theta_range_preview_timer.start(0)
        elif not self._theta_range_preview_timer.isActive():
            # Start once and keep the deadline stable while new slider values
            # arrive; the timeout always reads the newest stored interval.
            self._theta_range_preview_timer.start()

    def _accept_theta_range_indices(self, low: int, high: int) -> bool:
        """Return True only when a new discrete interval was selected."""
        indices = (int(low), int(high))
        if indices == self._theta_range_last_requested_indices:
            return False
        self._theta_range_last_requested_indices = indices
        return True

    def _on_map_theta_display_range_changed(self, low: int, high: int) -> None:
        if not self._accept_theta_range_indices(low, high):
            return
        if self._theta_range_linked and self._theta_range_link_available:
            self.volume_widget.set_theta_display_range_indices(int(low), int(high))
        self._schedule_theta_range_preview()

    def _on_volume_theta_display_range_changed(self, low: int, high: int) -> None:
        if not self._theta_range_linked or not self._theta_range_link_available:
            return
        if not self._accept_theta_range_indices(low, high):
            return
        self.map_widget.tools.set_theta_display_range_indices(
            int(low), int(high)
        )
        self._schedule_theta_range_preview()

    def _on_theta_range_interaction_started(self) -> None:
        """Use a lighter 3D screen sampling rate only while linked handles move."""
        if self._theta_range_linked and self._theta_range_link_available:
            self.volume_widget.set_theta_range_interaction_active(True)

    def _on_theta_range_interaction_finished(self) -> None:
        """Restore full 3D quality and finalize 2D contrast after a gesture."""
        self.volume_widget.finish_theta_range_interaction()
        self._theta_range_finalize_pending = True
        self._schedule_theta_range_preview()

    def _theta_projection_image(self, kind: str, low: int, high: int) -> np.ndarray:
        """Calculate the selected projection, reusing incremental 2D state."""
        state = self._analysis_view
        valid_mask = None
        if state is not None:
            # The user-facing scan.intensity_sum layer can intentionally use
            # the spike-interpolated signal. Reproduce the same source choice
            # as the workflow's peak-input selector; never silently integrate
            # the measured stack when the selected layer is explicitly based
            # on interpolation.
            data_source = state
            source_key_name = "scan.measured_intensity_sum"
            if (
                self._analysis_map_key == "scan.intensity_sum"
                and bool(state.metadata.get("peaks_used_interpolated_data", False))
            ):
                interpolated = (
                    state.curve_source("spikes.interpolated")
                    if hasattr(state, "curve_source") else None
                )
                if interpolated is None:
                    interpolated = next(
                        (
                            source for source in getattr(state, "curve_sources", ())
                            if getattr(source, "key", None) == "spikes.interpolated"
                        ),
                        None,
                    )
                if interpolated is None:
                    raise ValueError(
                        "Карта scan.intensity_sum использует spike-interpolation, "
                        "но источник spikes.interpolated недоступен."
                    )
                data_source = interpolated
                source_key_name = "spikes.interpolated"

            data = np.asarray(data_source.data_yxt)
            theta = np.asarray(data_source.theta, dtype=np.float64).reshape(-1)
            axis = 2
            valid_mask = getattr(data_source, "valid_mask_yxt", None)
            source_key = (
                "analysis", id(state), str(getattr(state, "node_id", "")),
                str(self._analysis_map_key), source_key_name, id(data),
                id(valid_mask),
            )
        else:
            scan = self._get_theta_scan()
            if scan is None:
                raise ValueError("Нет загруженного θ-объёма для пересчёта проекции.")
            data, theta, axis = scan
            source_key = (
                "hdf5", self._theta_scan_cache_key, id(data),
            )

        ascending_theta, order = normalize_theta_axis(theta)
        control_theta, _control_order = normalize_theta_axis(
            self._current_map_theta_axis()
        )
        if (
            ascending_theta.size != control_theta.size
            or not np.allclose(
                ascending_theta, control_theta, rtol=1e-10, atol=1e-12
            )
        ):
            raise ValueError(
                "Ось θ источника проекции не совпадает с осью θ текущего 2D/3D view."
            )
        if high >= ascending_theta.size:
            raise ValueError("Диапазон θ не согласован с текущим объёмом.")
        if axis not in (0, 2):
            raise ValueError(
                f"Неподдерживаемое расположение θ-оси: axis={axis}."
            )
        # The source stack remains in its native memory layout. For mean/sum,
        # small endpoint moves update only the entering/leaving detector plane.
        return self._theta_projection_accumulator.project(
            data,
            order,
            axis,
            kind,
            int(low),
            int(high),
            valid_mask=valid_mask,
            source_token=source_key,
        )

    def _apply_theta_projection_range(
        self, image: np.ndarray, title: str
    ) -> tuple[np.ndarray, str]:
        """Apply the current θ interval only to an eligible mean/max/sum map."""
        kind = self._current_theta_projection_kind()
        indices = self.map_widget.tools.theta_range_indices
        if kind is None:
            self._theta_projection_full_image = None
            self._theta_projection_full_title = ""
            return image, title
        self._theta_projection_full_image = np.asarray(image)
        self._theta_projection_full_title = str(title)
        if indices is None:
            return image, title
        try:
            axis, _order = normalize_theta_axis(self._current_map_theta_axis())
        except (TypeError, ValueError, AttributeError):
            return image, title
        low, high = indices
        if (low, high) == (0, axis.size - 1):
            return image, title
        if not (0 <= low < high < axis.size):
            return image, title
        cropped = self._theta_projection_image(kind, low, high)
        if cropped.shape != np.asarray(image).shape:
            raise ValueError(
                "Форма пересчитанной θ-проекции не совпала с текущей 2D-картой: "
                f"{cropped.shape} != {np.asarray(image).shape}."
            )
        return (
            cropped,
            f"{title} · θ ∈ [{axis[low]:.7g}°, {axis[high]:.7g}°]",
        )

    def _refresh_theta_range_preview(self) -> None:
        """Fast hot path for 2D range interaction; leave all static artists alone."""
        if self._theta_slice_mode:
            self._theta_range_finalize_pending = False
            return
        kind = self._current_theta_projection_kind()
        indices = self.map_widget.tools.theta_range_indices
        base_image = self._theta_projection_full_image
        if kind is None or indices is None or base_image is None:
            self._theta_range_finalize_pending = False
            return

        try:
            axis, _order = normalize_theta_axis(self._current_map_theta_axis())
            low, high = (int(indices[0]), int(indices[1]))
            if not (0 <= low < high < axis.size):
                return
            if (low, high) == (0, axis.size - 1):
                image = base_image
                title = self._theta_projection_full_title
            else:
                image = self._theta_projection_image(kind, low, high)
                if image.shape != base_image.shape:
                    raise ValueError(
                        "Форма пересчитанной θ-проекции не совпала с исходной картой: "
                        f"{image.shape} != {base_image.shape}."
                    )
                title = (
                    f"{self._theta_projection_full_title} · θ ∈ "
                    f"[{axis[low]:.7g}°, {axis[high]:.7g}°]"
                )

            clim = None
            if self._theta_range_finalize_pending:
                # Auto-contrast is refreshed once on release, not at every
                # mouse-move event. During the gesture the existing limits are
                # held fixed, which also makes adjacent θ intervals comparable.
                values = np.asarray(image, dtype=np.float64)
                finite = values[np.isfinite(values)]
                if finite.size:
                    p_lo = float(self.map_widget.low_pct.value())
                    p_hi = float(self.map_widget.high_pct.value())
                    if p_hi <= p_lo:
                        p_hi = min(100.0, p_lo + 1.0)
                    vmin, vmax = np.percentile(finite, [p_lo, p_hi])
                    if vmin >= vmax:
                        vmax = vmin + max(abs(float(vmin)) * 1e-9, 1e-12)
                    clim = (float(vmin), float(vmax))

            self.map_widget.update_theta_preview_image(image, title, clim=clim)
        except (ValueError, TypeError, IndexError, RuntimeError) as exc:
            self.status.showMessage(
                f"Предпросмотр диапазона θ не обновлён: {exc}", 8000
            )
        finally:
            self._theta_range_finalize_pending = False

    def _sync_volume_theta_context(self) -> None:
        """Tell the 3D panel whether its theta slider can link to the 2D map."""
        enabled = bool(self._theta_slice_mode)
        axis = self._current_map_theta_axis() if enabled else np.empty(0, dtype=np.float64)
        index = int(self.map_widget.theta_slider.value()) if axis.size else 0
        self._theta_index = max(0, min(index, max(0, axis.size - 1)))
        self.volume_widget.set_map_theta_context(enabled, axis, self._theta_index)

    def _set_theta_axis_for_current_scan(self):
        try:
            theta = self._current_map_theta_axis()
            if theta.size == 0:
                self.map_widget.set_theta_axis([])
                self._sync_volume_theta_context()
                return
            self._theta_index = max(0, min(self._theta_index, len(theta)-1))
            self.map_widget.set_theta_axis(theta, self._theta_index)
            self._sync_volume_theta_context()
        except Exception as exc:
            self.map_widget.set_theta_axis([])
            self._sync_volume_theta_context()
            if self._theta_slice_mode: self.status.showMessage(f"Срез по θ недоступен: {exc}", 8000)

    def _current_theta_value(self):
        if not self._theta_slice_mode: return None
        try:
            theta = self._current_map_theta_axis()
            idx=max(0,min(int(self.map_widget.theta_slider.value()),len(theta)-1))
            return float(theta[idx]) if len(theta) else None
        except Exception: return None

    def _on_theta_guide_toggled(self, enabled: bool) -> None:
        self._show_theta_guide=bool(enabled)
        self.curve_widget.set_theta_guide(self._current_theta_value() if self._show_theta_guide else None)

    def _on_theta_index_changed(self, index: int) -> None:
        self._theta_index = int(index)
        if self._theta_slice_mode:
            axis = self._current_map_theta_axis()
            if 0 <= self._theta_index < axis.size:
                self.volume_widget.set_map_theta_index(self._theta_index)
                self.volume_widget.set_theta_index_from_map(float(axis[self._theta_index]))
            self._theta_render_timer.start()
        if self._show_theta_guide: self.curve_widget.set_theta_guide(self._current_theta_value() if self._theta_slice_mode else None)

    def _on_vtk_theta_slice_changed(self, theta_value: float) -> None:
        """Mirror a user-driven 3D θ change back to the 2D map by value."""
        if not self._theta_slice_mode:
            return
        axis = self._current_map_theta_axis()
        if axis.size < 2:
            return
        try:
            value = float(theta_value)
        except (TypeError, ValueError, OverflowError):
            return
        if not np.isfinite(value):
            return
        index = int(np.argmin(np.abs(axis - value)))
        self._theta_index = index
        self.volume_widget.set_map_theta_index(index)
        self.map_widget.set_theta_index(index, emit=False)
        if self._show_theta_guide:
            self.curve_widget.set_theta_guide(self._current_theta_value())
        # Use the normal map refresh path; VTK has already updated synchronously.
        self._theta_render_timer.start(0)

    def _on_map_changed(self, *args) -> None:
        combo = self.map_widget.map_combo
        if combo.currentData() == "__theta_slice__":
            self._theta_slice_mode = True
            self._set_analysis_contrast_mode(None)
            self._set_theta_axis_for_current_scan(); self.map_widget.set_theta_slice_mode(True)
            self._sync_theta_range_context()
            self.refresh_map()
            if self._show_theta_guide: self.curve_widget.set_theta_guide(self._current_theta_value())
            return
        self._theta_slice_mode = False
        self._set_analysis_contrast_mode(self._analysis_map_key if self._analysis_view is not None else None)
        self.map_widget.set_theta_slice_mode(False)
        self._sync_volume_theta_context()
        if self._show_theta_guide: self.curve_widget.set_theta_guide(None)
        value = combo.currentData() or combo.currentText()
        if value:
            self.selected_map = str(value)
            self._sync_theta_range_context()
            self.refresh_map(); self.refresh_curves(update_volume=False)

    def _invalidate_curve_cache(self) -> None:
        """Очищает производные профили, зависящие от view или режима редукции."""
        for rois in self.session_scans.values():
            for saved in rois:
                saved.curve = None
                saved.count = 0
        self.preview_curve = None
        self.preview_th = None
        self.preview_count = 0    
        
    @staticmethod
    def _pool_label(pool: PoolSpec) -> str:
        return "Raw (1×1)" if not pool.active else f"Count-sum {pool.size}×{pool.size}, stride {pool.stride}"

    def _on_pool_changed(self, pool: PoolSpec) -> None:
        if self.repo is None or self.selected_scan is None:
            self.map_widget.set_pool_spec(PoolSpec())
            return
        old_pool = self.repo.pool
        source = self.repo.load_source(self.selected_scan)
        try:
            pooled_shape(source.source_shape_yx, pool)
            self.repo.set_pool(pool)
            self.repo.load_scan(self.selected_scan)
        except Exception as exc:
            self.repo.set_pool(old_pool)
            self.map_widget.set_pool_spec(old_pool, source_shape_yx=source.source_shape_yx)
            self.status.showMessage(f"Spatial view unchanged: {exc}")
            QMessageBox.warning(self, "Spatial pooling", str(exc))
            return

        self.map_widget.set_pool_spec(pool, source_shape_yx=source.source_shape_yx)
        self._theta_projection_accumulator.clear()
        self._invalidate_curve_cache()
        self.selected_pixel = None  # Индексы пикселей принадлежат конкретной grid.
        self._preview_timer.stop()
        self._populate_map_combo()  # Карты/профили теперь относятся к новому view.
        self.map_widget.reset_view()
        if self.preview_roi is not None:
            self._preview_timer.start(0)
        self._update_inspector()
        self.status.showMessage(f"Spatial view: {self._pool_label(pool)}")
        self._request_volume_refresh()

    def _set_processing_busy(self, busy: bool) -> None:
        # Во время исполнения нельзя переключать источник под активным workflow.
        self.open_h5_btn.setEnabled(not busy)
        self.refresh_scans_btn.setEnabled(not busy and self.repo is not None)
        self.map_widget.scan_combo.setEnabled(not busy)
        self.map_widget.pool_btn.setEnabled(not busy and self._analysis_view is None and self.repo is not None)

    def _analysis_display_key(self, state) -> tuple[str, str, str]:
        metadata = getattr(state, "metadata", {}) or {}
        source = str(metadata.get("source_file", self.current_file or ""))
        scan = str(metadata.get("scan_key", self.selected_scan or ""))
        node = str(getattr(state, "node_id", metadata.get("workflow_node_id", "")))
        return source, scan, node

    def _restore_analysis_source_colormap(self) -> None:
        """Restore the palette that was active before entering a result preview."""
        previous = self._analysis_previous_cmap
        self._analysis_previous_cmap = None
        if previous is None:
            return
        set_colormap = getattr(self.map_widget, "set_colormap", None)
        if callable(set_colormap):
            set_colormap(str(previous))
        else:  # compatibility with alternate map-widget adapters
            self.map_widget.current_cmap = str(previous)
            self.map_widget.tools.set_current_cmap(str(previous))

    def _remember_analysis_display_preferences(self) -> None:
        """Persist stable workflow-map preferences, never a transient branch map.

        ``branch.parameter.*`` layers are derived from the current branch
        selection and must not become the remembered map for the workflow
        node.  Otherwise switching to another preview can persist a stale
        branch raster key and the next visit may show another branch's data.
        """
        identity = self._analysis_display_identity
        if identity is None:
            return
        map_key = self._analysis_map_key
        branch_layer = self._branch_parameter_layer
        is_transient_branch_map = bool(
            branch_layer is not None and map_key == str(branch_layer.key)
        )
        if is_transient_branch_map:
            map_key = self._branch_parameter_return_map_key
        self._analysis_display_preferences[identity] = (
            map_key, self._analysis_curve_key,
        )
        if map_key and not is_transient_branch_map:
            self._analysis_layer_cmap_preferences[(identity, str(map_key))] = str(
                self.map_widget.current_cmap
            )

    def _apply_analysis_layer_colormap(self, identity, map_key: str | None, *, default_hkl: bool) -> None:
        if not map_key:
            return
        saved = self._analysis_layer_cmap_preferences.get((identity, str(map_key)))
        if saved:
            cmap_name = str(saved)
        elif default_hkl and str(map_key).startswith("hkl.delta_theta."):
            cmap_name = "RdBu_r"
        else:
            # Unsaved maps in the same preview session inherit the source-map
            # palette, not the semantic Δθ diverging palette left by the prior
            # layer. Saved per-layer preferences still take precedence above.
            cmap_name = self._analysis_previous_cmap or self.map_widget.current_cmap
        self.map_widget.current_cmap = cmap_name
        self.map_widget.tools.set_current_cmap(cmap_name)

    def _analysis_menu_items(self, state, attribute: str):
        items = tuple(getattr(state, attribute, ()) or ()) if state is not None else ()
        if (
            attribute == "map_layers" and state is self._analysis_view
            and self._branch_parameter_layer is not None
        ):
            items = (*items, self._branch_parameter_layer)
        return items

    def _build_analysis_result_menu(self, menu, items, selected_key, callback) -> None:
        previous_group = getattr(menu, "_analysis_result_action_group", None)
        if previous_group is not None:
            previous_group.deleteLater()
        menu.clear()
        groups = {}
        action_group = QActionGroup(menu)
        action_group.setObjectName("analysisResultExclusiveActions")
        action_group.setExclusive(True)
        menu._analysis_result_action_group = action_group
        for item in items:
            group_name = str(getattr(item, "group", "Результаты"))
            target_menu = groups.get(group_name)
            if target_menu is None:
                target_menu = menu.addMenu(group_name) if len({getattr(x, "group", "Результаты") for x in items}) > 1 else menu
                groups[group_name] = target_menu
            action = QAction(str(item.label), action_group)
            action.setData(str(item.key))
            action.setCheckable(True)
            action.setChecked(str(item.key) == str(selected_key))
            description = str(getattr(item, "description", "") or "")
            unit = str(getattr(item, "unit", "") or "")
            tooltip = description
            if unit:
                tooltip = (tooltip + "\n" if tooltip else "") + f"Единица: {unit}"
            if tooltip:
                action.setToolTip(tooltip)
            action.triggered.connect(lambda checked=False, key=str(item.key): callback(key))
            target_menu.addAction(action)

    @staticmethod
    def _set_result_menu_checked(menu, selected_key: str | None) -> None:
        def update(current_menu):
            for action in current_menu.actions():
                submenu = action.menu()
                if submenu is not None:
                    update(submenu)
                elif action.isCheckable():
                    action.setChecked(str(action.data()) == str(selected_key))
        update(menu)

    def _sync_analysis_result_menus(self, state) -> None:
        map_items = self._analysis_menu_items(state, "map_layers")
        curve_items = self._analysis_menu_items(state, "curve_sources")
        map_button = self.map_widget.tools.results_btn
        curve_button = self.curve_widget.toolbar.results_btn
        self._build_analysis_result_menu(
            self.map_widget.tools.results_menu, map_items, self._analysis_map_key,
            self.set_analysis_map_layer,
        )
        self._build_analysis_result_menu(
            self.curve_widget.toolbar.results_menu, curve_items, self._analysis_curve_key,
            self.set_analysis_curve_source,
        )
        active_theme = resolve_theme(self)
        apply_popup_theme(self.map_widget.tools.results_menu, active_theme)
        apply_popup_theme(self.curve_widget.toolbar.results_menu, active_theme)
        map_button.setEnabled(bool(state is not None and map_items and not self._theta_slice_mode))
        curve_button.setEnabled(bool(state is not None and curve_items))
        if state is None:
            map_button.setToolTip("Нет активного предпросмотра workflow")
            curve_button.setToolTip("Нет активного предпросмотра workflow")
        else:
            map_button.setToolTip("Выбрать пространственную карту текущего узла workflow")
            curve_button.setToolTip("Выбрать θ-зависимый сигнал текущего узла workflow")

    @staticmethod
    def _is_quantitative_analysis_map(key: str | None) -> bool:
        """Numerical workflow result maps, excluding source intensities and the bespoke peak map."""
        value = str(key or "")
        return bool(value and not value.startswith("scan.") and value != "peaks.count")

    def _set_contrast_percentiles(self, low: float, high: float) -> None:
        """Update both spin boxes and the range rail without triggering duplicate redraws."""
        widgets = (self.map_widget.low_pct, self.map_widget.high_pct, self.map_widget.range_slider)
        previous = [widget.blockSignals(True) for widget in widgets]
        try:
            self.map_widget.low_pct.setValue(float(low))
            self.map_widget.high_pct.setValue(float(high))
            self.map_widget.range_slider.setLow(float(low))
            self.map_widget.range_slider.setHigh(float(high))
        finally:
            for widget, was_blocked in zip(widgets, previous):
                widget.blockSignals(was_blocked)

    def _set_analysis_contrast_mode(self, key: str | None) -> None:
        """Temporarily use full linear percentiles for workflow result metrics.

        The user's previous source-map percentiles are captured once on entry
        and restored when leaving numerical results.  peaks.count remains on
        its dedicated hue/alpha renderer and is intentionally excluded.
        """
        quantitative = bool(
            self._analysis_view is not None
            and not self._theta_slice_mode
            and self._is_quantitative_analysis_map(key)
        )
        if quantitative and not self._analysis_quantitative_contrast:
            self._analysis_contrast_snapshot = (
                float(self.map_widget.low_pct.value()),
                float(self.map_widget.high_pct.value()),
            )
            self._set_contrast_percentiles(0.0, 100.0)
        elif not quantitative and self._analysis_quantitative_contrast:
            low, high = self._analysis_contrast_snapshot or (1.0, 99.0)
            self._set_contrast_percentiles(low, high)
            self._analysis_contrast_snapshot = None
        self._analysis_quantitative_contrast = quantitative

    def set_analysis_map_layer(self, key: str) -> None:
        state = self._analysis_view
        if state is None or not hasattr(state, "map_layers"):
            return
        key = str(key)
        available_layers = self._analysis_menu_items(state, "map_layers")
        if not any(layer.key == key for layer in available_layers):
            return

        is_current_branch_layer = bool(
            self._branch_parameter_layer is not None
            and key == str(self._branch_parameter_layer.key)
        )
        if is_current_branch_layer:
            branch_id = self._branch_parameter_branch_id
            parameter = self._branch_parameter_current_parameter
            if branch_id is not None and parameter:
                self._branch_parameter_view_enabled = True
                self._branch_parameter_choices.remember(int(branch_id), str(parameter))
        else:
            # Explicitly choosing a normal workflow map leaves branch-map
            # mode. Do not store the transient branch.parameter.* key as a
            # normal per-node display preference.
            if self._branch_parameter_layer is None:
                self._remember_analysis_display_preferences()
            self._branch_parameter_view_enabled = False
            self._branch_parameter_return_map_key = None
            self._branch_parameter_layer = None
            self._branch_parameter_branch_id = None
            self._branch_parameter_current_parameter = None

        self._analysis_map_key = key
        self._apply_analysis_layer_colormap(
            self._analysis_display_identity, self._analysis_map_key, default_hkl=True,
        )
        self._set_analysis_contrast_mode(self._analysis_map_key)
        if not is_current_branch_layer:
            self._remember_analysis_display_preferences()
        self._set_result_menu_checked(self.map_widget.tools.results_menu, self._analysis_map_key)
        self._sync_analysis_result_menus(state)
        self._sync_theta_range_context()
        self.refresh_map()
        self._update_inspector()

    def set_analysis_curve_source(self, key: str) -> None:
        state = self._analysis_view
        if state is None or not hasattr(state, "curve_sources"):
            return
        if not any(source.key == str(key) for source in state.curve_sources):
            return
        self._analysis_curve_key = str(key)
        self._remember_analysis_display_preferences()
        self._set_result_menu_checked(self.curve_widget.toolbar.results_menu, self._analysis_curve_key)
        self._analysis_curve_cache.clear()
        self.refresh_curves(update_volume=False)
        self._update_inspector()

    def _set_peak_pixel_pick_mode(
        self, enabled: bool, parent_node_id: str, use_interpolated_data: bool = False
    ) -> None:
        """Enable persistent detector-pixel picking for peak-preview QC."""
        if not enabled:
            self.map_widget.set_pixel_pick_mode(False)
            return
        state = self._analysis_view
        if state is None or str(getattr(state, "node_id", "")) != str(parent_node_id):
            self.map_widget.set_pixel_pick_mode(False)
            self.processing_panel.cancel_peak_pixel_pick_mode()
            self.status.showMessage(
                "2D-карта ещё не показывает вход выбранной ветви. Дождитесь её обновления и повторите выбор.",
                8000,
            )
            return
        # The selected detector coordinate must be visible on the input map,
        # not on the downstream spike/peak count layer.
        input_map_key = (
            "scan.intensity_sum" if use_interpolated_data
            else "scan.measured_intensity_sum"
        )
        if not any(
            layer.key == input_map_key for layer in getattr(state, "map_layers", ())
        ):
            input_map_key = "scan.intensity_sum"
        self.set_analysis_map_layer(input_map_key)
        self.map_widget.set_pixel_pick_mode(True)
        self.status.showMessage(
            "Выбор пикселя активен: щёлкайте по 2D-карте, чтобы менять реальную ROC. "
            "Закройте окно параметров кнопкой × или Esc, чтобы отменить выбор.",
            10000,
        )

    def show_analysis_preview(self, state) -> None:
        """Подключает один узел workflow к карте и ROC как согласованную версию."""
        if getattr(self.map_widget, "_pixel_pick_mode", False):
            self.map_widget.set_pixel_pick_mode(False)
            self.processing_panel.cancel_peak_pixel_pick_mode()
        if self.repo is None or self.selected_scan is None:
            return
        metadata = getattr(state, "metadata", {}) or {}
        if str(metadata.get("scan_key", "")) != str(self.selected_scan):
            self.status.showMessage("Предпросмотр относится к другому скану; переключение отменено", 8000)
            return
        source_file = metadata.get("source_file")
        if source_file and Path(str(source_file)).expanduser().resolve() != Path(self.current_file).resolve():
            self.status.showMessage("Предпросмотр относится к другому HDF5-файлу", 8000)
            return
        self._remember_analysis_display_preferences()
        if self._analysis_view is None and self._analysis_previous_cmap is None:
            self._analysis_previous_cmap = str(self.map_widget.current_cmap)
        identity = self._analysis_display_key(state)
        if self._analysis_previous_pool is None:
            self._analysis_previous_pool = self.repo.pool
        if self.repo.pool != PoolSpec():
            self.repo.set_pool(PoolSpec())
            source = self.repo.load_source(self.selected_scan)
            self.map_widget.set_pool_spec(PoolSpec(), source_shape_yx=source.source_shape_yx)
        self.map_widget.set_pool_enabled(False)
        self._analysis_view = state
        self._analysis_display_identity = identity
        self._install_branch_context(state)
        saved = self._analysis_display_preferences.get(identity, (None, None))
        map_keys = {layer.key for layer in getattr(state, "map_layers", ())}
        curve_keys = {source.key for source in getattr(state, "curve_sources", ())}
        self._analysis_map_key = saved[0] if saved[0] in map_keys else getattr(state, "default_map_key", None)
        self._analysis_curve_key = saved[1] if saved[1] in curve_keys else getattr(state, "default_curve_key", None)
        self._apply_analysis_layer_colormap(identity, self._analysis_map_key, default_hkl=True)
        self._set_analysis_contrast_mode(self._analysis_map_key)
        self._sync_analysis_result_menus(state)
        self._analysis_curve_cache.clear()
        self.selected_pixel = None
        self._theta_scan_cache_key = None; self._theta_scan_cache = None
        self._theta_projection_accumulator.clear()
        self._set_theta_axis_for_current_scan()
        self._sync_theta_range_context()
        self.refresh_map()
        self.refresh_curves()
        self._update_inspector()
        self._request_volume_refresh()
        hkl_preview_warning = metadata.get("hkl_identification_preview_warning")
        if hkl_preview_warning:
            self.status.showMessage(str(hkl_preview_warning), 15000)
        else:
            self.status.showMessage(f"Workflow preview: {state.label}")

    def _branch_display_path(self) -> Path | None:
        if self.current_file is None or self.selected_scan is None:
            return None
        try:
            root = resolve_analysis_projects_directory()
            project_dir = project_directory_for_scan(self.current_file, self.selected_scan, root)
            return project_dir / "branch_display_state.json"
        except (OSError, ValueError, TypeError):
            return None

    def _clear_branch_context(self) -> None:
        """Clear transient branch selections and overlays when leaving a preview."""
        self._branch_catalog = None
        self._branch_fingerprint = None
        self._branch_display_colors = {}
        self._branch_display_state_path = None
        self._branch_parameter_layer = None
        self._branch_parameter_branch_id = None
        self._branch_parameter_choices.reset()
        self._branch_parameter_view_enabled = False
        self._branch_parameter_return_map_key = None
        self._branch_parameter_current_parameter = None
        manager = getattr(self, "branch_manager", None)
        workspace = getattr(self, "plot_workspace", None)
        if manager is not None:
            manager.set_branches((), {}, ())
        if workspace is not None:
            workspace.set_branch_manager_visible(False)
        map_widget = getattr(self, "map_widget", None)
        if map_widget is not None:
            map_widget.clear_branch_footprints(draw=False)
        volume = getattr(self, "volume_widget", None)
        if volume is not None:
            volume.set_branch_display_state({}, (), focus_selected=False)

    def _install_branch_context(self, state) -> None:
        """Install the branch catalogue from the active immutable workflow node."""
        catalog = getattr(state, "branch_catalog", None)
        required = {"branch_id", "x", "y"}
        if catalog is None or getattr(catalog, "empty", True) or not required.issubset(getattr(catalog, "columns", ())):
            self._clear_branch_context()
            return
        try:
            summaries = summarize_branch_catalog(catalog)
            fingerprint = branch_catalog_fingerprint(catalog)
            branch_ids = [int(item["branch_id"]) for item in summaries]
            sidecar_path = self._branch_display_path()
            saved_colors = (
                load_branch_display_colors(sidecar_path, fingerprint)
                if sidecar_path is not None else {}
            )
            colors = normalize_branch_colors(branch_ids, saved_colors)
            parameter_fields = available_branch_parameters(catalog)
        except (ValueError, TypeError, KeyError) as exc:
            _logger.exception("Не удалось подготовить менеджер ветвей")
            self._clear_branch_context()
            self.status.showMessage(f"Менеджер ветвей недоступен: {exc}", 10000)
            return
        self._branch_catalog = catalog
        self._branch_fingerprint = fingerprint
        self._branch_display_colors = colors
        self._branch_display_state_path = sidecar_path
        context_signature = (self._analysis_display_identity, str(fingerprint))
        self._branch_parameter_choices.bind_context(context_signature)
        # Re-installing a workflow preview rebuilds the menu and branch model.
        # Do not keep a stale PreviewMapLayer object tied to the old state.
        self._branch_parameter_layer = None
        self._branch_parameter_branch_id = None
        self._branch_parameter_view_enabled = False
        self._branch_parameter_return_map_key = None
        self._branch_parameter_current_parameter = None
        self.branch_manager.set_branches(summaries, colors, parameter_fields)
        self.plot_workspace.set_branch_manager_visible(bool(summaries))
        self.volume_widget.set_branch_display_state(colors, (), focus_selected=False)
        self._sync_analysis_result_menus(state)

    def _sync_branch_display_to_volume(self) -> None:
        manager = getattr(self, "branch_manager", None)
        self.volume_widget.set_branch_display_state(
            self._branch_display_colors,
            manager.selected_branch_ids if manager is not None else (),
            focus_selected=bool(manager is not None and manager.focus_3d_enabled),
        )

    def _on_branch_selection_changed(self, branch_ids) -> None:
        selected = tuple(sorted({int(value) for value in (branch_ids or ())}))
        if len(selected) == 1 and self._branch_catalog is not None:
            fields = available_branch_parameters(self._branch_catalog, branch_id=selected[0])
            self.branch_manager.set_parameter_fields(fields)
        else:
            fields = []
            self.branch_manager.set_parameter_fields(())

        if self._branch_parameter_view_enabled:
            if len(selected) == 1:
                branch_id = selected[0]
                available = [str(item.get("key", "")) for item in fields]
                parameter = self._branch_parameter_choices.preferred(
                    branch_id, available, fallback=available[0] if available else None,
                )
                if parameter:
                    try:
                        self._activate_branch_parameter_layer(
                            branch_id, parameter, remember_return_map=False,
                            refresh_map=not self.branch_manager.footprints_enabled,
                        )
                    except (ValueError, TypeError, KeyError, IndexError) as exc:
                        # A model-specific value (for example eta) may not be
                        # present for a branch fitted with another model. Fall
                        # back once to the RCI position map, which is common to
                        # every valid branch event, rather than leaving stale
                        # pixels from the previously selected branch visible.
                        position = next(
                            (str(item["key"]) for item in fields
                             if str(item["key"]) in {"theta0", "theta", "theta_fit", "center_theta", "theta_detected"}),
                            None,
                        )
                        if position and position != parameter:
                            try:
                                self._activate_branch_parameter_layer(
                                    branch_id, position, remember_return_map=False,
                                    refresh_map=not self.branch_manager.footprints_enabled,
                                )
                            except (ValueError, TypeError, KeyError, IndexError):
                                _logger.debug(
                                    "Для ветви %s нет конечных значений параметров карты",
                                    branch_id, exc_info=True,
                                )
                                self._leave_branch_parameter_layer(restore=True, keep_mode=True)
                        else:
                            _logger.debug(
                                "Для ветви %s нет конечных значений параметров карты",
                                branch_id, exc_info=True,
                            )
                            self._leave_branch_parameter_layer(restore=True, keep_mode=True)
                else:
                    self._leave_branch_parameter_layer(restore=True, keep_mode=True)
            else:
                # Multi-selection / cleared selection hides the quantitative
                # branch map, but the per-branch choice remains remembered.
                self._leave_branch_parameter_layer(restore=True, keep_mode=True)
        elif self._branch_parameter_layer is not None:
            self._leave_branch_parameter_layer(restore=True, keep_mode=False)

        self._sync_branch_display_to_volume()
        # BranchManagerWidget emits footprints_toggled(True) after a selection
        # change when the overlay is enabled. That signal owns the one redraw;
        # doing it here as well would render the same detector map twice.

    def _on_branch_color_changed(self, branch_id: int, color: str) -> None:
        try:
            branch_id = int(branch_id)
            parsed = QColor(str(color))
            if not parsed.isValid() or branch_id not in self._branch_display_colors:
                return
            normalized = parsed.name(QColor.NameFormat.HexRgb).upper()
            conflicting_branch = next(
                (
                    other_id for other_id, other_color in self._branch_display_colors.items()
                    if int(other_id) != branch_id
                    and QColor(str(other_color)).name(QColor.NameFormat.HexRgb).upper() == normalized
                ),
                None,
            )
            if conflicting_branch is not None:
                self.status.showMessage(
                    f"Цвет {normalized} уже назначен ветви {conflicting_branch}. "
                    "Выберите другой цвет.",
                    8000,
                )
                return
            if not self.branch_manager.set_branch_color(branch_id, normalized):
                self.status.showMessage(
                    f"Цвет {normalized} уже используется другой ветвью.", 8000,
                )
                return
            self._branch_display_colors[branch_id] = normalized
            if self._branch_display_state_path is not None and self._branch_fingerprint:
                save_branch_display_colors(
                    self._branch_display_state_path, self._branch_fingerprint,
                    self._branch_display_colors,
                )
        except (OSError, ValueError, TypeError) as exc:
            self.status.showMessage(f"Не удалось сохранить цвет ветви: {exc}", 10000)
        self._sync_branch_display_to_volume()
        if self.branch_manager.footprints_enabled:
            self.refresh_map()

    def _on_branch_footprints_toggled(self, enabled: bool) -> None:
        if enabled and not self.branch_manager.selected_branch_ids:
            return
        if self._analysis_view is None:
            self.map_widget.clear_branch_footprints(draw=True)
            return
        self.refresh_map()

    def _on_branch_focus_3d_toggled(self, enabled: bool) -> None:
        self._sync_branch_display_to_volume()
        if enabled and not self.plot_workspace.mdi_area.is_panel_visible("volume"):
            self.plot_workspace.mdi_area.show_panel("volume", True)
            self._request_volume_refresh()

    def _build_branch_parameter_layer(self, branch_id: int, parameter: str):
        state = self._analysis_view
        catalog = self._branch_catalog
        if state is None or catalog is None:
            raise ValueError("Нет активного каталога ветвей")
        shape_yx = tuple(map(int, np.shape(state.data_yxt)[:2]))
        image = build_branch_parameter_map(catalog, int(branch_id), str(parameter), shape_yx)
        field = next(
            (item for item in available_branch_parameters(catalog, branch_id=int(branch_id))
             if item["key"] == str(parameter)),
            None,
        )
        if field is None:
            raise ValueError(f"Параметр {parameter!r} больше не доступен в текущем Fit-каталоге")
        if not np.isfinite(image).any():
            raise ValueError("У выбранной ветви нет конечных значений этого параметра")
        from roc_mcs.analysis.workflow_preview import PreviewMapLayer
        layer = PreviewMapLayer(
            key=f"branch.parameter.{int(branch_id)}.{str(parameter)}",
            label=f"Ветвь {int(branch_id)} · {field['label']}",
            data_yx=image,
            group=f"Параметры ветви {int(branch_id)}",
            unit=str(field.get("unit", "")),
            description=(
                f"Значение {parameter} для событий ветви {int(branch_id)}; "
                "при нескольких событиях одной ветви в одном пикселе отображается их среднее. "
                "Вне детекторных пикселей выбранной ветви значения отсутствуют."
            ),
        )
        return layer

    def _leave_branch_parameter_layer(
        self, *, restore: bool, keep_mode: bool,
    ) -> None:
        """Remove the transient branch layer and optionally restore the prior map."""
        old_layer = self._branch_parameter_layer
        was_active = bool(
            old_layer is not None and self._analysis_map_key == str(old_layer.key)
        )
        self._branch_parameter_layer = None
        self._branch_parameter_branch_id = None
        self._branch_parameter_current_parameter = None
        self._branch_parameter_view_enabled = bool(keep_mode)
        if restore and was_active:
            state = self._analysis_view
            map_keys = {str(item.key) for item in getattr(state, "map_layers", ())}
            fallback = self._branch_parameter_return_map_key
            if fallback not in map_keys:
                fallback = getattr(state, "default_map_key", None)
            self._analysis_map_key = str(fallback) if fallback is not None else None
            self._apply_analysis_layer_colormap(
                self._analysis_display_identity, self._analysis_map_key, default_hkl=True,
            )
            self._set_analysis_contrast_mode(self._analysis_map_key)
            self._sync_theta_range_context()
        if not keep_mode:
            self._branch_parameter_return_map_key = None
        self._sync_analysis_result_menus(self._analysis_view)
        if restore and was_active:
            self._set_result_menu_checked(
                self.map_widget.tools.results_menu, self._analysis_map_key,
            )
            if not self.branch_manager.footprints_enabled:
                self.refresh_map()
            self._update_inspector()

    def _activate_branch_parameter_layer(
        self, branch_id: int, parameter: str, *, remember_return_map: bool,
        refresh_map: bool = True,
    ) -> bool:
        """Activate one branch's map and remember its lightweight selection key."""
        branch_id = int(branch_id)
        parameter = str(parameter)
        layer = self._build_branch_parameter_layer(branch_id, parameter)
        if remember_return_map and not self._branch_parameter_view_enabled:
            self._branch_parameter_return_map_key = str(self._analysis_map_key or "") or None
        self._branch_parameter_view_enabled = True
        self._branch_parameter_choices.remember(branch_id, parameter)
        self._branch_parameter_current_parameter = parameter
        self._branch_parameter_layer = layer
        self._branch_parameter_branch_id = branch_id
        self._analysis_map_key = layer.key
        self._apply_analysis_layer_colormap(
            self._analysis_display_identity, self._analysis_map_key, default_hkl=False,
        )
        self._set_analysis_contrast_mode(self._analysis_map_key)
        self._sync_analysis_result_menus(self._analysis_view)
        self._set_result_menu_checked(self.map_widget.tools.results_menu, self._analysis_map_key)
        self._sync_theta_range_context()
        if refresh_map:
            self.refresh_map()
        self._update_inspector()
        return True

    def _on_branch_parameter_map_requested(self, branch_id: int, parameter: str) -> None:
        state = self._analysis_view
        catalog = self._branch_catalog
        if state is None or catalog is None:
            return
        selected = self.branch_manager.selected_branch_ids
        if len(selected) != 1 or int(selected[0]) != int(branch_id):
            self.status.showMessage("Карта параметра требует ровно одну выбранную ветвь", 8000)
            return
        try:
            self._activate_branch_parameter_layer(
                int(branch_id), str(parameter), remember_return_map=True,
            )
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            self.status.showMessage(f"Не удалось построить карту параметра ветви: {exc}", 10000)

    def _draw_branch_footprints_overlay(self) -> None:
        if (
            self._analysis_view is None or self._branch_catalog is None
            or not self.branch_manager.footprints_enabled
        ):
            self.map_widget.clear_branch_footprints(draw=False)
            return
        selected = self.branch_manager.selected_branch_ids
        if not selected:
            self.map_widget.clear_branch_footprints(draw=False)
            return
        shape_yx = tuple(map(int, np.shape(self._analysis_view.data_yxt)[:2]))
        layers = branch_footprint_layers(
            self._branch_catalog, selected, shape_yx, smooth_sigma=1.2,
        )
        count = self.map_widget.draw_branch_footprints(
            layers, self._branch_display_colors, alpha=0.45, contour_level=0.35,
        )
        if count == 0:
            self.status.showMessage("Для выбранных ветвей не найдено валидных XY-координат", 8000)

    def restore_analysis_preview(self) -> None:
        """Возвращает исходный HDF5-view, палитру и прежний spatial pooling."""
        self._remember_analysis_display_preferences()
        self._analysis_view = None
        self._clear_branch_context()
        self._restore_analysis_source_colormap()
        self._set_analysis_contrast_mode(None)
        self._analysis_map_key = None
        self._analysis_curve_key = None
        self._analysis_display_identity = None
        self._sync_analysis_result_menus(None)
        self._analysis_curve_cache.clear()
        if self.repo is not None and self._analysis_previous_pool is not None:
            previous = self._analysis_previous_pool
            self._analysis_previous_pool = None
            self.repo.set_pool(previous)
            source = self.repo.load_source(self.selected_scan) if self.selected_scan else None
            self.map_widget.set_pool_spec(previous, source_shape_yx=source.source_shape_yx if source else None)
        self.map_widget.set_pool_enabled(self.repo is not None)
        self._theta_scan_cache_key = None; self._theta_scan_cache = None
        self._theta_projection_accumulator.clear()
        self._set_theta_axis_for_current_scan()
        self._sync_theta_range_context()
        self.refresh_all()
        self.status.showMessage("Исходный HDF5-view восстановлен")

    def _analysis_curve(self, state, roi):
        source = state.curve_source(self._analysis_curve_key) if hasattr(state, "curve_source") else None
        data = source.data_yxt if source is not None else state.data_yxt
        theta = source.theta if source is not None else state.theta
        valid_mask = source.valid_mask_yxt if source is not None else state.valid_mask_yxt
        source_key = source.key if source is not None else "legacy"
        geometry = (round(float(roi.cx), 7), round(float(roi.cy), 7), round(float(roi.width), 7), round(float(roi.height), 7), round(float(roi.angle), 7), str(roi.type), self.reduction_mode)
        key = (state.node_id, source_key, geometry)
        cached = self._analysis_curve_cache.get(key)
        if cached is None:
            th, curve, count = extract_from_roi(
                data, theta, 2, roi,
                data.shape[1], data.shape[0],
                reduction_mode=self.reduction_mode, x_coords=state.x_coords,
                y_coords=state.y_coords, valid_mask=valid_mask,
            )
            cached = (np.asarray(th), np.asarray(curve), int(count))
            if len(self._analysis_curve_cache) > 128:
                self._analysis_curve_cache.pop(next(iter(self._analysis_curve_cache)))
            self._analysis_curve_cache[key] = cached
        return cached

    def _on_pixel_clicked(self, x_index: int, y_index: int) -> None:
        """Показывает ROC выбранного пикселя из активной версии данных."""
        if self.map_widget.roi_btn.isChecked() or self.repo is None or self.selected_scan is None:
            return
        if self._analysis_view is not None:
            data = self._analysis_view.data_yxt
            ny, nx = data.shape[:2]
            x_coords, y_coords = self._analysis_view.x_coords, self._analysis_view.y_coords
        else:
            scan = self.repo.load_scan(self.selected_scan)
            ny, nx = scan.shape_yx
            x_coords, y_coords = self.repo.centers_xy(self.selected_scan)
        if not (0 <= x_index < nx and 0 <= y_index < ny):
            return
        self.selected_pixel = (int(x_index), int(y_index))
        self.status.showMessage(
            f"Pixel view=({x_index}, {y_index}); detector center=({x_coords[x_index]:.1f}, {y_coords[y_index]:.1f})"
        )
        self.refresh_curves(update_volume=False)
        self._update_inspector()

    def _on_reduction_changed(self, *args) -> None:
        self.reduction_mode = self.map_widget.reduction_combo.currentText().lower()
        for rois in self.session_scans.values():
            for saved in rois:
                saved.roi.mode = self.reduction_mode
        self._invalidate_curve_cache()
        self.refresh_curves(update_volume=False)
        if self.preview_roi is not None:
            self._preview_timer.start(0)        
        self._update_inspector()
        
    
    def reset_contrast_defaults(self) -> None:
        """Сбрасывает контраст на дефолтные 1.0% - 99.0%"""
        # Блокируем сигналы, чтобы карта не перерисовывалась дважды
        self.map_widget.low_pct.blockSignals(True)
        self.map_widget.high_pct.blockSignals(True)
        
        low, high = (0.0, 100.0) if self._analysis_quantitative_contrast else (1.0, 99.0)
        self.map_widget.low_pct.setValue(low)
        self.map_widget.high_pct.setValue(high)
        
        self.map_widget.low_pct.blockSignals(False)
        self.map_widget.high_pct.blockSignals(False)
        
        # Передаем обновленные значения в слайдер
        self.map_widget.range_slider.setLow(low)
        self.map_widget.range_slider.setHigh(high)
        
        self.refresh_map()

    
    def reset_map_view(self, *args) -> None:
        self.map_widget.reset_view()

    def _clear_preview_state(self) -> None:
        """Согласованно очищает незакреплённый ROI и его кривую."""
        had_preview = any(value is not None for value in (self.preview_roi, self.preview_curve, self.preview_th)) or self.preview_count != 0
        self._preview_timer.stop()
        self.preview_roi = self.preview_curve = self.preview_th = None
        self.preview_count = 0
        self._preview_fit_pending = False
        self._sync_label_box()
        if had_preview:
            self.refresh_curves(update_volume=False)
            self._update_inspector()

    def _on_selector_change(self, roi) -> None:
        if roi is None:
            self._clear_preview_state()
            return
        new_roi = ROI.from_selector_roi(roi, label=self.label_edit.text().strip(), color=theme_colors(resolve_theme(self))["roi"], mode=self.reduction_mode).normalized()
        if not new_roi.is_valid:
            self._clear_preview_state()
            return

        # Оставляем последнюю рассчитанную кривую на экране, пока не готова новая: без мигания.
        self.preview_roi = new_roi
        self._sync_label_box()
        self._preview_timer.start(20)

    def _on_selector_finished(self, roi) -> None:
        """Подгоняет кривую один раз по окончательной геометрии ROI."""
        if roi is None or self.repo is None or self.selected_scan is None:
            return
        self._preview_fit_pending = True
        self._preview_timer.start(20)

    def _update_preview_curve(self) -> None:
        if self.repo is None or self.selected_scan is None or self.preview_roi is None:
            return
        try:
            if self._analysis_view is not None:
                state = self._analysis_view
                th, curve, count = extract_from_roi(
                    state.data_yxt, state.theta, 2, self.preview_roi,
                    state.data_yxt.shape[1], state.data_yxt.shape[0],
                    reduction_mode=self.reduction_mode, x_coords=state.x_coords,
                    y_coords=state.y_coords, valid_mask=state.valid_mask_yxt,
                )
            else:
                th, curve, count = self.repo.curve_from_roi(
                    self.selected_scan, self.preview_roi,
                    reduction_mode=self.reduction_mode,
                )
        except Exception as exc:
            self.status.showMessage(f"Preview update failed: {exc}")
            return

        theta_values = np.asarray(th, dtype=float)
        curve_values = np.asarray(curve, dtype=float)
        self.preview_th = theta_values if theta_values.ndim == 1 else None
        self.preview_curve = curve_values if curve_values.ndim == 1 else np.array([], dtype=float)
        self.preview_count = int(count)
        aligned_preview = (
            theta_values.ndim == 1 and curve_values.ndim == 1
            and theta_values.size > 0 and theta_values.shape == curve_values.shape
        )
        has_finite_preview = bool(
            aligned_preview
            and (np.isfinite(theta_values) & np.isfinite(curve_values)).any()
        )
        fit_view = self._preview_fit_pending and has_finite_preview
        self._preview_fit_pending = False
        self.refresh_curves(
            fit_curve_view=fit_view, fit_curve_y=has_finite_preview,
            update_volume=False,
        )
        self._update_inspector()

    def pin_preview_roi(self, *args) -> None:
        if self.repo is None or self.selected_scan is None or self.preview_roi is None:
            self.status.showMessage("No preview ROI to save.")
            return
        if not self.preview_roi.is_valid:    # Защита от дурака: Объект сам знает, можно ли его сохранять. 
            self.status.showMessage("Preview ROI geometry is invalid.")
            return

        new_id = self._next_available_roi_id()   # получаем новый ID для сохраняемого ROI
        saved = SavedROI(
            id=new_id,
            roi=self.preview_roi.normalized(),
            # A preview curve belongs to the selected workflow revision. Do not
            # persist it as the raw-session cache; recompute after restoring raw.
            curve=(self.preview_curve.copy() if self.preview_curve is not None and self._analysis_view is None else None),
            count=int(self.preview_count) if self._analysis_view is None else 0,
        )
        label = self._generate_unique_label("ROI")       # Мы ищем свободное имя, игнорируя то, что кто-то переименовал ROI 1 в ROI 2
        saved.roi.label = label
        saved.roi.mode = self.reduction_mode
        saved.roi.color = self._color_for_roi(new_id)  # заменяем белый цвет превью на цвет из нашей палитры
        self.current_rois().append(saved)              # сохранение в базу (список)
        self.preview_roi = None
        self.preview_curve = None
        self.preview_th = None
        self.preview_count = 0
        self.map_widget._apply_current_roi_tool(active=False)
        if self.map_widget.roi_btn.isChecked():
            self.map_widget._apply_current_roi_tool(active=True)
        
        # One selection refresh updates the map, curves, inspector and ROI table.
        # Repeating it here rebuilt the same UI several times without changing data.
        self._set_selected_roi_id(saved.id)
        # Сохранение ROI меняет только каталог доступных источников. Если активный
        # 3D-источник не менялся, не пересобираем payload/VTK actors и не трогаем камеру.
        update_sources = getattr(self.volume_widget, "update_saved_rois", None)
        if callable(update_sources):
            update_sources(self.current_rois())
        self.status.showMessage(f"Saved {saved.roi.label}")


    def _on_roi_table_edited(self, roi_id: int, field: str, new_val_str: str) -> None:
        """Умный парсинг изменений из таблицы."""
        # 1. Ищем нужный ROI в памяти
        target_pinned = None
        for pinned in self.current_rois():
            if pinned.id == roi_id:
                target_pinned = pinned
                break
        if target_pinned is None:
            return

        # Табличный label не влияет на 3D-геометрию. Обновлять 3D нужно
        # только после реально принятого изменения геометрического поля.
        geometry_changed = False
        try:
            if field == "label":
                target_pinned.roi.label = new_val_str
            else:
                val = float(new_val_str)
                old_val = getattr(target_pinned.roi, field)
                setattr(target_pinned.roi, field, val)
                if not target_pinned.roi.is_valid:
                    setattr(target_pinned.roi, field, old_val)
                    raise ValueError(f"Invalid geometry for {target_pinned.roi.type}")
                geometry_changed = float(val) != float(old_val)
                if geometry_changed:
                    target_pinned.curve = None
            self.status.showMessage(f"Updated {field} for ROI {roi_id}")
        except (TypeError, ValueError):
            self.status.showMessage(f"Invalid input for {field}. Reverting...")
        finally:
            self.refresh_roi_table()
            self.refresh_map()
            self.refresh_curves(update_volume=geometry_changed)
            self._update_inspector()

    def _on_roi_manager_edit_requested(self, roi_id: int, values: dict) -> None:
        """Применяет изменения одной ROI как единую проверяемую операцию."""
        target = next((item for item in self.current_rois() if int(item.id) == int(roi_id)), None)
        if target is None:
            return
        try:
            candidate = target.roi.copy()
            candidate.label = str(values.get("label", candidate.label)).strip()
            for field in ("cx", "cy", "width", "height", "angle"):
                if field in values:
                    setattr(candidate, field, float(values[field]))
            if "color" in values:
                if not QColor(str(values["color"])).isValid(): raise ValueError("Некорректный цвет ROI")
                candidate.color = QColor(str(values["color"])).name()
            candidate = candidate.normalized()
            if not candidate.is_valid:
                raise ValueError(f"Некорректная геометрия ROI типа {candidate.type}")
        except (TypeError, ValueError) as exc:
            self.status.showMessage(f"ROI {roi_id} was not changed: {exc}")
            return
        geometry_fields = ("cx", "cy", "width", "height", "angle")
        geometry_changed = (
            str(candidate.type) != str(target.roi.type)
            or any(
                float(getattr(candidate, name)) != float(getattr(target.roi, name))
                for name in geometry_fields
            )
        )
        color_changed = str(candidate.color) != str(target.roi.color)
        target.roi = candidate
        if geometry_changed:
            target.curve = None
        if color_changed:
            self.volume_widget.update_roi_color(int(roi_id), candidate.color)
        self.status.showMessage(f"ROI {roi_id} updated")
        self._set_selected_roi_id(int(roi_id), update_volume=geometry_changed)

    def _on_roi_manager_color_changed(self, roi_id: int, color: str) -> None:
        target = next((item for item in self.current_rois() if int(item.id) == int(roi_id)), None)
        parsed_color = QColor(str(color))
        if target is None or not parsed_color.isValid():
            return
        normalized_color = parsed_color.name(QColor.NameFormat.HexRgb)
        target.roi.color = normalized_color
        # The ROI outline is display-only: patch its existing Scatter3d trace.
        self.volume_widget.update_roi_color(int(roi_id), normalized_color)
        self.refresh_roi_table()
        self.refresh_map()
        self.refresh_curves(update_volume=False)
        self._update_inspector()

    def _on_roi_manager_delete_requested(self, roi_id: int) -> None:
        if not any(int(item.id) == int(roi_id) for item in self.current_rois()):
            return
        self._set_selected_roi_id(int(roi_id))
        self.delete_selected_roi()

    def _on_roi_manager_expanded(self, expanded: bool) -> None:
        """Выделяет место под компактную таблицу по запросу."""
        total = max(self.plot_splitter.width(), sum(self.plot_splitter.sizes()))
        target = 282 if expanded else 22
        rest = max(2, total - target)
        map_width = max(1, int(rest * 0.53))
        self.plot_splitter.setSizes([target, map_width, max(1, rest - map_width)])

    def delete_selected_roi(self, *args) -> None:
        if self.selected_scan is None or self.selected_roi_id is None:
            return
        rois = self.current_rois()
        before = len(rois)
        rois[:] = [r for r in rois if r.id != self.selected_roi_id]
        after = len(rois)
        if before == after:
            return
        self._set_selected_roi_id(None, update_volume=True)
        self.status.showMessage("Deleted selected ROI.")

    def clear_current_scan_rois(self, *args) -> None:
        if self.selected_scan is None:
            return
        if not self.current_rois():
            return
        reply = QMessageBox.question(
            self,
            "Clear ROIs",
            f"Delete all ROIs for scan {self.selected_scan}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.session_scans[self.selected_scan] = []
        self._set_selected_roi_id(None, update_volume=True)
        self.status.showMessage("Cleared all ROIs for current scan.")

    def _toggle_roi_selector_mode(self, _is_active: bool) -> None:
        """При смене режима убирает пиксельное превью; селектором управляет MapPlotWidget."""        
        if self.selected_pixel is not None:
            self.selected_pixel = None
            self.refresh_curves(update_volume=False)
            self._update_inspector()

    # -------- Контекстное Меню (ПКМ по ROI) ----------
    def _show_roi_context_menu(self, roi_id: int | None) -> None:
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { font-size: 10px; padding: 3px; } QMenu::item { padding: 4px 18px; } QMenu::separator { height: 1px; background: palette(mid); margin: 3px 5px; }")
        apply_popup_theme(menu, resolve_theme(self))
        if roi_id is not None and any(int(item.id) == int(roi_id) for item in self.current_rois()):
            self._set_selected_roi_id(int(roi_id))
            export_action = menu.addAction("Save rocking curve…")
            copy_menu = menu.addMenu("Copy")
            apply_popup_theme(copy_menu, resolve_theme(self))
            metadata_action = copy_menu.addAction("Metadata")
            data_action = copy_menu.addAction("Curve data (theta, I)")
            menu.addSeparator()
            delete_action = menu.addAction("Delete ROI")
            menu.addSeparator()
        else:
            export_action = metadata_action = data_action = delete_action = None
        clear_action = menu.addAction("Clear all ROIs")
        action = menu.exec(QCursor.pos())
        if action == export_action and export_action is not None: self._save_roi_curve(int(roi_id))
        elif action == metadata_action and metadata_action is not None: self._copy_roi_content(int(roi_id), "metadata")
        elif action == data_action and data_action is not None: self._copy_roi_content(int(roi_id), "data")
        elif action == delete_action and delete_action is not None: self.delete_selected_roi()
        elif action == clear_action: self.clear_current_scan_rois()

    def _roi_export_payload(self, roi_id: int):
        if self.repo is None or self.selected_scan is None:
            self.status.showMessage("Сначала откройте HDF5 и выберите скан.")
            return None
        pinned = next((item for item in self.current_rois() if int(item.id) == int(roi_id)), None)
        if pinned is None:
            self.status.showMessage(f"ROI {roi_id} больше не существует.")
            return None
        try:
            th, curve, count = self.repo.curve_from_roi(self.selected_scan, pinned.roi, reduction_mode=self.reduction_mode)
            th = np.asarray(th, dtype=float).reshape(-1)
            curve = np.asarray(curve, dtype=float).reshape(-1)
            if th.size != curve.size: raise ValueError(f"Число theta ({th.size}) не совпадает с числом I ({curve.size}).")
            if not curve.size: raise ValueError("Кривая пуста.")
        except Exception as exc:
            QMessageBox.warning(self, "ROI curve", f"Не удалось подготовить кривую ROI {roi_id}:\n{exc}")
            return None
        pinned.curve = curve.copy(); pinned.count = int(count)
        return pinned, th, curve, int(count)

    def _roi_metadata_text(self, pinned, th: np.ndarray, curve: np.ndarray, count: int) -> str:
        lines = ["Selected pinned ROI", f"File: {self.repo.h5_path}", f"Scan: {self.selected_scan}", f"Map: {self.selected_map or '(none)'}", f"Spatial view: {self._pool_label(self.repo.pool)}", f"Reduction: {self.reduction_mode}", f"ROI id: {pinned.id}"]
        lines.extend(self._roi_lines(pinned.roi, th, curve, count))
        return "\n".join(lines)

    def _copy_roi_content(self, roi_id: int, kind: str) -> None:
        payload = self._roi_export_payload(roi_id)
        if payload is None: return
        pinned, th, curve, count = payload
        if kind == "metadata":
            text = self._roi_metadata_text(pinned, th, curve, count)
        else:
            text = "theta\tI\n" + "\n".join(f"{theta:.12g}\t{intensity:.12g}" for theta, intensity in zip(th, curve))
        QApplication.clipboard().setText(text)
        self.status.showMessage(f"Скопировано в буфер: {'metadata' if kind == 'metadata' else 'theta, I'} для {pinned.roi.label or f'ROI {pinned.id}'}.")

    def _save_roi_curve(self, roi_id: int) -> None:
        payload = self._roi_export_payload(roi_id)
        if payload is None: return
        pinned, th, curve, count = payload
        label = re.sub(r"[^\w.-]+", "_", pinned.roi.label or f"ROI_{pinned.id}", flags=re.UNICODE).strip("._") or f"ROI_{pinned.id}"
        scan_tag = re.sub(r"[^\w.-]+", "_", str(self.selected_scan), flags=re.UNICODE).strip("._") or "scan"
        default_path = self.repo.h5_path.with_name(f"{self.repo.h5_path.stem}_{scan_tag}_{label}.txt")
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI rocking curve", str(default_path), "Tab-delimited text (*.txt *.dat);;All files (*)")
        if not path: return
        finite = np.isfinite(curve)
        if finite.any():
            values = curve[finite]; peak_idx = int(np.flatnonzero(finite)[np.argmax(values)])
            min_v, mean_v, max_v = float(values.min()), float(values.mean()), float(values.max())
            peak_theta = float(th[peak_idx])
        else:
            min_v = mean_v = max_v = peak_theta = float("nan")
        roi = pinned.roi.normalized()
        header = ["ROC-MCS rocking curve", f"source_file: {self.repo.h5_path}", f"scan_id: {self.selected_scan}", f"map: {self.selected_map or ''}", f"roi_id: {pinned.id}", f"roi_label: {roi.label or ''}", f"center_x: {roi.cx:.9g}", f"center_y: {roi.cy:.9g}", f"width: {roi.width:.9g}", f"height: {roi.height:.9g}", f"angle_deg: {roi.angle:.9g}", f"pixels: {count}", f"curve_points: {curve.size}", f"reduction: {self.reduction_mode}", f"spatial_view: {self._pool_label(self.repo.pool)}", f"intensity_min: {min_v:.12g}", f"intensity_mean: {mean_v:.12g}", f"intensity_max: {max_v:.12g}", f"peak_theta: {peak_theta:.12g}", "columns: theta\tI"]
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as stream:
                stream.write("# " + "\n# ".join(header) + "\n")
                np.savetxt(stream, np.column_stack((th, curve)), delimiter="\t", fmt="%.12g")
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, "Save ROI curve", f"Не удалось сохранить файл:\n{exc}")
            return
        self.status.showMessage(f"Кривая {roi.label or f'ROI {pinned.id}'} сохранена: {path}")

    # ---------- Rendering ----------
    def refresh_all(self) -> None:
        self.refresh_map()
        self.refresh_curves()
        self.refresh_roi_table()
        self._update_inspector()
        self._request_volume_refresh()

    def refresh_map(self, *args) -> None:
        """Отрисовывает карту выбранной версии данных либо исходного HDF5."""
        self._theta_range_preview_timer.stop()
        self._theta_range_finalize_pending = False
        if self._analysis_view is not None:
            state = self._analysis_view
            if self._theta_slice_mode:
                idx = max(0, min(int(self.map_widget.theta_slider.value()), state.theta.size - 1))
                img = np.asarray(state.data_yxt[:, :, idx], dtype=float).copy()
                if state.valid_mask_yxt is not None:
                    img[~np.asarray(state.valid_mask_yxt[:, :, idx], dtype=bool)] = np.nan
                title = f"{state.label} · θ={state.theta[idx]:.6g}"
                self.map_widget.tools.results_btn.setEnabled(False)
            else:
                if (
                    self._branch_parameter_layer is not None
                    and self._analysis_map_key == self._branch_parameter_layer.key
                ):
                    layer = self._branch_parameter_layer
                    img = layer.data_yx
                    title = f"{state.label} · {layer.label}"
                elif hasattr(state, "map_layer"):
                    layer = state.map_layer(self._analysis_map_key)
                    img = layer.data_yx
                    title = f"{state.label} · {layer.label}"
                else:
                    layer = None
                    img = state.map_data
                    title = f"{state.label} · {state.map_title}"
                self.map_widget.tools.results_btn.setEnabled(bool(self._analysis_menu_items(state, "map_layers")))
            if not self._theta_slice_mode and self._analysis_map_key == "peaks.count":
                # Match the display intensity to the signal actually consumed
                # by peaks.detect: measured counts or upstream interpolation.
                peak_input = state.data_yxt
                peak_valid = state.valid_mask_yxt
                if bool(state.metadata.get("peaks_used_interpolated_data", False)):
                    interpolated = next(
                        (source for source in state.curve_sources if source.key == "spikes.interpolated"),
                        None,
                    )
                    if interpolated is not None:
                        peak_input = interpolated.data_yxt
                        peak_valid = interpolated.valid_mask_yxt
                intensity, valid_yx = integrated_intensity_map(
                    peak_input,
                    state.theta,
                    valid_mask_yxt=peak_valid,
                    analysis_valid_mask_yx=getattr(state, "analysis_valid_mask_yx", None),
                )
                counts = np.asarray(img, dtype=float)
                if counts.shape != intensity.shape:
                    raise ValueError(
                        f"peak_count_map имеет форму {counts.shape}, ожидалась {intensity.shape}"
                    )
                counts = np.where(valid_yx, counts, np.nan)
                self.map_widget.draw_peak_count_map(
                    counts,
                    intensity,
                    title,
                    x_coords=state.x_coords,
                    y_coords=state.y_coords,
                    valid_mask_yx=valid_yx,
                )
                self._draw_branch_footprints_overlay()
                self._draw_pinned_rois(); self.map_widget.canvas.draw_idle()
                return

            branch_parameter_active = bool(
                self._branch_parameter_layer is not None
                and self._analysis_map_key == self._branch_parameter_layer.key
            )
            if not branch_parameter_active:
                try:
                    img, title = self._apply_theta_projection_range(img, title)
                except (ValueError, TypeError, IndexError) as exc:
                    self.status.showMessage(
                        f"Предпросмотр диапазона θ не применён: {exc}", 8000
                    )

            quantitative = self._is_quantitative_analysis_map(self._analysis_map_key)
            p_lo = float(self.map_widget.low_pct.value())
            p_hi = float(self.map_widget.high_pct.value())
            display_cmap = self.map_widget.current_cmap
            try:
                finite = np.asarray(img, dtype=float)
                finite = finite[np.isfinite(finite)]
                is_hkl_residual = str(self._analysis_map_key or "").startswith("hkl.delta_theta.")
                if is_hkl_residual:
                    # Δθ residuals are signed deviations around a physical zero.
                    # Keep zero exactly at the neutral midpoint and use one
                    # symmetric range for both signs; the high percentile is
                    # the only contrast control applied to the absolute values.
                    absolute = np.abs(finite)
                    limit = float(np.percentile(absolute, np.clip(p_hi, 0.0, 100.0))) if absolute.size else 1.0
                    if not np.isfinite(limit) or limit <= 0.0:
                        limit = max(float(np.max(absolute)) if absolute.size else 0.0, 1e-12)
                    vmin, vmax = -limit, limit
                else:
                    vmin, vmax = np.percentile(finite, [p_lo, p_hi]) if finite.size else (None, None)
                if quantitative and self._analysis_map_key in {
                    "spikes.count", "branches.count", "fit.n_detected", "fit.n_fitted", "fit.n_fitted_map",
                }:
                    # Count maps have a real zero baseline. Do not let an
                    # automatic percentile shift black to the value 1 or higher.
                    vmin = 0.0
                    if vmax is None or not np.isfinite(vmax) or vmax <= 0.0:
                        vmax = 1.0
                if vmin is not None and vmax is not None and vmin >= vmax:
                    vmax = vmin + max(abs(float(vmin)) * 1e-9, 1e-12)
            except Exception:
                vmin = vmax = None
            self.map_widget.draw_map(
                img, title, display_cmap, vmin, vmax,
                x_coords=state.x_coords, y_coords=state.y_coords,
                colorbar_mode="quantitative" if quantitative else "continuous",
            )
            if self.map_widget.cbar is not None:
                if branch_parameter_active and self._branch_parameter_layer is not None:
                    colorbar_unit = str(self._branch_parameter_layer.unit or "")
                else:
                    colorbar_unit = (
                        "Δθ, ″"
                        if str(self._analysis_map_key or "").startswith("hkl.delta_theta.")
                        else ""
                    )
                self.map_widget.cbar.set_label(colorbar_unit, labelpad=3)
            self._draw_branch_footprints_overlay()
            self._draw_pinned_rois(); self.map_widget.canvas.draw_idle()
            return
        if self.repo is None or self.selected_scan is None or (self.selected_map is None and not self._theta_slice_mode):
            self.map_widget.ax.clear()
            self.map_widget.ax.text(0.5, 0.5, "Open an HDF5 file", ha="center", va="center", transform=self.map_widget.ax.transAxes)
            self.map_widget.canvas.draw_idle(); return
        x_coords, y_coords = self.repo.centers_xy(self.selected_scan)
        if self._theta_slice_mode:
            try:
                scan = self._get_theta_scan()
                if scan is None: return
                data, theta, axis = scan
                idx = max(0, min(int(self.map_widget.theta_slider.value()), len(theta)-1))
                img = data[idx, :, :] if axis == 0 else data[:, :, idx]
                if img.shape != (len(y_coords), len(x_coords)):
                    raise ValueError(f"Срез θ имеет форму {img.shape}, а оси детектора ожидают {(len(y_coords), len(x_coords))}.")
                title = f"{self.selected_scan}  |  Срез по θ = {theta[idx]:.6g}°"
            except Exception as exc:
                self.status.showMessage(f"Не удалось построить срез по θ: {exc}", 8000); return
        else:
            img = self.repo.get_map(self.selected_scan, self.selected_map)
            title = f"{self.selected_scan}  |  {self.selected_map}"
            try:
                img, title = self._apply_theta_projection_range(img, title)
            except (ValueError, TypeError, IndexError) as exc:
                self.status.showMessage(
                    f"Предпросмотр диапазона θ не применён: {exc}", 8000
                )
        p_lo = float(self.map_widget.low_pct.value()); p_hi = float(self.map_widget.high_pct.value())
        if p_hi <= p_lo:p_hi=min(100.0,p_lo+1.0)
        try:
            level_key=(str(getattr(self.repo,"h5_path",self.current_file or "")),str(self.selected_scan),repr(getattr(self.repo,"pool",None)),int(self._theta_index),round(p_lo,2),round(p_hi,2)) if self._theta_slice_mode else None
            cached=self._theta_levels_cache.get(level_key) if level_key is not None else None
            if cached is not None:vmin,vmax=cached
            else:
                values=np.asarray(img);finite=values[np.isfinite(values)];vmin,vmax=np.percentile(finite,[p_lo,p_hi]) if finite.size else (None,None)
                if vmin is not None and vmax is not None and vmin>=vmax:vmax=vmin+max(abs(float(vmin))*1e-9,1e-12)
                if level_key is not None:
                    if len(self._theta_levels_cache)>=256:self._theta_levels_cache.pop(next(iter(self._theta_levels_cache)))
                    self._theta_levels_cache[level_key]=(vmin,vmax)
        except Exception:vmin,vmax=None,None
        self.map_widget.draw_map(img, title, self.map_widget.current_cmap, vmin, vmax, x_coords=x_coords, y_coords=y_coords)
        self._draw_pinned_rois()
        self.map_widget.canvas.draw_idle()

    def _draw_pinned_rois(self) -> None:
        """Подготавливает геометрию закреплённых ROI для MapPlotWidget."""
        roi_data_list = []
        for pinned in self.current_rois():
            roi = pinned.roi.normalized()
            if not roi.is_valid:                                  # Единая защита от дурака
                continue
            color = roi.color or self._color_for_roi(pinned.id)   # Собираем данные в словарь
            roi_data_list.append({
                'id': pinned.id,
                'type': roi.type,              # передаем тип ('point', 'line', 'rect')
                'corners': roi.corners(),      # передаем готовые координаты (очень упростит отрисовку)
                'cx': float(roi.cx),
                'cy': float(roi.cy),
                'width': roi.width,
                'height': roi.height,
                'angle': float(roi.angle),
                'color': color,
                'is_selected': pinned.id == self.selected_roi_id
            })
        self._map_patches = self.map_widget.draw_rois(roi_data_list)   # Отдаем словари в Панель на отрисовку и забираем патчи

        
    def refresh_curves(self, *, fit_curve_view: bool = False, fit_curve_y: bool = False, update_volume: bool = True) -> None:
        """Подготавливает профили из активной версии данных."""
        if self._analysis_view is not None:
            state = self._analysis_view
            curve_source = state.curve_source(self._analysis_curve_key) if hasattr(state, "curve_source") else None
            curve_data_yxt = curve_source.data_yxt if curve_source is not None else state.data_yxt
            curve_theta = curve_source.theta if curve_source is not None else state.theta
            curve_valid_mask = curve_source.valid_mask_yxt if curve_source is not None else state.valid_mask_yxt
            curve_view_label = f"{state.label} · {curve_source.label}" if curve_source is not None else state.label
            self.curve_widget.toolbar.results_btn.setEnabled(bool(getattr(state, "curve_sources", ())))
            curves_data = []
            for pinned in self.current_rois():
                roi = pinned.roi.normalized()
                if not roi.is_valid:
                    continue
                try:
                    th, curve, count = self._analysis_curve(state, roi)
                except Exception:
                    continue
                try:
                    theta_values = np.asarray(th, dtype=float)
                    curve_values = np.asarray(curve, dtype=float)
                except (TypeError, ValueError, OverflowError):
                    continue
                if (
                    theta_values.ndim != 1 or curve_values.ndim != 1
                    or theta_values.size == 0 or theta_values.shape != curve_values.shape
                ):
                    continue
                curves_data.append({
                    "id": pinned.id, "th": theta_values, "curve": curve_values,
                    "color": roi.color or self._color_for_roi(pinned.id),
                    "is_selected": pinned.id == self.selected_roi_id,
                    "label": roi.label or f"ROI {pinned.id}",
                })
            preview_data = None
            if self.preview_roi is not None and self.preview_roi.is_valid:
                try:
                    pth, pcurve, _ = self._analysis_curve(state, self.preview_roi)
                    pth = np.asarray(pth, dtype=float)
                    pcurve = np.asarray(pcurve, dtype=float)
                    self.preview_th, self.preview_curve = pth, pcurve
                    if (
                        pth.ndim == 1 and pcurve.ndim == 1
                        and pth.size > 0 and pth.shape == pcurve.shape
                    ):
                        preview_data = {"th": pth, "curve": pcurve}
                except Exception:
                    preview_data = None
            pixel_data = None
            if self.selected_pixel is not None:
                ix, iy = self.selected_pixel
                if 0 <= iy < curve_data_yxt.shape[0] and 0 <= ix < curve_data_yxt.shape[1]:
                    curve = np.asarray(curve_data_yxt[iy, ix, :], dtype=float).copy()
                    if curve_valid_mask is not None:
                        curve[~curve_valid_mask[iy, ix, :]] = np.nan
                    pixel_data = {"th": curve_theta, "curve": curve,
                                  "label": f"Pixel ({ix}, {iy}) · {curve_source.label if curve_source is not None else 'workflow data'}"}
                else:
                    self.selected_pixel = None
            theta_guide = self._current_theta_value() if self._show_theta_guide and self._theta_slice_mode else None
            self._curve_lines = self.curve_widget.draw_curves(
                curves_data, preview_data, pixel_data=pixel_data,
                view_label=curve_view_label, fit_view=fit_curve_view, fit_y=fit_curve_y,
                theta_guide=theta_guide,
            )
            if update_volume:
                self.refresh_volume()
            return
        th = self.repo.load_scan(self.selected_scan)["th"] if (self.repo is not None and self.selected_scan is not None) else None
        curves_data = []
        for pinned in self.current_rois():
            roi = pinned.roi.normalized()
            if not roi.is_valid:  # Единая, строгая защита от дурака
                continue            
            # --- Математика HDF5 остается здесь ---
            if pinned.curve is None or pinned.curve.size == 0:
                try:
                    _, curve, count = self.repo.curve_from_roi(self.selected_scan, roi, reduction_mode=self.reduction_mode) if self.repo and self.selected_scan else (None, np.array([]), 0)
                except Exception:
                    curve = np.array([])
                    count = 0
                pinned.curve = curve
                pinned.count = int(count)
            curve = np.asarray(pinned.curve, dtype=float) if pinned.curve is not None else np.array([])
            if curve.ndim != 1 or curve.size == 0:
                continue

            # A cached ROI profile may outlive a scan/theta-axis change. Never
            # hand Matplotlib an x/y pair with different lengths.
            current_th = np.asarray(th, dtype=float) if th is not None else np.arange(curve.size)
            if current_th.ndim != 1 or current_th.shape != curve.shape:
                continue
            color = roi.color or self._color_for_roi(pinned.id)
            curves_data.append({
                'id': pinned.id,
                'th': current_th,
                'curve': curve,
                'color': color,
                'is_selected': pinned.id == self.selected_roi_id,
                'label': roi.label or f"ROI {pinned.id}"
            })

        # Данные для предпросмотра
        preview_data = None
        if self.preview_curve is not None and self.preview_th is not None:
            preview_th = np.asarray(self.preview_th, dtype=float)
            preview_curve = np.asarray(self.preview_curve, dtype=float)
            if (
                preview_th.ndim == 1 and preview_curve.ndim == 1
                and preview_th.size > 0 and preview_th.shape == preview_curve.shape
            ):
                preview_data = {
                    'th': preview_th,
                    'curve': preview_curve,
                }
        pixel_data = None
        if self.repo is not None and self.selected_scan is not None and self.selected_pixel is not None:
            scan = self.repo.load_scan(self.selected_scan)
            ix, iy = self.selected_pixel
            if 0 <= ix < scan.shape_yx[1] and 0 <= iy < scan.shape_yx[0]:
                x_coords, y_coords = self.repo.centers_xy(self.selected_scan)
                pixel_data = {
                    "th": scan.th, "curve": scan.yxt[iy, ix, :],
                    "label": f"Pixel ({ix}, {iy}) · detector ({x_coords[ix]:.1f}, {y_coords[iy]:.1f})",
                }
            else:
                self.selected_pixel = None
        view_label = self._pool_label(self.repo.pool) if self.repo is not None else None
        theta_guide=self._current_theta_value() if self._show_theta_guide and self._theta_slice_mode else None
        self._curve_lines = self.curve_widget.draw_curves(
            curves_data, preview_data, pixel_data=pixel_data, view_label=view_label, fit_view=fit_curve_view,
            fit_y=fit_curve_y, theta_guide=theta_guide
        )
        if update_volume: self.refresh_volume()


    def _request_volume_refresh(self) -> None:
        """Отложить обновление 3D до показа окна и объединить частые события."""
        self._volume_refresh_pending = True
        if self.plot_workspace.mdi_area.is_panel_visible("volume"):
            self._volume_refresh_timer.start(0)

    def _on_plot_panel_visibility_changed(self, key: str, visible: bool) -> None:
        if key == "volume" and visible and (self._volume_refresh_pending or not self.volume_widget.has_dataset_request):
            self._volume_refresh_timer.start(0)

    def _on_plot_panel_restored(self, key: str) -> None:
        if key == "volume" and (self._volume_refresh_pending or not self.volume_widget.has_dataset_request):
            self._volume_refresh_timer.start(0)

    def refresh_volume(self) -> None:
        """Передаёт VTK выбранный снимок только при видимом окне."""
        if not self.plot_workspace.mdi_area.is_panel_visible("volume"):
            self._volume_refresh_pending = True
            return
        self._volume_refresh_pending = False
        if self._analysis_view is not None:
            state = self._analysis_view
            try:
                self.volume_widget.set_dataset(
                    state.data_yxt, state.theta, 2, state.x_coords, state.y_coords,
                    self.current_rois(), scan_label=self.selected_scan or "",
                    view_label=state.label, valid_mask=state.valid_mask_yxt,
                    source_path=self.current_file, operation_id=state.operation_id,
                    metadata=state.metadata,
                    peak_catalog=state.peak_points,
                    peak_position_kind=state.peak_position_kind,
                    branch_edges=getattr(state, "branch_edges", None),
                    branch_correlation_preview=getattr(
                        state, "branch_correlation_preview", None
                    ),
                )
            except Exception as exc:
                self.volume_widget.clear_dataset(f"3D: {type(exc).__name__}: {exc}")
            return
        if self.repo is None or self.selected_scan is None:
            self.volume_widget.clear_dataset()
            return
        try:
            scan = self.repo.load_scan(self.selected_scan)
            x_coords, y_coords = self.repo.centers_xy(self.selected_scan)
            self.volume_widget.set_dataset(
                scan.data, scan.th, scan.axis, x_coords, y_coords, self.current_rois(),
                scan_label=self.selected_scan, view_label=self._pool_label(self.repo.pool),
                source_path=self.current_file, operation_id="source.raw", metadata={},
            )
        except Exception as exc:
            self.volume_widget.clear_dataset(f"3D: {type(exc).__name__}: {exc}")

    def refresh_roi_table(self) -> None:
        """
        Перестраивает таблицу ROI на основе внутренних данных.
        Метод заполнения таблицы раскладывает данные строго по ячейкам
        """
        self.roi_table.blockSignals(True)
        self.roi_table.setRowCount(0) # Очищаем таблицу
        for pinned in self.current_rois():
            roi = pinned.roi.normalized()
            row = self.roi_table.rowCount()
            self.roi_table.insertRow(row)
            
            # Колонка 0: Label
            label_text = roi.label or f"ROI {pinned.id}"
            label_item = QTableWidgetItem(label_text)
            label_item.setData(Qt.ItemDataRole.UserRole, int(pinned.id))    # Прячем ID сюда
            self.roi_table.setItem(row, 0, label_item)
            
            # Колонки 1-5: X, Y, W, H, Angle (выравниваем по центру)
            values = [
                f"{roi.cx:.1f}", f"{roi.cy:.1f}", 
                f"{roi.width:.1f}", f"{roi.height:.1f}", f"{roi.angle:.1f}"
            ]
            for col, val in enumerate(values, start=1):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.roi_table.setItem(row, col, item)    
            if pinned.id == self.selected_roi_id:          # Если это выделенный ROI, выделяем строку
                self.roi_table.selectRow(row)  
        self.roi_table.blockSignals(False)
        self.roi_manager.sync_compact_view(self.current_rois(), self.selected_roi_id, self._color_for_roi)

    def _update_inspector(self) -> None:
        lines: List[str] = []
        if self.repo is None or self.selected_scan is None:
            self.stats_text.setPlainText("Open an HDF5 file to begin.")
            return

        lines.append(f"File: {self.repo.h5_path}")
        lines.append(f"Scan: {self.selected_scan}")
        if self.selected_map:
            lines.append(f"Map: {self.selected_map}")
        lines.append(f"Spatial view: {self._pool_label(self.repo.pool)}")
        lines.append(f"Reduction: {self.reduction_mode}")
        if self._analysis_view is not None:
            lines.append("")
            lines.append("Активная версия обработки")
            lines.append(f"Этап: {self._analysis_view.label}")
            lines.append(f"Операция: {self._analysis_view.operation_id}")
            lines.append(f"Grid: {self._analysis_view.metadata.get('grid_id', '—')}")
        if self.selected_pixel is not None:
            ix, iy = self.selected_pixel
            x_coords, y_coords = self.repo.centers_xy(self.selected_scan)
            lines.append(f"Selected pixel in view: ({ix}, {iy}); detector center: ({x_coords[ix]:.1f}, {y_coords[iy]:.1f})")


        if self.preview_roi is not None:
            lines.append("")
            lines.append("Preview ROI")
            lines.extend(self._roi_lines(self.preview_roi, self.preview_th, self.preview_curve, self.preview_count))

        pinned = self._selected_pinned_roi()
        if pinned is not None:
            lines.append("")
            lines.append("Selected pinned ROI")
            curve = pinned.curve
            th = self.repo.load_scan(self.selected_scan)["th"] if self.repo is not None else None
            lines.extend(self._roi_lines(pinned.roi, th, curve, pinned.count))

        if not self.current_rois():
            lines.append("")
            lines.append("No pinned ROIs for this scan yet.")
        self.stats_text.setPlainText("\n".join(lines))

    def _roi_lines(
        self,
        roi: ROI,
        th: Optional[np.ndarray],
        curve: Optional[np.ndarray],
        count: int,
    ) -> List[str]:
        roi = roi.normalized()
        out = [
            f"label: {roi.label or '(unnamed)'}",
            f"center: ({roi.cx:.3f}, {roi.cy:.3f})",
            f"size:   {roi.width:.3f} × {roi.height:.3f}",
            f"angle:  {roi.angle:.3f}°",
            f"pixels: {count}",
        ]
        if curve is not None and curve.size:
            finite = np.isfinite(curve)
            if finite.any():
                valid_idx = np.flatnonzero(finite); idx = int(valid_idx[np.argmax(curve[finite])])
                theta_peak = float(th[idx]) if th is not None and idx < len(th) else float(idx)
                stats = f"{np.min(curve[finite]):.6g} / {np.mean(curve[finite]):.6g} / {np.max(curve[finite]):.6g}"
                peak_text = f"{theta_peak:.6g}"
            else:
                stats, peak_text = "n/a / n/a / n/a", "n/a"
            out.extend([f"curve points: {curve.size}", f"min / mean / max: {stats}", f"peak θ: {peak_text}"])
        return out

    def _color_for_roi(self, roi_id: int) -> str:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
        return cycle[(roi_id - 1) % len(cycle)]

    def _on_pick_event(self, event) -> None:
        artist = getattr(event, "artist", None)
        roi_id = getattr(artist, "_roi_id", None)
        if roi_id is None:
            return
        self._set_selected_roi_id(int(roi_id))

    
    def _on_table_selection_changed(self):
        """Обработчик выделения кликом поля в ROI-таблице."""
        selected_items = self.roi_table.selectedItems()
        if not selected_items:
            self._set_selected_roi_id(None)
            return
            
        # ID всегда лежит в 0-й колонке текущей строки
        row = selected_items[0].row()
        id_item = self.roi_table.item(row, 0)
        roi_id = id_item.data(Qt.ItemDataRole.UserRole)
        self._set_selected_roi_id(roi_id)


    def closeEvent(self, event) -> None:
        """Stop pending UI refreshes, finalize VTK once, and release HDF5."""
        # Prevent queued GUI slots from requesting map/volume updates while
        # the embedded native QVTK surface is being finalized.
        for timer_name in ("_theta_render_timer", "_theta_range_preview_timer", "_preview_timer", "_volume_refresh_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                try:
                    timer.stop()
                except RuntimeError:
                    pass
        try: self.plot_workspace.mdi_area.save_geometry()
        except Exception: pass
        # Only the terminal Explorer close finalizes the reusable 3D scene.
        # Closing the 3D MDI child itself only hides that panel.
        try:
            self.volume_widget.release_vtk_scene(final=True)
        except Exception:
            _logger.exception("Ошибка финального закрытия встроенной VTK-сцены")
        closer = getattr(self.repo, "close", None)
        if callable(closer):
            try: closer()
            except Exception: pass
        self.repo = None
        super().closeEvent(event)
