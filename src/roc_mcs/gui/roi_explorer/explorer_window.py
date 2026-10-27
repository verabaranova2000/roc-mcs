import sys

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

Qt = QtCore.Qt
QTimer = QtCore.QTimer
QAction = QtGui.QAction
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
from roc_mcs.gui.components.icon_provider import resource_svg_icon
from roc_mcs.gui.app_settings import resolve_initial_database, remember_hdf5_database, settings_path
from roc_mcs.gui.roi_explorer.theme import apply_theme as apply_roi_theme, apply_popup_theme, resolve_theme, theme_colors
from roc_mcs.gui.roi_explorer.plot_workspace_widget import PlotWorkspaceWidget
from roc_mcs.roi.roi_types import ROI, SavedROI


class RockingCurveExplorerWindow(QMainWindow):
    def __init__(self, h5_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.setWindowTitle("Rocking Curve Explorer")
        self.resize(1100, 700)

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

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview_curve)

        self._build_ui()
        self._build_actions()
        self._connect_signals()
        self._apply_theme()
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
        self.left_sidebar = QWidget(self.splitter); self.left_sidebar.setObjectName("leftSidebarSurface"); self.left_sidebar.setMinimumWidth(262)
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
        self._left_sidebar_collapsed = False; self._left_sidebar_width = 262
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

        # Inspector использует общий раскрываемый компонент из gui/components.
        self.inspector_section = CollapsibleSection("Inspector", expanded=True)
        self.stats_text = QPlainTextEdit(self.inspector_section.body); self.stats_text.setReadOnly(True); self.stats_text.setMaximumBlockCount(1000)
        self.inspector_section.content_layout.addWidget(self.stats_text); left_layout.addWidget(self.inspector_section, 1)

        left_layout.addStretch(0)

       
        # Правая рабочая область: ROI Manager, карта и кривая.
        self.plot_workspace = PlotWorkspaceWidget(self.splitter)
        self.plot_splitter = self.plot_workspace                  # Совместимость обработчиков размеров
        self.roi_manager = self.plot_workspace.roi_manager
        self.roi_table = self.plot_workspace.roi_table
        self.map_widget = self.plot_workspace.map_widget
        self.curve_widget = self.plot_workspace.curve_widget

        
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
        self.left_scroll.setMinimumWidth(240)

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
            self.map_widget.apply_theme(active_theme); self.curve_widget.apply_theme(active_theme); self.roi_manager.apply_theme(active_theme)
            for popup in self.findChildren(QWidget):
                if popup.objectName() in ("roiEditorPopup", "compactColorPicker"):
                    update_popup_theme = getattr(popup, "apply_theme", None)
                    if callable(update_popup_theme): update_popup_theme(active_theme)
            self.open_h5_btn.setIcon(resource_svg_icon("folder_open.svg", widget=self.open_h5_btn, size=18))
            self.refresh_scans_btn.setIcon(resource_svg_icon("refresh.svg", widget=self.refresh_scans_btn, size=18))
        finally:
            self._applying_theme = False

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
            self._left_sidebar_collapsed = False; self.left_sidebar.setMaximumWidth(16777215); self.left_sidebar.setMinimumWidth(262); self.left_scroll.show()
            self.left_sidebar_toggle.setIcon(_pane_toggle_icon(True, self.palette().color(QPalette.ColorRole.WindowText))); self.left_sidebar_toggle.setToolTip("Collapse sidebar")
            total = max(sum(sizes), self.width()); left_width = max(262, self._left_sidebar_width); self.splitter.setSizes([left_width, max(1, total - left_width)])

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

    def _connect_signals(self) -> None:
        """ 'Провода', которые соединяют кнопки с функциями """
        self.map_widget.scan_combo.currentIndexChanged.connect(self._on_scan_changed)
        self.map_widget.map_combo.currentIndexChanged.connect(self._on_map_changed)
        self.map_widget.reduction_combo.currentIndexChanged.connect(self._on_reduction_changed)
        self.map_widget.pool_changed.connect(self._on_pool_changed)
        self.map_widget.pixel_clicked.connect(self._on_pixel_clicked)
        
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
        self.roi_manager.expanded_changed.connect(self._on_roi_manager_expanded)

        # --- Сигналы тулбара над картой ---
        self.map_widget.low_pct.valueChanged.connect(self.refresh_map)
        self.map_widget.high_pct.valueChanged.connect(self.refresh_map)
        self.map_widget.auto_contrast_btn.clicked.connect(self.reset_contrast_defaults)

        # Сигнал для включения/выключения рисования ROI
        self.map_widget.roi_btn.toggled.connect(self._toggle_roi_selector_mode)
        # Когда селектор на карте изменяется (тянут за ручку, рисуют линию) -> пересчитываем рокинги и обновляем графики
        self.map_widget.preview_roi_changed.connect(self._on_selector_change)
        self.map_widget.preview_roi_finished.connect(self._on_selector_finished)

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
                self.refresh_curves()
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

    def _set_selected_roi_id(self, roi_id: Optional[int]) -> None:
        self.selected_roi_id = roi_id
        self._sync_label_box()
        self.refresh_map()
        self.refresh_curves()
        self.refresh_roi_table()
        self._update_inspector()

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
        self.selected_scan = self.map_widget.scan_combo.currentText()
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
        self.status.showMessage(pool_warning or f"Selected scan: {self.selected_scan} · {self.repo.view}")


    def _populate_map_combo(self) -> None:
        if self.repo is None or self.selected_scan is None:
            return
        self.map_widget.map_combo.blockSignals(True)
        self.map_widget.map_combo.clear()

        maps = self.repo.available_maps(self.selected_scan)
        for map_name in maps:
            self.map_widget.map_combo.addItem(map_name)
            self.map_widget.map_combo.setItemData(
                self.map_widget.map_combo.count() - 1, 
                map_name, # Текст подсказки (полное имя без обрезки)
                Qt.ItemDataRole.ToolTipRole
            )
        current_items = [self.map_widget.map_combo.itemText(i) for i in range(self.map_widget.map_combo.count())]
        if self.selected_map not in current_items:
            self.selected_map = self.map_widget.map_combo.itemText(0) if self.map_widget.map_combo.count() else None
        if self.selected_map is not None:
            idx = self.map_widget.map_combo.findText(self.selected_map)
            if idx >= 0:
                self.map_widget.map_combo.setCurrentIndex(idx)
        self.map_widget.map_combo.blockSignals(False)
        self.refresh_map()         # Важно: Принудительный запуск обновления сцены после смены списка карт
        self.refresh_curves()
        
    def _on_map_changed(self, *args) -> None:
        if self.map_widget.map_combo.currentText():
            self.selected_map = self.map_widget.map_combo.currentText()
            self.refresh_map()
            self.refresh_curves()

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
        self._invalidate_curve_cache()
        self.selected_pixel = None  # Индексы пикселей принадлежат конкретной grid.
        self._preview_timer.stop()
        self._populate_map_combo()  # Карты/профили теперь относятся к новому view.
        self.map_widget.reset_view()
        if self.preview_roi is not None:
            self._preview_timer.start(0)
        self._update_inspector()
        self.status.showMessage(f"Spatial view: {self._pool_label(pool)}")

    def _on_pixel_clicked(self, x_index: int, y_index: int) -> None:
        """Обрабатывает явный запрос на пиксельный профиль, если режим доступен."""
        if self.map_widget.roi_btn.isChecked() or self.repo is None or self.selected_scan is None:
            return
        scan = self.repo.load_scan(self.selected_scan)
        ny, nx = scan.shape_yx
        if not (0 <= x_index < nx and 0 <= y_index < ny):
            return
        self.selected_pixel = (int(x_index), int(y_index))
        x_coords, y_coords = self.repo.centers_xy(self.selected_scan)
        self.status.showMessage(
            f"Pixel view=({x_index}, {y_index}); detector center=({x_coords[x_index]:.1f}, {y_coords[y_index]:.1f}); {self.repo.view}"
        )
        self.refresh_curves()
        self._update_inspector()
    

    def _on_reduction_changed(self, *args) -> None:
        self.reduction_mode = self.map_widget.reduction_combo.currentText().lower()
        for rois in self.session_scans.values():
            for saved in rois:
                saved.roi.mode = self.reduction_mode
        self._invalidate_curve_cache()
        self.refresh_curves()
        if self.preview_roi is not None:
            self._preview_timer.start(0)        
        self._update_inspector()
        
    
    def reset_contrast_defaults(self) -> None:
        """Сбрасывает контраст на дефолтные 1.0% - 99.0%"""
        # Блокируем сигналы, чтобы карта не перерисовывалась дважды
        self.map_widget.low_pct.blockSignals(True)
        self.map_widget.high_pct.blockSignals(True)
        
        self.map_widget.low_pct.setValue(1.0)
        self.map_widget.high_pct.setValue(99.0)
        
        self.map_widget.low_pct.blockSignals(False)
        self.map_widget.high_pct.blockSignals(False)
        
        # Передаем обновленные значения в слайдер
        self.map_widget.range_slider.setLow(1.0)
        self.map_widget.range_slider.setHigh(99.0)
        
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
            self.refresh_curves()
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
            th, curve, count = self.repo.curve_from_roi(
                self.selected_scan,
                self.preview_roi,
                reduction_mode=self.reduction_mode,
            )
        except Exception as exc:
            self.status.showMessage(f"Preview update failed: {exc}")
            return

        self.preview_th = th
        self.preview_curve = curve
        self.preview_count = count
        curve_values = np.asarray(curve, dtype=float)
        fit_view = self._preview_fit_pending and curve_values.size > 0 and np.isfinite(curve_values).any()
        self._preview_fit_pending = False
        self.refresh_curves(fit_curve_view=fit_view)
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
            curve=self.preview_curve.copy() if self.preview_curve is not None else None,
            count=int(self.preview_count),
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
        
        # --- ОБНОВЛЕНИЕ UI ЧЕРЕЗ ТАБЛИЦУ ---
        self._set_selected_roi_id(saved.id)            # Сначала говорим системе, что этот ROI теперь выбран
        self.refresh_roi_table()                       # Полностью перестраиваем таблицу (она сама отрисует новый ROI и выделит его строку)
        
        self._set_selected_roi_id(saved.id)            # _set_selected_roi_id выделяет элемент в UI
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

        # 2. Безопасно применяем изменения
        try:
            if field == "label":
                target_pinned.roi.label = new_val_str
            else:      
                val = float(new_val_str)                              # Если это не имя, значит это число. Пытаемся конвертировать!
                old_val = getattr(target_pinned.roi, field)           # ТРАНЗАКЦИЯ: Запоминаем текущее валидное значение
                setattr(target_pinned.roi, field, val)                # Применяем новое значение               
                if not target_pinned.roi.is_valid:                    # Спрашиваем сам объект, не нарушилась ли его геометрия?
                    setattr(target_pinned.roi, field, old_val)        # Откатываемся к старому значению и вызываем искусственную ошибку
                    raise ValueError(f"Invalid geometry for {target_pinned.roi.type}")
                target_pinned.curve = None                            # Сбрасываем кэш кривой, чтобы заставить систему пересчитать её!
            self.status.showMessage(f"Updated {field} for ROI {roi_id}")
        except ValueError as e:                                       # Юзер ввел дичь (буквы вместо цифр или отрицательную ширину)
            self.status.showMessage(f"Invalid input for {field}. Reverting...")
        finally:                                                      # В любом случае (успех или ошибка) перерисовываем всё!
            self.refresh_roi_table()                                  # Если была ошибка, таблица просто откатится к старым валидным значениям из self.current_rois()
            self.refresh_map()
            self.refresh_curves()
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
        target.roi = candidate
        target.curve = None
        self.status.showMessage(f"ROI {roi_id} updated")
        self._set_selected_roi_id(int(roi_id))

    def _on_roi_manager_color_changed(self, roi_id: int, color: str) -> None:
        target = next((item for item in self.current_rois() if int(item.id) == int(roi_id)), None)
        if target is None or not QColor(color).isValid():
            return
        target.roi.color = color
        self.refresh_roi_table()
        self.refresh_map()
        self.refresh_curves()
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
        self._set_selected_roi_id(None)
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
        self._set_selected_roi_id(None)
        self.status.showMessage("Cleared all ROIs for current scan.")

    def _toggle_roi_selector_mode(self, _is_active: bool) -> None:
        """При смене режима убирает пиксельное превью; селектором управляет MapPlotWidget."""        
        if self.selected_pixel is not None:
            self.selected_pixel = None
            self.refresh_curves()
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

    def refresh_map(self, *args) -> None:
        """Получает карту и параметры контраста, затем передаёт их MapPlotWidget."""
        if self.repo is None or self.selected_scan is None or self.selected_map is None:
            self.map_widget.ax.clear()
            self.map_widget.ax.text(0.5, 0.5, "Open an HDF5 file", ha="center", va="center", transform=self.map_widget.ax.transAxes)
            self.map_widget.canvas.draw_idle()
            return
        img = self.repo.get_map(self.selected_scan, self.selected_map)
        x_coords, y_coords = self.repo.centers_xy(self.selected_scan)
        p_lo = float(self.map_widget.low_pct.value())
        p_hi = float(self.map_widget.high_pct.value())
        if p_hi <= p_lo:
            p_hi = min(100.0, p_lo + 1.0)
        try:
            vmin = float(np.percentile(img, p_lo))
            vmax = float(np.percentile(img, p_hi))
        except Exception:
            vmin, vmax = None, None
        cmap = self.map_widget.current_cmap
        title = f"{self.selected_scan}  |  {self.selected_map}"
        self.map_widget.draw_map(img, title, cmap, vmin, vmax, x_coords=x_coords, y_coords=y_coords)  # 1. Виджет сам нарисует карту и САМ восстановит нужный селектор (линию/прямоугольник)
        self._draw_pinned_rois()                                                                      # 2. Рисуем закрепившиеся ROI
        self.map_widget.canvas.draw_idle()                       # draw_map восстанавливает нужный селектор.
        
        
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

        
    def refresh_curves(self, *, fit_curve_view: bool = False) -> None:
        """Подготавливает профили; по запросу подгоняет график под новое preview."""
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
            curve = pinned.curve
            if curve is None or curve.size == 0:
                continue
                
            # --- Формируем пакет данных для Панели ---
            current_th = th if th is not None else np.arange(curve.size)
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
        if self.preview_curve is not None and self.preview_th is not None and self.preview_curve.size > 0:
            preview_data = {
                'th': self.preview_th,
                'curve': self.preview_curve
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
        self._curve_lines = self.curve_widget.draw_curves(
            curves_data, preview_data, pixel_data=pixel_data, view_label=view_label, fit_view=fit_curve_view
        )

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
        """Освобождает HDF5-дескриптор при закрытии окна."""
        closer = getattr(self.repo, "close", None)
        if callable(closer):
            try: closer()
            except Exception: pass
        self.repo = None
        super().closeEvent(event)
