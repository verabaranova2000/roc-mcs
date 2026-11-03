"""Standalone Qt desktop viewer: HDF5 -> ROI Explorer extraction -> VTK GPU volume."""
from __future__ import annotations

import argparse
import faulthandler
import gc
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from roc_mcs.gui.app_settings import (
    load_detector_bad_pixels,
    normalize_detector_bad_pixels,
    save_detector_bad_pixels,
)
from .h5_volume_source import H5VolumeLoader, VolumePayload
from .cursor_link import detector_cursor_bounds, map_detector_cursor_to_volume
from .vtk_scene import TransferSettings, VTKVolumeScene


def _trace(stage: str, /, **fields: Any) -> None:
    """Flush concise diagnostics; positional-only name avoids field collisions.

    A field named ``stage`` is intentionally supported, e.g. to report the
    current QVTK paint lifecycle stage without colliding with the event name.
    """
    details = " ".join(f"{key}={value!r}" for key, value in fields.items())
    print(
        f"[VTK-H5 {time.strftime('%Y-%m-%d %H:%M:%S')}] {stage}"
        + (f" | {details}" if details else ""),
        flush=True,
    )


def _resource_snapshot() -> dict[str, Any]:
    """Best-effort process memory and Win32 GDI/USER handle counters."""
    result: dict[str, Any] = {}
    try:
        import psutil
        process = psutil.Process()
        memory = process.memory_info()
        result["rss_mb"] = round(memory.rss / (1024.0 ** 2), 1)
        result["vms_mb"] = round(memory.vms / (1024.0 ** 2), 1)
        try:
            # USS is the best available approximation of memory uniquely
            # owned by this process. RSS may include shared libraries and
            # driver mappings and must not be treated as a leak counter alone.
            full_memory = process.memory_full_info()
            if hasattr(full_memory, "uss"):
                result["uss_mb"] = round(full_memory.uss / (1024.0 ** 2), 1)
            if hasattr(full_memory, "pss"):
                result["pss_mb"] = round(full_memory.pss / (1024.0 ** 2), 1)
        except Exception:
            pass
        result["threads"] = int(process.num_threads())
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            user32.GetGuiResources.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            user32.GetGuiResources.restype = wintypes.DWORD
            handle = kernel32.GetCurrentProcess()
            result["gdi_objects"] = int(user32.GetGuiResources(handle, 0))
            result["user_objects"] = int(user32.GetGuiResources(handle, 1))
        except Exception as exc:
            result["win_resource_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _trace_resources(stage: str) -> None:
    snapshot = _resource_snapshot()
    if snapshot:
        _trace("PROCESS_RESOURCES", stage=stage, **snapshot)


def _qt_modules():
    try:
        # CRITICAL: register concrete VTK rendering implementations BEFORE a
        # QVTKRenderWindowInteractor instance is constructed. Its constructor
        # calls vtkRenderWindow(), which selects the concrete implementation
        # through VTK's object factory. If RenderingOpenGL2 has not been
        # imported yet, Windows can leave the widget with the abstract
        # vtkRenderWindow base object; registering OpenGL2 later does not
        # replace that already-created object.
        #
        # This is the same import ordering used by Kitware's VTK 9.6.2
        # QVTKRenderWidgetConeExample. The volume backend is also registered
        # here, before the first vtkGPUVolumeRayCastMapper is constructed.
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401

        from PyQt6 import QtCore, QtWidgets
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except Exception as exc:
        raise ImportError(
            "Не удалось загрузить PyQt6 и VTK OpenGL2/Qt backend. "
            "Проверьте установку VTK с модулями vtkRenderingOpenGL2, "
            "vtkRenderingVolumeOpenGL2, vtkInteractionStyle и PyQt6. "
            f"Исходная ошибка: {type(exc).__name__}: {exc}"
        ) from exc
    return QtCore, QtWidgets, QVTKRenderWindowInteractor


def _create_window_class(QtCore, QtWidgets, QVTKRenderWindowInteractor):
    class TracedQVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
        """QVTK widget with completed-paint accounting for safe deferred init."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.paint_count = 0
            self.paint_stage = "pre-init"
            self._preinit_refresh_requested = False
            self._finalized = False

        def closeEvent(self, event):
            # QVTKRenderWindowInteractor's own closeEvent calls Finalize().
            # Invoke it exactly once while the widget still has its native
            # parent/window resources; do not manually finalize the render
            # window from the scene or from the QMainWindow destructor.
            if not self._finalized:
                self._finalized = True
                _trace("QVTK_FINALIZE_BEGIN", render_window=self.GetRenderWindow().GetClassName())
                try:
                    super().closeEvent(event)
                finally:
                    _trace("QVTK_FINALIZE_END")
            event.accept()

        def paintEvent(self, event):
            next_count = self.paint_count + 1
            stage = self.paint_stage
            if next_count <= 8 or next_count % 25 == 0:
                _trace("QVTK_PAINT_BEGIN", stage=stage, count=next_count,
                       size=(self.width(), self.height()))
            try:
                super().paintEvent(event)
            except BaseException as exc:
                _trace("QVTK_PAINT_EXCEPTION", stage=stage, count=next_count,
                       error=f"{type(exc).__name__}: {exc}")
                raise
            self.paint_count = next_count
            if next_count <= 8 or next_count % 25 == 0:
                _trace("QVTK_PAINT_END", stage=stage, count=next_count)

    class LoadWorker(QtCore.QObject):
        completed = QtCore.pyqtSignal()

        def __init__(self, loader: H5VolumeLoader, scan_key: str, mode: str, roi: Any,
                     remove_spikes: bool, smoothing_sigma: float,
                     theta_range: tuple[float, float] | None, outcome: dict):
            super().__init__()
            self.loader = loader
            self.scan_key = scan_key
            self.mode = mode
            self.roi = roi
            self.remove_spikes = remove_spikes
            self.smoothing_sigma = smoothing_sigma
            self.theta_range = theta_range
            # Plain Python dict is owned by the GUI window and is read only after
            # QThread.finished. This avoids delivering a large payload to GUI
            # code before the worker thread has actually stopped.
            self.outcome = outcome

        @QtCore.pyqtSlot()
        def run(self):
            try:
                self.outcome["payload"] = self.loader.load(
                    self.scan_key,
                    mode=self.mode,
                    roi=self.roi,
                    remove_spikes=self.remove_spikes,
                    smoothing_sigma=self.smoothing_sigma,
                    theta_range=self.theta_range,
                )
            except Exception as exc:
                self.outcome["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self.completed.emit()

    class VTKH5ViewerWindow(QtWidgets.QMainWindow):
        def __init__(self, h5_path: str | None = None, scan_key: str | None = None):
            super().__init__()
            self.setWindowTitle("ROI Explorer — VTK HDF5 Volume Viewer")
            # The launcher opens maximized (respecting Windows' work area and
            # taskbar). Keep a modest minimum size for smaller laptop displays.
            self.resize(1480, 860)
            self.setMinimumSize(900, 540)
            self.loader: H5VolumeLoader | None = None
            self.payload: VolumePayload | None = None
            self._roi_geometry_initialized = False
            self._roi_by_scan: dict[str, tuple[float, float, float, float, float]] = {}
            self._active_roi_scan_key: str | None = None
            self._thread = None
            self._worker = None
            self._pending_load_outcome = None
            self._busy = False
            self._closing = False
            self._initial_scan_key = scan_key
            self._auto_load_enabled = True
            self._auto_load_pending = False
            self._pending_theta_slice_index: int | None = None
            self._cursor_detector_bounds: tuple[float, float, float, float] | None = None
            self._last_map_cursor_position: tuple[float, float] | None = None
            self._pending_map_cursor_position: tuple[float, float] | None = None
            self._cursor_input_source = "sliders"
            self.vtk_widget = None
            self.render_window = None
            self.interactor = None
            self.scene = None
            self._vtk_init_pending = False
            self._vtk_init_attempts = 0
            self._build_ui()
            # Coalesce rapid slider signals into bounded updates. The selected
            # theta cell is updated through existing GPU volume shader uniforms;
            # no overlay actor, geometry, volume texture, or HDF5 read is created.
            self._theta_slice_update_timer = QtCore.QTimer(self)
            self._theta_slice_update_timer.setSingleShot(True)
            self._theta_slice_update_timer.setInterval(25)
            self._theta_slice_update_timer.timeout.connect(self._apply_pending_theta_slice_index)
            # Coalesce slider or future 2D-map mouse-move events. Only the most
            # recent detector XY is transferred to existing GPU uniforms at
            # most once per ~16 ms; no data reload or volume rebuild is involved.
            self._cursor_line_update_timer = QtCore.QTimer(self)
            self._cursor_line_update_timer.setSingleShot(True)
            self._cursor_line_update_timer.setInterval(16)
            self._cursor_line_update_timer.timeout.connect(self._apply_pending_map_cursor_position)
            # IMPORTANT: construct the QVTK child BEFORE the top-level window is
            # shown. The production-scene diagnostic that successfully rendered
            # the real H5 payload creates QVTK in this order. The previous viewer
            # created it later, after the parent window was already visible, and
            # initialized it on a timer without waiting for any completed paint.
            # Do not initialize OpenGL yet. The child is part of the widget tree
            # before the top-level window is shown; the initial modal file dialog
            # runs while the parent (and therefore this child) is still unshown.
            self._create_vtk_widget_shell()

            # Initial H5 selection/opening is orchestrated by main() before
            # QMainWindow.show(). This guarantees that the QVTK child exists
            # before the parent window creates its native platform surface.
            if h5_path:
                self.open_h5(h5_path)

        def _build_ui(self):
            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)
            root.setContentsMargins(10, 10, 10, 10)
            root.setSpacing(10)

            self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
            root.addWidget(self.splitter)

            # All settings live in one vertical scroll area. The old fixed-height
            # panel forced the lower camera controls below the Windows taskbar
            # and made the group boxes visually collide on shorter displays.
            self.panel_scroll = QtWidgets.QScrollArea()
            self.panel_scroll.setWidgetResizable(True)
            self.panel_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.panel_scroll.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.panel_scroll.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self.panel_scroll.setMinimumWidth(300)
            self.panel_scroll.setMaximumWidth(430)
            panel = QtWidgets.QWidget()
            self.panel_content = panel
            self.panel_layout = QtWidgets.QVBoxLayout(panel)
            self.panel_layout.setContentsMargins(8, 8, 8, 8)
            self.panel_layout.setSpacing(9)
            self.panel_scroll.setWidget(panel)

            file_row = QtWidgets.QHBoxLayout()
            self.path_label = QtWidgets.QLabel("HDF5 не выбран")
            self.path_label.setWordWrap(True)
            self.open_button = QtWidgets.QPushButton("Открыть H5…")
            self.open_button.clicked.connect(self._choose_file)
            file_row.addWidget(self.path_label, 1)
            file_row.addWidget(self.open_button)
            self.panel_layout.addLayout(file_row)

            data_group = QtWidgets.QGroupBox("Источник данных")
            self.data_group = data_group
            data_form = QtWidgets.QFormLayout(data_group)
            data_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            data_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            self.scan_combo = QtWidgets.QComboBox()
            self.scan_combo.currentIndexChanged.connect(self._scan_changed)
            self.mode_combo = QtWidgets.QComboBox()
            self.mode_combo.addItem("Полный детектор", "full")
            self.mode_combo.addItem("Прямоугольная ROI", "roi")
            self.mode_combo.setToolTip(
                "Полный детектор временно игнорирует ROI, но не изменяет её геометрию. "
                "Возврат в режим ROI использует тот же прямоугольник."
            )
            self.spikes_check = QtWidgets.QCheckBox("Удалять спайки (штатный алгоритм)")
            self.smoothing_spin = self._double_spin(0.0, 3.0, 0.0, 2, 0.1)
            self.theta_crop_check = QtWidgets.QCheckBox("Ограничить диапазон θ при загрузке")
            self.theta_min_spin = self._double_spin(-1e12, 1e12, 0.0, 6, 0.1)
            self.theta_max_spin = self._double_spin(-1e12, 1e12, 1.0, 6, 0.1)
            self.theta_min_spin.setEnabled(False)
            self.theta_max_spin.setEnabled(False)
            self.theta_crop_check.toggled.connect(self.theta_min_spin.setEnabled)
            self.theta_crop_check.toggled.connect(self.theta_max_spin.setEnabled)
            self.bad_pixels_button = QtWidgets.QPushButton("Битые пиксели…")
            self.bad_pixels_button.setToolTip(
                "Глобальный список исключений детектора; каждый пиксель маскируется для всех θ."
            )
            self.bad_pixels_button.clicked.connect(self._edit_bad_pixels)
            data_form.addRow("Scan:", self.scan_combo)
            data_form.addRow("Область:", self.mode_combo)
            data_form.addRow("Предобработка:", self.spikes_check)
            data_form.addRow("Сглаживание σ, воксели:", self.smoothing_spin)
            data_form.addRow(self.theta_crop_check)
            data_form.addRow("θ от:", self.theta_min_spin)
            data_form.addRow("θ до:", self.theta_max_spin)
            data_form.addRow("Дефекты детектора:", self.bad_pixels_button)
            self.panel_layout.addWidget(data_group)

            roi_group = QtWidgets.QGroupBox("Геометрия ROI (detector-grid; сохраняется при переключении режимов)")
            roi_form = QtWidgets.QFormLayout(roi_group)
            roi_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            roi_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            self.roi_cx = self._double_spin(-1e9, 1e9, 0.0, 3, 0.1)
            self.roi_cy = self._double_spin(-1e9, 1e9, 0.0, 3, 0.1)
            self.roi_width = self._double_spin(0.001, 1e9, 100.0, 3, 0.1)
            self.roi_height = self._double_spin(0.001, 1e9, 100.0, 3, 0.1)
            self.roi_angle = self._double_spin(-180.0, 180.0, 0.0, 3, 0.5)
            roi_form.addRow("Центр X:", self.roi_cx)
            roi_form.addRow("Центр Y:", self.roi_cy)
            roi_form.addRow("Ширина:", self.roi_width)
            roi_form.addRow("Высота:", self.roi_height)
            roi_form.addRow("Угол, °:", self.roi_angle)
            self.panel_layout.addWidget(roi_group)
            self.roi_group = roi_group
            self.mode_combo.currentIndexChanged.connect(
                lambda *_: self.roi_group.setEnabled(self.mode_combo.currentData() == "roi")
            )
            # Full-detector GPU rendering can exceed shared-GPU memory or texture
            # limits on laboratory laptops. Start with a centered ROI; full detector
            # remains explicitly selectable after the user sees the dimensions.
            self.mode_combo.setCurrentIndex(self.mode_combo.findData("roi"))
            roi_group.setEnabled(True)

            render_group = QtWidgets.QGroupBox("Отображение объёма")
            self.render_group = render_group
            render_form = QtWidgets.QFormLayout(render_group)
            render_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            render_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            self.palette_combo = QtWidgets.QComboBox()
            from roc_mcs.gui.roi_explorer.colormap_catalog import NAPARI_STYLE_COLORMAPS
            for name in NAPARI_STYLE_COLORMAPS:
                self.palette_combo.addItem(name, name)
            self.theme_combo = QtWidgets.QComboBox()
            self.theme_combo.addItem("Тёмная", "dark")
            self.theme_combo.addItem("Светлая", "light")
            # Match ROI_Explorer_3D_VTK_LAB: threshold controls are fractions
            # of robust p99.9 positive intensity, not percentiles of the array.
            self.low_pct = self._double_spin(0.0, 99.0, 16.0, 2, 1.0)
            self.high_pct = self._double_spin(1.0, 100.0, 95.0, 2, 1.0)
            self.low_pct.valueChanged.connect(self._on_low_threshold_changed)
            self.high_pct.valueChanged.connect(self._on_high_threshold_changed)
            self._updating_threshold_widgets = False
            self.max_opacity = self._double_spin(0.0, 1.0, 0.18, 3, 0.01)
            self.gamma = self._double_spin(0.05, 8.0, 2.0, 2, 0.05)
            self.sample_factor = self._double_spin(0.25, 8.0, 1.0, 2, 0.1)
            self.opacity_unit_factor = self._double_spin(0.05, 20.0, 1.0, 2, 0.1)
            self.projection_check = QtWidgets.QCheckBox("Показывать XY-проекцию объёма")
            self.projection_check.setChecked(True)
            self.projection_opacity = self._double_spin(0.0, 1.0, 0.75, 2, 0.05)
            self.theta_scale_spin = self._double_spin(0.1, 10.0, 1.0, 2, 0.1)
            render_form.addRow("Палитра:", self.palette_combo)
            render_form.addRow("Тема:", self.theme_combo)
            render_form.addRow("Vmin, % от p99.9:", self.low_pct)
            render_form.addRow("Vmax, % от p99.9:", self.high_pct)
            render_form.addRow("Макс. opacity:", self.max_opacity)
            render_form.addRow("Gamma:", self.gamma)
            render_form.addRow("Шаг луча ×:", self.sample_factor)
            render_form.addRow("Opacity unit distance ×:", self.opacity_unit_factor)
            render_form.addRow(self.projection_check)
            render_form.addRow("Opacity XY:", self.projection_opacity)
            render_form.addRow("Масштаб θ (только отображение) ×:", self.theta_scale_spin)
            self.panel_layout.addWidget(render_group)

            # Theta highlighting is applied inside the existing GPU ray-casting
            # shader. It recolors only volume samples in the selected theta-cell
            # slab, preserving the native scalar opacity and compositing path.
            theta_slice_group = QtWidgets.QGroupBox("Выделение вклада выбранного θ в 3D-объёме")
            self.theta_slice_group = theta_slice_group
            theta_slice_form = QtWidgets.QFormLayout(theta_slice_group)
            theta_slice_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            theta_slice_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            self.theta_slice_check = QtWidgets.QCheckBox("Подсвечивать выбранный слой θ внутри облака")
            self.theta_slice_check.setToolTip(
                "Меняет цвет только тех выборок исходного GPU-объёма, которые принадлежат "
                "объёмной ячейке выбранного кадра θ и уже имеют ненулевую прозрачность "
                "по текущим Vmin/Vmax/Gamma. Не рисует плоскость, точки поверх объёма "
                "или отдельный actor; исходная прозрачность и композитинг VTK сохраняются."
            )
            self.theta_slice_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.theta_slice_slider.setRange(0, 0)
            self.theta_slice_slider.setSingleStep(1)
            self.theta_slice_slider.setPageStep(1)
            self.theta_slice_slider.setEnabled(False)
            self.theta_slice_value_label = QtWidgets.QLabel("θ = —")
            self.theta_slice_value_label.setMinimumWidth(105)
            self.theta_slice_palette_combo = QtWidgets.QComboBox()
            for name in NAPARI_STYLE_COLORMAPS:
                self.theta_slice_palette_combo.addItem(name, name)
            default_slice_palette = self.theta_slice_palette_combo.findData("turbo")
            if default_slice_palette >= 0:
                self.theta_slice_palette_combo.setCurrentIndex(default_slice_palette)
            self.theta_slice_strength = self._double_spin(0.0, 1.0, 0.90, 2, 0.05)
            theta_slice_form.addRow(self.theta_slice_check)
            theta_slice_form.addRow("Кадр по θ:", self.theta_slice_slider)
            theta_slice_form.addRow("Физический угол:", self.theta_slice_value_label)
            theta_slice_form.addRow("Палитра выделения:", self.theta_slice_palette_combo)
            theta_slice_form.addRow("Сила цветового выделения:", self.theta_slice_strength)
            theta_slice_hint = QtWidgets.QLabel(
                "Выделение выполняется внутри GPU ray casting: цвет меняется у вкладов "
                "объёмных выборок выбранной ячейки θ, но их исходные Vmin/Vmax/Gamma, "
                "прозрачность и глубинный композитинг не меняются."
            )
            theta_slice_hint.setWordWrap(True)
            theta_slice_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
            theta_slice_form.addRow(theta_slice_hint)
            self.panel_layout.addWidget(theta_slice_group)

            self.theta_slice_check.toggled.connect(self._on_theta_slice_toggled)
            self.theta_slice_slider.valueChanged.connect(self._on_theta_slice_slider_changed)
            self.theta_slice_slider.sliderReleased.connect(self._apply_pending_theta_slice_index)
            self.theta_slice_palette_combo.currentIndexChanged.connect(self._on_theta_slice_palette_changed)
            self.theta_slice_strength.valueChanged.connect(self._on_theta_slice_strength_changed)

            # The standalone viewer uses detector-coordinate sliders as a
            # temporary input source. Embedded ROI Explorer will call the same
            # public set_map_cursor_position((x, y)) slot from its 2D map.
            cursor_group = QtWidgets.QGroupBox("Линия положения курсора 2D ↔ 3D")
            self.cursor_line_group = cursor_group
            cursor_form = QtWidgets.QFormLayout(cursor_group)
            cursor_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            cursor_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            self.cursor_line_check = QtWidgets.QCheckBox("Показывать линию координаты курсора")
            self.cursor_line_check.setToolTip(
                "Непрерывная навигационная линия проходит от одного края θ-объёма до другого "
                "по выбранным координатам детектора. Она остаётся видимой даже в "
                "пустых областях и композитится внутри того же GPU ray casting: "
                "передние облака ослабляют/закрывают её по своей прозрачности, "
                "а линия ослабляет расположенные за ней слои. Отдельного actor нет."
            )
            self.cursor_x_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.cursor_y_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            for slider in (self.cursor_x_slider, self.cursor_y_slider):
                slider.setRange(0, 10000)
                slider.setSingleStep(10)
                slider.setPageStep(100)
                slider.setEnabled(False)
            self.cursor_x_value_label = QtWidgets.QLabel("X детектора = —")
            self.cursor_y_value_label = QtWidgets.QLabel("Y детектора = —")
            self.cursor_x_value_label.setMinimumWidth(120)
            self.cursor_y_value_label.setMinimumWidth(120)
            cursor_form.addRow(self.cursor_line_check)
            cursor_form.addRow("Координата X:", self.cursor_x_slider)
            cursor_form.addRow(self.cursor_x_value_label)
            cursor_form.addRow("Координата Y:", self.cursor_y_slider)
            cursor_form.addRow(self.cursor_y_value_label)
            self.cursor_line_width = self._double_spin(0.5, 4.0, 2.0, 2, 0.1)
            self.cursor_line_width.setToolTip(
                "Ширина окрашиваемого цилиндрического следа в локальных шагах сетки X/Y. "
                "Не меняет объёмные данные и глубинное перекрытие."
            )
            cursor_form.addRow("Ширина линии, вокселей:", self.cursor_line_width)
            self.cursor_line_density = self._double_spin(0.05, 3.0, 0.95, 2, 0.05)
            self.cursor_line_density.setToolTip(
                "Оптическая плотность самостоятельной линии в единицах 1/воксель. "
                "Линия существует и в пустом объёме; видимый перед ней объём "
                "ослабляет её, а маркер ослабляет облако за собой. Увеличьте "
                "значение, если линия слишком тонкая или малозаметная."
            )
            cursor_form.addRow("Оптическая плотность, 1/воксель:", self.cursor_line_density)
            cursor_hint = QtWidgets.QLabel(
                "Координаты задаются ползунками в системе детектора. "
                "Позже сигнал мыши 2D-карты подключится к тому же входу. "
                "Линия непрерывна по всему θ; вне текущей ROI она скрывается."
            )
            cursor_hint.setWordWrap(True)
            cursor_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
            cursor_form.addRow(cursor_hint)
            self.panel_layout.addWidget(cursor_group)

            self.cursor_line_check.toggled.connect(self._on_cursor_line_toggled)
            self.cursor_x_slider.valueChanged.connect(self._on_cursor_slider_changed)
            self.cursor_y_slider.valueChanged.connect(self._on_cursor_slider_changed)
            self.cursor_line_width.valueChanged.connect(self._on_cursor_line_width_changed)
            self.cursor_line_density.valueChanged.connect(self._on_cursor_line_density_changed)

            self.load_button = QtWidgets.QPushButton("Загрузить / перестроить объём")
            self.load_button.setMinimumHeight(36)
            self.load_button.clicked.connect(self.load_volume)
            self.apply_display_button = QtWidgets.QPushButton("Применить отображение")
            self.apply_display_button.clicked.connect(self.apply_display)
            self.panel_layout.addWidget(self.load_button)
            self.panel_layout.addWidget(self.apply_display_button)

            self.status_label = QtWidgets.QLabel(
                "Откройте HDF5. После выбора файла будет автоматически загружена "
                "безопасная центральная ROI; полный детектор доступен отдельно."
            )
            self.status_label.setWordWrap(True)
            self.status_label.setMinimumHeight(80)
            self.panel_layout.addWidget(self.status_label)

            camera_group = QtWidgets.QGroupBox("Камера и оси")
            self.camera_group = camera_group
            camera_layout = QtWidgets.QVBoxLayout(camera_group)
            camera_row = QtWidgets.QHBoxLayout()
            self.perspective_button = QtWidgets.QPushButton("Perspective")
            self.perspective_button.setCheckable(True)
            self.perspective_button.setChecked(True)
            self.perspective_button.clicked.connect(lambda checked: self._camera_perspective(checked))
            self.axis_x_button = QtWidgets.QPushButton("X")
            self.axis_y_button = QtWidgets.QPushButton("Y")
            self.axis_z_button = QtWidgets.QPushButton("Z")
            self.axis_x_button.clicked.connect(lambda: self._camera_axis("x"))
            self.axis_y_button.clicked.connect(lambda: self._camera_axis("y"))
            self.axis_z_button.clicked.connect(lambda: self._camera_axis("z"))
            self.axes_button = QtWidgets.QPushButton("Оси")
            self.axes_button.setCheckable(True)
            self.axes_button.setChecked(True)
            self.axes_button.clicked.connect(self._toggle_axes)
            for widget in (self.perspective_button, self.axis_x_button, self.axis_y_button,
                           self.axis_z_button):
                camera_row.addWidget(widget)
            camera_layout.addLayout(camera_row)
            camera_row2 = QtWidgets.QHBoxLayout()
            camera_row2.addWidget(self.axes_button)
            try:
                from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings
                grid_default = bool(load_plot_display_settings()["volume"].get("show_grid", True))
            except Exception:
                grid_default = True
            self.grid_check = QtWidgets.QCheckBox("Показывать сетку")
            self.grid_check.setChecked(grid_default)
            self.grid_check.setToolTip(
                "Показывает или скрывает координатные линии сетки; подписи осей остаются."
            )
            self.grid_check.toggled.connect(self._toggle_grid)
            camera_row2.addWidget(self.grid_check)
            camera_layout.addLayout(camera_row2)
            interaction_hint = QtWidgets.QLabel(
                "Мышь: левая кнопка — вращение; средняя — сдвиг; "
                "правая кнопка и колесо — масштабирование."
            )
            interaction_hint.setWordWrap(True)
            interaction_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
            camera_layout.addWidget(interaction_hint)
            self.panel_layout.addWidget(camera_group)

            self.viewer = QtWidgets.QWidget()
            self.viewer_layout = QtWidgets.QVBoxLayout(self.viewer)
            self.viewer_layout.setContentsMargins(0, 0, 0, 0)
            self.placeholder_label = QtWidgets.QLabel(
                "Выберите HDF5-файл, чтобы создать VTK/OpenGL-сцену."
            )
            self.placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.placeholder_label.setWordWrap(True)
            self.viewer_layout.addWidget(self.placeholder_label, 1)
            self.splitter.addWidget(self.panel_scroll)
            self.splitter.addWidget(self.viewer)
            self.splitter.setStretchFactor(1, 1)
            self.splitter.setSizes([365, 1080])

            self.setStyleSheet("""
                QGroupBox { font-weight: 600; margin-top: 8px; }
                QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
                QPushButton { padding: 6px 9px; }
                QComboBox, QDoubleSpinBox { min-height: 24px; }
                QScrollArea { border: 0; }
            """)

        @staticmethod
        def _double_spin(minimum, maximum, value, decimals, step):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setValue(value)
            return spin

        def _on_low_threshold_changed(self, value: float):
            if self._updating_threshold_widgets:
                return
            self._updating_threshold_widgets = True
            try:
                if float(value) >= float(self.high_pct.value()):
                    self.high_pct.setValue(min(100.0, float(value) + 0.01))
            finally:
                self._updating_threshold_widgets = False

        def _on_high_threshold_changed(self, value: float):
            if self._updating_threshold_widgets:
                return
            self._updating_threshold_widgets = True
            try:
                if float(value) <= float(self.low_pct.value()):
                    self.low_pct.setValue(max(0.0, float(value) - 0.01))
            finally:
                self._updating_threshold_widgets = False

        def _refresh_controls(self):
            busy = bool(self._busy)
            has_loader = self.loader is not None
            has_scene = self.scene is not None
            has_payload = self.payload is not None
            self.open_button.setEnabled(not busy)
            self.scan_combo.setEnabled(not busy and has_loader)
            self.mode_combo.setEnabled(not busy and has_loader)
            self.spikes_check.setEnabled(not busy and has_loader)
            self.smoothing_spin.setEnabled(not busy and has_loader)
            self.theta_crop_check.setEnabled(not busy and has_loader)
            self.theta_min_spin.setEnabled(not busy and has_loader and self.theta_crop_check.isChecked())
            self.theta_max_spin.setEnabled(not busy and has_loader and self.theta_crop_check.isChecked())
            self.bad_pixels_button.setEnabled(not busy)
            self.roi_group.setEnabled(not busy and has_loader and self.mode_combo.currentData() == "roi")
            self.load_button.setEnabled(not busy and has_loader and has_scene)
            self.apply_display_button.setEnabled(not busy and has_payload and has_scene)
            self.data_group.setEnabled(not busy)
            self.render_group.setEnabled(not busy)
            self.camera_group.setEnabled(not busy and has_scene)
            if hasattr(self, "grid_check"):
                self.grid_check.setEnabled(not busy and has_scene)
            if hasattr(self, "theta_slice_group"):
                self.theta_slice_group.setEnabled(not busy and has_scene and has_payload)
                self.theta_slice_slider.setEnabled(not busy and has_scene and has_payload)
                self.theta_slice_check.setEnabled(not busy and has_scene and has_payload)
                self.theta_slice_palette_combo.setEnabled(not busy and has_scene and has_payload)
                self.theta_slice_strength.setEnabled(not busy and has_scene and has_payload)
            if hasattr(self, "cursor_line_group"):
                cursor_ready = not busy and has_scene and has_payload and self._cursor_detector_bounds is not None
                self.cursor_line_group.setEnabled(cursor_ready)
                self.cursor_line_check.setEnabled(cursor_ready)
                self.cursor_x_slider.setEnabled(cursor_ready)
                self.cursor_y_slider.setEnabled(cursor_ready)
                self.cursor_line_width.setEnabled(cursor_ready)
                self.cursor_line_density.setEnabled(cursor_ready)

        def _theta_slice_label_text(self, index: int) -> str:
            if self.payload is None or not len(self.payload.theta):
                return "θ = —"
            index = int(np.clip(int(index), 0, len(self.payload.theta) - 1))
            return f"θ = {float(self.payload.theta[index]):.7g}°  [{index + 1}/{len(self.payload.theta)}]"

        def _on_theta_slice_toggled(self, enabled: bool):
            if self.scene is None or self.payload is None:
                return
            try:
                if self._theta_slice_update_timer.isActive():
                    self._theta_slice_update_timer.stop()
                self._pending_theta_slice_index = None
                self.scene.set_theta_slice_enabled(bool(enabled), render=False)
                self.vtk_widget.Render()
                _trace(
                    "THETA_VOLUME_SHADER_HIGHLIGHT_TOGGLED",
                    enabled=bool(enabled),
                    index=self.scene.theta_slice_index,
                    theta=(float(self.payload.theta[self.scene.theta_slice_index])
                           if self.scene.theta_slice_index is not None else None),
                )
            except Exception as exc:
                self._show_error("Не удалось изменить выделение θ", str(exc))

        def _on_theta_slice_slider_changed(self, index: int):
            if self.payload is None or self.scene is None:
                return
            index = int(np.clip(int(index), 0, len(self.payload.theta) - 1))
            self.theta_slice_value_label.setText(self._theta_slice_label_text(index))
            self._pending_theta_slice_index = index
            if not self._theta_slice_update_timer.isActive():
                self._theta_slice_update_timer.start()

        def _apply_pending_theta_slice_index(self):
            if self._pending_theta_slice_index is None:
                return
            index = int(self._pending_theta_slice_index)
            self._pending_theta_slice_index = None
            if self.payload is None or self.scene is None or self.vtk_widget is None:
                return
            try:
                applied = self.scene.set_theta_slice_index(index, render=False)
                self.vtk_widget.Render()
                self.theta_slice_value_label.setText(self._theta_slice_label_text(applied))
                _trace(
                    "THETA_VOLUME_SHADER_SLICE_CHANGED",
                    index=applied,
                    theta=float(self.payload.theta[applied]),
                    selected_theta_cell_bounds=self.scene.diagnostics().get("theta_slice_display_bounds"),
                )
            except Exception as exc:
                self._show_error("Не удалось обновить срез θ", str(exc))

        def _on_theta_slice_palette_changed(self, _index: int):
            if self.scene is None or self.payload is None:
                return
            try:
                self.scene.set_theta_slice_palette(str(self.theta_slice_palette_combo.currentData()), render=False)
                self.vtk_widget.Render()
            except Exception as exc:
                self._show_error("Не удалось изменить палитру среза θ", str(exc))

        def _on_theta_slice_strength_changed(self, value: float):
            if self.scene is None or self.payload is None:
                return
            try:
                self.scene.set_theta_slice_strength(float(value), render=False)
                self.vtk_widget.Render()
            except Exception as exc:
                self._show_error("Не удалось изменить силу выделения θ", str(exc))

        def _reset_theta_slice_controls(self):
            self._reset_cursor_line_controls()
            timer = getattr(self, "_theta_slice_update_timer", None)
            if timer is not None and timer.isActive():
                timer.stop()
            self._pending_theta_slice_index = None
            if hasattr(self, "theta_slice_check"):
                self.theta_slice_check.blockSignals(True)
                self.theta_slice_check.setChecked(False)
                self.theta_slice_check.blockSignals(False)
            if hasattr(self, "theta_slice_slider"):
                self.theta_slice_slider.blockSignals(True)
                self.theta_slice_slider.setRange(0, 0)
                self.theta_slice_slider.setValue(0)
                self.theta_slice_slider.blockSignals(False)
            if hasattr(self, "theta_slice_value_label"):
                self.theta_slice_value_label.setText("θ = —")

        @staticmethod
        def _slider_to_detector_coordinate(value: int, low: float, high: float) -> float:
            fraction = float(np.clip(int(value), 0, 10000)) / 10000.0
            return float(low + fraction * (high - low))

        @staticmethod
        def _detector_coordinate_to_slider(value: float, low: float, high: float) -> int:
            if high <= low:
                return 5000
            fraction = float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))
            return int(round(fraction * 10000.0))

        def _cursor_position_from_sliders(self) -> tuple[float, float] | None:
            bounds = self._cursor_detector_bounds
            if bounds is None:
                return None
            x0, x1, y0, y1 = bounds
            return (
                self._slider_to_detector_coordinate(self.cursor_x_slider.value(), x0, x1),
                self._slider_to_detector_coordinate(self.cursor_y_slider.value(), y0, y1),
            )

        def _set_cursor_sliders_from_detector_position(self, x: float, y: float) -> None:
            bounds = self._cursor_detector_bounds
            if bounds is None:
                return
            x0, x1, y0, y1 = bounds
            self.cursor_x_slider.blockSignals(True)
            self.cursor_y_slider.blockSignals(True)
            try:
                self.cursor_x_slider.setValue(self._detector_coordinate_to_slider(x, x0, x1))
                self.cursor_y_slider.setValue(self._detector_coordinate_to_slider(y, y0, y1))
            finally:
                self.cursor_x_slider.blockSignals(False)
                self.cursor_y_slider.blockSignals(False)
            self._update_cursor_coordinate_labels()

        def _update_cursor_coordinate_labels(self) -> None:
            position = self._cursor_position_from_sliders()
            if position is None:
                self.cursor_x_value_label.setText("X детектора = —")
                self.cursor_y_value_label.setText("Y детектора = —")
                return
            self.cursor_x_value_label.setText(f"X детектора = {position[0]:.4f}")
            self.cursor_y_value_label.setText(f"Y детектора = {position[1]:.4f}")

        def _configure_cursor_controls(self, payload: VolumePayload) -> None:
            """Reset the standalone input domain for the newly loaded payload."""
            timer = getattr(self, "_cursor_line_update_timer", None)
            if timer is not None and timer.isActive():
                timer.stop()
            self._pending_map_cursor_position = None
            self._last_map_cursor_position = None
            self._cursor_input_source = "sliders"
            self._cursor_detector_bounds = detector_cursor_bounds(payload)
            self.cursor_line_check.blockSignals(True)
            self.cursor_line_check.setChecked(False)
            self.cursor_line_check.blockSignals(False)
            self.cursor_x_slider.blockSignals(True)
            self.cursor_y_slider.blockSignals(True)
            try:
                self.cursor_x_slider.setValue(5000)
                self.cursor_y_slider.setValue(5000)
            finally:
                self.cursor_x_slider.blockSignals(False)
                self.cursor_y_slider.blockSignals(False)
            self._update_cursor_coordinate_labels()
            if self.scene is not None:
                self.scene.clear_cursor_line(render=False)

        def _reset_cursor_line_controls(self) -> None:
            timer = getattr(self, "_cursor_line_update_timer", None)
            if timer is not None and timer.isActive():
                timer.stop()
            self._pending_map_cursor_position = None
            self._last_map_cursor_position = None
            self._cursor_input_source = "sliders"
            self._cursor_detector_bounds = None
            if hasattr(self, "cursor_line_check"):
                self.cursor_line_check.blockSignals(True)
                self.cursor_line_check.setChecked(False)
                self.cursor_line_check.blockSignals(False)
            if hasattr(self, "cursor_x_slider"):
                self.cursor_x_slider.blockSignals(True)
                self.cursor_y_slider.blockSignals(True)
                try:
                    self.cursor_x_slider.setValue(5000)
                    self.cursor_y_slider.setValue(5000)
                finally:
                    self.cursor_x_slider.blockSignals(False)
                    self.cursor_y_slider.blockSignals(False)
                self._update_cursor_coordinate_labels()
            if self.scene is not None:
                self.scene.clear_cursor_line(render=False)

        def _on_cursor_slider_changed(self, _value: int) -> None:
            self._update_cursor_coordinate_labels()
            position = self._cursor_position_from_sliders()
            if position is not None:
                self._queue_cursor_position(position, source="sliders")

        def _on_cursor_line_toggled(self, enabled: bool) -> None:
            if self.scene is None or self.payload is None:
                return
            if not enabled:
                if self._cursor_line_update_timer.isActive():
                    self._cursor_line_update_timer.stop()
                self._pending_map_cursor_position = None
                try:
                    self.scene.set_cursor_line_enabled(False, render=False)
                    self.vtk_widget.Render()
                except Exception as exc:
                    self._show_error("Не удалось скрыть линию курсора", str(exc))
                return
            position = self._last_map_cursor_position
            if position is None and self._cursor_input_source == "sliders":
                position = self._cursor_position_from_sliders()
            if position is None:
                # A connected 2D map explicitly sent None (cursor left the
                # map); do not resurrect a stale slider/map position.
                return
            self._last_map_cursor_position = position
            try:
                self.scene.set_cursor_line_density(float(self.cursor_line_density.value()), render=False)
                self.scene.set_cursor_line_enabled(True, render=False)
                self._pending_map_cursor_position = position
                if not self._cursor_line_update_timer.isActive():
                    self._cursor_line_update_timer.start()
            except Exception as exc:
                self._show_error("Не удалось включить линию курсора", str(exc))

        def _on_cursor_line_width_changed(self, _value: float) -> None:
            if self.cursor_line_check.isChecked() and self._last_map_cursor_position is not None:
                self._pending_map_cursor_position = self._last_map_cursor_position
                if not self._cursor_line_update_timer.isActive():
                    self._cursor_line_update_timer.start()

        def _on_cursor_line_density_changed(self, value: float) -> None:
            """Adjust independent marker opacity without remapping coordinates."""
            if self.scene is None or not self.cursor_line_check.isChecked():
                return
            try:
                self.scene.set_cursor_line_density(float(value), render=False)
                if self.vtk_widget is not None:
                    self.vtk_widget.Render()
            except Exception as exc:
                self._show_error("Не удалось изменить оптическую плотность линии", str(exc))

        def _queue_cursor_position(self, position, *, source: str) -> None:
            self._cursor_input_source = str(source)
            if position is None:
                self._last_map_cursor_position = None
                self._pending_map_cursor_position = None
                if self._cursor_line_update_timer.isActive():
                    self._cursor_line_update_timer.stop()
                if self.scene is not None and self.cursor_line_check.isChecked():
                    self.scene.set_cursor_line_position(None, None, render=False)
                    if self.vtk_widget is not None:
                        self.vtk_widget.Render()
                return
            try:
                if isinstance(position, dict):
                    x, y = float(position["x"]), float(position["y"])
                else:
                    x, y = float(position[0]), float(position[1])
                if not (np.isfinite(x) and np.isfinite(y)):
                    raise ValueError("Координаты курсора должны быть конечными.")
            except Exception as exc:
                raise ValueError("Позиция курсора должна быть парой detector X/Y или словарём {'x','y'}.") from exc

            self._last_map_cursor_position = (x, y)
            if source == "map":
                self._set_cursor_sliders_from_detector_position(x, y)
                # If the map is larger than the active ROI, retain the true
                # cursor coordinates in the readout; the slider itself is
                # bounded by the current ROI's detector-space bounding box.
                self.cursor_x_value_label.setText(f"X детектора = {x:.4f}")
                self.cursor_y_value_label.setText(f"Y детектора = {y:.4f}")
            if self.cursor_line_check.isChecked() and self.payload is not None:
                self._pending_map_cursor_position = (x, y)
                if not self._cursor_line_update_timer.isActive():
                    self._cursor_line_update_timer.start()

        @QtCore.pyqtSlot(object)
        def set_map_cursor_position(self, position) -> None:
            """Public detector-XY input for a future ROI Explorer 2D→3D link.

            Accepts ``(x, y)``, ``[x, y]`` or ``{"x": x, "y": y}`` in
            detector coordinates. ``None`` means that the 2D cursor is outside
            the map (or is not currently valid) and hides the in-volume marker.
            A future map's Qt signal can connect to this slot; the standalone
            X/Y sliders use the same private scheduling/mapping path.
            """
            self._queue_cursor_position(position, source="map")

        def _apply_pending_map_cursor_position(self) -> None:
            position = self._pending_map_cursor_position
            self._pending_map_cursor_position = None
            if (
                position is None or not self.cursor_line_check.isChecked()
                or self.payload is None or self.scene is None or self.vtk_widget is None
            ):
                return
            try:
                target = map_detector_cursor_to_volume(
                    self.payload,
                    position[0],
                    position[1],
                    line_width_voxels=float(self.cursor_line_width.value()),
                )
                if target is None:
                    self.scene.set_cursor_line_position(None, None, render=False)
                    mapped = None
                else:
                    self.scene.set_cursor_line_position(
                        target.volume_x,
                        target.volume_y,
                        target.radius_x,
                        target.radius_y,
                        render=False,
                    )
                    mapped = (target.volume_x, target.volume_y)
                # Render() is QVTK's non-blocking QWidget.update() path. The
                # GPU mapper, volume texture and transfer functions are reused.
                self.vtk_widget.Render()
                _trace(
                    "CURSOR_LINE_POSITION_APPLIED",
                    detector_xy=position,
                    volume_local_xy=mapped,
                    radii=(target.radius_x, target.radius_y) if target is not None else None,
                    visible=target is not None,
                )
            except Exception as exc:
                self._show_error("Не удалось обновить линию курсора", str(exc))

        def _choose_file(self):
            # Hide an already-created QVTK/OpenGL canvas while a subsequent modal
            # file dialog is active. On first launch the parent window is still
            # hidden; the pre-created QVTK child therefore has no visible surface.
            parent_is_visible = self.isVisible()
            canvas_was_visible = bool(
                parent_is_visible and self.vtk_widget is not None
                and self.vtk_widget.isVisible()
            )
            if parent_is_visible and self.vtk_widget is not None:
                self.vtk_widget.setUpdatesEnabled(False)
                self.vtk_widget.hide()

            # For first launch, keep the parent hidden while the file dialog is
            # open. A parentless non-native dialog avoids transient-window
            # interactions with an unshown QMainWindow/QVTK child.
            dialog_parent = self if parent_is_visible else None
            dialog = QtWidgets.QFileDialog(dialog_parent)
            dialog.setOption(
                QtWidgets.QFileDialog.Option.DontUseNativeDialog, True
            )
            dialog.setWindowTitle("Выберите HDF5-измерение")
            if self.loader is not None:
                dialog.setDirectory(str(self.loader.h5_path.parent))
            dialog.setNameFilters([
                "HDF5 files (*.h5 *.hdf5)",
                "All files (*)",
            ])
            dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
            dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)

            try:
                accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
                selected = dialog.selectedFiles() if accepted else []
                filename = selected[0] if selected else ""
            finally:
                # The dialog is fully closed before a previously visible QVTK
                # canvas is restored. On first launch the parent is still hidden.
                if parent_is_visible and self.vtk_widget is not None and canvas_was_visible:
                    self.vtk_widget.setUpdatesEnabled(True)
                    self.vtk_widget.show()
                    self.vtk_widget.update()
                elif self.vtk_widget is not None:
                    self.vtk_widget.setUpdatesEnabled(True)

            if filename:
                try:
                    self.open_h5(filename)
                    if self.isVisible():
                        self._ensure_vtk_widget()
                except Exception as exc:
                    self._show_error("Не удалось открыть HDF5", str(exc))

        def _create_vtk_widget_shell(self):
            """Construct QVTK before the top-level window is shown; do not Initialize yet."""
            if self.vtk_widget is not None:
                return
            self.vtk_widget = TracedQVTKRenderWindowInteractor(self.viewer)
            self.render_window = self.vtk_widget.GetRenderWindow()

            # Fail fast before Initialize()/paintEvent if the OpenGL2 object
            # factory was not registered before QVTK construction. Rendering a
            # base vtkRenderWindow is not a recoverable Qt paint error and may
            # terminate the process in native VTK code.
            render_window_class = self.render_window.GetClassName()
            is_opengl_render_window = bool(
                self.render_window.IsA("vtkOpenGLRenderWindow")
            )
            _trace(
                "VTK_RENDER_WINDOW_BACKEND",
                render_window=render_window_class,
                is_opengl=is_opengl_render_window,
            )
            if not is_opengl_render_window:
                raise RuntimeError(
                    "QVTK создал базовый "
                    f"{render_window_class}, а не OpenGL RenderWindow. "
                    "Модули vtkRenderingOpenGL2 должны быть импортированы "
                    "ДО создания QVTK-виджета. Инициализация отменена, "
                    "чтобы не допустить нативного access violation."
                )

            # These settings must be applied before the native surface is created.
            self.render_window.SetMultiSamples(0)
            self.vtk_widget.setUpdatesEnabled(True)
            self.viewer_layout.addWidget(self.vtk_widget, 1)
            # Do not call hide(): the child must participate in the parent
            # show() sequence exactly as in the passing production-scene test.
            self.vtk_widget.paint_stage = "pre-init"
            _trace(
                "QVTK_WIDGET_PRECREATED",
                render_window=self.render_window.GetClassName(),
                multisamples=self.render_window.GetMultiSamples(),
            )

        def _ensure_vtk_widget(self):
            """Initialize QVTK after the Qt top-level window has been shown.

            QVTK's documented Qt lifecycle does not require a fixed number of
            paintEvent calls. The window must be visible and its native widget
            resource must exist; after that, Initialize() is called from the Qt
            event loop. This mirrors both Kitware's QVTK example and the
            successful standalone production-scene diagnostic in this project.
            """
            if self.scene is not None or self._vtk_init_pending:
                return
            if self.vtk_widget is None:
                raise RuntimeError("QVTK child was not pre-created before the main window was shown.")
            if not self.isVisible():
                # main() calls this again immediately after showing the parent.
                self._vtk_init_pending = False
                return
            self._vtk_init_pending = True
            self._vtk_init_attempts = 0
            self.open_button.setEnabled(False)
            self.placeholder_label.hide()
            # The child was created before QMainWindow.show() and is already
            # shown by Qt with its parent. QVTK's official example initializes
            # after show(); the project's successful production diagnostic uses
            # the same order with a 150 ms event-loop delay. Paint-event counts
            # are deliberately NOT used as a readiness condition: QWidget may
            # legitimately receive only one initial paint before VTK is ready.
            self.vtk_widget.setUpdatesEnabled(True)
            _trace(
                "QVTK_INIT_SCHEDULED",
                visible=self.vtk_widget.isVisible(),
                created=self.vtk_widget.testAttribute(
                    QtCore.Qt.WidgetAttribute.WA_WState_Created
                ),
            )
            # _ensure_vtk_widget itself is called by a queued Qt timer after
            # QMainWindow.show(). Run the short initialization here, in the GUI
            # thread, matching the known-good diagnostic ordering exactly.
            self._initialize_vtk_widget()

        def _initialize_vtk_widget(self):
            if self.vtk_widget is None:
                self._vtk_init_pending = False
                return
            self._vtk_init_attempts += 1
            try:
                widget_created = self.vtk_widget.testAttribute(
                    QtCore.Qt.WidgetAttribute.WA_WState_Created
                )
                if not self.isVisible() or not self.vtk_widget.isVisible() or not widget_created:
                    if self._vtk_init_attempts < 40:
                        QtCore.QTimer.singleShot(25, self._initialize_vtk_widget)
                        return
                    raise RuntimeError(
                        "Qt не создал видимый оконный ресурс QVTK после показа главного окна. "
                        "Проверьте parent/centralWidget и native window handle."
                    )

                self.render_window = self.vtk_widget.GetRenderWindow()
                if not bool(self.render_window.IsA("vtkOpenGLRenderWindow")):
                    raise RuntimeError(
                        "Перед Initialize() обнаружен не-OpenGL RenderWindow: "
                        f"{self.render_window.GetClassName()}."
                    )
                # Match vtk_production_scene_diagnostic.py: disable multisampling
                # before the first interactor initialization/render.
                self.render_window.SetMultiSamples(0)
                self.interactor = self.render_window.GetInteractor()
                if self.interactor is None:
                    raise RuntimeError(
                        "QVTK не создал vtkRenderWindowInteractor для render window."
                    )
                # Ask Qt for one explicit pre-initialization repaint. The
                # successful native diagnostic observed two completed paints
                # before Initialize(); this request encourages the same stable
                # backing-store/native-surface sequence without requiring a
                # fixed event count (Qt is free to coalesce update requests).
                if (not bool(self.interactor.GetInitialized())
                        and not self.vtk_widget._preinit_refresh_requested
                        and self.vtk_widget.paint_count < 2):
                    self.vtk_widget._preinit_refresh_requested = True
                    before_paints = int(self.vtk_widget.paint_count)
                    # Match the successful native diagnostic: force one
                    # synchronous pre-init paint while the child is visible.
                    # This is only a best-effort surface warm-up, never a
                    # readiness gate; if Qt coalesces/skips it, Initialize still
                    # proceeds using the actual native-widget checks below.
                    self.vtk_widget.repaint()
                    _trace(
                        "QVTK_PREINIT_REPAINT_DONE",
                        paint_count_before=before_paints,
                        paint_count_after=int(self.vtk_widget.paint_count),
                    )
                _trace_resources("before-qvtk-initialize")
                _trace(
                    "QVTK_INITIALIZE_BEGIN",
                    render_window=self.render_window.GetClassName(),
                    multisamples=self.render_window.GetMultiSamples(),
                    widget_size=(self.vtk_widget.width(), self.vtk_widget.height()),
                    paint_count=getattr(self.vtk_widget, "paint_count", None),
                    interactor=self.interactor.GetClassName(),
                )
                # vtkRenderWindowInteractor.Initialize() is the documented VTK
                # lifecycle entry point. QVTK forwards this method to its
                # vtkGenericRenderWindowInteractor. Never call Start(): Qt owns
                # the GUI event loop in an embedded QVTK application.
                if not bool(self.interactor.GetInitialized()):
                    self.vtk_widget.Initialize()
                initialized = bool(self.interactor.GetInitialized())
                if not initialized:
                    raise RuntimeError(
                        "QVTK Initialize() вернулся, но vtkRenderWindowInteractor "
                        "сообщает GetInitialized() == 0."
                    )
                self.vtk_widget.paint_stage = "post-init-empty"
                _trace_resources("after-qvtk-initialize")
                _trace(
                    "QVTK_INITIALIZE_DONE",
                    initialized=initialized,
                    enabled=self.interactor.GetEnabled(),
                    render_window=self.render_window.GetClassName(),
                    multisamples=self.render_window.GetMultiSamples(),
                )
                self.scene = VTKVolumeScene(
                    self.render_window,
                    self.interactor,
                    render_callback=self.vtk_widget.Render,
                )
                if hasattr(self, "grid_check"):
                    self.grid_check.blockSignals(True)
                    self.grid_check.setChecked(self.scene.grid_visible)
                    self.grid_check.blockSignals(False)
                self._vtk_init_pending = False
                _trace("VTK_SCENE_CREATED")
                self._refresh_controls()
                _trace_resources("qvtk-initialized")
                if self.loader is not None:
                    self.status_label.setText(
                        "VTK готов. Загружается выбранный scan в режиме ROI."
                    )
                    if self._auto_load_pending:
                        QtCore.QTimer.singleShot(0, self._run_pending_auto_load)
                else:
                    self.status_label.setText("Выберите HDF5-файл для загрузки объёма.")
            except Exception as exc:
                self._vtk_init_pending = False
                self.load_button.setEnabled(False)
                self.open_button.setEnabled(True)
                self._show_error("Не удалось инициализировать VTK/OpenGL", str(exc))

        def open_h5(self, path: str):
            loader = H5VolumeLoader(path)
            keys = loader.scan_keys
            if not keys:
                raise ValueError("В HDF5 не найдено ни одного поддерживаемого scan.")
            if self._initial_scan_key and self._initial_scan_key not in keys:
                raise KeyError(
                    f"Запрошенный scan {self._initial_scan_key!r} не найден. "
                    f"Доступно: {keys}"
                )
            preferred_scan = self._initial_scan_key
            self.loader = loader
            self._initial_scan_key = None
            self.placeholder_label.hide()
            _trace("H5_OPENED", path=str(loader.h5_path), scan_keys=keys)
            self.payload = None
            self._reset_theta_slice_controls()
            self._roi_geometry_initialized = False
            self._roi_by_scan.clear()
            self._active_roi_scan_key = None
            if self.scene is not None:
                self.scene.clear()
            self.path_label.setText(str(loader.h5_path))
            self.scan_combo.blockSignals(True)
            self.scan_combo.clear()
            self.scan_combo.addItems(keys)
            if preferred_scan and preferred_scan in keys:
                self.scan_combo.setCurrentText(preferred_scan)
            self.scan_combo.blockSignals(False)
            active_scan = self.scan_combo.currentText().strip()
            if active_scan:
                self._seed_roi_from_h5_metadata(loader, active_scan)
                self._seed_theta_controls(loader, active_scan)
                self._active_roi_scan_key = active_scan
            self._auto_load_pending = bool(self._auto_load_enabled)
            self.status_label.setText(
                f"Найдено scan: {len(keys)}. Выполняется подготовка центральной ROI; "
                "полный детектор можно выбрать отдельно."
            )
            if self.scene is not None and self._auto_load_pending:
                QtCore.QTimer.singleShot(0, self._run_pending_auto_load)

        def _run_pending_auto_load(self):
            if not self._auto_load_pending or self.loader is None or self.scene is None:
                return
            if self._thread is not None:
                return
            self._auto_load_pending = False
            self.load_volume()

        def _scan_changed(self, _index: int):
            # ROI is session state scoped to one (open HDF5, scan) identity.
            # Save the outgoing scan before switching; restore only that exact
            # scan's geometry, otherwise seed a new metadata-derived rectangle.
            previous_scan = self._active_roi_scan_key
            if previous_scan and self._roi_geometry_initialized:
                self._roi_by_scan[previous_scan] = self._current_roi_geometry()
            self.payload = None
            self._reset_theta_slice_controls()
            self._roi_geometry_initialized = False
            scan_key = self.scan_combo.currentText().strip()
            if self.loader is not None and scan_key:
                saved_roi = self._roi_by_scan.get(scan_key)
                if saved_roi is not None:
                    self._set_roi_geometry(saved_roi)
                    self._roi_geometry_initialized = True
                    _trace("ROI_RESTORED_FOR_SCAN", scan=scan_key, geometry=saved_roi)
                else:
                    self._seed_roi_from_h5_metadata(self.loader, scan_key)
                self._seed_theta_controls(self.loader, scan_key)
            self._active_roi_scan_key = scan_key or None
            if self.scene is not None:
                self.scene.clear()
            self._refresh_controls()

        def _seed_roi_from_h5_metadata(self, loader: H5VolumeLoader, scan_key: str):
            """Seed a bounded center ROI using HDF5 dataset metadata only (no data read)."""
            try:
                import h5py
                info = loader.repository.scans[scan_key]
                with h5py.File(loader.h5_path, "r") as h5:
                    raw_ds = h5[info.raw_path]
                    theta_ds = h5[info.th_path]
                    raw_shape = tuple(int(v) for v in raw_ds.shape)
                    theta_count = int(np.prod(theta_ds.shape, dtype=np.int64))
                candidates = [axis for axis in (0, 2) if raw_shape[axis] == theta_count]
                if len(candidates) != 1:
                    raise ValueError(
                        f"Не удалось однозначно определить theta-axis: "
                        f"raw_shape={raw_shape}, theta_count={theta_count}."
                    )
                ny, nx = (raw_shape[1], raw_shape[2]) if candidates[0] == 0 else (raw_shape[0], raw_shape[1])
                if nx < 1 or ny < 1:
                    raise ValueError(f"Некорректная detector shape: {(ny, nx)}")
                self.roi_cx.setValue((nx - 1) / 2.0)
                self.roi_cy.setValue((ny - 1) / 2.0)
                self.roi_width.setValue(min(float(nx), 256.0, max(32.0, float(nx) * 0.35)))
                self.roi_height.setValue(min(float(ny), 256.0, max(32.0, float(ny) * 0.35)))
                _trace(
                    "ROI_INITIALIZED_FROM_DETECTOR_SHAPE",
                    scan=scan_key,
                    detector_shape_yx=(ny, nx),
                    center=(self.roi_cx.value(), self.roi_cy.value()),
                    size=(self.roi_width.value(), self.roi_height.value()),
                )
                # The initial rectangle belongs to this scan now. Loading a
                # full-detector payload must never overwrite this geometry.
                self._roi_geometry_initialized = True
            except Exception as exc:
                self._roi_geometry_initialized = False
                _trace("ROI_METADATA_SEED_FAILED", scan=scan_key, error=f"{type(exc).__name__}: {exc}")
                self.status_label.setText(
                    "Не удалось автоматически определить размер detector-grid. "
                    "Проверьте центр и размеры ROI вручную."
                )

        def _current_roi_geometry(self) -> tuple[float, float, float, float, float]:
            """Return the exact editable rectangle in detector-grid coordinates."""
            return (
                float(self.roi_cx.value()),
                float(self.roi_cy.value()),
                float(self.roi_width.value()),
                float(self.roi_height.value()),
                float(self.roi_angle.value()),
            )

        def _set_roi_geometry(self, geometry: tuple[float, float, float, float, float]) -> None:
            cx, cy, width, height, angle = geometry
            self.roi_cx.setValue(cx)
            self.roi_cy.setValue(cy)
            self.roi_width.setValue(width)
            self.roi_height.setValue(height)
            self.roi_angle.setValue(angle)

        def _current_roi(self):
            from roc_mcs.roi.roi_types import ROI
            return ROI(
                cx=float(self.roi_cx.value()),
                cy=float(self.roi_cy.value()),
                width=float(self.roi_width.value()),
                height=float(self.roi_height.value()),
                angle=float(self.roi_angle.value()),
                label="VTK ROI",
                type="rect",
            )

        def _estimate_load_memory(self, scan_key: str, remove_spikes: bool) -> tuple[int, int | None]:
            try:
                import h5py
                info = self.loader.repository.scans[scan_key]
                with h5py.File(self.loader.h5_path, "r") as h5:
                    dataset = h5[info.raw_path]
                    voxels = int(np.prod(dataset.shape, dtype=np.int64))
                    source_bytes = int(voxels * dataset.dtype.itemsize)
                # Metadata-only estimate; do not read the scan on the GUI thread.
                # Conservative CPU peak: HDF5 source + the payload-owned
                # display array + the renderer's Fortran copy + VTK's deep
                # scalar copy + the finite percentile sample.  Spike removal
                # adds masks and its own working arrays. GPU-driver allocations
                # are intentionally not claimed by this CPU estimate.
                estimate = source_bytes + voxels * (32 + (64 if remove_spikes else 0))
                try:
                    import psutil
                    available = int(psutil.virtual_memory().available)
                except Exception:
                    available = None
                return estimate, available
            except Exception:
                return 0, None

        def load_volume(self):
            if self._closing or self._busy or self._thread is not None:
                return
            if self.loader is None:
                self._choose_file()
                if self.loader is None:
                    return
            if self.scene is None:
                self.status_label.setText(
                    "Инициализируется VTK/OpenGL. Нажмите загрузку объёма после появления сцены."
                )
                return
            scan_key = self.scan_combo.currentText().strip()
            if not scan_key:
                self._show_error("Нет scan", "В HDF5 не выбран scan.")
                return
            mode = str(self.mode_combo.currentData())
            try:
                roi = self._current_roi() if mode == "roi" else None
            except Exception as exc:
                self._show_error("Некорректная ROI", str(exc))
                return
            theta_range = None
            if self.theta_crop_check.isChecked():
                theta_range = (
                    float(self.theta_min_spin.value()),
                    float(self.theta_max_spin.value()),
                )
                if not theta_range[0] < theta_range[1]:
                    self._show_error("Некорректный диапазон θ", "Нужно задать θ от < θ до.")
                    return

            estimated, available = self._estimate_load_memory(scan_key, self.spikes_check.isChecked())
            _trace_resources("before-load-volume")
            _trace(
                "LOAD_MEMORY_PREFLIGHT",
                scan=scan_key,
                mode=mode,
                estimated_extra_mb=round(estimated / (1024.0 ** 2), 1),
                available_mb=(round(available / (1024.0 ** 2), 1) if available is not None else None),
            )
            if available is not None and estimated > available * 0.45:
                answer = QtWidgets.QMessageBox.warning(
                    self,
                    "Большой объём данных",
                    f"Консервативная оценка дополнительной CPU-памяти: {estimated / (1024**3):.2f} GiB; "
                    f"свободно сейчас: {available / (1024**3):.2f} GiB.\n\n"
                    "Загрузка и удаление спайков могут потребовать больше памяти из-за "
                    "временных массивов. Память GPU/драйвера в этой оценке не учитывается. Продолжить?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel,
                )
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    return

            self._set_busy(True, "Загрузка HDF5 и подготовка объёма…")
            self._pending_load_outcome = {}
            self._thread = QtCore.QThread(self)
            self._worker = LoadWorker(
                self.loader, scan_key, mode, roi,
                self.spikes_check.isChecked(), float(self.smoothing_spin.value()),
                theta_range,
                self._pending_load_outcome,
            )
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            # Quit from the worker's completion signal using a direct connection;
            # the window consumes the result only after QThread.finished.
            self._worker.completed.connect(
                self._thread.quit, QtCore.Qt.ConnectionType.DirectConnection
            )
            self._thread.finished.connect(self._thread_finished)
            self._thread.finished.connect(self._worker.deleteLater)
            self._thread.start()

        @QtCore.pyqtSlot(object)
        def _load_finished(self, payload: VolumePayload):
            try:
                axis_spacing = {}
                for name, axis in (("x", payload.x), ("y", payload.y), ("theta", payload.theta)):
                    diffs = np.abs(np.diff(np.asarray(axis, dtype=np.float64)))
                    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                    axis_spacing[name] = {
                        "min": float(np.min(diffs)) if diffs.size else None,
                        "median": float(np.median(diffs)) if diffs.size else None,
                        "max": float(np.max(diffs)) if diffs.size else None,
                    }
                _trace(
                    "PAYLOAD_READY",
                    scan=payload.scan_key,
                    mode=payload.view_mode,
                    shape=payload.shape,
                    dtype=payload.scalar_dtype,
                    nbytes=payload.nbytes,
                    finite=payload.finite_count,
                    invalid=payload.invalid_count,
                    masked_detector_pixels=payload.masked_detector_pixels,
                    coordinate_system=payload.coord_system,
                    axis_spacing=axis_spacing,
                )
                _trace("SCENE_SET_PAYLOAD_BEGIN")
                _trace_resources("before-scene-rebuild")
                info = self.scene.set_payload(
                    payload,
                    self._transfer_settings(),
                    theta_scale=float(self.theta_scale_spin.value()),
                )
                self.payload = payload
                self._configure_cursor_controls(payload)
                theta_count = int(len(payload.theta))
                theta_index = int(self.scene.theta_slice_index or 0)
                self.theta_slice_slider.blockSignals(True)
                self.theta_slice_slider.setRange(0, max(0, theta_count - 1))
                self.theta_slice_slider.setValue(min(max(theta_index, 0), max(0, theta_count - 1)))
                self.theta_slice_slider.blockSignals(False)
                self.theta_slice_value_label.setText(self._theta_slice_label_text(theta_index))
                self.theta_slice_check.blockSignals(True)
                self.theta_slice_check.setChecked(bool(self.scene.theta_slice_enabled))
                self.theta_slice_check.blockSignals(False)
                self.theta_slice_palette_combo.blockSignals(True)
                palette_index = self.theta_slice_palette_combo.findData(self.scene.theta_slice_palette)
                if palette_index >= 0:
                    self.theta_slice_palette_combo.setCurrentIndex(palette_index)
                self.theta_slice_palette_combo.blockSignals(False)
                self.theta_slice_strength.blockSignals(True)
                self.theta_slice_strength.setValue(float(self.scene.theta_slice_strength))
                self.theta_slice_strength.blockSignals(False)
                self.vtk_widget.paint_stage = "payload-set"
                _trace(
                    "SCENE_SET_PAYLOAD_DONE",
                    grid_points=info.get("point_count"),
                    render_window=self.render_window.GetClassName(),
                    volume_mapper=self.scene.mapper.GetClassName(),
                    sample_distance=self.scene.mapper.GetSampleDistance(),
                    auto_adjust=self.scene.mapper.GetAutoAdjustSampleDistances(),
                    jitter=self.scene.mapper.GetUseJittering() if hasattr(self.scene.mapper, "GetUseJittering") else None,
                    projection_visible=info.get("projection_visible"),
                    projection_dimensions=info.get("projection_dimensions"),
                    threshold_reference=info.get("threshold_reference"),
                    threshold_min=info.get("threshold_min"),
                    threshold_max=info.get("threshold_max"),
                    opacity_unit_distance=info.get("opacity_unit_distance"),
                )
                _trace("SCENE_DIAGNOSTICS", **self.scene.diagnostics())
                _trace_resources("after-scene-rebuild")
                _trace_resources("payload-set")
                if not self._roi_geometry_initialized and payload.view_mode == "full":
                    self._seed_roi_from_payload(payload)
                s = payload.spike_report
                spike_text = (
                    f"Спайки: κ={s.kappa:.3g}; удалено {s.invalid_count:,} "
                    f"({s.invalid_fraction:.3%}); σфон={s.sigma:.4g}"
                    if s.enabled else "Спайки: не удалялись"
                )
                self.status_label.setText(
                    f"{payload.scan_key} | {payload.view_mode} | {payload.shape} | "
                    f"{payload.finite_count:,} конечных / {payload.invalid_count:,} invalid\n"
                    f"Координаты: {payload.coord_system}; dtype={payload.scalar_dtype}; "
                    f"сетка={info['point_count']:,} узлов\n{spike_text}\n"
                    f"Исключённые пиксели: {len(payload.masked_detector_pixels)}; "
                    f"масштаб θ: {float(self.theta_scale_spin.value()):g}×; "
                    f"диапазон θ: [{payload.theta[0]:g}, {payload.theta[-1]:g}]\n"
                    f"Сглаживание σ={payload.smoothing_sigma:g} (воксели); "
                    f"пороговый sample={info['threshold_sample_count']:,} "
                    f"(stride {info['threshold_sample_stride']})\n"
                    f"Пороги интенсивности: Vmin={info['threshold_min']:.5g}; "
                    f"Vmax={info['threshold_max']:.5g}; reference p99.9={info['threshold_reference']:.5g}\n"
                    f"GPU sampling: {info['sample_distance']:.5g}; "
                    f"opacity unit distance={info['opacity_unit_distance']:.5g}; "
                    f"XY projection={'on' if info['projection_visible'] else 'off'} "
                    f"dims={info.get('projection_dimensions')}; "
                    f"opacity={float(self.projection_opacity.value()):.3g}; "
                    f"auto-adjust={info['auto_adjust_sample_distances']}; jitter={info['jittering']}"
                )
                if info.get("gpu_supported") is False:
                    self._show_error(
                        "GPU volume rendering не поддерживается",
                        "VTK сообщает, что GPU mapper не поддерживается текущим OpenGL-контекстом. "
                        "Проверьте драйвер видеокарты и возможности OpenGL.",
                    )
            except Exception as exc:
                self._show_error("Ошибка построения VTK-сцены", str(exc))
                _trace("SCENE_SET_PAYLOAD_FAILED", error=f"{type(exc).__name__}: {exc}")

        def _seed_roi_from_payload(self, payload: VolumePayload):
            # Fallback used only if HDF5 metadata could not initialize the
            # detector geometry. Keep the same 35% default as the metadata path.
            x0, x1 = float(payload.x[0]), float(payload.x[-1])
            y0, y1 = float(payload.y[0]), float(payload.y[-1])
            self.roi_cx.setValue(0.5 * (x0 + x1))
            self.roi_cy.setValue(0.5 * (y0 + y1))
            self.roi_width.setValue(max((x1 - x0 + 1.0) * 0.35, 0.001))
            self.roi_height.setValue(max((y1 - y0 + 1.0) * 0.35, 0.001))
            self.roi_angle.setValue(0.0)
            self._roi_geometry_initialized = True

        def _seed_theta_controls(self, loader: H5VolumeLoader, scan_key: str):
            """Initialize crop controls from theta metadata without reading detector data."""
            self.theta_crop_check.setChecked(False)
            try:
                theta_low, theta_high = loader.theta_bounds(scan_key)
                self.theta_min_spin.setValue(theta_low)
                self.theta_max_spin.setValue(theta_high)
                self.theta_crop_check.setChecked(False)
                _trace("THETA_RANGE_INITIALIZED", scan=scan_key, bounds=(theta_low, theta_high))
            except Exception as exc:
                _trace("THETA_RANGE_METADATA_FAILED", scan=scan_key,
                       error=f"{type(exc).__name__}: {exc}")

        def _edit_bad_pixels(self):
            current = load_detector_bad_pixels()
            text = "; ".join(f"{x},{y}" for x, y in current)
            value, accepted = QtWidgets.QInputDialog.getText(
                self,
                "Исключаемые пиксели детектора",
                "Нулевые индексы x,y через точку с запятой.\n"
                "Каждый пиксель маскируется на всех углах θ и во всех scan.\n"
                "Оставьте поле пустым, чтобы отключить маску:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                text,
            )
            if not accepted:
                return
            try:
                normalized = normalize_detector_bad_pixels(value)
                save_detector_bad_pixels(normalized)
                if self.loader is not None:
                    self.loader.set_bad_pixels(normalized)
                _trace("DETECTOR_BAD_PIXELS_SAVED", pixels=[(p["x"], p["y"]) for p in normalized])
                self.status_label.setText(
                    "Список дефектных пикселей сохранён глобально. "
                    "Перестройте объём, чтобы применить его к текущему scan."
                )
            except Exception as exc:
                self._show_error("Некорректный список пикселей", str(exc))

        def _load_failed(self, message: str):
            self.status_label.setText("Загрузка не выполнена.")
            self._show_error("Не удалось подготовить объём", message)

        def _thread_finished(self):
            # This slot runs in the GUI thread, after the worker has stopped.
            # Only now may another load start or controls become interactive.
            thread = self._thread
            if thread is None:
                return
            outcome = self._pending_load_outcome or {}
            self._thread = None
            self._worker = None
            self._pending_load_outcome = None
            try:
                thread.deleteLater()
            except Exception:
                pass

            if self._closing:
                _trace("LOAD_THREAD_FINISHED_DURING_CLOSE")
                self.close()
                return

            if "payload" in outcome:
                self._load_finished(outcome["payload"])
            else:
                self._load_failed(str(outcome.get("error", "Рабочий поток завершился без результата.")))
            # Drop the worker-result dictionary as soon as the new payload is
            # owned by the active viewer/scene. A full detector contains tens
            # of MB and retaining an obsolete worker result would look like a
            # renderer leak during repeated ROI/full rebuilds.
            outcome.clear()
            collected = gc.collect()
            _trace("PYTHON_GC_AFTER_LOAD", collected_objects=collected)
            self._set_busy(False)
            _trace_resources("load-thread-finished")

        def _set_busy(self, busy: bool, message: str = ""):
            self._busy = bool(busy)
            if message:
                self.status_label.setText(message)
            self._refresh_controls()

        def _transfer_settings(self) -> TransferSettings:
            low = float(self.low_pct.value()) / 100.0
            high = float(self.high_pct.value()) / 100.0
            if not (0.0 <= low < high <= 1.0):
                raise ValueError("Нужно 0 <= Vmin < Vmax <= 100% от p99.9.")
            return TransferSettings(
                lower_fraction=low,
                upper_fraction=high,
                max_opacity=float(self.max_opacity.value()),
                gamma=float(self.gamma.value()),
                palette=str(self.palette_combo.currentData()),
                sample_distance_factor=float(self.sample_factor.value()),
                opacity_unit_distance_factor=float(self.opacity_unit_factor.value()),
                show_projection=bool(self.projection_check.isChecked()),
                projection_opacity=float(self.projection_opacity.value()),
            )

        def apply_display(self):
            if self.payload is None or self.scene is None:
                self.status_label.setText("Сначала загрузите объём.")
                return
            try:
                settings = self._transfer_settings()
                self.scene.set_theme(str(self.theme_combo.currentData()))
                self.scene.set_theta_scale(float(self.theta_scale_spin.value()), render=False)
                # Recompute ray sampling/opacity distance from displayed voxel
                # geometry after changing theta scale.
                info = self.scene.update_transfer_functions(settings, render=False)
                # Theme/property updates are applied in the GUI thread; render is
                # requested once after all controls have been updated.
                self.vtk_widget.Render()
                self.status_label.setText(
                    f"Отображение обновлено: Vmin={info['threshold_min']:.5g}; "
                    f"Vmax={info['threshold_max']:.5g}; p99.9={info['threshold_reference']:.5g}; "
                    f"max opacity={settings.max_opacity:g}; gamma={settings.gamma:g}; "
                    f"sample distance={info['sample_distance']:.5g}; "
                    f"opacity unit distance={info['opacity_unit_distance']:.5g}; "
                    f"palette={settings.palette}; θ scale={float(self.theta_scale_spin.value()):g}×; "
                    f"XY projection={'on' if settings.show_projection else 'off'} "
                    f"(opacity {settings.projection_opacity:g})."
                )
                _trace("DISPLAY_SETTINGS_APPLIED", **info, settings={
                    "lower_fraction": settings.lower_fraction,
                    "upper_fraction": settings.upper_fraction,
                    "max_opacity": settings.max_opacity,
                    "gamma": settings.gamma,
                    "sample_distance_factor": settings.sample_distance_factor,
                    "opacity_unit_distance_factor": settings.opacity_unit_distance_factor,
                    "show_projection": settings.show_projection,
                    "projection_opacity": settings.projection_opacity,
                    "theta_scale": float(self.theta_scale_spin.value()),
                })
            except Exception as exc:
                self._show_error("Некорректные настройки отображения", str(exc))

        def _camera_axis(self, axis: str):
            if self.payload is not None and self.scene is not None:
                self.scene.reset_camera_axis(axis)

        def _camera_perspective(self, enabled: bool):
            if self.payload is not None and self.scene is not None:
                self.scene.set_perspective(enabled)

        def _toggle_axes(self, visible: bool):
            if self.scene is not None:
                self.scene.set_axes_visible(visible)

        def _toggle_grid(self, visible: bool):
            if self.scene is not None:
                self.scene.set_grid_visible(visible)

        def _show_error(self, title: str, text: str):
            parent = self if self.isVisible() else None
            QtWidgets.QMessageBox.critical(parent, title, text)

        def closeEvent(self, event):
            if self._thread is not None:
                self._closing = True
                if self._thread.isRunning():
                    self.status_label.setText(
                        "Завершается подготовка данных; окно закроется после её окончания."
                    )
                else:
                    QtCore.QTimer.singleShot(0, self._thread_finished)
                event.ignore()
                return
            _trace_resources("before-window-close")
            try:
                if self._cursor_line_update_timer.isActive():
                    self._cursor_line_update_timer.stop()
                self._pending_map_cursor_position = None
                self._last_map_cursor_position = None
            except Exception:
                pass
            try:
                if self.scene is not None:
                    self.scene.close()
            except Exception as exc:
                _trace("SCENE_CLOSE_WARNING", error=f"{type(exc).__name__}: {exc}")
            try:
                if self.vtk_widget is not None and not self.vtk_widget._finalized:
                    self.vtk_widget.setUpdatesEnabled(False)
                    self.vtk_widget.close()  # QVTK closeEvent performs Finalize exactly once.
            except Exception as exc:
                _trace("QVTK_CLOSE_WARNING", error=f"{type(exc).__name__}: {exc}")
            _trace_resources("after-qvtk-close")
            event.accept()

    return VTKH5ViewerWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone ROI Explorer VTK HDF5 volume viewer")
    parser.add_argument("--h5", type=Path, help="Путь к HDF5-файлу")
    parser.add_argument("--scan", help="Ключ scan, например entry459 или entry459_CALC")
    parser.add_argument("--mode", choices=("roi", "full"), default="roi",
                        help="Начальный режим загрузки (по умолчанию roi)")
    parser.add_argument("--no-auto-load", action="store_true",
                        help="Не загружать объём автоматически после открытия HDF5")
    args = parser.parse_args(argv)

    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    QtCore, QtWidgets, QVTKRenderWindowInteractor = _qt_modules()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("ROI Explorer VTK H5 Viewer")
    app.setQuitOnLastWindowClosed(False)
    vtk_qt = __import__("vtkmodules.qt", fromlist=["PyQtImpl", "QVTKRWIBase"])
    from vtkmodules.vtkCommonCore import vtkVersion
    _trace(
        "APP_START",
        python=sys.executable,
        vtk_version=vtkVersion.GetVTKVersion(),
        vtk_qt_binding=getattr(vtk_qt, "PyQtImpl", None),
        vtk_qt_base=getattr(vtk_qt, "QVTKRWIBase", None),
        qt_version=QtCore.qVersion(),
    )
    _trace_resources("application-start")
    Window = _create_window_class(QtCore, QtWidgets, QVTKRenderWindowInteractor)
    window = Window(None, args.scan)
    window._auto_load_enabled = not bool(args.no_auto_load)
    window._initial_mode = str(args.mode)
    window.mode_combo.setCurrentIndex(window.mode_combo.findData(args.mode))
    if args.h5:
        try:
            window.open_h5(str(args.h5))
        except Exception as exc:
            _trace("INITIAL_H5_OPEN_FAILED", error=f"{type(exc).__name__}: {exc}")
            return 2
    else:
        # First choose the H5 while the main window and QVTK child are both
        # unshown. QVTK already exists in the widget tree before QMainWindow.show.
        window._choose_file()
        if window.loader is None:
            _trace("INITIAL_H5_SELECTION_CANCELLED")
            return 0

    # Maximized is preferable to a hard-coded 920 px window: Qt uses the
    # screen's available work area, so the lower controls remain above the
    # Windows taskbar instead of being clipped behind it.
    window.showMaximized()
    app.setQuitOnLastWindowClosed(True)
    # Match the known-good native diagnostic and Kitware's Qt example: show
    # the parent first, then initialize QVTK from the active Qt event loop.
    # A fixed number of paintEvent calls is NOT a QVTK readiness requirement.
    QtCore.QTimer.singleShot(150, window._ensure_vtk_widget)
    _trace("QT_EVENT_LOOP_ENTER")
    result = int(app.exec())
    _trace("QT_EVENT_LOOP_RETURN", result=result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
