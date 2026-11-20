"""Ленивый VTK-рендерер ROI Explorer; источник данных задаёт само приложение."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import logging

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from roc_mcs.gui.roi_explorer.theme import resolve_theme, theme_colors
from .cursor_link import map_detector_cursor_to_volume
from .scan_view_adapter import build_volume_payload
from roc_mcs.roi.roi_types import normalize_rectangular_roi
from .vtk_scene import TransferSettings, VTKVolumeScene
from .vtk_tools_widget import VTKToolsWidget
from .vtk_volume_display_widget import VTKVolumeDisplayWidget


logger = logging.getLogger(__name__)


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(int, object, object)


class _PayloadWorker(QtCore.QRunnable):
    def __init__(self, token: int, request: dict):
        super().__init__()
        self.token = int(token)
        self.request = request
        self.signals = _WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            payload = build_volume_payload(**self.request)
            self.signals.finished.emit(self.token, payload, None)
        except Exception as exc:
            self.signals.finished.emit(self.token, None, exc)


class EmbeddedVTKVolumeWidget(QtWidgets.QWidget):
    """Лёгкая оболочка; QVTK/OpenGL создаются только при первом показе окна."""

    data_rebuild_requested = QtCore.pyqtSignal()
    # Physical theta coordinate (degrees) changed by the user in the 3D HUD.
    # The explorer maps this value to its own 2D theta axis; indices are not
    # assumed to match because either source may be stored in descending order.
    theta_slice_value_changed = QtCore.pyqtSignal(float)

    # Choose one of the four supplied scientific source symbols (1..4).
    # 1 is the default nested-volume/cube glyph.
    SOURCE_ICON_VARIANT = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("EmbeddedVTKVolumeWidget")
        self._theme = resolve_theme(self)
        self.setProperty("rocMcsTheme", self._theme)
        self._qvtk = None
        self._render_window = None
        self._interactor = None
        self._scene = None
        self._display = None
        self._payload = None
        self._dataset_request: dict | None = None
        self._active_worker = None
        self._pending_request: tuple[int, dict] | None = None
        self._generation = 0
        self._closed = False
        self._init_pending = False
        self._init_generation = 0
        self._first_payload = True
        self._map_cursor_position = None
        self._roi_preview = None
        self._rois: list = []
        self._selected_roi_id: int | None = None
        self._source_scan_key: str | None = None
        self._map_theta_mode = False
        self._map_theta_values = np.empty(0, dtype=np.float64)
        self._map_theta_index = 0
        self._transfer = TransferSettings()
        self._settings = {}
        self._settings_timer = QtCore.QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(80)
        self._settings_timer.timeout.connect(self._apply_transfer_settings)
        self._roi_preview_timer = QtCore.QTimer(self)
        self._roi_preview_timer.setSingleShot(True)
        self._roi_preview_timer.setInterval(16)
        self._roi_preview_timer.timeout.connect(self._flush_roi_preview)
        self._thread_pool = QtCore.QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        self.toolbar = QtWidgets.QWidget(self)
        self.toolbar.setObjectName("EmbeddedVTKToolbar")
        row = QtWidgets.QHBoxLayout(self.toolbar)
        row.setContentsMargins(3, 1, 3, 1)
        row.setSpacing(4)
        # Keep the combo as an internal selection model for API compatibility,
        # but expose the source choice through one compact icon-only menu.
        self.source_combo = QtWidgets.QComboBox(self.toolbar)
        self.source_combo.setObjectName("vtkExplorerVolumeSource")
        self.source_combo.addItem("Полный детектор", "full")
        self.source_combo.hide()
        self.source_button = QtWidgets.QToolButton(self.toolbar)
        self.source_button.setObjectName("vtkExplorerSourceButton")
        self.source_button.setAutoRaise(True)
        self.source_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.source_button.setFixedSize(28, 28)
        self.source_button.setIconSize(QtCore.QSize(19, 19))
        self.source_button.setToolTip("Выбрать источник 3D-объёма")
        self.source_button.setAccessibleName("Источник 3D-объёма")
        self.source_menu = QtWidgets.QMenu(self.source_button)
        self.source_menu.setObjectName("vtkExplorerSourceMenu")
        self.source_menu.setToolTipsVisible(True)
        self.source_button.setMenu(self.source_menu)
        row.addWidget(self.source_button, 0)
        self.tools = VTKToolsWidget(self.toolbar)
        self.tools.volume_source_btn.hide()
        self.tools.theme_btn.hide()
        row.addWidget(self.tools, 1)
        root.addWidget(self.toolbar, 0)

        self.viewport = QtWidgets.QWidget(self)
        self.viewport.setObjectName("EmbeddedVTKViewport")
        self.viewport_layout = QtWidgets.QVBoxLayout(self.viewport)
        self.viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.viewport_layout.setSpacing(0)
        self.placeholder = QtWidgets.QLabel("3D-сцена будет создана при первом открытии окна.", self.viewport)
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.viewport_layout.addWidget(self.placeholder)
        root.addWidget(self.viewport, 1)

        self.status = QtWidgets.QLabel("3D не инициализировано", self)
        self.status.setObjectName("vtkEmbeddedStatus")
        self.status.setMinimumHeight(18)
        self.status.setWordWrap(True)
        root.addWidget(self.status, 0)

        self._build_overlay_settings()
        self._connect_tools()
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self._update_source_icon()
        self._rebuild_source_menu()
        self.source_button.setEnabled(False)
        self.tools.set_scene_available(False)
        self.tools.set_volume_source_available(False)
        self.tools.set_overlay_controls_enabled(cursor=False, theta=False, projection=False, roi_preview=False)
        self.apply_theme(self._theme)

    @property
    def scene_initialized(self) -> bool:
        return self._scene is not None

    @property
    def vtk_widget(self):
        return self._qvtk

    @property
    def has_dataset_request(self) -> bool:
        return self._dataset_request is not None

    def _spin(self, minimum, maximum, value, decimals=2, suffix=""):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(float(minimum), float(maximum))
        widget.setDecimals(int(decimals))
        widget.setValue(float(value))
        widget.setSuffix(str(suffix))
        widget.setKeyboardTracking(False)
        widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        widget.setFixedWidth(76)
        return widget

    def _build_overlay_settings(self):
        self.cursor_width = self._spin(0.5, 4.0, 1.6, 1, " vox")
        self.cursor_density = self._spin(0.05, 3.0, 0.45, 2)
        cursor = self._settings_panel(("Ширина", self.cursor_width), ("Оптич. плотность", self.cursor_density))

        self.theta_index = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.theta_index.setRange(0, 0)
        self.theta_label = QtWidgets.QLabel("θ = —")
        self.theta_palette = QtWidgets.QComboBox()
        for name in ("viridis", "inferno", "magma", "plasma", "cividis", "gray"):
            self.theta_palette.addItem(name, name)
        self.theta_strength = self._spin(0.0, 1.0, 0.75, 2)
        theta = self._settings_panel(("Срез", self.theta_label), ("Индекс θ", self.theta_index), ("Палитра", self.theta_palette), ("Сила", self.theta_strength))
        self.theta_link_checkbox = QtWidgets.QCheckBox("Связать с 2D-картой")
        self.theta_link_checkbox.setObjectName("vtkThetaSliceLinkCheckbox")
        self.theta_link_checkbox.setChecked(True)
        self.theta_link_checkbox.setEnabled(False)
        self.theta_link_checkbox.setToolTip(
            "Синхронизация доступна только в режиме 2D-карты «Срез по θ» "
            "и при совпадающих физических θ-осях."
        )
        self.theta_link_checkbox.setAccessibleName("Связать срез θ в 3D с 2D-картой")
        theta.layout().addRow(self.theta_link_checkbox)

        self.projection_opacity = self._spin(0.0, 1.0, 0.75, 2)
        self.projection_palette = QtWidgets.QComboBox()
        for name in ("auto", "Greys", "Greys_r", "viridis", "inferno", "magma"):
            self.projection_palette.addItem(name, None if name == "auto" else name)
        projection = self._settings_panel(("Непрозрачность", self.projection_opacity), ("Палитра", self.projection_palette))

        self.roi_preview_opacity = self._spin(0.02, 0.35, 0.10, 2)
        self.roi_preview_strength = self._spin(0.0, 1.0, 0.92, 2)
        self.roi_preview_mode = QtWidgets.QComboBox()
        for label, value in (("Внутри ROI", "inside"), ("Контекст", "context"), ("Только контур", "off")):
            self.roi_preview_mode.addItem(label, value)
        roi = self._settings_panel(("Непрозрачность", self.roi_preview_opacity), ("Подсветка", self.roi_preview_strength), ("Режим", self.roi_preview_mode))
        self.tools.set_overlay_settings_widgets(cursor_line=cursor, theta_slice=theta, projection=projection, roi_preview=roi)

        self.cursor_width.valueChanged.connect(self._update_cursor)
        self.cursor_density.valueChanged.connect(self._update_cursor_density)
        self.theta_index.valueChanged.connect(self._apply_theta_index)
        self.theta_link_checkbox.toggled.connect(self._on_theta_link_toggled)
        self.theta_palette.currentIndexChanged.connect(self._apply_theta_palette)
        self.theta_strength.valueChanged.connect(self._apply_theta_strength)
        self.projection_opacity.valueChanged.connect(self._queue_transfer_settings)
        self.projection_palette.currentIndexChanged.connect(self._queue_transfer_settings)
        self.roi_preview_opacity.valueChanged.connect(self._update_roi_preview)
        self.roi_preview_strength.valueChanged.connect(self._update_roi_preview)
        self.roi_preview_mode.currentIndexChanged.connect(self._update_roi_preview)

        self.display_panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.display_panel)
        form.setContentsMargins(8, 7, 8, 7)
        form.setSpacing(5)
        self.low_threshold = self._spin(0.0, 99.0, 16.0, 1, " %")
        self.high_threshold = self._spin(1.0, 100.0, 95.0, 1, " %")
        self.opacity = self._spin(0.0, 1.0, 0.18, 2)
        self.gamma = self._spin(0.05, 8.0, 2.0, 2)
        self.sample_factor = self._spin(0.25, 8.0, 1.0, 2, " ×")
        self.unit_distance = self._spin(0.05, 20.0, 1.0, 2, " ×")
        self.palette = QtWidgets.QComboBox(self.display_panel)
        from roc_mcs.gui.roi_explorer.colormap_catalog import NAPARI_STYLE_COLORMAPS
        for name in NAPARI_STYLE_COLORMAPS:
            self.palette.addItem(str(name), str(name))
        for label, control in (("Vmin", self.low_threshold), ("Vmax", self.high_threshold), ("Макс. opacity", self.opacity), ("Gamma", self.gamma), ("Ray sample", self.sample_factor), ("Opacity distance", self.unit_distance), ("Colormap", self.palette)):
            form.addRow(label, control)
        self.display = None
        self._threshold_sync = False
        for control in (self.low_threshold, self.high_threshold, self.opacity, self.gamma, self.sample_factor, self.unit_distance):
            control.valueChanged.connect(self._queue_transfer_settings)
        self.palette.currentIndexChanged.connect(self._queue_transfer_settings)

    def _update_source_icon(self) -> None:
        """Apply the selected monochrome source glyph to the compact button."""
        from roc_mcs.gui.components.icon_provider import svg_icon
        from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR

        variant = int(self.SOURCE_ICON_VARIANT)
        if variant not in (1, 2, 3, 4):
            raise ValueError("SOURCE_ICON_VARIANT должен быть 1..4.")
        foreground = theme_colors(self._theme)["text"]
        icon_path = ICON_DIR / f"vtk_volume_source_{variant}.svg"
        self.source_button.setIcon(svg_icon(
            icon_path, color=foreground, widget=self.source_button, monochrome=True,
        ))

    def _rebuild_source_menu(self) -> None:
        """Mirror the hidden selection model into a concise icon-button menu."""
        self.source_menu.clear()
        self._source_menu_actions = {}
        for index in range(self.source_combo.count()):
            key = self.source_combo.itemData(index)
            label = self.source_combo.itemText(index)
            action = QtGui.QAction(self.source_combo.itemIcon(index), label, self.source_menu)
            action.setCheckable(True)
            action.setData(key)
            action.setChecked(index == self.source_combo.currentIndex())
            action.triggered.connect(
                lambda _checked=False, selected=key: self._select_source_from_menu(selected)
            )
            self.source_menu.addAction(action)
            self._source_menu_actions[key] = action
        self._sync_source_menu_state()

    def _sync_source_menu_state(self) -> None:
        selected = self.source_combo.currentData() or "full"
        labels = []
        for key, action in getattr(self, "_source_menu_actions", {}).items():
            active = key == selected
            action.setChecked(active)
            if active:
                labels.append(action.text())
        detail = labels[0] if labels else "Полный детектор"
        self.source_button.setToolTip(f"Источник 3D-объёма: {detail}")
        self.source_button.setAccessibleDescription(f"Текущий источник: {detail}")

    def _select_source_from_menu(self, key) -> None:
        index = self.source_combo.findData(key)
        if index < 0 or index == self.source_combo.currentIndex():
            self._sync_source_menu_state()
            return
        self.source_combo.setCurrentIndex(index)

    @staticmethod
    def _settings_panel(*rows):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(panel)
        layout.setContentsMargins(3, 2, 3, 2)
        layout.setSpacing(4)
        for label, widget in rows:
            layout.addRow(label, widget)
        return panel

    def _connect_tools(self):
        t = self.tools
        t.reset_view_requested.connect(self._reset_camera)
        t.axes_toggled.connect(lambda value: self._scene_call("set_axes_visible", value))
        t.bounding_box_toggled.connect(lambda value: self._scene_call("set_bounding_box_visible", value))
        t.axis_ticks_toggled.connect(lambda value: self._scene_call("set_axis_ticks_visible", value))
        t.grid_toggled.connect(lambda value: self._scene_call("set_grid_visible", value))
        t.perspective_view_requested.connect(self._perspective_view)
        t.camera_axis_direction_requested.connect(self._camera_axis)
        t.axis_scales_apply_requested.connect(self._set_axis_scales)
        t.theta_display_range_changed.connect(self._set_theta_range)
        t.theta_boundary_planes_toggled.connect(self._set_theta_boundary_planes)
        t.theta_boundary_plane_opacity_changed.connect(self._set_theta_boundary_opacity)
        t.cursor_line_toggled.connect(self._toggle_cursor)
        t.theta_slice_toggled.connect(self._toggle_theta)
        t.projection_toggled.connect(self._toggle_projection)
        t.overlay_toggled.connect(self._toggle_overlay)
        t.cursor_line_color_changed.connect(self._set_cursor_color)
        t.roi_preview_color_changed.connect(self._set_roi_color)

    def _scene_call(self, method, *args):
        scene = self._scene
        if scene is not None:
            getattr(scene, method)(*args)
            self._render()

    def set_dataset(self, raw, th, axis, x_coords, y_coords, saved_rois, *, scan_label="", view_label="", valid_mask=None, source_path=None, operation_id="source.raw", metadata=None):
        """Передать выбранную версию ROI Explorer; отдельный HDF5 не открывается."""
        scan_key = str(scan_label)
        source_identity = f"{Path(source_path).expanduser().resolve() if source_path else ''}|{scan_key}"
        if self._source_scan_key != source_identity:
            # ID ROI могут повторяться в разных сканах: чужую геометрию не переносим.
            self.source_combo.blockSignals(True)
            full_index = self.source_combo.findData("full")
            self.source_combo.setCurrentIndex(max(0, full_index))
            self.source_combo.blockSignals(False)
            self._source_scan_key = source_identity
        self._rois = []
        for item in (saved_rois or []):
            try:
                normalize_rectangular_roi(item.roi)
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            self._rois.append(item)
        self._rebuild_source_choices()
        details = dict(metadata or {})
        details.setdefault("view_label", str(view_label))
        self._dataset_request = {
            "raw": raw, "theta": th, "axis": axis, "x_coords": x_coords, "y_coords": y_coords,
            "valid_mask": valid_mask, "roi": self._selected_roi(), "scan_key": scan_key,
            "h5_path": source_path, "operation_id": str(operation_id), "metadata": details,
            "smoothing_sigma": float(self._settings.get("smooth_sigma", 0.5)),
        }
        self.source_button.setEnabled(True)
        self._refresh_theta_link_control()
        self._closed = False
        if self._scene is not None:
            self._submit_payload(self._dataset_request)
        else:
            self._schedule_vtk_init()

    def _rebuild_source_choices(self):
        selected = self.source_combo.currentData() or "full"
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("Полный детектор", "full")
        valid_ids = set()
        for item in self._rois:
            roi_id = int(item.id)
            valid_ids.add(roi_id)
            label = str(item.roi.label or f"ROI {roi_id}")
            self.source_combo.addItem(f"ROI {roi_id} · {label}", roi_id)
            index = self.source_combo.count() - 1
            color = QtGui.QColor(str(item.roi.color))
            if color.isValid():
                pix = QtGui.QPixmap(12, 12); pix.fill(color)
                self.source_combo.setItemData(index, QtGui.QIcon(pix), QtCore.Qt.ItemDataRole.DecorationRole)
        target = selected if selected == "full" or selected in valid_ids else "full"
        index = self.source_combo.findData(target)
        self.source_combo.setCurrentIndex(max(0, index))
        self.source_combo.setEnabled(bool(self._rois))
        self.source_combo.blockSignals(False)
        self._selected_roi_id = None if target == "full" else int(target)
        self._rebuild_source_menu()

    def _selected_roi(self):
        key = self.source_combo.currentData() or "full"
        if key == "full":
            return None
        item = next((value for value in self._rois if int(value.id) == int(key)), None)
        return item.roi.normalized() if item is not None else None

    def _source_changed(self, _index):
        self._sync_source_menu_state()
        if self._dataset_request is None:
            return
        self._selected_roi_id = None if self.source_combo.currentData() == "full" else int(self.source_combo.currentData())
        request = dict(self._dataset_request)
        request["roi"] = self._selected_roi()
        self._dataset_request = request
        if self._scene is not None:
            self._submit_payload(request)

    def _submit_payload(self, request):
        self._generation += 1
        token = self._generation
        # Не оставляем на экране предыдущий scan/ROI во время подготовки нового.
        # Иначе пользователь может принять старый объём за уже выбранный источник.
        self._payload = None
        if self._scene is not None:
            self._scene.clear(render=False)
            self.tools.set_scene_available(False)
            if self.display is not None:
                self.display.set_scene_available(False)
            self._render()
        if self._active_worker is not None:
            self._pending_request = (token, dict(request))
            self.status.setText("Подготавливается новая версия объёма…")
            return
        worker = _PayloadWorker(token, dict(request))
        worker.signals.finished.connect(self._payload_ready)
        self._active_worker = worker
        self._thread_pool.start(worker)
        self.status.setText("Подготовка данных для 3D…")

    @QtCore.pyqtSlot(int, object, object)
    def _payload_ready(self, token, payload, error):
        self._active_worker = None
        if int(token) == self._generation and not self._closed:
            if error is not None:
                self.status.setText(f"3D не построено: {type(error).__name__}: {error}")
            else:
                self._payload = payload
                try:
                    self._apply_payload(payload)
                except Exception as exc:
                    self._payload = None
                    # При частичной ошибке не оставляем неполный старый/новый объём.
                    if self._scene is not None:
                        try:
                            self._scene.clear(render=False)
                            self.tools.set_scene_available(False)
                            if self.display is not None:
                                self.display.set_scene_available(False)
                            self._render()
                        except Exception:
                            logger.exception("Не удалось очистить частично построенную VTK-сцену")
                    self.status.setText(f"Ошибка построения VTK-сцены: {type(exc).__name__}: {exc}")
        pending, self._pending_request = self._pending_request, None
        if pending is not None and pending[0] == self._generation and not self._closed:
            worker = _PayloadWorker(pending[0], pending[1])
            worker.signals.finished.connect(self._payload_ready)
            self._active_worker = worker
            self._thread_pool.start(worker)

    def _apply_payload(self, payload):
        if self._scene is None:
            self._payload = payload
            return
        settings = self._transfer_settings()
        info = self._scene.set_payload(payload, settings, axis_scales=self.tools.axis_scales)
        self.tools.set_scene_available(True)
        self.tools.set_theta_axis_values(payload.theta)
        self.tools.set_axis_scales(*self._scene.axis_scales)
        self.tools.set_overlay_controls_enabled(cursor=True, theta=len(payload.theta) > 2, projection=True, roi_preview=True)
        self.tools.set_projection_enabled(settings.show_projection)
        if self.display is not None:
            self.display.set_colormap(settings.palette)
            self.display.set_thresholds(self.low_threshold.value(), self.high_threshold.value(), absolute_low=info.get("threshold_min"), absolute_high=info.get("threshold_max"), reference=info.get("threshold_reference"))
            self.display.set_scene_available(True)
        blocked = self.theta_index.blockSignals(True)
        try:
            self.theta_index.setRange(0, max(0, len(payload.theta) - 1))
            self.theta_index.setValue(int(self._scene.theta_slice_index or 0))
        finally:
            self.theta_index.blockSignals(blocked)
        self._set_theta_index(int(self.theta_index.value()), notify_map=False)
        self._refresh_theta_link_control()
        if self._first_payload:
            self._scene.reset_camera_fit_preserve_direction()
            self._first_payload = False
        self._apply_cursor_position()
        if self._roi_preview is not None:
            self._update_roi_preview()
        self._render()
        self.status.setText(f"{payload.scan_key} · {payload.view_mode} · {payload.shape} · {payload.scalar_dtype} · invalid {payload.invalid_count:,} · {info.get('point_count', 0):,} узлов")

    def _load_vtk_modules(self):
        # Регистрация OpenGL backend обязательна ДО создания QVTK.
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        class _TrackedQVTK(QVTKRenderWindowInteractor):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._roc_finalized = False
                self._paint_count = 0

            def paintEvent(self, event):
                super().paintEvent(event)
                self._paint_count += 1

            def closeEvent(self, event):
                if not self._roc_finalized:
                    self._roc_finalized = True
                    super().closeEvent(event)  # Единственный вызов Finalize внутри QVTK.
                event.accept()

        return _TrackedQVTK

    def _schedule_vtk_init(self):
        if self._scene is not None or self._closed or self._dataset_request is None:
            return
        if not self.isVisible() or not self.window().isVisible() or self._init_pending:
            return
        self._init_pending = True
        self._init_generation += 1
        ticket = self._init_generation
        if self._qvtk is not None:
            QtCore.QTimer.singleShot(0, lambda t=ticket: self._initialize_vtk(t))
        else:
            QtCore.QTimer.singleShot(80, lambda t=ticket: self._ensure_vtk(t))

    def showEvent(self, event):
        active_theme = resolve_theme(self.parentWidget() or self)
        if active_theme != self._theme:
            self.apply_theme(active_theme)
        super().showEvent(event)
        self._schedule_vtk_init()

    def _ensure_vtk(self, ticket: int):
        if ticket != self._init_generation:
            return
        if self._scene is not None or self._closed or not self._init_pending:
            self._init_pending = False
            return
        if not self.isVisible() or not self.window().isVisible() or self._dataset_request is None:
            self._init_pending = False
            return
        if self._qvtk is not None:
            self._initialize_vtk(ticket)
            return
        try:
            QVTKRenderWindowInteractor = self._load_vtk_modules()
            widget = QVTKRenderWindowInteractor(self.viewport)
            self._qvtk = widget
            widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontCreateNativeAncestors, False)
            self.viewport_layout.removeWidget(self.placeholder)
            self.placeholder.hide()
            self.viewport_layout.addWidget(widget, 1)
            widget.show()
            widget.winId()
            QtCore.QTimer.singleShot(100, lambda t=ticket: self._initialize_vtk(t))
        except Exception as exc:
            self._init_pending = False
            request = self._dataset_request
            self.release_vtk_scene()
            self._dataset_request = request
            message = f"VTK недоступен: {type(exc).__name__}: {exc}"
            self.status.setText(message)
            self.placeholder.setText("Не удалось инициализировать VTK. Проверьте PyQt6, VTK OpenGL и драйвер GPU.")
            self.placeholder.show()

    def _initialize_vtk(self, ticket: int):
        if ticket != self._init_generation:
            return
        widget = self._qvtk
        if widget is None or self._closed or not self._init_pending or self._scene is not None:
            return
        if not self.isVisible() or not self.window().isVisible():
            self._init_pending = False
            return
        try:
            # Обновляем тему до первого paint QVTK/HUD, а не после загрузки объёма.
            self.apply_theme(resolve_theme(self.parentWidget() or self))
            render_window = widget.GetRenderWindow()
            if not bool(render_window.IsA("vtkOpenGLRenderWindow")):
                raise RuntimeError(f"Ожидался OpenGL RenderWindow, получен {render_window.GetClassName()}.")
            render_window.SetMultiSamples(0)
            interactor = render_window.GetInteractor()
            if interactor is None:
                raise RuntimeError("QVTK не создал vtkRenderWindowInteractor.")
            if not bool(interactor.GetInitialized()):
                # Создаём native surface и просим Qt выполнить первый paint до Initialize.
                if getattr(widget, "_paint_count", 0) < 1:
                    widget.repaint()
                widget.Initialize()
            if not bool(interactor.GetInitialized()):
                raise RuntimeError("VTK interactor не перешёл в состояние Initialized.")
            self._render_window = render_window
            self._interactor = interactor
            self._scene = VTKVolumeScene(render_window, interactor, render_callback=widget.Render)
            self._scene.set_theme(self._theme)
            self.tools.apply_theme(self._theme)
            self.tools.set_scene_available(False)
            if self.display is None:
                # HUD должен быть дочерним QVTK native widget: соседний QWidget
                # может оказаться под native OpenGL surface и стать невидимым.
                self.display = VTKVolumeDisplayWidget(widget)
                self.display.attach_settings_panel(self.display_panel)
                self.display.bind_threshold_spins(self.low_threshold, self.high_threshold)
                self.display.thresholds_changed.connect(self._queue_transfer_settings)
            else:
                self.display.set_viewport(widget)
            self.display.set_colormap(self._transfer.palette)
            self.display.apply_theme(self._theme)
            self.display.set_colorbar_visible(bool(self._settings.get("show_colorbar", True)))
            self.display.set_scene_available(False)
            self._apply_cached_scene_controls()
            if self._payload is not None:
                self._apply_payload(self._payload)
            elif self._dataset_request is not None:
                self._submit_payload(self._dataset_request)
            else:
                self.status.setText("Выберите HDF5 и скан в ROI Explorer, затем откройте 3D.")
            self._init_pending = False
        except Exception as exc:
            request = self._dataset_request
            self.release_vtk_scene()
            self._dataset_request = request
            message = f"Ошибка запуска VTK: {type(exc).__name__}: {exc}"
            self.placeholder.setText(message)
            self.placeholder.show()
            self.status.setText(message)
            self._init_pending = False

    def _apply_cached_scene_controls(self):
        if self._scene is None:
            return
        self._scene.set_axes_visible(True)
        self._scene.set_grid_visible(bool(self._settings.get("show_grid", True)), render=False)

    def _transfer_settings(self):
        return replace(
            self._transfer,
            lower_fraction=float(self.low_threshold.value()) / 100.0,
            upper_fraction=float(self.high_threshold.value()) / 100.0,
            max_opacity=float(self.opacity.value()),
            gamma=float(self.gamma.value()),
            palette=str(self.palette.currentData() or "magma"),
            sample_distance_factor=float(self.sample_factor.value()),
            opacity_unit_distance_factor=float(self.unit_distance.value()),
            show_projection=self.tools.projection_button.isChecked(),
            projection_opacity=float(self.projection_opacity.value()),
            projection_palette=self.projection_palette.currentData(),
        )

    def _queue_transfer_settings(self, *_args):
        if self._scene is not None:
            self._settings_timer.start()

    def _apply_transfer_settings(self):
        if self._scene is None or self._payload is None:
            return
        try:
            self._transfer = self._transfer_settings()
            info = self._scene.update_transfer_functions(self._transfer, render=False)
            self.display.set_colormap(self._transfer.palette)
            self.display.set_thresholds(self.low_threshold.value(), self.high_threshold.value(), absolute_low=info.get("threshold_min"), absolute_high=info.get("threshold_max"), reference=info.get("threshold_reference"))
            self._render()
        except Exception as exc:
            self.status.setText(f"Настройки отображения не применены: {exc}")

    def apply_display_settings(self, settings=None):
        if settings is None:
            return
        values = dict(settings)
        old_sigma = float(self._settings.get("smooth_sigma", 0.5))
        new_sigma = float(values.get("smooth_sigma", 0.5))
        self._settings = values
        self.low_threshold.setValue(100.0 * float(values.get("isomin_fraction", 0.16)))
        self.high_threshold.setValue(100.0 * float(values.get("isomax_fraction", 0.95)))
        self.opacity.setValue(float(values.get("volume_opacity", 0.18)))
        self.gamma.setValue(2.0 if values.get("volume_opacity_mode", "uniform") == "soft" else 1.0)
        self.sample_factor.setValue({"fast": 1.75, "balanced": 1.0, "full": 0.65, "auto": 1.0}.get(str(values.get("quality", "auto")), 1.0))
        self.unit_distance.setValue(1.0)
        idx = self.palette.findData(str(values.get("volume_colormap", "magma")))
        if idx >= 0:
            self.palette.setCurrentIndex(idx)
        self.projection_opacity.setValue(float(values.get("projection_opacity", 0.75)))
        pidx = self.projection_palette.findData(values.get("projection_colorscale"))
        if pidx >= 0:
            self.projection_palette.setCurrentIndex(pidx)
        self.tools.set_grid_visible(bool(values.get("show_grid", True)))
        self.tools.set_projection_enabled(bool(values.get("show_projection", True)))
        if self.display is not None:
            self.display.set_colorbar_visible(bool(values.get("show_colorbar", True)))
        if self._scene is not None:
            self._scene.set_theme(self._theme)  # применить изменения шрифта осей
            self._scene.set_perspective(str(values.get("projection_type", "perspective")) == "perspective")
            self._apply_transfer_settings()
        if self._dataset_request is not None and not np.isclose(old_sigma, new_sigma):
            request = dict(self._dataset_request)
            request["smoothing_sigma"] = new_sigma
            self._dataset_request = request
            # Explorer решает, когда можно перестраивать данные: скрытая 3D-панель
            # только помечается устаревшей, без фонового выделения больших массивов.
            self.data_rebuild_requested.emit()

    def apply_theme(self, theme=None):
        active = str(theme or resolve_theme(self)).lower()
        self._theme = active if active in ("light", "dark") else "light"
        self.setProperty("rocMcsTheme", self._theme)
        colors = theme_colors(self._theme)
        self.setStyleSheet(f"QWidget#EmbeddedVTKVolumeWidget{{background:{colors['window']};color:{colors['text']};}} QWidget#EmbeddedVTKToolbar{{background:{colors['window']};}} QLabel#vtkEmbeddedStatus{{color:{colors['muted']};font-size:9px;padding:1px 4px;}}")
        if hasattr(self, "tools"):
            self.tools.apply_theme(self._theme)
        if hasattr(self, "source_button"):
            self._update_source_icon()
        if self._scene is not None:
            self._scene.set_theme(self._theme)
            self._render()
        if self.display is not None:
            self.display.apply_theme(self._theme)

    def set_map_cursor_position(self, position):
        self._map_cursor_position = None if position is None else (float(position[0]), float(position[1]))
        self._apply_cursor_position()

    def _apply_cursor_position(self):
        if self._scene is None or self._payload is None or not self.tools.cursor_line_button.isChecked():
            return
        if self._map_cursor_position is None:
            self._scene.set_cursor_line_position(None, None, render=False)
        else:
            target = map_detector_cursor_to_volume(self._payload, *self._map_cursor_position, line_width_voxels=float(self.cursor_width.value()))
            if target is None:
                self._scene.set_cursor_line_position(None, None, render=False)
            else:
                self._scene.set_cursor_line_position(target.volume_x, target.volume_y, target.radius_x, target.radius_y, render=False)
        self._render()

    def _toggle_cursor(self, enabled):
        if self._scene is None or self._payload is None:
            self.tools.set_cursor_line_enabled(False)
            return
        self._scene.set_cursor_line_color(self.tools.cursor_line_color, render=False)
        self._scene.set_cursor_line_density(float(self.cursor_density.value()), render=False)
        self._scene.set_cursor_line_enabled(bool(enabled), render=False)
        self._apply_cursor_position()
        self._render()

    def _update_cursor(self, *_args):
        self._apply_cursor_position()

    def _update_cursor_density(self, value):
        if self._scene is not None:
            self._scene.set_cursor_line_density(float(value), render=False)
            self._render()

    def _set_cursor_color(self, color):
        if self._scene is not None:
            self._scene.set_cursor_line_color(str(color), render=False)
            self._render()

    def _toggle_theta(self, enabled):
        if self._scene is not None and self._payload is not None:
            self._scene.set_theta_slice_enabled(bool(enabled), render=False)
            self._render()

    def _apply_theta_index(self, index):
        self._set_theta_index(int(index), notify_map=True)

    def _set_theta_index(self, index: int, *, notify_map: bool) -> None:
        if self._scene is not None and self._payload is not None:
            value = self._scene.set_theta_slice_index(int(index), render=False)
            self.theta_label.setText(f"θ = {float(self._payload.theta[value]):.7g}° [{value + 1}/{len(self._payload.theta)}]")
            self._render()
            if notify_map and self.is_theta_link_active:
                self.theta_slice_value_changed.emit(float(self._payload.theta[value]))

    @property
    def is_theta_link_active(self) -> bool:
        """Whether bidirectional 2D↔3D theta synchronization is effective now."""
        return bool(
            self._map_theta_mode
            and self.theta_link_checkbox.isEnabled()
            and self.theta_link_checkbox.isChecked()
        )

    def set_map_theta_context(self, enabled: bool, theta_values=None, current_index: int = 0) -> None:
        """Publish the current 2D theta-map state to the embedded volume view."""
        axis = np.asarray(theta_values if theta_values is not None else [], dtype=np.float64)
        if axis.ndim != 1 or not np.isfinite(axis).all():
            axis = np.empty(0, dtype=np.float64)
            enabled = False
        if axis.size > 1:
            differences = np.diff(axis)
            if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
                axis = np.empty(0, dtype=np.float64)
                enabled = False
        self._map_theta_mode = bool(enabled and axis.size > 1)
        self._map_theta_values = axis.copy()
        self.set_map_theta_index(current_index)
        self._refresh_theta_link_control()

    def set_map_theta_index(self, index: int) -> None:
        """Update the 2D map's current index without rebuilding its theta axis."""
        maximum = max(0, int(self._map_theta_values.size) - 1)
        self._map_theta_index = max(0, min(int(index), maximum))

    def _volume_theta_axis(self):
        if self._payload is not None:
            return np.asarray(self._payload.theta, dtype=np.float64)
        if self._dataset_request is not None:
            try:
                axis = np.asarray(self._dataset_request.get("theta"), dtype=np.float64)
                if axis.ndim == 1 and np.isfinite(axis).all():
                    return axis
            except (TypeError, ValueError):
                pass
        return np.empty(0, dtype=np.float64)

    def _refresh_theta_link_control(self) -> None:
        volume_axis = self._volume_theta_axis()
        map_axis = self._map_theta_values
        compatible = (
            self._map_theta_mode
            and map_axis.size > 1
            and volume_axis.size == map_axis.size
            and volume_axis.size > 1
            and np.allclose(
                np.sort(volume_axis), np.sort(map_axis),
                rtol=1e-10, atol=1e-12,
            )
        )
        self.theta_link_checkbox.setEnabled(bool(compatible))
        if compatible:
            self.theta_link_checkbox.setToolTip(
                "Связать физическое положение θ-среза в 2D-карте и 3D. "
                "Изменение любого ползунка обновляет второй."
            )
            if self.is_theta_link_active and self._payload is not None:
                map_value = float(self._map_theta_values[self._map_theta_index])
                self.set_theta_index_from_map(map_value)
        elif self._map_theta_mode:
            self.theta_link_checkbox.setToolTip(
                "Синхронизация недоступна: физические θ-оси 2D-карты и 3D-источника не совпадают."
            )
        else:
            self.theta_link_checkbox.setToolTip(
                "Доступно только в режиме 2D-карты «Срез по θ». В остальных режимах 3D-срез независимый."
            )

    def _on_theta_link_toggled(self, checked: bool) -> None:
        if checked and self.is_theta_link_active and self._map_theta_values.size:
            map_value = float(self._map_theta_values[self._map_theta_index])
            self.set_theta_index_from_map(map_value)

    def set_theta_index_from_map(self, theta_value: float) -> None:
        """Synchronize the 3D slider from a physical 2D theta value, no echo."""
        if not self.is_theta_link_active or self._payload is None:
            return
        try:
            value = float(theta_value)
        except (TypeError, ValueError, OverflowError):
            return
        if not np.isfinite(value) or not self._payload.theta.size:
            return
        axis = np.asarray(self._payload.theta, dtype=np.float64)
        index = int(np.argmin(np.abs(axis - value)))
        slider = self.theta_index
        blocked = slider.blockSignals(True)
        try:
            slider.setValue(index)
        finally:
            slider.blockSignals(blocked)
        self._set_theta_index(index, notify_map=False)

    def _apply_theta_palette(self, _index):
        if self._scene is not None:
            self._scene.set_theta_slice_palette(str(self.theta_palette.currentData()), render=False)
            self._render()

    def _apply_theta_strength(self, value):
        if self._scene is not None:
            self._scene.set_theta_slice_strength(float(value), render=False)
            self._render()

    def _toggle_projection(self, enabled):
        self._transfer = replace(self._transfer, show_projection=bool(enabled))
        self._queue_transfer_settings()

    def _toggle_overlay(self, key, enabled):
        if key == "roi_preview" and self._scene is not None:
            self._scene.set_roi_preview_visible(bool(enabled), render=False)
            self._render()

    def _set_roi_color(self, color):
        if self._scene is not None:
            self._scene.set_roi_preview_color(str(color), render=False)
            self._render()

    def set_roi_preview_geometry(self, roi):
        self._roi_preview = None if roi is None else (roi.copy() if callable(getattr(roi, "copy", None)) else roi)
        self._update_roi_preview()

    def _update_roi_preview(self, *_args):
        """Coalesce selector moves; never render VTK inside a Matplotlib callback."""
        if self._scene is None or self._payload is None or self._closed:
            return
        if not self._roi_preview_timer.isActive():
            self._roi_preview_timer.start()

    def _flush_roi_preview(self):
        scene = self._scene
        if scene is None or self._payload is None or self._closed:
            return
        try:
            if self._roi_preview is None:
                scene.set_roi_preview_visible(False, render=False)
            else:
                mode = str(self.roi_preview_mode.currentData() or "inside")
                scene.set_roi_preview(
                    self._roi_preview, show_fill=True, show_outline=True,
                    opacity=float(self.roi_preview_opacity.value()),
                    highlight_mode=mode, highlight_strength=float(self.roi_preview_strength.value()),
                    highlight_color=self.tools.roi_preview_color, render=False,
                )
                scene.set_roi_preview_visible(self.tools.roi_preview_button.isChecked(), render=False)
            self._render()
        except Exception as exc:
            # Исключения из Qt slot не должны покидать event dispatcher.
            logger.exception("Не удалось обновить предпросмотр ROI в VTK")
            try:
                scene.set_roi_preview_visible(False, render=False)
                self._render()
            except Exception:
                logger.exception("Не удалось скрыть некорректный ROI preview")
            self.status.setText(f"Предпросмотр ROI не обновлён: {type(exc).__name__}: {exc}")

    def _camera_axis(self, axis, polarity):
        if self._scene is not None:
            self._scene.reset_camera_axis(str(axis), polarity=int(polarity))
            self.tools.set_perspective_enabled(self._scene.perspective_enabled)
            self._render()

    def _perspective_view(self):
        if self._scene is not None:
            self._scene.set_perspective_view()
            self.tools.set_perspective_enabled(self._scene.perspective_enabled)
            self._render()

    def _reset_camera(self):
        if self._scene is not None:
            self._scene.reset_camera_fit_preserve_direction()
            self._render()

    def _set_axis_scales(self, x, y, theta):
        if self._scene is not None:
            self._scene.set_axis_scales(float(x), float(y), float(theta), render=False)
            self.tools.set_axis_scales(float(x), float(y), float(theta))
            self._render()

    def _set_theta_range(self, low, high):
        if self._scene is not None:
            self._scene.set_theta_display_range_indices(int(low), int(high), render=False)
            self._render()

    def _set_theta_boundary_planes(self, enabled):
        if self._scene is not None:
            self._scene.set_theta_boundary_planes_visible(bool(enabled), render=False)
            self._render()

    def _set_theta_boundary_opacity(self, value):
        if self._scene is not None:
            self._scene.set_theta_boundary_plane_opacity(float(value), render=False)
            self._render()

    def update_roi_color(self, roi_id, color):
        for item in self._rois:
            if int(item.id) == int(roi_id):
                item.roi.color = str(color)
                self._rebuild_source_choices()
                break

    def clear_dataset(self, message="Выберите HDF5 и скан для построения 3D"):
        self._roi_preview_timer.stop()
        self._generation += 1
        self._init_pending = False
        self._pending_request = None
        self._dataset_request = None
        self._payload = None
        self._rois.clear()
        self._source_scan_key = None
        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentIndex(max(0, self.source_combo.findData("full")))
        self.source_combo.blockSignals(False)
        self._rebuild_source_choices()
        self.source_button.setEnabled(False)
        if self._scene is not None:
            self._scene.clear(render=False)
            self._render()
        if self.display is not None:
            self.display.set_scene_available(False)
        self.tools.set_scene_available(False)
        self.status.setText(str(message))

    def release_vtk_scene(self):
        """Закрыть VTK/OpenGL полностью; обычное hide/minimize этот метод не вызывает."""
        self._closed = True
        self._roi_preview_timer.stop()
        self._settings_timer.stop()
        self._init_pending = False
        self._init_generation += 1
        self._generation += 1
        self._pending_request = None
        self._dataset_request = None
        self._payload = None
        self._roi_preview = None
        scene, self._scene = self._scene, None
        cleanup_errors = []
        if scene is not None:
            try:
                scene.close()
            except Exception as exc:
                cleanup_errors.append(f"VTK resources: {type(exc).__name__}: {exc}")
                logger.exception("Не удалось полностью освободить ресурсы VTK-сцены")
        # HUD and its settings panel are ordinary Qt controls, not VTK/GPU
        # resources. Keep them alive so closing/reopening 3D does not leave a
        # deleted QWidget wrapper in display_panel or duplicate signal bindings.
        if self.display is not None:
            try:
                self.display.set_scene_available(False)
                # HUD/settings persist across scene teardown, but must not be
                # destroyed as a child of the QVTK widget being finalized.
                self.display.set_viewport(self.viewport)
            except (RuntimeError, AttributeError) as exc:
                cleanup_errors.append(f"display HUD detach: {exc}")
        widget, self._qvtk = self._qvtk, None
        self._render_window = None
        self._interactor = None
        if widget is not None:
            try:
                widget.setUpdatesEnabled(False)
                widget.close()  # QVTK closeEvent performs Finalize exactly once.
            except Exception as exc:
                cleanup_errors.append(f"QVTK Finalize: {type(exc).__name__}: {exc}")
                logger.exception("Ошибка Finalize для QVTK render window")
            try:
                self.viewport_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
            except RuntimeError as exc:
                cleanup_errors.append(f"QVTK widget detach: {exc}")
        self.placeholder.setText("3D-сцена закрыта. Нажмите кнопку 3D, чтобы создать её заново.")
        if self.placeholder.parent() is None:
            self.placeholder.setParent(self.viewport)
        self.viewport_layout.addWidget(self.placeholder)
        self.placeholder.show()
        self.tools.set_scene_available(False)
        self.tools.set_overlay_controls_enabled(cursor=False, theta=False, projection=False, roi_preview=False)
        if cleanup_errors:
            self.status.setText("3D закрыто; предупреждение очистки: " + " | ".join(cleanup_errors))
        else:
            self.status.setText("VTK/OpenGL-сцена закрыта; ресурсы рендерера освобождены.")
        self._closed = False
        self._first_payload = True

    def _render(self):
        if self._qvtk is not None and self._scene is not None:
            self._qvtk.Render()

    def closeEvent(self, event):
        self.release_vtk_scene()
        super().closeEvent(event)
