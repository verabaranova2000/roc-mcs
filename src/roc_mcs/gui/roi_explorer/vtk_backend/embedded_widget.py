"""Ленивый VTK-рендерер ROI Explorer; источник данных задаёт само приложение."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import logging
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from roc_mcs.gui.roi_explorer.theme import resolve_theme, theme_colors
from roc_mcs.analysis.project import project_directory_for_scan
from roc_mcs.gui.app_settings import resolve_analysis_projects_directory
from roc_mcs.gui.components.floating_legend import FloatingLegendPanel, LegendEntry
from roc_mcs.gui.roi_explorer.map_tools_widget import SmartDropdownButton
from roc_mcs.gui.roi_explorer.theta_range_preview import THETA_RANGE_PREVIEW_INTERVAL_MS
from roc_mcs.gui.roi_explorer.plot_toolbar_style import (
    SMART_DROPDOWN_INDICATOR_RGBA,
    SMART_DROPDOWN_INDICATOR_SIZE,
)
from .cursor_link import map_detector_cursor_to_volume
from .scan_view_adapter import build_volume_payload
from roc_mcs.roi.roi_types import normalize_rectangular_roi
from .vtk_scene import TransferSettings, VTKVolumeScene, _rebase_vtk_mouse_move
from .vtk_tools_widget import VTKToolsWidget
from .vtk_volume_display_widget import VTKVolumeDisplayWidget
from .scan_view_state import load_view_state, save_view_state


logger = logging.getLogger(__name__)

# Keep the VTK console output sink alive for the process lifetime.  VTK uses a
# process-wide singleton; retaining the Python wrapper prevents it from being
# collected while native VTK objects still hold the instance.
_VTK_OUTPUT_WINDOW_KEEPALIVE = None


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


class _SourceDropdownButton(SmartDropdownButton):
    """Smart dropdown with only the scientific corner marker, no native chevron.

    QToolButton's platform style paints its own menu indicator even when a QSS
    rule hides it on some native styles.  Paint the normal toolbutton control
    with HasMenu removed, then draw the same lower-right marker as
    SmartDropdownButton.  The attached QMenu and InstantPopup interaction are
    unchanged.
    """

    def paintEvent(self, event):
        option = QtWidgets.QStyleOptionToolButton()
        self.initStyleOption(option)
        option.features &= ~QtWidgets.QStyleOptionToolButton.ToolButtonFeature.HasMenu
        option.subControls &= ~QtWidgets.QStyle.SubControl.SC_ToolButtonMenu
        painter = QtWidgets.QStylePainter(self)
        if not painter.isActive():
            return
        painter.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_ToolButton, option)
        painter.end()

        marker = QtGui.QPainter(self)
        if not marker.isActive():
            return
        marker.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        marker.setPen(QtCore.Qt.PenStyle.NoPen)
        marker.setBrush(QtGui.QColor(*SMART_DROPDOWN_INDICATOR_RGBA))
        rect = self.rect()
        size = SMART_DROPDOWN_INDICATOR_SIZE
        marker.drawPolygon(QtGui.QPolygon([
            QtCore.QPoint(rect.right() - size, rect.bottom()),
            QtCore.QPoint(rect.right(), rect.bottom() - size),
            QtCore.QPoint(rect.right(), rect.bottom()),
        ]))
        marker.end()


class EmbeddedVTKVolumeWidget(QtWidgets.QWidget):
    """Лёгкая оболочка; QVTK/OpenGL создаются только при первом показе окна."""

    data_rebuild_requested = QtCore.pyqtSignal()
    # Physical theta coordinate (degrees) changed by the user in the 3D HUD.
    # The explorer maps this value to its own 2D theta axis; indices are not
    # assumed to match because either source may be stored in descending order.
    theta_slice_value_changed = QtCore.pyqtSignal(float)
    # Emitted after the VTK toolbar receives a new physical theta axis.
    # The explorer uses it to re-evaluate 2D/3D range-link compatibility.
    theta_range_axis_changed = QtCore.pyqtSignal(object)

    # Choose one of the four supplied scientific source symbols (1..4).
    # 1 is the default nested-volume/cube glyph.
    SOURCE_ICON_VARIANT = 1
    # Small outline variants are provided in resources/icons/vtk_theta_link_*.svg.
    THETA_LINK_ICON_VARIANT = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        # The host application can use a pixel-sized QApplication font, for
        # which QFont.pointSizeF() is -1.  Give this pane a valid inherited
        # point size before creating child controls, but preserve the host
        # family/weight/style and its existing point size when it is valid.
        inherited_font = QtGui.QFont(QtWidgets.QApplication.font())
        inherited_point_size = float(inherited_font.pointSizeF())
        if not np.isfinite(inherited_point_size) or inherited_point_size <= 0.0:
            inherited_font.setPointSizeF(9.0)
        self.setFont(inherited_font)
        self.setObjectName("EmbeddedVTKVolumeWidget")
        self._theme = resolve_theme(self)
        self.setProperty("rocMcsTheme", self._theme)
        self._qvtk = None
        self._render_window = None
        self._interactor = None
        self._scene = None
        self._display = None
        self._peak_hover_observer_id = None
        self._last_hover_peak_key = None
        self._last_hover_pick_time = 0.0
        self._payload = None
        self._peak_catalog = None
        self._peak_position_kind = "detected"
        self._branch_edges = None
        self._branch_display_colors: dict[int, str] = {}
        self._branch_display_selected_ids: tuple[int, ...] = ()
        self._branch_display_focus_selected = False
        self._show_excluded_branches = False
        # Immutable, renderer-neutral view model built from the stored
        # correlation artifact. It is deliberately excluded from the volume
        # worker request and never triggers a second scientific calculation.
        self._branch_correlation_preview = None
        # None means "auto-show when data first become available".  Once the
        # user explicitly toggles a layer, preserve that choice on payload refresh.
        self._peak_overlay_preference: bool | None = None
        self._branch_overlay_preference: bool | None = None
        self._branch_correspondence_overlay_preference: bool | None = None
        self._dataset_request: dict | None = None
        self._active_worker = None
        self._pending_request: tuple[int, dict] | None = None
        self._generation = 0
        self._closed = False
        self._permanently_closed = False
        self._release_in_progress = False
        self._final_shutdown_pending = False
        self._vtk_init_attempts = 0
        self._init_pending = False
        self._init_generation = 0
        self._first_payload = True
        self._map_cursor_position = None
        self._roi_preview = None
        self._rois: list = []
        self._selected_roi_id: int | None = None
        self._source_scan_key: str | None = None
        # Display-only state belongs to a scan/project, not to the application
        # preferences or the immutable scientific workflow manifest.
        self._view_state_identity: str | None = None
        self._view_state_path: Path | None = None
        self._pending_view_state: dict | None = None
        self._pending_view_camera_state: dict | None = None
        self._default_view_controls: dict | None = None
        self._view_state_interaction_observer_id = None
        self._view_state_timer = QtCore.QTimer(self)
        self._view_state_timer.setSingleShot(True)
        self._view_state_timer.setInterval(300)
        self._view_state_timer.timeout.connect(self._save_scan_view_state)
        self._map_theta_mode = False
        self._map_theta_values = np.empty(0, dtype=np.float64)
        self._map_theta_index = 0
        # Remember the user's link preference while theta-map mode is unavailable.
        self._theta_link_preference = True
        self._transfer = TransferSettings()
        self._settings = {}
        self._settings_timer = QtCore.QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(80)
        self._settings_timer.timeout.connect(self._apply_transfer_settings)
        # Dedicated low-latency path for the colorbar drag.  Other staged
        # display controls retain their existing settings-apply debounce.
        self._volume_contrast_timer = QtCore.QTimer(self)
        self._volume_contrast_timer.setSingleShot(True)
        self._volume_contrast_timer.setInterval(25)
        self._volume_contrast_timer.timeout.connect(self._apply_pending_volume_contrast)
        self._pending_volume_contrast: tuple[float, float] | None = None
        self._roi_preview_timer = QtCore.QTimer(self)
        self._roi_preview_timer.setSingleShot(True)
        self._roi_preview_timer.setInterval(16)
        self._roi_preview_timer.timeout.connect(self._flush_roi_preview)
        # Coalesce the high-frequency θ-slider events. The 3D mapper only
        # needs the newest discrete interval; applying every intermediate
        # mouse event needlessly rebuilds tiny VTK props and queues renders.
        self._theta_range_timer = QtCore.QTimer(self)
        self._theta_range_timer.setSingleShot(True)
        self._theta_range_timer.setInterval(THETA_RANGE_PREVIEW_INTERVAL_MS)
        self._theta_range_timer.timeout.connect(self._apply_pending_theta_range)
        self._pending_theta_range: tuple[int, int] | None = None
        self._thread_pool = QtCore.QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        self.toolbar = QtWidgets.QWidget(self)
        self.toolbar.setObjectName("EmbeddedVTKToolbar")
        row = QtWidgets.QHBoxLayout(self.toolbar)
        row.setContentsMargins(3, 1, 3, 1)
        row.setSpacing(0)
        # Keep the combo as an internal selection model for API compatibility,
        # but expose the source choice through one compact icon-only menu.
        self.source_combo = QtWidgets.QComboBox(self.toolbar)
        self.source_combo.setObjectName("vtkExplorerVolumeSource")
        self.source_combo.addItem("Полный детектор", "full")
        self.source_combo.hide()
        self.source_button = _SourceDropdownButton(self.toolbar)
        self.source_button.setObjectName("vtkExplorerSourceButton")
        self.source_button.setAutoRaise(True)
        self.source_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        # SmartDropdownButton paints the custom lower-right triangle itself.
        # Suppress Qt's second native menu chevron (the unwanted "v").
        self.source_button.setStyleSheet(
            "QToolButton::menu-indicator { image: none; width: 0px; height: 0px; }"
        )
        self.source_button.setFixedSize(28, 28)
        self.source_button.setIconSize(QtCore.QSize(19, 19))
        self.source_button.setToolTip("Выбрать источник 3D-объёма")
        self.source_button.setAccessibleName("Источник 3D-объёма")
        self.source_menu = QtWidgets.QMenu(self.source_button)
        self.source_menu.setObjectName("vtkExplorerSourceMenu")
        self.source_menu.setToolTipsVisible(True)
        # The complete palette-aware menu stylesheet is reapplied in
        # apply_theme(); never append to an existing stylesheet on theme
        # changes, otherwise identical rules accumulate indefinitely.
        self.source_button.setMenu(self.source_menu)
        row.addWidget(self.source_button, 0)
        self.tools = VTKToolsWidget(self.toolbar)
        # The first VTK tool starts directly beside the source dropdown; remove
        # the nested toolbar's default left inset for this embedded layout only.
        tools_layout = self.tools.layout()
        if tools_layout is not None:
            tools_layout.setContentsMargins(0, 2, 5, 2)
        self.tools.volume_source_btn.hide()
        self.tools.theme_btn.hide()
        row.addWidget(self.tools, 1)
        root.addWidget(self.toolbar, 0)

        self.viewport = QtWidgets.QWidget(self)
        self.viewport.setObjectName("EmbeddedVTKViewport")
        # A grid permits QVTK and the HUD to occupy the same viewport cell.
        # QVTK fills it; the fixed-size native HUD is aligned top-right by the
        # layout, so it is never clipped or left at QWidget's default height.
        self.viewport_layout = QtWidgets.QGridLayout(self.viewport)
        self.viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.viewport_layout.setSpacing(0)
        self.viewport_layout.setRowStretch(0, 1)
        self.viewport_layout.setColumnStretch(0, 1)
        self.placeholder = QtWidgets.QLabel("3D-сцена будет создана при первом открытии окна.", self.viewport)
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.viewport_layout.addWidget(self.placeholder, 0, 0)
        self.branch_legend = FloatingLegendPanel(
            self.viewport,
            title="BRANCH ID · ЦВЕТА",
            default_size=QtCore.QSize(224, 248),
            margin=8,
        )
        self.branch_legend.entry_toggled.connect(self._on_branch_legend_entry_toggled)
        self.branch_legend.hide_on_canvas()
        self.branch_correlation_legend = FloatingLegendPanel(
            self.viewport,
            title="3D-КОРРЕЛЯЦИЯ · СООТВЕТСТВИЯ",
            default_size=QtCore.QSize(292, 208),
            margin=8,
        )
        self.branch_correlation_legend.hide_on_canvas()
        root.addWidget(self.viewport, 1)

        self.status = QtWidgets.QLabel("3D не инициализировано", self)
        self.status.setObjectName("vtkEmbeddedStatus")
        self.status.setMinimumHeight(18)
        self.status.setWordWrap(True)
        root.addWidget(self.status, 0)

        self.apply_global_icon_size()
        self._build_overlay_settings()
        self._connect_tools()
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self._update_source_icon()
        self._rebuild_source_menu()
        self.source_button.setEnabled(False)
        self.tools.set_scene_available(False)
        self.tools.set_volume_source_available(False)
        self.tools.set_overlay_controls_enabled(
            cursor=False, theta=False, projection=False, roi_preview=False, peaks=False
        )
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

    def _spin(self, minimum, maximum, value, decimals=2, suffix="", step=None):
        """Create a compact floating-point control with a precision-safe wheel step.

        QDoubleSpinBox defaults to a 1.0 wheel step even when it displays two
        decimal places.  That made values such as opacity=0.18 jump directly to
        1.00/0.00.  Unless a control supplies a domain-specific step, use its
        smallest displayed decimal place.
        """
        widget = QtWidgets.QDoubleSpinBox()
        precision = max(0, int(decimals))
        widget.setRange(float(minimum), float(maximum))
        widget.setDecimals(precision)
        wheel_step = float(step) if step is not None else (10.0 ** -precision if precision else 1.0)
        widget.setSingleStep(max(wheel_step, 10.0 ** -precision if precision else 1.0))
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
        # Match the independently tested standalone VTK viewer defaults.
        for name in ("turbo", "viridis", "inferno", "magma", "plasma", "cividis", "gray"):
            self.theta_palette.addItem(name, name)
        turbo_index = self.theta_palette.findData("turbo")
        if turbo_index >= 0:
            self.theta_palette.setCurrentIndex(turbo_index)
        self.theta_strength = self._spin(0.0, 1.0, 0.90, 2, step=0.05)
        theta = self._settings_panel(
            ("Срез", self.theta_label),
            ("Индекс θ", self.theta_index),
            ("Палитра", self.theta_palette),
            ("Подсветка", self.theta_strength),
        )

        # A compact icon + caption replaces the oversized native checkbox. The
        # parent row is disabled as a whole when physical 2D/3D theta axes cannot
        # be linked, so neither the icon nor its label suggests an available action.
        self.theta_link_row = QtWidgets.QWidget(theta)
        link_layout = QtWidgets.QHBoxLayout(self.theta_link_row)
        link_layout.setContentsMargins(0, 0, 0, 0)
        link_layout.setSpacing(4)
        self.theta_link_checkbox = QtWidgets.QToolButton(self.theta_link_row)
        self.theta_link_checkbox.setObjectName("vtkThetaSliceLinkButton")
        self.theta_link_checkbox.setAutoRaise(True)
        self.theta_link_checkbox.setCheckable(True)
        self.theta_link_checkbox.setChecked(True)
        self.theta_link_checkbox.setFixedSize(20, 20)
        self.theta_link_checkbox.setIconSize(QtCore.QSize(14, 14))
        self.theta_link_checkbox.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.theta_link_checkbox.setAccessibleName("Связать срез θ в 3D с 2D-картой")
        self.theta_link_label = QtWidgets.QLabel("Связать с 2D-картой", self.theta_link_row)
        self.theta_link_label.setObjectName("vtkThetaSliceLinkLabel")
        self.theta_link_label.setToolTip(
            "Синхронизация доступна только в режиме 2D-карты «Срез по θ» "
            "и при совпадающих физических θ-осях."
        )
        link_layout.addWidget(self.theta_link_checkbox, 0)
        link_layout.addWidget(self.theta_link_label, 1)
        theta.layout().addRow("Связь", self.theta_link_row)

        # Optional abstract geometry marker at the selected theta coordinate.
        # It is deliberately NOT a colored copy of the XY data slice: the
        # intensity highlight remains the original in-volume shader path above.
        self.theta_marker_checkbox = QtWidgets.QCheckBox("Плоскость среза θ")
        self.theta_marker_checkbox.setObjectName("vtkThetaSliceMarkerCheckbox")
        self.theta_marker_checkbox.setToolTip(
            "Показать дополнительную полупрозрачную геометрическую XY-плоскость "
            "в координате выбранного θ. Интенсивности и палитра данных не меняются."
        )
        self.theta_marker_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.theta_marker_opacity.setObjectName("vtkThetaSliceMarkerOpacity")
        self.theta_marker_opacity.setRange(2, 25)
        self.theta_marker_opacity.setValue(12)
        self.theta_marker_opacity.setFixedWidth(82)
        self.theta_marker_opacity.setToolTip("Непрозрачность геометрической плоскости: 2–25 %")
        self.theta_marker_opacity_label = QtWidgets.QLabel("12 %")
        marker_row = QtWidgets.QWidget(theta)
        marker_layout = QtWidgets.QHBoxLayout(marker_row)
        marker_layout.setContentsMargins(0, 0, 0, 0)
        marker_layout.setSpacing(5)
        marker_layout.addWidget(self.theta_marker_checkbox, 0)
        marker_layout.addStretch(1)
        marker_layout.addWidget(self.theta_marker_opacity_label, 0)
        marker_layout.addWidget(self.theta_marker_opacity, 0)
        theta.layout().addRow(marker_row)
        self._update_theta_link_icon()

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

        self.peak_marker_size = self._spin(1.0, 20.0, 7.0, 0, " px", step=1.0)
        self.peak_marker_size.setToolTip(
            "Диаметр сферических центров в экранных пикселях. Размер автоматически "
            "пересчитывается при приближении, повороте и изменении размера 3D-окна; "
            "угловой масштаб θ не делает маркеры микроскопическими."
        )
        self.peak_marker_opacity = self._spin(0.05, 1.0, 0.95, 2, step=0.05)
        self.peak_color_by_branch = QtWidgets.QCheckBox("Цвет по branch_id")
        self.peak_color_by_branch.setChecked(True)
        self.peak_color_by_branch.setToolTip(
            "Каждая пространственно связная ветвь получает устойчивый отдельный цвет. "
            "Для результатов поиска/fit без branch_id используется единый цвет."
        )
        peaks = self._settings_panel(
            ("Размер", self.peak_marker_size),
            ("Непрозрачность", self.peak_marker_opacity),
        )
        peaks.layout().addRow(self.peak_color_by_branch)
        self.branch_line_width = self._spin(0.5, 8.0, 2.0, 1, " px", step=0.5)
        self.branch_line_opacity = self._spin(0.05, 1.0, 0.92, 2, step=0.05)
        branches = self._settings_panel(
            ("Толщина", self.branch_line_width),
            ("Непрозрачность", self.branch_line_opacity),
        )
        self.correlation_pair_combo = QtWidgets.QComboBox()
        self.correlation_pair_combo.setObjectName("vtkBranchCorrelationPairSelector")
        self.correlation_pair_combo.addItem("Все пары", None)
        self.correlation_pair_combo.setMinimumContentsLength(15)
        self.correlation_pair_combo.setToolTip(
            "Показать все пары регистрации или только одну пару source ↔ partner."
        )
        self.correlation_show_candidates = QtWidgets.QCheckBox(
            "Альтернативные поддержанные связи"
        )
        self.correlation_show_candidates.setObjectName("vtkBranchCorrelationCandidates")
        self.correlation_show_candidates.setToolTip(
            "Пунктиром показать поддержанные one-to-one кандидаты, которые не были "
            "выбраны итоговым назначением. Они не являются принятыми соответствиями."
        )
        self.correlation_emphasize_core = QtWidgets.QCheckBox(
            "Подсветить совпавшее ядро"
        )
        self.correlation_emphasize_core.setObjectName("vtkBranchCorrelationCoreEmphasis")
        self.correlation_emphasize_core.setChecked(True)
        self.correlation_emphasize_core.setToolTip(
            "Пики, участвующие в принятых соответствиях, остаются яркими; остальные "
            "пики выбранных ветвей приглушаются. Координаты и цвета ветвей не меняются."
        )
        self.correlation_residual_opacity = self._spin(
            0.05, 1.0, 0.28, 2, step=0.05
        )
        self.correlation_line_width = self._spin(
            0.5, 8.0, 2.0, 1, " px", step=0.5
        )
        self.correlation_line_opacity = self._spin(
            0.05, 1.0, 0.82, 2, step=0.05
        )
        correspondences = self._settings_panel(
            ("Пара", self.correlation_pair_combo),
            ("Непрозрачность связей", self.correlation_line_opacity),
            ("Толщина линий", self.correlation_line_width),
            ("Непрозрачность остатка", self.correlation_residual_opacity),
        )
        correspondences.layout().addRow(self.correlation_show_candidates)
        correspondences.layout().addRow(self.correlation_emphasize_core)
        self.tools.set_overlay_settings_widgets(
            cursor_line=cursor, theta_slice=theta, projection=projection,
            roi_preview=roi, peaks=peaks, branches=branches,
            correspondences=correspondences,
        )

        self.cursor_width.valueChanged.connect(self._update_cursor)
        self.cursor_density.valueChanged.connect(self._update_cursor_density)
        self.theta_index.valueChanged.connect(self._apply_theta_index)
        self.theta_link_checkbox.toggled.connect(self._on_theta_link_toggled)
        self.theta_palette.currentIndexChanged.connect(self._apply_theta_palette)
        self.theta_strength.valueChanged.connect(self._apply_theta_strength)
        self.theta_marker_checkbox.toggled.connect(self._toggle_theta_marker)
        self.theta_marker_opacity.valueChanged.connect(self._apply_theta_marker_opacity)
        self.projection_opacity.valueChanged.connect(self._queue_transfer_settings)
        self.projection_palette.currentIndexChanged.connect(self._queue_transfer_settings)
        self.roi_preview_opacity.valueChanged.connect(self._update_roi_preview)
        self.roi_preview_strength.valueChanged.connect(self._update_roi_preview)
        self.roi_preview_mode.currentIndexChanged.connect(self._update_roi_preview)
        self.peak_marker_size.valueChanged.connect(self._apply_peak_marker_settings)
        self.peak_marker_opacity.valueChanged.connect(self._apply_peak_marker_settings)
        self.peak_color_by_branch.toggled.connect(self._apply_peak_marker_coloring)
        self.branch_line_width.valueChanged.connect(self._apply_branch_edge_settings)
        self.branch_line_opacity.valueChanged.connect(self._apply_branch_edge_settings)
        self.correlation_pair_combo.currentIndexChanged.connect(
            self._apply_branch_correlation_selection
        )
        self.correlation_show_candidates.toggled.connect(
            self._apply_branch_correlation_settings
        )
        self.correlation_emphasize_core.toggled.connect(
            self._apply_branch_correlation_settings
        )
        self.correlation_residual_opacity.valueChanged.connect(
            self._apply_branch_correlation_settings
        )
        self.correlation_line_width.valueChanged.connect(
            self._apply_branch_correlation_settings
        )
        self.correlation_line_opacity.valueChanged.connect(
            self._apply_branch_correlation_settings
        )

        self.display_panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.display_panel)
        form.setContentsMargins(8, 7, 8, 7)
        form.setSpacing(5)
        self.low_threshold = self._spin(0.0, 99.0, 16.0, 1, " %", step=0.1)
        self.high_threshold = self._spin(1.0, 100.0, 95.0, 1, " %", step=0.1)
        self.low_threshold.setToolTip("Нижняя граница в процентах от положительного p99.9; число на шкале показывает абсолютную интенсивность.")
        self.high_threshold.setToolTip("Верхняя граница в процентах от положительного p99.9; число на шкале показывает абсолютную интенсивность.")
        self.opacity = self._spin(0.0, 1.0, 0.18, 2, step=0.01)
        self.opacity.setToolTip("Максимальная opacity VTK transfer function (0–1). Это параметр VTK-рендера, не Plotly-opacity.")
        self.gamma = self._spin(0.05, 8.0, 2.0, 2, step=0.05)
        self.gamma.setToolTip("Форма нарастания opacity между Vmin и Vmax; 2.0 плавнее подавляет слабые значения, чем 1.0.")
        self.sample_factor = self._spin(0.25, 8.0, 1.0, 2, " ×", step=0.1)
        self.unit_distance = self._spin(0.05, 20.0, 1.0, 2, " ×", step=0.1)
        self.palette = QtWidgets.QComboBox(self.display_panel)
        from roc_mcs.gui.roi_explorer.colormap_catalog import NAPARI_STYLE_COLORMAPS
        for name in NAPARI_STYLE_COLORMAPS:
            self.palette.addItem(str(name), str(name))
        for label, control in (("Vmin · % p99.9", self.low_threshold), ("Vmax · % p99.9", self.high_threshold), ("Макс. opacity", self.opacity), ("Gamma", self.gamma), ("Ray sample", self.sample_factor), ("Opacity distance", self.unit_distance), ("Colormap", self.palette)):
            form.addRow(label, control)
        self.display = None
        self._threshold_sync = False
        for control in (self.low_threshold, self.high_threshold, self.opacity, self.gamma, self.sample_factor, self.unit_distance):
            control.valueChanged.connect(self._queue_transfer_settings)
        self.palette.currentIndexChanged.connect(self._queue_transfer_settings)

    def _update_theta_link_icon(self) -> None:
        variant = int(self.THETA_LINK_ICON_VARIANT)
        if variant not in (1, 2, 3):
            raise ValueError("THETA_LINK_ICON_VARIANT должен быть 1..3.")
        from roc_mcs.gui.components.icon_provider import svg_icon
        from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR

        colors = theme_colors(self._theme)
        foreground = colors["text"] if self.theta_link_row.isEnabled() else colors["muted"]
        icon_path = ICON_DIR / f"vtk_theta_link_{variant}.svg"
        self.theta_link_checkbox.setIcon(svg_icon(
            icon_path, color=foreground, widget=self.theta_link_checkbox, monochrome=True,
        ))

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
        for signal in (t.axes_toggled, t.bounding_box_toggled, t.axis_ticks_toggled, t.grid_toggled):
            signal.connect(self._schedule_view_state_save)
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
        t.peak_marker_color_changed.connect(self._set_peak_marker_color)

    def set_branch_display_state(
        self,
        branch_colors: dict[int, str] | None = None,
        selected_branch_ids=(),
        *,
        focus_selected: bool = False,
    ) -> None:
        """Apply shared branch-manager state to the current/future 3D payload."""
        colors: dict[int, str] = {}
        for raw_id, raw_color in (branch_colors or {}).items():
            try:
                branch_id = int(raw_id)
                color = str(raw_color).strip().upper()
                if branch_id < 0 or len(color) != 7 or not color.startswith("#"):
                    continue
                int(color[1:], 16)
            except (TypeError, ValueError, OverflowError):
                continue
            colors[branch_id] = color
        selected = tuple(sorted({int(value) for value in selected_branch_ids if int(value) >= 0}))
        self._branch_display_colors = colors
        self._branch_display_selected_ids = selected
        self._branch_display_focus_selected = bool(focus_selected and selected)
        if self._scene is not None:
            self._scene.set_branch_display_style(
                self._branch_display_colors,
                self._branch_display_selected_ids,
                focus_selected=self._branch_display_focus_selected,
                render=True,
            )
        self._update_branch_legend()

    def _scene_call(self, method, *args):
        scene = self._scene
        if scene is not None:
            getattr(scene, method)(*args)
            self._render()

    def set_dataset(
        self, raw, th, axis, x_coords, y_coords, saved_rois, *,
        scan_label="", view_label="", valid_mask=None, source_path=None,
        operation_id="source.raw", metadata=None, peak_catalog=None,
        peak_position_kind="detected", branch_edges=None,
        branch_correlation_preview=None,
    ):
        """Передать выбранную версию ROI Explorer; отдельный HDF5 не открывается."""
        self._theta_range_timer.stop()
        self._pending_theta_range = None
        scan_key = str(scan_label)
        source_identity = f"{Path(source_path).expanduser().resolve() if source_path else ''}|{scan_key}"
        analysis_root = resolve_analysis_projects_directory() if source_path and scan_key else None
        # Include the configured root in the storage identity. If the user changes
        # the global projects path while a viewer remains open, the next dataset
        # selection must switch sidecars rather than silently writing to the old root.
        view_storage_identity = f"{source_identity}|{analysis_root or ''}"
        if self._view_state_identity != view_storage_identity:
            self._save_scan_view_state(flush=True)
            if self._default_view_controls is None:
                self._default_view_controls = self._capture_scan_view_controls()
            self._view_state_identity = view_storage_identity
            self._view_state_path = None
            self._pending_view_state = None
            self._pending_view_camera_state = None
            if source_path and scan_key and analysis_root is not None:
                try:
                    project_dir = project_directory_for_scan(
                        source_path, scan_key, analysis_root
                    )
                    self._view_state_path = project_dir / "view_state.json"
                    self._pending_view_state = load_view_state(self._view_state_path)
                except (OSError, ValueError, TypeError) as exc:
                    logger.warning(
                        "Не удалось загрузить сохранённый вид скана %s: %s",
                        scan_key, exc,
                    )
            if self._pending_view_state is None and self._default_view_controls is not None:
                self._apply_scan_view_controls(self._default_view_controls, restore_camera=False)
            self._first_payload = True
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
        # The peak catalogue is a renderer overlay, not an input to
        # build_volume_payload. Keep it out of the worker request to preserve
        # the data-adapter contract and avoid copying the DataFrame.
        self._peak_catalog = peak_catalog
        self._peak_position_kind = str(peak_position_kind or "detected")
        self._branch_edges = branch_edges
        self._branch_correlation_preview = branch_correlation_preview
        self._sync_branch_correlation_pair_combo()
        self._dataset_request = {
            "raw": raw, "theta": th, "axis": axis, "x_coords": x_coords, "y_coords": y_coords,
            "valid_mask": valid_mask, "roi": self._selected_roi(), "scan_key": scan_key,
            "h5_path": source_path, "operation_id": str(operation_id), "metadata": details,
            # Match the standalone VTK viewer's crisp, unsmoothed starting point.
            # A user-selected non-zero value remains fully supported.
            "smoothing_sigma": float(self._settings.get("smooth_sigma", 0.0)),
        }
        self.source_button.setEnabled(True)
        self._refresh_theta_link_control()
        self._closed = False
        if self._scene is not None:
            self._submit_payload(self._dataset_request)
        else:
            self._schedule_vtk_init()

    def apply_global_icon_size(self) -> None:
        """Refresh embedded 3D source/toolbar geometry after global icon changes."""
        size = VTKToolsWidget._icon_size()
        self.source_button.setFixedSize(size + 10, size + 6)
        self.source_button.setIconSize(QtCore.QSize(size, size))
        self.tools._set_icon_sizes()

    def _capture_scan_view_controls(self) -> dict:
        """Capture Qt-owned display controls without touching scientific data."""
        return {
            "axis_scales": list(self.tools.axis_scales),
            "transfer": {
                "low_percent": float(self.low_threshold.value()),
                "high_percent": float(self.high_threshold.value()),
                "opacity": float(self.opacity.value()),
                "gamma": float(self.gamma.value()),
                "sample_factor": float(self.sample_factor.value()),
                "unit_distance": float(self.unit_distance.value()),
                "palette": str(self.palette.currentData() or "magma"),
                "projection_enabled": bool(self.tools.projection_button.isChecked()),
                "projection_opacity": float(self.projection_opacity.value()),
                "projection_palette": self.projection_palette.currentData(),
            },
            "scene": {
                "axes_visible": bool(self.tools.axes_button.isChecked()),
                "bounding_box_visible": bool(self.tools.bounding_box_button.isChecked()),
                "axis_ticks_visible": bool(self.tools.axis_ticks_action.isChecked()),
                "grid_visible": bool(self.tools.grid_action.isChecked()),
            },
        }

    def _capture_scan_view_state(self) -> dict:
        state = self._capture_scan_view_controls()
        if self._scene is not None:
            camera = self._scene.capture_camera_state()
            if camera is not None:
                state["camera"] = camera
            state["scene"] = {
                "axes_visible": bool(self._scene.axes_visible),
                "bounding_box_visible": bool(self._scene.bounding_box_visible),
                "axis_ticks_visible": bool(self._scene.axis_ticks_visible),
                "grid_visible": bool(self._scene.grid_visible),
            }
        return state

    def _apply_scan_view_controls(self, state: dict, *, restore_camera: bool = True) -> None:
        """Apply a validated sidecar to controls before the next payload build."""
        controls = [
            self.low_threshold, self.high_threshold, self.opacity, self.gamma,
            self.sample_factor, self.unit_distance, self.palette,
            self.projection_opacity, self.projection_palette,
            self.tools.projection_button, self.tools.axes_button,
            self.tools.bounding_box_button, self.tools.axis_ticks_button,
        ]
        blockers = [QtCore.QSignalBlocker(widget) for widget in controls]
        try:
            transfer = state.get("transfer", {}) if isinstance(state, dict) else {}
            for key, widget in (
                ("low_percent", self.low_threshold), ("high_percent", self.high_threshold),
                ("opacity", self.opacity), ("gamma", self.gamma),
                ("sample_factor", self.sample_factor), ("unit_distance", self.unit_distance),
                ("projection_opacity", self.projection_opacity),
            ):
                if key in transfer:
                    widget.setValue(float(transfer[key]))
            palette_index = self.palette.findData(transfer.get("palette"))
            if palette_index >= 0:
                self.palette.setCurrentIndex(palette_index)
            projection_palette_index = self.projection_palette.findData(
                transfer.get("projection_palette")
            )
            if projection_palette_index >= 0:
                self.projection_palette.setCurrentIndex(projection_palette_index)
            if isinstance(transfer.get("projection_enabled"), bool):
                self.tools.projection_button.setChecked(transfer["projection_enabled"])
            axis_scales = state.get("axis_scales") if isinstance(state, dict) else None
            if axis_scales is not None:
                self.tools.set_axis_scales(*axis_scales)
            scene_state = state.get("scene", {}) if isinstance(state, dict) else {}
            for key, method in (
                ("axes_visible", self.tools.set_axes_visible),
                ("bounding_box_visible", self.tools.set_bounding_box_visible),
                ("axis_ticks_visible", self.tools.set_axis_ticks_visible),
                ("grid_visible", self.tools.set_grid_visible),
            ):
                if key in scene_state:
                    method(scene_state[key])
        finally:
            del blockers
        self._pending_view_camera_state = (
            state.get("camera") if restore_camera and isinstance(state, dict) else None
        )

    def _schedule_view_state_save(self, *_args) -> None:
        if (
            self._view_state_path is not None and self._payload is not None
            and self._scene is not None and not self._closed
        ):
            self._view_state_timer.start()

    def _save_scan_view_state(self, *, flush: bool = False) -> None:
        if flush and self._view_state_timer.isActive():
            self._view_state_timer.stop()
        if (
            self._view_state_path is None or self._payload is None
            or self._scene is None or self._closed
        ):
            return
        try:
            save_view_state(self._view_state_path, self._capture_scan_view_state())
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "Не удалось сохранить состояние 3D-вида (%s): %s",
                self._view_state_path, exc,
            )

    def _on_vtk_interaction_finished(self, *_args) -> None:
        self._schedule_view_state_save()

    def update_saved_rois(self, saved_rois) -> None:
        """Refresh the source menu without rebuilding an unchanged volume.

        Saving a new ROI changes the catalogue, not the active dataset. In that
        common case this method deliberately leaves the current payload, VTK
        scene, camera, clipping range, and render-worker generation untouched.
        If the active ROI was actually removed, the source falls back to the full
        detector and only then is a new payload requested.
        """
        previous_source = self.source_combo.currentData() or "full"
        previous_roi_ids = {int(item.id) for item in self._rois}
        validated = []
        seen_roi_ids: set[int] = set()
        for item in (saved_rois or []):
            try:
                normalize_rectangular_roi(item.roi)
                roi_id = int(item.id)
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            if roi_id in seen_roi_ids:
                continue
            seen_roi_ids.add(roi_id)
            validated.append(item)
        new_roi_ids = {int(item.id) for item in validated}
        self._rois = validated
        self._rebuild_source_choices()
        self._refresh_theta_link_control()
        active_roi_removed = (
            previous_source != "full"
            and int(previous_source) not in new_roi_ids
            and int(previous_source) in previous_roi_ids
        )
        if active_roi_removed and self._dataset_request is not None:
            request = dict(self._dataset_request)
            request["roi"] = None
            self._dataset_request = request
            if self._scene is not None:
                self._submit_payload(request)

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
        self._hide_peak_tooltip()
        pending_view_state, self._pending_view_state = self._pending_view_state, None
        if pending_view_state is not None:
            self._apply_scan_view_controls(pending_view_state)
        # Excluded-component visibility is a per-dataset inspection choice;
        # never carry it silently into another scan/preview.
        self._show_excluded_branches = False
        settings = self._transfer_settings()
        info = self._scene.set_payload(payload, settings, axis_scales=self.tools.axis_scales)
        if pending_view_state is not None:
            scene_state = pending_view_state.get("scene", {})
            if "axes_visible" in scene_state:
                self._scene.set_axes_visible(scene_state["axes_visible"])
            for key, method in (
                ("bounding_box_visible", self._scene.set_bounding_box_visible),
                ("axis_ticks_visible", self._scene.set_axis_ticks_visible),
                ("grid_visible", self._scene.set_grid_visible),
            ):
                if key in scene_state:
                    method(scene_state[key], render=False)
        request = self._dataset_request or {}
        marker_error = None
        branch_error = None
        correlation_error = None
        marker_count = 0
        branch_edge_count = 0
        correlation_info = {"matched_links": 0, "candidate_links": 0}
        request_x = request.get("x_coords", payload.x)
        request_y = request.get("y_coords", payload.y)
        try:
            marker_count = self._scene.set_peak_markers(
                self._peak_catalog, request_x, request_y,
                visible=False,
                point_size=float(self.peak_marker_size.value()),
                opacity=float(self.peak_marker_opacity.value()),
                color_by_branch=bool(self.peak_color_by_branch.isChecked()),
                uniform_color=self.tools.peak_marker_color,
                show_excluded_branches=False,
                render=False,
            )
        except Exception as exc:
            # Overlay construction is optional presentation. A malformed or
            # legacy table must never make a valid scientific volume disappear.
            marker_error = exc
            logger.exception("Не удалось построить дополнительный слой центров пиков")
            self._scene.set_peak_markers(None, payload.x, payload.y, visible=False, render=False)
        try:
            branch_edge_count = self._scene.set_branch_edges(
                self._peak_catalog, self._branch_edges, request_x, request_y,
                visible=False,
                color_by_branch=True,
                line_width=float(self.branch_line_width.value()),
                opacity=float(self.branch_line_opacity.value()),
                render=False,
            )
        except Exception as exc:
            branch_error = exc
            logger.exception("Не удалось построить слой связей дифракционных ветвей")
            self._scene.set_branch_edges(
                None, None, payload.x, payload.y, visible=False, render=False
            )

        correlation_model = self._branch_correlation_preview
        has_correlation = bool(
            correlation_model is not None
            and getattr(correlation_model, "pairs", ())
        )
        self._sync_branch_correlation_pair_combo()
        correlation_visible = (
            bool(self._branch_correspondence_overlay_preference)
            if self._branch_correspondence_overlay_preference is not None
            else has_correlation
        ) and has_correlation
        self.tools.set_overlay_tool_checked("correspondences", correlation_visible)
        if has_correlation:
            try:
                correlation_info = self._update_branch_correlation_scene(
                    visible=correlation_visible, render=False
                )
            except Exception as exc:
                correlation_error = exc
                logger.exception("Не удалось построить слой 3D-корреляции ветвей")
                self._scene.set_branch_correspondences(
                    np.empty((0, 2, 3), dtype=np.float64),
                    np.empty(0, dtype=np.int64),
                    visible=False,
                    render=False,
                )
                correlation_visible = False
                self.tools.set_overlay_tool_checked("correspondences", False)
        else:
            # set_payload() clears all previous-scene actors. Reset the logical
            # overlay as well, so no stale legend or preference can suggest
            # that a prior correlation is still attached to this scan.
            self._scene.set_branch_correspondences(
                np.empty((0, 2, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                visible=False,
                render=False,
            )

        self.tools.set_scene_available(True)
        self.tools.set_theta_axis_values(payload.theta)
        # vtk_scene.set_payload() intentionally resets its GPU crop to the full
        # volume. Reapply the toolbar's retained interval after every payload
        # rebuild so the slider and actual scene cannot diverge when only the
        # underlying view/ROI changes but the physical θ-axis stays identical.
        theta_range = self.tools.theta_range_indices
        if theta_range is not None:
            self._scene.set_theta_display_range_indices(
                int(theta_range[0]), int(theta_range[1]), render=False
            )
        self.theta_range_axis_changed.emit(tuple(float(v) for v in payload.theta))
        self.tools.set_axis_scales(*self._scene.axis_scales)
        has_peak_catalog = marker_count > 0
        has_branch_edges = branch_edge_count > 0
        peak_visible = (
            bool(self._peak_overlay_preference)
            if self._peak_overlay_preference is not None else has_peak_catalog
        ) and has_peak_catalog
        branches_visible = (
            bool(self._branch_overlay_preference)
            if self._branch_overlay_preference is not None else has_branch_edges
        ) and has_branch_edges
        self.tools.set_overlay_controls_enabled(
            cursor=True, theta=len(payload.theta) > 2, projection=True,
            roi_preview=True, peaks=has_peak_catalog,
            branches=has_branch_edges, correspondences=has_correlation and correlation_error is None,
            volume=True,
        )
        self.tools.set_overlay_tool_checked("peaks", peak_visible)
        self.tools.set_overlay_tool_checked("branches", branches_visible)
        self._scene.set_peak_markers_visible(peak_visible, render=False)
        self._scene.set_branch_edges_visible(branches_visible, render=False)
        self._scene.set_branch_display_style(
            self._branch_display_colors,
            self._branch_display_selected_ids,
            focus_selected=self._branch_display_focus_selected,
            render=False,
        )
        # The correlation actor is independent of both volume and the ordinary
        # adjacent-pixel branch-edge layer. Its visibility was established
        # above from the stored correlation artifact.
        self._scene.set_branch_correspondences_visible(
            correlation_visible,
            show_candidates=bool(self.correlation_show_candidates.isChecked()),
            render=False,
        )
        self._scene.set_volume_visible(self.tools.is_overlay_tool_checked("volume"), render=False)
        self._update_branch_legend()
        self._update_branch_correlation_legend()
        self.tools.set_projection_enabled(settings.show_projection)
        self._scene.set_theta_slice_palette(str(self.theta_palette.currentData() or "viridis"), render=False)
        self._scene.set_theta_slice_strength(float(self.theta_strength.value()), render=False)
        self._scene.set_theta_slice_marker_opacity(
            float(self.theta_marker_opacity.value()) / 100.0, render=False
        )
        self._scene.set_theta_slice_marker_visible(
            self.theta_marker_checkbox.isChecked(), render=False
        )
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
        self._restore_theta_slice_overlay_state(payload)
        self._refresh_theta_link_control()
        if self._first_payload:
            self._scene.reset_camera_fit_preserve_direction()
            if self._pending_view_camera_state is not None:
                self._scene.restore_camera_state(
                    self._pending_view_camera_state, render=False
                )
                self._pending_view_camera_state = None
            self._first_payload = False
        self._apply_cursor_position()
        if self._roi_preview is not None:
            self._update_roi_preview()
        self._render()
        self._schedule_view_state_save()
        marker_status = ""
        if marker_count:
            marker_state = "показаны" if bool(self._scene._peak_marker_actor.GetVisibility()) else "скрыты"
            marker_size_px = float(getattr(self._scene, "_peak_marker_size", self.peak_marker_size.value()))
            marker_status = (
                f" · центров пиков {marker_count:,} ({self._peak_position_kind}; "
                f"{marker_state}; Ø={marker_size_px:g} px)"
            )
        edge_status = f" · связей ветвей {branch_edge_count:,}" if branch_edge_count else ""
        status_text = (
            f"{payload.scan_key} · {payload.view_mode} · {payload.shape} · "
            f"{payload.scalar_dtype} · invalid {payload.invalid_count:,} · "
            f"{info.get('point_count', 0):,} узлов{marker_status}{edge_status}"
        )
        if marker_error is not None:
            status_text += f" · центры пиков недоступны: {type(marker_error).__name__}"
        if branch_error is not None:
            status_text += f" · связи недоступны: {type(branch_error).__name__}"
        if has_correlation and correlation_error is None:
            model = self._branch_correlation_preview
            total_matches = int(getattr(model, "total_matched", 0))
            total_candidates = int(getattr(model, "total_candidates", 0))
            drawn_matches = int(correlation_info.get("matched_links", 0))
            pair_count = len(getattr(model, "pairs", ()))
            pair_word = "пара" if pair_count == 1 else "пары" if 2 <= pair_count <= 4 else "пар"
            status_text += (
                f" · 3D-корреляция: {total_matches:,} соответствий / {pair_count} {pair_word}"
                f" (в текущем ROI показано {drawn_matches:,})"
            )
            if total_candidates:
                status_text += f" · альтернативных кандидатов {total_candidates:,}"
            assignment_counts = dict(getattr(model, "assignment_counts", {}) or {})
            if assignment_counts:
                status_text += (
                    f" · назначения: принято {int(assignment_counts.get('partner', 0)):,}, "
                    f"неоднозначно {int(assignment_counts.get('ambiguous', 0)):,}, "
                    f"остаток {int(assignment_counts.get('residual', 0)):,}"
                )
            if any(
                bool(getattr(pair, "matched_truncated", False))
                or bool(getattr(pair, "candidate_truncated", False))
                for pair in getattr(model, "pairs", ())
            ):
                status_text += " · геометрия связей прорежена только для отображения"
            warnings = tuple(getattr(model, "warnings", ()) or ())
            if warnings:
                status_text += f" · предупреждение корреляции: {warnings[0]}"
            preview_warning = (request.get("metadata") or {}).get(
                "branch_correlation_preview_warning"
            )
            if preview_warning:
                status_text += f" · {preview_warning}"
        elif correlation_error is not None:
            status_text += f" · 3D-корреляция недоступна: {type(correlation_error).__name__}: {correlation_error}"
        preview_warning = (request.get("metadata") or {}).get(
            "branch_correlation_preview_warning"
        )
        if preview_warning and not has_correlation:
            status_text += f" · {preview_warning}"
        self.status.setText(status_text)

    def _load_vtk_modules(self):
        # Регистрация OpenGL backend обязательна ДО создания QVTK.
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
        # VTK may choose the platform-specific vtkWin32OutputWindow singleton
        # on Windows.  Its native diagnostics window can appear outside the
        # Qt application even when the Python process already has a console.
        # Install the generic vtkOutputWindow explicitly: it writes messages
        # to stderr/stdout and does not create a second native window.  Keep a
        # Python reference because this is a process-wide VTK singleton.
        from vtkmodules.vtkCommonCore import vtkOutputWindow

        global _VTK_OUTPUT_WINDOW_KEEPALIVE
        previous_output_window = vtkOutputWindow.GetInstance()
        if previous_output_window is not None:
            previous_output_window.SetPromptUser(False)
            if hasattr(previous_output_window, "ShowWindowOff"):
                previous_output_window.ShowWindowOff()
            elif hasattr(previous_output_window, "SetShowWindow"):
                previous_output_window.SetShowWindow(False)

        if (
            previous_output_window is not None
            and previous_output_window.GetClassName() == "vtkOutputWindow"
        ):
            # Reuse an already-installed generic console sink.  This makes
            # lazy/repeated 3D initialization idempotent.
            previous_output_window.SetDisplayModeToAlways()
            _VTK_OUTPUT_WINDOW_KEEPALIVE = previous_output_window
        else:
            output_window = vtkOutputWindow()
            output_window.SetPromptUser(False)
            output_window.SetDisplayModeToAlways()
            vtkOutputWindow.SetInstance(output_window)
            _VTK_OUTPUT_WINDOW_KEEPALIVE = output_window

        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        class _TrackedQVTK(QVTKRenderWindowInteractor):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._roc_finalized = False
                self._roc_mouse_move_dispatch = False
                self._roc_drag_previous_vtk_position = None

            def _setEventInformation(self, x, y, ctrl, shift, key, repeat=0, keysum=None):
                # Preserve QVTK's own device-pixel-ratio and Y-flip conversion,
                # then correct LastEventPosition before mouseMoveEvent dispatches
                # the event to vtkInteractorStyleTrackballCamera.
                super()._setEventInformation(x, y, ctrl, shift, key, repeat, keysum)
                if self._roc_mouse_move_dispatch:
                    self._roc_drag_previous_vtk_position = _rebase_vtk_mouse_move(
                        self._Iren, self._roc_drag_previous_vtk_position
                    )

            def mousePressEvent(self, event):
                # Recover if a release was lost while a native toolbar/menu was
                # active. The normal VTK release path still owns ordinary drags.
                self._roc_drag_previous_vtk_position = None
                style = None
                try:
                    style = self._Iren.GetInteractorStyle()
                    if style is not None and int(style.GetState()) != 0:
                        style.StopState()
                        style.ReleaseFocus()
                except Exception:
                    logger.debug("Could not recover a stale QVTK mouse gesture", exc_info=True)
                super().mousePressEvent(event)
                try:
                    if style is not None and int(style.GetState()) != 0:
                        self._roc_drag_previous_vtk_position = tuple(
                            map(int, self._Iren.GetEventPosition())
                        )
                except Exception:
                    self._roc_drag_previous_vtk_position = None
                    logger.debug("Could not capture the QVTK drag start", exc_info=True)

            def mouseMoveEvent(self, event):
                # VTK's GrabFocus() sends drag events exclusively to the style,
                # so an interactor MouseMoveEvent observer cannot rebase them.
                # Interpose inside QVTK's standard adapter after its coordinate
                # conversion and before it invokes the event; keep every other
                # platform-specific conversion and event path unchanged.
                self._roc_mouse_move_dispatch = True
                try:
                    super().mouseMoveEvent(event)
                finally:
                    self._roc_mouse_move_dispatch = False

            def mouseReleaseEvent(self, event):
                try:
                    super().mouseReleaseEvent(event)
                finally:
                    self._roc_drag_previous_vtk_position = None

            def closeEvent(self, event):
                if not self._roc_finalized:
                    self._roc_finalized = True
                    # QVTK owns both the interactor lifecycle and RenderWindow
                    # finalization. Stop only its Qt timer here; do not call
                    # interactor.Disable() or detach its style separately.
                    try:
                        timer = getattr(self, "_Timer", None)
                        if timer is not None:
                            timer.stop()
                    except RuntimeError:
                        pass
                    super().closeEvent(event)  # QVTK performs the single Finalize.
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
        # Geometry is committed by the MDI area before show(). Create the
        # native QVTK surface on the next event-loop turn; readiness is
        # checked explicitly below instead of sleeping for fixed 80/100 ms.
        if self._qvtk is not None:
            QtCore.QTimer.singleShot(0, lambda t=ticket: self._initialize_vtk(t))
        else:
            QtCore.QTimer.singleShot(0, lambda t=ticket: self._ensure_vtk(t))

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
            self.viewport_layout.addWidget(widget, 0, 0)
            # Activate the already-sized viewport layout before creating the
            # native child surface. This uses the final MDI geometry, not a
            # temporary default-size QVTK window.
            self.viewport_layout.activate()
            widget.show()
            widget.winId()
            self._vtk_init_attempts = 0
            QtCore.QTimer.singleShot(0, lambda t=ticket: self._initialize_vtk(t))
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
            native_created = bool(
                widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_WState_Created)
            )
            surface_ready = (
                widget.isVisible()
                and native_created
                and widget.width() > 1
                and widget.height() > 1
            )
            if not surface_ready:
                self._vtk_init_attempts += 1
                if self._vtk_init_attempts <= 40:
                    QtCore.QTimer.singleShot(25, lambda t=ticket: self._initialize_vtk(t))
                    return
                raise RuntimeError(
                    "Qt не подготовил видимую native surface QVTK за 1 секунду "
                    "(видимость, HWND или размер viewport не готовы)."
                )
            # Не вызываем repaint()/processEvents(): QVTK владеет paint-on-screen
            self.apply_theme(resolve_theme(self.parentWidget() or self))
            render_window = widget.GetRenderWindow()
            if not bool(render_window.IsA("vtkOpenGLRenderWindow")):
                raise RuntimeError(f"Ожидался OpenGL RenderWindow, получен {render_window.GetClassName()}.")
            render_window.SetMultiSamples(0)
            interactor = render_window.GetInteractor()
            if interactor is None:
                raise RuntimeError("QVTK не создал vtkRenderWindowInteractor.")
            if not bool(interactor.GetInitialized()):
                # QVTK deliberately uses WA_PaintOnScreen and returns no
                # QPaintEngine.  A synchronous QWidget.repaint() here can
                # re-enter its paintEvent/backing-store path and produce
                # recursive-repaint / inactive-QPainter warnings.  winId()
                # was already materialized in _ensure_vtk(); Initialize()
                # is safe here and the first normal Qt paint performs Render.
                widget.Initialize()
            if not bool(interactor.GetInitialized()):
                raise RuntimeError("VTK interactor не перешёл в состояние Initialized.")
            self._render_window = render_window
            self._interactor = interactor
            # QWidget.update() queues/coalesces a Qt paint event. QVTK's
            # paintEvent then calls vtkGenericRenderWindowInteractor.Render()
            # with the widget's native paint lifecycle in control. Calling
            # widget.Render() here would resolve through __getattr__ to VTK and
            # render synchronously, allowing nested paints during Qt events.
            self._scene = VTKVolumeScene(render_window, interactor, render_callback=widget.update)
            self._scene.set_theme(self._theme)
            try:
                self._view_state_interaction_observer_id = interactor.AddObserver(
                    "EndInteractionEvent", self._on_vtk_interaction_finished
                )
            except Exception:
                self._view_state_interaction_observer_id = None
            if self._peak_hover_observer_id is None:
                self._peak_hover_observer_id = interactor.AddObserver(
                    "MouseMoveEvent", self._on_peak_hover, 0.10
                )
            self.tools.apply_theme(self._theme)
            self.tools.set_scene_available(False)
            if self.display is None:
                # The HUD is a native sibling of QVTK in the shared viewport.
                # A child of the render widget can be clipped by its paint-on-screen
                # surface or be destroyed when the QVTK widget is finalized.
                self.display = VTKVolumeDisplayWidget(self.viewport)
                self.display.attach_settings_panel(self.display_panel)
                self.display.bind_threshold_spins(self.low_threshold, self.high_threshold)
                self.display.thresholds_changed.connect(self._queue_volume_contrast_preview)
                self.display.thresholds_interaction_finished.connect(
                    self._finish_volume_contrast_interaction
                )
                self.viewport_layout.addWidget(
                    self.display, 0, 0,
                    alignment=(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight),
                )
            else:
                self.display.set_viewport(self.viewport)
                if self.viewport_layout.indexOf(self.display) < 0:
                    self.viewport_layout.addWidget(
                        self.display, 0, 0,
                        alignment=(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight),
                    )
            self.display.set_colormap(self._transfer.palette)
            self.display.apply_theme(self._theme)
            self.display.set_colorbar_visible(bool(self._settings.get("show_vtk_colorbar", True)))
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

    def _queue_volume_contrast_preview(self, low: float, high: float) -> None:
        """Queue a fast contrast-only update while the colorbar is dragged."""
        if self._scene is None or self._payload is None or self._closed:
            return
        self._pending_volume_contrast = (float(low), float(high))
        if not self._volume_contrast_timer.isActive():
            self._volume_contrast_timer.start()

    def _finish_volume_contrast_interaction(self) -> None:
        """Ensure the final handle position is rendered immediately on release."""
        if (
            self._scene is None or self._payload is None or self._closed
            or self._pending_volume_contrast is None
        ):
            return
        if self._volume_contrast_timer.isActive():
            self._volume_contrast_timer.stop()
        QtCore.QTimer.singleShot(0, self._apply_pending_volume_contrast)

    def _apply_pending_volume_contrast(self) -> None:
        if (
            self._scene is None
            or self._payload is None
            or self._closed
            or self._pending_volume_contrast is None
        ):
            return
        if self._volume_contrast_timer.isActive():
            self._volume_contrast_timer.stop()
        low, high = self._pending_volume_contrast
        self._pending_volume_contrast = None
        try:
            info = self._scene.update_volume_contrast(
                float(low) / 100.0,
                float(high) / 100.0,
                render=False,
            )
            self._transfer = replace(
                self._transfer,
                lower_fraction=float(low) / 100.0,
                upper_fraction=float(high) / 100.0,
                lower_percentile=None,
                upper_percentile=None,
            )
            # Percentage controls are the single source of truth during a
            # pointer drag. This throttled VTK callback may update the physical
            # min/max labels, but must never rewrite spinboxes or slider handles.
            self.display.update_absolute_thresholds(
                absolute_low=info.get("threshold_min"),
                absolute_high=info.get("threshold_max"),
                reference=info.get("threshold_reference"),
            )
            self._render()
            self._schedule_view_state_save()
        except Exception as exc:
            self.status.setText(f"Не удалось обновить контраст объёма: {exc}")

    def _queue_transfer_settings(self, *_args):
        if self._scene is not None:
            self._settings_timer.start()

    def _apply_transfer_settings(self):
        if self._scene is None or self._payload is None:
            return
        if self._volume_contrast_timer.isActive():
            self._volume_contrast_timer.stop()
        self._pending_volume_contrast = None
        try:
            self._transfer = self._transfer_settings()
            info = self._scene.update_transfer_functions(self._transfer, render=False)
            self.display.set_colormap(self._transfer.palette)
            self.display.set_thresholds(self.low_threshold.value(), self.high_threshold.value(), absolute_low=info.get("threshold_min"), absolute_high=info.get("threshold_max"), reference=info.get("threshold_reference"))
            self._render()
            self._schedule_view_state_save()
        except Exception as exc:
            self.status.setText(f"Настройки отображения не применены: {exc}")

    def apply_display_settings(self, settings=None):
        if settings is None:
            return
        values = dict(settings)
        old_sigma = float(self._settings.get("smooth_sigma", 0.0))
        new_sigma = float(values.get("smooth_sigma", 0.0))
        self._settings = values
        self.low_threshold.setValue(100.0 * float(values.get("isomin_fraction", 0.16)))
        self.high_threshold.setValue(100.0 * float(values.get("isomax_fraction", 0.95)))
        # Plotly's volume_opacity/opacity_mode are not numerically equivalent to
        # VTK's scalar-opacity maximum and gamma.  Keep the VTK controls on the
        # standalone viewer's baseline (0.18 / 2.0), and let the VTK HUD own
        # subsequent edits instead of silently overwriting them with Plotly
        # settings every time the global display dialog is applied.
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
            self.display.set_colorbar_visible(bool(values.get("show_vtk_colorbar", True)))
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
        self.setStyleSheet(f"""
            QWidget#EmbeddedVTKVolumeWidget {{ background:{colors['window']}; color:{colors['text']}; }}
            QWidget#EmbeddedVTKToolbar {{ background:{colors['window']}; }}
            QLabel#vtkEmbeddedStatus {{ color:{colors['muted']}; font-size: 7pt; padding:1px 4px; }}
            QToolButton#vtkThetaSliceLinkButton {{
                min-width:20px; max-width:20px; min-height:20px; max-height:20px;
                padding:0; margin:0; border:1px solid transparent; border-radius:3px;
                background:transparent;
            }}
            QToolButton#vtkThetaSliceLinkButton:hover {{ background:{colors['hover']}; border-color:{colors['border']}; }}
            QToolButton#vtkThetaSliceLinkButton:checked {{ background:{colors['button']}; border-color:{colors['highlight']}; }}
            QToolButton#vtkThetaSliceLinkButton:disabled {{ background:transparent; border-color:transparent; }}
            QLabel#vtkThetaSliceLinkLabel {{ font-size: 7pt; color:{colors['text']}; }}
            QLabel#vtkThetaSliceLinkLabel:disabled {{ color:{colors['muted']}; }}
        """)
        if hasattr(self, "tools"):
            self.tools.apply_theme(self._theme)
        self._update_theta_link_icon()
        if hasattr(self, "source_button"):
            self._update_source_icon()
            from roc_mcs.gui.roi_explorer.theme import apply_popup_theme
            apply_popup_theme(self.source_menu, self._theme)
            menu_font = QtGui.QFont(self.font())
            if menu_font.pointSizeF() <= 0.0:
                menu_font = QtGui.QFont("Sans Serif")
            menu_font.setPointSizeF(8.0)
            self.source_menu.setFont(menu_font)
            menu_colors = theme_colors(self._theme)
            self.source_menu.setStyleSheet(f"""
                QMenu#vtkExplorerSourceMenu {{
                    background:{menu_colors['base']}; color:{menu_colors['text']};
                    border:1px solid {menu_colors['border']}; padding:2px;
                    font-size: 7pt;
                }}
                QMenu#vtkExplorerSourceMenu::item {{
                    background:transparent; color:{menu_colors['text']};
                    padding:3px 20px 3px 8px; min-height: 16px;
                }}
                QMenu#vtkExplorerSourceMenu::item:selected {{
                    background:{menu_colors['highlight']};
                    color:{menu_colors['highlight_text']};
                }}
                QMenu#vtkExplorerSourceMenu::separator {{
                    height:1px; margin:3px 7px; background:{menu_colors['border']};
                }}
            """)
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

    def _restore_theta_slice_overlay_state(self, payload) -> None:
        """Restore the toolbar's theta emphasis state on a newly built volume."""
        has_theta_frames = int(len(payload.theta)) > 2
        requested = bool(self.tools.theta_slice_button.isChecked())
        enabled = bool(has_theta_frames and requested)

        # A scan with too few angular samples cannot provide a meaningful
        # theta selection. Keep the toolbar and scene state consistent.
        if not has_theta_frames and requested:
            self.tools.set_theta_slice_enabled(False)

        if self._scene is not None:
            self._scene.set_theta_slice_enabled(enabled, render=False)

        logger.debug(
            "VTK theta-slice state restored after payload: scan=%s requested=%s "
            "available=%s enabled=%s index=%s",
            getattr(payload, "scan_key", ""),
            requested,
            has_theta_frames,
            enabled,
            getattr(self._scene, "theta_slice_index", None),
        )

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
            and self.theta_link_row.isEnabled()
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
        available = bool(compatible)
        self.theta_link_row.setEnabled(available)
        if available:
            tip = (
                "Связать физическое положение θ-среза в 2D-карте и 3D. "
                "Изменение любого ползунка обновляет второй."
            )
        elif self._map_theta_mode:
            tip = "Синхронизация недоступна: физические θ-оси 2D-карты и 3D-источника не совпадают."
        else:
            tip = "Доступно только в режиме 2D-карты «Срез по θ». В остальных режимах 3D-срез независимый."
        self.theta_link_checkbox.setToolTip(tip)
        self.theta_link_label.setToolTip(tip)
        # Do not lose the user's preference when the control is temporarily
        # unavailable; an inactive disabled row must not look checked/clickable.
        if not available:
            blocked = self.theta_link_checkbox.blockSignals(True)
            try:
                self.theta_link_checkbox.setChecked(False)
            finally:
                self.theta_link_checkbox.blockSignals(blocked)
        else:
            blocked = self.theta_link_checkbox.blockSignals(True)
            try:
                self.theta_link_checkbox.setChecked(self._theta_link_preference)
            finally:
                self.theta_link_checkbox.blockSignals(blocked)
            if self.is_theta_link_active and self._payload is not None:
                map_value = float(self._map_theta_values[self._map_theta_index])
                self.set_theta_index_from_map(map_value)
        self._update_theta_link_icon()

    def _on_theta_link_toggled(self, checked: bool) -> None:
        if not self.theta_link_row.isEnabled():
            return
        self._theta_link_preference = bool(checked)
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

    def _toggle_theta_marker(self, enabled: bool) -> None:
        if self._scene is None or self._payload is None:
            if enabled:
                self.theta_marker_checkbox.setChecked(False)
            return
        self._scene.set_theta_slice_marker_visible(bool(enabled), render=False)
        self._render()

    def _apply_theta_marker_opacity(self, value: int) -> None:
        percent = int(value)
        self.theta_marker_opacity_label.setText(f"{percent} %")
        if self._scene is not None:
            self._scene.set_theta_slice_marker_opacity(percent / 100.0, render=False)
            self._render()

    def _apply_theta_strength(self, value):
        if self._scene is not None:
            self._scene.set_theta_slice_strength(float(value), render=False)
            self._render()

    def _hide_peak_tooltip(self) -> None:
        self._last_hover_peak_key = None
        QtWidgets.QToolTip.hideText()

    def _on_peak_hover(self, _caller=None, _event=None) -> None:
        """Low-cost hover inspection for the currently rendered peak glyphs."""
        scene = self._scene
        interactor = self._interactor
        marker_actor = getattr(scene, "_peak_marker_actor", None) if scene is not None else None
        if scene is None or interactor is None or marker_actor is None or not marker_actor.GetVisibility():
            self._hide_peak_tooltip()
            return
        now = time.monotonic()
        # vtkPointPicker is invoked at most 24 times/s. The camera interactor
        # still receives every event; this observer never aborts interaction.
        if now - self._last_hover_pick_time < (1.0 / 24.0):
            return
        self._last_hover_pick_time = now
        try:
            position = interactor.GetEventPosition()
            window_size = self._render_window.GetSize() if self._render_window is not None else (800, 600)
            min_dimension = max(1, min(int(window_size[0]), int(window_size[1])))
            tolerance = max(
                0.006,
                min(0.030, 1.8 * float(self.peak_marker_size.value()) / min_dimension),
            )
            info = scene.pick_peak_at_display(
                float(position[0]), float(position[1]), tolerance=tolerance
            )
        except Exception:
            logger.debug("Не удалось определить пик под курсором в 3D", exc_info=True)
            info = None
        if info is None:
            self._hide_peak_tooltip()
            return

        key = (info.get("peak_id"), info.get("branch_id"))
        if key == self._last_hover_peak_key:
            return
        self._last_hover_peak_key = key
        color = str(info.get("color", "#FFC857"))
        branch_id = info.get("branch_id")
        if branch_id is None:
            branch_label = "branch_id не назначен"
            count_label = "число точек ветви: —"
        else:
            branch_label = f"branch_id {int(branch_id)}"
            count_label = f"точек в ветви: {int(info.get('branch_event_count', 0)):,}"
        selection_label = (
            "включена в анализ" if bool(info.get("analysis_included", True))
            else "исключена по порогу размера"
        )
        peak_id = info.get("peak_id")
        peak_label = f"peak_id {int(peak_id)}" if peak_id is not None else "peak_id —"
        html = (
            '<div style="white-space:nowrap; padding:2px 3px;">'
            f'<span style="color:{color}; font-size:15px;">●</span> '
            f'<b style="color:{color};">{branch_label}</b><br>'
            f'<span>{count_label}</span><br>'
            f'<span style="color:#888;">{selection_label}</span><br>'
            f'<span>{peak_label}</span><br>'
            f'<span style="color:#888;">X={float(info["x"]):.4g}; '
            f'Y={float(info["y"]):.4g}; θ={float(info["theta"]):.5g}</span>'
            '</div>'
        )
        if self._qvtk is not None:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), html, self._qvtk)

    def _update_branch_legend(self) -> None:
        """Show palette entries for the ordinary overlay or focused selection."""
        panel = getattr(self, "branch_legend", None)
        scene = self._scene
        if panel is None:
            return
        if scene is None:
            panel.hide_on_canvas()
            return
        data = scene.branch_legend_data(limit=48)
        focus_selected = bool(data.get("focus_selected", False))
        peaks_visible = self.tools.is_overlay_tool_checked("peaks")
        if not peaks_visible and not focus_selected:
            panel.hide_on_canvas()
            return
        entries = data.get("entries", ())
        excluded_count = int(data.get("excluded_branches", 0))
        color_by_branch = bool(self.peak_color_by_branch.isChecked())
        # Explicit Branch Manager focus has its own branch palette contract;
        # it remains visible even if the global uniform-marker preference is
        # active, because the manager action is an explicit object selection.
        show_palette = color_by_branch or focus_selected
        if (not show_palette or not entries) and excluded_count <= 0:
            panel.hide_on_canvas()
            return

        legend_entries = [
            LegendEntry(
                label=f"branch_id {int(item['branch_id'])}",
                color=str(item["color"]),
                detail=(
                    f"{int(item['event_count']):,} пиков"
                    + (" · исключена из анализа" if not item.get("analysis_included", True) else "")
                ),
            )
            for item in (entries if show_palette else ())
        ]
        if excluded_count and not focus_selected:
            legend_entries.append(LegendEntry(
                label=(
                    f"Исключённые · {excluded_count:,} ветвей · "
                    f"{int(data.get('excluded_events', 0)):,} пиков"
                ),
                color="#858B95",
                detail="Показать или скрыть отдельным нейтральным слоем; в корреляцию не входят.",
                action_key="show_excluded",
                checked=bool(data.get("show_excluded", False)),
            ))
        if focus_selected:
            panel.set_title(f"ВЫБРАННЫЕ ВЕТВИ · {len(entries)}")
        elif color_by_branch:
            suffix = f" · TOP {len(entries)} / {data['total_branches']}" if data.get("truncated") else ""
            panel.set_title(f"ВЕТВИ · {len(entries)} ПРИНЯТЫХ{suffix}")
        else:
            panel.set_title("ВЕТВИ · ФИЛЬТР ОТОБРАЖЕНИЯ")
        panel.set_entries(legend_entries)
        panel.show_on_canvas(
            position=QtCore.QPoint(
                8,
                max(8, self.viewport.height() - panel.height() - 8),
            )
        )

    def _on_branch_legend_entry_toggled(self, action_key: str, checked: bool) -> None:
        if action_key != "show_excluded" or self._scene is None:
            return
        self._show_excluded_branches = bool(checked)
        self._scene.set_excluded_branches_visible(bool(checked), render=False)
        self._update_branch_legend()
        self._render()

    def _sync_branch_correlation_pair_combo(self) -> None:
        """Populate the pair filter without causing intermediate scene updates."""
        combo = getattr(self, "correlation_pair_combo", None)
        if combo is None:
            return
        model = self._branch_correlation_preview
        pairs = tuple(getattr(model, "pairs", ()) or ()) if model is not None else ()
        previous = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("Все пары", None)
            for pair in pairs:
                source_id = int(pair.source_branch_id)
                partner_id = int(pair.partner_branch_id)
                combo.addItem(
                    f"branch {source_id} ↔ {partner_id}",
                    (source_id, partner_id),
                )
            restore = combo.findData(previous) if previous is not None else 0
            combo.setCurrentIndex(max(0, restore))
            combo.setEnabled(bool(pairs))
        finally:
            combo.blockSignals(False)

    def _selected_branch_correlation_pairs(self) -> tuple[Any, ...]:
        model = self._branch_correlation_preview
        pairs = tuple(getattr(model, "pairs", ()) or ()) if model is not None else ()
        selected = self.correlation_pair_combo.currentData()
        if selected is None:
            return pairs
        try:
            selected_key = (int(selected[0]), int(selected[1]))
        except (TypeError, ValueError, IndexError):
            return pairs
        return tuple(
            pair for pair in pairs
            if (int(pair.source_branch_id), int(pair.partner_branch_id)) == selected_key
        )

    def _update_branch_correlation_scene(
        self, *, visible: bool | None = None, render: bool = True
    ) -> dict[str, int]:
        """Install the selected, already-computed peak correspondences in VTK."""
        if self._scene is None:
            return {"matched_links": 0, "candidate_links": 0, "matched_peaks": 0, "focus_branches": 0}
        selected_pairs = self._selected_branch_correlation_pairs()
        matched_parts: list[np.ndarray] = []
        partner_parts: list[np.ndarray] = []
        candidate_parts: list[np.ndarray] = []
        peak_id_parts: list[np.ndarray] = []
        focus_ids: list[int] = []
        for pair in selected_pairs:
            segments = np.asarray(pair.matched_segments, dtype=np.float64)
            if segments.size:
                matched_parts.append(segments)
                partner_parts.append(
                    np.full(segments.shape[0], int(pair.partner_branch_id), dtype=np.int64)
                )
            alternatives = np.asarray(pair.candidate_segments, dtype=np.float64)
            if alternatives.size:
                candidate_parts.append(alternatives)
            ids = np.asarray(pair.matched_peak_ids, dtype=np.int64).reshape(-1)
            if ids.size:
                # The scene itself deterministically caps the marker cloud.
                # Intersect before concatenating so emphasis work/memory is
                # proportional to rendered glyphs, not the complete catalog.
                displayed_ids = np.asarray(
                    getattr(self._scene, "_peak_marker_peak_ids", np.empty(0)),
                    dtype=np.int64,
                )
                visible_ids = np.intersect1d(ids, displayed_ids, assume_unique=False)
                if visible_ids.size:
                    peak_id_parts.append(visible_ids)
            focus_ids.extend((int(pair.source_branch_id), int(pair.partner_branch_id)))

        matched = (
            np.concatenate(matched_parts, axis=0)
            if matched_parts else np.empty((0, 2, 3), dtype=np.float64)
        )
        partner_ids = (
            np.concatenate(partner_parts, axis=0)
            if partner_parts else np.empty(0, dtype=np.int64)
        )
        candidates = (
            np.concatenate(candidate_parts, axis=0)
            if candidate_parts else np.empty((0, 2, 3), dtype=np.float64)
        )
        matched_peak_ids = (
            np.unique(np.concatenate(peak_id_parts))
            if peak_id_parts else np.empty(0, dtype=np.int64)
        )
        if visible is None:
            visible = self.tools.is_overlay_tool_checked("correspondences")
        return self._scene.set_branch_correspondences(
            matched,
            partner_ids,
            candidate_segments=candidates,
            matched_peak_ids=matched_peak_ids,
            focus_branch_ids=np.unique(np.asarray(focus_ids, dtype=np.int64)),
            visible=bool(visible),
            show_candidates=bool(self.correlation_show_candidates.isChecked()),
            emphasize_core=bool(self.correlation_emphasize_core.isChecked()),
            residual_opacity=float(self.correlation_residual_opacity.value()),
            line_width=float(self.correlation_line_width.value()),
            opacity=float(self.correlation_line_opacity.value()),
            candidate_opacity=float(self.correlation_line_opacity.value()),
            render=render,
        )

    def _update_branch_correlation_legend(self) -> None:
        """Show per-pair shifts and exact match counts in the floating key."""
        panel = getattr(self, "branch_correlation_legend", None)
        if panel is None:
            return
        pairs = self._selected_branch_correlation_pairs()
        should_show = bool(
            self._scene is not None
            and pairs
            and self.tools.is_overlay_tool_checked("correspondences")
        )
        if not should_show:
            panel.hide_on_canvas()
            return

        def signed(value, digits=2):
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                return "—"
            if not np.isfinite(number):
                return "—"
            return f"{number:+.{digits}f}"

        entries = []
        for pair in pairs:
            summary = dict(getattr(pair, "summary", {}) or {})
            x_shift = summary.get("shift_x_px")
            y_shift = summary.get("shift_y_px")
            theta_shift = summary.get("shift_theta_arcsec")
            mad = summary.get("theta_residual_mad_arcsec")
            source_count = int(summary.get("source_event_count", 0))
            matched_count = int(getattr(pair, "matched_total_count", 0))
            candidate_count = int(getattr(pair, "candidate_total_count", 0))
            details = (
                f"{matched_count:,}/{source_count:,} принятых пар · "
                f"ΔX={signed(x_shift, 0)} px · ΔY={signed(y_shift, 0)} px · "
                f"Δθ={signed(theta_shift, 2)}″"
            )
            details += f" · MADθ={signed(mad, 2)}″" if mad is not None else ""
            details += " · сдвиг партнёра → source; цвет линии = branch партнёра"
            invalid_links = int(summary.get("invalid_assignment_count", 0))
            if invalid_links:
                details += f" · не восстановлено по peak_id: {invalid_links:,}"
            if candidate_count:
                details += f" · альтернативы {candidate_count:,}"
            if bool(getattr(pair, "matched_truncated", False)):
                details += " · линии прорежены для экрана"
            entries.append(LegendEntry(
                label=f"branch {int(pair.source_branch_id)} ↔ {int(pair.partner_branch_id)}",
                color=VTKVolumeScene._branch_hex_color(int(pair.partner_branch_id)),
                detail=details,
            ))
        entries.extend((
            LegendEntry(
                label="Сплошная линия",
                color="#4C72B0",
                detail="Принятое соответствие peak_id ↔ peak_id",
            ),
            LegendEntry(
                label="Пунктир",
                color="#EFA542",
                detail="Поддержанный альтернативный кандидат (не назначен)",
            ),
            LegendEntry(
                label="Приглушённый остаток",
                color="#858585",
                detail="Остальные пики выбранных ветвей; исходный цвет branch_id сохранён",
            ),
        ))
        pair_word = "ПАРА" if len(pairs) == 1 else "ПАРЫ"
        panel.set_title(f"3D-КОРРЕЛЯЦИЯ · {len(pairs)} {pair_word}")
        panel.set_entries(entries)
        panel.show_on_canvas(
            position=QtCore.QPoint(
                max(8, self.viewport.width() - panel.width() - 8),
                8,
            )
        )

    def _toggle_projection(self, enabled):
        self._transfer = replace(self._transfer, show_projection=bool(enabled))
        self._queue_transfer_settings()
        self._schedule_view_state_save()

    def _toggle_overlay(self, key, enabled):
        if key == "peaks":
            self._peak_overlay_preference = bool(enabled)
            if self._scene is not None:
                self._scene.set_peak_markers_visible(bool(enabled), render=False)
                marker_count = int(self._scene._peak_marker_polydata.GetNumberOfPoints())
                logger.info(
                    "3D peak-centre overlay toggled: requested=%s visible=%s "
                    "count=%d mapper=%s diameter_px=%.3g world_radius_range=%s bounds=%s",
                    bool(enabled), bool(self._scene._peak_marker_actor.GetVisibility()),
                    marker_count, self._scene._peak_marker_mapper.GetClassName(),
                    float(self._scene._peak_marker_size),
                    tuple(float(v) for v in getattr(self._scene, "_peak_marker_world_radius_range", ())),
                    self._scene._peak_marker_actor.GetBounds(),
                )
                status_text = self.status.text()
                new_state = "показаны" if enabled else "скрыты"
                for old_state in ("показаны", "скрыты"):
                    status_text = status_text.replace(
                        f"; {old_state}; Ø=", f"; {new_state}; Ø=", 1
                    )
                self.status.setText(status_text)
                self._update_branch_legend()
                self._hide_peak_tooltip()
                self._render()
        elif key == "branches":
            self._branch_overlay_preference = bool(enabled)
            if self._scene is not None:
                self._scene.set_branch_edges_visible(bool(enabled), render=False)
                self._render()
        elif key == "correspondences":
            self._branch_correspondence_overlay_preference = bool(enabled)
            if self._scene is not None:
                self._scene.set_branch_correspondences_visible(
                    bool(enabled),
                    show_candidates=bool(self.correlation_show_candidates.isChecked()),
                    render=False,
                )
                self._update_branch_correlation_legend()
                self._render()
        elif key == "volume":
            if self._scene is not None:
                self._scene.set_volume_visible(bool(enabled), render=False)
                self._render()
        elif self._scene is not None and key == "roi_preview":
            self._scene.set_roi_preview_visible(bool(enabled), render=False)
            self._render()

    def _apply_peak_marker_settings(self, *_args):
        if self._scene is not None:
            self._scene.set_peak_marker_settings(
                point_size=float(self.peak_marker_size.value()),
                opacity=float(self.peak_marker_opacity.value()),
                render=False,
            )
            self._render()

    def _apply_peak_marker_coloring(self, *_args):
        if self._scene is None or self._payload is None or self._dataset_request is None:
            return
        request = self._dataset_request
        marker_count = self._scene.set_peak_markers(
            self._peak_catalog,
            request.get("x_coords", self._payload.x),
            request.get("y_coords", self._payload.y),
            visible=self.tools.is_overlay_tool_checked("peaks"),
            point_size=float(self.peak_marker_size.value()),
            opacity=float(self.peak_marker_opacity.value()),
            color_by_branch=bool(self.peak_color_by_branch.isChecked()),
            uniform_color=self.tools.peak_marker_color,
            show_excluded_branches=self._show_excluded_branches,
            render=False,
        )
        self._update_branch_legend()
        self._render()

    def _apply_branch_edge_settings(self, *_args):
        if self._scene is not None:
            self._scene.set_branch_edge_settings(
                line_width=float(self.branch_line_width.value()),
                opacity=float(self.branch_line_opacity.value()),
                render=False,
            )
            self._render()

    def _apply_branch_correlation_selection(self, *_args):
        if self._scene is None or self._branch_correlation_preview is None:
            self._update_branch_correlation_legend()
            return
        visible = self.tools.is_overlay_tool_checked("correspondences")
        self._update_branch_correlation_scene(visible=visible, render=False)
        self._update_branch_correlation_legend()
        self._render()

    def _apply_branch_correlation_settings(self, *_args):
        if self._scene is None:
            return
        self._scene.set_branch_correspondence_settings(
            show_candidates=bool(self.correlation_show_candidates.isChecked()),
            emphasize_core=bool(self.correlation_emphasize_core.isChecked()),
            residual_opacity=float(self.correlation_residual_opacity.value()),
            line_width=float(self.correlation_line_width.value()),
            opacity=float(self.correlation_line_opacity.value()),
            render=False,
        )
        self._update_branch_correlation_legend()
        self._render()

    def _set_peak_marker_color(self, color):
        if self._scene is None or self._payload is None or self._dataset_request is None:
            return
        request = self._dataset_request
        self._scene.set_peak_markers(
            self._peak_catalog,
            request.get("x_coords", self._payload.x),
            request.get("y_coords", self._payload.y),
            visible=self.tools.is_overlay_tool_checked("peaks"),
            point_size=float(self.peak_marker_size.value()),
            opacity=float(self.peak_marker_opacity.value()),
            color_by_branch=bool(self.peak_color_by_branch.isChecked()),
            uniform_color=str(color),
            show_excluded_branches=self._show_excluded_branches,
            render=False,
        )
        self._update_branch_legend()
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
            self._schedule_view_state_save()

    def _perspective_view(self):
        if self._scene is not None:
            self._scene.set_perspective_view()
            self.tools.set_perspective_enabled(self._scene.perspective_enabled)
            self._render()
            self._schedule_view_state_save()

    def _reset_camera(self):
        if self._scene is not None:
            self._scene.reset_camera_fit_preserve_direction()
            self._render()
            self._schedule_view_state_save()

    def _set_axis_scales(self, x, y, theta):
        if self._scene is not None:
            self._scene.set_axis_scales(float(x), float(y), float(theta), render=False)
            self.tools.set_axis_scales(float(x), float(y), float(theta))
            self._render()
            self._schedule_view_state_save()

    def _set_theta_range(self, low, high):
        """Queue the newest range for a bounded-rate VTK scene update."""
        scene = self._scene
        if scene is None or self._closed:
            return
        try:
            low_index, high_index = int(low), int(high)
        except (TypeError, ValueError, OverflowError):
            return
        axis_count = len(self.tools.theta_axis_values)
        if not (0 <= low_index < high_index < axis_count):
            return
        requested = (low_index, high_index)
        if self._pending_theta_range == requested:
            return
        full_range = (0, axis_count - 1)
        normalized = None if requested == full_range else requested
        if self._pending_theta_range is None and scene.theta_display_range_indices == normalized:
            return
        self._pending_theta_range = requested
        # Do not restart an active timer: a continuous drag must not postpone
        # rendering forever. The timeout applies the latest interval received.
        if not self._theta_range_timer.isActive():
            self._theta_range_timer.start()

    def _apply_pending_theta_range(self, *, render: bool = True) -> None:
        pending, self._pending_theta_range = self._pending_theta_range, None
        scene = self._scene
        if pending is None or scene is None or self._closed:
            return
        try:
            low, high = pending
            axis_count = len(self.tools.theta_axis_values)
            requested = None if (low == 0 and high == axis_count - 1) else (low, high)
            if scene.theta_display_range_indices != requested:
                scene.set_theta_display_range_indices(low, high, render=False)
                if render:
                    self._render()
        except (ValueError, RuntimeError, TypeError) as exc:
            logger.debug("Не удалось применить отложенный диапазон θ: %s", exc)

    def set_theta_display_range_indices(self, low: int, high: int) -> None:
        """Apply a linked inclusive theta range without emitting a UI change."""
        self.tools.set_theta_display_range_indices(int(low), int(high))
        self._set_theta_range(int(low), int(high))

    def set_theta_range_interaction_active(self, active: bool) -> None:
        """Temporarily lower only the 3D screen sampling during linked drags."""
        scene = self._scene
        if scene is None or self._closed:
            return
        try:
            changed = scene.set_theta_range_interaction_active(
                bool(active), render=False
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Не удалось изменить интерактивное качество VTK: %s", exc)
            return
        if changed:
            self._render()

    def finish_theta_range_interaction(self) -> None:
        """Apply the last queued crop and restore full quality in one repaint."""
        self._theta_range_timer.stop()
        has_pending_range = self._pending_theta_range is not None
        if has_pending_range:
            self._apply_pending_theta_range(render=False)

        scene = self._scene
        if scene is None or self._closed:
            return
        try:
            quality_changed = scene.set_theta_range_interaction_active(
                False, render=False
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Не удалось восстановить качество VTK: %s", exc)
            quality_changed = False
        if has_pending_range or quality_changed:
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
        self.tools.set_theta_axis_values(None)
        self.theta_range_axis_changed.emit(tuple())
        self._peak_catalog = None
        self._peak_position_kind = "detected"
        self._branch_edges = None
        self._branch_correlation_preview = None
        self.tools.set_overlay_tool_checked("peaks", False)
        self.tools.set_overlay_tool_checked("correspondences", False)
        self._branch_correspondence_overlay_preference = None
        self._sync_branch_correlation_pair_combo()
        if self.branch_legend is not None:
            self.branch_legend.hide_on_canvas()
        if self.branch_correlation_legend is not None:
            self.branch_correlation_legend.hide_on_canvas()
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

    def release_vtk_scene(self, *, final: bool = False):
        """Освободить одну QVTK-сцену через владельца native Qt surface.

        Закрытие MDI-окна только скрывает панель и не вызывает этот метод.
        При окончательном закрытии HUD скрывается, снимаются только observers
        сцены, затем QVTK closeEvent останавливает свой timer и вызывает один
        штатный Finalize без ручного Disable/detach interactor.
        """
        if final:
            self._permanently_closed = True
        if self._release_in_progress:
            if final:
                self._final_shutdown_pending = True
            return

        # A previous X-button close may already have released the scene/QVTK.
        # Final Explorer close must still retire the native HUD and its popup.
        if self._permanently_closed and self._scene is None and self._qvtk is None:
            self._closed = True
            self._roi_preview_timer.stop()
            self._settings_timer.stop()
            self._volume_contrast_timer.stop()
            self._theta_range_timer.stop()
            self._pending_theta_range = None
            self._init_pending = False
            self._init_generation += 1
            self._generation += 1
            try:
                self._thread_pool.clear()
            except RuntimeError:
                pass
            self._active_worker = None
            self._pending_volume_contrast = None
            if self.display is not None:
                try:
                    self.display.prepare_for_shutdown(final=True)
                except (RuntimeError, AttributeError):
                    logger.exception("Не удалось окончательно остановить native VTK HUD")
            return

        self._release_in_progress = True
        self._save_scan_view_state(flush=True)
        self._view_state_timer.stop()
        self._closed = True
        try:
            self._roi_preview_timer.stop()
            self._settings_timer.stop()
            self._volume_contrast_timer.stop()
            self._theta_range_timer.stop()
            self._pending_theta_range = None
            self._init_pending = False
            self._init_generation += 1
            self._generation += 1
            self._pending_request = None
            self._dataset_request = None
            self._payload = None
            self._pending_volume_contrast = None
            self._roi_preview = None

            # Инвалидируем незавершённые CPU-задачи; не завершаем потоки насильно.
            try:
                self._thread_pool.clear()
            except RuntimeError:
                pass
            self._active_worker = None

            self._hide_peak_tooltip()
            if self.branch_legend is not None:
                self.branch_legend.hide_on_canvas()
            if self.branch_correlation_legend is not None:
                self.branch_correlation_legend.hide_on_canvas()
            if self._interactor is not None and self._peak_hover_observer_id is not None:
                try:
                    self._interactor.RemoveObserver(self._peak_hover_observer_id)
                except Exception:
                    logger.debug("Не удалось снять peak-hover observer", exc_info=True)
            self._peak_hover_observer_id = None
            if self._interactor is not None and self._view_state_interaction_observer_id is not None:
                try:
                    self._interactor.RemoveObserver(self._view_state_interaction_observer_id)
                except Exception:
                    logger.debug("Не удалось снять observer состояния 3D-вида", exc_info=True)
            self._view_state_interaction_observer_id = None
            scene, self._scene = self._scene, None
            widget, self._qvtk = self._qvtk, None
            self._render_window = None
            self._interactor = None
            cleanup_errors = []

            if self.display is not None:
                try:
                    # Закрывает popup при обоих сценариях. При temporary close
                    # HUD/event filter остаются переиспользуемыми; final удаляет
                    # popup и снимает event filter перед уничтожением родителей.
                    self.display.prepare_for_shutdown(final=final)
                except (RuntimeError, AttributeError) as exc:
                    cleanup_errors.append(f"display HUD: {exc}")
                    logger.exception("Не удалось остановить VTK HUD")

            if widget is not None:
                try:
                    timer = getattr(widget, "_Timer", None)
                    if timer is not None:
                        timer.stop()
                except Exception as exc:
                    cleanup_errors.append(f"QVTK timer: {type(exc).__name__}: {exc}")
                    logger.exception("Не удалось остановить QVTK timer перед очисткой сцены")

            # Сначала снимаем application observers/widgets сцены при живом
            # interactor. Native OpenGL ресурсы здесь вручную не трогаем.
            if scene is not None:
                try:
                    scene.close()
                except Exception as exc:
                    cleanup_errors.append(f"VTK scene detach: {type(exc).__name__}: {exc}")
                    logger.exception("Не удалось отключить наблюдатели VTK-сцены")

            if widget is not None:
                try:
                    # QVTK is WA_PaintOnScreen and deliberately has no
                    # QPaintEngine.  Hide/close it through its own closeEvent;
                    # do not toggle QWidget updates during native teardown.
                    widget.close()
                except Exception as exc:
                    cleanup_errors.append(f"QVTK Finalize: {type(exc).__name__}: {exc}")
                    logger.exception("Ошибка Finalize для QVTK render window")
                try:
                    self.viewport_layout.removeWidget(widget)
                    # Не setParent(None): native child остаётся под прежним
                    # Qt-родителем и уничтожается стандартным parent lifecycle.
                    widget.deleteLater()
                except RuntimeError as exc:
                    cleanup_errors.append(f"QVTK widget detach: {exc}")

            if not self._permanently_closed:
                if self.placeholder.parent() is None:
                    self.placeholder.setParent(self.viewport)
                self.viewport_layout.addWidget(self.placeholder, 0, 0)
                self.placeholder.setText(
                    "3D-сцена закрыта. Нажмите кнопку 3D, чтобы создать её заново."
                )
                self.placeholder.show()
                self.tools.set_scene_available(False)
                self.tools.set_overlay_controls_enabled(
                    cursor=False, theta=False, projection=False, roi_preview=False
                )
                if cleanup_errors:
                    self.status.setText(
                        "3D закрыто; предупреждение очистки: " + " | ".join(cleanup_errors)
                    )
                else:
                    self.status.setText("VTK/OpenGL-сцена закрыта.")

            # Локальные ссылки удерживают scene/widget до завершения Finalize.
            scene = None
            widget = None
            self._first_payload = True
        finally:
            self._release_in_progress = False
            self._closed = self._permanently_closed
            if self._final_shutdown_pending:
                self._final_shutdown_pending = False
                if self.display is not None:
                    try:
                        self.display.prepare_for_shutdown(final=True)
                    except (RuntimeError, AttributeError):
                        logger.exception("Не удалось окончательно остановить native VTK HUD")

    def _render(self):
        widget = self._qvtk
        if widget is not None and self._scene is not None and not self._closed:
            # update() is asynchronous and coalesced by Qt.  The QVTK
            # paintEvent owns the actual VTK render; never call the delegated
            # vtkGenericRenderWindowInteractor.Render() synchronously here.
            widget.update()

    def closeEvent(self, event):
        self.release_vtk_scene(final=True)
        super().closeEvent(event)
