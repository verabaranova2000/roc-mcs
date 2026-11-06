"""Viewport-anchored volume contrast controls for the standalone VTK viewer.

The widget is a small HUD attached to the QVTK viewport (not to a 3D actor),
so its color scale and contrast handles remain fixed while the camera moves.
The dual-range slider reuses ROI Explorer's map slider implementation.
"""
from __future__ import annotations

from PyQt6.QtCore import QLineF, QPoint, QPointF, QRectF, QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from roc_mcs.gui.roi_explorer.colormap_catalog import mpl_colormap
from roc_mcs.gui.roi_explorer.map_tools_widget import QRangeSlider, ColorbarOptionsButton
from roc_mcs.gui.roi_explorer.theme import (
    apply_popup_theme,
    apply_widget_palette,
    theme_colors,
)


class _VolumeContrastRangeSlider(QRangeSlider):
    """Theme-explicit slider used only by the VTK volume HUD.

    The shared map slider keeps its existing palette-driven painting. This
    dedicated subclass paints every surface from ROI Explorer's resolved theme
    so Windows native QMenu/QWidgetAction palette inheritance cannot make the
    dark VTK HUD look like a light widget.
    """

    def __init__(self, *args, **kwargs):
        self._theme = "light"
        self._theme_palette = theme_colors(self._theme)
        super().__init__(*args, **kwargs)

    def set_theme(self, theme: str) -> None:
        active = str(theme).lower()
        self._theme = active if active in ("light", "dark") else "light"
        self._theme_palette = theme_colors(self._theme)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colors = self._theme_palette
        is_dark = self._theme == "dark"
        surface = QColor(colors["window"])
        track = QColor(colors["border"])
        active = QColor("#76A9D5" if is_dark else "#477FA8")
        handle_fill = QColor(colors["button"] if is_dark else colors["base"])
        handle_edge = QColor(colors["text"] if is_dark else colors["border"])

        # Explicitly paint the tiny slider canvas; do not rely on inherited
        # QPalette.Base, which may belong to the host application's light theme.
        painter.fillRect(self.rect(), surface)
        if self._orientation == Qt.Orientation.Vertical:
            x = self.width() * 0.5
            low_y = self._val_to_pos(self._low)
            high_y = self._val_to_pos(self._high)
            painter.setPen(QPen(track, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QLineF(x, 6.0, x, self.height() - 6.0))
            painter.setPen(QPen(active, 2.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QLineF(x, high_y, x, low_y))
            for y in (low_y, high_y):
                handle = QRectF(x - 3.4, y - 2.2, 6.8, 4.4)
                painter.setPen(QPen(handle_edge, 0.85))
                painter.setBrush(handle_fill)
                painter.drawRoundedRect(handle, 1.1, 1.1)
                painter.setPen(QPen(active, 0.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QLineF(x - 2.0, y, x + 2.0, y))
        else:
            y = self.height() * 0.5
            low_x = self._val_to_pos(self._low)
            high_x = self._val_to_pos(self._high)
            painter.setPen(QPen(track, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QLineF(6.0, y, self.width() - 6.0, y))
            painter.setPen(QPen(active, 2.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QLineF(low_x, y, high_x, y))
            for x in (low_x, high_x):
                painter.setPen(QPen(handle_edge, 0.85))
                painter.setBrush(handle_fill)
                painter.drawEllipse(QPointF(x, y), 4.2, 4.2)
        painter.end()


class _VerticalColorRamp(QWidget):
    """Theme-aware vertical rendering of the active volume transfer palette."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = "inferno"
        self.setFixedSize(14, 154)
        self.setAccessibleName("Цветовая шкала интенсивности объёма")

    def set_colormap(self, name: str) -> None:
        self._name = str(name or "inferno")
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        gradient = QLinearGradient(0.0, float(rect.bottom()), 0.0, float(rect.top()))
        cmap = mpl_colormap(self._name)
        for index in range(33):
            position = index / 32.0
            r, g, b, a = cmap(position)
            gradient.setColorAt(
                position,
                QColor.fromRgbF(float(r), float(g), float(b), float(a)),
            )
        edge = self.palette().color(QPalette.ColorRole.Mid)
        painter.setPen(QPen(edge, 1.0))
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, 2.0, 2.0)
        painter.end()


class VTKVolumeDisplayWidget(QFrame):
    """Pinned volume colorbar, dual contrast slider and parameter flyout."""

    thresholds_changed = pyqtSignal(float, float)

    def __init__(self, viewport: QWidget, parent=None):
        super().__init__(viewport if parent is None else parent)
        self._viewport = viewport
        self._theme = "light"
        self._low_spin = None
        self._high_spin = None
        self._syncing_thresholds = False
        self._threshold_reference: float | None = None
        self._last_low: float | None = None
        self._last_high: float | None = None
        self.setObjectName("VTKVolumeDisplayOverlay")
        self.setFixedWidth(86)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(3)

        title = QLabel("INTENSITY", self)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("VTKVolumeColorbarTitle")
        root.addWidget(title)

        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 0, 0, 0)
        scale_row.setSpacing(4)
        self.range_slider = _VolumeContrastRangeSlider(
            low=16.0,
            high=95.0,
            min_val=0.0,
            max_val=100.0,
            parent=self,
            orientation=Qt.Orientation.Vertical,
            minimum_gap=0.01,
        )
        self.range_slider.setObjectName("vtkVolumeContrastRangeSlider")
        self.range_slider.setFixedHeight(154)
        self.range_slider.setToolTip(
            "Нижняя и верхняя границы контраста · доли от положительного p99.9"
        )
        self.range_slider.valueChanged.connect(self._on_slider_changed)
        scale_row.addWidget(self.range_slider, 0, Qt.AlignmentFlag.AlignVCenter)

        self.color_ramp = _VerticalColorRamp(self)
        scale_row.addWidget(self.color_ramp, 0, Qt.AlignmentFlag.AlignVCenter)

        value_column = QVBoxLayout()
        value_column.setContentsMargins(0, 0, 0, 0)
        value_column.setSpacing(0)
        self.vmax_label = QLabel("max: —", self)
        self.vmax_label.setObjectName("VTKVolumeMaxLabel")
        self.vmin_label = QLabel("min: —", self)
        self.vmin_label.setObjectName("VTKVolumeMinLabel")
        for label in (self.vmax_label, self.vmin_label):
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            label.setWordWrap(True)
            label.setToolTip("Абсолютное значение интенсивности на соответствующей границе")
        value_column.addWidget(self.vmax_label)
        value_column.addStretch(1)
        value_column.addWidget(self.vmin_label)
        scale_row.addLayout(value_column, 1)
        root.addLayout(scale_row)

        # Reuse the exact compact ellipsis control already used below the
        # map colorbar; only its hit target is adapted to this vertical HUD.
        self.options_button = ColorbarOptionsButton(self)
        self.options_button.setObjectName("VTKVolumeDisplayOptionsButton")
        self.options_button.setAutoRaise(False)
        self.options_button.set_visual_width(12)
        self.options_button.setFixedSize(28, 14)
        self.options_button.setToolTip("Параметры отображения объёма")
        self.options_button.setAccessibleName("Параметры отображения объёма")
        self.options_button.clicked.connect(self._open_settings_menu)
        options_row = QHBoxLayout()
        options_row.setContentsMargins(0, 1, 0, 0)
        options_row.addStretch(1)
        options_row.addWidget(self.options_button)
        options_row.addStretch(1)
        root.addLayout(options_row)

        self.settings_menu = QMenu(self)
        self.settings_menu.setObjectName("VTKVolumeDisplayMenu")
        self.settings_menu.setToolTipsVisible(True)
        self.settings_panel: QWidget | None = None
        self._settings_action: QWidgetAction | None = None

        viewport.installEventFilter(self)
        self._apply_overlay_theme(self._theme)
        self._position_overlay()
        self.hide()

    def eventFilter(self, watched, event):
        if watched is self._viewport and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.LayoutRequest,
        ):
            self._position_overlay()
        return super().eventFilter(watched, event)

    def _position_overlay(self) -> None:
        if self._viewport is None:
            return
        margin = 12
        x = max(margin, self._viewport.width() - self.width() - margin)
        self.move(x, margin)
        if self.isVisible():
            self.raise_()

    def attach_settings_panel(self, panel: QWidget) -> None:
        """Attach the caller-owned editor to the compact ellipsis menu."""
        if self._settings_action is not None:
            self.settings_menu.removeAction(self._settings_action)
            self._settings_action.deleteLater()
        self.settings_panel = panel
        panel.setParent(self.settings_menu)
        panel.setObjectName("VTKVolumeDisplaySettingsPanel")
        panel.setMinimumWidth(286)
        action = QWidgetAction(self.settings_menu)
        action.setDefaultWidget(panel)
        self.settings_menu.addAction(action)
        self._settings_action = action
        self._apply_settings_panel_theme()

    def bind_threshold_spins(self, low_spin: QWidget, high_spin: QWidget) -> None:
        """Keep precise percentage editors and the viewport slider synchronized."""
        self._low_spin = low_spin
        self._high_spin = high_spin
        low_spin.valueChanged.connect(self._on_spin_changed)
        high_spin.valueChanged.connect(self._on_spin_changed)
        self.set_thresholds(float(low_spin.value()), float(high_spin.value()))

    def _on_spin_changed(self, *_args) -> None:
        if self._syncing_thresholds or self._low_spin is None or self._high_spin is None:
            return
        low = float(self._low_spin.value())
        high = float(self._high_spin.value())
        if low >= high:
            self._syncing_thresholds = True
            if self.sender() is self._low_spin:
                high = min(100.0, low + self.range_slider.minimum_gap)
                self._high_spin.setValue(high)
            else:
                low = max(0.0, high - self.range_slider.minimum_gap)
                self._low_spin.setValue(low)
            self._syncing_thresholds = False
        self.range_slider.blockSignals(True)
        self.range_slider.setValues(low, high)
        self.range_slider.blockSignals(False)
        low, high = self.range_slider.low(), self.range_slider.high()
        self._last_low, self._last_high = low, high
        self.thresholds_changed.emit(low, high)

    def _on_slider_changed(self, low: float, high: float) -> None:
        if self._syncing_thresholds:
            return
        self._syncing_thresholds = True
        try:
            if self._low_spin is not None:
                self._low_spin.blockSignals(True)
                self._low_spin.setValue(round(float(low), 2))
                self._low_spin.blockSignals(False)
            if self._high_spin is not None:
                self._high_spin.blockSignals(True)
                self._high_spin.setValue(round(float(high), 2))
                self._high_spin.blockSignals(False)
        finally:
            self._syncing_thresholds = False
        self._last_low, self._last_high = float(low), float(high)
        self.thresholds_changed.emit(float(low), float(high))

    def set_thresholds(
        self,
        low: float,
        high: float,
        *,
        absolute_low: float | None = None,
        absolute_high: float | None = None,
        reference: float | None = None,
    ) -> None:
        self._syncing_thresholds = True
        try:
            low = max(0.0, min(100.0 - self.range_slider.minimum_gap, float(low)))
            high = max(low + self.range_slider.minimum_gap, min(100.0, float(high)))
            self.range_slider.blockSignals(True)
            self.range_slider.setValues(low, high)
            self.range_slider.blockSignals(False)
            if self._low_spin is not None:
                self._low_spin.blockSignals(True)
                self._low_spin.setValue(low)
                self._low_spin.blockSignals(False)
            if self._high_spin is not None:
                self._high_spin.blockSignals(True)
                self._high_spin.setValue(high)
                self._high_spin.blockSignals(False)
        finally:
            self._syncing_thresholds = False
        self._last_low, self._last_high = low, high
        self._threshold_reference = float(reference) if reference is not None else None
        if absolute_low is None and self._threshold_reference is not None:
            absolute_low = self._threshold_reference * low / 100.0
        if absolute_high is None and self._threshold_reference is not None:
            absolute_high = self._threshold_reference * high / 100.0
        self.vmin_label.setText(f"min: {self._format_value(absolute_low)}")
        self.vmax_label.setText(f"max: {self._format_value(absolute_high)}")

    @staticmethod
    def _format_value(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{float(value):.3g}"

    def set_colormap(self, name: str) -> None:
        self.color_ramp.set_colormap(str(name))

    def set_scene_available(self, available: bool) -> None:
        self.setVisible(bool(available))
        self.setEnabled(bool(available))
        if available:
            self._position_overlay()
            self.raise_()

    def _open_settings_menu(self) -> None:
        if self.settings_panel is None:
            return
        self._apply_settings_panel_theme()
        self.settings_panel.adjustSize()
        self.settings_menu.adjustSize()
        point = self.options_button.mapToGlobal(QPoint(0, self.options_button.height() + 2))
        screen = QApplication.screenAt(point) or QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            if point.y() + self.settings_menu.height() > area.bottom() + 1:
                point.setY(self.options_button.mapToGlobal(QPoint(0, -self.settings_menu.height() - 2)).y())
            point.setX(max(area.left(), min(point.x(), area.right() - self.settings_menu.width() + 1)))
            point.setY(max(area.top(), min(point.y(), area.bottom() - self.settings_menu.height() + 1)))
        self.settings_menu.popup(point)

    def _apply_settings_panel_theme(self) -> None:
        if self.settings_panel is None:
            return
        c = theme_colors(self._theme)
        apply_widget_palette(self.settings_panel, self._theme)
        self.settings_panel.setAutoFillBackground(True)
        self.settings_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.settings_panel.setStyleSheet(f"""
            QWidget#VTKVolumeDisplaySettingsPanel {{
                background: {c['window']}; color: {c['text']};
                border: 0;
            }}
            QWidget#VTKVolumeDisplaySettingsPanel QWidget {{ color: {c['text']}; }}
            QWidget#VTKVolumeDisplaySettingsPanel QLabel {{ background: transparent; color: {c['text']}; border: 0; }}
            QWidget#VTKVolumeDisplaySettingsPanel QDoubleSpinBox,
            QWidget#VTKVolumeDisplaySettingsPanel QComboBox {{
                background: {c['base']}; color: {c['text']};
                border: 1px solid {c['border']}; border-radius: 3px;
                selection-background-color: {c['highlight']};
                selection-color: {c['highlight_text']};
            }}
            QWidget#VTKVolumeDisplaySettingsPanel QComboBox QAbstractItemView {{
                background: {c['base']}; color: {c['text']};
                selection-background-color: {c['highlight']};
                selection-color: {c['highlight_text']};
            }}
            QWidget#VTKVolumeDisplaySettingsPanel QPushButton {{
                background: {c['button']}; color: {c['button_text']};
                border: 1px solid {c['border']}; border-radius: 3px;
                padding: 3px 8px;
            }}
            QWidget#VTKVolumeDisplaySettingsPanel QPushButton:hover {{
                background: {c['hover']}; border-color: {c['highlight']};
            }}
        """)

    def _apply_overlay_theme(self, theme: str) -> None:
        self._theme = str(theme).lower() if str(theme).lower() in ("light", "dark") else "light"
        c = theme_colors(self._theme)
        apply_widget_palette(self, self._theme)
        self.setStyleSheet(f"""
            QFrame#VTKVolumeDisplayOverlay {{
                background: {c['window']}; color: {c['text']};
                border: 1px solid {c['border']}; border-radius: 5px;
            }}
            QFrame#VTKVolumeDisplayOverlay QLabel {{
                background: transparent; color: {c['text']}; border: 0;
                font-size: 8px; padding: 0;
            }}
            QLabel#VTKVolumeColorbarTitle {{ font-size: 8px; font-weight: 700; }}
            QLabel#VTKVolumeMaxLabel, QLabel#VTKVolumeMinLabel {{ font-size: 8px; }}
            QToolButton#VTKVolumeDisplayOptionsButton {{
                background: {c['button']}; color: {c['button_text']};
                border: 1px solid {c['border']}; border-radius: 3px;
                padding: 0; font-size: 11px; font-weight: 700;
            }}
            QToolButton#VTKVolumeDisplayOptionsButton:hover {{
                background: {c['hover']}; border-color: {c['highlight']};
            }}
        """)
        apply_popup_theme(self.settings_menu, self._theme)
        self.settings_menu.setStyleSheet(self.settings_menu.styleSheet() + f"""
            QMenu#VTKVolumeDisplayMenu {{
                background: {c['window']}; color: {c['text']};
                border: 1px solid {c['border']}; padding: 0;
            }}
        """)
        self._apply_settings_panel_theme()
        self.range_slider.set_theme(self._theme)
        self.color_ramp.update()

    def apply_theme(self, theme: str | None = None) -> None:
        active = str(theme or self._theme).lower()
        self._apply_overlay_theme(active)

    def closeEvent(self, event):
        try:
            self._viewport.removeEventFilter(self)
        except RuntimeError:
            pass
        super().closeEvent(event)
