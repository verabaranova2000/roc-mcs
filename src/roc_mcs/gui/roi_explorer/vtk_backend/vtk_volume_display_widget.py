"""Viewport-anchored volume contrast controls for the standalone VTK viewer.

The widget is a small HUD attached to the QVTK viewport (not to a 3D actor),
so its color scale and contrast handles remain fixed while the camera moves.
The dual-range slider reuses ROI Explorer's map slider implementation.
"""
from __future__ import annotations

from PyQt6.QtCore import QLineF, QPoint, QPointF, QRectF, QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
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
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colors = self._theme_palette
        is_dark = self._theme == "dark"
        track = QColor(colors["border"])
        active = QColor("#76A9D5" if is_dark else "#477FA8")
        handle_fill = QColor(colors["button"] if is_dark else colors["base"])
        handle_edge = QColor(colors["text"] if is_dark else colors["border"])

        # Leave the track canvas transparent so the compact HUD reads as one
        # integrated overlay rather than a framed panel.
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
        self.setFixedSize(8, 154)
        self.setToolTip("Интенсивность")
        self.setAccessibleName("Цветовая шкала интенсивности объёма")

    def set_colormap(self, name: str) -> None:
        self._name = str(name or "inferno")
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
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
        painter.setPen(QPen(edge, 0.45))
        painter.setBrush(gradient)
        painter.drawRect(rect)
        painter.end()


class VTKVolumeDisplayWidget(QFrame):
    """Pinned volume colorbar, dual contrast slider and parameter flyout."""

    thresholds_changed = pyqtSignal(float, float)

    @staticmethod
    def _root_window_widget(widget: QWidget) -> QWidget:
        """Return the outermost QWidget owner, not an MDI/native child window."""
        owner = widget
        while owner.parentWidget() is not None:
            owner = owner.parentWidget()
        return owner

    def __init__(self, viewport: QWidget, parent=None):
        super().__init__(viewport if parent is None else parent)
        # Give the entire HUD a valid point-sized font before constructing any
        # child controls.  Some host applications use pixel-sized application
        # fonts (pointSize() == -1); native popup widgets can otherwise emit
        # QFont::setPointSize(-1) when deriving their menu fonts.
        self._menu_font = self._make_positive_point_font(8.0)
        self.setFont(self._menu_font)
        self._viewport = viewport
        self._viewport_layout = viewport.layout()
        self._theme = "light"
        self._scene_available = False
        self._show_colorbar = True
        self._native_shutdown_prepared = False
        self._low_spin = None
        self._high_spin = None
        self._syncing_thresholds = False
        self._threshold_reference: float | None = None
        self._last_low: float | None = None
        self._last_high: float | None = None
        self.setObjectName("VTKVolumeDisplayOverlay")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setLineWidth(0)
        self.setFixedWidth(44)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(True)
        # Embedded mode parents the HUD to the viewport container, as a sibling
        # of QVTK, never to QVTK itself. Both are native children of the same
        # parent; this avoids clipping/reparenting when QVTK is finalized.
        # WA_AlwaysStackOnTop is important for QVTK's paint-on-screen surface.
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.winId()  # force native-handle creation before the first volume render

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        # Numeric bounds sit above/below the combined slider + color strip.
        # They no longer occupy a side column, so long values cannot widen the
        # gap between the two vertical controls.
        self.vmax_label = QLabel("—", self)
        self.vmax_label.setObjectName("VTKVolumeMaxLabel")
        self.vmin_label = QLabel("—", self)
        self.vmin_label.setObjectName("VTKVolumeMinLabel")
        for label in (self.vmax_label, self.vmin_label):
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            label.setWordWrap(False)
            label.setMinimumWidth(0)
            label.setFixedHeight(10)
            label.setToolTip("Абсолютная интенсивность; подробности — при наведении на цветовую шкалу.")

        root.addWidget(self.vmax_label, 0, Qt.AlignmentFlag.AlignHCenter)

        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 0, 0, 0)
        scale_row.setSpacing(4)
        scale_row.addStretch(1)

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
        self.range_slider.setFixedSize(7, 154)
        self.range_slider.setToolTip(
            "Ползунки задают проценты от положительного p99.9. "
            "Абсолютные значения min/max показаны над и под шкалой."
        )
        self.range_slider.valueChanged.connect(self._on_slider_changed)
        scale_row.addWidget(self.range_slider, 0, Qt.AlignmentFlag.AlignVCenter)

        self.color_ramp = _VerticalColorRamp(self)
        self.color_ramp.setAccessibleName("Цветовая шкала интенсивности объёма")
        scale_row.addWidget(self.color_ramp, 0, Qt.AlignmentFlag.AlignVCenter)
        scale_row.addStretch(1)
        root.addLayout(scale_row)

        root.addWidget(self.vmin_label, 0, Qt.AlignmentFlag.AlignHCenter)

        self.options_button = ColorbarOptionsButton(self)
        self.options_button.setObjectName("VTKVolumeDisplayOptionsButton")
        self.options_button.setAutoRaise(False)
        self.options_button.set_visual_width(12)
        self.options_button.setFixedSize(28, 14)
        self.options_button.setToolTip("Параметры отображения объёма")
        self.options_button.setAccessibleName("Параметры отображения объёма")
        self.options_button.clicked.connect(self._open_settings_menu)
        options_row = QHBoxLayout()
        options_row.setContentsMargins(0, 0, 0, 0)
        options_row.addStretch(1)
        options_row.addWidget(self.options_button)
        options_row.addStretch(1)
        root.addLayout(options_row)

        # This HUD is a native child QWidgetWindow above QVTK, not a top-level
        # owner for a popup. Give QMenu the real root application window so
        # Windows transient-parenting never targets VTKVolumeDisplayOverlayWindow.
        self._popup_owner = self._root_window_widget(self)
        self.settings_menu = QMenu(self._popup_owner)
        self.settings_menu.setObjectName("VTKVolumeDisplayMenu")
        self.settings_menu.setToolTipsVisible(True)
        # Native popup menus can outlive the stylesheet context from which
        # they were created; explicitly inherit the HUD's valid point font.
        self.settings_menu.setFont(self._menu_font)
        self.settings_panel: QWidget | None = None
        self._settings_action: QWidgetAction | None = None

        viewport.installEventFilter(self)
        self._apply_overlay_theme(self._theme)
        # Materialize both dimensions before the HUD is inserted into the
        # shared viewport grid. A fixed width alone leaves a QWidget at its
        # default height and clips the color ramp; QGridLayout owns final
        # top-right placement once the widget is registered in that layout.
        self.ensurePolished()
        self.adjustSize()
        content_height = max(190, int(self.sizeHint().height()))
        self.setFixedSize(44, content_height)
        self._position_overlay()
        self.hide()

    def set_viewport(self, viewport: QWidget) -> None:
        """Перепривязать HUD к текущему QVTK, сохранив все Qt-контролы."""
        if viewport is self._viewport and self.parentWidget() is viewport:
            return
        self.settings_menu.close()
        self.hide()
        old_viewport = self._viewport
        old_layout = self._viewport_layout
        if old_layout is not None:
            try:
                old_layout.removeWidget(self)
            except RuntimeError:
                pass
        if old_viewport is not None:
            try:
                old_viewport.removeEventFilter(self)
            except RuntimeError:
                pass
        self._viewport = viewport
        self._viewport_layout = viewport.layout()
        self.setParent(viewport)
        if not self._native_shutdown_prepared:
            new_popup_owner = self._root_window_widget(viewport)
            if new_popup_owner is not self._popup_owner:
                menu_flags = self.settings_menu.windowFlags()
                self.settings_menu.setParent(new_popup_owner, menu_flags)
                self._popup_owner = new_popup_owner
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.winId()
        viewport.installEventFilter(self)
        if isinstance(self._viewport_layout, QGridLayout):
            self._viewport_layout.addWidget(
                self, 0, 0, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight
            )
        self.ensurePolished()
        self.adjustSize()
        self.setFixedSize(44, max(190, int(self.sizeHint().height())))
        self._position_overlay()
        if self._scene_available and self._show_colorbar:
            self.show()
            self.raise_()

    def eventFilter(self, watched, event):
        # Geometry events are infrequent and are the only times the overlay
        # needs repositioning.  LayoutRequest can be generated by the HUD's
        # own popup and controls; reacting to it would create an unnecessary
        # layout/raise feedback loop.
        if watched is self._viewport and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self._position_overlay()
            if self.isVisible():
                self.raise_()
        return super().eventFilter(watched, event)

    def _position_overlay(self) -> None:
        if self._viewport is None:
            return
        # Embedded Explorer uses a QGridLayout with QVTK and this fixed-size
        # HUD in the same cell.  Let Qt own the child geometry; this avoids a
        # manual move() racing the QVTK/native-window resize and clipping pass.
        layout = self._viewport_layout or self._viewport.layout()
        if isinstance(layout, QGridLayout) and layout.indexOf(self) >= 0:
            # Alignment is assigned exactly once when the HUD is inserted via
            # QGridLayout.addWidget(..., alignment=...).  Qt 6 exposes no
            # QLayout.alignment(QWidget) getter; calling it with a widget raises
            # TypeError and aborts VTK scene construction.  The grid owns the
            # HUD geometry on resize, so there is nothing to query or mutate here.
            return
        else:
            margin = 12
            x = max(margin, self._viewport.width() - self.width() - margin)
            self.move(x, margin)

    def attach_settings_panel(self, panel: QWidget) -> None:
        """Attach the caller-owned editor to the compact ellipsis menu."""
        if self._settings_action is not None:
            self.settings_menu.removeAction(self._settings_action)
            self._settings_action.deleteLater()
        self.settings_panel = panel
        panel.setParent(self.settings_menu)
        panel.setObjectName("VTKVolumeDisplaySettingsPanel")
        panel.setFont(self._menu_font)
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
        self._update_bound_labels(
            self._absolute_from_percent(low), self._absolute_from_percent(high)
        )
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
        self._update_bound_labels(
            self._absolute_from_percent(float(low)),
            self._absolute_from_percent(float(high)),
        )
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
        self._update_bound_labels(absolute_low, absolute_high)

    def _absolute_from_percent(self, value: float) -> float | None:
        if self._threshold_reference is None:
            return None
        return float(self._threshold_reference) * float(value) / 100.0

    def _update_bound_labels(
        self, absolute_low: float | None, absolute_high: float | None
    ) -> None:
        min_text = self._format_value(absolute_low)
        max_text = self._format_value(absolute_high)
        self.vmin_label.setText(min_text)
        self.vmax_label.setText(max_text)
        self.color_ramp.setToolTip(
            "Интенсивность"
            f"\nmax: {max_text}"
            f"\nmin: {min_text}"
        )

    @staticmethod
    def _format_value(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{float(value):.3g}"

    def set_colormap(self, name: str) -> None:
        self.color_ramp.set_colormap(str(name))

    def set_colorbar_visible(self, visible: bool) -> None:
        self._show_colorbar = bool(visible)
        self._sync_visibility()

    def set_scene_available(self, available: bool) -> None:
        self._scene_available = bool(available)
        self._sync_visibility()

    def prepare_for_shutdown(self, *, final: bool = False) -> None:
        """Stop popup/paint traffic without manually destroying the native HUD.

        Temporary 3D-window close keeps the HUD reusable. Final Explorer close
        removes its viewport event filter and retires the top-level-owned menu;
        Qt still owns the native child HWND and destroys it with its viewport.
        """
        if final and self._native_shutdown_prepared:
            return
        self._scene_available = False
        try:
            self.settings_menu.close()
        except RuntimeError:
            pass
        try:
            self.hide()
            self.setEnabled(False)
        except RuntimeError:
            pass

        if final:
            self._native_shutdown_prepared = True
            viewport = self._viewport
            if viewport is not None:
                try:
                    viewport.removeEventFilter(self)
                except RuntimeError:
                    pass
            try:
                self.settings_menu.deleteLater()
            except RuntimeError:
                pass

    def _sync_visibility(self) -> None:
        visible = self._scene_available and self._show_colorbar
        was_visible = self.isVisible()
        self.setEnabled(visible)
        if not visible:
            if was_visible:
                self.hide()
            return

        # Recompute the non-zero layout-derived height before the first show.
        self.ensurePolished()
        self.adjustSize()
        self.setFixedSize(44, max(190, int(self.sizeHint().height())))
        self._position_overlay()
        if not was_visible:
            self.show()
            self.raise_()

    def restack_over_viewport(self) -> None:
        """Reposition and restack this native HUD above a paint-on-screen viewport."""
        if not (self._scene_available and self._show_colorbar):
            return
        self._position_overlay()
        if not self.isVisible():
            self.show()
        self.raise_()
        # Do not call update() here.  This method is scheduled after every
        # QVTK paintEvent; repainting an unchanged overlay on every camera
        # frame creates unnecessary nested Qt paint traffic over a native
        # OpenGL widget.  Child controls update themselves when their actual
        # values, palette, or labels change.

    @staticmethod
    def _make_positive_point_font(point_size: float) -> QFont:
        """Return a point-sized font suitable for native popup widgets."""
        font = QFont(QApplication.font())
        if font.pointSizeF() <= 0.0:
            font = QFont("Sans Serif")
        font.setPointSizeF(float(point_size))
        return font

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
        # Match the exact VTK renderer clear colour. The HUD is a native
        # always-on-top sibling of QVTK, so a transparent background exposes
        # whatever MDI window happens to sit behind it after maximize/restore.
        surface = "#111316" if self._theme == "dark" else "#ffffff"
        apply_widget_palette(self, self._theme)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(surface))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(c["text"]))
        self.setPalette(palette)
        self.setStyleSheet(f"""
            QFrame#VTKVolumeDisplayOverlay {{
                background-color: {surface}; color: {c['text']};
                border: 0; border-radius: 0;
            }}
            QFrame#VTKVolumeDisplayOverlay QLabel {{
                background: transparent; color: {c['text']}; border: 0;
                font-size: 6pt; padding: 0;
            }}
            QLabel#VTKVolumeMaxLabel, QLabel#VTKVolumeMinLabel {{
                font-size: 6pt; font-weight: 500; padding: 0; margin: 0;
                background: transparent; border: 0;
            }}
            QWidget#vtkVolumeContrastRangeSlider {{ background: transparent; border: 0; }}
            QToolButton#VTKVolumeDisplayOptionsButton {{
                background: {c['button']}; color: {c['button_text']};
                border: 1px solid {c['border']}; border-radius: 3px;
                padding: 0; font-size: 8pt; font-weight: 700;
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
            self.settings_menu.close()
        except RuntimeError:
            pass
        viewport = self._viewport
        if viewport is not None:
            try:
                viewport.removeEventFilter(self)
            except RuntimeError:
                pass
        super().closeEvent(event)
