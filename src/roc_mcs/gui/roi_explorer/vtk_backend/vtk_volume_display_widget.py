"""Viewport-anchored volume contrast controls for the standalone VTK viewer.

The widget is a small HUD attached to the QVTK viewport (not to a 3D actor),
so its color scale and contrast handles remain fixed while the camera moves.
The dual-range slider reuses ROI Explorer's map slider implementation.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QEvent, QLineF, QPoint, QPointF, QRectF, QTimer, Qt, pyqtSignal
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
    """Volume-only dual range slider with stable pointer capture.

    The shared map slider deliberately keeps its existing interaction.  The
    VTK HUD uses a wider *invisible* hit area, preserves the pointer-to-handle
    offset while dragging, and snaps to the precision of its numeric editors
    (tenths in embedded mode, hundredths in the desktop viewer).  This prevents
    a click on the edge of a tiny handle
    from moving the threshold before the drag even begins.
    """

    interactionFinished = pyqtSignal()

    DISPLAY_DECIMALS = 1
    VALUE_STEP = 10.0 ** -DISPLAY_DECIMALS
    HANDLE_HIT_RADIUS = 7.0

    def __init__(self, *args, **kwargs):
        self._theme = "light"
        self._theme_palette = theme_colors(self._theme)
        self._drag_offset = 0.0
        self._last_active_handle = "low"
        self._gesture_active = False
        self._gesture_changed = False
        super().__init__(*args, **kwargs)
        # The painted track remains narrow and centered.  Extra width is only
        # an interaction target, not a thicker visual control.
        self.setFixedSize(14, 154)
        self.setMouseTracking(True)
        # This widget is painted over a native QVTK sibling.  Declare the
        # surface opaque and paint its background explicitly on every expose;
        # transparent child painting can otherwise retain old handle pixels
        # in the native overlay backing store until a parent resize occurs.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)

    def set_theme(self, theme: str) -> None:
        active = str(theme).lower()
        self._theme = active if active in ("light", "dark") else "light"
        self._theme_palette = theme_colors(self._theme)
        self.update()

    def set_display_decimals(self, decimals: int) -> None:
        """Match the precision of the two numeric editors bound to this slider."""
        precision = max(0, min(4, int(decimals)))
        self.DISPLAY_DECIMALS = precision
        self.VALUE_STEP = 10.0 ** -precision if precision else 1.0
        self._minimum_gap = min(self.VALUE_STEP, max(0.0, self._max - self._min))
        self.setValues(self._low, self._high)

    def setValues(self, low: float, high: float) -> None:
        """Keep programmatic values aligned with the bound numeric editors."""
        low = round(float(low), self.DISPLAY_DECIMALS)
        high = round(float(high), self.DISPLAY_DECIMALS)
        super().setValues(low, high)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        self._gesture_active = True
        self._gesture_changed = False
        coordinate = self._coordinate(event)
        low_position = self._val_to_pos(self._low)
        high_position = self._val_to_pos(self._high)
        low_distance = abs(coordinate - low_position)
        high_distance = abs(coordinate - high_position)

        low_hit = low_distance <= self.HANDLE_HIT_RADIUS
        high_hit = high_distance <= self.HANDLE_HIT_RADIUS

        if low_hit and high_hit:
            # When the two handles are close, their enlarged hit regions
            # overlap.  Select by the side of the midpoint; on an exact tie,
            # retain the last handle so repeated clicks remain deterministic.
            midpoint = (low_position + high_position) * 0.5
            if coordinate < midpoint - 0.05:
                active = "high" if self._orientation == Qt.Orientation.Vertical else "low"
            elif coordinate > midpoint + 0.05:
                active = "low" if self._orientation == Qt.Orientation.Vertical else "high"
            else:
                active = self._last_active_handle
            self._active_handle = active
            handle_position = high_position if active == "high" else low_position
            self._drag_offset = coordinate - handle_position
        elif low_hit or high_hit:
            active = "low" if low_hit and (not high_hit or low_distance <= high_distance) else "high"
            self._active_handle = active
            handle_position = low_position if active == "low" else high_position
            # Keep the exact grab offset: grabbing the top/bottom edge of a
            # handle must not make it jump to the pointer's centre.
            self._drag_offset = coordinate - handle_position
        else:
            # A click on the track (outside either handle's hit zone) moves
            # the nearest bound to the clicked position, as expected.
            active = "low" if low_distance <= high_distance else "high"
            self._active_handle = active
            self._drag_offset = 0.0
            self._set_position(coordinate)

        self._last_active_handle = self._active_handle
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._active_handle is None:
            event.ignore()
            return
        self._set_position(self._coordinate(event) - self._drag_offset)
        # The widget is only 14 px wide; repainting the complete track is cheap
        # and guarantees the previous handle locations are erased as well.
        self.update()
        event.accept()

    def cancel_drag(self) -> None:
        """Cancel a pointer grab before the native viewport changes geometry."""
        self._active_handle = None
        self._drag_offset = 0.0

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        gesture_changed = self._gesture_active and self._gesture_changed
        self._gesture_active = False
        self._gesture_changed = False
        self.cancel_drag()
        if gesture_changed:
            self.interactionFinished.emit()
        self.update()
        event.accept()

    def resizeEvent(self, event):
        # The handle positions are calculated from the current widget height.
        # A resize invalidates an in-progress pointer offset, so never continue
        # a drag using coordinates from the previous viewport geometry.
        if event.oldSize() != event.size():
            self.cancel_drag()
        super().resizeEvent(event)
        self.update()

    def _set_position(self, position: float) -> None:
        if self._active_handle not in ("low", "high"):
            return
        value = round(self._pos_to_val(float(position)), self.DISPLAY_DECIMALS)
        old_values = (self._low, self._high)
        if self._active_handle == "low":
            self.setLow(min(99.0, max(0.0, value)))
        else:
            self.setHigh(min(100.0, max(1.0, value)))
        new_values = (self._low, self._high)
        if new_values != old_values:
            if self._gesture_active:
                self._gesture_changed = True
            self.valueChanged.emit(self._low, self._high)

    def event(self, event):
        # If the OS interrupts a drag (for example, focus changes while the
        # pointer is outside the viewport), never leave a stale active handle.
        if event.type() in (QEvent.Type.UngrabMouse, QEvent.Type.FocusOut):
            self._active_handle = None
            self._drag_offset = 0.0
        return super().event(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            # The slider sits on an opaque HUD whose surface colour matches
            # the VTK renderer clear colour.  Clearing before drawing removes
            # every previous anti-aliased handle/track pixel, even when Qt
            # delivers only a partial paint event after mouse movement.
            background = self.palette().color(QPalette.ColorRole.Window)
            # Clear the complete slider surface. QPainter's system clip still
            # honours Qt's expose region, while update() below invalidates the
            # whole 14 px widget for each pointer movement.
            painter.fillRect(self.rect(), background)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            colors = self._theme_palette
            is_dark = self._theme == "dark"
            track = QColor(colors["border"])
            active = QColor("#76A9D5" if is_dark else "#477FA8")
            handle_fill = QColor(colors["button"] if is_dark else colors["base"])
            handle_edge = QColor(colors["text"] if is_dark else colors["border"])

            # The cleared surface uses the exact HUD/renderer background, so
            # the track remains visually integrated without stale alpha pixels.
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
        finally:
            painter.end()


class _VerticalColorRamp(QWidget):
    """Theme-aware vertical rendering of the active volume transfer palette."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = "inferno"
        self.setFixedWidth(8)
        self.setFixedHeight(154)
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
    thresholds_interaction_finished = pyqtSignal()

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
        self._window_owner = self._root_window_widget(viewport)
        # A restore/maximize transition can deliver several native resize
        # notifications before Qt has finished laying out the viewport.  A
        # single-shot timer coalesces them and measures the final geometry.
        self._viewport_sync_timer = QTimer(self)
        self._viewport_sync_timer.setSingleShot(True)
        self._viewport_sync_timer.setInterval(0)
        self._viewport_sync_timer.timeout.connect(self._sync_viewport_geometry)
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
            # The bound editors select the actual precision on bind:
            # 0.1 pp in embedded mode, 0.01 pp in standalone mode.
            minimum_gap=0.1,
        )
        self.range_slider.setObjectName("vtkVolumeContrastRangeSlider")
        # The subclass paints a 7 px visual track inside this 14 px hit area.
        self.range_slider.setFixedWidth(14)
        self.range_slider.setFixedHeight(154)
        self.range_slider.setToolTip(
            "Ползунки задают проценты от положительного p99.9. "
            "Абсолютные значения min/max показаны над и под шкалой."
        )
        self.range_slider.valueChanged.connect(self._on_slider_changed)
        self.range_slider.interactionFinished.connect(
            self.thresholds_interaction_finished.emit
        )
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
        if self._window_owner is not viewport:
            self._window_owner.installEventFilter(self)
        self._apply_overlay_theme(self._theme)
        # Materialize both dimensions before the HUD is inserted into the
        # shared viewport grid. A fixed width alone leaves a QWidget at its
        # default height and clips the color ramp; QGridLayout owns final
        # top-right placement once the widget is registered in that layout.
        self.ensurePolished()
        self._fit_to_viewport()
        self._position_overlay()
        self.hide()

    def set_viewport(self, viewport: QWidget) -> None:
        """Перепривязать HUD к текущему QVTK, сохранив все Qt-контролы."""
        if viewport is self._viewport and self.parentWidget() is viewport:
            return
        self.settings_menu.close()
        self.hide()
        self._viewport_sync_timer.stop()
        old_viewport = self._viewport
        old_layout = self._viewport_layout
        old_window_owner = self._window_owner
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
        if old_window_owner is not None and old_window_owner is not old_viewport:
            try:
                old_window_owner.removeEventFilter(self)
            except RuntimeError:
                pass
        self._viewport = viewport
        self._viewport_layout = viewport.layout()
        self._window_owner = self._root_window_widget(viewport)
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
        if self._window_owner is not viewport:
            self._window_owner.installEventFilter(self)
        if isinstance(self._viewport_layout, QGridLayout):
            self._viewport_layout.addWidget(
                self, 0, 0, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight
            )
        self.ensurePolished()
        self._fit_to_viewport()
        self._position_overlay()
        self._schedule_viewport_sync()
        if self._scene_available and self._show_colorbar:
            self.show()
            self.raise_()

    def eventFilter(self, watched, event):
        # Resize is observed before the target widget's own resize handler and
        # before every dependent layout has necessarily settled.  Defer the
        # actual measurement until the event loop has completed that pass.
        # Also watch the owning window: some window managers report a
        # maximize/restore state change before the QVTK child receives its
        # final native resize notification.
        viewport_events = (
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.ShowToParent,
            QEvent.Type.WinIdChange,
        )
        window_events = (
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.ShowToParent,
            QEvent.Type.WindowStateChange,
        )
        if watched is self._viewport and event.type() in viewport_events:
            self._schedule_viewport_sync()
        elif watched is self._window_owner and event.type() in window_events:
            self._schedule_viewport_sync()
        return super().eventFilter(watched, event)

    def _schedule_viewport_sync(self) -> None:
        """Coalesce native resize/state events into one post-layout sync."""
        if self._native_shutdown_prepared:
            return
        if not self._viewport_sync_timer.isActive():
            self._viewport_sync_timer.start()

    def _sync_viewport_geometry(self) -> None:
        """Re-read settled viewport geometry and repaint the percentage handles."""
        if self._native_shutdown_prepared or self._viewport is None:
            return
        self.ensurePolished()
        self._fit_to_viewport()
        self._position_overlay()
        if self._scene_available and self._show_colorbar and self.isVisible():
            self.raise_()
        self.update()
        self.range_slider.update()
        self.color_ramp.update()

    def _fit_to_viewport(self) -> None:
        """Fit the HUD and both vertical scales inside the current viewport.

        The former 154 px slider/ramp plus a hard 190 px HUD minimum were
        clipped in a restored (non-maximized) MDI window whose viewport was
        shorter than the overlay.  Keep the designed 154 px scale when there
        is room; otherwise shrink slider and ramp together so percentage
        positions remain visible and use the same pixel geometry.
        """
        viewport = self._viewport
        viewport_height = int(viewport.contentsRect().height()) if viewport is not None else 0
        current_track_height = max(1, int(self.range_slider.height()))

        # A parent/window geometry change invalidates the local pointer offset
        # even when the compact track remains at its preferred 154 px height.
        # Keep the editors as the authoritative percentage state and discard
        # any unfinished pointer gesture before projecting it into new geometry.
        self.range_slider.cancel_drag()

        if viewport_height > 0:
            layout = self.layout()
            content_height = int(layout.sizeHint().height()) if layout is not None else current_track_height + 37
            chrome_height = max(0, content_height - current_track_height)
            track_height = max(
                56,
                min(154, viewport_height - chrome_height),
            )
        else:
            track_height = 154

        if track_height != current_track_height:
            self.range_slider.setFixedHeight(track_height)
            self.color_ramp.setFixedHeight(track_height)

        # The spin boxes are the canonical displayed percentages. Reapply
        # them after a viewport resize so the handles are projected from the
        # current values, never from an intermediate pointer coordinate.
        if self._low_spin is not None and self._high_spin is not None:
            low, high = self._normalise_threshold_pair(
                float(self._low_spin.value()), float(self._high_spin.value())
            )
            self.range_slider.setValues(low, high)
        self.range_slider.updateGeometry()
        self.color_ramp.updateGeometry()
        self.range_slider.update()
        self.color_ramp.update()

        self.adjustSize()
        target_height = max(1, int(self.sizeHint().height()))
        if viewport_height > 0:
            target_height = min(target_height, viewport_height)
        if self.width() != 44 or self.height() != target_height:
            self.setFixedSize(44, target_height)

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
        # The embedded ROI Explorer editors display tenths; the separate
        # desktop viewer displays hundredths.  The slider must use the shared
        # precision, otherwise a debounced update rounds its handles back.
        precision = min(int(low_spin.decimals()), int(high_spin.decimals()))
        self.range_slider.set_display_decimals(precision)
        low_spin.valueChanged.connect(self._on_spin_changed)
        high_spin.valueChanged.connect(self._on_spin_changed)
        self.set_thresholds(float(low_spin.value()), float(high_spin.value()))

    @staticmethod
    def _restore_blocked(widget: QWidget, previous_state: bool) -> None:
        """Restore the exact signal-blocking state that was active before a sync."""
        widget.blockSignals(previous_state)

    def _normalise_threshold_pair(
        self, low: float, high: float, *, moved: str | None = None
    ) -> tuple[float, float]:
        """Return bounds representable by the 0.1 %-precision editors."""
        gap = float(self.range_slider.minimum_gap)
        precision = int(self.range_slider.DISPLAY_DECIMALS)
        low = round(max(0.0, min(99.0, float(low))), precision)
        high = round(max(1.0, min(100.0, float(high))), precision)
        if high - low < gap - 1e-9:
            if moved == "high":
                low = round(max(0.0, high - gap), precision)
            else:
                high = round(min(100.0, low + gap), precision)
        # Boundary-safe fallback.  With the configured [0, 99] / [1, 100]
        # editor ranges, a 0.1 gap is always representable.
        if high - low < gap - 1e-9:
            low = round(max(0.0, min(99.0, high - gap)), precision)
            high = round(min(100.0, low + gap), precision)
        return low, high

    def _on_spin_changed(self, *_args) -> None:
        if self._syncing_thresholds or self._low_spin is None or self._high_spin is None:
            return

        moved = "low" if self.sender() is self._low_spin else "high"
        low, high = self._normalise_threshold_pair(
            self._low_spin.value(), self._high_spin.value(), moved=moved
        )

        self._syncing_thresholds = True
        low_blocked = high_blocked = slider_blocked = None
        try:
            low_blocked = self._low_spin.blockSignals(True)
            high_blocked = self._high_spin.blockSignals(True)
            slider_blocked = self.range_slider.blockSignals(True)
            self._low_spin.setValue(low)
            self._high_spin.setValue(high)
            self.range_slider.setValues(low, high)
        finally:
            if slider_blocked is not None:
                self._restore_blocked(self.range_slider, slider_blocked)
            if high_blocked is not None:
                self._restore_blocked(self._high_spin, high_blocked)
            if low_blocked is not None:
                self._restore_blocked(self._low_spin, low_blocked)
            self._syncing_thresholds = False

        # Read the values back from the actual editors so the signal, cbar,
        # transfer settings and displayed numbers all share the same values.
        low = float(self._low_spin.value())
        high = float(self._high_spin.value())
        self._last_low, self._last_high = low, high
        self._update_bound_labels(
            self._absolute_from_percent(low), self._absolute_from_percent(high)
        )
        self.thresholds_changed.emit(low, high)

    def _on_slider_changed(self, low: float, high: float) -> None:
        if self._syncing_thresholds:
            return
        moved = getattr(self.range_slider, "_active_handle", None)
        low, high = self._normalise_threshold_pair(low, high, moved=moved)

        self._syncing_thresholds = True
        low_blocked = high_blocked = slider_blocked = None
        try:
            slider_blocked = self.range_slider.blockSignals(True)
            self.range_slider.setValues(low, high)
            if self._low_spin is not None:
                low_blocked = self._low_spin.blockSignals(True)
                self._low_spin.setValue(low)
            if self._high_spin is not None:
                high_blocked = self._high_spin.blockSignals(True)
                self._high_spin.setValue(high)
        finally:
            if self._high_spin is not None and high_blocked is not None:
                self._restore_blocked(self._high_spin, high_blocked)
            if self._low_spin is not None and low_blocked is not None:
                self._restore_blocked(self._low_spin, low_blocked)
            if slider_blocked is not None:
                self._restore_blocked(self.range_slider, slider_blocked)
            self._syncing_thresholds = False

        # Use editor values, not the pre-rounded mouse coordinates.  This
        # prevents the slider from being pulled back by the debounced VTK
        # settings update after every drag.
        if self._low_spin is not None:
            low = float(self._low_spin.value())
        if self._high_spin is not None:
            high = float(self._high_spin.value())
        self._last_low, self._last_high = low, high
        self._update_bound_labels(
            self._absolute_from_percent(low), self._absolute_from_percent(high)
        )
        self.thresholds_changed.emit(low, high)

    def set_thresholds(
        self,
        low: float,
        high: float,
        *,
        absolute_low: float | None = None,
        absolute_high: float | None = None,
        reference: float | None = None,
    ) -> None:
        low, high = self._normalise_threshold_pair(low, high)
        self._syncing_thresholds = True
        low_blocked = high_blocked = slider_blocked = None
        try:
            slider_blocked = self.range_slider.blockSignals(True)
            self.range_slider.setValues(low, high)
            if self._low_spin is not None:
                low_blocked = self._low_spin.blockSignals(True)
                self._low_spin.setValue(low)
            if self._high_spin is not None:
                high_blocked = self._high_spin.blockSignals(True)
                self._high_spin.setValue(high)
        finally:
            if self._high_spin is not None and high_blocked is not None:
                self._restore_blocked(self._high_spin, high_blocked)
            if self._low_spin is not None and low_blocked is not None:
                self._restore_blocked(self._low_spin, low_blocked)
            if slider_blocked is not None:
                self._restore_blocked(self.range_slider, slider_blocked)
            self._syncing_thresholds = False

        if self._low_spin is not None:
            low = float(self._low_spin.value())
        if self._high_spin is not None:
            high = float(self._high_spin.value())
        self._last_low, self._last_high = low, high
        try:
            reference_value = float(reference) if reference is not None else None
        except (TypeError, ValueError):
            reference_value = None
        self._threshold_reference = (
            reference_value
            if reference_value is not None and math.isfinite(reference_value)
            else None
        )
        if absolute_low is None and self._threshold_reference is not None:
            absolute_low = self._threshold_reference * low / 100.0
        if absolute_high is None and self._threshold_reference is not None:
            absolute_high = self._threshold_reference * high / 100.0
        self._update_bound_labels(absolute_low, absolute_high)

    def _absolute_from_percent(self, value: float) -> float | None:
        if self._threshold_reference is None:
            return None
        if not math.isfinite(self._threshold_reference):
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
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{numeric:.3g}" if math.isfinite(numeric) else "—"

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
        self._viewport_sync_timer.stop()
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
            self._viewport_sync_timer.stop()
            viewport = self._viewport
            if viewport is not None:
                try:
                    viewport.removeEventFilter(self)
                except RuntimeError:
                    pass
            window_owner = self._window_owner
            if window_owner is not None and window_owner is not viewport:
                try:
                    window_owner.removeEventFilter(self)
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
        self._fit_to_viewport()
        self._position_overlay()
        if not was_visible:
            self.show()
            self.raise_()
        self._schedule_viewport_sync()

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
        slider_palette = self.range_slider.palette()
        slider_palette.setColor(QPalette.ColorRole.Window, QColor(surface))
        self.range_slider.setPalette(slider_palette)
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
        self._viewport_sync_timer.stop()
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
        window_owner = self._window_owner
        if window_owner is not None and window_owner is not viewport:
            try:
                window_owner.removeEventFilter(self)
            except RuntimeError:
                pass
        super().closeEvent(event)
