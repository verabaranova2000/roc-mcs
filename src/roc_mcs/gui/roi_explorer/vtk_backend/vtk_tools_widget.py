"""Compact scientific toolbar for the standalone ROI Explorer VTK viewer.

The widget owns presentation and interaction only. Scene/data operations are
reported through signals, keeping the numerical backend independent of Qt.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import math

from PyQt6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPalette, QPolygon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QSizePolicy,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.hex_color_picker import CompactColorPicker
from roc_mcs.gui.roi_explorer.map_tools_widget import QRangeSlider, SmartDropdownButton
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR
from roc_mcs.gui.roi_explorer.theme import (
    apply_popup_theme,
    apply_widget_palette,
    theme_colors,
)


class _OverlayToolButton(SmartDropdownButton):
    """LMB toggles an object; RMB opens its options without toggling it."""

    settings_requested = pyqtSignal()
    primary_pressed = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.settings_requested.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.primary_pressed.emit()
        super().mousePressEvent(event)


class _PopupHorizontalResizeGrip(QWidget):
    """A compact drag handle that widens its owning overlay popup only."""

    def __init__(self, menu: QMenu, panel: QWidget, base_width: int, parent=None):
        super().__init__(parent)
        self._menu = menu
        self._panel = panel
        self._base_width = max(1, int(base_width))
        self._base_panel_width = max(1, int(base_width))
        self._menu_chrome_width = 0
        self._drag_origin = None
        self._start_width = 0
        self.setFixedSize(15, 13)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("Потяните влево/вправо, чтобы изменить ширину окна")
        self.setAccessibleName("Изменить ширину окна настроек")

    def set_base_width(self, menu_width: int, panel_width: int) -> None:
        self._base_width = max(1, int(menu_width))
        self._base_panel_width = max(1, int(panel_width))
        self._menu_chrome_width = max(0, self._base_width - self._base_panel_width)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = self.palette().color(QPalette.ColorRole.Mid)
        painter.setPen(QPen(color, 1.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        # Three short diagonal strokes are a familiar, low-noise resize cue.
        for offset in (0, 4, 8):
            painter.drawLine(self.width() - 3 - offset, self.height() - 2,
                             self.width() - 2, self.height() - 3 - offset)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint()
            self._start_width = int(self._menu.width())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint().x() - self._drag_origin.x()
            screen = QApplication.screenAt(self._menu.frameGeometry().topLeft()) or QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                max_width = max(self._base_width, available.right() - self._menu.frameGeometry().left() + 1)
            else:
                max_width = max(self._base_width, self._start_width + 600)
            target = max(self._base_width, min(max_width, self._start_width + int(delta)))
            # The content widget and its dedicated slider rows are expanding;
            # labels and compact settings remain content-sized while the user
            # gains a longer travel rail without changing any slider values.
            panel_target = max(
                self._base_panel_width,
                target - self._menu_chrome_width,
            )
            # Fix the live width of the panel (rather than only raising its
            # minimum).  The popup's child layout can then allocate the extra
            # width to its expanding slider widgets, while labels and compact
            # controls retain their natural widths and fixed spacing.
            self._panel.setFixedWidth(panel_target)
            layout = self._panel.layout()
            if layout is not None:
                layout.activate()
            self._menu.setFixedWidth(target)
            self._menu.resize(target, self._menu.height())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _DirectionalAxisButton(QToolButton):
    """Deterministic camera command: LMB = +axis, RMB = -axis."""

    direction_requested = pyqtSignal(str, int)

    def __init__(self, axis: str, parent=None):
        super().__init__(parent)
        self._axis = str(axis).lower()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            polarity = 1 if event.button() == Qt.MouseButton.LeftButton else -1
            self.direction_requested.emit(self._axis, polarity)
            event.accept()
            return
        super().mousePressEvent(event)


class _AxisScalePreview(QWidget):
    """Theme-aware wireframe preview of the relative X/Y/θ unit cell."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scales = (1.0, 1.0, 1.0)
        self._theme = "light"
        self._theme_palette = theme_colors(self._theme)
        self.setFixedSize(100, 72)
        self.setAccessibleName("Предпросмотр пропорций единичной ячейки")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    def set_theme(self, theme: str) -> None:
        active = str(theme).lower()
        self._theme = active if active in ("light", "dark") else "light"
        self._theme_palette = theme_colors(self._theme)
        self.update()

    def set_scales(self, x: float, y: float, theta: float) -> None:
        self._scales = (float(x), float(y), float(theta))
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colors = self._theme_palette
        surface = QColor(colors["window"])
        edge = QColor(colors["border"])
        text = QColor(colors["text"])
        painter.fillRect(self.rect(), surface)
        painter.setPen(QPen(edge, 1.0))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 3.0, 3.0)

        sx, sy, sz = self._scales
        normalizer = max(sx, sy, sz, 1e-9)
        origin = QPointF(47.0, 62.0)
        vx = QPointF(27.0 * sx / normalizer, 0.0)
        vy = QPointF(-16.0 * sy / normalizer, -10.0 * sy / normalizer)
        vz = QPointF(0.0, -34.0 * sz / normalizer)
        p000 = origin
        p100 = origin + vx
        p010 = origin + vy
        p110 = origin + vx + vy
        p001 = origin + vz
        p101 = origin + vx + vz
        p011 = origin + vy + vz
        p111 = origin + vx + vy + vz
        for a, b in (
            (p000, p100), (p000, p010), (p100, p110), (p010, p110),
            (p001, p101), (p001, p011), (p101, p111), (p011, p111),
            (p000, p001), (p100, p101), (p010, p011), (p110, p111),
        ):
            painter.setPen(QPen(edge, 0.9))
            painter.drawLine(a, b)

        # Preserve the scientific X/Y/θ convention while all neutral surfaces
        # and labels follow the active application theme.
        for vector, color in (
            (vx, QColor("#D65A4A")),
            (vy, QColor("#459B70")),
            (vz, QColor("#477BC5")),
        ):
            painter.setPen(QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(origin, origin + vector)
        painter.setPen(QPen(text, 1.0))
        painter.drawText(78, 69, "X")
        painter.drawText(22, 48, "Y")
        painter.drawText(50, 20, "θ")
        painter.end()


class VTKToolsWidget(QWidget):
    """Small, embeddable toolbar for camera, axes, overlays, and appearance."""

    reset_view_requested = pyqtSignal()
    axes_toggled = pyqtSignal(bool)
    bounding_box_toggled = pyqtSignal(bool)
    axis_ticks_toggled = pyqtSignal(bool)
    grid_toggled = pyqtSignal(bool)
    perspective_view_requested = pyqtSignal()
    perspective_toggled = pyqtSignal(bool)  # legacy compatibility
    camera_axis_requested = pyqtSignal(str)  # legacy: positive direction only
    camera_axis_direction_requested = pyqtSignal(str, int)

    overlay_toggled = pyqtSignal(str, bool)
    overlay_settings_requested = pyqtSignal(str)
    cursor_line_toggled = pyqtSignal(bool)
    theta_slice_toggled = pyqtSignal(bool)
    projection_toggled = pyqtSignal(bool)
    cursor_line_color_changed = pyqtSignal(str)
    roi_preview_color_changed = pyqtSignal(str)
    axis_scales_apply_requested = pyqtSignal(float, float, float)
    # Index bounds are inclusive and refer to the currently loaded physical theta axis.
    theta_display_range_changed = pyqtSignal(int, int)
    theta_boundary_planes_toggled = pyqtSignal(bool)
    theta_boundary_plane_opacity_changed = pyqtSignal(float)
    theme_changed = pyqtSignal(str)

    DEFAULT_CURSOR_LINE_COLOR = "#FFBD55"
    DEFAULT_ROI_PREVIEW_COLOR = "#FFB547"
    # 0 = restore the exact legacy icons from the supplied project archive.
    # 1..3 = the first monochrome proposal set; 4..7 = the new concepts A..D.
    SCENE_OBJECTS_ICON_VARIANT = 0
    # 0 = restore the original cursor / theta / XY icons; 1..3 = Round 6 sets.
    SCENE_OBJECTS_CHILD_ICON_VARIANT = 0
    AXIS_TICKS_ICON_VARIANT = 1
    # CAMERA_AXES_ICON_VARIANT=0 keeps the current plain triad. Values 1..3
    # select the optional monochrome "axes + settings" concepts shipped with
    # this update. Icon choices do not affect camera behavior.
    CAMERA_AXES_ICON_VARIANT = 0
    AXIS_SCALE_ICON_VARIANT = 1
    # 1..3 select alternate scientific icons for the live theta-volume crop tool.
    THETA_RANGE_ICON_VARIANT = 1
    # 1..3 are alternative compact compositions for live ROI selector preview.
    ROI_PREVIEW_ICON_VARIANT = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VTKToolBar")
        self._theme = "light"
        self._cursor_line_color = self.DEFAULT_CURSOR_LINE_COLOR
        self._roi_preview_color = self.DEFAULT_ROI_PREVIEW_COLOR
        self._color_picker: CompactColorPicker | None = None
        self._active_color_picker_key: str | None = None
        self._overlay_settings: dict[str, QWidget] = {}
        self._overlay_settings_menus: dict[str, QMenu] = {}
        self._overlay_settings_panels: dict[str, QWidget] = {}
        self._overlay_popup_base_widths: dict[str, int] = {}
        self._overlay_popup_base_menu_widths: dict[str, int] = {}
        self._overlay_popup_base_heights: dict[str, int] = {}
        self._overlay_buttons: dict[str, _OverlayToolButton] = {}
        self._overlay_tool_specs: dict[str, dict[str, object]] = {}
        self._active_overlay_settings: str | None = None
        self._perspective_enabled = True
        self._axis_scales = (1.0, 1.0, 1.0)
        self._applied_axis_scales = (1.0, 1.0, 1.0)
        self._theta_axis_values: list[float] = []
        self._theta_range_indices: tuple[int, int] | None = None
        self.setFixedHeight(34)
        self.setStyleSheet(
            """
            QWidget#VTKToolBar QToolButton {
                max-height: 27px; min-height: 25px; min-width: 27px;
                padding: 1px; margin: 0; border: 1px solid transparent;
                border-radius: 3px; background: transparent;
                color: palette(button-text);
            }
            QWidget#VTKToolBar QToolButton:hover {
                background: palette(midlight); border-color: palette(mid);
            }
            QWidget#VTKToolBar QToolButton:checked {
                background: palette(highlight); color: palette(highlighted-text);
                border-color: palette(highlight);
            }
            QWidget#VTKToolBar QToolButton[settingsActive="true"] {
                border-color: palette(highlight); background: palette(alternate-base);
            }
            QWidget#VTKToolBar QToolButton::menu-indicator { image: none; }
            QWidget#VTKToolBar QMenu::indicator {
                background: palette(base); border: 1px solid palette(mid);
            }
            QWidget#VTKToolBar QMenu::indicator:checked { background: palette(highlight); }
            """
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(5, 2, 5, 2)
        row.setSpacing(3)

        self.reset_btn = self._toolbar_button(
            "zoom_out_map.svg", "Вписать объём, сохранив направление камеры"
        )
        self.reset_btn.clicked.connect(self.reset_view_requested.emit)
        row.addWidget(self.reset_btn)

        # Standard camera orientations are grouped in their own compact flyout,
        # immediately after Fit.  The four child commands reuse the existing
        # camera signals, so the VTK camera implementation remains unchanged.
        self.orientation_btn = SmartDropdownButton(self)
        self.orientation_btn.setObjectName("vtkOrientationMenuButton")
        self.orientation_btn.setAutoRaise(True)
        self.orientation_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.orientation_btn.setFixedSize(27, 27)
        self.orientation_btn.setIconSize(QSize(18, 18))
        self.orientation_btn.setToolTip("Стандартные ракурсы 3D-сцены")
        self.orientation_btn.setAccessibleName("Стандартные ракурсы 3D-сцены")
        self.orientation_menu = QMenu(self.orientation_btn)
        self.orientation_menu.setObjectName("VTKOrientationMenu")
        self.orientation_menu.setToolTipsVisible(True)
        self.orientation_menu.setStyleSheet(self._popup_button_stylesheet())
        self._build_orientation_menu()
        self.orientation_btn.setMenu(self.orientation_menu)
        row.addWidget(self.orientation_btn)

        # These launchers use the very same SmartDropdownButton class as the
        # map selector/pooling tools, including its lower-right corner marker.
        self.camera_btn = SmartDropdownButton(self)
        self.camera_btn.setObjectName("vtkCameraMenuButton")
        self.camera_btn.setAutoRaise(True)
        self.camera_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.camera_btn.setFixedSize(27, 27)
        self.camera_btn.setIconSize(QSize(18, 18))
        self.camera_btn.setToolTip("Камера и координатные оси")
        self.camera_btn.setAccessibleName("Камера и координатные оси")
        self.camera_menu = QMenu(self.camera_btn)
        self.camera_menu.setObjectName("VTKCameraMenu")
        self.camera_menu.setToolTipsVisible(True)
        self.camera_menu.setStyleSheet(self._popup_button_stylesheet())
        self._build_camera_menu()
        self.camera_btn.setMenu(self.camera_menu)
        row.addWidget(self.camera_btn)

        self.overlays_btn = SmartDropdownButton(self)
        self.overlays_btn.setObjectName("vtkSceneObjectsButton")
        self.overlays_btn.setAutoRaise(True)
        self.overlays_btn.setFixedSize(27, 27)
        self.overlays_btn.setIconSize(QSize(self._icon_size(), self._icon_size()))
        self.overlays_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.overlays_btn.setToolTip("Дополнительные элементы 3D-сцены")
        self.overlays_btn.setAccessibleName("Дополнительные элементы 3D-сцены")
        self.overlays_menu = QMenu(self.overlays_btn)
        self.overlays_menu.setObjectName("VTKOverlaysMenu")
        self.overlays_menu.setToolTipsVisible(True)
        self.overlays_menu.setStyleSheet(self._popup_button_stylesheet())
        self._build_overlays_menu()
        self.overlays_menu.aboutToHide.connect(self._close_overlay_settings)
        self.overlays_btn.setMenu(self.overlays_menu)
        row.addWidget(self.overlays_btn)

        # Live theta display crop.  The slider is a rendering-only control:
        # it sends inclusive sample indices to the viewer and never edits data.
        self.theta_range_btn = SmartDropdownButton(self)
        self.theta_range_btn.setObjectName("vtkThetaDisplayRangeButton")
        self.theta_range_btn.setAutoRaise(True)
        self.theta_range_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.theta_range_btn.setFixedSize(27, 27)
        self.theta_range_btn.setIconSize(QSize(18, 18))
        self.theta_range_btn.setToolTip(
            "Предпросмотр диапазона θ в 3D · данные и XY-проекция не изменяются"
        )
        self.theta_range_btn.setAccessibleName("Предпросмотр диапазона θ в 3D")
        self.theta_range_menu = QMenu(self.theta_range_btn)
        self.theta_range_menu.setObjectName("VTKThetaDisplayRangeMenu")
        self.theta_range_menu.setToolTipsVisible(True)
        self.theta_range_menu.setStyleSheet(self._popup_button_stylesheet())
        self._build_theta_range_menu()
        self.theta_range_btn.setMenu(self.theta_range_menu)
        row.addWidget(self.theta_range_btn)

        # Compatibility aliases retained for the existing desktop viewer API.
        self.axes_action = self.axes_button
        self.perspective_action = self.perspective_button
        self.cursor_line_action = self.cursor_line_button
        # Created when the cursor's independent settings popup is attached.
        self.cursor_line_color_action = None

        row.addStretch(1)
        self.theme_btn = QToolButton(self)
        self.theme_btn.setObjectName("vtkThemeToggle")
        self.theme_btn.setAutoRaise(True)
        self.theme_btn.setToolTip("Переключить тему viewer")
        self.theme_btn.setAccessibleName("Переключить тему viewer")
        self.theme_btn.clicked.connect(self._toggle_theme)
        row.addWidget(self.theme_btn)

        self.set_cursor_line_color(self._cursor_line_color)
        self.set_roi_preview_color(self._roi_preview_color)
        self._set_icon_sizes()
        self.apply_theme("light")

    # ------------------------------------------------------------------
    # Shared toolbar primitives
    # ------------------------------------------------------------------
    def _toolbar_button(self, icon_name: str, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setAutoRaise(True)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setIconSize(QSize(self._icon_size(), self._icon_size()))
        button.setIcon(svg_icon(ICON_DIR / icon_name, widget=button))
        return button

    def _icon_button(
        self,
        parent: QWidget,
        icon_name: str,
        tooltip: str,
        *,
        checkable: bool = False,
        size: int = 19,
    ) -> QToolButton:
        button = QToolButton(parent)
        button.setAutoRaise(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setFixedSize(30, 30)
        button.setIconSize(QSize(size, size))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setCheckable(checkable)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    @staticmethod
    def _popup_button_stylesheet() -> str:
        return """
        QMenu { padding: 2px; }
        QWidget#VTKCameraCommandStrip QToolButton,
        QWidget#VTKOrientationCommandStrip QToolButton,
        QWidget#VTKOverlayIconRow QToolButton {
            min-width: 27px; min-height: 27px; max-width: 27px; max-height: 27px;
            padding: 1px; margin: 0; border: 1px solid transparent;
            border-radius: 3px; background: transparent; color: palette(button-text);
        }
        QWidget#VTKCameraCommandStrip QToolButton:hover,
        QWidget#VTKOrientationCommandStrip QToolButton:hover,
        QWidget#VTKOverlayIconRow QToolButton:hover {
            background: palette(midlight); border-color: palette(mid);
        }
        QWidget#VTKCameraCommandStrip QToolButton:checked,
        QWidget#VTKOrientationCommandStrip QToolButton:checked,
        QWidget#VTKOverlayIconRow QToolButton:checked {
            background: palette(highlight); color: palette(highlighted-text);
            border-color: palette(highlight);
        }
        QWidget#VTKOverlayIconRow QToolButton[settingsActive="true"] {
            background: palette(alternate-base); border-color: palette(highlight);
        }
        QWidget#VTKOverlaySettingsPanel {
            background: palette(window); color: palette(text);
        }
        QWidget#VTKOverlaySettingsPanel QLabel {
            color: palette(text); font-size: 9px; padding: 0;
        }
        QWidget#VTKOverlaySettingsPanel QDoubleSpinBox,
        QWidget#VTKOverlaySettingsPanel QComboBox {
            min-height: 20px; max-height: 21px; padding: 0 2px; font-size: 9px;
        }
        QWidget#VTKOverlaySettingsPanel QSlider {
            min-height: 15px; max-height: 16px;
        }
        QWidget#VTKOverlaySettingsPanel QToolButton#VTKCursorColorButton {
            min-width: 25px; max-width: 25px; min-height: 19px; max-height: 19px;
            padding: 0; border: 1px solid palette(mid); border-radius: 3px;
        }
        QWidget#VTKAxisScalePanel {
            background: palette(window); color: palette(text);
        }
        QWidget#VTKAxisScalePanel QLabel {
            color: palette(text); font-size: 10px; padding: 0;
        }
        QWidget#VTKAxisScalePanel QDoubleSpinBox {
            min-height: 20px; max-height: 20px; min-width: 56px;
            padding: 0 2px; font-size: 10px;
        }
        QWidget#VTKAxisScalePanel QPushButton {
            min-height: 21px; max-height: 21px; padding: 0 8px;
            border: 1px solid palette(mid); border-radius: 3px;
            background: palette(button); color: palette(button-text); font-size: 10px;
        }
        QWidget#VTKAxisScalePanel QPushButton:hover {
            background: palette(midlight); border-color: palette(highlight);
        }
        QWidget#VTKThetaRangePanel {
            background: palette(window); color: palette(text);
        }
        QWidget#VTKThetaRangePanel QLabel {
            color: palette(text); background: transparent; border: 0; padding: 0;
            font-size: 9px;
        }
        QWidget#VTKThetaRangePanel QLabel#VTKThetaRangeTitle {
            font-size: 8px; font-weight: 700; letter-spacing: 0.35px;
        }
        QWidget#VTKThetaRangePanel QPushButton {
            min-height: 20px; max-height: 20px; padding: 0 7px;
            border: 1px solid palette(mid); border-radius: 3px;
            background: palette(button); color: palette(button-text); font-size: 9px;
        }
        QWidget#VTKThetaRangePanel QPushButton:hover {
            background: palette(midlight); border-color: palette(highlight);
        }
        """

    @staticmethod
    def _apply_popup_surface_theme(menu: QMenu, theme: str, *, padding: int = 0) -> None:
        """Match the outer native popup surface to its embedded settings panel."""
        c = theme_colors(theme)
        name = menu.objectName().replace('"', "")
        menu.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        menu.setStyleSheet(menu.styleSheet() + f"""
            QMenu#{name} {{
                background: {c['window']}; color: {c['text']};
                border: 1px solid {c['border']}; padding: {int(padding)}px;
            }}
        """)

    @staticmethod
    def _apply_popup_panel_theme(panel: QWidget, theme: str) -> None:
        """Paint custom QWidgetAction panels explicitly, not via app defaults.

        QMenu and its QWidgetAction child are separate native popup surfaces on
        Windows. Explicit role colours prevent the menu frame using one theme
        while spin boxes/combos in the embedded panel retain another.
        """
        c = theme_colors(theme)
        apply_widget_palette(panel, theme)
        panel.setAutoFillBackground(True)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        name = panel.objectName().replace('"', "")
        if name in {"VTKCameraCommandStrip", "VTKOrientationCommandStrip", "VTKOverlayIconRow"}:
            panel.setStyleSheet(f"""
                QWidget#{name} {{ background: transparent; color: {c['text']}; border: 0; }}
            """)
            return
        panel.setStyleSheet(f"""
            QWidget#{name} {{
                background: {c['window']}; color: {c['text']};
                border: 0;
            }}
            QWidget#{name} QWidget {{ color: {c['text']}; }}
            QWidget#{name} QLabel {{ background: transparent; color: {c['text']}; border: 0; }}
            QWidget#{name} QDoubleSpinBox,
            QWidget#{name} QSpinBox,
            QWidget#{name} QComboBox,
            QWidget#{name} QLineEdit {{
                background: {c['base']}; color: {c['text']};
                border: 1px solid {c['border']}; border-radius: 3px;
                selection-background-color: {c['highlight']};
                selection-color: {c['highlight_text']};
            }}
            QWidget#{name} QComboBox QAbstractItemView {{
                background: {c['base']}; color: {c['text']};
                border: 1px solid {c['border']};
                selection-background-color: {c['highlight']};
                selection-color: {c['highlight_text']};
            }}
            QWidget#{name} QPushButton {{
                background: {c['button']}; color: {c['button_text']};
                border: 1px solid {c['border']}; border-radius: 3px;
            }}
            QWidget#{name} QPushButton:hover {{
                background: {c['hover']}; border-color: {c['highlight']};
            }}
            QWidget#{name} QCheckBox,
            QWidget#{name} QRadioButton {{
                color: {c['text']}; spacing: 5px;
            }}
            QWidget#{name} QCheckBox::indicator,
            QWidget#{name} QRadioButton::indicator {{
                width: 12px; height: 12px;
                background: {c['base']}; border: 1px solid {c['border']};
                border-radius: 3px;
            }}
            QWidget#{name} QCheckBox::indicator:checked,
            QWidget#{name} QRadioButton::indicator:checked {{
                background: {c['highlight']}; border-color: {c['highlight']};
            }}
            QWidget#{name} QCheckBox::indicator:disabled,
            QWidget#{name} QRadioButton::indicator:disabled {{
                background: {c['window']}; border-color: {c['muted']};
            }}
            QWidget#VTKAxisScalePanel QToolButton {{
                background: {c['button']}; color: {c['button_text']};
                border: 1px solid {c['border']}; border-radius: 3px;
                padding: 0; font-size: 15px;
            }}
            QWidget#VTKAxisScalePanel QToolButton:hover {{
                background: {c['hover']}; border-color: {c['highlight']};
            }}
            QWidget#VTKAxisScalePanel QToolButton#VTKAxisScaleCloseButton {{
                font-size: 10px; font-weight: 500; padding: 0;
            }}
            QWidget#VTKAxisScalePanel QLabel#VTKAxisScaleTitle {{
                font-size: 8px; font-weight: 700; letter-spacing: 0.4px;
            }}
            QWidget#VTKThetaRangePanel QLabel#VTKThetaRangeTitle {{
                font-size: 8px; font-weight: 700; letter-spacing: 0.45px;
            }}
            QWidget#VTKThetaRangePanel QLabel#VTKThetaRangeValue {{
                font-size: 10px; font-weight: 600;
            }}
            QWidget#VTKThetaRangePanel QLabel#VTKThetaRangeEndpoint {{
                color: {c['muted']}; font-size: 8px;
            }}
            QWidget#VTKThetaRangePanel QCheckBox {{
                spacing: 4px; font-size: 9px; color: {c['text']};
            }}
            QWidget#VTKThetaRangePanel QCheckBox::indicator {{
                width: 11px; height: 11px;
            }}
            QWidget#VTKThetaRangePanel QSlider {{
                min-height: 15px; max-height: 16px;
            }}
            QWidget#VTKROIPreviewSettings QComboBox,
            QWidget#VTKROIPreviewSettings QDoubleSpinBox,
            QWidget#VTKROIPreviewSettings QSpinBox {{
                min-height: 20px; max-height: 22px; padding: 0 4px;
                background: {c['base']}; color: {c['text']};
                border: 1px solid {c['border']}; border-radius: 3px;
                selection-background-color: {c['highlight']};
                selection-color: {c['highlight_text']}; font-size: 9px;
            }}
            QWidget#VTKROIPreviewSettings QComboBox QAbstractItemView {{
                background: {c['base']}; color: {c['text']};
                border: 1px solid {c['border']};
                selection-background-color: {c['highlight']};
                selection-color: {c['highlight_text']}; outline: 0;
            }}
            QWidget#VTKROIPreviewSettings QToolButton#VTKROIPreviewColorButton {{
                background: {c['button']}; border: 1px solid {c['border']};
                border-radius: 3px; padding: 1px;
            }}
            QWidget#VTKROIPreviewSettings QToolButton#VTKROIPreviewColorButton:hover {{
                background: {c['hover']}; border-color: {c['highlight']};
            }}
            QWidget#VTKROIPreviewSettings QCheckBox,
            QWidget#VTKROIPreviewSettings QRadioButton {{
                color: {c['text']}; spacing: 5px; font-size: 9px;
            }}
            QWidget#VTKROIPreviewSettings QCheckBox::indicator,
            QWidget#VTKROIPreviewSettings QRadioButton::indicator {{
                width: 12px; height: 12px;
                background: {c['base']}; border: 1px solid {c['border']};
                border-radius: 3px;
            }}
            QWidget#VTKROIPreviewSettings QCheckBox::indicator:checked,
            QWidget#VTKROIPreviewSettings QRadioButton::indicator:checked {{
                background: {c['highlight']}; border-color: {c['highlight']};
            }}
            QWidget#VTKROIPreviewSettings QCheckBox::indicator:disabled,
            QWidget#VTKROIPreviewSettings QRadioButton::indicator:disabled {{
                background: {c['window']}; border-color: {c['muted']};
            }}
        """)

    # ------------------------------------------------------------------
    # Live theta display-range popup
    # ------------------------------------------------------------------
    def _build_theta_range_menu(self) -> None:
        """Compact dual-handle slider for GPU-side theta cropping."""
        panel = QWidget(self.theta_range_menu)
        panel.setObjectName("VTKThetaRangePanel")
        panel.setMinimumWidth(258)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title = QLabel("ОБЛАСТЬ ОТОБРАЖЕНИЯ θ", panel)
        title.setObjectName("VTKThetaRangeTitle")
        title.setToolTip(
            "Предпросмотр в 3D через clipping planes GPU-маппера. "
            "Исходные данные, XY-проекция и другие окна не меняются."
        )
        layout.addWidget(title)

        self.theta_range_value_label = QLabel("θ: —", panel)
        self.theta_range_value_label.setObjectName("VTKThetaRangeValue")
        self.theta_range_value_label.setMinimumWidth(240)
        layout.addWidget(self.theta_range_value_label)

        self.theta_range_slider = QRangeSlider(
            low=0.0,
            high=1.0,
            min_val=0.0,
            max_val=1.0,
            parent=panel,
            orientation=Qt.Orientation.Horizontal,
            minimum_gap=0.01,
        )
        self.theta_range_slider.setObjectName("vtkThetaDisplayRangeSlider")
        self.theta_range_slider.setFixedWidth(238)
        self.theta_range_slider.setToolTip(
            "Ограничить видимую часть объёма по θ. Перемещение ручек не перестраивает данные."
        )
        self.theta_range_slider.setEnabled(False)
        self.theta_range_slider.valueChanged.connect(self._on_theta_range_slider_changed)
        layout.addWidget(self.theta_range_slider, 0, Qt.AlignmentFlag.AlignHCenter)

        endpoints = QHBoxLayout()
        endpoints.setContentsMargins(0, 0, 0, 0)
        endpoints.setSpacing(4)
        self.theta_range_min_label = QLabel("min: —", panel)
        self.theta_range_max_label = QLabel("max: —", panel)
        self.theta_range_min_label.setObjectName("VTKThetaRangeEndpoint")
        self.theta_range_max_label.setObjectName("VTKThetaRangeEndpoint")
        endpoints.addWidget(self.theta_range_min_label)
        endpoints.addStretch(1)
        endpoints.addWidget(self.theta_range_max_label)
        layout.addLayout(endpoints)

        # Optional range markers: a single checkable control and one restrained
        # opacity slider. They affect only two lightweight VTK plane props;
        # the GPU crop and all scientific data remain entirely independent.
        boundary_row = QHBoxLayout()
        boundary_row.setContentsMargins(0, 2, 0, 0)
        boundary_row.setSpacing(6)
        self.theta_boundary_planes_check = QCheckBox("Граничные плоскости", panel)
        self.theta_boundary_planes_check.setObjectName("VTKThetaBoundaryPlanesToggle")
        self.theta_boundary_planes_check.setToolTip(
            "Показать две полупрозрачные плоскости на границах выбранного диапазона θ"
        )
        self.theta_boundary_planes_check.setEnabled(False)
        self.theta_boundary_planes_check.toggled.connect(
            self._on_theta_boundary_planes_toggled
        )
        boundary_row.addWidget(self.theta_boundary_planes_check)
        boundary_row.addStretch(1)
        self.theta_boundary_opacity_value = QLabel("12 %", panel)
        self.theta_boundary_opacity_value.setObjectName("VTKThetaBoundaryOpacityValue")
        boundary_row.addWidget(self.theta_boundary_opacity_value)
        layout.addLayout(boundary_row)

        self.theta_boundary_opacity_slider = QSlider(
            Qt.Orientation.Horizontal, panel
        )
        self.theta_boundary_opacity_slider.setObjectName(
            "VTKThetaBoundaryOpacitySlider"
        )
        self.theta_boundary_opacity_slider.setRange(2, 25)
        self.theta_boundary_opacity_slider.setValue(12)
        self.theta_boundary_opacity_slider.setFixedWidth(238)
        self.theta_boundary_opacity_slider.setToolTip(
            "Прозрачность заливки граничных плоскостей: 2–25 %. Контур остаётся чётким."
        )
        self.theta_boundary_opacity_slider.setEnabled(False)
        self.theta_boundary_opacity_slider.valueChanged.connect(
            self._on_theta_boundary_plane_opacity_changed
        )
        layout.addWidget(
            self.theta_boundary_opacity_slider, 0, Qt.AlignmentFlag.AlignHCenter
        )

        self.theta_range_reset_button = QPushButton("Полный диапазон", panel)
        self.theta_range_reset_button.setObjectName("VTKThetaRangeReset")
        self.theta_range_reset_button.setToolTip("Вернуть отображение всего диапазона θ")
        self.theta_range_reset_button.clicked.connect(self.reset_theta_display_range)
        layout.addWidget(self.theta_range_reset_button, 0, Qt.AlignmentFlag.AlignRight)

        action = QWidgetAction(self.theta_range_menu)
        action.setDefaultWidget(panel)
        self.theta_range_menu.addAction(action)
        self.theta_range_panel = panel
        self._theta_range_action = action

    def _on_theta_boundary_planes_toggled(self, enabled: bool) -> None:
        active = bool(enabled)
        self.theta_boundary_opacity_slider.setEnabled(active)
        self.theta_boundary_opacity_value.setEnabled(active)
        self.theta_boundary_planes_toggled.emit(active)

    def _on_theta_boundary_plane_opacity_changed(self, value: int) -> None:
        percent = int(value)
        self.theta_boundary_opacity_value.setText(f"{percent} %")
        self.theta_boundary_plane_opacity_changed.emit(percent / 100.0)

    def _replace_theta_range_slider(self, theta_values: list[float]) -> None:
        """Recreate the shared slider with the physical axis' exact limits."""
        differences = [
            right - left for left, right in zip(theta_values, theta_values[1:])
        ]
        minimum_gap = min(differences)
        old_slider = self.theta_range_slider
        new_slider = QRangeSlider(
            low=float(theta_values[0]),
            high=float(theta_values[-1]),
            min_val=float(theta_values[0]),
            max_val=float(theta_values[-1]),
            parent=self.theta_range_panel,
            orientation=Qt.Orientation.Horizontal,
            minimum_gap=float(minimum_gap),
        )
        new_slider.setObjectName("vtkThetaDisplayRangeSlider")
        new_slider.setFixedWidth(238)
        new_slider.setToolTip(
            "Ограничить видимую часть объёма по θ. Значения привязываются к реальным отсчётам."
        )
        new_slider.valueChanged.connect(self._on_theta_range_slider_changed)
        self.theta_range_panel.layout().replaceWidget(old_slider, new_slider)
        old_slider.hide()
        old_slider.deleteLater()
        self.theta_range_slider = new_slider
        apply_widget_palette(self.theta_range_slider, self._theme)

    def set_theta_axis_values(self, values) -> None:
        """Configure the preview slider from an ascending physical theta axis."""
        if values is None:
            self._theta_axis_values = []
            self._theta_range_indices = None
            self.theta_range_slider.setEnabled(False)
            self.theta_range_reset_button.setEnabled(False)
            self.theta_boundary_planes_check.blockSignals(True)
            self.theta_boundary_planes_check.setChecked(False)
            self.theta_boundary_planes_check.blockSignals(False)
            self.theta_boundary_planes_check.setEnabled(False)
            self.theta_boundary_opacity_slider.blockSignals(True)
            self.theta_boundary_opacity_slider.setValue(12)
            self.theta_boundary_opacity_slider.blockSignals(False)
            self.theta_boundary_opacity_slider.setEnabled(False)
            self.theta_boundary_opacity_value.setText("12 %")
            self.theta_boundary_opacity_value.setEnabled(False)
            self.theta_range_value_label.setText("θ: —")
            self.theta_range_min_label.setText("min: —")
            self.theta_range_max_label.setText("max: —")
            return

        theta_values = [float(value) for value in values]
        if len(theta_values) < 2:
            raise ValueError("Для предпросмотра диапазона θ нужны минимум два отсчёта.")
        if any(not math.isfinite(value) for value in theta_values):
            raise ValueError("Ось θ содержит нечисловые или бесконечные значения.")
        if any(right <= left for left, right in zip(theta_values, theta_values[1:])):
            raise ValueError("Ось θ должна быть строго возрастающей.")

        self._theta_axis_values = theta_values
        self._replace_theta_range_slider(theta_values)
        self._theta_range_indices = (0, len(theta_values) - 1)
        self.theta_range_slider.setEnabled(len(theta_values) > 2)
        self.theta_range_reset_button.setEnabled(len(theta_values) > 2)
        self.theta_boundary_planes_check.setEnabled(len(theta_values) > 2)
        show_boundary_options = bool(
            len(theta_values) > 2 and self.theta_boundary_planes_check.isChecked()
        )
        self.theta_boundary_opacity_slider.setEnabled(show_boundary_options)
        self.theta_boundary_opacity_value.setEnabled(show_boundary_options)
        self._update_theta_range_labels(0, len(theta_values) - 1)

    def _update_theta_range_labels(self, low_index: int, high_index: int) -> None:
        values = self._theta_axis_values
        if not values:
            return
        low = values[int(low_index)]
        high = values[int(high_index)]
        self.theta_range_value_label.setText(f"θ: {low:.7g}°  —  {high:.7g}°")
        self.theta_range_min_label.setText(f"min: {values[0]:.7g}°")
        self.theta_range_max_label.setText(f"max: {values[-1]:.7g}°")

    def _on_theta_range_slider_changed(self, low_value: float, high_value: float) -> None:
        values = self._theta_axis_values
        if len(values) < 2:
            return
        low_index = max(0, min(len(values) - 2, bisect_left(values, float(low_value))))
        high_index = max(
            low_index + 1,
            min(len(values) - 1, bisect_right(values, float(high_value)) - 1),
        )
        snapped_low = values[low_index]
        snapped_high = values[high_index]
        # Snap both handles to actual sample coordinates so the displayed
        # interval is reproducible and can be committed later by the data layer.
        self.theta_range_slider.blockSignals(True)
        self.theta_range_slider.setValues(snapped_low, snapped_high)
        self.theta_range_slider.blockSignals(False)
        self._theta_range_indices = (low_index, high_index)
        self._update_theta_range_labels(low_index, high_index)
        self.theta_display_range_changed.emit(low_index, high_index)

    def reset_theta_display_range(self) -> None:
        values = self._theta_axis_values
        if len(values) < 2:
            return
        self.theta_range_slider.blockSignals(True)
        self.theta_range_slider.setValues(values[0], values[-1])
        self.theta_range_slider.blockSignals(False)
        self._theta_range_indices = (0, len(values) - 1)
        self._update_theta_range_labels(0, len(values) - 1)
        self.theta_display_range_changed.emit(0, len(values) - 1)

    def set_theta_display_range_indices(self, low_index: int, high_index: int) -> None:
        values = self._theta_axis_values
        if len(values) < 2:
            return
        low = int(low_index)
        high = int(high_index)
        if not (0 <= low < high < len(values)):
            raise ValueError("Индексы предпросмотра θ должны удовлетворять 0 ≤ low < high < N.")
        self._theta_range_indices = (low, high)
        self.theta_range_slider.blockSignals(True)
        self.theta_range_slider.setValues(values[low], values[high])
        self.theta_range_slider.blockSignals(False)
        self._update_theta_range_labels(low, high)

    # ------------------------------------------------------------------
    # Camera / axes popup
    # ------------------------------------------------------------------
    def _build_orientation_menu(self) -> None:
        """Compact four-command strip for standard camera orientations."""
        self._orientation_panel = QWidget(self.orientation_menu)
        self._orientation_panel.setObjectName("VTKOrientationCommandStrip")
        row = QHBoxLayout(self._orientation_panel)
        # Match the ROI selector flyout: 1 px margins and 1 px between buttons.
        row.setContentsMargins(1, 1, 1, 1)
        row.setSpacing(1)

        self.perspective_button = QToolButton(self._orientation_panel)
        self.perspective_button.setAutoRaise(True)
        self.perspective_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.perspective_button.setFixedSize(27, 27)
        self.perspective_button.setIconSize(QSize(17, 17))
        self.perspective_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.perspective_button.setToolTip(
            "Стандартный перспективный изометрический ракурс"
        )
        self.perspective_button.setAccessibleName(
            "Установить стандартный перспективный изометрический ракурс"
        )
        self.perspective_button.clicked.connect(self._emit_perspective_view_requested)
        row.addWidget(self.perspective_button)

        self.camera_axis_buttons: dict[str, QToolButton] = {}
        for axis in ("x", "y", "z"):
            button = _DirectionalAxisButton(axis, self._orientation_panel)
            button.setAutoRaise(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setFixedSize(27, 27)
            button.setIconSize(QSize(17, 17))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(
                f"Вид вдоль оси {axis.upper()} · ЛКМ: +{axis.upper()}, ПКМ: −{axis.upper()}"
            )
            button.setAccessibleName(f"Вид вдоль оси {axis.upper()}")
            button.direction_requested.connect(self._emit_camera_axis_direction)
            self.camera_axis_buttons[axis] = button
            row.addWidget(button)

        action = QWidgetAction(self.orientation_menu)
        action.setDefaultWidget(self._orientation_panel)
        self.orientation_menu.addAction(action)

    def _build_camera_menu(self) -> None:
        """Camera display controls only; orientation commands live next to Fit."""
        self._camera_panel = QWidget(self.camera_menu)
        self._camera_panel.setObjectName("VTKCameraCommandStrip")
        strip = QHBoxLayout(self._camera_panel)
        strip.setContentsMargins(1, 1, 1, 1)
        strip.setSpacing(1)

        self.axes_button = self._icon_button(
            self._camera_panel,
            "vtk_axes_info.svg",
            "Показывать/скрывать индикатор ориентации X/Y/Z в левом нижнем углу",
            checkable=True,
        )
        self.axes_button.setFixedSize(27, 27)
        self.axes_button.setIconSize(QSize(17, 17))
        self.axes_button.setChecked(True)
        self.axes_button.toggled.connect(self.axes_toggled.emit)
        strip.addWidget(self.axes_button)

        self.bounding_box_button = self._icon_button(
            self._camera_panel,
            "vtk_bounding_box.svg",
            "Показывать/скрывать каркас ограничивающего параллелепипеда",
            checkable=True,
        )
        self.bounding_box_button.setFixedSize(27, 27)
        self.bounding_box_button.setIconSize(QSize(17, 17))
        self.bounding_box_button.setChecked(True)
        self.bounding_box_button.toggled.connect(self.bounding_box_toggled.emit)
        strip.addWidget(self.bounding_box_button)

        self.axis_ticks_button = SmartDropdownButton(self._camera_panel)
        self.axis_ticks_button.setAutoRaise(True)
        self.axis_ticks_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.axis_ticks_button.setFixedSize(27, 27)
        self.axis_ticks_button.setIconSize(QSize(17, 17))
        self.axis_ticks_button.setToolTip("Настройки делений и координатной сетки")
        self.axis_ticks_button.setAccessibleName("Настройки делений и координатной сетки")
        self.axis_ticks_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.axis_display_menu = QMenu(self.axis_ticks_button)
        self.axis_display_menu.setObjectName("VTKAxisDisplayMenu")
        self.axis_ticks_action = self.axis_display_menu.addAction("Деления и числовые подписи осей")
        self.axis_ticks_action.setCheckable(True)
        self.axis_ticks_action.setChecked(True)
        self.axis_ticks_action.toggled.connect(self.axis_ticks_toggled.emit)
        self.grid_action = self.axis_display_menu.addAction("Координатная сетка")
        self.grid_action.setCheckable(True)
        self.grid_action.setChecked(True)
        self.grid_action.toggled.connect(self.grid_toggled.emit)
        self.axis_ticks_button.setMenu(self.axis_display_menu)
        strip.addWidget(self.axis_ticks_button)

        self.axis_scale_btn = SmartDropdownButton(self._camera_panel)
        self.axis_scale_btn.setAutoRaise(True)
        self.axis_scale_btn.setFixedSize(27, 27)
        self.axis_scale_btn.setIconSize(QSize(18, 18))
        self.axis_scale_btn.setToolTip("Масштаб геометрии по X, Y и θ")
        self.axis_scale_btn.setAccessibleName("Масштаб координатных осей")
        self.axis_scale_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.axis_scale_menu = QMenu(self.camera_menu)
        self.axis_scale_menu.setObjectName("VTKAxisScaleMenu")
        self.axis_scale_menu.setStyleSheet(self._popup_button_stylesheet())
        self._build_axis_scale_menu()
        self.axis_scale_btn.setMenu(self.axis_scale_menu)
        strip.addWidget(self.axis_scale_btn)

        action = QWidgetAction(self.camera_menu)
        action.setDefaultWidget(self._camera_panel)
        self.camera_menu.addAction(action)

    def _emit_camera_axis_direction(self, axis: str, polarity: int) -> None:
        axis = str(axis).lower()
        polarity = 1 if int(polarity) >= 0 else -1
        self.camera_axis_direction_requested.emit(axis, polarity)
        if polarity > 0:
            self.camera_axis_requested.emit(axis)

    def _emit_perspective_view_requested(self) -> None:
        self._perspective_enabled = True
        self.perspective_view_requested.emit()
        self.perspective_toggled.emit(True)

    def set_perspective_enabled(self, enabled: bool) -> None:
        self._perspective_enabled = bool(enabled)

    @property
    def perspective_enabled(self) -> bool:
        return bool(self._perspective_enabled)

    # ------------------------------------------------------------------
    # Axis-scale popup (display-only affine scaling)
    # ------------------------------------------------------------------
    def _build_axis_scale_menu(self) -> None:
        """Build a compact transactional editor for display-only axis scale."""
        panel = QWidget(self.axis_scale_menu)
        self.axis_scale_panel = panel
        panel.setObjectName("VTKAxisScalePanel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(7, 5, 7, 6)
        outer.setSpacing(5)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(5)
        self.axis_scale_title = QLabel("МАСШТАБ ГЕОМЕТРИИ", panel)
        self.axis_scale_title.setObjectName("VTKAxisScaleTitle")
        header.addWidget(self.axis_scale_title)
        header.addStretch(1)
        self.axis_scale_close_btn = QToolButton(panel)
        self.axis_scale_close_btn.setObjectName("VTKAxisScaleCloseButton")
        self.axis_scale_close_btn.setText("×")
        self.axis_scale_close_btn.setFixedSize(18, 18)
        self.axis_scale_close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.axis_scale_close_btn.setToolTip("Отмена: закрыть без применения изменений")
        self.axis_scale_close_btn.setAccessibleName("Отменить изменения масштаба и закрыть окно")
        self.axis_scale_close_btn.clicked.connect(self.axis_scale_menu.close)
        header.addWidget(self.axis_scale_close_btn)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)
        self.axis_scale_spins: dict[str, QDoubleSpinBox] = {}
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(3)
        for row, (axis, label_text) in enumerate((
            ("x", "X"), ("y", "Y"), ("theta", "θ")
        )):
            caption = QLabel(label_text, panel)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption.setToolTip(f"Коэффициент отображаемого масштаба {axis.upper()}")
            spin = QDoubleSpinBox(panel)
            spin.setRange(0.10, 10.00)
            spin.setDecimals(2)
            spin.setSingleStep(0.10)
            spin.setValue(1.00)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.setKeyboardTracking(False)
            spin.setFixedWidth(64)
            spin.setToolTip(
                "Масштабирует только координатную геометрию относительно середины оси. "
                "Исходные координаты и массив интенсивностей не изменяются."
            )
            spin.valueChanged.connect(self._refresh_axis_scale_preview)
            grid.addWidget(caption, row, 0)
            grid.addWidget(spin, row, 1)
            self.axis_scale_spins[axis] = spin
        controls.addLayout(grid)
        body.addLayout(controls)

        self.axis_scale_preview = _AxisScalePreview(panel)
        self.axis_scale_preview.setObjectName("VTKAxisScalePreview")
        self.axis_scale_preview.setToolTip(
            "Предпросмотр пропорций ячейки: X — красный, Y — зелёный, θ — синий."
        )
        body.addWidget(self.axis_scale_preview, 0, Qt.AlignmentFlag.AlignCenter)
        outer.addLayout(body)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 1, 0, 0)
        footer.setSpacing(5)
        self.axis_scale_reset_btn = QToolButton(panel)
        self.axis_scale_reset_btn.setObjectName("VTKAxisScaleResetButton")
        self.axis_scale_reset_btn.setFixedSize(24, 23)
        self.axis_scale_reset_btn.setIconSize(QSize(15, 15))
        self.axis_scale_reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.axis_scale_reset_btn.setToolTip(
            "Сбросить коэффициенты к 1,00× (только черновик; нажмите «Применить» для фиксации)"
        )
        self.axis_scale_reset_btn.setAccessibleName("Сбросить коэффициенты масштаба к единице")
        self.axis_scale_reset_btn.clicked.connect(self._reset_axis_scale_edit)
        footer.addWidget(self.axis_scale_reset_btn)
        footer.addStretch(1)
        self.axis_scale_apply_btn = QPushButton("Применить", panel)
        self.axis_scale_apply_btn.setObjectName("VTKAxisScaleApplyButton")
        self.axis_scale_apply_btn.setFixedWidth(82)
        self.axis_scale_apply_btn.setFixedHeight(24)
        self.axis_scale_apply_btn.setDefault(True)
        self.axis_scale_apply_btn.clicked.connect(self._apply_axis_scale_edit)
        footer.addWidget(self.axis_scale_apply_btn)
        outer.addLayout(footer)

        action = QWidgetAction(self.axis_scale_menu)
        action.setDefaultWidget(panel)
        self.axis_scale_menu.addAction(action)
        # Esc, clicking outside, or the close button discards only the draft.
        self.axis_scale_menu.aboutToHide.connect(self._restore_axis_scale_editor)
        self._refresh_axis_scale_preview()

    def _refresh_axis_scale_preview(self, *_args) -> None:
        if not hasattr(self, "axis_scale_preview"):
            return
        values = tuple(
            float(self.axis_scale_spins[key].value())
            for key in ("x", "y", "theta")
        )
        self.axis_scale_preview.set_scales(*values)

    @property
    def axis_scales(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self._applied_axis_scales)

    def set_axis_scales(self, x: float, y: float, theta: float) -> None:
        """Synchronize the editor and committed scale factors from the scene."""
        values = self._validate_axis_scales((x, y, theta))
        self._axis_scales = values
        self._applied_axis_scales = values
        for key, value in zip(("x", "y", "theta"), values):
            spin = self.axis_scale_spins[key]
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._refresh_axis_scale_preview()

    @staticmethod
    def _validate_axis_scales(values) -> tuple[float, float, float]:
        result = tuple(float(value) for value in values)
        if len(result) != 3 or any(not (0.1 <= value <= 10.0) for value in result):
            raise ValueError("Коэффициенты масштабирования X/Y/θ должны лежать в [0.1, 10].")
        return result

    def _read_axis_scale_editor(self) -> tuple[float, float, float]:
        return self._validate_axis_scales(
            tuple(self.axis_scale_spins[key].value() for key in ("x", "y", "theta"))
        )

    def _restore_axis_scale_editor(self) -> None:
        # Closing via Esc, outside click, or the window manager is a cancel.
        for key, value in zip(("x", "y", "theta"), self._applied_axis_scales):
            spin = self.axis_scale_spins[key]
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._axis_scales = self._applied_axis_scales
        self._refresh_axis_scale_preview()

    def _reset_axis_scale_edit(self) -> None:
        """Reset the uncommitted editor draft to neutral 1× proportions."""
        for spin in self.axis_scale_spins.values():
            spin.setValue(1.0)
        self._refresh_axis_scale_preview()

    def _apply_axis_scale_edit(self) -> None:
        """Request scene update; host commits the editor only after VTK succeeds."""
        try:
            values = self._read_axis_scale_editor()
        except (TypeError, ValueError):
            self._restore_axis_scale_editor()
            return
        # Do not update _applied_axis_scales and do not close the popup here.
        # desktop_viewer calls set_axis_scales() only after the scene transaction
        # succeeds; on error it restores the last committed values.
        self.axis_scales_apply_requested.emit(*values)

    # ------------------------------------------------------------------
    # Additional scene objects — persistent icon rail + independent popups
    # ------------------------------------------------------------------
    def _build_overlays_menu(self) -> None:
        """Show three adjacent, equally sized scene-object tool buttons."""
        self._overlays_panel = QWidget(self.overlays_menu)
        self._overlays_panel.setObjectName("VTKOverlayIconRow")
        row = QHBoxLayout(self._overlays_panel)
        row.setContentsMargins(1, 1, 1, 1)
        row.setSpacing(1)
        self._overlay_icon_row = row

        child_variant = int(self.SCENE_OBJECTS_CHILD_ICON_VARIANT)
        if child_variant == 0:
            # Exact icon files used by the user's project before the Round 6
            # proposal. Keep their original multicolour treatment unchanged.
            child_icons = {
                "cursor": "vtk_cursor_line_1.svg",
                "theta": "vtk_theta_plane_1.svg",
                "projection": "vtk_xy_projection.svg",
            }
            child_monochrome = False
        elif 1 <= child_variant <= 3:
            child_icons = {
                "cursor": f"vtk_cursor_line_set_{child_variant}.svg",
                "theta": f"vtk_theta_plane_set_{child_variant}.svg",
                "projection": f"vtk_xy_projection_set_{child_variant}.svg",
            }
            child_monochrome = True
        else:
            raise ValueError(
                "SCENE_OBJECTS_CHILD_ICON_VARIANT must be 0 (legacy) or 1..3."
            )

        self.cursor_line_button = self.register_overlay_tool(
            "cursor",
            child_icons["cursor"],
            "Линия положения курсора 2D ↔ 3D · ЛКМ — включить/выключить, ПКМ — параметры",
            settings_title="ЛИНИЯ КУРСОРА",
            monochrome=child_monochrome,
            icon_size=16,
        )
        self.theta_slice_button = self.register_overlay_tool(
            "theta",
            child_icons["theta"],
            "Сечение по θ · ЛКМ — включить/выключить, ПКМ — параметры",
            settings_title="СЕЧЕНИЕ ПО θ",
            monochrome=child_monochrome,
            icon_size=16,
        )
        self.projection_button = self.register_overlay_tool(
            "projection",
            child_icons["projection"],
            "XY-проекция объёма · ЛКМ — показать/скрыть, ПКМ — параметры",
            settings_title="XY-ПРОЕКЦИЯ",
            monochrome=child_monochrome,
            icon_size=16,
        )
        roi_icon_variant = int(self.ROI_PREVIEW_ICON_VARIANT)
        if roi_icon_variant not in (1, 2, 3):
            raise ValueError("ROI_PREVIEW_ICON_VARIANT должен быть 1, 2 или 3.")
        self.roi_preview_button = self.register_overlay_tool(
            "roi_preview",
            f"vtk_roi_preview_{roi_icon_variant}.svg",
            "Предпросмотр ROI и выделение соответствующей части облака · ЛКМ — показать/скрыть, ПКМ — параметры",
            settings_title="ПРЕДПРОСМОТР ROI",
            monochrome=True,
            icon_size=18,
        )
        action = QWidgetAction(self.overlays_menu)
        action.setDefaultWidget(self._overlays_panel)
        self.overlays_menu.addAction(action)

    def register_overlay_tool(
        self,
        key: str,
        icon_name: str,
        tooltip: str,
        *,
        settings_title: str,
        monochrome: bool = True,
        icon_size: int = 23,
        checkable: bool = True,
    ) -> QToolButton:
        tool_key = str(key).strip()
        if not tool_key:
            raise ValueError("Ключ overlay-инструмента не может быть пустым.")
        if tool_key in self._overlay_buttons:
            raise ValueError(f"Overlay-инструмент {tool_key!r} уже зарегистрирован.")
        if not str(icon_name).lower().endswith(".svg"):
            raise ValueError("Для overlay-инструмента требуется имя SVG-иконки.")

        button = _OverlayToolButton(self._overlays_panel)
        button.setObjectName(f"vtkOverlay_{tool_key}")
        button.setAutoRaise(True)
        button.setCheckable(bool(checkable))
        button.setFixedSize(27, 27)
        button.setIconSize(QSize(int(icon_size), int(icon_size)))
        button.setToolTip(str(tooltip))
        button.setAccessibleName(str(tooltip).split("·", 1)[0].strip())
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.settings_requested.connect(
            lambda selected_key=tool_key: self._show_overlay_settings(selected_key)
        )
        button.primary_pressed.connect(self._close_overlay_settings)
        button.toggled.connect(
            lambda enabled, selected_key=tool_key:
                self._emit_overlay_toggled(selected_key, bool(enabled))
        )
        self._overlay_icon_row.addWidget(button)
        self._overlay_buttons[tool_key] = button
        self._overlay_tool_specs[tool_key] = {
            "icon_name": str(icon_name),
            "tooltip": str(tooltip),
            "settings_title": str(settings_title),
            "monochrome": bool(monochrome),
            "icon_size": int(icon_size),
        }
        apply_widget_palette(button, self._theme)
        button.setIcon(
            svg_icon(
                ICON_DIR / str(icon_name),
                color=theme_colors(self._theme)["text"],
                widget=button,
                monochrome=bool(monochrome),
            )
        )
        return button

    def _emit_overlay_toggled(self, key: str, enabled: bool) -> None:
        self.overlay_toggled.emit(str(key), bool(enabled))
        if key == "cursor":
            self.cursor_line_toggled.emit(bool(enabled))
        elif key == "theta":
            self.theta_slice_toggled.emit(bool(enabled))
        elif key == "projection":
            self.projection_toggled.emit(bool(enabled))

    def _create_overlay_settings_menu(self, key: str, widget: QWidget) -> QMenu:
        menu = QMenu(self.overlays_menu)
        menu.setObjectName(f"VTKOverlaySettingsMenu_{key}")
        menu.setStyleSheet(self._popup_button_stylesheet())
        panel = QWidget(menu)
        panel.setObjectName("VTKOverlaySettingsPanel")
        root = QVBoxLayout(panel)
        root.setContentsMargins(8, 6, 8, 7)
        root.setSpacing(5)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(5)
        title = QLabel(str(self._overlay_tool_specs[key].get("settings_title") or "ПАРАМЕТРЫ"), panel)
        title.setStyleSheet("font-size: 9px; font-weight: 650; letter-spacing: 0.35px; color: palette(mid);")
        header.addWidget(title)

        if key in ("cursor", "roi_preview"):
            header.addWidget(QLabel("Цвет", panel))
            color_button = QToolButton(panel)
            color_button.setObjectName(
                "VTKCursorColorButton" if key == "cursor" else "VTKROIPreviewColorButton"
            )
            color_button.setAutoRaise(True)
            color_button.setFixedSize(25, 19)
            color_button.setAccessibleName(
                "Цвет линии курсора" if key == "cursor" else "Цвет подсветки облака внутри ROI"
            )
            color_button.clicked.connect(
                self._open_cursor_line_picker if key == "cursor"
                else self._open_roi_preview_color_picker
            )
            header.addWidget(color_button)
            if key == "cursor":
                color_button.setToolTip("Цвет линии курсора")
                self.cursor_color_btn = color_button
                self.cursor_line_color_action = color_button
                self.set_cursor_line_color(self._cursor_line_color)
            else:
                color_button.setToolTip("Цвет выделяемой части облака")
                self.roi_preview_color_btn = color_button
                self.set_roi_preview_color(self._roi_preview_color)
        root.addLayout(header)

        widget.setParent(panel)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        widget.setMinimumSize(0, 0)
        apply_widget_palette(widget, self._theme)
        root.addWidget(widget, 1)
        widget.show()
        if key in ("cursor", "theta"):
            grip_row = QHBoxLayout()
            grip_row.setContentsMargins(0, 0, 0, 0)
            grip_row.setSpacing(0)
            grip_row.addStretch(1)
            grip = _PopupHorizontalResizeGrip(menu, panel, 220, panel)
            grip_row.addWidget(grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
            root.addLayout(grip_row)
        panel.adjustSize()
        base_panel_width = max(220, panel.sizeHint().width())
        panel.setFixedWidth(base_panel_width)
        action = QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)
        menu.adjustSize()
        # Capture the compact, natural popup dimensions exactly once. A drag
        # is temporary: closing the popup restores this baseline, and opening
        # it again always starts from the original layout without stale width.
        base_menu_width = max(base_panel_width, menu.sizeHint().width())
        menu.setFixedWidth(base_menu_width)
        menu.adjustSize()
        base_menu_height = max(1, menu.sizeHint().height())
        self._overlay_popup_base_widths[key] = base_panel_width
        self._overlay_popup_base_menu_widths[key] = base_menu_width
        self._overlay_popup_base_heights[key] = base_menu_height
        if key in ("cursor", "theta"):
            grip.set_base_width(base_menu_width, base_panel_width)
        menu.aboutToHide.connect(lambda selected_key=key: self._on_overlay_settings_hidden(selected_key))
        self._overlay_settings_panels[key] = panel
        return menu

    def set_overlay_settings_widget(
        self,
        key: str,
        widget: QWidget,
        *,
        settings_title: str | None = None,
    ) -> None:
        """Attach one compact, independently-lived options popup per object."""
        tool_key = str(key).strip()
        if tool_key not in self._overlay_buttons:
            raise KeyError(f"Сначала зарегистрируйте overlay-инструмент {tool_key!r}.")
        old = self._overlay_settings.get(tool_key)
        if old is widget and tool_key in self._overlay_settings_menus:
            return
        if settings_title is not None:
            self._overlay_tool_specs[tool_key]["settings_title"] = str(settings_title)
        old_menu = self._overlay_settings_menus.pop(tool_key, None)
        if old_menu is not None:
            old_menu.close()
            old_menu.deleteLater()
        if old is not None and old is not widget:
            old.setParent(None)
        self._overlay_settings[tool_key] = widget
        menu = self._create_overlay_settings_menu(tool_key, widget)
        self._overlay_settings_menus[tool_key] = menu
        apply_popup_theme(menu, self._theme)
        self._apply_popup_surface_theme(menu, self._theme)
        self._apply_popup_panel_theme(self._overlay_settings_panels[tool_key], self._theme)

    def set_overlay_settings_widgets(
        self,
        *,
        cursor_line: QWidget | None = None,
        theta_slice: QWidget | None = None,
        projection: QWidget | None = None,
        roi_preview: QWidget | None = None,
    ) -> None:
        if cursor_line is not None:
            self.set_overlay_settings_widget(
                "cursor", cursor_line, settings_title="ЛИНИЯ КУРСОРА"
            )
        if theta_slice is not None:
            self.set_overlay_settings_widget(
                "theta", theta_slice, settings_title="СЕЧЕНИЕ ПО θ"
            )
        if projection is not None:
            self.set_overlay_settings_widget(
                "projection", projection, settings_title="XY-ПРОЕКЦИЯ"
            )
        if roi_preview is not None:
            self.set_overlay_settings_widget(
                "roi_preview", roi_preview, settings_title="ПРЕДПРОСМОТР ROI"
            )

    def _popup_position_below(self, anchor: QWidget, popup_size: QSize) -> QPoint:
        point = anchor.mapToGlobal(QPoint(0, anchor.height() + 2))
        screen = QApplication.screenAt(point) or QApplication.primaryScreen()
        if screen is None:
            return point
        bounds = screen.availableGeometry()
        x = min(max(point.x(), bounds.left()), max(bounds.left(), bounds.right() - popup_size.width() + 1))
        y = point.y()
        if y + popup_size.height() > bounds.bottom() + 1:
            y = anchor.mapToGlobal(QPoint(0, -popup_size.height() - 2)).y()
        y = min(max(y, bounds.top()), max(bounds.top(), bounds.bottom() - popup_size.height() + 1))
        return QPoint(x, y)

    def _reset_overlay_settings_popup_size(self, key: str) -> None:
        """Restore one resizable popup to its compact opening dimensions."""
        if key not in ("cursor", "theta"):
            return
        menu = self._overlay_settings_menus.get(key)
        panel = self._overlay_settings_panels.get(key)
        panel_width = self._overlay_popup_base_widths.get(key)
        menu_width = self._overlay_popup_base_menu_widths.get(key)
        if menu is None or panel is None or panel_width is None or menu_width is None:
            return
        # setFixedWidth also removes any larger minimum left by a previous drag.
        panel.setFixedWidth(panel_width)
        layout = panel.layout()
        if layout is not None:
            layout.activate()
        panel.adjustSize()
        menu.setFixedWidth(menu_width)
        menu.resize(
            menu_width,
            self._overlay_popup_base_heights.get(key, menu.sizeHint().height()),
        )
    def _show_overlay_settings(self, key: str) -> None:
        if key not in self._overlay_buttons:
            return
        menu = self._overlay_settings_menus.get(key)
        if menu is None:
            return
        if self._active_overlay_settings and self._active_overlay_settings != key:
            if self._color_picker is not None:
                try:
                    self._color_picker.close()
                except RuntimeError:
                    self._color_picker = None
            previous = self._overlay_settings_menus.get(self._active_overlay_settings)
            if previous is not None:
                previous.close()
        self._active_overlay_settings = key
        for item_key, button in self._overlay_buttons.items():
            button.setProperty("settingsActive", item_key == key)
            button.style().unpolish(button)
            button.style().polish(button)
        menu.ensurePolished()
        panel = self._overlay_settings_panels.get(key)
        if panel is not None:
            self._reset_overlay_settings_popup_size(key)
            layout = panel.layout()
            if layout is not None:
                layout.activate()
        menu.adjustSize()
        base_menu_width = self._overlay_popup_base_menu_widths.get(key)
        if base_menu_width is not None and key in ("cursor", "theta"):
            menu.setFixedWidth(base_menu_width)
            menu.resize(
                base_menu_width,
                self._overlay_popup_base_heights.get(key, menu.sizeHint().height()),
            )
        menu.popup(self._popup_position_below(self._overlay_buttons[key], menu.sizeHint()))
        self.overlay_settings_requested.emit(key)

    def _on_overlay_settings_hidden(self, key: str) -> None:
        # A manually dismissed settings popup must not leave its nested color
        # picker visible as an orphan. Applying/closing the picker itself does
        # not hide this parent menu, so the normal Apply path is unaffected.
        if key in ("cursor", "roi_preview") and self._active_color_picker_key == key and self._color_picker is not None:
            try:
                if self._color_picker.isVisible():
                    self._color_picker.close()
            except RuntimeError:
                self._color_picker = None
                self._active_color_picker_key = None
        button = self._overlay_buttons.get(key)
        if button is not None:
            button.setProperty("settingsActive", False)
            button.style().unpolish(button)
            button.style().polish(button)
        if self._active_overlay_settings == key:
            self._active_overlay_settings = None
        self._reset_overlay_settings_popup_size(key)

    def _close_overlay_settings(self) -> None:
        # Closing an options child never closes the primary two-button menu.
        # Any nested palette belongs to the previous child and must not become
        # an orphan popup when switching object or dismissing the parent.
        if self._color_picker is not None:
            try:
                self._color_picker.close()
            except RuntimeError:
                self._color_picker = None
        for menu in tuple(self._overlay_settings_menus.values()):
            if menu.isVisible():
                menu.close()
        self._active_overlay_settings = None
        for button in self._overlay_buttons.values():
            button.setProperty("settingsActive", False)
            button.style().unpolish(button)
            button.style().polish(button)

    # ------------------------------------------------------------------
    # Theme and state synchronization
    # ------------------------------------------------------------------
    @staticmethod
    def _icon_size() -> int:
        app = QApplication.instance()
        raw = app.property("rocMcsIconSize") if app is not None else None
        try:
            return max(12, min(20, int(raw)))
        except (TypeError, ValueError, OverflowError):
            return 16

    def _set_icon_sizes(self) -> None:
        size = self._icon_size()
        self.reset_btn.setIconSize(QSize(size, size))
        self.camera_btn.setIconSize(QSize(18, 18))
        self.orientation_btn.setIconSize(QSize(18, 18))
        self.overlays_btn.setIconSize(QSize(18, 18))
        self.theta_range_btn.setIconSize(QSize(18, 18))
        self.theme_btn.setIconSize(QSize(17, 17))

    def _color_swatch_icon(self, color: str) -> QIcon:
        pixmap = QPixmap(22, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1))
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(3, 3, 16, 10, 2, 2)
        painter.end()
        return QIcon(pixmap)

    @property
    def cursor_line_color(self) -> str:
        return self._cursor_line_color

    def set_cursor_line_color(self, color: str) -> None:
        candidate = QColor(str(color))
        if not candidate.isValid():
            raise ValueError(f"Некорректный цвет линии: {color!r}")
        self._cursor_line_color = candidate.name(QColor.NameFormat.HexRgb).upper()
        if not hasattr(self, "cursor_color_btn"):
            return
        self.cursor_color_btn.setIcon(self._color_swatch_icon(self._cursor_line_color))
        self.cursor_color_btn.setToolTip(
            f"Цвет линии курсора · {self._cursor_line_color}. Нажмите, чтобы выбрать цвет."
        )

    @property
    def roi_preview_color(self) -> str:
        return self._roi_preview_color

    def set_roi_preview_color(self, color: str) -> None:
        candidate = QColor(str(color))
        if not candidate.isValid():
            raise ValueError(f"Некорректный цвет подсветки ROI: {color!r}")
        self._roi_preview_color = candidate.name(QColor.NameFormat.HexRgb).upper()
        button = getattr(self, "roi_preview_color_btn", None)
        if button is not None:
            button.setIcon(self._color_swatch_icon(self._roi_preview_color))
            button.setToolTip(
                f"Цвет выделения облака внутри ROI · {self._roi_preview_color}. Нажмите, чтобы выбрать."
            )

    def _open_cursor_line_picker(self) -> None:
        self._open_overlay_color_picker("cursor")

    def _open_roi_preview_color_picker(self) -> None:
        self._open_overlay_color_picker("roi_preview")

    def _open_overlay_color_picker(self, key: str) -> None:
        try:
            if self._color_picker is not None:
                self._color_picker.close()
        except RuntimeError:
            pass
        if key == "cursor":
            color = self._cursor_line_color
            button = self.cursor_color_btn
            handler = self._on_cursor_color_selected
        elif key == "roi_preview":
            color = self._roi_preview_color
            button = self.roi_preview_color_btn
            handler = self._on_roi_preview_color_selected
        else:
            raise KeyError(f"Неизвестный ключ color picker: {key!r}")
        parent_menu = self._overlay_settings_menus.get(key, self.overlays_menu)
        picker = CompactColorPicker(color, parent_menu, popup=True)
        picker.color_selected.connect(handler)
        picker.destroyed.connect(
            lambda *_args, selected=picker: self._clear_color_picker_reference(selected)
        )
        picker.ensurePolished()
        point = self._popup_position_below(button, picker.sizeHint())
        picker.move(point)
        picker.apply_theme(self._theme)
        self._color_picker = picker
        self._active_color_picker_key = key
        # Qt.Popup is deliberately parented to the options QMenu. Qt therefore
        # treats it as a nested popup: swatch clicks and Apply/Cancel do not
        # dismiss the two-button launcher or its settings popup.
        picker.show()

    def _clear_color_picker_reference(self, picker: CompactColorPicker) -> None:
        if self._color_picker is picker:
            self._color_picker = None
            self._active_color_picker_key = None

    def _on_cursor_color_selected(self, color: str) -> None:
        self.set_cursor_line_color(color)
        self.cursor_line_color_changed.emit(self._cursor_line_color)

    def _on_roi_preview_color_selected(self, color: str) -> None:
        self.set_roi_preview_color(color)
        self.roi_preview_color_changed.emit(self._roi_preview_color)

    def _toggle_theme(self) -> None:
        self.theme_changed.emit("dark" if self._theme == "light" else "light")

    def apply_theme(self, theme: str | None = None) -> None:
        self._theme = apply_widget_palette(self, theme)
        foreground = theme_colors(self._theme)["text"]

        self.reset_btn.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", color=foreground, widget=self.reset_btn))
        theta_range_icon = f"vtk_theta_range_{int(self.THETA_RANGE_ICON_VARIANT)}.svg"
        self.theta_range_btn.setIcon(
            svg_icon(ICON_DIR / theta_range_icon, color=foreground,
                     widget=self.theta_range_btn, monochrome=False)
        )
        camera_icon_name = (
            "vtk_axes_triad_1.svg" if int(self.CAMERA_AXES_ICON_VARIANT) == 0
            else f"vtk_camera_axes_settings_{self.CAMERA_AXES_ICON_VARIANT}.svg"
        )
        self.camera_btn.setIcon(
            svg_icon(ICON_DIR / camera_icon_name, color=foreground, widget=self.camera_btn, monochrome=True)
        )
        self.orientation_btn.setIcon(
            svg_icon(ICON_DIR / "vtk_camera_perspective.svg", color=foreground, widget=self.orientation_btn, monochrome=True)
        )
        scene_variant = int(self.SCENE_OBJECTS_ICON_VARIANT)
        if scene_variant == 0:
            # Preserve the exact original project icon until the user chooses
            # one of the new monochrome alternatives explicitly.
            scene_icon_name = "vtk_scene_objects_1.svg"
            scene_icon_monochrome = False
        elif 1 <= scene_variant <= 7:
            scene_icon_name = f"vtk_scene_add_{scene_variant}.svg"
            scene_icon_monochrome = True
        else:
            raise ValueError(
                "SCENE_OBJECTS_ICON_VARIANT must be 0 (legacy) or 1..7."
            )
        self.overlays_btn.setIcon(
            svg_icon(
                ICON_DIR / scene_icon_name,
                color=foreground, widget=self.overlays_btn,
                monochrome=scene_icon_monochrome,
            )
        )
        self.axes_button.setIcon(
            svg_icon(ICON_DIR / "vtk_axes_info.svg", color=foreground, widget=self.axes_button, monochrome=False)
        )
        self.bounding_box_button.setIcon(
            svg_icon(ICON_DIR / "vtk_bounding_box.svg", color=foreground, widget=self.bounding_box_button)
        )
        self.axis_ticks_button.setIcon(
            svg_icon(ICON_DIR / f"vtk_axis_ticks_{self.AXIS_TICKS_ICON_VARIANT}.svg", color=foreground, widget=self.axis_ticks_button)
        )
        self.perspective_button.setIcon(
            svg_icon(ICON_DIR / "vtk_camera_perspective.svg", color=foreground, widget=self.perspective_button)
        )
        for axis, button in self.camera_axis_buttons.items():
            button.setIcon(svg_icon(ICON_DIR / f"vtk_camera_axis_{axis}.svg", color=foreground, widget=button))

        self.axis_scale_btn.setIcon(
            svg_icon(
                ICON_DIR / f"vtk_axis_scale_{self.AXIS_SCALE_ICON_VARIANT}.svg",
                color=foreground, widget=self.axis_scale_btn, monochrome=True,
            )
        )
        self.axis_scale_reset_btn.setIcon(
            svg_icon(
                ICON_DIR / "vtk_axis_scale_reset.svg",
                color=foreground, widget=self.axis_scale_reset_btn, monochrome=True,
            )
        )
        self.axis_scale_preview.set_theme(self._theme)
        for key, button in self._overlay_buttons.items():
            spec = self._overlay_tool_specs[key]
            button.setIcon(
                svg_icon(
                    ICON_DIR / str(spec["icon_name"]), color=foreground, widget=button,
                    monochrome=bool(spec["monochrome"]),
                )
            )
        self.theme_btn.setIcon(
            svg_icon(ICON_DIR / ("theme_moon.svg" if self._theme == "light" else "theme_sun.svg"), color=foreground, widget=self.theme_btn)
        )
        target_theme = "тёмную" if self._theme == "light" else "светлую"
        self.theme_btn.setToolTip(f"Переключить viewer на {target_theme} тему")

        self.set_cursor_line_color(self._cursor_line_color)
        self.set_roi_preview_color(self._roi_preview_color)
        self._set_icon_sizes()
        for menu in (self.orientation_menu, self.camera_menu, self.overlays_menu,
                     self.theta_range_menu, self.axis_scale_menu, *self._overlay_settings_menus.values()):
            apply_popup_theme(menu, self._theme)
        apply_popup_theme(self.axis_display_menu, self._theme)
        for menu, padding in (
            (self.orientation_menu, 0), (self.camera_menu, 0), (self.overlays_menu, 0),
            (self.theta_range_menu, 0), (self.axis_scale_menu, 0), (self.axis_display_menu, 2),
            *[(item, 0) for item in self._overlay_settings_menus.values()],
        ):
            self._apply_popup_surface_theme(menu, self._theme, padding=padding)
        for widget in (self._orientation_panel, self._camera_panel, self._overlays_panel,
                       self.theta_range_panel, self.axis_scale_panel, *self._overlay_settings_panels.values()):
            self._apply_popup_panel_theme(widget, self._theme)
        apply_widget_palette(self.theta_range_slider, self._theme)
        apply_widget_palette(self.theta_boundary_planes_check, self._theme)
        apply_widget_palette(self.theta_boundary_opacity_slider, self._theme)
        if self._color_picker is not None:
            try:
                self._color_picker.apply_theme(self._theme)
            except RuntimeError:
                self._color_picker = None
        self.axis_scale_preview.set_theme(self._theme)
        self._refresh_axis_scale_preview()

    # ------------------------------------------------------------------
    # Compatibility and scene-state methods
    # ------------------------------------------------------------------
    def set_axes_visible(self, visible: bool) -> None:
        self.axes_button.blockSignals(True)
        self.axes_button.setChecked(bool(visible))
        self.axes_button.blockSignals(False)

    def set_bounding_box_visible(self, visible: bool) -> None:
        self.bounding_box_button.blockSignals(True)
        self.bounding_box_button.setChecked(bool(visible))
        self.bounding_box_button.blockSignals(False)

    def set_axis_ticks_visible(self, visible: bool) -> None:
        self.axis_ticks_action.blockSignals(True)
        self.axis_ticks_action.setChecked(bool(visible))
        self.axis_ticks_action.blockSignals(False)

    def set_grid_visible(self, visible: bool) -> None:
        self.grid_action.blockSignals(True)
        self.grid_action.setChecked(bool(visible))
        self.grid_action.blockSignals(False)

    def set_overlay_tool_enabled(self, key: str, enabled: bool) -> None:
        button = self._overlay_buttons.get(str(key))
        if button is not None:
            button.setEnabled(bool(enabled))

    def set_overlay_tool_checked(self, key: str, checked: bool) -> None:
        button = self._overlay_buttons.get(str(key))
        if button is not None:
            button.blockSignals(True)
            button.setChecked(bool(checked))
            button.blockSignals(False)

    def set_cursor_line_enabled(self, enabled: bool) -> None:
        self.set_overlay_tool_checked("cursor", enabled)

    def set_theta_slice_enabled(self, enabled: bool) -> None:
        self.set_overlay_tool_checked("theta", enabled)

    def set_projection_enabled(self, enabled: bool) -> None:
        self.set_overlay_tool_checked("projection", enabled)

    def set_scene_available(self, available: bool) -> None:
        enabled = bool(available)
        for button in (
            self.reset_btn, self.orientation_btn, self.camera_btn, self.overlays_btn,
            self.theta_range_btn, self.perspective_button, self.axis_scale_btn,
            self.bounding_box_button, self.axes_button, self.axis_ticks_button,
            *self.camera_axis_buttons.values(), *self._overlay_buttons.values(),
        ):
            button.setEnabled(enabled)
        has_theta_range = len(self._theta_axis_values) > 2
        self.theta_range_slider.setEnabled(enabled and has_theta_range)
        self.theta_range_reset_button.setEnabled(enabled and has_theta_range)
        self.theta_boundary_planes_check.setEnabled(enabled and has_theta_range)
        show_boundary_options = bool(
            enabled and has_theta_range and self.theta_boundary_planes_check.isChecked()
        )
        self.theta_boundary_opacity_slider.setEnabled(show_boundary_options)
        self.theta_boundary_opacity_value.setEnabled(show_boundary_options)

    def set_overlay_controls_enabled(
        self, *, cursor: bool, theta: bool, projection: bool = True,
        roi_preview: bool = True,
    ) -> None:
        self.set_overlay_tool_enabled("cursor", cursor)
        self.set_overlay_tool_enabled("theta", theta)
        self.set_overlay_tool_enabled("projection", projection)
        self.set_overlay_tool_enabled("roi_preview", roi_preview)
        for key, available in (
            ("cursor", cursor), ("theta", theta), ("projection", projection),
            ("roi_preview", roi_preview),
        ):
            widget = self._overlay_settings.get(key)
            if widget is not None:
                widget.setEnabled(bool(available))
        color_button = getattr(self, "cursor_color_btn", None)
        if color_button is not None:
            color_button.setEnabled(bool(cursor))
        roi_color_button = getattr(self, "roi_preview_color_btn", None)
        if roi_color_button is not None:
            roi_color_button.setEnabled(bool(roi_preview))
