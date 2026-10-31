"""Отображение rocking curves с компактной панелью и мышиной навигацией."""
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from roc_mcs.gui.roi_explorer.curve_tools_widget import CurveToolsWidget
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, theme_colors
from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings


class CurvePlotWidget(QWidget):
    """График кривых; подбор и вычисление данных остаются снаружи."""
    settings_requested = pyqtSignal()
    theta_guide_toggled = pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CurvePlotWidget")
        self.setMinimumSize(230, 190); self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display_settings = load_plot_display_settings()["curve"]
        self._pan_active = False
        self._pan_start_px = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        self._base_xlim = None
        self._base_ylim = None
        self._view_initialized = False
        self.roi_right_click_callback = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar = CurveToolsWidget(self)
        self.toolbar.reset_view_requested.connect(self.reset_view)
        self.toolbar.settings_requested.connect(self.settings_requested.emit)
        self.toolbar.theta_guide_toggled.connect(self.theta_guide_toggled.emit)
        layout.addWidget(self.toolbar)
        self.header_widget = QWidget(self)
        self.header_widget.setFixedHeight(24)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(10, 0, 10, 0)
        header_layout.setSpacing(0)
        self.header_label = QLabel("Rocking curves", self.header_widget)
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setStyleSheet("QLabel { color: palette(window-text); font-size: 10px; font-weight: 600; background: transparent; }")
        header_layout.addWidget(self.header_label)
        layout.addWidget(self.header_widget)

        self._theme = "light"
        self.figure = Figure(figsize=(5, 5), facecolor=theme_colors(self._theme)["canvas"])
        self.canvas = FigureCanvasQTAgg(self.figure); self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.set_facecolor(theme_colors(self._theme)["axes"])
        # Художники переиспользуются: во время ROI-preview не очищаем Axes и не мигаем.
        self._curve_artists = {}
        self._preview_artist = None
        self._pixel_artist = None
        self._empty_artist = None
        self._theta_guide_value = None
        self._theta_guide_artist = self.ax.axvline(0, color="#d99b25", linewidth=1.15, linestyle=(0,(3,2)), alpha=0.95, zorder=6, visible=False)
        # Верх графика отделён от заголовка постоянными 4 px, а не 11% высоты Figure.
        self.figure.subplots_adjust(left=0.17, right=0.985, bottom=0.15, top=0.985)
        self.cbar = None
        self.map_image = None
        self.coord_label = QLabel("")
        self.coord_label.setObjectName("MapCoordLabel")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.coord_label.setStyleSheet("QLabel#MapCoordLabel { font-family: monospace; font-size: 10px; font-weight: 600; color: palette(window-text); padding: 2px 5px; background: transparent; }")
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.coord_label, 0)
        self.coord_label.setFixedHeight(18)
        layout.setStretch(0, 0)
        layout.setStretch(1, 0)
        layout.setStretch(2, 1)
        layout.setStretch(3, 0)
        self._axes_sync_pending = False
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self._style_axes()
        self.apply_display_settings(self._display_settings)
        QTimer.singleShot(0, self._sync_axes_top)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._axes_sync_pending:
            self._axes_sync_pending = True
            QTimer.singleShot(0, self._sync_axes_top)

    def _sync_axes_top(self) -> None:
        """Фиксирует верхнюю границу ROC-графика в пикселях при любом resize."""
        self._axes_sync_pending = False
        height = max(1, self.canvas.height())
        top = max(0.55, min(0.998, 1.0 - 4.0 / height))
        current_top = float(self.ax.get_position().y1)
        if abs(current_top - top) * height >= 0.5:
            self.figure.subplots_adjust(top=top)
            self.canvas.draw_idle()

    def apply_display_settings(self, settings: dict | None = None) -> None:
        if settings is not None: self._display_settings = dict(settings)
        self._style_axes(); self.canvas.draw_idle()

    def apply_theme(self, theme: str | None = None) -> None:
        self._theme = str(theme or self._theme).lower()
        if self._theme not in ("dark", "light"): self._theme = "light"
        apply_widget_palette(self, self._theme)
        c = theme_colors(self._theme); self.figure.set_facecolor(c["canvas"]); self.ax.set_facecolor(c["axes"])
        self.header_label.setStyleSheet(f"QLabel {{ color:{c['text']}; font-size:10px; font-weight:600; background:transparent; }}")
        self.coord_label.setStyleSheet(f"QLabel#MapCoordLabel {{ font-family:monospace; font-size:10px; font-weight:600; color:{c['muted']}; padding:2px 5px; background:transparent; }}")
        self._style_axes(); self.toolbar.apply_theme(self._theme)
        if self._preview_artist is not None: self._preview_artist.set_color(c["text"])
        if self._pixel_artist is not None: self._pixel_artist.set_color(c["muted"])
        if self._empty_artist is not None: self._empty_artist.set_color(c["muted"])
        self.canvas.setStyleSheet(f"background:{c['canvas']};")
        self.canvas.draw_idle()

    def _style_axes(self) -> None:
        c = theme_colors(self._theme); s = self._display_settings; font = str(s.get("font_family", "Segoe UI"))
        self.ax.set_xlabel("θ", fontsize=int(s.get("label_size", 9)), color=c["text"], labelpad=4, fontname=font)
        self.ax.set_ylabel("Intensity", fontsize=int(s.get("label_size", 9)), color=c["text"], labelpad=5, fontname=font)
        self.ax.grid(bool(s.get("show_grid", True)), which="major", color=c["grid"], linewidth=0.55, alpha=0.22)
        self.ax.tick_params(axis="both", colors=c["muted"], labelsize=int(s.get("tick_size", 8)), direction="in", length=3, width=0.6, pad=3)
        for label in (*self.ax.get_xticklabels(), *self.ax.get_yticklabels()): label.set_fontname(font)
        for spine in self.ax.spines.values(): spine.set_color(c["border"]); spine.set_linewidth(0.7)
        self.ax.yaxis.set_major_locator(MaxNLocator(nbins=4)); self.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)); self.ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        self.ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        scale_label = self.ax.yaxis.get_offset_text(); scale_label.set_fontsize(max(6, int(s.get("tick_size", 8))-1)); scale_label.set_fontname(font); scale_label.set_color(c["muted"]); scale_label.set_verticalalignment("bottom")

    @staticmethod
    def _left_click_picker(artist, mouseevent):
        """Разрешает событие выбора линии только при нажатии левой кнопкой мыши"""
        if mouseevent.button != 1:
            return False, {}
        return artist.contains(mouseevent)

    def _has_visible_curve_data(self) -> bool:
        """Направляющая θ разрешена только при наличии хотя бы одной реальной кривой."""
        artists=list(self._curve_artists.values())
        if self._preview_artist is not None: artists.append(self._preview_artist)
        if self._pixel_artist is not None: artists.append(self._pixel_artist)
        for artist in artists:
            if not artist.get_visible(): continue
            x,y=artist.get_data()
            if len(x) and len(y):
                try:
                    import numpy as np
                    xa=np.asarray(x,dtype=float); ya=np.asarray(y,dtype=float)
                    if xa.shape==ya.shape and (np.isfinite(xa)&np.isfinite(ya)).any(): return True
                except (TypeError,ValueError):
                    return True
        return False

    def set_theta_guide(self, theta_value: float | None) -> None:
        """Показывает θ только поверх существующей pinned/preview/pixel-кривой."""
        self._theta_guide_value = float(theta_value) if theta_value is not None else None
        visible=self._theta_guide_value is not None and self._has_visible_curve_data()
        if visible: self._theta_guide_artist.set_xdata([self._theta_guide_value,self._theta_guide_value])
        self._theta_guide_artist.set_visible(bool(visible)); self.canvas.draw_idle()

    def draw_curves(self, curves_data: list, preview_data: dict = None, *, pixel_data: dict = None,
                    view_label: str | None = None, fit_view: bool = False, fit_y: bool = False, theta_guide: float | None = None) -> dict:
        """Обновляет линии на месте; во время preview подбирает Y, сохраняя ручной X-зум."""
        previous_xlim = self.ax.get_xlim() if self._view_initialized else None
        previous_ylim = self.ax.get_ylim() if self._view_initialized else None
        self._theta_guide_value = float(theta_guide) if theta_guide is not None else None
        self._theta_guide_artist.set_visible(False)
        title = f"Rocking curves · {view_label}" if view_label else "Rocking curves"
        if self.header_label.text() != title:
            self.header_label.setText(title)

        curve_lines = {}
        active_ids = set()
        for curve in curves_data:
            roi_id = curve["id"]
            active_ids.add(roi_id)
            line = self._curve_artists.get(roi_id)
            if line is None or line.axes is not self.ax:
                line, = self.ax.plot([], [], picker=self._left_click_picker)
                line.set_pickradius(5)
                self._curve_artists[roi_id] = line
            line.set_data(curve["th"], curve["curve"])
            line.set_color(curve["color"])
            line.set_linewidth(2.4 if curve["is_selected"] else 1.5)
            line.set_alpha(1.0 if curve["is_selected"] else 0.88)
            line.set_linestyle("-")
            line.set_label(curve["label"])
            line._roi_id = roi_id
            line.set_visible(True)
            curve_lines[roi_id] = line

        for roi_id in tuple(self._curve_artists):
            if roi_id not in active_ids:
                line = self._curve_artists.pop(roi_id)
                try: line.remove()
                except ValueError: pass

        c = theme_colors(self._theme)
        if preview_data:
            if self._preview_artist is None or self._preview_artist.axes is not self.ax:
                self._preview_artist, = self.ax.plot([], [], linestyle="--", linewidth=1.8, alpha=0.95, label="ROI preview", zorder=4)
            self._preview_artist.set_data(preview_data["th"], preview_data["curve"])
            self._preview_artist.set_color(c["text"])
            self._preview_artist.set_visible(True)
        elif self._preview_artist is not None:
            self._preview_artist.set_visible(False)
            self._preview_artist.set_data([], [])

        if pixel_data:
            if self._pixel_artist is None or self._pixel_artist.axes is not self.ax:
                self._pixel_artist, = self.ax.plot([], [], linestyle="-.", linewidth=1.7, alpha=0.95, zorder=3)
            self._pixel_artist.set_data(pixel_data["th"], pixel_data["curve"])
            self._pixel_artist.set_color(c["muted"])
            self._pixel_artist.set_label(pixel_data["label"])
            self._pixel_artist.set_visible(True)
        elif self._pixel_artist is not None:
            self._pixel_artist.set_visible(False)
            self._pixel_artist.set_data([], [])

        has_data = bool(curves_data or preview_data or pixel_data)
        if not has_data:
            if self._empty_artist is None or self._empty_artist.axes is not self.ax:
                self._empty_artist = self.ax.text(0.5, 0.5, "No saved ROIs or selected pixel", ha="center", va="center",
                    transform=self.ax.transAxes, color=c["muted"], fontsize=9)
            self._empty_artist.set_color(c["muted"])
            self._empty_artist.set_visible(True)
            self._view_initialized = False
            self._base_xlim = self._base_ylim = None
            self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        else:
            if self._empty_artist is not None: self._empty_artist.set_visible(False)
            # Включаем autoscale только для измерения полного extent: set_xlim/set_ylim пользователя его отключают.
            # Затем текущий зум восстанавливается ниже, кроме финального fit_view после отпускания ROI.
            self.ax.set_autoscalex_on(True); self.ax.set_autoscaley_on(True)
            self.ax.relim(); self.ax.autoscale_view()
            self._base_xlim, self._base_ylim = self.ax.get_xlim(), self.ax.get_ylim()
            live_ylim=self.ax.get_ylim()
            if previous_xlim is not None and previous_ylim is not None and not fit_view:
                self.ax.set_xlim(previous_xlim)
                self.ax.set_ylim(live_ylim if (fit_y and preview_data) else previous_ylim)
            self._view_initialized = True
        guide_visible=self._theta_guide_value is not None and has_data and self._has_visible_curve_data()
        if guide_visible:self._theta_guide_artist.set_xdata([self._theta_guide_value,self._theta_guide_value])
        self._theta_guide_artist.set_visible(bool(guide_visible))
        self.canvas.draw_idle()
        return curve_lines

    def reset_view(self) -> None:
        if self._base_xlim is not None and self._base_ylim is not None:
            self.ax.set_xlim(self._base_xlim)
            self.ax.set_ylim(self._base_ylim)
            self.canvas.draw_idle()

    def _on_scroll_zoom(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None: return
        step = getattr(event, "step", 0)
        if not step: step = 1 if getattr(event, "button", None) == "up" else -1 if getattr(event, "button", None) == "down" else 0
        if not step: return
        scale = 1 / 1.15 if step > 0 else 1.15
        x, y = event.xdata, event.ydata; xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim(x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale)
        self.ax.set_ylim(y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale)
        self.canvas.draw_idle()

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is not self.ax: return
        if event.button == 2:
            self._pan_active = True
            self._pan_start_px = (event.x, event.y)
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button != 3: 
            return
        clicked_roi_id = None
        for line in reversed(self.ax.lines):
            contains, _ = line.contains(event)
            if contains and hasattr(line, "_roi_id"):
                clicked_roi_id = line._roi_id
                break
        if self.roi_right_click_callback: 
            self.roi_right_click_callback(clicked_roi_id)

    def _on_mouse_move(self, event) -> None:
        if self._pan_active and event.x is not None and event.y is not None and self._pan_start_px is not None:
            dx = event.x - self._pan_start_px[0]
            dy = event.y - self._pan_start_px[1]
            bbox = self.ax.bbox
            if bbox.width > 0 and bbox.height > 0:
                sx = (self._pan_start_xlim[1] - self._pan_start_xlim[0]) / bbox.width
                sy = (self._pan_start_ylim[1] - self._pan_start_ylim[0]) / bbox.height
                self.ax.set_xlim(self._pan_start_xlim[0] - dx * sx, self._pan_start_xlim[1] - dx * sx)
                self.ax.set_ylim(self._pan_start_ylim[0] - dy * sy, self._pan_start_ylim[1] - dy * sy)
                self.canvas.draw_idle()
        if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
            self.coord_label.setText(f"θ: {event.xdata:.5g}   I: {event.ydata:.5g}")

    def _on_mouse_release(self, event) -> None:
        if self._pan_active and event.button == 2:
            self._pan_active = False
            self._pan_start_px = None
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_axes_leave(self, event) -> None:
        if not self._pan_active: self.coord_label.clear()

