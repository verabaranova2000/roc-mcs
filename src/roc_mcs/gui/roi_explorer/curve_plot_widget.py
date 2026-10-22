"""Отображение rocking curves с компактной панелью и мышиной навигацией."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from roc_mcs.gui.roi_explorer.curve_tools_widget import CurveToolsWidget


class CurvePlotWidget(QWidget):
    """График кривых; подбор и вычисление данных остаются снаружи."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMinimumHeight(350)
        self._pan_active = False
        self._pan_start_px = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        self._base_xlim = None
        self._base_ylim = None
        self.roi_right_click_callback = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar = CurveToolsWidget(self)
        self.toolbar.reset_view_requested.connect(self.reset_view)
        layout.addWidget(self.toolbar)
        self.header_widget = QWidget(self)
        self.header_widget.setFixedHeight(24)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(10, 2, 10, 1)
        self.header_label = QLabel("Rocking curves", self.header_widget)
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setStyleSheet("QLabel { color: #343a40; font-size: 10px; font-weight: 600; background: transparent; }")
        header_layout.addWidget(self.header_label)
        layout.addWidget(self.header_widget)

        self.figure = Figure(figsize=(5, 5), facecolor="#ffffff")
        self.canvas = FigureCanvasQTAgg(self.figure); self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.set_facecolor("#ffffff")
        self.figure.subplots_adjust(left=0.17, right=0.985, bottom=0.15, top=0.89)              # отступы Figure
        self.cbar = None
        self.map_image = None
        self.coord_label = QLabel("")
        self.coord_label.setObjectName("MapCoordLabel")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.coord_label.setStyleSheet("QLabel#MapCoordLabel { font-family: monospace; font-size: 10px; font-weight: 600; color: #244c73; padding: 2px 5px; background: transparent; }")
        layout.addWidget(self.canvas)
        layout.addWidget(self.coord_label)
        layout.addStretch(1)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self._style_axes()

    def _style_axes(self) -> None:
        self.ax.set_xlabel("θ", fontsize=9, color="#343a40", labelpad=4)
        self.ax.set_ylabel("Intensity", fontsize=9, color="#343a40", labelpad=5)
        self.ax.grid(True, which="major", color="#9aa4b2", linewidth=0.6, alpha=0.25)
        self.ax.tick_params(axis="both", colors="#4b5563", labelsize=8, direction="out", length=3, width=0.6)
        for spine in self.ax.spines.values(): 
            spine.set_color("#aab2bd")
            spine.set_linewidth(0.7)
        self.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        self.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        self.ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        self.ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        # self.ax.yaxis.get_offset_text().set_fontsize(8)
        scale_label = self.ax.yaxis.get_offset_text()
        scale_label.set_fontsize(8)
        scale_label.set_color("#4b5563")
        scale_label.set_verticalalignment("bottom")        

    @staticmethod
    def _left_click_picker(artist, mouseevent):
        """Разрешает событие выбора линии только при нажатии левой кнопкой мыши"""
        if mouseevent.button != 1:
            return False, {}
        return artist.contains(mouseevent)

    def draw_curves(self, curves_data: list, preview_data: dict = None, *, pixel_data: dict = None, view_label: str | None = None) -> dict:
        """Перерисовывает данные и запоминает полный extent для Reset View."""
        self.ax.clear()
        self.ax.set_facecolor("#ffffff")
        self.header_label.setText(f"Rocking curves · {view_label}" if view_label else "Rocking curves")
        self._style_axes()
        curve_lines = {}
        for curve in curves_data:
            # line, = self.ax.plot(curve["th"], curve["curve"], color=curve["color"],
            #     linewidth=2.4 if curve["is_selected"] else 1.5,
            #     alpha=1.0 if curve["is_selected"] else 0.88, label=curve["label"], picker=5)
            # line._roi_id = curve["id"]
            line, = self.ax.plot(curve["th"], curve["curve"], color=curve["color"],
                linewidth=2.4 if curve["is_selected"] else 1.5,
                alpha=1.0 if curve["is_selected"] else 0.88,
                label=curve["label"], picker=self._left_click_picker)
            line.set_pickradius(5)
            line._roi_id = curve["id"]            
            curve_lines[curve["id"]] = line
        if preview_data:
            self.ax.plot(preview_data["th"], preview_data["curve"], color="#20252b", linestyle="--", linewidth=1.8, alpha=0.95, label="ROI preview")
        if pixel_data:
            self.ax.plot(pixel_data["th"], pixel_data["curve"], color="#505965", linestyle="-.", linewidth=1.7, alpha=0.95, label=pixel_data["label"])
        # if curves_data or preview_data or pixel_data:
        #     self.ax.legend(frameon=False, loc="best", fontsize=8, labelspacing=0.3, handlelength=1.8, borderaxespad=0.5)
        # else:
        #     self.ax.text(0.5, 0.5, "No saved ROIs or selected pixel", ha="center", va="center",
        #                  transform=self.ax.transAxes, color="#737c87", fontsize=9)
        if not (curves_data or preview_data or pixel_data):
            self.ax.text(0.5, 0.5, "No saved ROIs or selected pixel",
                         ha="center", va="center", transform=self.ax.transAxes,
                         color="#737c87", fontsize=9)        
        self.ax.relim()
        self.ax.autoscale_view()
        self._base_xlim = self.ax.get_xlim()
        self._base_ylim = self.ax.get_ylim()
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Совпадает с картой: квадратный холст и одинаковые строки над/под графиком.
        self.canvas.setFixedHeight(max(120, self.width()))
