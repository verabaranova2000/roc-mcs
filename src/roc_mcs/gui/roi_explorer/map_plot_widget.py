from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSizePolicy, QToolButton
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.colors import to_rgba
import numpy as np

from roc_mcs.roi.spatial_pooling import PoolSpec
from  roc_mcs.gui.roi_explorer.map_tools_widget import MapToolsWidget, pooling_preview_pixmap as _pooling_preview_pixmap
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, theme_colors
from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings
from roc_mcs.gui.roi_explorer.map_tools_widget import SmartDropdownButton, QRangeSlider
from roc_mcs.gui.roi_explorer.selectors import RotatedRectangleSelector
from .selectors.line import LineSelector
from .selectors.point import PointSelector

# =====================================================================
# СТИЛИ И КОНСТАНТЫ ВИДЖЕТА КАРТЫ (MapPlotWidget)
# =====================================================================

# Стиль для выпадающих списков заголовка (плоские, без рамок и стрелочек, с автошироким выпадающим списком)
# Делаем списки визуально "плоскими", как текст заголовка
STYLE_FLAT_COMBOBOX = """
    QComboBox { border:none; background:transparent; font-size:10px; font-family:'DejaVu Sans','Arial',sans-serif; font-weight:bold; padding:2px; color:palette(window-text); }
    QComboBox::drop-down { border:none; width:0px; }
    QComboBox:hover { color:palette(highlight); }
    QComboBox QAbstractItemView { border:1px solid palette(mid); background:palette(base); color:palette(text); selection-background-color:palette(highlight); selection-color:palette(highlighted-text); min-width:250px; }
"""
STYLE_COORD_LABEL = """
    QLabel#MapCoordLabel { font-family:monospace; font-size:10px; font-weight:bold; color:palette(window-text); padding:2px 5px; background:transparent; }
"""



class MapPlotWidget(QWidget):
    """
    Автономный виджет для визуализации 2D карт.
    Сам управляет выбором скана, карты и редукции.
    """
    # --- Сигналы наружу (чтобы Главное окно знало об изменениях) ---
    scan_changed = pyqtSignal(str)
    map_changed =  pyqtSignal(str)
    reduction_changed =  pyqtSignal(str)
    pool_changed = pyqtSignal(object)      # Новый PoolSpec выбран пользователем
    pixel_clicked = pyqtSignal(int, int)   # Индексы пикселя в текущем view: x, y
    data_updated =  pyqtSignal()             # Сигнал, что картинка перерисовалась
    settings_requested = pyqtSignal()
    pin_roi_requested = pyqtSignal()         # сигнал умного списка ROI
    preview_roi_changed = pyqtSignal(object) # Геометрия ROI при движении селектора.
    preview_roi_finished = pyqtSignal(object) # Финальная геометрия после отпускания мыши.

    # =========================================================================
    # РЕЕСТР ИНСТРУМЕНТОВ (Статичная конфигурация)
    # Ключи обязаны строго совпадать с полем `type` в датаклассе ROI!
    # =========================================================================
    ROI_TOOLS_CONFIG = {
        "rect": {
            "name": "Rectangle ROI", 
            "icon": str(ICON_DIR / "roi_rectangle.svg"),  # ⛶ ⧉
            "selector_class": RotatedRectangleSelector 
        },
        "line": {
            "name": "Line ROI",      
            "icon": str(ICON_DIR / "roi_line.svg"),       # \    
            "selector_class": LineSelector      
        },
        "point": {
            "name": "Point ROI",     
            "icon": str(ICON_DIR / "roi_point.svg"),      # 𖥔 ݁✠    
            "selector_class": PointSelector
        },
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MapPlotWidget")
        self.repo = None                    # Ссылка на данные
        self._pool_spec = PoolSpec()

        self.setMinimumSize(340, 190); self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display_settings = load_plot_display_settings()["map"]
        self._scalebar_artist = None
        
        # ---Внутреннее состояние ---
        self.current_scan = None
        self.reduction_mode = "mean"
        self._current_roi_type = "rect"  # Дефолтное состояние селектора. Теперь оно на 100% бьется с ROI(type="rect")
        self._current_selector = None    # Здесь будет жить инстанс активного селектора
        
        # --- Состояние для Pan (перемещения) ---
        self._pan_active = False
        self._pan_start_x_px = None  # Теперь используем физические пиксели экрана!
        self._pan_start_y_px = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        # # --- Сохраняем "базовые" лимиты для ограничения зума ---
        # self._base_xlim = None
        # self._base_ylim = None

        # ГЛАВНЫЙ СЛОЙ ВИДЖЕТА
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)                 # Ноль лишних отступов

        # ИНИЦИАЛИЗАЦИЯ MATPLOTLIB
        self.figure = Figure(figsize=(5, 5), tight_layout=False)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.figure.subplots_adjust(left=0.12, right=0.91, bottom=0.11, top=0.97)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # self.canvas.setFixedHeight(450)  # Подобрать цифру, при которой карта выглядит идеально квадратной
        
        self.ax = self.figure.add_subplot(1, 1, 1)
        self._theme = "light"
        self.cbar = None
        self.map_image = None
        self.roi_right_click_callback = None
        self.cax = None                                                             # Заготовим переменную для оси colorbar'а
        
        # Панель вынесена отдельно; публичные атрибуты сохранены для совместимости.
        self.current_cmap = "inferno"
        self.tools = MapToolsWidget(self.ROI_TOOLS_CONFIG, self._current_roi_type, SmartDropdownButton, QRangeSlider, self)
        self.toolbar = self.tools
        for name in ("home_btn", "roi_btn", "roi_menu", "roi_action_group", "roi_tool_buttons",
                     "pool_btn", "pool_menu", "pool_size_spin", "pool_stride_spin", "pool_apply_btn",
                     "pool_preview_label", "pin_roi_btn", "cmap_btn", "cmap_menu", "auto_contrast_btn",
                     "low_pct", "high_pct", "range_slider", "settings_btn"):
            setattr(self, name, getattr(self.tools, name))
        self.tools.reset_view_requested.connect(self.reset_view)
        self.tools.settings_requested.connect(self.settings_requested.emit)
        self.tools.pin_roi_requested.connect(self.pin_roi_requested.emit)
        self.tools.cmap_selected.connect(self._on_cmap_selected)
        self.tools.roi_toggled.connect(self._on_roi_toggled)
        self.tools.roi_tool_clicked.connect(self._on_flyout_btn_clicked)
        self.tools.pool_size_changed.connect(self._on_pool_size_changed)
        self.tools.pool_stride_changed.connect(self._update_pool_preview)
        self.tools.pool_apply_requested.connect(self._apply_pool_selection)
        self.set_pool_spec(PoolSpec())
        self.pool_btn.setEnabled(False)

        # ========================================================
        # ЭТАЖ 2: УМНЫЙ ЗАГОЛОВОК (Интерактивная замена title)
        # ========================================================
        self.header_widget = QWidget()
        self.header_widget.setObjectName("MapHeader")
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_widget.setFixedHeight(25)
        self.header_layout.setContentsMargins(6, 0, 6, 0); self.header_layout.setSpacing(5)

        self.scan_combo = QComboBox()
        self.map_combo = QComboBox()
        self.reduction_combo = QComboBox()
        self.reduction_combo.addItems(["mean", "sum"])
        # Подсказки к полям заголовка
        self.scan_combo.setToolTip("Выбор набора данных (скана) из HDF5 архива.")
        self.map_combo.setToolTip("Выбор типа 2D карты (распределение интенсивности, ширины пика, положения и др.).")
        self.reduction_combo.setToolTip("Режим агрегации пикселей (mean/sum) при проецировании ROI в 1D профиль.")
        
        for combo, minimum in ((self.scan_combo, 80), (self.map_combo, 76), (self.reduction_combo, 55)):
            combo.setStyleSheet(STYLE_FLAT_COMBOBOX); combo.setMinimumWidth(minimum)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); combo.setMinimumContentsLength(5)
        
        def create_separator():
            """Создает вертикальный разделитель ' | ' для умного заголовка."""
            lbl = QLabel(" | ")
            lbl.setObjectName("MapHeaderSeparator")
            lbl.setStyleSheet("font-size: 15px; font-weight: normal; color: palette(window-text);")
            return lbl
    
        # Управление данными остаётся непосредственно над соответствующим полотном.
        self.header_layout.addWidget(QLabel("Scan", self.header_widget))
        self.header_layout.addWidget(self.scan_combo, 2)
        self.header_layout.addWidget(create_separator())
        self.header_layout.addWidget(QLabel("Map", self.header_widget))
        self.header_layout.addWidget(self.map_combo, 1)
        self.header_layout.addWidget(create_separator())
        self.header_layout.addWidget(QLabel("ROI", self.header_widget))
        self.header_layout.addWidget(self.reduction_combo)

        # ========================================================
        # ЭТАЖ 3: Строка координат мыши (x,y)
        # ========================================================
        self.coord_label = QLabel("")
        self.coord_label.setObjectName("MapCoordLabel")   # имя для селектора стилей 
        self.coord_label.setStyleSheet(STYLE_COORD_LABEL) # применяем константу стилей
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)   # Выравниваем текст по правому краю, чтобы он был аккуратно под графиком
        
        # ========================================================
        # СБОРКА ГЛАВНОГО СЛОЯ (main_layout)
        # ========================================================
        self.main_layout.addWidget(self.toolbar)         # 1. Тулбар наверху
        self.main_layout.addWidget(self.header_widget)   # 2. Интерактивный заголовок
        self.main_layout.addWidget(self.canvas, 1)       # 3. График (занимает всё оставшееся место)
        self.main_layout.addWidget(self.coord_label)

 
        # ========================================================
        # ПОДКЛЮЧЕНИЕ СИГНАЛОВ И СОБЫТИЙ
        # ========================================================
        self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)         # вместо self._on_mouse_click и self._on_mouse_press_pan 
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)     # вместо self._on_mouse_move и self._on_mouse_move_pan
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release) # вместо self._on_mouse_release_pan

        
        # Подключаем комбобоксы к сигналам виджета
        self.scan_combo.currentTextChanged.connect(self.scan_changed.emit)
        self.map_combo.currentTextChanged.connect(self.map_changed.emit)
        self.reduction_combo.currentTextChanged.connect(self.reduction_changed.emit)
        self.apply_display_settings(self._display_settings)

    
    # ============================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ КЛАССА
    # ============================================================
    def _on_cmap_selected(self, cmap_name: str) -> None:
        self.current_cmap = cmap_name; self.tools.set_current_cmap(cmap_name)
        self.low_pct.valueChanged.emit(self.low_pct.value())   # вызываем перерисовку холста (сигнал будет подхвачен Главным окном)

    def _on_pool_size_changed(self, value: int) -> None:
        self._keep_pool_size_odd(value)
        self._update_pool_preview()

    def _update_pool_preview(self, *_args) -> None:
        if hasattr(self, "pool_preview_label"):
            self.pool_preview_label.setPixmap(
                _pooling_preview_pixmap(self.pool_size_spin.value(), self.pool_stride_spin.value(), self.palette())
            )
    
    def _keep_pool_size_odd(self, value: int) -> None:
        if value % 2 == 0:
            maximum = self.pool_size_spin.maximum()
            self.pool_size_spin.setValue(value + 1 if value < maximum else value - 1)

    def set_pool_size_limit(self, source_shape_yx) -> None:
        ny, nx = map(int, source_shape_yx)
        max_size = min(ny, nx)
        max_size = max_size if max_size % 2 else max_size - 1
        self.pool_size_spin.setRange(1, max(1, max_size))
        self.pool_stride_spin.setRange(1, max(1, ny, nx, self._pool_spec.stride))


    def set_pool_spec(self, pool: PoolSpec, *, source_shape_yx=None) -> None:
        self._pool_spec = pool
        if source_shape_yx is not None:
            self.set_pool_size_limit(source_shape_yx)
        for spin, value in ((self.pool_size_spin, pool.size), (self.pool_stride_spin, pool.stride)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_pool_preview()
        label = ("Raw (window 1×1, stride 1)" if not pool.active else
                 f"Count-sum window {pool.size}×{pool.size}, stride {pool.stride}")
        self.pool_btn.setToolTip(f"Spatial view: {label}")
        self.pool_btn.setProperty("poolActive", bool(pool.active))
        self.pool_btn.style().unpolish(self.pool_btn); self.pool_btn.style().polish(self.pool_btn); self.pool_btn.update()
        
    def set_pool_enabled(self, enabled: bool) -> None:
        self.pool_btn.setEnabled(bool(enabled))

    def _apply_pool_selection(self) -> None:
        pool = PoolSpec(size=int(self.pool_size_spin.value()), stride=int(self.pool_stride_spin.value()))
        self.pool_menu.close()
        if pool != self._pool_spec:
            self.pool_changed.emit(pool)

    def reset_preview_selector(self) -> None:
        """Сбрасывает незакреплённый ROI после смены скана."""
        if self._current_selector is not None:
            self._current_selector.remove()
            self._current_selector = None
        # До первой карты селектор создавать нельзя: первый draw_map() очищает Axes.
        if self.roi_btn.isChecked() and self.map_image is not None:
            self._apply_current_roi_tool(active=True)

    # --- Графики ---  
    
    def apply_theme(self, theme: str | None = None) -> None:
        self._theme = str(theme or self._theme).lower()
        if self._theme not in ("dark", "light"): self._theme = "light"
        apply_widget_palette(self, self._theme)
        c = theme_colors(self._theme)
        self.header_widget.setStyleSheet(f"QWidget#MapHeader {{ background:{c['window']}; color:{c['text']}; }} QWidget#MapHeader QLabel {{ color:{c['muted']}; font-size:9px; background:transparent; }}")
        combo_style = f"""
            QComboBox {{ border:none; background:transparent; font-size:10px; font-family:'DejaVu Sans','Arial',sans-serif; font-weight:bold; padding:2px; color:{c['text']}; }}
            QComboBox::drop-down {{ border:none; width:0px; }}
            QComboBox:hover {{ color:{c['highlight']}; }}
            QComboBox QAbstractItemView {{ border:1px solid {c['border']}; background:{c['base']}; color:{c['text']}; selection-background-color:{c['highlight']}; selection-color:{c['highlight_text']}; min-width:250px; }}
        """
        for combo in (self.scan_combo, self.map_combo, self.reduction_combo):
            combo.setPalette(self.palette()); combo.setStyleSheet(combo_style)
            view = combo.view()
            view.setPalette(combo.palette())
            view.setAutoFillBackground(True)
            view.setStyleSheet(f"QListView, QAbstractItemView {{ background:{c['base']}; color:{c['text']}; border:1px solid {c['border']}; outline:0; }} QListView::item, QAbstractItemView::item {{ color:{c['text']}; padding:3px 5px; }} QListView::item:selected, QAbstractItemView::item:selected {{ background:{c['highlight']}; color:{c['highlight_text']}; }}")
            view.viewport().setPalette(combo.palette()); view.viewport().setAutoFillBackground(True)
        for label in self.header_widget.findChildren(QLabel, "MapHeaderSeparator"):
            label.setStyleSheet(f"font-size:15px; font-weight:normal; color:{c['muted']}; background:transparent;")
        self._apply_axes_theme(); self._update_scalebar(); self.tools.apply_theme(self._theme); self._update_pool_preview()
        self.coord_label.setStyleSheet(f"QLabel#MapCoordLabel {{ font-family:monospace; font-size:10px; font-weight:600; color:{c['muted']}; padding:1px 4px; background:transparent; }}")
        self.canvas.draw_idle()

    def apply_display_settings(self, settings: dict | None = None) -> None:
        if settings is not None: self._display_settings = dict(settings)
        if self.map_image is not None:
            show_cbar = bool(self._display_settings.get("show_colorbar", True))
            if show_cbar and self.cbar is None:
                divider = make_axes_locatable(self.ax); self.cax = divider.append_axes("right", size="4.2%", pad=0.055); self.cbar = self.figure.colorbar(self.map_image, cax=self.cax)
                formatter = ScalarFormatter(useMathText=True); formatter.set_powerlimits((-3, 4)); self.cbar.ax.yaxis.set_major_formatter(formatter); self.cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            elif not show_cbar and self.cbar is not None:
                self.cbar.remove(); self.cbar = None; self.cax = None
            elif show_cbar and self.cbar is not None:
                formatter = ScalarFormatter(useMathText=True); formatter.set_powerlimits((-3, 4)); self.cbar.ax.yaxis.set_major_formatter(formatter); self.cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        self._apply_axes_theme()
        if self.map_image is not None: self._update_scalebar()
        self.canvas.draw_idle()

    def _apply_axes_theme(self) -> None:
        c = theme_colors(self._theme); s = self._display_settings; font = str(s.get("font_family", "Segoe UI"))
        self.figure.set_facecolor(c["canvas"]); self.ax.set_facecolor(c["axes"])
        for axis, label in ((self.ax.xaxis, self.ax.xaxis.label), (self.ax.yaxis, self.ax.yaxis.label)):
            label.set_color(c["text"]); label.set_fontname(font); label.set_fontsize(int(s.get("label_size", 9)))
        self.ax.tick_params(axis="both", colors=c["muted"], labelsize=int(s.get("tick_size", 8)), direction="in", length=3, width=0.65, top=True, right=True, pad=3)
        for label in (*self.ax.get_xticklabels(), *self.ax.get_yticklabels()): label.set_fontname(font)
        for spine in self.ax.spines.values(): spine.set_color(c["border"]); spine.set_linewidth(0.7)
        if self.cbar is not None:
            self.cbar.ax.set_facecolor(c["axes"]); self.cbar.ax.tick_params(colors=c["muted"], labelsize=int(s.get("colorbar_tick_size", 8)), direction="in", length=2.5, width=0.6, pad=2)
            for label in self.cbar.ax.get_yticklabels(): label.set_fontname(font)
            offset = self.cbar.ax.yaxis.get_offset_text(); offset.set_fontsize(max(6, int(s.get("colorbar_tick_size", 8))-1)); offset.set_fontname(font); offset.set_color(c["muted"])
            self.cbar.outline.set_edgecolor(c["border"]); self.cbar.outline.set_linewidth(0.7)
            for spine in self.cbar.ax.spines.values(): spine.set_color(c["border"]); spine.set_linewidth(0.7)
        self.canvas.setStyleSheet(f"background:{c['canvas']};")

    def _update_scalebar(self) -> None:
        if self._scalebar_artist is not None:
            try: self._scalebar_artist.remove()
            except (ValueError, AttributeError): pass
            self._scalebar_artist = None
        s = self._display_settings
        if not bool(s.get("show_scalebar", False)) or self.map_image is None: return
        try:
            pixel_um, length_um = float(s.get("pixel_size_um", 55.0)), float(s.get("scalebar_um", 2000.0))
            if not np.isfinite(pixel_um) or not np.isfinite(length_um) or pixel_um <= 0 or length_um <= 0: return
            size_data = length_um / pixel_um
            xlim = self.ax.get_xlim()
            if size_data >= abs(xlim[1] - xlim[0]) * 0.72: return
            label = f"{length_um/1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm"
            font = FontProperties(family=str(s.get("font_family", "Segoe UI")), size=int(s.get("scalebar_font_size", 9)), weight="bold")
            bar_color = "#f5f7fa" if self._theme == "dark" else "#263445"
            self._scalebar_artist = AnchoredSizeBar(self.ax.transData, size_data, label, loc="lower left", pad=0.45,
                color=bar_color, frameon=False, size_vertical=1.0, fontproperties=font, sep=3, borderpad=0.45)
            self._scalebar_artist.set_zorder(8); self.ax.add_artist(self._scalebar_artist)
        except (TypeError, ValueError, RuntimeError):
            self._scalebar_artist = None

    @staticmethod
    def _axis_centers(values, size, name):
        a = np.arange(size, dtype=float) if values is None else np.asarray(values, dtype=float)
        if a.ndim != 1 or a.size != size or not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: ожидалась конечная 1D-ось длины {size}")
        if a.size > 1 and (not np.all(np.diff(a) > 0) or not np.allclose(np.diff(a), np.diff(a)[0])):
            raise ValueError(f"{name} должна быть равномерной и строго возрастающей")
        step = float(np.diff(a)[0]) if a.size > 1 else 1.0
        return a, (float(a[0] - step / 2), float(a[-1] + step / 2))

    @staticmethod
    def _nearest_axis_index(values, coordinate):
        a = np.asarray(values, dtype=float)
        step = float(np.diff(a)[0]) if a.size > 1 else 1.0
        if coordinate < a[0] - step / 2 or coordinate > a[-1] + step / 2:
            return None
        j = int(np.searchsorted(a, coordinate).clip(0, len(a) - 1))
        if j and abs(a[j - 1] - coordinate) <= abs(a[j] - coordinate): j -= 1
        return j
        

    def draw_map(self, img, title: str, cmap: str, vmin: float, vmax: float, *, x_coords=None, y_coords=None) -> None:
        """Обновляет карту в координатах исходной detector-grid, сохраняя ROI и селектор."""
        img = np.asarray(img)
        if img.ndim != 2:
            raise ValueError(f"Ожидалась 2D-карта, получена форма {img.shape}")
        self._x_coords, x_edges = self._axis_centers(x_coords, img.shape[1], "x_coords")
        self._y_coords, y_edges = self._axis_centers(y_coords, img.shape[0], "y_coords")
        extent = (x_edges[0], x_edges[1], y_edges[0], y_edges[1])
        first_draw = self.map_image is None

        if first_draw:
            self.ax.clear()
            self.ax.set_xlabel("Detector X (source pixels)")
            self.ax.set_ylabel("Detector Y (source pixels)")
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True)); self.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            self.ax.set_aspect("equal", adjustable="box")
            self.map_image = self.ax.imshow(img, origin="lower", cmap=cmap, interpolation="nearest",
                                            extent=extent, vmin=vmin, vmax=vmax)
            self.ax.set_xlim(x_edges)
            self.ax.set_ylim(y_edges)
        else:
            # Обновляем artist на месте: оси не очищаются, поэтому ROI и handles селектора остаются на месте.
            old_xlim, old_ylim = self.ax.get_xlim(), self.ax.get_ylim()
            self.map_image.set_data(img)
            self.map_image.set_extent(extent)
            self.map_image.set_cmap(cmap)
            if vmin is not None and vmax is not None:
                self.map_image.set_clim(vmin, vmax)
            else:
                self.map_image.autoscale()
            self.ax.set_xlabel("Detector X (source pixels)")
            self.ax.set_ylabel("Detector Y (source pixels)")
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True)); self.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)

        show_cbar = bool(self._display_settings.get("show_colorbar", True))
        if show_cbar:
            if self.cbar is None:
                divider = make_axes_locatable(self.ax); self.cax = divider.append_axes("right", size="4.2%", pad=0.055)
                self.cbar = self.figure.colorbar(self.map_image, cax=self.cax)
            else: self.cbar.update_normal(self.map_image)
            formatter = ScalarFormatter(useMathText=True); formatter.set_powerlimits((-3, 4))
            self.cbar.ax.yaxis.set_major_formatter(formatter); self.cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        elif self.cbar is not None:
            self.cbar.remove(); self.cbar = None; self.cax = None
        self._apply_axes_theme(); self._update_scalebar()
        if first_draw and self.roi_btn.isChecked(): self._apply_current_roi_tool(active=True)
        self.canvas.draw_idle()

    @staticmethod
    def _hollow_picker(artist, event) -> tuple[bool, dict]:
        """
        Статический пикер с постоянным жизненным циклом.
        Перехватывает клики только на контуре (границе) рамки.
        """
        if event.xdata is None or event.ydata is None:
            return False, {}
        w2 = artist._roi_width / 2.0
        h2 = artist._roi_height / 2.0
        u, v = artist._roi_inv_trans.transform((event.xdata, event.ydata)) # Перевод координат мыши в локальную плоскую систему координат (u, v)
                                                                           # с центром в (0,0) и без учета поворота рамки с помощью обратной матрицы
        # (динамически подстраивается под шаг сетки):
        EDGE_TOLERANCE = float(getattr(artist, "_roi_edge_tolerance", 1.5))   # Толерантность захвата границы (в единицах осей - пикселях матрицы).
                                                                              # Если матрица 256x256, значение 1.5 означает полтора пикселя в обе стороны.        
        in_outer = (abs(u) <= w2 + EDGE_TOLERANCE) and (abs(v) <= h2 + EDGE_TOLERANCE)                # Проверяем попадание во внешний габарит (границы рамки + допуск)
        in_inner = (abs(u) < max(0, w2 - EDGE_TOLERANCE)) and (abs(v) < max(0, h2 - EDGE_TOLERANCE))  # Проверяем попадание во внутреннюю пустоту (границы рамки - допуск)
        is_hit = in_outer and not in_inner                                                            # Клик валиден ТОЛЬКО если мы в габарите, но НЕ в пустоте
        return is_hit, {}




    def draw_rois(self, roi_data_list: list) -> dict:
        """Отрисовывает квадратики, линии и точки ROI поверх карты."""
        # 1. Безопасная очистка: удаляем только те объекты, которые принадлежат ROI
        [p.remove() for p in reversed(self.ax.patches) if hasattr(p, '_roi_id')]
        [l.remove() for l in reversed(self.ax.lines) if hasattr(l, '_roi_id')]
        
        map_patches = {}        
        for data in roi_data_list:
            roi_id = data['id']
            roi_type = data.get('type', 'rect')  # По умолчанию rect для совместимости
            
            # Общие визуальные параметры выносим наверх
            lw = 2.2 if data['is_selected'] else 1.6
            alpha = 1.0 if data['is_selected'] else 0.85
            color = data['color']
            
            # --- ТОЧКА ---
            if roi_type == "point":
                point = Line2D(
                    [data['cx']], [data['cy']], 
                    marker='+', markersize=12, 
                    color=color, markeredgewidth=lw, 
                    linestyle='None', alpha=alpha
                )
                point._roi_id = roi_id
                point.set_picker(5)  # Стандартный радиус клика 5 пикселей (т.к. у точки нет рамки)
                self.ax.add_line(point)
                map_patches[roi_id] = point

            # --- ЛИНИЯ ---
            elif roi_type == "line":
                corners = data['corners']
                line = Line2D(
                    corners[:, 0], corners[:, 1], 
                    color=color, linewidth=lw, alpha=alpha
                )
                line._roi_id = roi_id
                line.set_picker(5)   # Радиус захвата линии 5 пикселей
                self.ax.add_line(line)
                map_patches[roi_id] = line

            # --- ПРЯМОУГОЛЬНИК (Твоя идеальная логика без изменений) ---
            else:
                rect = Rectangle(
                    (-data['width'] / 2.0, -data['height'] / 2.0),
                    data['width'],
                    data['height'],
                    fill=False,
                    linewidth=lw,
                    linestyle="-",
                    edgecolor=color,
                    alpha=alpha,
                )
                # Изолируем аффинное преобразование
                affine = Affine2D().rotate_deg(data['angle']).translate(data['cx'], data['cy'])
                rect.set_transform(affine + self.ax.transData)
                
                # Инжектируем атрибуты геометрии для пикера
                rect._roi_id = roi_id 
                rect._roi_width = data['width']
                rect._roi_height = data['height']
                rect._roi_inv_trans = affine.inverted()
                dx = float(np.diff(self._x_coords)[0]) if len(self._x_coords) > 1 else 1.0
                dy = float(np.diff(self._y_coords)[0]) if len(self._y_coords) > 1 else 1.0
                rect._roi_edge_tolerance = max(1.5, dx / 2, dy / 2)
                
                rect.set_picker(self._hollow_picker)  # Здесь оставляем твой умный пикер!
                self.ax.add_patch(rect)
                map_patches[roi_id] = rect

        self.canvas.draw_idle()
        return map_patches
    


    # ========================================================
    # ЕДИНЫЕ ОБРАБОТЧИКИ СОБЫТИЙ МЫШИ
    # ========================================================
    def _on_mouse_press(self, event) -> None:
        """Единый обработчик нажатия кнопок мыши"""
        if event.inaxes != self.ax:
            return
            
        # --- ПРАВЫЙ КЛИК (button == 3 - левая кнопка): Логика ROI ---
        if event.button == 3:
            clicked_roi_id = None                     # Если кликнули по карте: ищем, попали ли в прямоугольник
            for patch in reversed(self.ax.patches):   # reversed, чтобы сначала проверять верхние
                contains, _ = patch.contains(event)
                if contains and hasattr(patch, "_roi_id"):
                    clicked_roi_id = patch._roi_id
                    break        
            if self.roi_right_click_callback:                    # Если нашли ROI и окно подписано на это событие - передаем ID
                self.roi_right_click_callback(clicked_roi_id)    # Если кликнули в пустое место графика, передастся None.
        
        # --- СРЕДНИЙ КЛИК (button == 2 - зажато колесико): Начало Pan (перемещения) ---
        elif event.button == 2:
            self._pan_active = True
            self._pan_start_x_px = event.x    # ВАЖНО: Берем event.x и event.y (физические пиксели экрана), а не xdata!
            self._pan_start_y_px = event.y
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)  # Курсор-кулачок
        # В выключенном состоянии карта нейтральна: клик не выбирает пиксель и не строит кривую.

    def _on_mouse_move(self, event) -> None:
        """Единый обработчик движения мыши (Координаты + Перемещение)"""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        # 1. ЛОГИКА ТЕКСТА КООРДИНАТ (работает всегда) (X, Y и интенсивность Z под курсором)
        x, y = event.xdata, event.ydata   
        z_str = ""
        if hasattr(self, 'map_image') and self.map_image is not None:
            arr = self.map_image.get_array()
            x_axis = getattr(self, "_x_coords", np.arange(arr.shape[1], dtype=float))
            y_axis = getattr(self, "_y_coords", np.arange(arr.shape[0], dtype=float))
            ix = self._nearest_axis_index(x_axis, x); iy = self._nearest_axis_index(y_axis, y)
            if ix is not None and iy is not None:
                z_val = arr[iy, ix]
                if z_val is np.ma.masked or z_val is None:
                    z_str = "  |  Int: N/A"
                else:
                    z_str = f"  |  Int: {float(z_val):.1f}"
        coord_text = f"X: {x:.1f}   Y: {y:.1f}{z_str}"
        self.coord_label.setText(coord_text)

        # 2. ЛОГИКА ПЕРЕМЕЩЕНИЯ КАРТЫ (работает только если зажато колесико)
        if self._pan_active and event.x is not None and event.y is not None:
            # dx = event.xdata - self._pan_start_x
            # dy = event.ydata - self._pan_start_y
            # Считаем сдвиг мыши в пикселях монитора
            dx_px = event.x - self._pan_start_x_px
            dy_px = event.y - self._pan_start_y_px            
            # self.ax.set_xlim([self._pan_start_xlim[0] - dx, self._pan_start_xlim[1] - dx])
            # self.ax.set_ylim([self._pan_start_ylim[0] - dy, self._pan_start_ylim[1] - dy])
            # self.canvas.draw_idle()
            # Вычисляем масштаб: сколько единиц данных в одном пикселе
            bbox = self.ax.bbox
            x_scale = (self._pan_start_xlim[1] - self._pan_start_xlim[0]) / bbox.width
            y_scale = (self._pan_start_ylim[1] - self._pan_start_ylim[0]) / bbox.height
            
            # Переводим пиксели в данные и сдвигаем
            self.ax.set_xlim([self._pan_start_xlim[0] - dx_px * x_scale, 
                              self._pan_start_xlim[1] - dx_px * x_scale])
            self.ax.set_ylim([self._pan_start_ylim[0] - dy_px * y_scale, 
                              self._pan_start_ylim[1] - dy_px * y_scale])
            self.canvas.draw_idle()


    def _on_mouse_release(self, event):
        """Конец перемещения"""
        if event.button == 2 and self._pan_active:
            self._pan_active = False
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor) # Возвращаем обычную стрелку
            
    def _on_axes_leave(self, event) -> None:
        """Метод для слушателя мыши (координаты x,y). Если мышка ушла с карты, очищаем строку координат"""
        self.coord_label.setText("")



    # ========================================================
    # ЛОГИКА НАВИГАЦИИ (С ограничениями!)
    # ========================================================
    def reset_view(self):
        """Показывает полный extent текущего spatial-view."""
        if self.map_image is not None:
            x0, x1, y0, y1 = self.map_image.get_extent()
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y0, y1)
            self._update_scalebar(); self.canvas.draw_idle()

    def _on_scroll_zoom(self, event):
        """Приближение/Отдаление колесиком мыши туда, где находится курсор"""
        if event.inaxes != self.ax:
            return
            
        base_scale = 1.15                                                   # Насколько сильно зумит один "щелчок" колесика
        scale_factor = 1 / base_scale if event.step > 0 else base_scale     # event.step > 0 это скролл вверх (зум), < 0 - вниз (отдаление)

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Считаем новые лимиты осей (с центром в точке курсора)
        new_xlim = [xdata - (xdata - cur_xlim[0]) * scale_factor,
                    xdata + (cur_xlim[1] - xdata) * scale_factor]
        new_ylim = [ydata - (ydata - cur_ylim[0]) * scale_factor,
                    ydata + (cur_ylim[1] - ydata) * scale_factor]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._update_scalebar(); self.canvas.draw_idle()


    # ---------------------------------------------------------------------
    # Слоты для управления ROI-инструментами
    # ---------------------------------------------------------------------
    def _on_flyout_btn_clicked(self, btn: QToolButton) -> None:
        """Обработка клика по иконке внутри кастомного выпадающего меню."""
        self.roi_menu.close() # Прячем меню
        
        roi_id = btn.property("roi_id")
        if roi_id == self._current_roi_type:
            return
            
        self._current_roi_type = roi_id
        info = self.ROI_TOOLS_CONFIG[roi_id]
        # --- Обновляем главную кнопку ---
        self.roi_btn.setIcon(svg_icon(info["icon"], widget=self.roi_btn))       
        self.roi_btn.setToolTip(info["name"])
        # --- Переключаем "зажатость" кнопок внутри flyout ---
        for i in range(btn.parent().layout().count()):  
            child_btn = btn.parent().layout().itemAt(i).widget()
            if isinstance(child_btn, QToolButton) and child_btn != btn:
                child_btn.setChecked(False)
        btn.setChecked(True)
        # --- Синхронизируем главную кнопку-тумблер ---
        if not self.roi_btn.isChecked():
            self.roi_btn.setChecked(True)
        else:
            self._apply_current_roi_tool(active=True)        


    def _on_roi_toggled(self, checked: bool) -> None:
        """Срабатывает при клике на саму кнопку (включение/выключение селектора)."""
        self._apply_current_roi_tool(active=checked)

    
    def _apply_current_roi_tool(self, active: bool) -> None:
        """
        Единая точка управления селекторами. 
        Переключает активный инструмент на графике.
        Создает/удаляет/прячет нужный класс-наследник BaseROISelector.
        Работает без единого if/elif через словарь конфигурации.
        """
        # 1. Уничтожаем старый селектор, если он есть
        if self._current_selector is not None:
            self._current_selector.remove()
            self._current_selector = None
            
        # 2. Если инструмент выключили отжатием кнопки
        if not active:
            self.preview_roi_changed.emit(None)  # <-- ВАЖНО: говорим Главному окну стереть кривую
            self.canvas.draw_idle()
            return

        # 2. Достаем нужный класс прямо из словаря по ID
        config = self.ROI_TOOLS_CONFIG[self._current_roi_type]
        SelectorClass = config["selector_class"]
        
        if SelectorClass is not None:
            colors = theme_colors(self._theme)
            self._current_selector = SelectorClass(   # Создаем новый инстанс нужного селектора (RotatedRectangle, Line или Point)
                self.ax,
                on_change=self._on_selector_changed,
                onselect=self._on_selector_selected,  # чтобы откликивались и промежуточные движения, и завершение
                min_size=1.0,
                handle_size=7.0,
                rotation_handle_offset=18.0,
                edgecolor=colors["roi"],
                facecolor=to_rgba(colors["roi"], alpha=0.08),
                activecolor=colors["roi"],
                hovercolor=colors["roi_hover"],
                lw=2.0,
                allow_create=True,
                allow_move=True,
                allow_resize=True,
                allow_rotate=True,
                keep_aspect=False,
                snap_grid=0.0,
                snap_angle=0.0,
                bounds=None,
                use_blit=True,  # технология быстрой перерисовки только части экрана
            )
            self.canvas.draw_idle()
        else:
            print(f"Инструмент {self._current_roi_type} еще в разработке!")

    def _on_selector_changed(self, roi):
        """Передаёт наружу промежуточную геометрию, не вмешиваясь в расчёт кривой."""
        self.preview_roi_changed.emit(roi)

    def _on_selector_selected(self, roi):
        """Сообщает отдельно о завершении рисования/перемещения ROI."""
        self.preview_roi_changed.emit(roi)
        self.preview_roi_finished.emit(roi)
