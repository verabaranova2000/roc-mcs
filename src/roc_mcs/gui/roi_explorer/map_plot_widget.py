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
import numpy as np

from roc_mcs.roi.spatial_pooling import PoolSpec
from  roc_mcs.gui.roi_explorer.map_tools_widget import MapToolsWidget, pooling_preview_pixmap as _pooling_preview_pixmap
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR
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
    QComboBox {
        border: none;
        background: transparent;
        font-size: 10px;
        font-family: 'DejaVu Sans', 'Arial', sans-serif;
        font-weight: bold;
        padding: 2px;
        color: #000000; 
    }
    QComboBox::drop-down { border: none; width: 0px; } /* Прячем стрелочку для чистоты */
    QComboBox:hover { color: #000080; } /* Ink blue акцент при наведении */
    QComboBox QAbstractItemView {
        border: 1px solid #cccccc;
        background: #ffffff;
        selection-background-color: #e6f2ff;
        selection-color: #000080;
        min-width: 250px; /* <--- ЭТА СТРОКА РЕШАЕТ ПРОБЛЕМУ обрезки текста "..." в выпадающих списках */
    }
"""

# Стиль для строки координат мыши в подвале графика (моноширинный научный шрифт)
# Здесь QLabel#MapCoordLabel означает: «Примени этот стиль к любому QLabel, у которого ID равен MapCoordLabel»
STYLE_COORD_LABEL = """
    QLabel#MapCoordLabel {
        font-family: monospace;
        font-size: 10px;
        font-weight: bold;
        color: #000080;
        padding: 2px 5px;
        background-color: transparent;
    }
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
    pin_roi_requested = pyqtSignal()         # сигнал умного списка ROI
    preview_roi_changed = pyqtSignal(object) # Сигнал: передает объект ROI (LineROI, PointROI, RectROI) наверх

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
        self.repo = None                    # Ссылка на данные
        self._pool_spec = PoolSpec()

        self.setMinimumWidth(300)           # минимальные размеры для всего виджета. Сплиттер физически упрется в этот лимит и не сломает Matplotlib
        self.setMinimumHeight(350)
        
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
        self.figure = Figure(figsize=(5, 5)) 
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.figure.subplots_adjust(left=0.12, right=0.85, bottom=0.1, top=0.99)            # Жесткие отступы. Цифры будут расти в правое поле (right=0.85), не толкая график
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)   # жесткая фиксация высоты как в моем классе PLOTBOX!
        # self.canvas.setFixedHeight(450)  # Подобрать цифру, при которой карта выглядит идеально квадратной
        
        self.ax = self.figure.add_subplot(1, 1, 1)
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
                     "low_pct", "high_pct", "range_slider"):
            setattr(self, name, getattr(self.tools, name))
        self.tools.reset_view_requested.connect(self.reset_view)
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
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_layout.setContentsMargins(10, 4, 10, 0)   # Отступы: чуть-чуть сверху, чтобы отделить от тулбара, снизу 0, чтобы прижать к графику
        self.header_layout.setSpacing(6)

        self.scan_combo = QComboBox()
        self.map_combo = QComboBox()
        self.reduction_combo = QComboBox()
        self.reduction_combo.addItems(["mean", "sum"])
        # Подсказки к полям заголовка
        self.scan_combo.setToolTip("Выбор набора данных (скана) из HDF5 архива.")
        self.map_combo.setToolTip("Выбор типа 2D карты (распределение интенсивности, ширины пика, положения и др.).")
        self.reduction_combo.setToolTip("Режим агрегации пикселей (mean/sum) при проецировании ROI в 1D профиль.")
        
        for combo in (self.scan_combo, self.map_combo, self.reduction_combo):
            combo.setStyleSheet(STYLE_FLAT_COMBOBOX)                                 # применяем константу стилей
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)   # ВАЖНО: Исправлено для PyQt6. Список сам подгоняет ширину под текст.
        
        def create_separator():
            """Создает вертикальный разделитель ' | ' для умного заголовка."""
            lbl = QLabel(" | ")
            lbl.setStyleSheet("font-size: 15px; font-weight: normal; color: #555555;")
            return lbl
    
        # Собираем заголовок по центру
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.scan_combo)
        self.header_layout.addWidget(create_separator())
        self.header_layout.addWidget(self.map_combo)
        self.header_layout.addWidget(create_separator())
        self.header_layout.addWidget(self.reduction_combo)
        self.header_layout.addStretch()

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
        self.main_layout.addWidget(self.coord_label)     # 4. Строка координат мыши (x,y)
        self.main_layout.addStretch(1)                   # 5. Пружина (ставим в самый низ.  Она заберет на себя всё свободное серое пространство в нижней части окна.

 
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

    
    # ============================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ КЛАССА
    # ============================================================
    def _on_cmap_selected(self, cmap_name: str) -> None:
        self.current_cmap = cmap_name
        self.low_pct.valueChanged.emit(self.low_pct.value())   # вызываем перерисовку холста (сигнал будет подхвачен Главным окном)

    def _on_pool_size_changed(self, value: int) -> None:
        self._keep_pool_size_odd(value)
        self._update_pool_preview()

    def _update_pool_preview(self, *_args) -> None:
        if hasattr(self, "pool_preview_label"):
            self.pool_preview_label.setPixmap(
                _pooling_preview_pixmap(self.pool_size_spin.value(), self.pool_stride_spin.value())
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
        self.pool_btn.setStyleSheet(
            "SmartDropdownButton::menu-indicator { image: none; }" +
            (" QToolButton { color: #000080; background: #e6f2ff; border: 1px solid #000080; font-weight: bold; }" if pool.active else "")
        )
        
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
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)

        if self.cbar is None:
            divider = make_axes_locatable(self.ax)
            self.cax = divider.append_axes("right", size="5%", pad=0.08)
            self.cbar = self.figure.colorbar(self.map_image, cax=self.cax)
        else:
            self.cbar.update_normal(self.map_image)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        self.cbar.ax.yaxis.set_major_formatter(formatter)
        self.cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if first_draw and self.roi_btn.isChecked():
            self._apply_current_roi_tool(active=True)
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
    

    # --- Авто-адаптация квадрата ---
    def resizeEvent(self, event):
        """
        Перехватываем изменение размера виджета (например, работу сплиттера).
        Динамически заставляем высоту холста Matplotlib равняться его ширине.
        """
        super().resizeEvent(event)
        current_width = self.width()   # Узнаем текущую ширину виджета (которую нам выделил сплиттер)
        # Принудительно делаем высоту холста равной его ширине (идеальный квадрат).
        # А нижняя пружина addStretch(1) из main_layout аккуратно заберет всю оставшуюся пустоту под себя!
        self.canvas.setFixedHeight(current_width)        

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
            self.canvas.draw_idle()

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
        self.canvas.draw_idle()


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
        self.roi_btn.setIcon(QIcon(info["icon"]))       
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
            self._current_selector = SelectorClass(   # Создаем новый инстанс нужного селектора (RotatedRectangle, Line или Point)
                self.ax,
                on_change=self._on_selector_changed,
                onselect=self._on_selector_changed,  # чтобы откликивались и промежуточные движения, и завершение
                min_size=1.0,
                handle_size=7.0,
                rotation_handle_offset=18.0,
                edgecolor="white",
                facecolor=(1.0, 1.0, 1.0, 0.05),
                activecolor="#ffd54a",
                hovercolor="#7ad7ff",
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
        """
        Прокси-метод передачи объекта ROI наружу.
        Селектор изменил геометрию (потянули за ручку) -> передаем объект ROI наружу.
        MapPlotWidget не "лезет" в главное окно и не пытается самостоятельно вызывать расчет кривых качания.
        Он лишь говорит во внешний мир: 
            «Привет! У меня тут пользователь потянул за ручку ROI, вот новые координаты (object), делайте с ними что хотите»
        RockingCurveExplorerWindow поймает это через self.map_widget.preview_roi_changed.connect(...)
        """
        self.preview_roi_changed.emit(roi)