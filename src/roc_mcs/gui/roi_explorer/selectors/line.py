from matplotlib.lines import Line2D
from typing import Optional, Tuple, Any, Dict
import math
import numpy as np

try:
    from matplotlib.backend_bases import Cursors
except ImportError:
    Cursors = None  # type: ignore

from roc_mcs.gui.roi_explorer.selectors.enums import Action
from roc_mcs.gui.roi_explorer.selectors.base import BaseROISelector
from roc_mcs.roi.roi_types import LineROI
from roc_mcs.gui.roi_explorer.selectors.enums import Action, LineHandle


class LineSelector(BaseROISelector):
    """
    Этот класс управляет тремя артистами: 
    самой линией и двумя квадратными маркерами на концах.
    """
    # --- Рисуем селектор-линию ---
    def _create_artists(self) -> None:
        """Создаем графические элементы для линии и её концов."""
        # Сама линия
        self._line = Line2D(
            [], [], linestyle="-", color=self.edgecolor, 
            linewidth=self.lw, visible=False, zorder=4
        )
        self._line.set_clip_on(True)
        self.ax.add_line(self._line)

        # Ручки на концах (P1 и P2)
        self._handles: Dict[LineHandle, Line2D] = {}
        for hname in [LineHandle.P1, LineHandle.P2]:
            h = Line2D(
                [], [], linestyle="", marker="s", markersize=self.handle_size,
                markerfacecolor="white", markeredgecolor=self.edgecolor,
                markeredgewidth=1.2, visible=False, zorder=6,
            )
            h.set_clip_on(True)
            self.ax.add_line(h)
            self._handles[hname] = h

        # Список для blitting (передается в BaseROISelector)
        self._animated_artists = [self._line, *self._handles.values()]

    # ---------------------------------------------------------------------
    # Geometry & Hit Testing (В экранных пикселях!)
    # ---------------------------------------------------------------------
    def _dist_to_segment_px(self, px: float, py: float, roi: LineROI) -> float:
        """Вычисляет кратчайшее расстояние от курсора мыши до отрезка в пикселях дисплея."""
        p1 = self.ax.transData.transform((roi.x1, roi.y1))
        p2 = self.ax.transData.transform((roi.x2, roi.y2))
        p = np.array([px, py], dtype=float)
        
        v = p2 - p1
        w = p - p1
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            return float(math.hypot(px - p1[0], py - p1[1]))
            
        c2 = np.dot(v, v)
        if c2 <= c1:
            return float(math.hypot(px - p2[0], py - p2[1]))
            
        b = c1 / c2
        pb = p1 + b * v
        return float(math.hypot(px - pb[0], py - pb[1]))

    def _event_in_roi(self, event) -> bool:
        """Проверяет, попал ли клик по самой линии."""
        if self.roi is None or event.x is None or event.y is None:
            return False
        if not self._line.get_visible():
            return False

        dist_px = self._dist_to_segment_px(event.x, event.y, self.roi)
        return dist_px <= self.hover_tolerance_px

    def _hit_test_handle(self, event, tol_px: Optional[float] = None) -> Optional[LineHandle]:
        """Проверяет, схватили ли мы один из концов линии."""
        if self.roi is None or event.x is None or event.y is None:
            return None
        if tol_px is None:
            tol_px = self.hover_tolerance_px

        # Координаты ручек в данных
        positions = {
            LineHandle.P1: (self.roi.x1, self.roi.y1),
            LineHandle.P2: (self.roi.x2, self.roi.y2)
        }

        for name, (x_data, y_data) in positions.items():
            px, py = self.ax.transData.transform((x_data, y_data))
            if math.hypot(event.x - px, event.y - py) <= tol_px:
                return name
        return None

    def _resize_from_handle(
        self,
        base: LineROI,
        handle: LineHandle,
        mouse_local: Tuple[float, float],  # Предполагается, что Base передает координаты данных
        *,
        keep_aspect: bool,
        from_center: bool,
    ) -> LineROI:
        """Перемещает один из концов линии."""
        mx, my = mouse_local
        
        if handle == LineHandle.P1:
            return LineROI(x1=mx, y1=my, x2=base.x2, y2=base.y2)
        elif handle == LineHandle.P2:
            return LineROI(x1=base.x1, y1=base.y1, x2=mx, y2=my)
        
        return base

    def _constrain_roi(self, roi: LineROI) -> LineROI: # 🟨
        """
        Удержание концов отрезка в пределах осей графика и привязка к сетке.
        У линии нет ширины и высоты, только две точки: P₁(x₁, y₁) и P₂(x₂, y₂).
        Значит, ограничивать и привязывать к сетке нужно независимо каждую из них.
        """
        r = roi.copy()
        # --- Snapping (привязка к сетке) ---
        if self.snap_grid > 0:
            def snap(val: float) -> float:
                return round(val / self.snap_grid) * self.snap_grid
            r.x1, r.y1 = snap(r.x1), snap(r.y1)
            r.x2, r.y2 = snap(r.x2), snap(r.y2)
            
        # Примечание: snap_angle для произвольной линии реализовать можно 
        # (вращая r.x2, r.y2 вокруг r.x1, r.y1), но обычно для отрезка это избыточно.

        # --- Bounds (контроль границ) ---
        bounds = self._axes_bounds()
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            def clamp(val: float, vmin: float, vmax: float) -> float:
                """ Жестко отсекаем координаты, чтобы концы не выходили за оси """
                return max(vmin, min(val, vmax))
            r.x1 = clamp(r.x1, xmin, xmax)
            r.y1 = clamp(r.y1, ymin, ymax)
            r.x2 = clamp(r.x2, xmin, xmax)
            r.y2 = clamp(r.y2, ymin, ymax) 
        return r
        
    # ---------------------------------------------------------------------
    # Artist updates
    # ---------------------------------------------------------------------
    def _update_artists(self) -> None:
        if self.roi is None:
            self._line.set_visible(False)
            for h in self._handles.values():
                h.set_visible(False)
            return

        roi = self.roi
        # Обновляем линию
        self._line.set_data([roi.x1, roi.x2], [roi.y1, roi.y2])
        self._line.set_visible(True)
        self._line.set_color(self._current_edgecolor())
        self._line.set_linewidth(self.lw + (0.5 if self._action != Action.IDLE else 0.0))

        # Обновляем ручки
        edge = self._current_edgecolor()
        positions = {LineHandle.P1: (roi.x1, roi.y1), LineHandle.P2: (roi.x2, roi.y2)}
        
        for name, artist in self._handles.items():
            x, y = positions[name]
            artist.set_data([x], [y])
            artist.set_visible(True)
            artist.set_markeredgecolor(edge)

    def contains(self, x: float, y: float) -> bool:
        """
        Метод для интеграции: проверка, находится ли точка (в координатах данных)
        достаточно близко к линии. 
        """
        if self.roi is None:
            return False
        # Для проверки в данных конвертируем в пиксели
        p_disp = self.ax.transData.transform((x, y))
        dist_px = self._dist_to_segment_px(p_disp[0], p_disp[1], self.roi)
        return dist_px <= self.hover_tolerance_px

    # ---------------------------------------------------------------------
    # State / cursor helpers
    # ---------------------------------------------------------------------
    def _cursor_for_handle(self, handle: Optional[LineHandle]) -> Optional[Any]:  # 🟨
        """Возвращает курсор в зависимости от того, над какой ручкой находится мышь."""
        if Cursors is None or handle is None:
            return None
        # Использование словаря делает логику масштабируемой (O(1) поиск)
        mapping = {        # Для концов линии отлично подходит крестик или рука
            LineHandle.P1: getattr(Cursors, "CROSSHAIR", None),
            LineHandle.P2: getattr(Cursors, "CROSSHAIR", None),
        }
        return mapping.get(handle, getattr(Cursors, "POINTER", None))  # Дефолтный fallback, если ручка по какой-то причине не найдена


    # -------------------------------------------------------------------------
    # Реализация геометрических хуков (специфично для ЛИНИИ) 🟨
    # -------------------------------------------------------------------------
    def _process_press(self, event) -> bool:
        """
        Определяет, за что схватились: за конец отрезка, за саму линию, или кликнули в пустоту.
        """
        handle = self._hit_test_handle(event, tol_px=self.hover_tolerance_px)
        
        # 1. Попали в один из концов отрезка (P1 или P2)?
        if handle is not None and self.roi is not None:
            if self.allow_resize:  # Для линии изменение длины/угла — это resize
                self._action = Action.RESIZING
                self._active_handle = handle
                self._press_roi = self.roi.copy()
                return True

        # 2. Попали в саму линию (перемещение целиком)?
        if self.roi is not None and self.allow_move and self._event_in_roi(event):
            self._action = Action.MOVING
            self._press_roi = self.roi.copy()
            self._active_handle = None
            return True

        # 3. Кликнули в пустоту — создаем новую линию
        if self.allow_create:
            # Начальная и конечная точки совпадают
            # self.roi = LineROI(event.xdata, event.ydata, event.xdata, event.ydata)
            # self._press_roi = self.roi.copy()
            self._press_roi = LineROI(event.xdata, event.ydata, event.xdata, event.ydata)   # ⚓ Сохраняем математический "якорь" для старта в _press_roi, 
            self.roi = None                                                                 # но оставляем self.roi = None. Фигура не отрисуется до первого сдвига мыши!
            
            self._action = Action.CREATING
            self._active_handle = LineHandle.P2  # МАГИЯ: при создании мы сразу притворяемся, что тянем за второй конец! 
            self._update_artists()           # Полная отрисовка
            self._emit_change()
            self._request_redraw(full=True)
            return False                     # База может отдыхать
            
        return False

    def _process_motion(self, event) -> bool:
        """
        Математика перетаскивания концов отрезка или линии целиком.
        Главная фишка отрезка: создание линии — это то же самое, 
        что изменение её размера (RESIZING) за вторую точку (P2). 
        Мы можем использовать этот факт, чтобы не дублировать код в _process_motion.
        """
        base = self._press_roi
        
        # CREATING и RESIZING для линии обрабатываются абсолютно одинаково, 
        # так как при создании мы установили active_handle = P2.
        if self._action in (Action.CREATING, Action.RESIZING):
            r = base.copy()
            if self._active_handle == LineHandle.P1:
                r.x1, r.y1 = event.xdata, event.ydata
            elif self._active_handle == LineHandle.P2:
                r.x2, r.y2 = event.xdata, event.ydata
                
            self.roi = self._constrain_roi(r)
            return True

        elif self._action == Action.MOVING:
            dx = event.xdata - self._press_event.xdata
            dy = event.ydata - self._press_event.ydata
            
            r = base.copy()
            r.x1 += dx
            r.y1 += dy
            r.x2 += dx
            r.y2 += dy
            
            self.roi = self._constrain_roi(r)
            return True
        return False

    def _process_release(self, event) -> bool:
        """Проверка, не выродилась ли линия в точку (длина меньше min_size)."""
        if self.roi is None:
            return False
        dx = self.roi.x2 - self.roi.x1
        dy = self.roi.y2 - self.roi.y1
        length = math.hypot(dx, dy)      # Используем math.hypot для быстрого вычисления длины вектора
        return length >= self.min_size