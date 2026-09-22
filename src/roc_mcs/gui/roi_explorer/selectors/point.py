from matplotlib.lines import Line2D
from typing import Optional, Tuple, Any
import math

from roc_mcs.gui.roi_explorer.selectors.enums import Action
from roc_mcs.gui.roi_explorer.selectors.base import BaseROISelector
from roc_mcs.roi.roi_types import PointROI


class PointSelector(BaseROISelector):
    """
    Точка — это самый простой селектор. 
    У неё нет ручек изменения размера (_hit_test_handle всегда возвращает None), 
    она просто перемещается целиком.
    """
    # --- Рисуем селектор-точку ---
    def _create_artists(self) -> None:
        """Создаем один маркер, символизирующий точку."""
        self._point = Line2D(
            [], [], linestyle="", marker="+", markersize=12,  # Крестик удобнее для прицеливания
            markeredgecolor=self.edgecolor, markeredgewidth=self.lw,
            visible=False, zorder=5
        )
        self._point.set_clip_on(True)
        self.ax.add_line(self._point)

        self._animated_artists = [self._point]

    # ---------------------------------------------------------------------
    # Geometry & Hit Testing
    # ---------------------------------------------------------------------
    def _event_in_roi(self, event) -> bool:
        """Попали ли мы курсором в окрестность точки."""
        if self.roi is None or event.x is None or event.y is None:
            return False
        if not self._point.get_visible():
            return False

        px, py = self.ax.transData.transform((self.roi.x, self.roi.y))
        dist_px = math.hypot(event.x - px, event.y - py)
        return dist_px <= self.hover_tolerance_px

    def _hit_test_handle(self, event, tol_px: Optional[float] = None) -> None:
        """У точки нет ручек масштабирования, ее можно только двигать."""
        return None

    def _resize_from_handle(self, base: PointROI, handle: any, mouse_local: Tuple[float, float], **kwargs) -> PointROI:
        """Этот метод никогда не будет вызван, так как _hit_test_handle возвращает None."""
        return base

    def _constrain_roi(self, roi: PointROI) -> PointROI: # 🟨
        """
        Удержание точки в пределах видимых осей и привязка к сетке.
        Нам нужно зажать всего одну координату (x, y).
        """
        r = roi.copy()
        # --- Snapping (привязка к сетке) ---
        if self.snap_grid > 0:
            r.x = round(r.x / self.snap_grid) * self.snap_grid
            r.y = round(r.y / self.snap_grid) * self.snap_grid

        # --- Bounds (контроль границ) ---
        bounds = self._axes_bounds()
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            r.x = min(max(r.x, xmin), xmax)
            r.y = min(max(r.y, ymin), ymax)
        return r
        
    # ---------------------------------------------------------------------
    # Artist updates
    # ---------------------------------------------------------------------
    def _update_artists(self) -> None:
        if self.roi is None:
            self._point.set_visible(False)
            return

        self._point.set_data([self.roi.x], [self.roi.y])
        self._point.set_visible(True)
        self._point.set_markeredgecolor(self._current_edgecolor())
        
        # Утолщаем при взаимодействии
        active_lw = self.lw + (1.0 if self._action != Action.IDLE else 0.0)
        self._point.set_markeredgewidth(active_lw)

    def contains(self, x: float, y: float) -> bool:
        """Точное совпадение не имеет смысла, берем радиус допуска."""
        if self.roi is None:
            return False
        px, py = self.ax.transData.transform((self.roi.x, self.roi.y))
        p_test = self.ax.transData.transform((x, y))
        dist_px = math.hypot(p_test[0] - px, p_test[1] - py)
        return dist_px <= self.hover_tolerance_px

    # ---------------------------------------------------------------------
    # State / cursor helpers
    # ---------------------------------------------------------------------
    def _cursor_for_handle(self, handle) -> Optional[Any]: # 🟨
        """
        У точечного селектора нет ручек масштабирования (всегда возвращается None).
        Поэтому здесь не должно быть логики с LineHandle.
        Смена курсора при наведении на саму точку должна обрабатываться 
        на уровне BaseROISelector (например, через возврат Cursors.MOVE).
        """
        return None


    # -------------------------------------------------------------------------
    # Реализация геометрических хуков (специфично для ТОЧКИ) 🟨
    # У точки нет вращения, нет ручек ресайза. Только создание и перемещение.
    # -------------------------------------------------------------------------
    def _process_press(self, event) -> bool:
        """Определяет: кликнули в существующую точку для переноса, или ставим новую."""
        # 1. Попали в существующую точку? (считаем это телом фигуры)
        if self.roi is not None and self.allow_move and self._event_in_roi(event):
            self._action = Action.MOVING
            self._press_roi = self.roi.copy()
            self._active_handle = None
            return True

        # 2. Ставим новую точку
        if self.allow_create:
            self.roi = PointROI(event.xdata, event.ydata)
            self._press_roi = self.roi.copy()
            self._action = Action.CREATING
            self._active_handle = None
            
            self._update_artists()
            self._emit_change()
            self._request_redraw(full=True)
            return False
            
        return False

    def _process_motion(self, event) -> bool:
        """Обновление координат точки при перетаскивании."""
        if self._action in (Action.CREATING, Action.MOVING):
            # Для точки логика едина: она просто следует за курсором.
            # Если нужно учитывать смещение при клике (dx/dy), можно считать от base,
            # но для точки обычно логичнее жестко привязывать её центр к курсору.
            
            r = self._press_roi.copy()
            if self._action == Action.MOVING:
                # Оставляем плавное перетаскивание с учетом того, где именно внутри
                # радиуса точки произошел клик (чтобы точка не "прыгала" центром под мышь)
                dx = event.xdata - self._press_event.xdata
                dy = event.ydata - self._press_event.ydata
                r.x += dx
                r.y += dy
            else:
                # При создании просто ставим под курсор
                r.x = event.xdata
                r.y = event.ydata
                
            self.roi = self._constrain_roi(r)
            return True
            
        return False

    def _process_release(self, event) -> bool:
        """Для точки любая валидная координата имеет смысл. Главное, чтобы объект существовал."""
        return self.roi is not None