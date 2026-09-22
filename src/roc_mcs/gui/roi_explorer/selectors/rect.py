from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from typing import Optional, Tuple, Any, Dict
import math
import numpy as np



try:
    from matplotlib.backend_bases import Cursors
except ImportError:
    Cursors = None  # type: ignore

from roc_mcs.roi.roi_types import RotatedRectROI
from roc_mcs.gui.roi_explorer.selectors.enums import Action, RectHandle
from roc_mcs.gui.roi_explorer.selectors.base import BaseROISelector


class RotatedRectangleSelector(BaseROISelector):
    _HANDLE_ORDER = (RectHandle.NW, RectHandle.N, RectHandle.NE, RectHandle.E, RectHandle.SE, RectHandle.S, RectHandle.SW, RectHandle.W)

    # --- Рисуем прямоугольный селектор ---
    def _create_artists(self) -> None:
        """Реализация создания графики специально для прямоугольника."""
        # Ставим .set_clip_on(True). Если True (по умолчанию), элемент обрезается рамкой осей (Axes)
        # --- Сам прямоугольник ---
        self._rect = Rectangle(
            (-0.5, -0.5), 1.0, 1.0, fill=True,
            facecolor=self.facecolor, edgecolor=self.edgecolor,
            linewidth=self.lw, visible=False, zorder=4,
        )
        self._rect.set_clip_on(True) 
        self.ax.add_patch(self._rect)

        # --- Квадратные ручки по краям ---
        self._handles: Dict[RectHandle, Line2D] = {}
        for hname in self._HANDLE_ORDER:
            h = Line2D(
                [], [], linestyle="", marker="s", markersize=self.handle_size,
                markerfacecolor="white", markeredgecolor=self.edgecolor,
                markeredgewidth=1.2, visible=False, zorder=6,
            )
            h.set_clip_on(True)
            self.ax.add_line(h)
            self._handles[hname] = h
            
        # --- Круглая ручка поворота ---
        self._rot_handle = Line2D(
            [], [],
            linestyle="", marker="o", markersize=self.handle_size,
            markerfacecolor="white", markeredgecolor=self.edgecolor,
            markeredgewidth=1.2, visible=False, zorder=7,
        )
        self._rot_handle.set_clip_on(True)
        self.ax.add_line(self._rot_handle)

        # --- Линия к ручке поворота ---
        self._rot_line = Line2D([], [], linestyle="-", linewidth=1.0, color=self.edgecolor, visible=False, zorder=5)
        self._rot_line.set_clip_on(True)
        self.ax.add_line(self._rot_line)

        # Заполняем список для blitting (Просто отдаем список базовому классу; там включится анимацию для графики, если используем blitting)
        self._animated_artists = [self._rect, self._rot_handle, self._rot_line, *self._handles.values()]

    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------
    # --- Общая логика перевода координат из данных на экран и обратно с учетом угла поворота ---
    #    Эти методы строятся на центрах и углах вращения рамки. (Для Линии аффинные преобразования вообще не нужны, там просто две точки)
    def _roi_affine(self, roi: RotatedRectROI) -> Affine2D:
        return Affine2D().rotate_deg(roi.angle).translate(roi.cx, roi.cy)

    def _data_to_local(self, x: float, y: float, roi: RotatedRectROI) -> Tuple[float, float]:
        local = self._roi_affine(roi).inverted().transform((x, y))
        return float(local[0]), float(local[1])

    def _local_to_data(self, ux: float, uy: float, roi: RotatedRectROI) -> Tuple[float, float]:
        data = self._roi_affine(roi).transform((ux, uy))
        return float(data[0]), float(data[1])

    def _rotated_half_extents(self, width: float, height: float, angle_deg: float) -> Tuple[float, float]:
        a = math.radians(angle_deg)
        c = abs(math.cos(a))
        s = abs(math.sin(a))
        ex = 0.5 * (width * c + height * s)
        ey = 0.5 * (width * s + height * c)
        return ex, ey       

    def _constrain_roi(self, roi: RotatedRectROI) -> RotatedRectROI: # 🟨
        """
        Контроль границ
        Clamp to min size, snap if requested, and keep inside bounds if enabled.
        """
        r = roi.copy()
        # --- minimum size ---
        r.width = max(self.min_size, float(r.width))
        r.height = max(self.min_size, float(r.height))
        r.angle = self._normalize_angle(r.angle)
        # --- snapping ---
        if self.snap_grid > 0:
            r.cx = round(r.cx / self.snap_grid) * self.snap_grid
            r.cy = round(r.cy / self.snap_grid) * self.snap_grid
            r.width = max(self.min_size, round(r.width / self.snap_grid) * self.snap_grid)
            r.height = max(self.min_size, round(r.height / self.snap_grid) * self.snap_grid)
        if self.snap_angle > 0:
            r.angle = round(r.angle / self.snap_angle) * self.snap_angle
            r.angle = self._normalize_angle(r.angle)
        # --- bounds ---
        bounds = self._axes_bounds()
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            ex, ey = self._rotated_half_extents(r.width, r.height, r.angle)

            # If the ROI is larger than the visible area, clamp the size conservatively.
            max_w = 2.0 * max(1e-12, xmax - xmin)
            max_h = 2.0 * max(1e-12, ymax - ymin)
            r.width = min(r.width, max_w)
            r.height = min(r.height, max_h)
            ex, ey = self._rotated_half_extents(r.width, r.height, r.angle)

            r.cx = min(max(r.cx, xmin + ex), xmax - ex)
            r.cy = min(max(r.cy, ymin + ey), ymax - ey)
        return r
    
    def _handle_positions(self, roi: RotatedRectROI) -> Dict[RectHandle, Tuple[float, float]]:
        w2 = roi.width / 2.0
        h2 = roi.height / 2.0
        local = {
            RectHandle.NW: (-w2, +h2),
            RectHandle.N: (0.0, +h2),
            RectHandle.NE: (+w2, +h2),
            RectHandle.E: (+w2, 0.0),
            RectHandle.SE: (+w2, -h2),
            RectHandle.S: (0.0, -h2),
            RectHandle.SW: (-w2, -h2),
            RectHandle.W: (-w2, 0.0),
        }
        R = self._rotmat(roi.angle)
        c = np.array([roi.cx, roi.cy], dtype=float)
        out: Dict[RectHandle, Tuple[float, float]] = {}
        for key, (ux, uy) in local.items():
            p = c + R @ np.array([ux, uy], dtype=float)
            out[key] = (float(p[0]), float(p[1]))
        return out

    
    def _roi_top_center(self, roi: RotatedRectROI) -> Tuple[float, float]:
        return self._local_to_data(0.0, roi.height / 2.0, roi)

    def _roi_rotate_handle(self, roi: _roi_rotate_handle) -> Tuple[float, float]:
        """ 
        Математика ручки вращения. 
        Берет заданную выше высоту ручки (18.0) и правильно конвертировать их в экранные размеры 
        с учетом плотности пикселей монитора.
        """
        top_x_data, top_y_data = self._roi_top_center(roi)
        # вектор направления
        dir_x_data, dir_y_data = self._local_to_data(0.0, roi.height / 2.0 + 1.0, roi)
        # В пиксели дисплея
        top_disp = self.ax.transData.transform((top_x_data, top_y_data))
        dir_disp = self.ax.transData.transform((dir_x_data, dir_y_data))
        
        vx = dir_disp[0] - top_disp[0]
        vy = dir_disp[1] - top_disp[1]
        v_len = math.hypot(vx, vy)
        if v_len < 1e-8:
            return top_x_data, top_y_data
            
        dpi = self.ax.figure.dpi
        offset_px = self.rotation_handle_offset * (dpi / 72.0)   # ПРАВИЛЬНЫЙ РАСЧЕТ РАЗМЕРА: переводим offset (пункты) в экранные пиксели (теперь длина ручки будет стабильной при зуме)
        
        hx_disp = top_disp[0] + (vx / v_len) * offset_px
        hy_disp = top_disp[1] + (vy / v_len) * offset_px

        hx_data, hy_data = self.ax.transData.inverted().transform((hx_disp, hy_disp))  # Возвращаем в данные
        return float(hx_data), float(hy_data)

    def _event_in_roi(self, event) -> bool:
        """
        Проверка, попали ли мы в рамку.
        Внутри него есть явное обращение к self._rect (которого нет в базе), 
        а математика рассчитывает попадание курсора в прямоугольник (w2, h2).
        """
        if self.roi is None or event.xdata is None or event.ydata is None:
            return False
        if not self._rect.get_visible():
            return False
            
        u, v = self._data_to_local(event.xdata, event.ydata, self.roi)
        
        # Физический размер допуска в масштабе данных
        p0 = self.ax.transData.inverted().transform((event.x, event.y))
        p1 = self.ax.transData.inverted().transform((event.x + self.hover_tolerance_px, event.y))
        tol_data = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        
        w2 = self.roi.width / 2.0
        h2 = self.roi.height / 2.0
        
        # 1. Внешний контур (захватываем площадь рамки + допуск наружу)
        in_outer = (abs(u) <= w2 + tol_data) and (abs(v) <= h2 + tol_data)
        
        # 2. Внутреннее "ядро" (свободная зона для создания новых рамок).
        # Отступаем от краев внутрь на величину допуска.
        # max(0.0, ...) защищает от ошибок, если рамка стянута в микроскопическую точку.
        inner_w = max(0.0, w2 - tol_data)
        inner_h = max(0.0, h2 - tol_data)
        in_inner = (abs(u) < inner_w) and (abs(v) < inner_h)
        # Мы попали по рамке ТОЛЬКО если курсор в зоне внешнего контура, но НЕ провалился во внутреннюю пустоту.
        return in_outer and not in_inner  
        
    def _hit_test_handle(self, event, tol_px: Optional[float] = None) -> Optional[RectHandle]:
        """
        Перебирает ручки из списка self._HANDLE_ORDER, который мы определили именно в классе прямоугольника.
        """
        if self.roi is None or event.x is None or event.y is None:
            return None
        if tol_px is None:
            tol_px = self.hover_tolerance_px

        positions = self._handle_positions(self.roi)
        for name in list(self._HANDLE_ORDER) + [RectHandle.ROTATE]:
            if name == RectHandle.ROTATE:
                x, y = self._roi_rotate_handle(self.roi)
            else:
                x, y = positions[name]
            px, py = self.ax.transData.transform((x, y))
            if math.hypot(event.x - px, event.y - py) <= tol_px:
                return name
        return None

    def _resize_from_handle(
        self,
        base: RotatedRectROI,
        handle: RectHandle,
        mouse_local: Tuple[float, float],
        *,
        keep_aspect: bool,
        from_center: bool,
    ) -> RotatedRectROI:
        """
        Блок логики, который описывает, как именно тянутся углы (NW, NE) и грани (N, S) прямоугольника. 
        Для круга или линии логика ресайза будет другой
        """
        mx, my = mouse_local
        w2 = base.width / 2.0
        h2 = base.height / 2.0
        aspect = base.width / base.height if base.height != 0 else 1.0

        if from_center:
            # Symmetric resize around the center in local coordinates.
            if handle in (RectHandle.E, RectHandle.W):
                new_w = max(self.min_size, 2.0 * abs(mx))
                new_h = base.height if not keep_aspect else max(self.min_size, new_w / aspect)
            elif handle in (RectHandle.N, RectHandle.S):
                new_h = max(self.min_size, 2.0 * abs(my))
                new_w = base.width if not keep_aspect else max(self.min_size, new_h * aspect)
            else:
                new_w = max(self.min_size, 2.0 * abs(mx))
                new_h = max(self.min_size, 2.0 * abs(my))
                if keep_aspect:
                    if new_w / max(new_h, 1e-12) > aspect:
                        new_h = max(self.min_size, new_w / aspect)
                    else:
                        new_w = max(self.min_size, new_h * aspect)

            if keep_aspect and handle in (RectHandle.E, RectHandle.W):
                new_h = max(self.min_size, new_w / aspect)
            elif keep_aspect and handle in (RectHandle.N, RectHandle.S):
                new_w = max(self.min_size, new_h * aspect)

            return RotatedRectROI(base.cx, base.cy, float(new_w), float(new_h), base.angle)

        # Anchored resize: opposite side is fixed.
        if handle == RectHandle.NE:
            ax0, ay0 = -w2, -h2
            mx = max(mx, ax0 + self.min_size)
            my = max(my, ay0 + self.min_size)
            xmin, xmax = ax0, mx
            ymin, ymax = ay0, my
        elif handle == RectHandle.NW:
            ax0, ay0 = +w2, -h2
            mx = min(mx, ax0 - self.min_size)
            my = max(my, ay0 + self.min_size)
            xmin, xmax = mx, ax0
            ymin, ymax = ay0, my
        elif handle == RectHandle.SE:
            ax0, ay0 = -w2, +h2
            mx = max(mx, ax0 + self.min_size)
            my = min(my, ay0 - self.min_size)
            xmin, xmax = ax0, mx
            ymin, ymax = my, ay0
        elif handle == RectHandle.SW:
            ax0, ay0 = +w2, +h2
            mx = min(mx, ax0 - self.min_size)
            my = min(my, ay0 - self.min_size)
            xmin, xmax = mx, ax0
            ymin, ymax = my, ay0
        elif handle == RectHandle.E:
            ax0 = -w2
            mx = max(mx, ax0 + self.min_size)
            xmin, xmax = ax0, mx
            ymin, ymax = -h2, +h2
            if keep_aspect:
                new_w = xmax - xmin
                new_h = max(self.min_size, new_w / aspect)
                ymin, ymax = -new_h / 2.0, +new_h / 2.0
        elif handle == RectHandle.W:
            ax0 = +w2
            mx = min(mx, ax0 - self.min_size)
            xmin, xmax = mx, ax0
            ymin, ymax = -h2, +h2
            if keep_aspect:
                new_w = xmax - xmin
                new_h = max(self.min_size, new_w / aspect)
                ymin, ymax = -new_h / 2.0, +new_h / 2.0
        elif handle == RectHandle.N:
            ay0 = -h2
            my = max(my, ay0 + self.min_size)
            xmin, xmax = -w2, +w2
            ymin, ymax = ay0, my
            if keep_aspect:
                new_h = ymax - ymin
                new_w = max(self.min_size, new_h * aspect)
                xmin, xmax = -new_w / 2.0, +new_w / 2.0
        elif handle == RectHandle.S:
            ay0 = +h2
            my = min(my, ay0 - self.min_size)
            xmin, xmax = -w2, +w2
            ymin, ymax = my, ay0
            if keep_aspect:
                new_h = ymax - ymin
                new_w = max(self.min_size, new_h * aspect)
                xmin, xmax = -new_w / 2.0, +new_w / 2.0
        else:
            return base

        # If keep_aspect and corner resize, adjust the second dimension.
        if keep_aspect and handle.is_corner:
            new_w = xmax - xmin
            new_h = ymax - ymin
            if new_w / max(new_h, 1e-12) > aspect:
                new_h = max(self.min_size, new_w / aspect)
            else:
                new_w = max(self.min_size, new_h * aspect)

            if handle in (RectHandle.NE, RectHandle.SE):
                # x anchored on left side for east handles
                xmax = xmin + new_w
                if handle == RectHandle.NE:
                    ymin = ymax - new_h
                else:
                    ymax = ymin + new_h
            else:
                xmin = xmax - new_w
                if handle == RectHandle.NW:
                    ymin = ymax - new_h
                else:
                    ymax = ymin + new_h

        new_w = max(self.min_size, float(xmax - xmin))
        new_h = max(self.min_size, float(ymax - ymin))
        local_cx = 0.5 * (xmin + xmax)
        local_cy = 0.5 * (ymin + ymax)
        dxg, dyg = self._rotmat(base.angle) @ np.array([local_cx, local_cy], dtype=float)

        r = RotatedRectROI(
            cx=base.cx + float(dxg),
            cy=base.cy + float(dyg),
            width=float(new_w),
            height=float(new_h),
            angle=base.angle,
        )
        return r


    # ---------------------------------------------------------------------
    # Artist updates and drawing
    # ---------------------------------------------------------------------
    def _update_artists(self) -> None:
        if self.roi is None or self.roi.width <= 0 or self.roi.height <= 0:
            self._rect.set_visible(False)
            for h in self._handles.values():
                h.set_visible(False)
            self._rot_handle.set_visible(False)
            self._rot_line.set_visible(False)
            return

        roi = self.roi
        self._rect.set_visible(True)
        self._rect.set_xy((-roi.width / 2.0, -roi.height / 2.0))
        self._rect.set_width(roi.width)
        self._rect.set_height(roi.height)
        self._rect.set_facecolor(self.facecolor)
        self._rect.set_edgecolor(self._current_edgecolor())
        self._rect.set_linewidth(self.lw + (0.5 if self._action != Action.IDLE else 0.0))
        self._rect.set_transform(self._roi_affine(roi) + self.ax.transData)

        pos = self._handle_positions(roi)
        edge = self._current_edgecolor()
        for name, artist in self._handles.items():
            x, y = pos[name]
            artist.set_data([x], [y])
            artist.set_visible(True)
            artist.set_markerfacecolor("white")
            artist.set_markeredgecolor(edge)

        topx, topy = self._roi_top_center(roi)
        rotx, roty = self._roi_rotate_handle(roi)
        self._rot_handle.set_data([rotx], [roty])
        self._rot_handle.set_visible(True)
        self._rot_handle.set_markerfacecolor("white")
        self._rot_handle.set_markeredgecolor(edge)

        self._rot_line.set_data([topx, rotx], [topy, roty])
        self._rot_line.set_visible(True)
        self._rot_line.set_color(edge)
        
    # ---------------------------------------------------------------------
    # State / cursor helpers
    # ---------------------------------------------------------------------
    def _cursor_for_handle(self, handle: Optional[RectHandle]) -> Optional[Any]: # 🟨
        if Cursors is None or handle is None:
            return None
        mapping = {
            RectHandle.ROTATE: getattr(Cursors, "HAND", None),
            RectHandle.N: getattr(Cursors, "RESIZE_VERTICAL", None),
            RectHandle.S: getattr(Cursors, "RESIZE_VERTICAL", None),
            RectHandle.E: getattr(Cursors, "RESIZE_HORIZONTAL", None),
            RectHandle.W: getattr(Cursors, "RESIZE_HORIZONTAL", None),
            RectHandle.NE: getattr(Cursors, "RESIZE_NESW", None),
            RectHandle.SW: getattr(Cursors, "RESIZE_NESW", None),
            RectHandle.NW: getattr(Cursors, "RESIZE_NWSE", None),
            RectHandle.SE: getattr(Cursors, "RESIZE_NWSE", None),
        }
        return mapping.get(handle, getattr(Cursors, "POINTER", None))    


    # -------------------------------------------------------------------------
    # Реализация геометрических хуков (специфично для прямоугольника) 🟨
    # -------------------------------------------------------------------------
    def _process_press(self, event) -> None:
        """
        Обрабатывает нажатие мыши: а какую часть схватился пользователь при нажатии (ручка, тело, пустота) 
        и переводит селектор в соответствующий Action.

        Returns:
            bool:
                - True: Действие распознано. Базовый класс сам обновит графику _update_artists()
                  и выполнит быстрый блит _request_redraw(full=False).
                - False: Либо клик мимо (ничего не произошло), либо создание 
                  фигуры обработано автономно с полным рендером _request_redraw(full=True).
        """
        handle = self._hit_test_handle(event, tol_px=self.hover_tolerance_px)
        
        # 1. Попали в ручку вращения или ресайза?
        if handle is not None and self.roi is not None:
            if handle == RectHandle.ROTATE and self.allow_rotate:
                self._action = Action.ROTATING
                self._active_handle = handle
                self._press_roi = self.roi.copy()
                return True  
            if handle.is_resize and self.allow_resize:
                self._action = Action.RESIZING
                self._active_handle = handle
                self._press_roi = self.roi.copy()
                return True

        # 2. Попали внутрь тела фигуры (перемещение)?
        if self.roi is not None and self.allow_move and self._event_in_roi(event):
            self._action = Action.MOVING
            self._press_roi = self.roi.copy()
            self._active_handle = None
            return True

        # 3. Кликнули в пустоту — создаем новую фигуру
        if self.allow_create:
            self._press_roi = RotatedRectROI(event.xdata, event.ydata, 0.0, 0.0, 0.0) # ⚓ Сохраняем "якорь" начальной точки, но саму фигуру пока не создаем.
            self.roi = None                  # Прямоугольник материализуется в self.roi только при реальном сдвиге мыши в _process_motion.
            
            self._action = Action.CREATING
            self._active_handle = None
            self._update_artists()           # Всё делаем сами и просим тяжелый redraw с полной перерисовкой фона
            self._emit_change()              # Сигнализируем главному окну, что селектор сброшен
            self._request_redraw(full=True)  # Жесткая перерисовка фона
            return False                     # СТОП для базы! Мы всё сделали сами, базовый класс свободен.
        return False # Ничего не произошло (например, кликнули мимо при allow_create=False)


    def _process_motion(self, event) -> None:
        """Математика изменения геометрии прямоугольника при перетаскивании."""
        base = self._press_roi
        keep_aspect, from_center = self._apply_modifiers(event) # Считываем модификаторы (Shift / Ctrl) (Этот метод в базовом классе)

        if self._action == Action.CREATING:
            x0, y0 = self._press_event.xdata, self._press_event.ydata
            x1, y1 = event.xdata, event.ydata
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            if self.snap_grid > 0:
                cx = round(cx / self.snap_grid) * self.snap_grid
                cy = round(cy / self.snap_grid) * self.snap_grid
                w = max(self.min_size, round(w / self.snap_grid) * self.snap_grid)
                h = max(self.min_size, round(h / self.snap_grid) * self.snap_grid)
            self.roi = self._constrain_roi(RotatedRectROI(cx, cy, w, h, 0.0))
            return True

        elif self._action == Action.MOVING:
            dx = event.xdata - self._press_event.xdata
            dy = event.ydata - self._press_event.ydata
            r = RotatedRectROI(base.cx + dx, base.cy + dy, base.width, base.height, base.angle)
            self.roi = self._constrain_roi(r)
            return True

        elif self._action == Action.ROTATING:
            a0 = math.degrees(math.atan2(self._press_event.ydata - base.cy, self._press_event.xdata - base.cx))
            a1 = math.degrees(math.atan2(event.ydata - base.cy, event.xdata - base.cx))
            new_angle = self._normalize_angle(base.angle + (a1 - a0))
            if self.snap_angle > 0:
                new_angle = round(new_angle / self.snap_angle) * self.snap_angle
                new_angle = self._normalize_angle(new_angle)
            self.roi = self._constrain_roi(RotatedRectROI(base.cx, base.cy, base.width, base.height, new_angle))
            return True

        elif self._action == Action.RESIZING:
            handle = self._active_handle
            if handle is None:
                return False
            px, py = self._data_to_local(event.xdata, event.ydata, base)
            r = self._resize_from_handle(base, handle, (px, py), keep_aspect=keep_aspect, from_center=from_center)
            self.roi = self._constrain_roi(r)
            return True

    def _process_release(self, event) -> None:  # 🟨 Наследник проверяет, имеет ли фигура смысл
        """Проверка, не выродился ли прямоугольник в точку/линию."""
        if self.roi is None:
            return False
        return self.roi.width >= self.min_size and self.roi.height >= self.min_size

        
    
    # ---------------------------------------------------------------------
    # Optional helpers for integration
    # ---------------------------------------------------------------------
    def contains(self, x: float, y: float) -> bool:
        """
        Проверка, кликнули ли рядом с линией (с учетом tolerance)
        (расчет расстояния от точки до отрезка)
        """
        """Check whether a data-space point lies inside the current ROI."""
        if self.roi is None:
            return False
        u, v = self._data_to_local(x, y, self.roi)
        return (-self.roi.width / 2.0 <= u <= self.roi.width / 2.0) and (-self.roi.height / 2.0 <= v <= self.roi.height / 2.0)

    