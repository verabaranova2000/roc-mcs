from typing import Optional, Tuple, Any
from matplotlib.backend_bases import MouseButton
import math
import numpy as np
from typing import Callable, Optional, Tuple, Any


try:
    from matplotlib.backend_bases import Cursors
except ImportError:
    Cursors = None  # type: ignore

from roc_mcs.roi.roi_types import BaseROI
from roc_mcs.gui.roi_explorer.selectors.enums import Action, RectHandle


# ======================
# БАЗОВЫЙ КЛАСС 
# ======================

class BaseROISelector:
    """
    Базовый класс для интерактивных селекторов ROI в Matplotlib.

    Взаимодействие (Interaction)
    --------------------------
    - Перетаскивание на пустом месте: создание новой фигуры.
    - Перетаскивание внутри фигуры: перемещение (MOVING).
    - Перетаскивание за маркеры (ручки): изменение размера (RESIZING) или длины.
    - Перетаскивание маркера вращения: поворот (ROTATING) (если поддерживается).
    - Клавиши Delete/Backspace: удаление фигуры.
    - Клавиша Escape: отмена текущего действия (создания/перемещения).

    Особенности (Features)
    ----------------------
    - Стабильная геометрия в координатах данных (data coordinates).
    - Подсветка при наведении (hover) и изменение курсора.
    - Опциональная привязка к сетке (snapping).
    - Опциональное ограничение выхода за пределы осей (bounds constraint).
    - Использование blitting для плавной отрисовки и высокой производительности.
    - Модификаторы Shift/Alt для дополнительного контроля (например, сохранение пропорций).
    """
    
    def __init__(
        self,
        ax,
        onselect: Optional[Callable[[BaseROI], None]] = None,
        on_change: Optional[Callable[[BaseROI], None]] = None,
        roi: Optional[BaseROI] = None,
        *,
        min_size: float = 2.0,
        handle_size: float = 7.0,
        rotation_handle_offset: float = 18.0,
        edgecolor: str = "#00AEEF",
        facecolor=(0.0, 0.67, 0.94, 0.08),
        activecolor: str = "#FF8C00",
        hovercolor: str = "#62C4FF",
        lw: float = 2.0,
        allow_create: bool = True,
        allow_move: bool = True,
        allow_resize: bool = True,
        allow_rotate: bool = True,
        keep_aspect: bool = False,
        snap_grid: float = 0.0,
        snap_angle: float = 0.0,
        bounds: Optional[str] = None,
        use_blit: bool = False,
        hover_tolerance_px: float = 10.0,
        cursor_tolerance_px: float = 12.0,
    ):
        # --- Основное окружение и коллбэки ---
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.onselect = onselect
        self.on_change = on_change
        
        # --- Настройки цветов, допусков ---
        self.min_size = float(min_size)
        self.handle_size = float(handle_size)
        self.rotation_handle_offset = float(rotation_handle_offset)
        self.edgecolor = edgecolor
        self.facecolor = facecolor
        self.activecolor = activecolor
        self.hovercolor = hovercolor
        self.lw = float(lw)

        self.allow_create = bool(allow_create)
        self.allow_move = bool(allow_move)
        self.allow_resize = bool(allow_resize)
        self.allow_rotate = bool(allow_rotate)
        self.keep_aspect = bool(keep_aspect)
        self.snap_grid = float(snap_grid)
        self.snap_angle = float(snap_angle)
        self.bounds = bounds  # None | "axes"
        self.use_blit = bool(use_blit)
        self.hover_tolerance_px = float(hover_tolerance_px)
        self.cursor_tolerance_px = float(cursor_tolerance_px)

        if self.bounds not in (None, "axes"):
            raise ValueError("bounds must be None or 'axes'")

        # --- 3. Хранение состояния ROI ---
        self.roi: Optional[BaseROI] = roi.copy() if roi is not None else None
        
        # --- 4. Машина состояний мыши и интерактивности ---
        self._action: Action = Action.IDLE
        self._active_handle: Optional[RectHandle] = None
        self._hover_handle: Optional[RectHandle] = None
        self._press_event = None
        self._press_roi: Optional[BaseROI] = None   
        self._press_modifiers: str = ""

        self._bg = None
        self._needs_full_draw = True

        # ==========================================
        # ВЫЗЫВАЕМ МЕТОД НАСЛЕДНИКА ДЛЯ ОТРИСОВКИ!
        # ==========================================
        self._animated_artists = [] # Наследник заполнит этот список
        self._create_artists()      # 1. Наследник создает артистов и наполняет self._animated_artists
        # --- Подключение событий Matplotlib ---
        if self.use_blit:           # 2. Базовый класс единообразно настраивает blitting для всех
            for artist in self._animated_artists:
                artist.set_animated(True)
            self._cid_draw = self.canvas.mpl_connect("draw_event", self._on_draw)
        else:
            self._cid_draw = None


        self._cid_press = self.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_move = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_release = self.canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_key = self.canvas.mpl_connect("key_press_event", self._on_key)
        
        # Синхронизируем ручку вращения с панорамированием и зумом (слушать изменения осей)
        self._cid_xlim = self.ax.callbacks.connect('xlim_changed', self._on_view_lim_changed) # связь с обработчиком зума
        self._cid_ylim = self.ax.callbacks.connect('ylim_changed', self._on_view_lim_changed)
        
        self._update_artists()
        self._request_redraw(full=True)
        
    # ---------------------------------------------------------------------
    # Public API  # 🟧
    # ---------------------------------------------------------------------
    def set_roi(self, roi: Optional[BaseROI]) -> None: # 🟧
        self.roi = roi.copy() if roi is not None else None
        self._cancel_interaction(update=False)
        self._update_artists()
        self._request_redraw(full=True)

    def clear(self) -> None: # 🟧
        self.set_roi(None)

    def get_roi(self) -> Optional[BaseROI]:  # 🟧
        return self.roi.copy() if self.roi is not None else None

    def disconnect(self) -> None: # 🟧
        self.canvas.mpl_disconnect(self._cid_press)
        self.canvas.mpl_disconnect(self._cid_move)
        self.canvas.mpl_disconnect(self._cid_release)
        self.canvas.mpl_disconnect(self._cid_key)
        if self._cid_draw is not None:                       # Connection ID (идентификатор подключения)
            self.canvas.mpl_disconnect(self._cid_draw)
        # --- Отписываемся от событий осей (зум, панорамирование) ---
        if hasattr(self, '_cid_xlim'):                       # Проверяем через hasattr на случай, если disconnect вызовут до того, 
            self.ax.callbacks.disconnect(self._cid_xlim)     # как переменные были созданы (хорошая практика для защиты от багов)
        if hasattr(self, '_cid_ylim'):
            self.ax.callbacks.disconnect(self._cid_ylim)            

    def remove(self) -> None: # 🟧
        """
        Полностью удаляет селектор: отключает события мыши 
        и стирает все графические элементы (артисты) с осей.
        """
        self.disconnect()                      # 1. Отключаем все события (мышь, зум, клавиатуру)
        for artist in self._animated_artists:  # 2. Удаляем всю графику с графика Matplotlib
            try:
                if getattr(artist, 'axes', None) is not None or getattr(artist, 'figure', None) is not None: # Проверяем, прикреплен ли артист к графику, чтобы избежать NotImplementedError
                    artist.remove()
            except Exception:     # Ловим ВСЁ (ValueError, NotImplementedError и т.д.)
                pass              # Игнорируем, если артист уже был удален ранее
        self._animated_artists.clear()         # 3. ОЧИЩАЕМ СПИСОК (очень важно!)
        self.canvas.draw_idle()                # 4. Перерисовываем холст
        
    
    def set_active(self, is_active: bool) -> None: # 🟧
        """
        Включает или отключает взаимодействие с селектором.
        Если отключено, селектор отображается, но не перехватывает события мыши.
        """
        self.allow_create = is_active
        self.allow_move = is_active
        self.allow_resize = is_active
        self.allow_rotate = is_active
        # Если мы отключаем селектор прямо во время перетаскивания,
        # нужно безопасно сбросить текущее действие и вернуть стандартный курсор.
        if not is_active:
            self._cancel_interaction(update=True)
            # self._set_cursor
            # возвращает стандартную стрелочку, когда мы отключаем селектор
            self._set_cursor(getattr(Cursors, "POINTER", None) if Cursors is not None else None)
    
        
    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------
    # --- Чистая тригонометрия ---
    @staticmethod
    def _normalize_angle(angle: float) -> float: # 🟧
        return ((float(angle) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _rotmat(angle_deg: float) -> np.ndarray: # 🟧
        t = math.radians(angle_deg)
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s], [s, c]], dtype=float)
    
    # --- Математика удержания любой рамки внутри осей графика ( _axes_bounds, _rotated_half_extents, _constrain_roi) ---
    def _axes_bounds(self) -> Optional[Tuple[float, float, float, float]]: # 🟧
        if self.bounds != "axes":
            return None
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        return (min(xmin, xmax), max(xmin, xmax), min(ymin, ymax), max(ymin, ymax))


    def _constrain_roi(self, roi: BaseROI) -> BaseROI: # 🟨 ✅
        """
        Абстрактный метод ограничений.
        Отвечает за clipping (математическое удержание координат фигуры 
        внутри осей графика xₘᵢₙ, xₘₐₓ, yₘᵢₙ, yₘₐₓ и привязку к сетке).
        
        Базовый класс знает, что фигуру надо ограничивать, но НЕ ЗНАЕТ КАК.
        Должен быть переопределен в наследниках для контроля границ.
        """
        raise NotImplementedError
        


    def _apply_modifiers(self, event) -> Tuple[bool, bool]: # 🟧
        """
        Проверка нажатия Shift/Alt
        Возвращает (keep_aspect, resize_from_center)
        """
        mods = (self._press_modifiers or "").lower()
        keep = self.keep_aspect or ("shift" in mods)
        from_center = "alt" in mods or "option" in mods
        return keep, from_center

    @staticmethod
    def _signed_scale(value: float, minimum: float) -> float: # 🟧
        """ вспомогательная математика """
        if abs(value) < minimum:
            return math.copysign(minimum, value if value != 0 else 1.0)
        return float(value)


    def _on_view_lim_changed(self, ax) -> None: # 🟧
        """Обработчик зума. Пересчитывает пимпочку строго в момент изменения масштаба (до отрисовки кадра)."""
        if self.roi is not None:
            self._update_artists()
            
    # ---------------------------------------------------------------------
    # Artist updates and drawing
    # ---------------------------------------------------------------------

    def _current_edgecolor(self) -> str: # 🟧
        """
        Учим метод цвета реагировать на новый флаг. 
        Если мы навели курсор на рамку (а не только на квадратики в углах), 
        она моментально отреагирует и подсветится синим.
        """
        if self._action != Action.IDLE:
            return self.activecolor
        is_hovering_body = getattr(self, "_hover_body", False)  # Рамка вспыхнет синим (hovercolor), если курсор над ручкой-маркером 
        if self._hover_handle is not None or is_hovering_body:  # ИЛИ если он находится над самой линией/площадью прямоугольника
            return self.hovercolor
        return self.edgecolor

    def _emit_change(self) -> None: # 🟧
        """Триггер коллбеков."""
        if self.roi is not None and self.on_change is not None:
            self.on_change(self.roi.copy())

    def _emit_select(self) -> None: # 🟧
        """Триггер коллбеков."""
        if self.roi is not None and self.onselect is not None:
            self.onselect(self.roi.copy())

    # Управление отрисовкой (_request_redraw, _blit_current, _on_draw) — это чисто машинерия Matplotlib, ей самое место в базовом классе,
    """
    Это самое сложное и грязное место (blitting в Matplotlib работает капризно). 
    Оставить эту логику инкапсулированной в базовом классе — лучшее архитектурное решение. 
    Наследники вообще не должны знать, как именно Matplotlib кэширует фон и перерисовывает кадры.
    """
    def _request_redraw(self, *, full: bool = False) -> None: # 🟧
        if not self.use_blit:
            self.canvas.draw_idle()
            return

        if full or self._needs_full_draw:
            self._needs_full_draw = False
            self.canvas.draw_idle()
            return

        if self._bg is None:
            self.canvas.draw_idle()
            return

        self._blit_current()

    def _on_draw(self, event) -> None:  # pragma: no cover - backend dependent 🟧
        if not self.use_blit:
            return
        try:
            self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
            self._blit_current()
        except Exception:
            self._bg = None

    def _blit_current(self) -> None:  # pragma: no cover - backend dependent 🟧
        if self._bg is None:
            return
        try:
            self.canvas.restore_region(self._bg)
            for artist in self._animated_artists:
                if artist.get_visible():
                    self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        except Exception:
            self.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # State / cursor helpers
    # ---------------------------------------------------------------------
    # Вся логика смены иконок курсора мыши одинакова для любой фигуры (навел на угол — появилась диагональная стрелочка) (_set_cursor и _cursor_for_handle)
    def _set_cursor(self, cursor: Optional[Any]) -> None: # 🟧
        if cursor is None:
            return
        try:
            self.canvas.set_cursor(cursor)
        except Exception:
            pass

    def _cursor_for_handle(self, handle: Any) -> Optional[Any]: # 🟨 ✅
        """
        Возвращает иконку курсора для конкретной ручки.
        Переопределяется в наследниках, так как у каждой фигуры свои типы ручек.
        """
        return getattr(Cursors, "POINTER", None) if Cursors is not None else None


    def _update_hover(self, event) -> None: # 🟧
        """
        Метод управляет флагом наведения. 
        Обрати внимание: внутри он вызывает self._hit_test_handle() и self._event_in_roi(). 
        В базовом классе этих методов больше нет, но Python достаточно умен! 
        При работе он поймет, что ты создала объект RotatedRectangleSelector, и вызовет эти методы оттуда.
        
        Сюда мы добавляем недостающее звено — флаг self._hover_body, 
        который будет запоминать, что курсор находится именно над телом или гранью рамки.
        """
        if event.inaxes != self.ax or event.x is None or event.y is None:
            self._hover_handle = None
            self._hover_body = False  # Сбрасываем флаг тела рамки
            self._set_cursor(getattr(Cursors, "POINTER", None) if Cursors is not None else None)
            return

        handle = self._hit_test_handle(event, tol_px=self.cursor_tolerance_px)
        self._hover_handle = handle
        self._hover_body = False  # Инициализируем флаг по умолчанию

        if self._action == Action.IDLE:
            if handle is not None:
                self._set_cursor(self._cursor_for_handle(handle))
            elif self.roi is not None and self._event_in_roi(event):
                self._hover_body = True  # Мы находимся над гранью или внутри рамки!
                self._set_cursor(getattr(Cursors, "MOVE", None) if Cursors is not None else None)
            else:
                self._set_cursor(getattr(Cursors, "POINTER", None) if Cursors is not None else None)

        
    def _cancel_interaction(self, *, update: bool = True) -> None: # 🟧
        """ Универсальный сброс состояния """
        self._action = Action.IDLE
        self._active_handle = None
        self._hover_handle = None
        self._press_event = None
        self._press_roi = None
        self._press_modifiers = ""
        if update:
            self._update_artists()
            self._request_redraw()

    # ---------------------------------------------------------------------
    # Геометрические "хуки" для наследников (🟨)
    # ---------------------------------------------------------------------
    def _process_press(self, event) -> bool:
        """Определяет действие (MOVING, RESIZING и т.д.) и устанавливает self._action. Возвращает True, если нужна перерисовка."""
        raise NotImplementedError

    def _process_motion(self, event) -> bool:
        """Изменяет self.roi в зависимости от текущего self._action. Возвращает True, если геометрия изменилась."""
        raise NotImplementedError

    def _process_release(self, event) -> bool:
        """Проверяет валидность фигуры в конце действия. Возвращает True, если фигура корректна."""
        raise NotImplementedError
        
    # ---------------------------------------------------------------------
    # Общие методы мыши (Mouse / key handlers) 🟧
    # ---------------------------------------------------------------------
    def _on_press(self, event) -> None:
        """Перехватывает клик, фильтрует мусор и отдает команду наследнику."""
        """
        Определяет, что именно мы сейчас делаем — двигаем рамку, вращаем или тянем за угол 
        (переводит стейт в Action.MOVING, Action.ROTATING и т.д.)
        """
        if event.button != MouseButton.LEFT:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        self._press_event = event
        self._press_modifiers = str(getattr(event, "key", "") or "")
        # 🟨 Вся логика "попали ли в ручку/тело" теперь в наследнике!
        if self._process_press(event):
            self._update_artists()
            self._request_redraw()

    
    def _on_motion(self, event) -> None:
        """Управляет Hover-эффектом, а при зажатой мыши отдает математику движения наследнику."""
        """
        Hover feedback when idle.
        
        Высчитывает математику движения. И снова магия: при ресайзе он вызывает self._resize_from_handle(...), 
        который мы на предыдущем шаге перенесли в класс прямоугольника. 
        База просто говорит: «Эй, наследник, я тут сдвинул мышь, пересчитай-ка свои грани!»
        """
        if self._action == Action.IDLE:
            self._update_hover(event)
            self._update_artists()
            self._request_redraw()
            return

        if self._press_event is None or self._press_roi is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        # 🟨 Вся математика растягивания/движения/вращения теперь в наследнике!
        if self._process_motion(event):
            self._update_artists()
            self._emit_change()
            self._request_redraw()


    def _on_release(self, event) -> None:
        """Завершает действие и проверяет (через наследника), не выродилась ли фигура в ноль."""
        if self._action == Action.IDLE:
            return

        is_valid = self._process_release(event) # 🟨 Наследник проверяет, имеет ли фигура смысл (например, ширина > min_size)
        if is_valid and self.roi is not None:
            self._emit_select()
        else:
            if self._action == Action.CREATING:
                self.roi = None
                self._emit_change()
                
        self._cancel_interaction(update=False)
        self._update_artists()
        self._request_redraw()

    def _on_key(self, event) -> None:  # 🟧
        """ Удаление рамки """
        key = str(event.key or "").lower()
        if key in ("delete", "backspace"):
            self.clear()
            return

        if key == "escape":
            if self._action == Action.CREATING:
                self.clear()
                return
            self._cancel_interaction(update=True)

       


    # --- Абстрактные методы, которые реализуют наследники ---
    def _create_artists(self): raise NotImplementedError
    def _update_artists(self): raise NotImplementedError
    def _handle_positions(self, roi) -> dict: raise NotImplementedError
    def contains(self, x: float, y: float) -> bool: raise NotImplementedError