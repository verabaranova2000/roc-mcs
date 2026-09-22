import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any

# ===========================================
# Датаклассы (Модели данных) BaseROI и др.
# ===========================================

@dataclass
class BaseROI:
    """Абстрактный класс для всех ROI."""
    def copy(self) -> "BaseROI": raise NotImplementedError
    def to_dict(self) -> dict: return asdict(self)


@dataclass
class RotatedRectROI(BaseROI):    # Разньше называлось RotatedROI 
    """Вращающийся прямоугольник в координатах данных."""
    cx: float
    cy: float
    width: float
    height: float
    angle: float = 0.0  # degrees, CCW

    def copy(self) -> "RotatedRectROI":
        return RotatedRectROI(self.cx, self.cy, self.width, self.height, self.angle)

    @classmethod
    def from_dict(cls, d: dict) -> "RotatedRectROI":
        return cls(**d)



@dataclass
class LineROI(BaseROI):
    """Отрезок (линия) в координатах данных."""
    x1: float
    y1: float
    x2: float
    y2: float

    def copy(self) -> "LineROI":
        return LineROI(self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_dict(cls, d: dict) -> "LineROI":
        return cls(**d)

@dataclass
class PointROI(BaseROI):
    """Точка в координатах данных."""
    x: float
    y: float

    def copy(self) -> "PointROI":
        return PointROI(self.x, self.y)

    @classmethod
    def from_dict(cls, d: dict) -> "PointROI":
        return cls(**d)



@dataclass
class ROI:
    """
    Universal region of interest (Rectangle, Line, or Point) in image coordinates.
    """
    cx: float
    cy: float
    width: float        # Для линии - это длина (length), для точки = 0.0
    height: float       # Для линии и точки = 0.0
    angle: float = 0.0  # degrees, CCW
    label: str = ""
    color: str = ""
    mode: str = "mean"
    type: str = "rect"  # <-- НОВОЕ ПОЛЕ: "rect", "line", "point"

    def copy(self) -> "ROI":
        return ROI(
            cx=float(self.cx),
            cy=float(self.cy),
            width=float(self.width),
            height=float(self.height),
            angle=float(self.angle),
            label=self.label,
            color=self.color,
            mode=self.mode,
            type=self.type,
        )

    def normalized(self) -> "ROI":
        """
        Точки и линии не нуждаются в нормализации габаритов так же строго, 
        но мы оставляем логику безопасной для всех.
        """
        return ROI(
            cx=float(self.cx),
            cy=float(self.cy),
            width=max(0.0, abs(float(self.width))),
            height=max(0.0, abs(float(self.height))),
            angle=self._normalize_angle(float(self.angle)),
            label=self.label,
            color=self.color,
            mode=self.mode,
            type=self.type,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return ((float(angle) + 180.0) % 360.0) - 180.0

    @property
    def area(self) -> float:
        if self.type in ("line", "point"):
            return 0.0
        return max(0.0, float(self.width)) * max(0.0, float(self.height))

    def corners(self) -> np.ndarray:
        """Return the four corner points in data coordinates."""
        r = self.normalized()
        if r.type == "point":         # Для точки возвращаем одну и ту же координату (или небольшой крестик, если нужно для bbox)
            return np.array([[r.cx, r.cy]], dtype=float)
            
        elif r.type == "line":        # Для линии возвращаем только 2 точки - ее концы
            l2 = 0.5 * float(r.width)
            local = np.array([[-l2, 0.0], [l2, 0.0]], dtype=float)
            
        else:                         # Прямоугольник "rect"
            w2 = 0.5 * float(r.width)
            h2 = 0.5 * float(r.height)
            local = np.array([
                [-w2, -h2],
                [ w2, -h2],
                [ w2,  h2],
                [-w2,  h2],
            ], dtype=float)
        theta = math.radians(float(r.angle))
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        return local @ rot.T + np.array([r.cx, r.cy], dtype=float)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        pts = self.corners()
        xs = pts[:, 0]
        ys = pts[:, 1]
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    @property
    def x1(self) -> int:
        return int(math.floor(self.bbox[0]))

    @property
    def x2(self) -> int:
        return int(math.ceil(self.bbox[1]))

    @property
    def y1(self) -> int:
        return int(math.floor(self.bbox[2]))

    @property
    def y2(self) -> int:
        return int(math.ceil(self.bbox[3]))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_selector_roi(cls, roi: BaseROI, **kwargs) -> "ROI":
        """
        Строгий парсер. Явно проверяем тип пришедшего объекта.
        Никаких угадываний и getattr-магии для базовых классов.
        Умный парсер. Распознает, пришла ли к нам точка, линия или прямоугольник,
        и безопасно вычисляет нужные параметры (cx, cy, width, angle).
        """
        label = str(kwargs.get("label", ""))
        color = str(kwargs.get("color", ""))
        mode = str(kwargs.get("mode", "mean"))

        if isinstance(roi, LineROI):           # 1. Если пришла ЛИНИЯ
            cx = (roi.x1 + roi.x2) / 2.0
            cy = (roi.y1 + roi.y2) / 2.0
            width = math.hypot(roi.x2 - roi.x1, roi.y2 - roi.y1) # Длина идет в width
            angle = math.degrees(math.atan2(roi.y2 - roi.y1, roi.x2 - roi.x1))
            return cls(
                cx=cx, cy=cy, 
                width=width, height=0.0, angle=angle,
                type="line", label=label, color=color, mode=mode
            )

        elif isinstance(roi, PointROI):        # 2. Если пришла ТОЧКА
            return cls(
                cx=float(roi.x), cy=float(roi.y), 
                width=0.0, height=0.0, angle=0.0,
                type="point", label=label, color=color, mode=mode
            )

        elif isinstance(roi, RotatedRectROI):  # 3. Если пришел ПРЯМОУГОЛЬНИК
            return cls(
                cx=float(roi.cx), cy=float(roi.cy),
                width=float(roi.width), height=float(roi.height), angle=float(roi.angle),
                type="rect", label=label, color=color, mode=mode
            )
        else:                                  # 4. Fallback (если вдруг прилетел сырой словарь или старый объект)
            # Здесь оставляем минимальный фоллбэк только для нераспознанных структур
            roi_type = str(getattr(roi, "type", "rect")).lower()
            return cls(
                cx=float(getattr(roi, "cx", 0.0)),
                cy=float(getattr(roi, "cy", 0.0)),
                width=float(getattr(roi, "width", 0.0)),
                height=float(getattr(roi, "height", 0.0)),
                angle=float(getattr(roi, "angle", 0.0)),
                type=roi_type, label=label, color=color, mode=mode
            )
        
    @classmethod
    def from_dict(cls, d: dict) -> "ROI":
        if {"cx", "cy", "width", "height"}.issubset(d.keys()):
            return cls(
                cx=float(d["cx"]),
                cy=float(d["cy"]),
                width=float(d["width"]),
                height=float(d["height"]),
                angle=float(d.get("angle", 0.0)),
                label=str(d.get("label", "")),
                color=str(d.get("color", "")),
                mode=str(d.get("mode", "mean")),
                type=str(d.get("type", "rect")), # <-- Дефолт "rect" спасет старые JSON!
            )
        # Backward compatibility with the old axis-aligned ROI schema.
        if {"x1", "x2", "y1", "y2"}.issubset(d.keys()):
            x1 = float(d["x1"])
            x2 = float(d["x2"])
            y1 = float(d["y1"])
            y2 = float(d["y2"])
            return cls(
                cx=0.5 * (x1 + x2),
                cy=0.5 * (y1 + y2),
                width=abs(x2 - x1),
                height=abs(y2 - y1),
                angle=float(d.get("angle", 0.0)),
                label=str(d.get("label", "")),
                color=str(d.get("color", "")),
                mode=str(d.get("mode", "mean")),
                type="rect", # НОВОЕ: явно указываем тип для старых сохранений
            )
        raise ValueError(f"Unrecognized ROI schema: {sorted(d.keys())}")

    @property
    def is_valid(self) -> bool:
        """
        Строгая валидация геометрии в зависимости от типа сущности.
        Не позволяет селекторам создавать вырожденные (нулевые) фигуры.
        """
        if self.type == 'point':                          # Точка всегда валидна, если у нее есть координаты cx, cy
            return self.width == 0 and self.height == 0   # Любая попытка задать ширину или высоту отличную от 0 должна отбраковываться.
        elif self.type == 'line':                         # Линия должна иметь хотя бы какую-то длину (ширину ИЛИ высоту проекции)
            return self.width > 0 and self.height == 0    # Чтобы случайный клик (0x0) не считался линией
        elif self.type == 'rect':                         # Прямоугольник обязан иметь площадь. 
            return self.width > 0 and self.height > 0     # Линия нулевой толщины (клик+протяжка по одной оси) - это не прямоугольник.
        return False                                      # Неизвестный тип на всякий случай блокируем


@dataclass
class SavedROI:
    id: int
    roi: ROI
    curve: Optional[np.ndarray] = None
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "roi": self.roi.to_dict(),
            "count": int(self.count),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SavedROI":
        roi_data = d.get("roi", d)
        roi = ROI.from_dict(roi_data).normalized()
        return cls(id=int(d.get("id", 0)), roi=roi, curve=None, count=int(d.get("count", 0)))
