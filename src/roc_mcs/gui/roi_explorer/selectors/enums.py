from enum import Enum, auto
# ===================================================
# ВСПОМОГАТЕЛЬНЫЕ ENUM (ДЛЯ УПРАВЛЕНИЯ СОСТОЯНИЕМ)
# ===================================================

class Action(Enum):
    """Текущее действие пользователя."""
    IDLE = auto()        # ничего
    CREATING = auto()    # создаем
    MOVING = auto()      # двигаем фигуру целиком
    RESIZING = auto()    # тянем за край
    ROTATING = auto()    # вращаем


class RectHandle(Enum):
    """Маркеры изменения размера для прямоугольника."""
    NW = "nw"
    N = "n"
    NE = "ne"
    E = "e"
    SE = "se"
    S = "s"
    SW = "sw"
    W = "w"
    ROTATE = "rotate"

    @property
    def is_resize(self) -> bool:
        return self in {
            RectHandle.NW,
            RectHandle.N,
            RectHandle.NE,
            RectHandle.E,
            RectHandle.SE,
            RectHandle.S,
            RectHandle.SW,
            RectHandle.W,
        }

    @property
    def is_corner(self) -> bool:
        return self in {RectHandle.NW, RectHandle.NE, RectHandle.SE, RectHandle.SW}


class LineHandle(Enum):
    """Маркеры изменения длины для линии."""
    P1 = "p1"
    P2 = "p2"
    
    @property
    def is_resize(self) -> bool:
        return True
