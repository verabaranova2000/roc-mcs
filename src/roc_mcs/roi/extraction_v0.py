import numpy as np
import re

from roc_mcs.roi.roi_types import ROI

"""
Извлечение ROC-кривых из ROI

        текст инспектора
              ↓
        parse_roi_text()           # парсит текст из Inspector и выдает ROI
              ↓
        ROI(...)                   # объект ROI
              ↓
        extract_from_roi()         # извлекает ROC-кривые из заданной ROI
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
     point   line   rect
       │      │      │
       └──────┼──────┘
              ▼
       ROC-данные ROI
            reduction_mode:
                None  -> все индивидуальные кривые пикселей
                "mean" -> одна средняя кривая
                "sum"  -> одна суммарная кривая
"""


def parse_roi_text(text):
    """
    Восстанавливает ROI из текста инспектора приложения.

    Тип ROI определяется по размерам:
    прямоугольник — width > 0 и height > 0;
    линия — width > 0 и height = 0;
    точка — width = 0 и height = 0.

    Parameters
    ----------
    text : str
        Текст с описанием выбранной ROI, скопированный из инспектора.

    Returns
    -------
    ROI
        Восстановленный объект ROI с координатами, размерами,
        углом, меткой и типом геометрии.
    """
    label = re.search(r"label:\s*(.+)", text)
    center = re.search(r"center:\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)", text)
    size = re.search(r"size:\s*([-+0-9.eE]+)\s*[×x]\s*([-+0-9.eE]+)", text)
    angle = re.search(r"angle:\s*([-+0-9.eE]+)", text)

    if center is None or size is None:
        raise ValueError("Не удалось прочитать параметры ROI: отсутствуют center или size.")

    cx = float(center.group(1))
    cy = float(center.group(2))
    width = float(size.group(1))
    height = float(size.group(2))
    roi_angle = float(angle.group(1)) if angle else 0.0
    roi_label = label.group(1).strip() if label else ""

    if width > 0 and height > 0:
        roi_type = "rect"
    elif width > 0 and height == 0:
        roi_type = "line"
    elif width == 0 and height == 0:
        roi_type = "point"
    else:
        raise ValueError(f"Некорректные размеры ROI: width={width}, height={height}.")

    return ROI(
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        angle=roi_angle,
        label=roi_label,
        type=roi_type,
    )


def extract_from_roi(
    raw,
    th,
    axis,
    roi,
    nx,
    ny,
    reduction_mode=None,
):
    """
    Извлекает ROC-кривые из заданной ROI.

    Parameters
    ----------
    raw : ndarray
        Трёхмерный массив измеренных данных.
    th : ndarray
        Угловая ось ROC-кривых.
    axis : int
        Индекс оси массива `raw`, соответствующей углу θ.
    roi : ROI
        Описание выбранной области интереса.
    nx : int
        Число пикселей по оси X.
    ny : int
        Число пикселей по оси Y.
    reduction_mode : {None, "mean", "sum"}, optional
        Режим получения данных:
        - None — вернуть все индивидуальные ROC-кривые выбранных пикселей;
        - "mean" — вернуть среднюю ROC-кривую;
        - "sum" — вернуть суммарную ROC-кривую.

    Returns
    -------
    th : ndarray
        Угловая ось.
    data : ndarray
        При `reduction_mode=None` — массив индивидуальных ROC-кривых
        формы `(N_pixels, N_theta)`.
        При `"mean"` или `"sum"` — одна агрегированная кривая
        формы `(N_theta,)`.
    count : int
        Число пикселей, вошедших в ROI.
    """
    if raw.ndim != 3:
        raise ValueError(f"Ожидался трёхмерный массив raw, получен массив формы {raw.shape}.")
    if axis not in (0, 2):
        raise ValueError(f"Неподдерживаемая ось угла axis={axis}. Допустимые значения: 0 или 2.")
    
    mode = None if reduction_mode is None else reduction_mode.lower()
    if mode not in (None, "mean", "sum"):
        raise ValueError(
            f"Неподдерживаемый режим агрегации reduction_mode: {reduction_mode!r}. "
            "Допустимые значения: None, 'mean' или 'sum'."
        )
    roi_type = roi.type.lower()

    if roi_type == "point":
        return _extract_from_point_roi(raw, th, axis, roi, nx, ny, mode)
    if roi_type == "line":
        return _extract_from_line_roi(raw, th, axis, roi, nx, ny, mode)
    if roi_type == "rect":
        return _extract_from_rect_roi(raw, th, axis, roi, nx, ny, mode)
    
    raise ValueError(
        f"Неподдерживаемый тип ROI: {roi.type!r}. "
        "Допустимые типы: 'point', 'line' или 'rect'."
    )


# ===========================================================================
# Внутренние функции для extract_from_roi(...)
# ===========================================================================

def _extract_from_point_roi(raw, th, axis, roi, nx, ny, reduction_mode=None):
    """
    Извлекает ROC-кривую пикселя, заданного точечной ROI.

    При reduction_mode=None возвращается массив shape (1, N_theta).
    При "mean" или "sum" возвращается массив shape (N_theta,).
    """
    ix, iy = int(round(roi.cx)), int(round(roi.cy))
    if not (0 <= ix < nx and 0 <= iy < ny):          # Проверка на выход за границы
        return th.copy(), np.empty((0, len(th))), 0
    
    if axis == 0:
        curve = raw[:, iy, ix]
    else:
        curve = raw[iy, ix, :]
    curve = np.asarray(curve, dtype=float)

    if reduction_mode is None:
        return th.copy(), curve[np.newaxis, :], 1

    return th.copy(), curve, 1                       # Для одного пикселя mean и sum одинаковы.


def _extract_from_line_roi(raw, th, axis, roi, nx, ny, reduction_mode=None):
    """
    Извлекает ROC-кривые вдоль линейной ROI.

    При reduction_mode=None возвращаются все кривые выбранных
    точек линии, shape = (N_pixels, N_theta).

    При "mean" или "sum" возвращается одна агрегированная кривая.
    """
    length = max(1.0, float(np.hypot(roi.width, roi.height)))  # используем гипотенузу для получения реальной длины
    num_points = int(np.ceil(length))                          # Количество пикселей вдоль линии
    
    # Находим концы отрезка
    theta = np.deg2rad(float(roi.angle))
    dx = (length / 2.0) * np.cos(theta)
    dy = (length / 2.0) * np.sin(theta)
    
    x0, y0 = roi.cx - dx, roi.cy - dy
    x1, y1 = roi.cx + dx, roi.cy + dy
    
    # Генерируем координаты точек вдоль линии
    xs = np.linspace(x0, x1, num_points)
    ys = np.linspace(y0, y1, num_points)
    
    # Округляем до индексов пикселей
    ixs = np.round(xs).astype(int)
    iys = np.round(ys).astype(int)
    
    # Оставляем только те точки, которые попадают в границы картинки
    valid_mask = (ixs >= 0) & (ixs < nx) & (iys >= 0) & (iys < ny)
    valid_ixs = ixs[valid_mask]
    valid_iys = iys[valid_mask]
    count = len(valid_ixs)

    if count == 0:
        empty = (
            np.empty((0, len(th)))
            if reduction_mode is None
            else np.array([], dtype=float)
        )
        return th.copy(), empty, 0
        
    if axis == 0:
        profiles = raw[:, valid_iys, valid_ixs]   # (N_theta, N_pixels)
    else:
        profiles = raw[valid_iys, valid_ixs, :]   # (N_pixels, N_theta)

    if reduction_mode is None:
        if axis == 0:
            curves = profiles.T
        else:
            curves = profiles
        return th.copy(), np.asarray(curves, dtype=float), count

    if reduction_mode == "mean":
        if axis == 0:
            curve = profiles.mean(axis=1)
        else:
            curve = profiles.mean(axis=0)

    else:                                          # reduction_mode == "sum"
        if axis == 0:
            curve = profiles.sum(axis=1)
        else:
            curve = profiles.sum(axis=0)

    return th.copy(), np.asarray(curve, dtype=float), count


def _extract_from_rect_roi(raw, th, axis, roi, nx, ny, reduction_mode=None):
    """
    Извлекает ROC-кривые из прямоугольной ROI.

    При reduction_mode=None возвращаются все кривые выбранных
    пикселей, shape = (N_pixels, N_theta).

    При "mean" или "sum" возвращается одна агрегированная кривая.
    """
    roi = roi.normalized()
    x0, x1, y0, y1 = roi.bbox
    x_min = max(0, int(np.floor(x0)))
    x_max = min(nx, int(np.ceil(x1)))
    y_min = max(0, int(np.floor(y0)))
    y_max = min(ny, int(np.ceil(y1)))

    if x_max <= x_min or y_max <= y_min:
        empty = (
            np.empty((0, len(th)))
            if reduction_mode is None
            else np.array([], dtype=float)
        )
        return th.copy(), empty, 0
    
    xs = np.arange(x_min, x_max, dtype=float)
    ys = np.arange(y_min, y_max, dtype=float)
    xx, yy = np.meshgrid(xs, ys)

    theta = np.deg2rad(float(roi.angle))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    dx = xx - float(roi.cx)
    dy = yy - float(roi.cy)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    
    mask = (np.abs(u) <= 0.5 * float(roi.width) + 1e-9) & (np.abs(v) <= 0.5 * float(roi.height) + 1e-9)
    count = int(mask.sum())

    if count <= 0:
        empty = (
            np.empty((0, len(th)))
            if reduction_mode is None
            else np.array([], dtype=float)
        )
        return th.copy(), empty, 0

    if axis == 0:
        sub = raw[:, y_min:y_max, x_min:x_max]
        if reduction_mode is None:
            curves = sub[:, mask].T
            return th.copy(), np.asarray(curves, dtype=float), count
        curve = sub[:, mask].sum(axis=1)
    else:
        sub = raw[y_min:y_max, x_min:x_max, :]
        if reduction_mode is None:
            curves = sub[mask, :]
            return th.copy(), np.asarray(curves, dtype=float), count
        curve = sub[mask, :].sum(axis=0)

    if reduction_mode == "mean":
        curve = curve / float(count)

    return th.copy(), np.asarray(curve, dtype=float), count