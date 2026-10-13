import numpy as np
import re
from scipy.ndimage import map_coordinates

from roc_mcs.roi.roi_types import ROI, ROIVolume

"""
Работа с ROI: извлечение ROC-кривых и пространственный отбор

        текст инспектора → parse_roi_text() → ROI
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             extract_from_roi()        filter_peak_catalog_by_roi()
                    │                           │
              point / line / rect          roi_pixel_mask()
                    │                           │
                    ▼                           ▼
              ROC-кривые ROI              peak_catalog_roi
                                                │
                                                ▼
                                         fit_jax → D1 → E2/F → G

    E1 строит контрольный пул отдельно: в него при необходимости входят
    остатки и из пикселей вне ROI тестового семейства.
    extract_from_roi(): None — кривые пикселей; "mean"/"sum" — агрегация.
    roi_pixel_mask(): маска центров пикселей прямоугольной ROI.
"""


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
    if not np.all(np.isfinite([cx, cy, width, height, roi_angle])):
        raise ValueError("Параметры ROI должны быть конечными числами.")

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
    raw = np.asarray(raw)
    th = np.asarray(th, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"Ожидался трёхмерный массив raw, получен массив формы {raw.shape}.")
    if th.ndim != 1:
        raise ValueError(f"Ожидалась одномерная ось th, получена форма {th.shape}.")
    if axis not in (0, 2):
        raise ValueError(f"Неподдерживаемая ось угла axis={axis}. Допустимые значения: 0 или 2.")
    try:
        nx_i, ny_i = int(nx), int(ny)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("nx и ny должны быть положительными целыми числами.") from exc
    if nx_i != nx or ny_i != ny or nx_i <= 0 or ny_i <= 0:
        raise ValueError("nx и ny должны быть положительными целыми числами.")
    expected_yx = raw.shape[1:] if axis == 0 else raw.shape[:2]
    if expected_yx != (ny_i, nx_i) or raw.shape[axis] != len(th):
        raise ValueError(f"Несогласованные raw/th/размеры: raw={raw.shape}, th={th.shape}, axis={axis}, nx={nx_i}, ny={ny_i}.")
    nx, ny = nx_i, ny_i

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


def _rect_roi_window_mask(roi, nx, ny):
    """Возвращает ограниченную маску прямоугольной ROI и её срезы."""
    if str(getattr(roi, "type", "")).lower() != "rect":
        raise ValueError("Маска поддерживается только для прямоугольной ROI.")
    roi = roi.normalized()
    values = np.asarray([roi.cx, roi.cy, roi.width, roi.height, roi.angle], dtype=float)
    if not np.all(np.isfinite(values)) or roi.width <= 0 or roi.height <= 0:
        raise ValueError("Некорректные параметры прямоугольной ROI.")
    try:
        bbox = np.asarray(roi.bbox, dtype=float)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Не удалось получить bbox прямоугольной ROI.") from exc
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise ValueError("bbox ROI должен содержать четыре конечные координаты.")
    x0, x1, y0, y1 = map(float, bbox)
    if x1 < x0 or y1 < y0:
        raise ValueError("Некорректный bbox прямоугольной ROI.")
    x_min = int(np.clip(np.floor(x0), 0, nx))
    x_max = int(np.clip(np.ceil(x1), 0, nx))
    y_min = int(np.clip(np.floor(y0), 0, ny))
    y_max = int(np.clip(np.ceil(y1), 0, ny))
    xs = np.arange(x_min, x_max, dtype=float)
    ys = np.arange(y_min, y_max, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    # a = np.deg2rad(float(roi.angle)); c, s = float(np.cos(a)), float(np.sin(a))
    # dx = xx - float(roi.cx); dy = yy - float(roi.cy)
    # u = c * dx + s * dy; v = -s * dx + c * dy
    u, v = roi.to_local(xx, yy)   # 📍
    mask = (np.abs(u) <= 0.5 * float(roi.width) + 1e-9) & (np.abs(v) <= 0.5 * float(roi.height) + 1e-9)
    return roi, x_min, x_max, y_min, y_max, mask


def roi_pixel_mask(roi, nx, ny):
    """Возвращает маску (ny, nx) центров пикселей внутри прямоугольной ROI."""
    try:
        nx_i, ny_i = int(nx), int(ny)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("nx и ny должны быть положительными целыми числами.") from exc
    if nx_i != nx or ny_i != ny or nx_i <= 0 or ny_i <= 0:
        raise ValueError("nx и ny должны быть положительными целыми числами.")
    _, x_min, x_max, y_min, y_max, local = _rect_roi_window_mask(roi, nx_i, ny_i)
    mask = np.zeros((ny_i, nx_i), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = local
    return mask



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
    roi, x_min, x_max, y_min, y_max, mask = _rect_roi_window_mask(roi, nx, ny)
    count = int(mask.sum())

    if count <= 0:
        empty = np.empty((0, len(th))) if reduction_mode is None else np.array([], dtype=float)
        return th.copy(), empty, 0

    if axis == 0:
        sub = raw[:, y_min:y_max, x_min:x_max]
        if reduction_mode is None:
            return th.copy(), np.asarray(sub[:, mask].T, dtype=float), count
        curve = sub[:, mask].sum(axis=1)
    else:
        sub = raw[y_min:y_max, x_min:x_max, :]
        if reduction_mode is None:
            return th.copy(), np.asarray(sub[mask, :], dtype=float), count
        curve = sub[mask, :].sum(axis=0)

    if reduction_mode == "mean":
        curve = curve / float(count)
    return th.copy(), np.asarray(curve, dtype=float), count



def extract_roi_volume(raw, th, roi, *, axis=0, valid_mask=None, order=1):
    """Извлекает объём ROI: xy без поворота, uv с интерполяцией."""
    raw = np.asarray(raw); th = np.asarray(th, dtype=float)
    if raw.ndim != 3 or th.ndim != 1: raise ValueError("raw должен быть 3D, th — 1D")
    if axis not in (0, 2) or raw.shape[axis] != len(th): raise ValueError("Несогласованные raw, th или axis")
    if order not in (0, 1): raise ValueError("order должен быть 0 или 1")
    data = np.moveaxis(raw, axis, 0); nt, ny, nx = data.shape
    r = roi.normalized()
    if r.type.lower() != "rect" or not np.all(np.isfinite([r.cx, r.cy, r.width, r.height, r.angle])) or min(r.width, r.height) <= 0:
        raise ValueError("Ожидается корректная прямоугольная ROI")
    mask = None
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        if m.ndim == 2 and m.shape == (ny, nx): mask = np.broadcast_to(m, (nt, ny, nx))
        elif m.shape == raw.shape: mask = np.moveaxis(m, axis, 0)
        elif m.shape == (ny, nx, nt): mask = np.moveaxis(m, 2, 0)
        elif m.shape == data.shape: mask = m
        else: raise ValueError(f"Неверная форма valid_mask: {m.shape}")
    aligned = abs(np.sin(2 * np.deg2rad(r.angle))) < 1e-10
    if aligned:
        r, x0, x1, y0, y1, region = _rect_roi_window_mask(r, nx, ny)
        xs0, ys0 = np.arange(x0, x1), np.arange(y0, y1)
        xs, ys = xs0[region.any(axis=0)], ys0[region.any(axis=1)]
        if not len(xs) or not len(ys): raise ValueError("ROI не содержит центров пикселей детектора")
        sub = np.asarray(data[:, ys, :][:, :, xs], dtype=float)
        good = np.isfinite(sub)
        if mask is not None: good &= mask[:, ys, :][:, :, xs]
        vol = np.transpose(np.where(good, sub, np.nan), (2, 1, 0))
        return ROIVolume(xs.astype(float), ys.astype(float), th.copy(), vol, "xy", r.copy())
    u = np.linspace(-r.width / 2, r.width / 2, max(2, int(np.ceil(r.width)) + 1))
    v = np.linspace(-r.height / 2, r.height / 2, max(2, int(np.ceil(r.height)) + 1))
    U, V = np.meshgrid(u, v, indexing="ij")
    Xd, Yd = r.to_detector(U, V)
    coords = [Yd, Xd]; vol = np.empty((len(u), len(v), nt), dtype=float)
    for i, layer in enumerate(data):
        layer = np.asarray(layer, dtype=float); good = np.isfinite(layer)
        if mask is not None: good &= mask[i]
        num = map_coordinates(np.where(good, layer, 0.0), coords, order=order, mode="constant", cval=0.0)
        den = map_coordinates(good.astype(float), coords, order=order, mode="constant", cval=0.0)
        vol[:, :, i] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-8)
    return ROIVolume(u, v, th.copy(), vol, "uv", r.copy())