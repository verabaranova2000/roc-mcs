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
    raw, th, axis, roi, nx, ny, reduction_mode=None, *, x_coords=None, y_coords=None, valid_mask=None,
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
    if x_coords is not None or y_coords is not None:
        if x_coords is None or y_coords is None:
            raise ValueError("x_coords и y_coords нужно передавать вместе.")
        x_coords = _validate_grid_axis(x_coords, nx, "x_coords")
        y_coords = _validate_grid_axis(y_coords, ny, "y_coords")
        return _extract_from_roi_grid(raw, th, axis, roi, x_coords, y_coords, mode, valid_mask)
    if valid_mask is not None:
        raise ValueError("Для valid_mask передай также x_coords и y_coords.")

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
    u, v = roi.to_local(xx, yy)   # 📍
    mask = (np.abs(u) <= 0.5 * float(roi.width) + 1e-9) & (np.abs(v) <= 0.5 * float(roi.height) + 1e-9)
    return roi, x_min, x_max, y_min, y_max, mask


def roi_pixel_mask(roi, nx, ny, *, x_coords=None, y_coords=None):
    """Маска центров пикселей; без осей координаты трактуются как индексы."""
    try:
        nx_i, ny_i = int(nx), int(ny)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("nx и ny должны быть положительными целыми числами.") from exc
    if nx_i != nx or ny_i != ny or nx_i <= 0 or ny_i <= 0:
        raise ValueError("nx и ny должны быть положительными целыми числами.")
    if x_coords is not None or y_coords is not None:
        if x_coords is None or y_coords is None:
            raise ValueError("x_coords и y_coords нужно передавать вместе.")
        return roi_grid_mask(roi, _validate_grid_axis(x_coords, nx_i, "x_coords"), _validate_grid_axis(y_coords, ny_i, "y_coords"))
    _, x_min, x_max, y_min, y_max, local = _rect_roi_window_mask(roi, nx_i, ny_i)
    mask = np.zeros((ny_i, nx_i), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = local
    return mask


def _validate_grid_axis(values, size, name):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size != size or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} должна быть конечной непустой 1D-осью длины {size}.")
    if a.size > 1 and not np.all(np.diff(a) > 0):
        raise ValueError(f"{name} должна быть строго возрастающей.")
    return a


def roi_grid_mask(roi, x_coords, y_coords):
    """Маска (Ny_view, Nx_view) для ROI в координатах исходного детектора."""
    r = roi.normalized()
    if r.type.lower() != "rect":
        raise ValueError("roi_grid_mask поддерживает прямоугольную ROI.")
    x = np.asarray(x_coords, dtype=float); y = np.asarray(y_coords, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_coords и y_coords должны быть одномерными осями.")
    x = _validate_grid_axis(x, x.size, "x_coords"); y = _validate_grid_axis(y, y.size, "y_coords")
    xx, yy = np.meshgrid(x, y)
    return np.asarray(r.contains(xx, yy), dtype=bool)


def _nearest_axis_indices(axis_values, values):
    axis_values, values = np.asarray(axis_values, float), np.asarray(values, float)
    right = np.clip(np.searchsorted(axis_values, values), 0, len(axis_values) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(values - axis_values[left]) <= np.abs(values - axis_values[right])
    idx = np.where(choose_left, left, right)
    step_left = axis_values[1] - axis_values[0] if len(axis_values) > 1 else 1.0
    step_right = axis_values[-1] - axis_values[-2] if len(axis_values) > 1 else 1.0
    half_left, half_right = step_left / 2, step_right / 2
    inside = (values >= axis_values[0] - half_left) & (values <= axis_values[-1] + half_right)
    return idx.astype(int), inside


def _roi_grid_indices(roi, x_coords, y_coords):
    """Индексы узлов текущей сетки, центры которых попадают в ROI."""
    r = roi.normalized(); x = np.asarray(x_coords, float); y = np.asarray(y_coords, float)
    kind = r.type.lower()
    if kind == "point":
        ix, in_x = _nearest_axis_indices(x, np.asarray([r.cx]))
        iy, in_y = _nearest_axis_indices(y, np.asarray([r.cy]))
        if not (in_x[0] and in_y[0]): return np.empty(0, int), np.empty(0, int)
        return iy, ix
    if kind == "rect":
        mask = roi_grid_mask(r, x, y)
        iy, ix = np.nonzero(mask)
        return iy.astype(int), ix.astype(int)
    if kind == "line":
        length = float(r.width)
        if not np.isfinite(length) or length <= 0: return np.empty(0, int), np.empty(0, int)
        dx = np.median(np.diff(x)) if len(x) > 1 else 1.0
        dy = np.median(np.diff(y)) if len(y) > 1 else 1.0
        step = min(abs(dx), abs(dy))
        n = max(2, int(np.ceil(length / max(step, 1e-12))) + 1)
        a = np.deg2rad(float(r.angle)); d = np.linspace(-length / 2, length / 2, n)
        xs = r.cx + d * np.cos(a); ys = r.cy + d * np.sin(a)
        ix, in_x = _nearest_axis_indices(x, xs); iy, in_y = _nearest_axis_indices(y, ys)
        pairs = np.column_stack((iy[in_x & in_y], ix[in_x & in_y]))
        if not len(pairs): return np.empty(0, int), np.empty(0, int)
        _, first = np.unique(pairs, axis=0, return_index=True); pairs = pairs[np.sort(first)]
        return pairs[:, 0].astype(int), pairs[:, 1].astype(int)
    raise ValueError(f"Неизвестный тип ROI: {r.type!r}")


def _extract_from_roi_grid(raw, th, axis, roi, x_coords, y_coords, mode, valid_mask=None):
    """Извлекает данные из ROI в координатах детектора; маска относится к текущему view."""
    raw = np.asarray(raw); nt = len(th); ny, nx = (raw.shape[1:] if axis == 0 else raw.shape[:2])
    iy, ix = _roi_grid_indices(roi, x_coords, y_coords)
    yxt = np.moveaxis(raw, axis, -1) if axis != 2 else raw
    curves = np.asarray(yxt[iy, ix, :], dtype=float) if len(ix) else np.empty((0, nt), dtype=float)
    valid = np.isfinite(curves)
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        if m.shape == raw.shape: m = np.moveaxis(m, axis, -1) if axis != 2 else m
        elif m.shape == (ny, nx, nt): pass
        elif m.shape == (nt, ny, nx): m = np.moveaxis(m, 0, -1)
        else: raise ValueError(f"valid_mask имеет форму {m.shape}; ожидается current-view raw или Y×X×theta")
        if len(ix): valid &= m[iy, ix, :]
    curves = np.where(valid, curves, np.nan)
    count = len(ix)
    if mode is None: return th.copy(), curves, count
    if not count: return th.copy(), np.array([], dtype=float), 0
    n_valid = np.sum(np.isfinite(curves), axis=0)
    totals = np.nansum(curves, axis=0)
    if mode == "sum": curve = np.where(n_valid > 0, totals, np.nan)
    else: curve = np.divide(totals, n_valid, out=np.full(nt, np.nan), where=n_valid > 0)
    return th.copy(), np.asarray(curve, dtype=float), count



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



def extract_roi_volume(raw, th, roi, *, axis=0, valid_mask=None, order=1, x_coords=None, y_coords=None):
    """Извлекает объём ROI. Геометрия ROI задана в координатах исходной сетки."""
    raw = np.asarray(raw); th = np.asarray(th, dtype=float)
    if raw.ndim != 3 or th.ndim != 1: raise ValueError("raw должен быть 3D, th — 1D")
    if axis not in (0, 2) or raw.shape[axis] != len(th): raise ValueError("Несогласованные raw, th или axis")
    if order not in (0, 1): raise ValueError("order должен быть 0 или 1")
    data = np.moveaxis(raw, axis, 0); nt, ny, nx = data.shape
    x_axis = _validate_grid_axis(np.arange(nx) if x_coords is None else x_coords, nx, "x_coords")
    y_axis = _validate_grid_axis(np.arange(ny) if y_coords is None else y_coords, ny, "y_coords")
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
        else: raise ValueError(f"Неверная форма valid_mask: {m.shape}; ожидается маска текущего spatial-view")

    # Для углов, кратных 90°, берём реальные отсчёты сетки без интерполяции.
    aligned = abs(np.sin(2 * np.deg2rad(r.angle))) < 1e-10
    if aligned:
        xx, yy = np.meshgrid(x_axis, y_axis)
        region = r.contains(xx, yy)
        x_idx = np.flatnonzero(region.any(axis=0)); y_idx = np.flatnonzero(region.any(axis=1))
        if not len(x_idx) or not len(y_idx): raise ValueError("ROI не содержит центров пикселей текущей сетки")
        selected = region[np.ix_(y_idx, x_idx)]
        sub = np.asarray(data[:, y_idx, :][:, :, x_idx], dtype=float)
        good = np.isfinite(sub) & selected[None, :, :]
        if mask is not None: good &= mask[:, y_idx, :][:, :, x_idx]
        vol = np.transpose(np.where(good, sub, np.nan), (2, 1, 0))
        return ROIVolume(x_axis[x_idx].copy(), y_axis[y_idx].copy(), th.copy(), vol, "xy", r.copy())

    # Повернутый ROI пересэмплируется с шагом текущего grid, а не исходного пикселя.
    dx = float(np.median(np.diff(x_axis))) if nx > 1 else 1.0
    dy = float(np.median(np.diff(y_axis))) if ny > 1 else 1.0
    step = min(abs(dx), abs(dy))
    u = np.linspace(-r.width / 2, r.width / 2, max(2, int(np.ceil(r.width / step)) + 1))
    v = np.linspace(-r.height / 2, r.height / 2, max(2, int(np.ceil(r.height / step)) + 1))
    U, V = np.meshgrid(u, v, indexing="ij")
    Xd, Yd = r.to_detector(U, V)

    def to_index(coords, axis_values):
        if len(axis_values) == 1:
            idx = np.zeros_like(coords, dtype=float)
            inside = np.abs(coords - axis_values[0]) <= step / 2
        else:
            idx = np.interp(coords, axis_values, np.arange(len(axis_values)), left=np.nan, right=np.nan)
            inside = np.isfinite(idx)
        return idx, inside

    ix, inside_x = to_index(Xd, x_axis); iy, inside_y = to_index(Yd, y_axis)
    in_grid = inside_x & inside_y
    safe_coords = [np.where(in_grid, iy, -1.0), np.where(in_grid, ix, -1.0)]
    vol = np.full((len(u), len(v), nt), np.nan, dtype=float)
    for i, layer in enumerate(data):
        layer = np.asarray(layer, dtype=float); good = np.isfinite(layer)
        if mask is not None: good &= mask[i]
        num = map_coordinates(np.where(good, layer, 0.0), safe_coords, order=order, mode="constant", cval=0.0)
        den = map_coordinates(good.astype(float), safe_coords, order=order, mode="constant", cval=0.0)
        vals = np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 1e-8) & in_grid)
        vol[:, :, i] = vals
    return ROIVolume(u, v, th.copy(), vol, "uv", r.copy())

