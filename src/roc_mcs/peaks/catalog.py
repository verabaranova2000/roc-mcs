import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict

from roc_mcs.peaks.detection import PIXEL_PEAK_CONFIG, detect_peaks
from roc_mcs.utils import to_yxt
from roc_mcs.processing.background import BackgroundStats, estimate_background_stats
from roc_mcs.roi.extraction import roi_pixel_mask


def build_peak_catalog_v0(
    data_3d,
    th,
    window_size=3,
    peak_detection_config=PIXEL_PEAK_CONFIG,
):
    """
    Формирует каталог локальных максимумов дифракционного сигнала.
    
    Каждая строка соответствует одному peak event (обнаруженному пику в локальной
    области ROC), определяемой пространственной окрестностью 3×3.
    
    Столбцы:
        peak_id
        x, y
        theta_index
        theta
        prominence
        height
    
    Пространственная группировка пиков (clustering) и отслеживание ветвей
    на данном этапе не выполняются.
    """
    data = to_yxt(data_3d, th)
    Ny, Nx, Ntheta = data.shape

    # ------------------------------------------------------
    # Шум — ровно моя прежняя оценка
    # ------------------------------------------------------
    max_map = np.max(data, axis=-1)
    q30 = np.percentile(max_map, 30)
    bg_pixels = max_map[max_map <= q30]
    bg_median = np.median(bg_pixels)
    bg_mad = np.median( np.abs(bg_pixels - bg_median))
    bg_sigma = 1.4826 * bg_mad if bg_mad > 0 else 1.0
    bg_threshold = bg_median + 5 * bg_sigma

    print( f"background median   = {bg_median:.6g}")
    print(f"noise sigma          = {bg_sigma:.6g}")
    print(f"background threshold = {bg_threshold:.6g}")

    # ------------------------------------------------------
    # Peak catalogue
    # ------------------------------------------------------
    records = []
    half_w = window_size // 2
    peak_id = 0
    for y in tqdm(range(Ny), desc="Building peak catalogue", unit="row"):
        y_s = max(0, y - half_w)
        y_e = min(Ny, y + half_w + 1)
        for x in range(Nx):
            if max_map[y, x] < bg_threshold:
                continue
            x_s = max(0, x - half_w)
            x_e = min(Nx, x + half_w + 1)
            mean_curve = np.mean(data[y_s:y_e, x_s:x_e, :], axis=(0, 1))
            smooth, peaks, props, valid_mask = (
                detect_peaks(
                    mean_curve,
                    peak_detection_config,
                    noise_sigma=bg_sigma,
                )
            )
            for k, is_valid in enumerate(valid_mask):
                if not is_valid:
                    continue
                peak_index = int(peaks[k])
                records.append({
                    "peak_id": peak_id,
                    "x": x,
                    "y": y,
                    "theta_index": peak_index,
                    "theta": float(th[peak_index]),
                    "prominence": float(props["prominences"][k]),
                    "width_samples": float(props["widths"][k]),                         # ширина в отсчётах
                    "width_theta": float(props["widths"][k] * np.median(np.diff(th))),  # ширина в углах
                    "left_ip": float(props["left_ips"][k]),       # левая граница пика
                    "right_ip": float(props["right_ips"][k]),     # правая граница пика
                    "height": float(smooth[peak_index]),
                })
                peak_id += 1
    peak_catalog = pd.DataFrame(records)
    return (
        peak_catalog,
        bg_sigma,
        bg_threshold,
    )


def build_peak_catalog_v1(
    data_3d,
    th,
    window_size=3,
    peak_detection_config=PIXEL_PEAK_CONFIG,
    background_stats=None,
):
    """
    Формирует каталог локальных максимумов дифракционного сигнала.
    
    Каждая строка соответствует одному peak event (обнаруженному пику в локальной
    области ROC), определяемой пространственной окрестностью 3×3.
    
    Столбцы:
        peak_id
        x, y
        theta_index
        theta
        prominence
        height
    
    Пространственная группировка пиков (clustering) и отслеживание ветвей
    на данном этапе не выполняются.
    """
    data = to_yxt(data_3d, th)
    Ny, Nx, Ntheta = data.shape

    # ------------------------------------------------------
    # Шум — ровно моя прежняя оценка
    # ------------------------------------------------------
    # max_map = np.max(data, axis=-1)
    # q30 = np.percentile(max_map, 30)
    # bg_pixels = max_map[max_map <= q30]
    # bg_median = np.median(bg_pixels)
    # bg_mad = np.median( np.abs(bg_pixels - bg_median))
    # bg_sigma = 1.4826 * bg_mad if bg_mad > 0 else 1.0
    # bg_threshold = bg_median + 5 * bg_sigma

    # print( f"background median   = {bg_median:.6g}")
    # print(f"noise sigma          = {bg_sigma:.6g}")
    # print(f"background threshold = {bg_threshold:.6g}")

    background = estimate_background_stats(data) if background_stats is None else background_stats
    max_map = np.max(data, axis=-1)

    print(f"background median   = {background.median:.6g}")
    print(f"noise sigma          = {background.sigma:.6g}")
    print(f"background threshold = {background.threshold:.6g}")

    # ------------------------------------------------------
    # Peak catalogue
    # ------------------------------------------------------
    records = []
    half_w = window_size // 2
    peak_id = 0
    for y in tqdm(range(Ny), desc="Building peak catalogue", unit="row"):
        y_s = max(0, y - half_w)
        y_e = min(Ny, y + half_w + 1)
        for x in range(Nx):
            if max_map[y, x] < background.threshold:
                continue
            x_s = max(0, x - half_w)
            x_e = min(Nx, x + half_w + 1)
            mean_curve = np.mean(data[y_s:y_e, x_s:x_e, :], axis=(0, 1))
            smooth, peaks, props, valid_mask = (
                detect_peaks(
                    mean_curve,
                    peak_detection_config,
                    noise_sigma=background.sigma,
                )
            )
            for k, is_valid in enumerate(valid_mask):
                if not is_valid:
                    continue
                peak_index = int(peaks[k])
                records.append({
                    "peak_id": peak_id,
                    "x": x,
                    "y": y,
                    "theta_index": peak_index,
                    "theta": float(th[peak_index]),
                    "prominence": float(props["prominences"][k]),
                    "width_samples": float(props["widths"][k]),                         # ширина в отсчётах
                    "width_theta": float(props["widths"][k] * np.median(np.diff(th))),  # ширина в углах
                    "left_ip": float(props["left_ips"][k]),       # левая граница пика
                    "right_ip": float(props["right_ips"][k]),     # правая граница пика
                    "height": float(smooth[peak_index]),
                })
                peak_id += 1
    peak_catalog = pd.DataFrame(records)
    return peak_catalog, background.sigma, background.threshold



def build_peak_catalog(
    data_3d,
    th,
    window_size=3,
    peak_detection_config=PIXEL_PEAK_CONFIG,
    background_stats=None,
):
    """
    Формирует каталог локальных максимумов дифракционного сигнала.
    
    Каждая строка соответствует одному peak event (обнаруженному пику в локальной
    области ROC), определяемой пространственной окрестностью 3×3.
    
    Столбцы:
        peak_id
        x, y
        theta_index
        theta
        prominence
        height
    
    Пространственная группировка пиков (clustering) и отслеживание ветвей
    на данном этапе не выполняются.
    """
    data_3d = np.asarray(data_3d, dtype=float)

    # BackgroundStats работает с каноническим layout (Ntheta, Ny, Nx).
    background = (
        estimate_background_stats(data_3d)
        if background_stats is None
        else background_stats
    )

    # Для локальных ROC дальше используем (Ny, Nx, Ntheta).
    data = to_yxt(data_3d, th)
    Ny, Nx, Ntheta = data.shape

    max_map = np.max(data, axis=-1)

    print(f"background median   = {background.median:.6g}")
    print(f"noise sigma          = {background.sigma:.6g}")
    print(f"background threshold = {background.threshold:.6g}")

    records = []
    half_w = window_size // 2
    peak_id = 0

    for y in tqdm(range(Ny), desc="Building peak catalogue", unit="row"):
        y_s = max(0, y - half_w)
        y_e = min(Ny, y + half_w + 1)

        for x in range(Nx):
            if max_map[y, x] < background.threshold:
                continue

            x_s = max(0, x - half_w)
            x_e = min(Nx, x + half_w + 1)
            mean_curve = np.mean(data[y_s:y_e, x_s:x_e, :], axis=(0, 1))
            smooth, peaks, props, valid_mask = detect_peaks(
                mean_curve,
                peak_detection_config,
                noise_sigma=background.sigma,
            )

            for k, is_valid in enumerate(valid_mask):
                if not is_valid:
                    continue

                peak_index = int(peaks[k])

                records.append({
                    "peak_id": peak_id,
                    "x": x,
                    "y": y,
                    "theta_index": peak_index,
                    "theta": float(th[peak_index]),
                    "prominence": float(props["prominences"][k]),
                    "width_samples": float(props["widths"][k]),
                    "width_theta": float(
                        props["widths"][k] * np.median(np.diff(th))
                    ),
                    "left_ip": float(props["left_ips"][k]),
                    "right_ip": float(props["right_ips"][k]),
                    "height": float(smooth[peak_index]),
                })
                peak_id += 1
    return pd.DataFrame(records), background.sigma, background.threshold



def enrich_peak_catalog(peak_catalog, Nx):
    """
    Функция добавляет производные поля (исходный каталог обогащается метаданными):
        pixel_id
        component_id
        n_components
        
    Переводит сырой catalog в удобное для fitting представление:
        (x, y)
        ↓
        pixel_id
        ↓
        сортировка
        ↓
        component_id
        ↓
        n_components
    """
    pc = peak_catalog.copy()
    pc["pixel_id"] = pc["y"].astype(int) * Nx + pc["x"].astype(int)
    pc = pc.sort_values(["pixel_id", "theta"]).reset_index(drop=True)
    pc["component_id"] = pc.groupby("pixel_id", sort=False).cumcount()
    pc["n_components"] = pc.groupby("pixel_id")["component_id"].transform("size").astype(int)
    return pc


def build_pixel_peak_lookup(peak_catalog):
    """
    Формирует lookup обнаруженных пиков по координатам detector pixel.

    Lookup сохраняет идентичность и provenance каждого detected peak,
    необходимые для последующей pixel-level fitting.

    Parameters
    ----------
    peak_catalog : pandas.DataFrame
        Каталог detected peaks.

        Required columns:
            peak_id
            x
            y
            theta_index
            theta
            prominence
            height

    Returns
    -------
    lookup : defaultdict
        Словарь вида

            (x, y) -> [peak_dict, ...]

        Пики внутри каждого pixel отсортированы по theta.

    Notes
    -----
    peak_id является стабильным идентификатором detected peak
    и не изменяется после fitting.
    """
    required = {
        "peak_id",
        "x",
        "y",
        "theta_index",
        "theta",
        "prominence",
        "height",
    }
    missing = required - set(peak_catalog.columns)
    if missing:
        raise ValueError(f"peak_catalog is missing required columns: {sorted(missing)}")
    
    lookup = defaultdict(list)
    for row in peak_catalog.itertuples(index=False):
        lookup[(int(row.x), int(row.y))].append(
            {
                "peak_id": int(row.peak_id),
                "theta_index": int(row.theta_index),
                "theta": float(row.theta),
                "prominence": float(row.prominence),
                "height": float(row.height),
            }
        )
    for key in lookup:
        lookup[key].sort(key=lambda peak: peak["theta"])
    return lookup




def filter_peak_catalog_by_roi(peak_catalog, roi, nx, ny):
    """Фильтрует строки peak_catalog по прямоугольной ROI, сохраняя исходные столбцы/ID."""
    from roc_mcs.peaks.catalog import enrich_peak_catalog

    if not hasattr(peak_catalog, "columns") or not hasattr(peak_catalog, "iloc"):
        raise TypeError("peak_catalog должен быть pandas DataFrame.")
    if len(peak_catalog) == 0:
        raise ValueError("Исходный peak_catalog пуст.")
    mask = roi_pixel_mask(roi, nx, ny)
    pc = enrich_peak_catalog(peak_catalog, int(nx))
    required = {"x", "y", "pixel_id"}
    if not required.issubset(pc.columns):
        raise ValueError(f"enrich_peak_catalog должен возвращать столбцы {sorted(required)}.")
    x = np.asarray(pc["x"], dtype=float); y = np.asarray(pc["y"], dtype=float)
    if len(pc) != len(peak_catalog) or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Каталог содержит строки с некорректными координатами или enrich изменил число строк.")
    if not (np.all(np.abs(x - np.rint(x)) <= 1e-8) and np.all(np.abs(y - np.rint(y)) <= 1e-8)):
        raise ValueError("Координаты пиков должны соответствовать целым индексам пикселей.")
    x = np.rint(x).astype(np.int64); y = np.rint(y).astype(np.int64)
    if np.any((x < 0) | (x >= int(nx)) | (y < 0) | (y >= int(ny))):
        raise ValueError("В обогащённом peak_catalog есть координаты вне изображения.")
    pids = np.asarray(pc["pixel_id"], dtype=float)
    if not np.all(np.isfinite(pids)) or not np.all(np.abs(pids - np.rint(pids)) <= 1e-8):
        raise ValueError("pixel_id должен содержать конечные целочисленные идентификаторы.")
    pids = np.rint(pids).astype(np.int64)
    coords_per_pixel = pc.assign(_x=x, _y=y, _pid=pids).groupby("_pid")[ ["_x", "_y"] ].nunique()
    if (coords_per_pixel > 1).any().any():
        raise ValueError("Один pixel_id сопоставлен нескольким координатам.")
    inside = mask[y, x]
    selected_pixels = np.unique(pids[inside])
    if not len(selected_pixels):
        raise ValueError("В выбранной ROI нет обнаруженных пиков.")

    cols = set(peak_catalog.columns)
    if {"x", "y"}.issubset(cols):
        sx = np.asarray(peak_catalog["x"], dtype=float); sy = np.asarray(peak_catalog["y"], dtype=float)
        if not (np.all(np.isfinite(sx)) and np.all(np.isfinite(sy))):
            raise ValueError("Исходный peak_catalog содержит неконечные координаты.")
        if not (np.all(np.abs(sx - np.rint(sx)) <= 1e-8) and np.all(np.abs(sy - np.rint(sy)) <= 1e-8)):
            raise ValueError("Координаты исходного peak_catalog должны быть целочисленными.")
        sx = np.rint(sx).astype(np.int64); sy = np.rint(sy).astype(np.int64)
        if np.any((sx < 0) | (sx >= int(nx)) | (sy < 0) | (sy >= int(ny))):
            raise ValueError("В исходном peak_catalog есть координаты вне изображения.")
        keep = mask[sy, sx]
    elif "pixel_id" in cols:
        source_ids = np.asarray(peak_catalog["pixel_id"], dtype=float)
        if not np.all(np.isfinite(source_ids)) or not np.all(np.abs(source_ids - np.rint(source_ids)) <= 1e-8):
            raise ValueError("pixel_id исходного каталога должен содержать конечные целые значения.")
        keep = np.isin(np.rint(source_ids).astype(np.int64), selected_pixels)
    else:
        raise ValueError("Нельзя сопоставить строки исходного каталога: нужны x,y или pixel_id.")

    out = peak_catalog.iloc[np.flatnonzero(keep)].copy()
    if out.empty:
        raise ValueError("После фильтрации peak_catalog пуст.")
    check = enrich_peak_catalog(out, int(nx))
    cx = np.rint(np.asarray(check["x"], dtype=float)).astype(np.int64)
    cy = np.rint(np.asarray(check["y"], dtype=float)).astype(np.int64)
    if len(check) != len(out) or not np.all(mask[cy, cx]):
        raise RuntimeError("После фильтрации остались пики вне ROI либо изменилось число строк.")
    info = {"roi_pixels": int(mask.sum()), "rows_before": int(len(peak_catalog)), "rows_after": int(len(out)),
            "pixels_before": int(pc["pixel_id"].nunique()), "pixels_after": int(check["pixel_id"].nunique())}
    return out, info