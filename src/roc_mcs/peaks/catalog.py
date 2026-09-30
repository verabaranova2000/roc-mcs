import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict

from roc_mcs.peaks.detection import PIXEL_PEAK_CONFIG, detect_peaks
from roc_mcs.utils import to_yxt

def build_peak_catalog(
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