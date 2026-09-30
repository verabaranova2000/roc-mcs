from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PeakDetectionConfig:
    smooth_window: int
    smooth_polyorder: int
    min_distance: int
    prominence_fraction: float | None = None
    prominence_sigma: float | None = None
    min_relative_prominence: float | None = None

    def __post_init__(self):
        # Окно Savitzky–Golay должно быть положительным и нечётным
        if self.smooth_window < 3:
            raise ValueError("smooth_window должен быть >= 3.")

        if self.smooth_window % 2 == 0:
            raise ValueError("smooth_window должен быть нечётным.")

        # Степень полинома должна быть меньше ширины окна
        if self.smooth_polyorder < 0:
            raise ValueError("smooth_polyorder должен быть >= 0.")

        if self.smooth_polyorder >= self.smooth_window:
            raise ValueError("smooth_polyorder должен быть меньше smooth_window.")

        # Минимальная дистанция между пиками
        if self.min_distance < 1:
            raise ValueError("min_distance должен быть >= 1.")

        # Должен быть выбран ровно один способ задания prominence
        prominence_modes = [
            self.prominence_fraction is not None,
            self.prominence_sigma is not None,
        ]
        if sum(prominence_modes) != 1:
            raise ValueError("Нужно задать ровно один из параметров: prominence_fraction или prominence_sigma.")

        # Проверяем относительный порог, если он используется
        if self.prominence_fraction is not None:
            if not 0 < self.prominence_fraction:
                raise ValueError("prominence_fraction должен быть > 0.")

        if self.prominence_sigma is not None:
            if self.prominence_sigma <= 0:
                raise ValueError("prominence_sigma должен быть > 0.")

        if self.min_relative_prominence is not None:
            if not 0 < self.min_relative_prominence <= 1:
                raise ValueError("min_relative_prominence должен быть в интервале (0, 1].")    


PIXEL_PEAK_CONFIG = PeakDetectionConfig(
    smooth_window=11,               # 🟢 сглаживание
    smooth_polyorder=2,             # 🟢 степень сглаживания
    min_distance=6,                 # 🟢 минимальное расстояние между пиками 
    prominence_sigma=3.0,           # 🟢 prominence должна быть ≥ 3σ шума
    min_relative_prominence=0.07,   # 🟢 дополнительная фильтрация слабых пиков
)


from scipy.signal import find_peaks, savgol_filter, peak_widths

def detect_peaks(
    curve: np.ndarray,
    peak_detection_config: PeakDetectionConfig,
    noise_sigma: float | None = None,
):
    """
    Сглаживает ROC-кривую и обнаруживает потенциальные пики
    согласно переданной конфигурации.

    Parameters
    ----------
    curve : np.ndarray
        Одномерная ROC-кривая.
    peak_detection_config : PeakDetectionConfig
        Параметры сглаживания и обнаружения пиков.
    noise_sigma : float | None
        Оценка шума. Требуется, если в конфигурации
        используется prominence_sigma.

    Returns
    -------
    smooth : np.ndarray
        Сглаженная кривая.
    peaks : np.ndarray
        Индексы обнаруженных пиков.
    properties : dict
        Результаты scipy.signal.find_peaks.
        "width_heights":
            ширина на уровне половины prominence, 
            а не просто половины абсолютной высоты пика. Это ширина между left_ips и right_ips
    """
    curve = np.asarray(curve, dtype=float)
    smooth = savgol_filter(
        curve,
        window_length=peak_detection_config.smooth_window,
        polyorder=peak_detection_config.smooth_polyorder,
    )
    # --- 🔴 (ROI) ---
    if peak_detection_config.prominence_fraction is not None:
        prominence = (
            np.ptp(smooth)
            * peak_detection_config.prominence_fraction
        )
    # --- 🟢 (Pixel) ---
    elif peak_detection_config.prominence_sigma is not None:
        if noise_sigma is None:
            raise ValueError("noise_sigma должен быть передан, если используется prominence_sigma.")

        prominence = peak_detection_config.prominence_sigma * noise_sigma

    else:
        raise ValueError("Не задан ни prominence_fraction, ни prominence_sigma.")
    # --- Поиск всех кандидатов --- 
    peaks, properties = find_peaks(
        smooth,
        prominence=prominence,
        distance=peak_detection_config.min_distance,
    )
    # --- Ширина каждого найденного пика на уровне half-prominence ---
    if len(peaks) > 0:
        widths, width_heights, left_ips, right_ips = peak_widths(
            smooth,
            peaks,
            rel_height=0.5,
        )
        properties["widths"] = widths
        properties["width_heights"] = width_heights
        properties["left_ips"] = left_ips
        properties["right_ips"] = right_ips
    else:
        properties["widths"] = np.array([], dtype=float)
        properties["width_heights"] = np.array([], dtype=float)
        properties["left_ips"] = np.array([], dtype=float)
        properties["right_ips"] = np.array([], dtype=float)    
    
    # --- Какие кандидаты признать валидными ---
    valid_mask = np.ones(len(peaks), dtype=bool)

    if peak_detection_config.min_relative_prominence is not None:  # 🟢 (Pixel)
        if len(peaks) > 0:
            max_prominence = np.max(properties["prominences"])     # Главный (самый мощный) пик всегда валиден
            # Валидными считаем только те вершины, чья мощность составляет не менее 7% от главного пика
            valid_mask = (
                properties["prominences"]
                >= peak_detection_config.min_relative_prominence * max_prominence
            )
            
    return smooth, peaks, properties, valid_mask