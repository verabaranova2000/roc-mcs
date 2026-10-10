from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patheffects as pe
from tqdm.auto import tqdm


from roc_mcs.processing.background import BackgroundStats, estimate_background_stats


@dataclass(frozen=True, slots=True)
class KappaThresholdFit_v1:
    """Данные для анализа kappa-fit."""
    kappa: float
    n_spikes: np.ndarray
    kappas_positive: np.ndarray
    n_spikes_positive: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray
    p_tail: np.ndarray | None = None
    kappa_tail: float | None = None
    n_segments: int = 2
    bic_2: float = np.nan
    bic_3: float = np.nan

    @property
    def slopes(self) -> tuple[float, ...]:
        """Наклоны сегментов в координатах kappa–log10(N)."""
        slopes = [float(self.p_left[0]), float(self.p_right[0])]
        if self.p_tail is not None:
            slopes.append(float(self.p_tail[0]))
        return tuple(slopes)



@dataclass(frozen=True, slots=True)
class KappaThresholdFit:
    """Данные для анализа kappa-fit."""
    kappa: float
    n_spikes: np.ndarray
    kappas_positive: np.ndarray
    n_spikes_positive: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray
    p_tail: np.ndarray | None = None
    kappa_tail: float | None = None
    n_segments: int = 2
    bic_2: float = np.nan
    bic_3: float = np.nan

    @property
    def slopes(self) -> tuple[float, ...]:
        """Наклоны сегментов d log10(N) / d kappa."""
        slopes = [float(self.p_left[0]), float(self.p_right[0])]
        if self.p_tail is not None:
            slopes.append(float(self.p_tail[0]))
        return tuple(slopes)

    @property
    def delta_bic(self) -> float:
        return float(self.bic_2 - self.bic_3)

    @property
    def relative_tail_slope_change(self) -> float | None:
        if self.p_tail is None:
            return None
        return float(abs(self.p_tail[0] - self.p_right[0]) / max(abs(self.p_right[0]), abs(self.p_tail[0]), 1e-12))
    



@dataclass(frozen=True, slots=True)
class SpikeMaskResult:
    """Результат итеративного уточнения spike mask."""
    spike_mask: np.ndarray
    clean_data: np.ndarray
    background: BackgroundStats
    kappa: float
    history: list[dict]
    converged: bool
    iterations: int
    kappa_fits: list[KappaThresholdFit] | None = None





def interpolate_masked_points(
    data_3d: np.ndarray,
    spike_mask: np.ndarray,
) -> np.ndarray:
    """Заменяет spike-точки интерполяцией по theta."""
    data_3d = np.asarray(data_3d, dtype=float)
    spike_mask = np.asarray(spike_mask, dtype=bool)

    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено ndim={data_3d.ndim}")
    if spike_mask.shape != data_3d.shape:
        raise ValueError(
            f"spike_mask имеет форму {spike_mask.shape}, ожидалась {data_3d.shape}"
        )
    if data_3d.shape[0] < 2:
        raise ValueError("Для интерполяции требуется минимум два theta-отсчёта.")

    cleaned_data = data_3d.copy()

    interpolated = (data_3d[:-2] + data_3d[2:]) / 2.0
    middle_mask = spike_mask[1:-1]
    middle = cleaned_data[1:-1]
    middle[middle_mask] = interpolated[middle_mask]

    cleaned_data[0][spike_mask[0]] = data_3d[1][spike_mask[0]]
    cleaned_data[-1][spike_mask[-1]] = data_3d[-2][spike_mask[-1]]

    return cleaned_data


def _compute_spike_z_scores(data_3d: np.ndarray, background_variance: float) -> np.ndarray:
    """
    Вычисляет z-score (статистическую значимость) для точечного spike-критерия.
    Лапласиан (разность точки и среднего её соседей):
    - на линейном или плавном склоне физического пика Лапласиан близок к 0;
    - на вершине реального пика он немного положителен;
    - на космическом луче (спайке шириной в 1 пиксель) он выдает огромный скачок.
    
    Ожидаемая дисперсия Лапласиана:
        Var(yᵢ − 0.5yᵢ₋₁ − 0.5yᵢ₊₁) ≈ 1.5 · μ
    """
    data_3d = np.asarray(data_3d, dtype=float)
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено ndim={data_3d.ndim}")
    if not np.isfinite(background_variance) or background_variance < 0:
        raise ValueError("background_variance должен быть конечным и >= 0.")

    y_left = np.roll(data_3d, 1, axis=0)      # соседи слева и справа вдоль оси theta
    y_right = np.roll(data_3d, -1, axis=0)
    y_left[0] = data_3d[0]                    # защищаем края от циклических сдвигов np.roll
    y_right[-1] = data_3d[-1]

    laplacian = data_3d - (y_left + y_right) / 2.0                       # Лапласиан 
    local_mu = np.maximum(y_left, y_right)                                # чтобы защитить крутые склоны дифракционных пиков, берем максимум из соседей
    expected_variance = 1.5 * np.maximum(local_mu, background_variance)   # ожидаемая дисперсия Лапласиана
    sigma_laplacian = np.sqrt(expected_variance)
    z_scores = laplacian / np.maximum(sigma_laplacian, 1e-6)
    return z_scores 

    

def build_spike_mask(
    data_3d: np.ndarray,
    background_variance: float,
    kappa: float = 6.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Обнаруживает точечные высокоинтенсивные выбросы вдоль оси theta
    по лапласианному критерию с пуассоновской моделью шума.

    Спайк — это то, что пробивает порог kappa (выброс всегда > 0)
    Параметры:
      data_3d: 3D numpy array сырых данных
      background_variance: оценка дисперсии фона (sigma^2)
      kappa: порог отсечения в сигмах (обычно 5.0 - 7.0)

    Возвращает:
        spike_mask == True  → точка признана спайком
        spike_mask == False → точка нормальная
    """
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa должен быть конечным и > 0.")

    data_3d = np.asarray(data_3d, dtype=float)
    z_scores = _compute_spike_z_scores(data_3d, background_variance)
    
    spike_mask = np.isfinite(data_3d) & (z_scores >= kappa)
    n_spikes = np.count_nonzero(spike_mask)
    if verbose:
        print(f"Masking done: {n_spikes} true outliers detected (kappa = {kappa}).")
    return spike_mask

# ------------- ДЛЯ V2 ----------------
def _segment_fit(x, y, a, b, prefix):
    """Линейная регрессия y(x) на [a:b) через префиксные суммы."""
    n = b - a
    if n < 2:
        return None, np.inf

    sx = prefix[0][b] - prefix[0][a]
    sy = prefix[1][b] - prefix[1][a]
    sxx = prefix[2][b] - prefix[2][a]
    sxy = prefix[3][b] - prefix[3][a]
    syy = prefix[4][b] - prefix[4][a]

    den = n * sxx - sx * sx
    if den <= 0:
        return None, np.inf

    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    sse = syy - 2 * slope * sxy - 2 * intercept * sy + slope**2 * sxx + 2 * slope * intercept * sx + intercept**2 * n
    return np.array([slope, intercept]), max(float(sse), 0.0)


def find_kappa_threshold_v1(
    data_3d: np.ndarray,
    background_variance: float,
    kappas: np.ndarray | None = None,
    verbose: bool = False,
) -> KappaThresholdFit:
    """
    Находит первый устойчивый излом positive-tail distribution.
    
    spike_mask == True   → спайк
    spike_mask == False  → не спайк
    """
    data_3d = np.asarray(data_3d, dtype=float)
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено shape={data_3d.shape}")
    if not np.isfinite(background_variance) or background_variance < 0:
        raise ValueError("background_variance должен быть конечным и >= 0.")
    if kappas is None:
        kappas = np.linspace(2.0, 40.0, 39)

    kappas = np.asarray(kappas, dtype=float)
    if kappas.ndim != 1 or len(kappas) < 7 or np.any(~np.isfinite(kappas)) or np.any(kappas <= 0) or np.any(np.diff(kappas) <= 0):
        raise ValueError("kappas должен быть одномерным, конечным, положительным и строго возрастающим массивом минимум из 7 значений.")

    z_scores = _compute_spike_z_scores(data_3d, background_variance)
    finite = np.isfinite(data_3d)
    n_spikes = np.asarray([np.count_nonzero(finite & (z_scores >= k)) for k in kappas], dtype=int)

    positive = n_spikes > 0
    kappas_positive = kappas[positive]
    n_spikes_positive = n_spikes[positive]
    y = np.log10(n_spikes_positive)

    if len(kappas_positive) < 16:
        raise ValueError("Недостаточно точек с ненулевым числом спайков для поиска излома.")

    x = kappas_positive
    n = len(x)
    min_points = 8

    prefix = tuple(np.concatenate(([0.0], np.cumsum(v))) for v in (x, y, x * x, x * y, y * y))

    best2 = (np.inf, None, None, None)
    for i in range(min_points, n - min_points + 1):
        p1, e1 = _segment_fit(x, y, 0, i, prefix)
        p2, e2 = _segment_fit(x, y, i, n, prefix)
        error = e1 + e2
        if error < best2[0]:
            best2 = (error, i, p1, p2)

    bic_2 = n * np.log(max(best2[0] / n, np.finfo(float).tiny)) + 5 * np.log(n)

    best3 = (np.inf, None, None, None, None)
    if n >= 3 * min_points:
        for i in range(min_points, n - 2 * min_points + 1):
            p1, e1 = _segment_fit(x, y, 0, i, prefix)
            for j in range(i + min_points, n - min_points + 1):
                p2, e2 = _segment_fit(x, y, i, j, prefix)
                p3, e3 = _segment_fit(x, y, j, n, prefix)
                error = e1 + e2 + e3
                if error < best3[0]:
                    best3 = (error, i, j, p1, p2, p3)

    bic_3 = np.inf
    if best3[1] is not None:
        bic_3 = n * np.log(max(best3[0] / n, np.finfo(float).tiny)) + 8 * np.log(n)

    use_three = best3[1] is not None and bic_3 < bic_2

    if use_three:
        _, i, j, p_left, p_right, p_tail = best3
        split_1, split_2 = i, j
        n_segments = 3
    else:
        _, i, p_left, p_right = best2
        split_1, split_2 = i, None
        p_tail = None
        n_segments = 2

    def breakpoint(p1, p2, split):
        lo, hi = x[split - 1], x[split]
        denom = p1[0] - p2[0]
        if abs(denom) < 1e-12:
            return float(0.5 * (lo + hi))
        value = (p2[1] - p1[1]) / denom
        return float(value) if np.isfinite(value) and lo <= value <= hi else float(0.5 * (lo + hi))

    if abs(p_left[0] - p_right[0]) < 1e-10:
        raise ValueError("Не удалось надёжно определить первый излом.")

    kappa = breakpoint(p_left, p_right, split_1)
    kappa_tail = None
    if n_segments == 3:
        if abs(p_right[0] - p_tail[0]) < 1e-10:
            n_segments = 2
            p_tail = None
        else:
            kappa_tail = breakpoint(p_right, p_tail, split_2)

    if verbose:
        slopes = ", ".join(f"{s:.4g}" for s in (
            (p_left[0], p_right[0], p_tail[0]) if n_segments == 3 else (p_left[0], p_right[0])
        ))
        print(f"Selected kappa = {kappa:.4f} | segments = {n_segments} | slopes = ({slopes})")

    return KappaThresholdFit(
        kappa=kappa,
        n_spikes=n_spikes,
        kappas_positive=kappas_positive,
        n_spikes_positive=n_spikes_positive,
        p_left=p_left,
        p_right=p_right,
        p_tail=p_tail,
        kappa_tail=kappa_tail,
        n_segments=n_segments,
        bic_2=float(bic_2),
        bic_3=float(bic_3),
    )



def find_kappa_threshold_v2(
    data_3d: np.ndarray,
    background_variance: float,
    kappas: np.ndarray | None = None,
    verbose: bool = False,
    min_segment_points: int = 8,
    bic_margin: float = 10.0,
    min_relative_slope_change: float = 0.25,
) -> KappaThresholdFit:
    """Находит первый устойчивый change point positive-tail distribution."""
    data_3d = np.asarray(data_3d, dtype=float)
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено shape={data_3d.shape}")
    if not np.isfinite(background_variance) or background_variance < 0:
        raise ValueError("background_variance должен быть конечным и >= 0.")
    if kappas is None:
        kappas = np.linspace(2.0, 40.0, 39)

    kappas = np.asarray(kappas, dtype=float)
    if kappas.ndim != 1 or len(kappas) < 7 or np.any(~np.isfinite(kappas)) or np.any(kappas <= 0) or np.any(np.diff(kappas) <= 0):
        raise ValueError("kappas должен быть одномерным, конечным, положительным и строго возрастающим массивом минимум из 7 значений.")

    z_scores = _compute_spike_z_scores(data_3d, background_variance)
    finite = np.isfinite(data_3d)
    n_spikes = np.asarray([np.count_nonzero(finite & (z_scores >= k)) for k in kappas], dtype=int)

    positive = n_spikes > 0
    kappas_positive = kappas[positive]
    n_spikes_positive = n_spikes[positive]
    y = np.log10(n_spikes_positive)

    if len(kappas_positive) < 2 * min_segment_points:
        raise ValueError("Недостаточно точек с ненулевым числом спайков для поиска излома.")

    def bic(sse, n, n_params):
        return n * np.log(max(sse / n, np.finfo(float).tiny)) + n_params * np.log(n)

    x = kappas_positive
    n = len(x)

    # --- Лучшая двухсегментная модель ---
    best2 = (np.inf, None, None, None)

    for i in range(min_segment_points, n - min_segment_points + 1):
        p1 = np.polyfit(x[:i], y[:i], 1)
        p2 = np.polyfit(x[i:], y[i:], 1)

        if p1[0] > 0 or p2[0] > 0:
            continue

        e1 = np.sum((np.polyval(p1, x[:i]) - y[:i]) ** 2)
        e2 = np.sum((np.polyval(p2, x[i:]) - y[i:]) ** 2)
        error = e1 + e2

        if error < best2[0]:
            best2 = (error, i, p1, p2)

    if best2[1] is None:
        raise ValueError("Не удалось найти корректную двухсегментную модель.")

    bic_2 = bic(best2[0], n, 5)

    # --- Лучшая трёхсегментная модель ---
    best3 = (np.inf, None, None, None, None, None)

    if n >= 3 * min_segment_points:
        for i in range(min_segment_points, n - 2 * min_segment_points + 1):
            for j in range(i + min_segment_points, n - min_segment_points + 1):
                p1 = np.polyfit(x[:i], y[:i], 1)
                p2 = np.polyfit(x[i:j], y[i:j], 1)
                p3 = np.polyfit(x[j:], y[j:], 1)

                if p1[0] > 0 or p2[0] > 0 or p3[0] > 0:
                    continue

                e1 = np.sum((np.polyval(p1, x[:i]) - y[:i]) ** 2)
                e2 = np.sum((np.polyval(p2, x[i:j]) - y[i:j]) ** 2)
                e3 = np.sum((np.polyval(p3, x[j:]) - y[j:]) ** 2)
                error = e1 + e2 + e3

                if error < best3[0]:
                    best3 = (error, i, j, p1, p2, p3)

    bic_3 = np.inf
    use_three = False

    if best3[1] is not None:
        bic_3 = bic(best3[0], n, 8)
        p2 = best3[4]
        p3 = best3[5]
        relative_slope_change = abs(p3[0] - p2[0]) / max(abs(p2[0]), abs(p3[0]), 1e-12)
        use_three = (bic_2 - bic_3 >= bic_margin and relative_slope_change >= min_relative_slope_change)

    if use_three:
        _, i, j, p_left, p_right, p_tail = best3
        kappa = float(0.5 * (x[i - 1] + x[i]))
        kappa_tail = float(0.5 * (x[j - 1] + x[j]))
        n_segments = 3
    else:
        _, i, p_left, p_right = best2
        p_tail = None
        kappa = float(0.5 * (x[i - 1] + x[i]))
        kappa_tail = None
        n_segments = 2

    if verbose:
        slopes = ", ".join(f"{s:.4g}" for s in (
            (p_left[0], p_right[0], p_tail[0]) if n_segments == 3 else (p_left[0], p_right[0])
        ))
        print(f"Selected kappa = {kappa:.4f} | segments = {n_segments} | slopes = ({slopes})")
        print(f"BIC2 = {bic_2:.4f} | BIC3 = {bic_3:.4f}" if np.isfinite(bic_3) else f"BIC2 = {bic_2:.4f} | BIC3 = unavailable")

    return KappaThresholdFit(
        kappa=kappa,
        n_spikes=n_spikes,
        kappas_positive=kappas_positive,
        n_spikes_positive=n_spikes_positive,
        p_left=p_left,
        p_right=p_right,
        p_tail=p_tail,
        kappa_tail=kappa_tail,
        n_segments=n_segments,
        bic_2=float(bic_2),
        bic_3=float(bic_3),
    )



def _linear_segment_fit(x, y, prefix, start, stop):
    """Быстро оценивает OLS-прямую и SSE на отрезке [start:stop]."""
    n = stop - start
    if n < 2:
        return None, np.inf

    sx = prefix[0][stop] - prefix[0][start]
    sy = prefix[1][stop] - prefix[1][start]
    sxx = prefix[2][stop] - prefix[2][start]
    sxy = prefix[3][stop] - prefix[3][start]
    syy = prefix[4][stop] - prefix[4][start]

    sxx_c = sxx - sx * sx / n
    if sxx_c <= 0:
        return None, np.inf

    xbar = sx / n
    ybar = sy / n
    sxy_c = sxy - sx * sy / n
    syy_c = syy - sy * sy / n

    slope = sxy_c / sxx_c
    intercept = ybar - slope * xbar
    sse = max(float(syy_c - slope * sxy_c), 0.0)

    return np.array([slope, intercept]), sse


def find_kappa_threshold(
    data_3d: np.ndarray,
    background_variance: float,
    kappas: np.ndarray | None = None,
    verbose: bool = False,
    min_segment_points: int = 8,
    bic_margin: float = 10.0,
    min_relative_slope_change: float = 0.25,
) -> KappaThresholdFit:
    """Находит первый устойчивый change point positive-tail distribution."""
    data_3d = np.asarray(data_3d, dtype=float)
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено shape={data_3d.shape}")
    if not np.isfinite(background_variance) or background_variance < 0:
        raise ValueError("background_variance должен быть конечным и >= 0.")
    if kappas is None:
        kappas = np.linspace(2.0, 40.0, 39)

    kappas = np.asarray(kappas, dtype=float)
    if kappas.ndim != 1 or len(kappas) < 7 or np.any(~np.isfinite(kappas)) or np.any(kappas <= 0) or np.any(np.diff(kappas) <= 0):
        raise ValueError("kappas должен быть одномерным, конечным, положительным и строго возрастающим массивом минимум из 7 значений.")

    z_scores = _compute_spike_z_scores(data_3d, background_variance)
    finite = np.isfinite(data_3d)
    n_spikes = np.asarray([np.count_nonzero(finite & (z_scores >= k)) for k in kappas], dtype=int)

    positive = n_spikes > 0
    kappas_positive = kappas[positive]
    n_spikes_positive = n_spikes[positive]
    y = np.log10(n_spikes_positive)

    if len(kappas_positive) < 2 * min_segment_points:
        raise ValueError("Недостаточно точек с ненулевым числом спайков для поиска излома.")

    x = kappas_positive
    n = len(x)
    prefix = tuple(np.concatenate(([0.0], np.cumsum(v))) for v in (x, y, x * x, x * y, y * y))

    def bic(sse, n_points, n_params):
        return n_points * np.log(max(sse / n_points, np.finfo(float).tiny)) + n_params * np.log(n_points)

    # Кэшируем все возможные левый/правый хвосты.
    left_cache = {}
    right_cache = {}
    for i in range(min_segment_points, n - min_segment_points + 1):
        left_cache[i] = _linear_segment_fit(x, y, prefix, 0, i)
        right_cache[i] = _linear_segment_fit(x, y, prefix, i, n)

    # --- Лучшая двухсегментная модель ---
    best2 = (np.inf, None, None, None)
    for i in range(min_segment_points, n - min_segment_points + 1):
        p_left, e_left = left_cache[i]
        p_right, e_right = right_cache[i]
        if p_left is None or p_right is None or p_left[0] > 0 or p_right[0] > 0:
            continue

        error = e_left + e_right
        if error < best2[0]:
            best2 = (error, i, p_left, p_right)

    if best2[1] is None:
        raise ValueError("Не удалось найти корректную двухсегментную модель.")

    bic_2 = bic(best2[0], n, 5)

    # --- Лучшая трёхсегментная модель ---
    best3 = (np.inf, None, None, None, None, None)

    if n >= 3 * min_segment_points:
        for i in range(min_segment_points, n - 2 * min_segment_points + 1):
            p_left, e_left = left_cache[i]
            if p_left is None or p_left[0] > 0:
                continue

            for j in range(i + min_segment_points, n - min_segment_points + 1):
                p_middle, e_middle = _linear_segment_fit(x, y, prefix, i, j)
                if p_middle is None or p_middle[0] > 0:
                    continue

                p_tail, e_tail = right_cache[j]
                if p_tail is None or p_tail[0] > 0:
                    continue

                error = e_left + e_middle + e_tail
                if error < best3[0]:
                    best3 = (error, i, j, p_left, p_middle, p_tail)

    bic_3 = np.inf
    use_three = False

    if best3[1] is not None:
        bic_3 = bic(best3[0], n, 8)
        p_middle, p_tail = best3[4], best3[5]
        relative_slope_change = abs(p_tail[0] - p_middle[0]) / max(abs(p_middle[0]), abs(p_tail[0]), 1e-12)
        use_three = bic_2 - bic_3 >= bic_margin and relative_slope_change >= min_relative_slope_change

    if use_three:
        _, i, j, p_left, p_right, p_tail = best3
        kappa = float(0.5 * (x[i - 1] + x[i]))
        kappa_tail = float(0.5 * (x[j - 1] + x[j]))
        n_segments = 3
    else:
        _, i, p_left, p_right = best2
        p_tail = None
        kappa = float(0.5 * (x[i - 1] + x[i]))
        kappa_tail = None
        n_segments = 2

    if verbose:
        slopes = (p_left[0], p_right[0], p_tail[0]) if n_segments == 3 else (p_left[0], p_right[0])
        print(f"Selected kappa = {kappa:.4f} | segments = {n_segments} | slopes = ({', '.join(f'{s:.4g}' for s in slopes)})")
        print(f"BIC2 = {bic_2:.4f} | BIC3 = {bic_3:.4f}" if np.isfinite(bic_3) else f"BIC2 = {bic_2:.4f} | BIC3 = unavailable")

    return KappaThresholdFit(
        kappa=kappa,
        n_spikes=n_spikes,
        kappas_positive=kappas_positive,
        n_spikes_positive=n_spikes_positive,
        p_left=p_left,
        p_right=p_right,
        p_tail=p_tail,
        kappa_tail=kappa_tail,
        n_segments=n_segments,
        bic_2=float(bic_2),
        bic_3=float(bic_3),
    )




def refine_spike_mask(
    data_3d: np.ndarray,
    kappas: np.ndarray | None = None,
    max_iter: int = 5,
    tol: float = 1e-3,
    verbose: bool = False,
    collect_kappa_fits: bool = False,
) -> SpikeMaskResult:
    """Итеративно уточняет оценку фона и kappa до сходимости."""
    raw = np.asarray(data_3d, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено ndim={raw.ndim}")
    if max_iter < 1:
        raise ValueError("max_iter должен быть >= 1.")
    if tol <= 0:
        raise ValueError("tol должен быть > 0.")

    background = estimate_background_stats(raw)
    history = []
    kappa_fits = [] if collect_kappa_fits else None
    prev_kappa = None
    converged = False
    spike_mask = np.zeros(raw.shape, dtype=bool)
    clean_data = raw.copy()
    kappa = np.nan

    iterator = tqdm(range(max_iter), desc="Spike refinement", unit=" iter") if verbose else range(max_iter)
    for iteration in iterator:
        kappa_fit = find_kappa_threshold(raw, background_variance=background.sigma**2, kappas=kappas, verbose=False)
        kappa = kappa_fit.kappa
        if kappa_fits is not None:
            kappa_fits.append(kappa_fit)

        spike_mask = build_spike_mask(raw, background_variance=background.sigma**2, kappa=kappa, verbose=False)
        clean_data = interpolate_masked_points(raw, spike_mask)
        new_background = estimate_background_stats(clean_data)
        
        if verbose:
            tqdm.write(f"iter {iteration + 1}/{max_iter} | σ={background.sigma:.4g} | κ={kappa:.4f} | spikes={np.count_nonzero(spike_mask)}")
        sigma_rel_change = abs(new_background.sigma - background.sigma) / max(abs(background.sigma), 1e-12)
        kappa_rel_change = np.inf if prev_kappa is None else abs(kappa - prev_kappa) / max(abs(prev_kappa), 1e-12)

        history.append({
            "iteration": iteration,
            "sigma": background.sigma,
            "kappa": kappa,
            "n_spikes": int(np.count_nonzero(spike_mask)),
            "next_sigma": new_background.sigma,
            "sigma_rel_change": sigma_rel_change,
            "kappa_rel_change": kappa_rel_change,
        })
        background = new_background
        if prev_kappa is not None and sigma_rel_change < tol and kappa_rel_change < tol:
            converged = True
            break

        prev_kappa = kappa


    return SpikeMaskResult(
        spike_mask=spike_mask,
        clean_data=clean_data,
        background=background,
        kappa=float(kappa),
        history=history,
        converged=converged,
        iterations=len(history),
        kappa_fits=kappa_fits,
    )



def plot_kappa_threshold(fit: KappaThresholdFit, ax: Axes | None = None, scan_id: int | str | None = None) -> Axes:
    """Рисует сегментированный fit и выбранный первый излом."""
    title_prefix = f"Scan {scan_id}: " if scan_id is not None else ""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fit.kappas_positive, fit.n_spikes_positive, "ko-", markersize=3, label="Экспериментальные данные")

    xmin, xmax = fit.kappas_positive.min(), fit.kappas_positive.max()
    slope_labels = [f"{s:.3g}" for s in fit.slopes]
    ext = 2.0      # длина нахлеста (в сигмах) - экстраполяция линий за точки излома

    x1 = np.linspace(xmin, min(xmax, fit.kappa + ext), 80)
    ax.plot(x1, 10 ** np.polyval(fit.p_left, x1), "b--", linewidth=2, label=f"Сегмент 1, slope={slope_labels[0]}")

    if fit.n_segments == 3 and fit.kappa_tail is not None and fit.p_tail is not None:
        x2 = np.linspace(max(xmin, fit.kappa - ext), min(xmax, fit.kappa_tail + ext), 80)
        x3 = np.linspace(max(xmin, fit.kappa_tail - ext), xmax, 80)
        ax.plot(x2, 10 ** np.polyval(fit.p_right, x2), "r--", linewidth=2, label=f"Сегмент 2, slope={slope_labels[1]}")
        ax.plot(x3, 10 ** np.polyval(fit.p_tail, x3), "--", linewidth=2, label=f"Правый хвост, slope={slope_labels[2]}")
        ax.axvline(fit.kappa_tail, color="gray", linestyle="--", linewidth=1.5, label=f"2-й излом = {fit.kappa_tail:.2f}")
    else:
        x2 = np.linspace(max(xmin, fit.kappa - ext), xmax, 80)
        ax.plot(x2, 10 ** np.polyval(fit.p_right, x2), "r--", linewidth=2, label=f"Сегмент 2, slope={slope_labels[1]}")

    ax.axvline(fit.kappa, color="green", linestyle=":", linewidth=2, label=f"Выбранная kappa = {fit.kappa:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Порог kappa (сигмы)")
    ax.set_ylabel("Количество превышений N(z ≥ kappa)")
    ax.set_title(f"{title_prefix}Segmented tail fit: {fit.n_segments} сегм.")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax