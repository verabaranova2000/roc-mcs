import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter, peak_prominences
from tqdm.auto import tqdm
from dataclasses import dataclass

from roc_mcs.processing.moments import compute_profile_moments


"""
Применение калибровки
-----
calibrate_roc_map()         # применяет найденную калибровку к ROC-карте


Результат калибровки
-----
CalibrationSearchResult


Поиск калибровочных параметров
-----
guess_scan_ranges() /                # предлагает диапазоны поиска A и B: начальный (coarse) диапазон
guess_refinement_scan_ranges()       # предлагает диапазоны поиска A и B: локальное окно вокруг найденного минимума (A*, B*)
        │
        ▼
search_A_by_inner_B_minimum()       # перебирает A
        │
        ├── score_B_scan()          # вычисляет score(B) для фиксированного A
        └── find_inner_minimum()    # ищет внутренний минимум score(B) (внутри "_М_")
        │
        ▼
check_search_result()               # проверяет, что минимум не упирается в границы поиска

Использование
-------------
run_calibration() объединяет эти функции в полный цикл:
coarse → проверка → (опц.) refinement.

Выход:
-----
DataFrame с лучшими (A, B_best, score_best) и score_map
"""


# =============================
# Применение калибровки
# =============================

def calibrate_roc_map(roc_map, amplitude, reference_amplitude, reference_angle):
    """
    Калибровка угловой оси ROC-карты.

    Parameters
    ----------
    reference_angle : float
        Калибровочный коэффициент (arcsec), полученный при reference_amplitude.

    reference_amplitude : float
        Амплитуда пьезоактуатора (mVpp), для которой была выполнена калибровка.

    amplitude : float
        Амплитуда текущего эксперимента (mVpp).
    """
    scale = reference_angle * amplitude / reference_amplitude

    roc_map = roc_map.copy()
    roc_map["theta_axis"] = roc_map["s_axis"] * scale
    roc_map["calibration"] = {
        "reference_angle": reference_angle,
        "reference_amplitude": reference_amplitude,
        "amplitude": amplitude,
        "scale": scale,
    }
    return roc_map    



# =============================
# Результат калибровки
# =============================

@dataclass
class CalibrationSearchResult:
    """Результат одного этапа поиска калибровки."""
    A_scan: np.ndarray          # сетка A
    B_scan: np.ndarray          # сетка B
    estimate: dict | None       # начальная оценка диапазона (если есть) (Убрать?)
    df_res: pd.DataFrame        # минимум по B для каждого A
    score_map: np.ndarray       # карта S(A, B)
    
    @property
    def best_row(self) -> pd.Series:
        """Строка df_res с глобальным минимумом score."""
        return self.df_res.loc[self.df_res["score_best"].idxmin()]

    @property
    def A_best(self) -> float:
        """Оптимальный коэффициент масштаба A*."""
        return float(self.best_row["A"])

    @property
    def B_best(self) -> float:
        """Оптимальный сдвиг B*."""
        return float(self.best_row["B_best"])

    @property
    def boundary_report(self) -> dict:
        """Проверка, находится ли минимум внутри диапазона поиска."""
        return check_search_result(
            self.A_scan,
            self.B_scan,
            self.A_best,
            self.B_best,
            verbose=False,
        )

# =============================
# Поиск калибровочных параметров
# =============================

def normalize01(y):
    y = np.asarray(y, dtype=float)
    y = y - np.percentile(y, 5)   # устойчивее, чем min
    y = np.clip(y, 0, None)
    ymax = np.max(y)
    return y / ymax if ymax > 0 else y



def normalize01_v1(y, n_top=5):
    y = np.asarray(y, dtype=float)
    
    # 1. Вычитаем фон (здесь 5-й перцентиль работает отлично)
    y = y - np.percentile(y, 5) 
    y = np.clip(y, 0, None)
    
    if len(y) == 0:
        return y
        
    # 2. Находим робастный максимум (защита от 1-2 шумовых выбросов)
    # Берем n_top самых больших значений (или меньше, если массив короткий)
    k = min(n_top, len(y))
    top_values = np.sort(y)[-k:]   
    # Медиана топ-значений отсечет случайные одиночные спайки
    ymax = np.median(top_values)

    # 3. Нормируем
    if ymax > 0:
        y_norm = y / ymax
    else:
        y_norm = y  
    return y_norm


def normalize01(y, p_low=5, p_high=99.0):
    """
    Робастная нормировка 1D-сигнала в [0, 1].

    Идея:
    - снимаем фон по нижнему процентилю;
    - находим главный пик без сглаживания всего сигнала;
    - масштаб оцениваем по области пика, а не по абсолютному max всего массива.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y

    z = y - np.percentile(y, p_low)
    z = np.clip(z, 0, None)

    if not np.any(z):
        return z

    peaks, _ = find_peaks(z)
    if len(peaks) == 0:
        scale = np.percentile(z, p_high)
        return z / scale if scale > 0 else z

    main_peak = peaks[np.argmax(z[peaks])]
    prom, left_bases, right_bases = peak_prominences(z, [main_peak])

    lo = int(left_bases[0])
    hi = int(right_bases[0])

    if hi <= lo:
        scale = np.percentile(z, p_high)
    else:
        region = z[lo:hi + 1]
        scale = np.percentile(region, p_high)

    return z / scale if scale > 0 else z



def guess_scan_ranges(
    s_mca, y_mca,
    theta_motor, y_motor,
    a_factor=6.0,      # насколько широко брать A вокруг A0
    b_pad_factor=1.5,  # дополнительный запас по B
    nA=201,
    nB=2001,
    verbose=False,
):
    """Построить стартовые диапазоны поиска по грубой оценке параметров."""
    SHOW_TABLE = False
    s_mca = np.asarray(s_mca, dtype=float)
    theta_motor = np.asarray(theta_motor, dtype=float)
    y_mca = normalize01(y_mca)
    y_motor = normalize01(y_motor)

    # центры тяжести — только как seed, не как окончательные границы
    s_c = np.sum(s_mca * y_mca) / np.sum(y_mca)
    th_c = np.sum(theta_motor * y_motor) / np.sum(y_motor)

    # # грубая оценка ширины
    s_span = np.percentile(s_mca, 95) - np.percentile(s_mca, 5)
    th_span = np.percentile(theta_motor, 95) - np.percentile(theta_motor, 5)

    A0 = th_span / max(s_span, 1e-12)
    B0 = th_c - A0 * s_c

    # широкий диапазон A
    A_min = max(1e-6, A0 / a_factor)
    A_max = A0 * a_factor

    # широкий диапазон B:
    # позволяем MCS-кривой шириной ~2*A_max свободно сдвигаться относительно theta_motor
    B_min = theta_motor.min() - A_max - b_pad_factor * th_span
    B_max = theta_motor.max() + A_max + b_pad_factor * th_span

    A_scan = np.linspace(A_min, A_max, nA)
    B_scan = np.linspace(B_min, B_max, nB)

    estimate = {
        "A0": float(A0),
        "B0": float(B0),
        "s_c": float(s_c),
        "th_c": float(th_c),
    }

    if verbose:
        dA = A_scan[1] - A_scan[0]
        dB = B_scan[1] - B_scan[0]
        if SHOW_TABLE:
            print("Параметры поиска")
            print("─" * 68)
            print(
                f"{'Параметр':<20}"
                f"{'Оценка':>8}"
                f"{'Диапазон':>20}"
                f"{'Шаг':>10}"
            )
            print("─" * 68)
            print(
                f"{'A, угл. сек./канал':<20}"
                f"{estimate['A0']:>8.2f}"
                f"{f'{A_scan.min():.2f} … {A_scan.max():.2f}':>20}"
                f"{dA:>10.3f}"
            )
            print(
                f"{'B, угл. сек.':<20}"
                f"{estimate['B0']:>8.2f}"
                f"{f'{B_scan.min():.2f} … {B_scan.max():.2f}':>20}"
                f"{dB:>10.3f}"
            )
            print("─" * 68)
        else:
            print("Начальная оценка параметров")
            print(f"  A₀ (масштаб) : {estimate['A0']:.2f} угл. сек./канал")
            print(f"  B₀ (сдвиг)   : {estimate['B0']:.2f} угл. сек.")
            print()
            
            leftA = f"A ∈ [{A_scan.min():7.2f}, {A_scan.max():7.2f}]   ({len(A_scan):4d} точек)"
            leftB = f"B ∈ [{B_scan.min():7.2f}, {B_scan.max():7.2f}]   ({len(B_scan):4d} точек)"
            
            print(f"{'Диапазоны поиска':<46}Шаг")
            print(f"  {leftA:<44}  ΔA = {dA:.3f} угл. сек./канал")
            print(f"  {leftB:<44}  ΔB = {dB:.3f} угл. сек.")

    return A_scan, B_scan, estimate


def guess_refinement_scan_ranges(
    search_result: CalibrationSearchResult,
    refine_A_frac: float = 0.15,
    refine_B_frac: float = 0.10,
    nA: int = 401,
    nB: int = 1001,
    verbose: bool = False,
):
    """Построить локальные диапазоны поиска вокруг найденного минимума."""
    A_span = float(search_result.A_scan[-1] - search_result.A_scan[0])  # ширина текущего окна поиска: напр. A_scan=[10, 11, 12, ..., 300] -> A_span = 300 - 10 = 290 
    B_span = float(search_result.B_scan[-1] - search_result.B_scan[0])  # сужаем окно вокруг best: [best - 0.15*290, best + 0.15*290]

    A_min = max(1e-6, search_result.A_best - refine_A_frac * A_span)
    A_max = search_result.A_best + refine_A_frac * A_span
    B_min = search_result.B_best - refine_B_frac * B_span
    B_max = search_result.B_best + refine_B_frac * B_span

    A_scan = np.linspace(A_min, A_max, nA)
    B_scan = np.linspace(B_min, B_max, nB)

    estimate = {
        "A_center": float(search_result.A_best),
        "B_center": float(search_result.B_best),
        "A_span": A_span,
        "B_span": B_span,
    }

    if verbose:
        dA = A_scan[1] - A_scan[0]
        dB = B_scan[1] - B_scan[0]
        print("Локальное уточнение вокруг найденного минимума")
        print(f"  A_center = {estimate['A_center']:.3f}")
        print(f"  B_center = {estimate['B_center']:.3f}")
        print(f"  A ∈ [{A_min:.3f}, {A_max:.3f}] ({len(A_scan)} точек), ΔA = {dA:.4f}")
        print(f"  B ∈ [{B_min:.3f}, {B_max:.3f}] ({len(B_scan)} точек), ΔB = {dB:.4f}")

    return A_scan, B_scan, estimate



def score_B_scan(
    A,
    s_mca,
    y_mca,
    theta_motor,
    y_motor,
    B_scan,
    n_grid=500,
    delta=0.03,
    block_size=2048,
):
    """
    Векторизованный расчет score(B) для фиксированного A.
    """
    s_mca = np.asarray(s_mca, dtype=float)
    theta_motor = np.asarray(theta_motor, dtype=float)
    B_scan = np.asarray(B_scan, dtype=float)

    x_base = A * s_mca
    x0_min = x_base.min()
    x0_max = x_base.max()
    th_min = theta_motor.min()
    th_max = theta_motor.max()

    scores = np.full(B_scan.shape, np.inf, dtype=float)

    u = np.linspace(0.0, 1.0, n_grid, dtype=float)
    
    for start in range(0, len(B_scan), block_size):
        stop = min(start + block_size, len(B_scan))
        Bv = B_scan[start:stop]

        # есть ли вообще пересечение?
        valid = (x0_max + Bv > th_min) & (x0_min + Bv < th_max)
        if not np.any(valid):
            continue

        Bvv = Bv[valid]

        xmin = np.maximum(x0_min + Bvv, th_min)
        xmax = np.minimum(x0_max + Bvv, th_max)

        # Тот же равномерный grid, что и в score_AB
        grid = xmin[:, None] + (xmax - xmin)[:, None] * u[None, :]

        # Из-за сдвига MCS-кривой по B:
        # interp1d(x_base + B, y_mca)(grid) == interp1d(x_base, y_mca)(grid - B)
        y1 = np.interp(
            (grid - Bvv[:, None]).ravel(),
            x_base,
            y_mca,
            left=np.nan,
            right=np.nan,
        ).reshape(grid.shape)

        y2 = np.interp(
            grid.ravel(),
            theta_motor,
            y_motor,
            left=np.nan,
            right=np.nan,
        ).reshape(grid.shape)

        mask = np.isfinite(y1) & np.isfinite(y2)
        n_ok = mask.sum(axis=1)

        r = y1 - y2
        a = np.abs(r)
        loss = np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))
        loss[~mask] = 0.0

        sc = loss.sum(axis=1) / np.maximum(n_ok, 1)
        sc[n_ok < 50] = np.inf
        scores[start:stop][valid] = sc
    return scores


def find_inner_minimum_v0_bad(B_scan, scores, edge_frac=0.12, smooth=True):
    """
    Ищет внутренний локальный минимум на score(B),
    игнорируя хвосты и края.
    """
    B_scan = np.asarray(B_scan, dtype=float)
    scores = np.asarray(scores, dtype=float)

    finite = np.isfinite(scores)
    if finite.sum() < 5:
        return None, np.inf, {"reason": "too_few_finite_points"}

    idx = np.flatnonzero(finite)
    i0, i1 = idx[0], idx[-1]

    B_f = B_scan[finite]
    S_f = scores[finite]
    
    # лёгкое сглаживание, чтобы убрать мелкий шум
    if smooth and len(S_f) >= 9:
        win = min(len(S_f) if len(S_f) % 2 == 1 else len(S_f) - 1, 31)
        if win >= 5:
            S_work = savgol_filter(S_f, window_length=win, polyorder=3)
        else:
            S_work = S_f
    else:
        S_work = S_f

    # внутренний интервал (отрезаем края)
    left = i0 + int(edge_frac * (i1 - i0))
    right = i1 - int(edge_frac * (i1 - i0))
    if right <= left:
        left, right = i0, i1
        
    # локальные минимумы ищем как пики у -score
    cand_rel, _ = find_peaks(
        -S_work,
        prominence=np.nanstd(S_work) * 0.05 if np.nanstd(S_work) > 0 else 0.0,
    )

    if len(cand_rel) == 0:
        mask_inner = (B_scan >= B_scan[left]) & (B_scan <= B_scan[right]) & finite
        if mask_inner.sum() == 0:
            return None, np.inf, {"reason": "no_inner_points"}
        j = np.nanargmin(scores[mask_inner])
        B_best = B_scan[mask_inner][j]
        score_best = scores[mask_inner][j]
        return B_best, score_best, {"reason": "fallback_argmin"}

    cand_full = np.flatnonzero(finite)[cand_rel]                # переводим кандидатов обратно в индексы full-array
    cand_inner = [j for j in cand_full if left <= j <= right]   # оставляем только внутренние кандидаты

    if len(cand_inner) == 0:
        mask_inner = (B_scan >= B_scan[left]) & (B_scan <= B_scan[right]) & finite
        if mask_inner.sum() == 0:
            return None, np.inf, {"reason": "no_inner_candidates"}
        j = np.nanargmin(scores[mask_inner])
        B_best = B_scan[mask_inner][j]
        score_best = scores[mask_inner][j]
        return B_best, score_best, {"reason": "fallback_inner_argmin"}
    
    # выбираем лучший внутренний минимум по исходному score
    cand_scores = np.array([scores[j] for j in cand_inner], dtype=float)
    k = np.nanargmin(cand_scores)
    jbest = cand_inner[k]

    info = {
        "reason": "inner_local_minimum",
        "n_candidates": len(cand_inner),
        "candidate_indices": cand_inner,
    }
    return B_scan[jbest], scores[jbest], info



def find_inner_minimum(B_scan, scores, edge_frac=0.12, smooth=True):
    """
    Ищет внутренний локальный минимум (истинную долину) на score(B).
    Строго игнорирует краевые скаты в нуль (вырожденные решения).
    """
    B_scan = np.asarray(B_scan, dtype=float)
    scores = np.asarray(scores, dtype=float)

    finite = np.isfinite(scores)
    if finite.sum() < 5:
        return None, np.inf, {"reason": "too_few_finite_points"}

    idx = np.flatnonzero(finite)
    
    # Работаем только с конечными (finite) значениями, чтобы не тянуть индексы NaN-ов
    B_f = B_scan[finite]
    S_f = scores[finite]
    
    # Лёгкое сглаживание
    if smooth and len(S_f) >= 9:
        win = min(len(S_f) if len(S_f) % 2 == 1 else len(S_f) - 1, 31)
        if win >= 5:
            S_work = savgol_filter(S_f, window_length=win, polyorder=3)
        else:
            S_work = S_f
    else:
        S_work = S_f

    # Определяем внутренние границы (отбрасываем edge_frac от КОЛИЧЕСТВА валидных точек)
    margin = int(len(S_f) * edge_frac)
    left = margin
    right = len(S_f) - 1 - margin
    if right <= left:
        left, right = 0, len(S_f) - 1
        
    # Ищем истинные впадины. 
    s_range = np.nanmax(S_work) - np.nanmin(S_work)  # Prominence (глубину впадины) считаем от полного размаха ошибки, а не от std.
    prom = s_range * 0.05 if s_range > 0 else 0.0    # Настоящая впадина должна быть хотя бы на 5% глубже, чем самый высокий бугор несовпадения.

    cand_rel, _ = find_peaks(-S_work, prominence=prom)

    # Оставляем только те кандидаты, которые лежат внутри разрешенного окна
    valid_cands = [c for c in cand_rel if left <= c <= right]

    if len(valid_cands) == 0:
        # УБРАН FALLBACK НА ARGMIN!
        # Если нет ярко выраженной впадины, значит тут кривые просто разъезжаются (скат на краю).
        # Возвращаем inf, чтобы внешний цикл оптимизации выбросил этот масштаб A
        return None, np.inf, {"reason": "no_true_valley_found"}
    
    # Если нашлось несколько впадин, выбираем ту, где исходный score меньше
    best_idx_in_f = valid_cands[np.nanargmin(S_f[valid_cands])]
    jbest = idx[best_idx_in_f]
    info = {
        "reason": "inner_local_minimum",
        "n_candidates": len(valid_cands),
        "candidate_indices": [idx[c] for c in valid_cands],
    }
    return B_scan[jbest], scores[jbest], info



def search_A_by_inner_B_minimum(
    s_mca, y_mca,
    theta_motor, y_motor,
    A_scan, 
    B_scan,
    edge_frac=0.10,
    n_grid=500,
    delta=0.03,
    block_size=2048,
):
    """
    Parameters
    ----------
    s_mca : array-like
        Нормированная координата ROC-кривой, полученной из MCS.
    
    y_mca : array-like
        Интенсивность ROC-кривой из MCS.
    
    theta : array-like
        Угловая координата моторного скана (угл. сек.).
    
    y_motor : array-like
        Интенсивность моторного скана.
    
    A_scan : array-like
        Сетка масштабного коэффициента A.
    
    B_scan : array-like
        Сетка сдвига B.
    """
    # Если вдруг оси не отсортированы, отсортируем один раз
    if np.any(np.diff(s_mca) < 0):
        order = np.argsort(s_mca)
        s_mca = s_mca[order]
        y_mca = y_mca[order]

    if np.any(np.diff(theta_motor) < 0):
        order = np.argsort(theta_motor)
        theta_motor = theta_motor[order]
        y_motor = y_motor[order]

    y_mca_n = normalize01(y_mca)
    y_motor_n = normalize01(y_motor)
    
    score_map = np.empty((len(A_scan), len(B_scan)), dtype=float)
    rows = []
    for i, A in enumerate(tqdm(A_scan, desc="Searching A")):       
        scores = score_B_scan(
            A=A,
            s_mca=s_mca,
            y_mca=y_mca_n,
            theta_motor=theta_motor,
            y_motor=y_motor_n,
            B_scan=B_scan,
            n_grid=n_grid,
            delta=delta,
            block_size=block_size,
        )
        score_map[i] = scores

        B_best, score_best, info = find_inner_minimum(
            B_scan,
            scores,
            edge_frac=edge_frac,
            smooth=True,
        )

        rows.append({
            "A": A,
            "B_best": B_best,
            "score_best": score_best,
            "reason": info.get("reason", ""),
            "n_finite": np.isfinite(scores).sum(),
        })
    results = pd.DataFrame(rows).reset_index(drop=True)
    return results, score_map



def check_search_result(A_scan, B_scan, 
                        A_best, B_best, 
                        edge_cells=3, edge_frac=0.05,
                        verbose=True):
    """
    Проверка, не слишком ли близко найденный минимум к границам сетки.

    Parameters
    ----------
    A_scan, B_scan : array-like
        Сетки поиска.
    A_best, B_best : float
        Найденная лучшая точка.
    edge_cells : int
        Минимум крайних ячеек, которые считаются "опасной зоной".
    edge_frac : float
        Дополнительный критерий: доля диапазона от края.

    Returns
    -------
    report : dict
        Диагностический словарь.
    """
    A_scan = np.asarray(A_scan, dtype=float)
    B_scan = np.asarray(B_scan, dtype=float)

    iA = int(np.argmin(np.abs(A_scan - A_best)))
    iB = int(np.argmin(np.abs(B_scan - B_best)))

    nA = len(A_scan)
    nB = len(B_scan)

    A_min, A_max = float(A_scan.min()), float(A_scan.max())
    B_min, B_max = float(B_scan.min()), float(B_scan.max())

    dA_left = float(A_best - A_min)
    dA_right = float(A_max - A_best)
    dB_left = float(B_best - B_min)
    dB_right = float(B_max - B_best)

    A_span = A_max - A_min
    B_span = B_max - B_min

    A_too_close = (iA < edge_cells) or (iA >= nA - edge_cells) or (dA_left < edge_frac * A_span) or (dA_right < edge_frac * A_span)
    B_too_close = (iB < edge_cells) or (iB >= nB - edge_cells) or (dB_left < edge_frac * B_span) or (dB_right < edge_frac * B_span)

    if verbose:
        print("Проверка достаточности диапазонов поиска")
        print("─" * 40)
    
        print(f"A : {A_best:.3f}  (узел {iA+1} из {nA})")
        print(f"    диапазон [{A_min:.3f}, {A_max:.3f}]")
        print(f"    до границ: {dA_left:.3f} / {dA_right:.3f}")
        print()
    
        print(f"B : {B_best:.3f}  (узел {iB+1} из {nB})")
        print(f"    диапазон [{B_min:.3f}, {B_max:.3f}]")
        print(f"    до границ: {dB_left:.3f} / {dB_right:.3f}")
        print()
    
        if (not A_too_close) and (not B_too_close):
            print("✓ Диапазоны поиска достаточны.")
        else:
            print("⚠ Минимум расположен слишком близко к границе.")
    
            if A_too_close:
                side = "нижней" if dA_left < dA_right else "верхней"
                print(f"  A: ближе к {side} границе")
    
            if B_too_close:
                side = "нижней" if dB_left < dB_right else "верхней"
                print(f"  B: ближе к {side} границе")
    return {
        "iA": iA,
        "iB": iB,
        "nA": nA,
        "nB": nB,
        "A_best": float(A_best),
        "B_best": float(B_best),
        "A_bounds": (A_min, A_max),
        "B_bounds": (B_min, B_max),
        "dA_left": dA_left,
        "dA_right": dA_right,
        "dB_left": dB_left,
        "dB_right": dB_right,
        "A_too_close_to_edge": A_too_close,
        "B_too_close_to_edge": B_too_close,
        "ok": (not A_too_close) and (not B_too_close),
    }