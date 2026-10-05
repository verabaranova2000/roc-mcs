from dataclasses import dataclass
import numpy as np
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


from roc_mcs.processing.background import BackgroundStats, estimate_background_stats


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



# def interpolate_masked_points(data_3d: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
#     """Заменяет spike-точки линейной интерполяцией между соседними точками по theta."""
#     data_3d = np.asarray(data_3d, dtype=float)
#     spike_mask = np.asarray(spike_mask, dtype=bool)

#     if data_3d.ndim != 3:
#         raise ValueError(f"data_3d должен быть 3D, получено ndim={data_3d.ndim}")
#     if spike_mask.shape != data_3d.shape:
#         raise ValueError(f"spike_mask имеет форму {spike_mask.shape}, ожидалась {data_3d.shape}")

#     cleaned_data = data_3d.copy()
#     y_left = np.roll(data_3d, 1, axis=0)              # соседи по оси theta
#     y_right = np.roll(data_3d, -1, axis=0)
#     y_left[0] = data_3d[0]                            # чиним краевые эффекты (если спайк на самом первом или последнем кадре)
#     y_right[-1] = data_3d[-1]

#     cleaned_data[spike_mask] = (y_left[spike_mask] + y_right[spike_mask]) / 2.0    # интерполяция плохих точек (заменяем средним соседних)   
#     return cleaned_data


def interpolate_masked_points(data_3d: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
    """Заменяет spike-точки интерполяцией по theta."""
    data_3d = np.asarray(data_3d, dtype=float)
    spike_mask = np.asarray(spike_mask, dtype=bool)

    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено ndim={data_3d.ndim}")
    if spike_mask.shape != data_3d.shape:
        raise ValueError(f"spike_mask имеет форму {spike_mask.shape}, ожидалась {data_3d.shape}")
    if data_3d.shape[0] < 2:
        raise ValueError("Для интерполяции требуется минимум два theta-отсчёта.")

    cleaned_data = data_3d.copy()

    left = np.roll(data_3d, 1, axis=0)
    right = np.roll(data_3d, -1, axis=0)

    cleaned_data[1:-1][spike_mask[1:-1]] = (
        left[1:-1][spike_mask[1:-1]] + right[1:-1][spike_mask[1:-1]]
    ) / 2.0
    cleaned_data[0][spike_mask[0]] = data_3d[1][spike_mask[0]]
    cleaned_data[-1][spike_mask[-1]] = data_3d[-2][spike_mask[-1]]

    return cleaned_data


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


def find_kappa_threshold(
    data_3d: np.ndarray,
    background_variance: float,
    kappas: np.ndarray | None = None,
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """
    Находит порог kappa по излому зависимости числа спайков от kappa.
    Использует метод кусочно-линейной регрессии (Piecewise Linear Fit).
    
    spike_mask == True   → спайк
    spike_mask == False  → не спайк

    Возвращает:
        best_kappa : float
        n_spikes : np.ndarray
            Число обнаруженных спайков для каждого значения kappas.
    """
    data_3d = np.asarray(data_3d, dtype=float)
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено ndim={data_3d.ndim}")
    if not np.isfinite(background_variance) or background_variance < 0:
        raise ValueError("background_variance должен быть конечным и >= 0.")
    if kappas is None:
        kappas = np.linspace(2.0, 40.0, 39)
    kappas = np.asarray(kappas, dtype=float)
    if kappas.ndim != 1 or len(kappas) < 7 or np.any(~np.isfinite(kappas)) or np.any(np.diff(kappas) <= 0):
        raise ValueError("kappas должен быть одномерным, конечным и строго возрастающим массивом минимум из 7 значений.")
    if kappas.ndim != 1 or len(kappas) < 7 or np.any(~np.isfinite(kappas))  or np.any(kappas <= 0) or np.any(np.diff(kappas) <= 0):
        raise ValueError("kappas должен быть одномерным, конечным, положительным и строго возрастающим массивом минимум из 7 значений.")

    z_scores = _compute_spike_z_scores(data_3d, background_variance)
    finite = np.isfinite(data_3d)
    n_spikes = []
    for kappa in kappas:
        n_spikes.append(np.count_nonzero(finite & (z_scores >= kappa)))

    n_spikes = np.asarray(n_spikes, dtype=int)

    positive = n_spikes > 0                             # игнорируем точки, где n_spikes == 0, чтобы избежать log(0)
    x = kappas[positive]
    y = np.log10(n_spikes[positive])                    # логарифмический масштаб, так как процессы экспоненциальные
    if len(x) < 7:
        raise ValueError("Недостаточно точек с ненулевым числом спайков для поиска излома.")

    min_error = np.inf
    best_p_left = best_p_right = None

    # --- Поиск оптимальной точки разбиения (требуем минимум 3 точки для линии) ---
    for i in range(3, len(x) - 3):
        p_left = np.polyfit(x[:i], y[:i], 1)       # Линейная регрессия (полином 1 степени) для левого и правого участков
        p_right = np.polyfit(x[i:], y[i:], 1)
        error = np.sum((np.polyval(p_left, x[:i]) - y[:i]) ** 2) + np.sum((np.polyval(p_right, x[i:]) - y[i:]) ** 2)   # сумма квадратов ошибок (Residual Sum of Squares)
        if error < min_error:                      # Сохраняем модель с минимальной ошибкой
            min_error = error
            best_p_left, best_p_right = p_left, p_right

    if best_p_left is None or best_p_right is None or np.isclose(best_p_left[0], best_p_right[0]):
        raise ValueError("Не удалось надёжно определить точку излома.")

    # Точка пересечения двух прямых (точная координата X (kappa))
    # a₁x + b₁ = a₂x + b₂ ⇒ x = (b₂ − b₁) / (a₁ − a₂)
    kappa = (best_p_right[1] - best_p_left[1]) / (best_p_left[0] - best_p_right[0])
    if not np.isfinite(kappa) or not x.min() <= kappa <= x.max():
        raise ValueError(f"Найденный kappa={kappa:.6g} вне диапазона точек с ненулевым числом спайков.")

    # График
    if verbose:
        print(f"Selected kappa = {kappa:.4f}")

        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(x, 10**y, 'ko-', markersize=3, label='Экспериментальные данные')
        
        x_fit_left = np.linspace(min(x), kappa + 2, 50)            # генерация точек для отрисовки найденных прямых
        x_fit_right = np.linspace(kappa - 2, max(x), 50)
        
        ax.plot(x_fit_left, 10**np.polyval(best_p_left, x_fit_left), 'b--', linewidth=2, label='Тренд шума')
        ax.plot(x_fit_right, 10**np.polyval(best_p_right, x_fit_right), 'r--', linewidth=2, label='Тренд спайков')
        ax.axvline(kappa, color='green', linestyle=':', linewidth=2, label=f'Оптимальная kappa = {kappa:.2f}')
        
        ax.set_yscale('log')
        ax.set_xlabel('Порог kappa (сигмы)')
        ax.set_ylabel('Количество спайков')
        ax.set_title('Автоматический расчет точки излома (Piecewise Linear Fit)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()
    return float(kappa), n_spikes


def refine_spike_mask(
    data_3d: np.ndarray,
    kappas: np.ndarray | None = None,
    max_iter: int = 5,
    tol: float = 1e-3,
    verbose: bool = False,
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
    prev_kappa = None
    converged = False
    spike_mask = np.zeros(raw.shape, dtype=bool)
    clean_data = raw.copy()
    kappa = np.nan

    iterator = tqdm(range(max_iter), desc="Spike refinement", unit=" iter") if verbose else range(max_iter)
    for iteration in iterator:
        kappa, _ = find_kappa_threshold(
            raw,
            background_variance=background.sigma**2,
            kappas=kappas,
            verbose=verbose,
        )
        spike_mask = build_spike_mask(
            raw,
            background_variance=background.sigma**2,
            kappa=kappa,
            verbose=False,
        )
        clean_data = interpolate_masked_points(raw, spike_mask)
        new_background = estimate_background_stats(clean_data)
        
        if verbose:
            tqdm.write(f"iter {iteration + 1}/{max_iter} | σ={background.sigma:.4g} | κ={kappa:.4f} | spikes={np.count_nonzero(spike_mask)}")
        sigma_rel_change = abs(new_background.sigma - background.sigma) / max(abs(background.sigma), 1e-12)
        kappa_rel_change = (
            np.inf if prev_kappa is None
            else abs(kappa - prev_kappa) / max(abs(prev_kappa), 1e-12)
        )

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
        # background = new_background

    return SpikeMaskResult(
        spike_mask=spike_mask,
        clean_data=clean_data,
        background=background,
        kappa=float(kappa),
        history=history,
        converged=converged,
        iterations=len(history),
    )



