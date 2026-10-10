from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from roc_mcs.fitting.registry import ModelSpec

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticScan:
    frame: FloatArray
    theta: FloatArray
    peak_catalog: pd.DataFrame
    truth: dict[tuple[int, int], dict]


def make_synthetic_scan(
    spec: ModelSpec,
    theta: ArrayLike,
    Ny: int = 8,
    Nx: int = 8,
    max_components: int = 3,
    noise_sigma: float = 3.0,
    seed: int = 42,
) -> SyntheticScan:
    """
    Генерирует синтетический 2D-скан спектров с добавлением шума и эталонной разметкой.

    Parameters
    ----------
    spec : object
        Объект модели пика, содержащий метод `func(theta, **params)`.
    theta : array_like
        1D-массив или последовательность значений сетки углов theta.
    Ny : int, default 8
        Количество пикселей по пространственной оси Y.
    Nx : int, default 8
        Количество пикселей по пространственной оси X.
    max_components : int, default 3
        Максимальное случайное число пиковых компонент в одном пикселе.
    noise_sigma : float, default 3.0
        Стандартное отклонение аддитивного гауссова шума.
    seed : int, default 42
        Зерно генератора случайных чисел NumPy для воспроизводимости.

    Returns
    -------
    SyntheticScan
        Синтетический скан с полями:
        frame : np.ndarray
            Зашумленный 3D-скан формы (Ntheta, Ny, Nx).
        theta : np.ndarray
            1D-массив углов theta формы (Ntheta,).
        peak_catalog : pd.DataFrame
            Каталог сгенерированных пиков и их параметрами.
        truth : dict
            Ground truth по каждому пикселю: число пиков, фон,
            параметры компонент и чистый спектр.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, float)
    ntheta = len(theta)

    frame_yxt = np.zeros((Ny, Nx, ntheta), float)
    peak_rows, truth = [], {}
    
    global_peak_id = 0        # счетчик

    for y in range(Ny):
        for x in range(Nx):
            n = rng.integers(0, max_components + 1)
            bg = float(rng.uniform(2, 8))
            clean = np.full(ntheta, bg, float)
            components = []
            if n:
                idx = np.sort(rng.choice(np.arange(3, ntheta - 3), n, replace=False))
                for local_k, theta_index in enumerate(idx):     # используем local_k (или component_id) для локального порядка
                    params = {
                        "S": float(rng.uniform(800, 1800)),
                        "theta0": float(theta[theta_index]),
                        "H": float(rng.uniform(6, 12)),
                        "eta": float(rng.uniform(0.2, 0.8)),
                    }
                    component = np.asarray(spec.func(theta, **params), float)
                    clean += component

                    peak_rows.append({
                        "peak_id": global_peak_id,       # <--- Глобальный уникальный ID
                        "x": x,
                        "y": y,
                        "theta_index": int(theta_index),
                        "theta": params["theta0"],
                        "prominence": float(component.max()),
                        "width_samples": params["H"] / np.median(np.diff(theta)),
                        "width_theta": params["H"],
                        "left_ip": np.nan,
                        "right_ip": np.nan,
                        "height": float(clean[theta_index]),
                        "component_id": local_k,         # <--- Локальный номер пика в пикселе
                        "branch_id": np.nan,             # (пока ветвей нет, ставим NaN)
                    })
                    components.append({                  # В truth тоже сохраняем оба ID для удобства проверок
                        "peak_id": global_peak_id, 
                        "component_id": local_k, 
                        **params
                    })
                    global_peak_id += 1                  # Не забываем увеличивать счетчик!

            frame_yxt[y, x] = clean + rng.normal(0, noise_sigma, ntheta)
            truth[(x, y)] = {
                "n_peaks": n,
                "background": bg,
                "components": components,
                "clean_curve": clean,
            }

    frame = np.moveaxis(frame_yxt, -1, 0)  # (Ntheta, Ny, Nx)
    
    # 4. Обновляем список колонок (добавлен component_id)
    peak_catalog = pd.DataFrame(peak_rows, columns=[
        "peak_id", "x", "y", "theta_index", "theta",
        "prominence", "width_samples", "width_theta",
        "left_ip", "right_ip", "height", "branch_id", "component_id"
    ])
    return SyntheticScan(
        frame=frame,
        theta=theta,
        peak_catalog=peak_catalog,
        truth=truth,
    )