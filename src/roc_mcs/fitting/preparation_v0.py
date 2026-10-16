from dataclasses import dataclass
import numpy as np

from roc_mcs.peaks.catalog import enrich_peak_catalog

""" Подготовка и пакетирование задач fitting.

Пайплайн
--------
peak_catalog + data_yxt + ModelSpec
    │
    ▼
enrich_peak_catalog()
    │
    ▼
prepare_fit_data()
    │   ├─ initialize_component()
    │   ├─ spec.make_initial_guess()
    │   └─ spec.make_bounds()
    │
    ▼
PreparedFitData
    │
    ▼
iter_fit_batches()
    │   ├─ нарезка по K и batch size
    │   ├─ извлечение obs
    │   └─ построение valid
    │
    ▼
FitBatch
    │
    ▼
pad_batch_arrays()
    │
    ▼
JAX solver

Принцип
-------
Preparation выполняется один раз и не зависит от batch size B.
`iter_fit_batches()` только нарезает уже подготовленные данные.
`pad_batch_arrays()` подготавливает последний batch к фиксированной
форме, необходимой JAX/JIT.
"""

""" Датаклассы
PeakFitSeed — маленькая сущность входа.

PreparedFitGroup — данные всех пикселей одного K.

PreparedFitData — вся подготовленная задача.

FitBatch — уже конкретный кусок задачи, уходящий в optimizer.
"""


@dataclass(frozen=True, slots=True)
class PeakFitSeed:
    """Detected peak, используемый для инициализации одной компоненты."""
    peak_id: int
    theta_index: int
    theta_detected: float
    prominence: float
    height: float


@dataclass(frozen=True, slots=True)
class PreparedFitGroup:
    """Заранее подготовленные данные всех пикселей с одним числом компонент K."""
    K: int
    pixel_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    peak_id: np.ndarray
    theta_index: np.ndarray
    theta_detected: np.ndarray
    prominence: np.ndarray
    height: np.ndarray
    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    fixed: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class PreparedFitData:
    """Полностью подготовленная задача fitting до разбиения на batch'и."""
    groups: dict[int, PreparedFitGroup]
    theta_valid: np.ndarray
    active_names: tuple[str, ...]
    param_names: tuple[str, ...]
    add_background: bool

@dataclass(frozen=True)
class FitBatch:
    """Один batch пикселей, готовый к передаче в optimizer."""
    batch_id: int
    K: int
    pixel_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    obs: np.ndarray
    valid: np.ndarray
    peak_id: np.ndarray
    theta_index: np.ndarray
    theta_detected: np.ndarray
    prominence: np.ndarray
    height: np.ndarray
    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    fixed: dict[str, np.ndarray]
    active_names: tuple[str, ...]
    param_names: tuple[str, ...]
    add_background: bool




def initialize_component(
    spec, theta, intensity, seed, left_neighbor=None, right_neighbor=None,
    center_window=40.0, theta_span=None, theta_step=None,
):
    """
    Вычисляет начальные приближения и границы параметров для пиковой компоненты.

    Определяет адаптивное локальное окно вокруг `seed.theta_detected` с учётом
    расстояния до соседних пиков, извлекает окрестность спектра и формирует
    словари начальных оценок (guess) и ограничений (bounds) для фиттинга.

    PeakFitSeed + spectrum + ModelSpec
     ↓
    guess, bounds
     ↓
    theta0 override
     ↓
    VALIDATE(guess ∈ bounds)   < - одна обязательная проверка контракта
     ↓
    return

    Parameters
    ----------
    spec : object
        Спецификация модели пика (содержит `param_names` и функции оценок).
    theta : np.ndarray
        1D-массив значений углов theta.
    intensity : np.ndarray
        1D-массив интенсивностей спектра.
    seed : object
        Затравочный объект пика, содержащий атрибуты `theta_detected` и `peak_id`.
    left_neighbor : float, optional
        Позиция theta ближайшего левого соседа.
    right_neighbor : float, optional
        Позиция theta ближайшего правого соседа.

    Returns
    -------
    guess : dict
        Словарь начальных значений параметров компоненты.
    bounds : dict
        Словарь кортежей допустимых диапазонов параметров ``(min, max)``.

    Raises
    ------
    ValueError
        Если в вычисленном локальном окне вокруг пика менее 3 точек.
    """
    distances = []
    if left_neighbor is not None:
        distances.append(abs(seed.theta_detected - left_neighbor))
    if right_neighbor is not None:
        distances.append(abs(right_neighbor - seed.theta_detected))
    if theta_span is None:
        theta_span = float(theta.max() - theta.min())
    if theta_step is None:
        theta_step = float(np.median(np.diff(theta)))

    half_window = (
        0.45 * min(distances)
        if distances else 0.05 * theta_span
    )
    half_window = max(half_window, 4 * theta_step)
    mask = (
        (theta >= seed.theta_detected - half_window)
        & (theta <= seed.theta_detected + half_window)
        & np.isfinite(intensity)
    )
    theta_local = theta[mask]
    intensity_local = intensity[mask]
    if len(theta_local) < 3:
        raise ValueError(f"Too few points around peak {seed.peak_id}")

    guess = spec.make_initial_guess(theta_local, intensity_local)
    # if "theta0" in spec.param_names:
    #     guess["theta0"] = seed.theta_detected

    # bounds = spec.make_bounds(guess, dtheta=theta_step)
    # if "theta0" in spec.param_names:
    #     bounds["theta0"] = (
    #         seed.theta_detected - center_window,
    #         seed.theta_detected + center_window,
    #     )
    # return guess, bounds
    if "theta0" in spec.param_names:
        guess["theta0"]=seed.theta_detected
    bounds=spec.make_bounds(guess,dtheta=theta_step)
    if "theta0" in spec.param_names:
        lo0=max(float(theta[0]),seed.theta_detected-center_window)
        hi0=min(float(theta[-1]),seed.theta_detected+center_window)
        if lo0>=hi0: raise ValueError(f"Некорректное theta0 window для peak {seed.peak_id}")
        bounds["theta0"]=(lo0,hi0)
    return guess,bounds



def prepare_fit_data(
    data_yxt, 
    theta, 
    peak_catalog, 
    spec,
    valid_mask=None,            # <--- 1. ДОБАВИЛИ АРГУМЕНТ (по умолчанию None для совместимости)
    allow_center_shift=True, 
    center_window=40.0, 
    add_background=True,
):
    """
    Один раз вычисляет seeds, x0 и bounds для всех pixel.
    Подготовка данных:
        enrich_peak_catalog
            → unique pixel groups
            → определить K
            → создать массивы
            → initialize_component × K
            → x0/lb/ub
            → fixed
            → metadata
    """
    Ny, Nx, M = data_yxt.shape
    pc = enrich_peak_catalog(peak_catalog, Nx)

    active_names = tuple(
        name for name in spec.param_names
        if allow_center_shift or name != "theta0"
    )
    A = len(active_names)

    theta = np.asarray(theta, dtype=float)
    theta_valid = np.isfinite(theta)
    theta_span = float(theta.max() - theta.min())
    theta_step = float(np.median(np.diff(theta)))

    pixel_id_col = pc["pixel_id"].to_numpy(dtype=np.int64, copy=False)
    x_col = pc["x"].to_numpy(dtype=np.int64, copy=False)
    y_col = pc["y"].to_numpy(dtype=np.int64, copy=False)
    peak_id_col = pc["peak_id"].to_numpy(dtype=np.int64, copy=False)
    theta_index_col = pc["theta_index"].to_numpy(dtype=np.int64, copy=False)
    theta_col = pc["theta"].to_numpy(dtype=float, copy=False)
    prominence_col = pc["prominence"].to_numpy(dtype=float, copy=False)
    height_col = pc["height"].to_numpy(dtype=float, copy=False)

    pixel_ids, pixel_starts, pixel_counts = np.unique(
        pixel_id_col, return_index=True, return_counts=True
    )
    pixel_K = pixel_counts.astype(np.int16, copy=False)

    groups = {}
    for K in sorted(np.unique(pixel_K)):
        K = int(K)
        pixel_indices = np.flatnonzero(pixel_K == K)
        N = len(pixel_indices)

        pixel_id = np.empty(N, dtype=np.int64)
        x = np.empty(N, dtype=np.int64)
        y = np.empty(N, dtype=np.int64)
        peak_id = np.empty((N, K), dtype=np.int64)
        theta_index = np.empty((N, K), dtype=np.int64)
        theta_detected = np.empty((N, K), dtype=np.float64)
        prominence = np.empty((N, K), dtype=np.float64)
        height = np.empty((N, K), dtype=np.float64)

        Q = K * A + int(add_background)
        x0 = np.empty((N, Q), dtype=np.float64)
        lb = np.empty((N, Q), dtype=np.float64)
        ub = np.empty((N, Q), dtype=np.float64)

        fixed = {
            name: np.empty((N, K), dtype=np.float64)
            for name in spec.param_names
            if name not in active_names
        }

        for b, pidx in enumerate(pixel_indices):
            base = int(pixel_starts[pidx])
            end = base + K
            rows = slice(base, end)

            pid = int(pixel_ids[pidx])
            xx = int(x_col[base])
            yy = int(y_col[base])

            pixel_id[b] = pid
            x[b] = xx
            y[b] = yy

            obs_b = np.asarray(data_yxt[yy, xx], dtype=np.float64)
            valid_b = theta_valid & np.isfinite(obs_b)                # Базовая проверка: тета валидна и в данных нет NaN
            if valid_mask is not None:                                # Если пользователь передал маску, просто "докручиваем" ее сверху
                valid_b &= np.asarray(valid_mask[yy, xx], dtype=bool) # Явное приведение защитит от ошибки типов in-place
            obs_init = obs_b.copy()
            obs_init[~valid_b] = np.nan

            peak_id[b] = peak_id_col[rows]
            theta_index[b] = theta_index_col[rows]
            theta_detected[b] = theta_col[rows]
            prominence[b] = prominence_col[rows]
            height[b] = height_col[rows]

            centers = theta_detected[b]

            for k in range(K):
                seed = PeakFitSeed(
                    peak_id=int(peak_id[b, k]),
                    theta_index=int(theta_index[b, k]),
                    theta_detected=float(theta_detected[b, k]),
                    prominence=float(prominence[b, k]),
                    height=float(height[b, k]),
                )
                guess, bounds = initialize_component(
                    spec,
                    theta,
                    obs_init,
                    seed,
                    left_neighbor=centers[k - 1] if k else None,
                    right_neighbor=centers[k + 1] if k + 1 < K else None,
                    center_window=center_window,
                    theta_span=theta_span,
                    theta_step=theta_step,
                )
                for j, name in enumerate(active_names):
                    qj = k * A + j
                    x0[b, qj] = guess[name]
                    lo, hi = bounds.get(name, (-np.inf, np.inf))
                    lb[b, qj] = -np.inf if lo is None else lo
                    ub[b, qj] = np.inf if hi is None else hi
                for name in fixed:
                    fixed[name][b, k] = guess[name]

            if add_background:
                if not np.any(valid_b):
                    raise ValueError(f"Для pixel_id={pid} не осталось валидных точек.")
                bg0 = float(np.percentile(obs_b[valid_b], 10))
                x0[b, -1] = max(bg0, 0.0)
                lb[b, -1] = 0.0
                ub[b, -1] = np.inf

        groups[K] = PreparedFitGroup(
            K=K,
            pixel_id=pixel_id,
            x=x,
            y=y,
            peak_id=peak_id,
            theta_index=theta_index,
            theta_detected=theta_detected,
            prominence=prominence,
            height=height,
            x0=x0,
            lb=lb,
            ub=ub,
            fixed=fixed,
        )
    return PreparedFitData(
        groups=groups,
        theta_valid=theta_valid,
        active_names=active_names,
        param_names=tuple(spec.param_names),
        add_background=bool(add_background),
    )


def iter_fit_batches(data_yxt, prepared, valid_mask=None, max_batch_pixels=1024):
    """
    Возвращает batch'и из заранее подготовленных fit-данных.
    Выполняет:
        PreparedFitData
            ↓
           slice
            ↓
           obs
           valid   (с учетом theta_valid, isfinite и valid_mask)
            ↓
           FitBatch
    """
    for K in sorted(prepared.groups):
        group = prepared.groups[K]
        N = len(group.pixel_id)
        for start in range(0, N, max_batch_pixels):
            end = min(start + max_batch_pixels, N)
            sl = slice(start, end)
            x = group.x[sl]
            y = group.y[sl]
            obs = np.asarray(data_yxt[y, x], dtype=np.float64)                 # извлекаем срезы наблюдений размера (B, M)
            valid = prepared.theta_valid[None, :] & np.isfinite(obs)           # базовая маска: корректность углов theta и отсутствие NaN/Inf в скане
            if valid_mask is not None:                                         # интеграция внешней маски фильтрации спайков (valid_mask)
                mask_slice = np.asarray(valid_mask[y, x], dtype=bool)          # векторизованная выборка маски по батчу пикселей (y, x)
                # Защита от расхождения размерностей:
                # - Если valid_mask 3D (Ny, Nx, M): mask_slice имеет форму (B, M)
                # - Если valid_mask 2D (Ny, Nx): mask_slice имеет форму (B,), расширяем до (B, 1)
                if mask_slice.ndim == 1:
                    mask_slice = mask_slice[:, None]
                valid = valid & mask_slice                                      # поэлементное логическое И (учитываются ТОЛЬКО физически чистые точки)
            fixed = {name: values[sl] for name, values in group.fixed.items()}
            yield FitBatch(
                batch_id=start // max_batch_pixels,
                K=K,
                pixel_id=group.pixel_id[sl],
                x=x,
                y=y,
                obs=obs,
                valid=valid,
                peak_id=group.peak_id[sl],
                theta_index=group.theta_index[sl],
                theta_detected=group.theta_detected[sl],
                prominence=group.prominence[sl],
                height=group.height[sl],
                x0=group.x0[sl],
                lb=group.lb[sl],
                ub=group.ub[sl],
                fixed=fixed,
                active_names=prepared.active_names,
                param_names=prepared.param_names,
                add_background=prepared.add_background,
            )



def pad_batch_arrays(batch, target_B=128):
    """
    Дополняет последний batch до фиксированного B=128.
    Это последний адаптер:

    FitBatch variable B
            ↓
    fixed-shape arrays
            ↓
           JAX
    """
    n, M = batch.obs.shape
    if n > target_B:
        raise ValueError(f"Batch size {n} > target_B={target_B}")
    if n == target_B:
        return (
            np.asarray(batch.obs, dtype=float),
            np.asarray(batch.valid, dtype=bool),
            np.asarray(batch.x0, dtype=float),
            np.asarray(batch.lb, dtype=float),
            np.asarray(batch.ub, dtype=float),
            n,
        )

    pad = target_B - n
    x0 = np.concatenate([batch.x0, np.repeat(batch.x0[:1], pad, axis=0)], axis=0)
    lb = np.concatenate([batch.lb, np.repeat(batch.lb[:1], pad, axis=0)], axis=0)
    ub = np.concatenate([batch.ub, np.repeat(batch.ub[:1], pad, axis=0)], axis=0)

    obs = np.concatenate([
        np.asarray(batch.obs, dtype=float),
        np.zeros((pad, M), dtype=float),
    ], axis=0)

    valid = np.concatenate([
        np.asarray(batch.valid, dtype=bool),
        np.zeros((pad, M), dtype=bool),
    ], axis=0)

    return obs, valid, x0, lb, ub, n
