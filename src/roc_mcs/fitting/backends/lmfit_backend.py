import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from roc_mcs.fitting.registry import ModelSpec
from roc_mcs.fitting.preparation import PreparedFitGroup, prepare_fit_data


# ------------------------
# Модель + параметры 
# ------------------------

def _make_lmfit_fitter(spec: ModelSpec, K: int, add_background: bool = True):
    """Создаёт Model и шаблон Parameters для фиксированного K."""
    K = int(K)
    if K < 1:
        raise ValueError("K must be >= 1")
    model = None
    for k in range(K):
        component = Model(spec.func, prefix=f"peak_{k}_")
        model = component if model is None else model + component
    if add_background:
        background = Model(
            lambda theta, background:
                np.full_like(np.asarray(theta, dtype=float), background),
            prefix="bg_",
        )
        model = model + background
    return model, model.make_params()


def _lmfit_bound(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def _make_lmfit_params(
    template: Parameters, group: PreparedFitGroup, active_names: tuple[str, ...], row: int, 
    add_background: bool = True,
):
    """Создаёт параметры pixel из общей PreparedFitData."""
    params = template.copy()
    A = len(active_names)
    for k in range(group.K):
        for j, name in enumerate(active_names):
            qj = k * A + j
            params[f"peak_{k}_{name}"].set(
                value=float(group.x0[row, qj]),
                min=_lmfit_bound(group.lb[row, qj]),
                max=_lmfit_bound(group.ub[row, qj]),
                vary=True,
            )
        for name, values in group.fixed.items():
            params[f"peak_{k}_{name}"].set(
                value=float(values[row, k]),
                vary=False,
            )
    if add_background:
        params["bg_background"].set(
            value=float(group.x0[row, -1]),
            min=_lmfit_bound(group.lb[row, -1]),
            max=_lmfit_bound(group.ub[row, -1]),
            vary=True,
        )
    return params



# ------------------------
# Диагностики
# ------------------------
def _make_lmfit_peak_records(result, group, spec: ModelSpec, row: int):
    """Формирует fitted peak records из LMFit результата."""
    records = []
    K = int(group.K)
    for k in range(K):
        prefix = f"peak_{k}_"
        row_out = {
            "peak_id": int(group.peak_id[row, k]),
            "x": int(group.x[row]),
            "y": int(group.y[row]),
            "component_id": k,
            "n_components": K,
            "theta_index": int(group.theta_index[row, k]),
            "theta_detected": float(group.theta_detected[row, k]),
            "prominence": float(group.prominence[row, k]),
            "height": float(group.height[row, k]),
            "model": spec.name,
        }
        for name in spec.param_names:
            param = result.params[f"{prefix}{name}"]
            row_out[name] = float(param.value)
            row_out[f"{name}_error"] = np.nan if param.stderr is None else float(param.stderr)
        row_out["theta"] = float(row_out["theta0"])
        row_out["delta_theta"] = row_out["theta"] - row_out["theta_detected"]
        records.append(row_out)
    return records



# --- Схема выходной таблицы ---
_PIXEL_FIT_COLUMNS = (
    "x", "y", "n_detected", "n_fitted", "fit_status", "model",
    "n_data", "n_varying", "dof", "chisqr", "redchi", "rmse",
    "p95_abs_residual", "max_abs_residual", "rmse_over_noise",
    "aic", "bic", "fit_nfev", "message",
)

def _summarize_fit(result, intensity_fit, noise_sigma=None):
    """Считает pixel-level QC и residual."""
    residual = np.asarray(intensity_fit - result.best_fit, dtype=float)
    n_data = len(intensity_fit)
    n_varying = sum(p.vary for p in result.params.values())
    rmse = float(np.sqrt(np.mean(residual**2)))
    diagnostics = {
        "n_data": n_data,
        "n_varying": n_varying,
        "dof": n_data - n_varying,
        "chisqr": float(result.chisqr),
        "redchi": float(result.redchi),
        "rmse": rmse,
        "p95_abs_residual": float(np.percentile(np.abs(residual), 95)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "rmse_over_noise": rmse / noise_sigma if noise_sigma is not None and np.isfinite(noise_sigma) and noise_sigma > 0 else np.nan,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "fit_nfev": int(result.nfev),
    }
    return diagnostics, residual



# ------------------------
# Подгон 
# ------------------------

def _fit_lmfit_one(
    model, template, data_yxt, theta, prepared_fit, K,
    row, *, add_background=True, noise_sigma=None
):
    """Внутренний helper для одного fit. Общий для single-pixel и all-pixels подгона."""
    group = prepared_fit.groups[K]
    x, y = int(group.x[row]), int(group.y[row])
    intensity = np.asarray(data_yxt[y, x, :], dtype=float)
    valid = prepared_fit.theta_valid & np.isfinite(intensity)
    if not np.any(valid):
        raise ValueError(f"No valid data points at ({x}, {y})")

    theta_fit = theta[valid]
    intensity_fit = intensity[valid]
    params = _make_lmfit_params(template=template, group=group, active_names=prepared_fit.active_names, row=row, add_background=add_background)
    result = model.fit(intensity_fit, params=params, theta=theta_fit, method="least_squares")
    diagnostics, residual = _summarize_fit(result=result, intensity_fit=intensity_fit, noise_sigma=noise_sigma)
    return result, valid, residual, diagnostics


# --- Один пиксель ---
def _find_prepared_pixel(prepared_fit, x, y):
    x, y = int(x), int(y)
    for K, group in prepared_fit.groups.items():
        rows = np.flatnonzero((group.x == x) & (group.y == y))
        if len(rows):
            return int(K), int(rows[0])
    raise ValueError(f"No detected peaks at pixel ({x}, {y})")

def fit_lmfit_pixel(
    data_yxt, theta, peak_catalog, spec, *, x, y, prepared_fit=None,
    allow_center_shift=True, center_window=40.0, add_background=True, noise_sigma=None,
):
    """Фитирует одну rocking curve через общий LMFit backend."""
    data_yxt = np.asarray(data_yxt, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if len(theta) != data_yxt.shape[-1]:
        raise ValueError(f"theta length {len(theta)} != data theta dimension {data_yxt.shape[-1]}")

    if prepared_fit is None:
        prepared_fit = prepare_fit_data(
            data_yxt=data_yxt, theta=theta, peak_catalog=peak_catalog, spec=spec,
            allow_center_shift=allow_center_shift, center_window=center_window, add_background=add_background,
        )
    K, row = _find_prepared_pixel(prepared_fit, x, y)
    model, template = _make_lmfit_fitter(spec=spec, K=K, add_background=add_background)
    result, valid, residual, diagnostics = _fit_lmfit_one(
        model=model, template=template, data_yxt=data_yxt, theta=theta,
        prepared_fit=prepared_fit, K=K, row=row, add_background=add_background, noise_sigma=noise_sigma,
    )
    group = prepared_fit.groups[K]
    fitted_records = _make_lmfit_peak_records(result=result, group=group, spec=spec, row=row) if result.success else []
    return {
        "result": result, "fitted_peaks": pd.DataFrame(fitted_records), "valid": valid,
        "residual": residual, "diagnostics": diagnostics, "K": K, "x": int(x), "y": int(y),
    }


# --- Все пиксели ---
def fit_lmfit(
    data_yxt,
    theta,
    peak_catalog,
    spec,
    *,
    prepared_fit=None,
    allow_center_shift=True,
    center_window=40.0,
    add_background=True,
    noise_sigma=None,
    progress=True,
):
    """Multi-peak LMFit для всех pixels с detected peaks."""
    data_yxt = np.asarray(data_yxt, dtype=float)
    theta = np.asarray(theta, dtype=float)
    Ny, Nx, Ntheta = data_yxt.shape
    if len(theta) != Ntheta:
        raise ValueError(f"theta length {len(theta)} != data theta dimension {Ntheta}")
        
    if prepared_fit is None:
        prepared_fit = prepare_fit_data(
            data_yxt=data_yxt,
            theta=theta,
            peak_catalog=peak_catalog,
            spec=spec,
            allow_center_shift=allow_center_shift,
            center_window=center_window,
            add_background=add_background,
        )

    # Один Model + template на K.
    fitters = {
        K: _make_lmfit_fitter(spec=spec, K=K, add_background=add_background)
        for K in sorted(prepared_fit.groups)
    }
    fit_cube = np.full((Ny, Nx, Ntheta), np.nan, dtype=np.float32)
    residual_cube = np.full((Ny, Nx, Ntheta), np.nan, dtype=np.float32)
    fitted_records = []
    pixel_records = []
    total_pixels = sum(len(group.pixel_id) for group in prepared_fit.groups.values())
    detected_mask = np.zeros((Ny, Nx), dtype=bool)
    
    iterator = tqdm(total=total_pixels, desc="LMFit fitting", unit="pixel", disable=not progress)
    try:
        for K in sorted(prepared_fit.groups):
            group = prepared_fit.groups[K]
            model, template = fitters[K]
            for row in range(len(group.pixel_id)):
                x = int(group.x[row])
                y = int(group.y[row])
                detected_mask[y, x] = True
                try:
                    result, valid, residual, diagnostics = _fit_lmfit_one(
                        model=model, template=template,
                        data_yxt=data_yxt, theta=theta, prepared_fit=prepared_fit,
                        K=K, row=row, add_background=add_background, noise_sigma=noise_sigma,
                    )
                    success = bool(result.success)
                    if success:
                        peak_records = _make_lmfit_peak_records(result=result, group=group, spec=spec, row=row)
                        fit_cube[y, x, valid] = np.asarray(result.best_fit, dtype=np.float32)
                        residual_cube[y, x, valid] = np.asarray(residual, dtype=np.float32)
                        fitted_records.extend(peak_records)
                    pixel_records.append({
                        "x": x,
                        "y": y,
                        "n_detected": K,
                        "n_fitted": K if success else 0,
                        "fit_status": "success" if success else "fit_failed",
                        "model": spec.name,
                        **diagnostics,
                        "message": result.message,
                    })
                except Exception as exc:
                    pixel_records.append({
                        "x": x,
                        "y": y,
                        "n_detected": K,
                        "n_fitted": 0,
                        "fit_status": "exception",
                        "model": spec.name,
                        "message": repr(exc),
                    })
                finally:
                    iterator.update(1)
    finally:
        iterator.close()

    # Пиксели без detected peaks.
    for y, x in np.argwhere(~detected_mask):
        pixel_records.append({"x": int(x), "y": int(y), "n_detected": 0, "n_fitted": 0, "fit_status": "no_peaks", "model": spec.name})
    fitted_peak_catalog = pd.DataFrame(fitted_records)
    pixel_fit_catalog = pd.DataFrame.from_records(pixel_records, columns=_PIXEL_FIT_COLUMNS)
    if not fitted_peak_catalog.empty:
        fitted_peak_catalog = fitted_peak_catalog.sort_values(["y", "x", "component_id"]).reset_index(drop=True)
    pixel_fit_catalog = pixel_fit_catalog.sort_values(["y", "x"]).reset_index(drop=True)
    return (
        fitted_peak_catalog,
        pixel_fit_catalog,
        fit_cube,
        residual_cube,
    )




# ==============================
# Визуализация
# ==============================



# Палитра из 10 контрастных, высокополиграфических цветов для компонентов (Colorblind-friendly)
ACADEMIC_PEAK_PALETTE = [
    "#2CA02C",  # Зеленый
    "#9467BD",  # Фиолетовый
    "#17BECF",  # Голубой / Циан
    "#E377C2",  # Розовый
    "#8C564B",  # Коричневый
    "#BCBD22",  # Оливковый
    "#FF7F00",  # Оранжевый
    "#33A02C",  # Темно-зеленый
    "#6A3D9A",  # Темно-фиолетовый
    "#B15928",  # Терракотовый
]


def plot_pixel_fit(
    data_yxt, theta, x, y, pixel_result,
    normalize=True, show_diagnostics=False,
    figsize=(5.5, 4.0), legend_loc="upper left",
    palette="tab10",
):
    """Строит single-pixel fit и опциональные маркеры пиков."""
    data_yxt = np.asarray(data_yxt, dtype=float)
    theta = np.asarray(theta, dtype=float)
    x, y = int(x), int(y)

    result = pixel_result["result"]
    fitted_peaks = pixel_result["fitted_peaks"]
    valid = pixel_result["valid"]

    intensity = data_yxt[y, x, :]
    norm_factor = np.max(intensity) if normalize else 1.0
    intensity = intensity / norm_factor

    best_fit = np.full(theta.shape, np.nan, dtype=float)
    best_fit[valid] = result.best_fit / norm_factor

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    components = result.eval_components(params=result.params, theta=theta)

    # --- Палитра ---
    if isinstance(palette, str) or palette is None:
        cmap = plt.get_cmap(palette or "tab10")
        get_color = lambda i: cmap(i % cmap.N)           # цикл по встроенной палитре
    else:
        get_color = lambda i: palette[i % len(palette)]  # цикл по списку
    # --- Эксперимент и Суммарный фит ---
    ax.plot(theta, intensity, color="black", linewidth=1.2, label="Experimental", zorder=3)
    ax.plot(theta, best_fit, color="crimson", linewidth=1.2, label="Total fit", zorder=3)
    # --- Отрисовка компонентов (заливка + контур) ---
    for i, (_, row) in enumerate(fitted_peaks.reset_index(drop=True).iterrows()):
        color = get_color(i)
        component_name = f"peak_{int(row['component_id'])}_"
        if component_name not in components:
            continue

        comp_y = components[component_name] / norm_factor
        ax.fill_between(theta, 0, comp_y, facecolor=color, edgecolor=color, alpha=0.35, linewidth=0.8, label=f"Peak {row['peak_id']}")
        ax.plot(theta, comp_y, color=color, linestyle="--", linewidth=0.9, alpha=0.85)

        if show_diagnostics:                # диагностические маркеры (опционально)
            ax.axvline(row["theta_detected"], color=color, linestyle=":", linewidth=1.0, alpha=0.7)
            ax.axvline(row["theta"], color=color, linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel(r"$\theta$ (arcsec)", fontsize=11)
    ax.set_ylabel(r"$I/I_{\max}$ (rel. u.)" if normalize else "Intensity (counts)", fontsize=11)
    ax.set_title(f"Pixel ({x}, {y}) Profile Fit", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlim(theta.min(), theta.max())
    ax.set_ylim(bottom=0)                   # фиксируем низ на нуле
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc=legend_loc, frameon=False, fontsize=9.5, handletextpad=0.5, columnspacing=1.0)
    return fig, ax