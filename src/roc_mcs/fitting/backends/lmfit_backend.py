import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from lmfit import Model, Parameters

from roc_mcs.fitting.registry import ModelSpec
from roc_mcs.fitting.preparation import PreparedFitGroup, prepare_fit_data
from roc_mcs.fitting.result_normalization import broadcast_valid_mask, summarize_residuals, validate_valid_mask


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
    "p95_abs_residual", "max_abs_residual", "rmse_over_noise", "aic", "bic", "fit_nfev", "message",
    "bg_background", "bg_background_error",
)

def _summarize_fit(result, intensity_fit, noise_sigma=None):
    """Считает общие QC-метрики и residual = данные - модель."""
    n_varying = sum(p.vary for p in result.params.values())
    diagnostics, residual = summarize_residuals(intensity_fit, result.best_fit, n_varying=n_varying, noise_sigma=noise_sigma)
    diagnostics["fit_nfev"] = int(result.nfev)
    return diagnostics, residual



# ------------------------
# Подгон 
# ------------------------

def _fit_lmfit_one(
    model, template, data_yxt, theta, prepared_fit, K,
    row, *, add_background=True, noise_sigma=None, valid_mask=None
):
    """Внутренний helper для одного fit. Общий для single-pixel и all-pixels подгона."""
    group = prepared_fit.groups[K]
    x, y = int(group.x[row]), int(group.y[row])
    intensity = np.asarray(data_yxt[y, x, :], dtype=float)
    valid = prepared_fit.theta_valid & np.isfinite(intensity)   # базовая валидность: окно теты + отсутствие NaN в данных
    if valid_mask is not None:                                  # накладываем пользовательскую маску спайков, если она передана
        valid &= np.asarray(valid_mask, dtype=bool)
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
    valid_mask=None
):
    """Фитирует одну rocking curve через общий LMFit backend."""
    data_yxt = np.asarray(data_yxt, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if len(theta) != data_yxt.shape[-1]:
        raise ValueError(f"theta length {len(theta)} != data theta dimension {data_yxt.shape[-1]}")
    valid_mask = broadcast_valid_mask(valid_mask, data_yxt.shape)

    if prepared_fit is None:
        prepared_fit = prepare_fit_data(
            data_yxt=data_yxt, theta=theta, peak_catalog=peak_catalog, spec=spec,
            valid_mask=valid_mask,
            allow_center_shift=allow_center_shift, center_window=center_window, add_background=add_background,
        )
    K, row = _find_prepared_pixel(prepared_fit, x, y)
    model, template = _make_lmfit_fitter(spec=spec, K=K, add_background=add_background)
    pixel_mask = valid_mask[int(y), int(x)] if valid_mask is not None else None  # Вырезаем 1D маску для конкретного пикселя (y, x)

    result, valid, residual, diagnostics = _fit_lmfit_one(
        model=model, template=template, data_yxt=data_yxt, theta=theta,
        prepared_fit=prepared_fit, K=K, row=row, add_background=add_background, noise_sigma=noise_sigma,
        valid_mask=pixel_mask,
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
    valid_mask=None,
):
    """Multi-peak LMFit для всех pixels с detected peaks."""
    data_yxt = np.asarray(data_yxt, dtype=float)
    theta = np.asarray(theta, dtype=float)
    Ny, Nx, Ntheta = data_yxt.shape
    if len(theta) != Ntheta:
        raise ValueError(f"theta length {len(theta)} != data theta dimension {Ntheta}")
    valid_mask = broadcast_valid_mask(valid_mask, data_yxt.shape)

    if prepared_fit is None:
        prepared_fit = prepare_fit_data(
            data_yxt=data_yxt,
            theta=theta,
            peak_catalog=peak_catalog,
            spec=spec,
            valid_mask=valid_mask,
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
    if not peak_catalog.empty:
        detected_mask[peak_catalog["y"].to_numpy(dtype=int), peak_catalog["x"].to_numpy(dtype=int)] = True
    
    iterator = tqdm(total=total_pixels, desc="LMFit fitting", unit="pixel", disable=not progress)
    try:
        for K in sorted(prepared_fit.groups):
            group = prepared_fit.groups[K]
            model, template = fitters[K]
            for row in range(len(group.pixel_id)):
                x = int(group.x[row])
                y = int(group.y[row])
                detected_mask[y, x] = True
                pixel_mask = valid_mask[y, x] if valid_mask is not None else None   # Вырезаем маску для пикселя
                try:
                    result, valid, residual, diagnostics = _fit_lmfit_one(
                        model=model, template=template,
                        data_yxt=data_yxt, theta=theta, prepared_fit=prepared_fit,
                        K=K, row=row, add_background=add_background, noise_sigma=noise_sigma,
                        valid_mask=pixel_mask,
                    )
                    success = bool(result.success and np.isfinite(np.asarray(result.best_fit, dtype=float)).all())
                    if success and any(not np.isfinite(float(p.value)) for p in result.params.values()):
                        success = False
                    pixel_params = {"bg_background": np.nan, "bg_background_error": np.nan}
                    if success:
                        peak_records = _make_lmfit_peak_records(result=result, group=group, spec=spec, row=row)
                        fit_cube[y, x, valid] = np.asarray(result.best_fit, dtype=np.float32)
                        residual_cube[y, x, valid] = np.asarray(residual, dtype=np.float32)
                        fitted_records.extend(peak_records)
                        if add_background and "bg_background" in result.params:
                            bg = result.params["bg_background"]
                            pixel_params = {"bg_background": float(bg.value), "bg_background_error": np.nan if bg.stderr is None else float(bg.stderr)}
                    pixel_records.append({
                        "x": x, "y": y, "n_detected": K, "n_fitted": K if success else 0,
                        "fit_status": "success" if success else "fit_failed", "model": spec.name,
                        **diagnostics, **pixel_params, "message": result.message,
                    })
                except Exception as exc:
                    no_data = isinstance(exc, ValueError) and "No valid data points" in str(exc)
                    pixel_records.append({
                        "x": x, "y": y, "n_detected": K, "n_fitted": 0,
                        "fit_status": "fit_failed" if no_data else "exception", "model": spec.name,
                        "message": str(exc) if no_data else repr(exc),
                    })
                finally:
                    iterator.update(1)
    finally:
        iterator.close()

    # Пиксели, которые не удалось подготовить.
    counts = peak_catalog.groupby(["x", "y"], sort=False).size().to_dict() if not peak_catalog.empty else {}
    recorded = {(int(r["x"]), int(r["y"])) for r in pixel_records}
    for pixel_id, message in getattr(prepared_fit, "preparation_errors", {}).items():
        yy, xx = divmod(int(pixel_id), Nx)
        if (xx, yy) not in recorded:
            pixel_records.append({"x": xx, "y": yy, "n_detected": int(counts.get((xx, yy), 0)), "n_fitted": 0,
                                 "fit_status": "preparation_failed", "model": spec.name, "message": message})
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


