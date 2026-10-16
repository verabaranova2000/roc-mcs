"""Общий формат каталогов и диагностики для всех backend-ов."""
import numpy as np
import pandas as pd

from roc_mcs.peaks.catalog import enrich_peak_catalog


PIXEL_BASE_COLUMNS = (
    "x", "y", "n_detected", "n_fitted", "fit_status", "model", "n_data", "n_varying", "dof",
    "chisqr", "redchi", "rmse", "p95_abs_residual", "max_abs_residual", "rmse_over_noise",
    "aic", "bic", "fit_nfev", "n_iter", "cost", "optimality", "convergence_reason", "message",
)


def validate_valid_mask(valid_mask, data_shape):
    """Разрешает маску формы (Y,X,T), (Y,X) или (T); True означает использовать точку."""
    if valid_mask is None:
        return None
    mask = np.asarray(valid_mask, dtype=bool)
    ny, nx, nt = map(int, data_shape)
    allowed = ((ny, nx, nt), (ny, nx), (nt,))
    if mask.shape not in allowed:
        raise ValueError(f"valid_mask имеет форму {mask.shape}; ожидается одна из {allowed}")
    return mask


def broadcast_valid_mask(valid_mask, data_shape):
    """Приводит допустимые входные маски к (Y,X,T) без копирования, если возможно."""
    mask = validate_valid_mask(valid_mask, data_shape)
    if mask is None or mask.ndim == 3:
        return mask
    ny, nx, nt = map(int, data_shape)
    if mask.ndim == 2:
        return np.broadcast_to(mask[:, :, None], (ny, nx, nt))
    return np.broadcast_to(mask[None, None, :], (ny, nx, nt))


def pixel_valid_mask(data_yxt, theta_valid, valid_mask, *, x, y):
    intensity = np.asarray(data_yxt[int(y), int(x), :], dtype=float)
    valid = np.asarray(theta_valid, dtype=bool) & np.isfinite(intensity)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim == 3:
            valid &= mask[int(y), int(x), :]
        elif mask.ndim == 2:
            valid &= bool(mask[int(y), int(x)])
        elif mask.ndim == 1:
            valid &= mask
        else:
            raise ValueError(f"Недопустимая форма valid_mask: {mask.shape}")
    return intensity, valid


def summarize_residuals(observed, predicted, *, n_varying, noise_sigma=None):
    """Единые QC-метрики; residual имеет знак observed - predicted."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    residual = observed[keep] - predicted[keep]
    n_data = int(residual.size)
    n_varying = int(n_varying)
    if n_data == 0:
        return {"n_data": 0, "n_varying": n_varying, "dof": -n_varying, **{k: np.nan for k in ("chisqr", "redchi", "rmse", "p95_abs_residual", "max_abs_residual", "rmse_over_noise", "aic", "bic")}}, residual
    abs_residual = np.abs(residual)
    chisqr = float(np.dot(residual, residual))
    dof = n_data - n_varying
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    variance = max(chisqr / n_data, np.finfo(float).tiny)
    diagnostics = {
        "n_data": n_data, "n_varying": n_varying, "dof": dof, "chisqr": chisqr,
        "redchi": chisqr / dof if dof > 0 else np.nan, "rmse": rmse,
        "p95_abs_residual": float(np.percentile(abs_residual, 95)),
        "max_abs_residual": float(np.max(abs_residual)),
        "rmse_over_noise": rmse / noise_sigma if noise_sigma is not None and np.isfinite(noise_sigma) and noise_sigma > 0 else np.nan,
        "aic": float(n_data * np.log(variance) + 2 * n_varying),
        "bic": float(n_data * np.log(variance) + np.log(n_data) * n_varying),
    }
    return diagnostics, residual


def build_fitted_peak_catalog(peak_catalog, fitted_params, Nx, spec):
    """Связывает параметры успешных фитирований с исходными пиками по peak_id."""
    param_names = tuple(spec.param_names)
    center_name = "theta0" if "theta0" in param_names else None
    if center_name is None:
        raise ValueError(f"Модель {spec.name!r} должна объявлять центральный параметр 'theta0'")
    source = peak_catalog.copy()
    if source.empty:
        empty_dtypes = {"peak_id": "int64", "x": "int64", "y": "int64", "theta_index": "int64",
                        "theta": "float64", "prominence": "float64", "height": "float64",
                        "pixel_id": "int64", "component_id": "int64", "n_components": "int64"}
        for name, dtype in empty_dtypes.items():
            if name not in source.columns:
                source[name] = pd.Series(dtype=dtype)
    elif not {"pixel_id", "component_id", "n_components"}.issubset(source.columns):
        source = enrich_peak_catalog(source, Nx)
    if "theta_detected" not in source.columns:
        if "theta" not in source.columns:
            if source.empty:
                source["theta_detected"] = pd.Series(dtype=float)
            else:
                raise ValueError("В peak_catalog нет исходной координаты 'theta'")
        else:
            source = source.rename(columns={"theta": "theta_detected"})
    fit = pd.DataFrame(fitted_params).copy()
    required = ["peak_id", *param_names, *(f"{name}_error" for name in param_names)]
    for name in required:
        if name not in fit.columns:
            fit[name] = np.nan if name != "peak_id" else pd.Series(dtype=np.int64)
    fit = fit[required]
    if not fit.empty and fit["peak_id"].duplicated().any():
        duplicates = fit.loc[fit["peak_id"].duplicated(), "peak_id"].tolist()
        raise ValueError(f"В результатах подгонки повторяются peak_id: {duplicates[:5]}")
    metadata_cols = [c for c in source.columns if c not in set(param_names) | {f"{n}_error" for n in param_names} | {"theta", "delta_theta", "model", "fit_success"}]
    source = source[metadata_cols]
    out = source.merge(fit, on="peak_id", how="inner", validate="one_to_one")
    out["theta"] = pd.to_numeric(out[center_name], errors="coerce")
    out["delta_theta"] = out["theta"] - pd.to_numeric(out["theta_detected"], errors="coerce")
    out["model"] = str(spec.name)
    out["fit_success"] = True
    columns = [*metadata_cols, *param_names, *(f"{n}_error" for n in param_names), "theta", "delta_theta", "model", "fit_success"]
    out = out.reindex(columns=list(dict.fromkeys(columns)))
    if not out.empty:
        sort_cols = [c for c in ("y", "x", "component_id") if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, kind="stable")
    return out.reset_index(drop=True)


def normalize_pixel_fit_catalog(records, image_shape, model_id, peak_catalog, *, pixel_param_names=()):
    """Гарантирует одну строку на каждый пиксель и общую схему QC."""
    ny, nx = map(int, image_shape)
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame.from_records(records)
    if frame.empty or not {"x", "y"}.issubset(frame.columns):
        frame = pd.DataFrame(columns=["x", "y"])
    if not frame.empty:
        frame["x"] = pd.to_numeric(frame["x"], errors="raise").astype(int)
        frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(int)
        if frame.duplicated(["x", "y"]).any():
            raise ValueError("pixel_fit_catalog должен содержать не более одной строки на пиксель")
        if ((frame.x < 0) | (frame.x >= nx) | (frame.y < 0) | (frame.y >= ny)).any():
            raise ValueError("В pixel_fit_catalog есть координаты вне изображения")
    detected = {}
    if peak_catalog is not None and not peak_catalog.empty:
        counts = peak_catalog.groupby(["x", "y"], sort=False).size()
        detected = {(int(x), int(y)): int(n) for (x, y), n in counts.items()}
    keyed = {(int(r.x), int(r.y)): r._asdict() for r in frame.itertuples(index=False)}
    rows = []
    for y in range(ny):
        for x in range(nx):
            row = keyed.get((x, y))
            if row is None:
                n_detected = detected.get((x, y), 0)
                row = {"x": x, "y": y, "n_detected": n_detected, "n_fitted": 0,
                       "fit_status": "no_peaks" if n_detected == 0 else "not_processed", "model": model_id}
            row.setdefault("n_detected", detected.get((x, y), 0))
            row.setdefault("n_fitted", 0)
            row.setdefault("fit_status", "not_processed" if row["n_detected"] else "no_peaks")
            row.setdefault("model", model_id)
            rows.append(row)
    out = pd.DataFrame.from_records(rows)
    for name in PIXEL_BASE_COLUMNS:
        if name not in out:
            out[name] = np.nan
    for name in pixel_param_names:
        if name not in out:
            out[name] = np.nan
        if f"{name}_error" not in out:
            out[f"{name}_error"] = np.nan
    known_pixel_params = {"bg_background", "bg_background_error"} | {f"{n}_error" for n in pixel_param_names} | set(pixel_param_names)
    for name in known_pixel_params - set(pixel_param_names) - {f"{n}_error" for n in pixel_param_names}:
        if name in out.columns:
            out = out.drop(columns=name)
    extra = [c for c in out.columns if c not in PIXEL_BASE_COLUMNS and c not in {f"{n}_error" for n in pixel_param_names} and c not in pixel_param_names]
    columns = list(PIXEL_BASE_COLUMNS) + list(pixel_param_names) + [f"{n}_error" for n in pixel_param_names] + extra
    return out.reindex(columns=list(dict.fromkeys(columns))).sort_values(["y", "x"]).reset_index(drop=True)
