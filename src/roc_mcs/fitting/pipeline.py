"""Единая точка входа для подгонки скана разными backend-ами."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from roc_mcs.peaks.catalog import enrich_peak_catalog
from roc_mcs.fitting.preparation import prepare_fit_data
from roc_mcs.fitting.result_normalization import broadcast_valid_mask, build_fitted_peak_catalog, normalize_pixel_fit_catalog, validate_valid_mask
from roc_mcs.fitting.results import FitScanResult


def fit_scan(data_yxt, theta, peak_catalog, spec, *, backend="jax", valid_mask=None, allow_center_shift=True,
             center_window=40.0, add_background=True, noise_sigma=None, B=128, solver_config=None, progress=False, metadata=None):
    """Фитирует все пиксели и возвращает общий FitScanResult для JAX или LMFit."""
    data_yxt = np.asarray(data_yxt, dtype=float); theta = np.asarray(theta, dtype=float)
    if data_yxt.ndim != 3 or theta.ndim != 1 or theta.size != data_yxt.shape[-1]:
        raise ValueError("Ожидаются data_yxt формы (Y,X,T) и theta длины T")
    if not np.isfinite(theta).all() or np.any(np.diff(theta) <= 0):
        raise ValueError("theta должна быть конечной и строго возрастающей; при обратном порядке разверните theta и ось T в данных")
    input_valid_mask = validate_valid_mask(valid_mask, data_yxt.shape)
    fit_mask = broadcast_valid_mask(input_valid_mask, data_yxt.shape)
    backend = str(backend).lower()
    if backend not in {"jax", "lmfit"}:
        raise ValueError("backend должен быть 'jax' или 'lmfit'")
    ny, nx, _ = data_yxt.shape
    if "theta0" not in spec.param_names:
        raise ValueError(f"Модель {spec.name!r} должна содержать центральный параметр 'theta0'")
    source_catalog = peak_catalog.copy()
    required_peak_columns = {"peak_id", "x", "y", "theta", "theta_index", "prominence", "height"}
    if not source_catalog.empty:
        missing = required_peak_columns - set(source_catalog.columns)
        if missing:
            raise ValueError(f"В peak_catalog отсутствуют столбцы: {sorted(missing)}")
        if source_catalog["peak_id"].duplicated().any():
            raise ValueError("peak_catalog должен иметь уникальный peak_id")
        for axis_name, limit in (("x", nx), ("y", ny)):
            coords = pd.to_numeric(source_catalog[axis_name], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(coords).all() or not np.equal(coords, np.floor(coords)).all() or np.any((coords < 0) | (coords >= limit)):
                raise ValueError(f"Координаты {axis_name!r} должны быть целыми в диапазоне 0..{limit - 1}")
        if not np.isfinite(pd.to_numeric(source_catalog["theta"], errors="raise").to_numpy(dtype=float)).all():
            raise ValueError("peak_catalog['theta'] должен содержать конечные координаты")
    if source_catalog.empty:
        empty_dtypes = {"peak_id": "int64", "x": "int64", "y": "int64", "theta_index": "int64",
                        "theta": "float64", "prominence": "float64", "height": "float64",
                        "pixel_id": "int64", "component_id": "int64", "n_components": "int64"}
        for name, dtype in empty_dtypes.items():
            if name not in source_catalog.columns:
                source_catalog[name] = np.asarray([], dtype=dtype)
    else:
        source_catalog = enrich_peak_catalog(source_catalog, nx)
    prepared = prepare_fit_data(data_yxt=data_yxt, theta=theta, peak_catalog=source_catalog, spec=spec,
                                valid_mask=fit_mask, allow_center_shift=allow_center_shift,
                                center_window=center_window, add_background=add_background)
    default_solver_cfg = {"max_iter": 200, "ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8}
    solver_cfg = {**default_solver_cfg, **(solver_config or {})}
    if backend == "jax":
        from roc_mcs.fitting.backends.jax_backend import fit_jax, _decode_jax_results
        raw = fit_jax(data_yxt, theta, source_catalog, spec, valid_mask=fit_mask, B=B,
                      solver_config=solver_cfg, allow_center_shift=allow_center_shift,
                      center_window=center_window, add_background=add_background, progress=progress,
                      prepared_fit=prepared)
        decoded = _decode_jax_results(raw, data_yxt, theta, source_catalog, spec, prepared_fit=prepared,
                                      add_background=add_background, valid_mask=fit_mask, noise_sigma=noise_sigma)
        fitted_params = decoded["fitted_params"]
        pixel_records = decoded["pixel_records"]
        fit_cube, residual_cube = decoded["fit_cube"], decoded["residual_cube"]
        pixel_param_names = decoded["pixel_param_names"]
    else:
        from roc_mcs.fitting.backends.lmfit_backend import fit_lmfit
        fitted_raw, pixel_raw, fit_cube, residual_cube = fit_lmfit(
            data_yxt, theta, source_catalog, spec, prepared_fit=prepared, allow_center_shift=allow_center_shift,
            center_window=center_window, add_background=add_background, noise_sigma=noise_sigma,
            progress=progress, valid_mask=fit_mask)
        fitted_params, pixel_records = fitted_raw, pixel_raw
        pixel_param_names = ("bg_background",) if add_background else ()
        solver_cfg = {"method": "least_squares"}
    existing_pixels = {(int(r["x"]), int(r["y"])) for r in pixel_records}
    detected_counts = source_catalog.groupby(["x", "y"], sort=False).size().to_dict() if not source_catalog.empty else {}
    for pixel_id, message in getattr(prepared, "preparation_errors", {}).items():
        y, x = divmod(int(pixel_id), nx)
        if (x, y) not in existing_pixels:
            pixel_records.append({"x": x, "y": y, "n_detected": int(detected_counts.get((x, y), 0)), "n_fitted": 0,
                                  "fit_status": "preparation_failed", "model": str(spec.name), "message": message})
            existing_pixels.add((x, y))
    fitted_peak_catalog = build_fitted_peak_catalog(source_catalog, fitted_params, nx, spec)
    pixel_fit_catalog = normalize_pixel_fit_catalog(pixel_records, (ny, nx), spec.name, source_catalog,
                                                   pixel_param_names=pixel_param_names)
    fit_config = {"allow_center_shift": bool(allow_center_shift), "center_window": float(center_window),
                  "add_background": bool(add_background), "noise_sigma": noise_sigma,
                  "valid_mask_shape": None if input_valid_mask is None else tuple(input_valid_mask.shape),
                  "batch_size": int(B) if backend == "jax" else None, **solver_cfg}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{spec.name}_{backend}_{stamp}"
    return FitScanResult(
        peak_catalog=source_catalog, fitted_peak_catalog=fitted_peak_catalog,
        pixel_fit_catalog=pixel_fit_catalog, fit_cube=fit_cube, residual_cube=residual_cube,
        theta=theta, theta_valid=np.asarray(prepared.theta_valid, dtype=bool), image_shape=(ny, nx),
        model_id=str(spec.name), backend=backend, component_param_names=tuple(spec.param_names),
        pixel_param_names=tuple(pixel_param_names), valid_mask=input_valid_mask, run_id=run_id,
        fit_config=fit_config, model_version=str(getattr(spec, "version", "1")), metadata=dict(metadata or {}),
    )
