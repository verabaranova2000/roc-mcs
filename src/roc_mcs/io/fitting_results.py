"""Чтение и запись результатов fitting в /<entry>/fits/runs/<run_id>."""
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pandas as pd

from roc_mcs.fitting.results import FitScanResult


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"Не могу сериализовать {type(value).__name__} в JSON")


def _clean_json(value):
    if isinstance(value, dict): return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.ndarray): return _clean_json(value.tolist())
    return value


def _json_dump(value):
    return json.dumps(_clean_json(value), default=_json_default, ensure_ascii=False, allow_nan=False)


def _text_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _create_dataset(group, name, data, *, compression=True):
    arr = np.asarray(data)
    kwargs = {"compression": "gzip", "shuffle": True} if compression and arr.size > 0 and arr.ndim > 0 else {}
    if arr.dtype.kind in "UO":
        flat = ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in arr.reshape(-1)]
        arr = np.asarray(flat, dtype=object).reshape(arr.shape)
        return group.create_dataset(name, data=arr, dtype=_text_dtype(), **kwargs)
    return group.create_dataset(name, data=arr, **kwargs)


def _write_frame(group, frame):
    group.attrs["columns_json"] = _json_dump(list(frame.columns))
    group.attrs["dtypes_json"] = _json_dump({name: str(frame[name].dtype) for name in frame.columns})
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            data = series.to_numpy()
        elif series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            nonnull = series.dropna()
            if nonnull.map(lambda v: isinstance(v, (str, bytes, np.str_))).all():
                data = series.fillna("").astype(str).to_numpy(dtype=object)
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.notna().sum() == series.notna().sum():
                    data = numeric.to_numpy()
                else:
                    raise TypeError(f"Столбец {name!r} содержит неподдерживаемые смешанные типы")
        else:
            data = series.to_numpy()
        _create_dataset(group, name, data)


def _read_dataset(dataset):
    arr = dataset[()]
    if getattr(arr, "dtype", None) is not None and arr.dtype.kind in "SO":
        decode = np.vectorize(lambda x: x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x), otypes=[object])
        return decode(arr)
    return arr


def _read_frame(group):
    cols = json.loads(group.attrs.get("columns_json", "[]"))
    dtypes = json.loads(group.attrs.get("dtypes_json", "{}"))
    data = {}
    for name in cols:
        values = _read_dataset(group[name])
        dtype = dtypes.get(name, "")
        if dtype.startswith("int") and np.asarray(values).size:
            values = pd.to_numeric(values, errors="coerce").astype(dtype)
        elif dtype == "bool" and np.asarray(values).size:
            values = np.asarray(values, dtype=bool)
        elif dtype.startswith("float") and np.asarray(values).size:
            values = np.asarray(values, dtype=dtype)
        data[name] = values
    return pd.DataFrame(data, columns=cols)


def _open_target(target, mode):
    if isinstance(target, (h5py.File, h5py.Group)):
        return nullcontext(target)
    return h5py.File(target, mode)


def save_fit_scan_result(target, entry_key, result: FitScanResult, *, run_id=None, overwrite=False,
                         save_reconstruction=False, save_valid_mask=True):
    """Добавляет run в существующий entry, не меняя исходные HDF5 datasets."""
    if not isinstance(result, FitScanResult):
        raise TypeError("result должен быть FitScanResult")
    run_id = str(run_id or result.run_id or f"{result.model_id}_{result.backend}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
    with _open_target(target, "a") as h5:
        if entry_key not in h5:
            raise KeyError(f"В HDF5 нет группы {entry_key!r}; сначала укажите существующий scan entry")
        entry = h5[entry_key]
        runs = entry.require_group("fits").require_group("runs")
        if not run_id or "/" in run_id:
            raise ValueError("run_id должен быть непустым именем HDF5-группы без '/'")
        if run_id in runs and not overwrite:
            raise FileExistsError(f"Fit run {entry_key}/fits/runs/{run_id} уже существует")
        pending_id = f".__pending__{uuid4().hex}"
        backup_id = f".__backup__{uuid4().hex}" if run_id in runs else None
        run = runs.create_group(pending_id)
        try:
            run.attrs["run_id"] = run_id
            run.attrs["created_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            run.attrs["model_id"] = result.model_id
            run.attrs["model_version"] = result.model_version
            run.attrs["backend"] = result.backend
            run.attrs["fit_config_json"] = _json_dump(result.fit_config)
            run.attrs["metadata_json"] = _json_dump(result.metadata)
            run.attrs["image_shape_json"] = _json_dump(result.image_shape)
            model = run.create_group("model")
            _create_dataset(model, "theta", result.theta)
            _create_dataset(model, "theta_valid", result.theta_valid.astype(np.uint8))
            _create_dataset(model, "component_param_names", np.asarray(result.component_param_names, dtype=object), compression=False)
            _create_dataset(model, "pixel_param_names", np.asarray(result.pixel_param_names, dtype=object), compression=False)
            peaks = run.create_group("peaks").create_group("catalog")
            _write_frame(peaks, result.peak_catalog)

            ny, nx = result.image_shape
            pixels = run.create_group("pixels")
            pixel_df = result.pixel_fit_catalog
            def pixel_map(column, default, dtype=float):
                out = np.full((ny, nx), default, dtype=dtype)
                if column in pixel_df:
                    for row in pixel_df[["x", "y", column]].itertuples(index=False, name=None):
                        x, y, value = row
                        out[int(y), int(x)] = value
                return out
            _create_dataset(pixels, "status", pixel_map("fit_status", "", object))
            _create_dataset(pixels, "n_detected", pixel_map("n_detected", 0, np.int32).astype(np.int32))
            _create_dataset(pixels, "n_fitted", pixel_map("n_fitted", 0, np.int32).astype(np.int32))
            messages = pixels.create_group("message")
            _create_dataset(messages, "text", pixel_map("message", "", object))
            diagnostics = pixels.create_group("diagnostics")
            diag_names = ("n_data", "n_varying", "dof", "chisqr", "redchi", "rmse", "p95_abs_residual", "max_abs_residual", "rmse_over_noise", "aic", "bic", "fit_nfev", "n_iter", "cost", "optimality")
            for name in diag_names:
                _create_dataset(diagnostics, name, pixel_map(name, np.nan, float))
            convergence = pixel_df.get("convergence_reason", pd.Series(dtype=object))
            if len(convergence):
                _create_dataset(diagnostics, "convergence_reason", pixel_map("convergence_reason", "", object))
            if result.pixel_param_names:
                params_group = pixels.create_group("parameters")
                values = np.full((ny, nx, len(result.pixel_param_names)), np.nan, dtype=float)
                stderr = np.full_like(values, np.nan)
                indexed = pixel_df.set_index(["y", "x"], drop=False)
                for yi in range(ny):
                    for xi in range(nx):
                        if (yi, xi) not in indexed.index:
                            continue
                        row = indexed.loc[(yi, xi)]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        for j, name in enumerate(result.pixel_param_names):
                            values[yi, xi, j] = pd.to_numeric(row.get(name, np.nan), errors="coerce")
                            stderr[yi, xi, j] = pd.to_numeric(row.get(f"{name}_error", np.nan), errors="coerce")
                _create_dataset(params_group, "values", values)
                _create_dataset(params_group, "stderr", stderr)
            if save_valid_mask and result.valid_mask is not None:
                ds = _create_dataset(pixels, "input_valid_mask", result.valid_mask.astype(np.uint8))
                ds.attrs["semantics"] = "1=True: точка участвует в подгонке"

            components = run.create_group("components")
            fitted = result.fitted_peak_catalog.copy()
            sort_cols = [c for c in ("y", "x", "component_id") if c in fitted.columns]
            if not fitted.empty and sort_cols:
                fitted = fitted.sort_values(sort_cols, kind="stable").reset_index(drop=True)
            starts = np.full((ny, nx), -1, dtype=np.int64); counts = np.zeros((ny, nx), dtype=np.int32)
            if not fitted.empty and {"x", "y"}.issubset(fitted.columns):
                for (yi, xi), grp in fitted.groupby(["y", "x"], sort=False):
                    starts[int(yi), int(xi)] = int(grp.index.min())
                    counts[int(yi), int(xi)] = int(len(grp))
            index = components.create_group("index")
            _create_dataset(index, "start", starts); _create_dataset(index, "count", counts)
            fields = ("x", "y", "component_id", "peak_id", "theta_index", "theta_detected", "theta", "delta_theta", "prominence", "height")
            for field in fields:
                dest = "component_index" if field == "component_id" else field
                if field in fitted.columns:
                    values_field = fitted[field].to_numpy()
                else:
                    dtype = np.int64 if field in {"x", "y", "component_id", "peak_id", "theta_index"} else float
                    values_field = np.asarray([], dtype=dtype)
                _create_dataset(components, dest, values_field)
            n_comp = len(fitted); pnames = result.component_param_names
            comp_values = np.full((n_comp, len(pnames)), np.nan, dtype=float)
            comp_stderr = np.full_like(comp_values, np.nan)
            for j, name in enumerate(pnames):
                if name in fitted:
                    comp_values[:, j] = pd.to_numeric(fitted[name], errors="coerce").to_numpy(dtype=float)
                if f"{name}_error" in fitted:
                    comp_stderr[:, j] = pd.to_numeric(fitted[f"{name}_error"], errors="coerce").to_numpy(dtype=float)
            _create_dataset(components, "values", comp_values)
            _create_dataset(components, "stderr", comp_stderr)
            if save_reconstruction:
                recon = run.create_group("reconstruction")
                if result.fit_cube is not None: _create_dataset(recon, "fit_cube", result.fit_cube)
                if result.residual_cube is not None: _create_dataset(recon, "residual_cube", result.residual_cube)
            if backup_id is not None:
                runs.move(run_id, backup_id)
            try:
                runs.move(pending_id, run_id)
            except Exception:
                if backup_id is not None and backup_id in runs and run_id not in runs:
                    runs.move(backup_id, run_id)
                raise
            if backup_id is not None and backup_id in runs:
                del runs[backup_id]
        except Exception:
            if pending_id in runs:
                del runs[pending_id]
            raise
    return run_id


def load_fit_scan_result(target, entry_key, run_id) -> FitScanResult:
    """Загружает каталог, карты и метаданные; кубы реконструкции читаются, если сохранены."""
    with _open_target(target, "r") as h5:
        run = h5[entry_key]["fits"]["runs"][run_id]
        model = run["model"]
        theta = _read_dataset(model["theta"]).astype(float)
        theta_valid = _read_dataset(model["theta_valid"]).astype(bool)
        component_names = tuple(str(x) for x in _read_dataset(model["component_param_names"]).tolist())
        pixel_names = tuple(str(x) for x in _read_dataset(model["pixel_param_names"]).tolist())
        peak_catalog = _read_frame(run["peaks"]["catalog"])
        ny, nx = json.loads(run.attrs["image_shape_json"])
        pixels = run["pixels"]
        status = _read_dataset(pixels["status"])
        detected = _read_dataset(pixels["n_detected"])
        n_fitted = _read_dataset(pixels["n_fitted"])
        message = _read_dataset(pixels["message"]["text"])
        diag = pixels["diagnostics"]
        diag_arrays = {name: _read_dataset(ds) for name, ds in diag.items()}
        n_pixels = ny * nx
        expected_shape = (ny, nx)
        if any(np.asarray(a).shape != expected_shape for a in (status, detected, n_fitted, message)):
            raise ValueError(f"Повреждённые размеры пиксельных массивов: ожидалось {expected_shape}")
        pixel_data = {
            "x": np.tile(np.arange(nx, dtype=np.int32), ny),
            "y": np.repeat(np.arange(ny, dtype=np.int32), nx),
            "fit_status": np.asarray(status).reshape(-1).astype(str),
            "n_detected": np.asarray(detected).reshape(-1).astype(np.int32),
            "n_fitted": np.asarray(n_fitted).reshape(-1).astype(np.int32),
            "model": np.full(n_pixels, str(run.attrs["model_id"]), dtype=object),
            "message": np.asarray(message).reshape(-1).astype(str),
        }
        for name, arr in diag_arrays.items():
            arr = np.asarray(arr)
            if arr.shape != expected_shape:
                raise ValueError(f"Диагностика {name!r}: ожидалась форма {expected_shape}, получена {arr.shape}")
            pixel_data[name] = arr.reshape(-1)
        if pixel_names and "parameters" in pixels:
            values = _read_dataset(pixels["parameters"]["values"])
            stderr = _read_dataset(pixels["parameters"]["stderr"])
            expected_params_shape = (ny, nx, len(pixel_names))
            if values.shape != expected_params_shape or stderr.shape != expected_params_shape:
                raise ValueError(f"Неверная форма карт параметров: ожидалось {expected_params_shape}")
            for j, name in enumerate(pixel_names):
                pixel_data[name] = values[:, :, j].reshape(-1)
                pixel_data[f"{name}_error"] = stderr[:, :, j].reshape(-1)
        pixel_fit_catalog = pd.DataFrame(pixel_data)
        comp = run["components"]
        n_comp = len(comp["peak_id"])
        peak_ids = _read_dataset(comp["peak_id"]).astype(np.int64, copy=False)
        values = _read_dataset(comp["values"])
        stderr = _read_dataset(comp["stderr"])
        expected_component_shape = (n_comp, len(component_names))
        if values.shape != expected_component_shape or stderr.shape != expected_component_shape:
            raise ValueError(f"Неверная форма параметров компонент: ожидалось {expected_component_shape}")
        fitted_data = {"peak_id": peak_ids}
        for j, name in enumerate(component_names):
            fitted_data[name] = values[:, j]
            fitted_data[f"{name}_error"] = stderr[:, j]
        fitted_params = pd.DataFrame(fitted_data, columns=["peak_id", *component_names, *(f"{n}_error" for n in component_names)])
        source = peak_catalog.copy()
        if "theta_detected" not in source.columns and "theta" in source.columns:
            source = source.rename(columns={"theta": "theta_detected"})
        fitted_catalog = source.merge(fitted_params, on="peak_id", how="inner", validate="one_to_one")
        center_name = "theta0" if "theta0" in component_names else None
        if center_name is not None:
            fitted_catalog["theta"] = fitted_catalog[center_name]
            fitted_catalog["delta_theta"] = fitted_catalog["theta"] - fitted_catalog["theta_detected"]
        fitted_catalog["model"] = str(run.attrs["model_id"])
        fitted_catalog["fit_success"] = True
        sort_cols = [c for c in ("y", "x", "component_id") if c in fitted_catalog.columns]
        if sort_cols:
            fitted_catalog = fitted_catalog.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        fit_cube = residual_cube = None
        if "reconstruction" in run:
            recon = run["reconstruction"]
            if "fit_cube" in recon: fit_cube = _read_dataset(recon["fit_cube"])
            if "residual_cube" in recon: residual_cube = _read_dataset(recon["residual_cube"])
        valid_mask = _read_dataset(pixels["input_valid_mask"]).astype(bool) if "input_valid_mask" in pixels else None
        return FitScanResult(
            peak_catalog=peak_catalog, fitted_peak_catalog=fitted_catalog, pixel_fit_catalog=pixel_fit_catalog,
            fit_cube=fit_cube, residual_cube=residual_cube, theta=theta, theta_valid=theta_valid,
            image_shape=(ny, nx), model_id=str(run.attrs["model_id"]), backend=str(run.attrs["backend"]),
            component_param_names=component_names, pixel_param_names=pixel_names, valid_mask=valid_mask,
            run_id=str(run.attrs["run_id"]), fit_config=json.loads(run.attrs.get("fit_config_json", "{}")),
            model_version=str(run.attrs.get("model_version", "1")), metadata=json.loads(run.attrs.get("metadata_json", "{}")),
        )


def list_fit_runs(target, entry_key):
    """Возвращает метаданные сохранённых запусков, новые первыми."""
    with _open_target(target, "r") as h5:
        if entry_key not in h5:
            raise KeyError(f"В HDF5 нет группы {entry_key!r}")
        entry = h5[entry_key]
        if "fits" not in entry or "runs" not in entry["fits"]:
            return []
        records = []
        for name, run in entry["fits"]["runs"].items():
            if not isinstance(run, h5py.Group) or name.startswith(".__") or "run_id" not in run.attrs:
                continue
            records.append({"run_id": str(run.attrs["run_id"]), "model_id": str(run.attrs.get("model_id", "unknown")),
                            "model_version": str(run.attrs.get("model_version", "1")),
                            "backend": str(run.attrs.get("backend", "unknown")),
                            "created_utc": str(run.attrs.get("created_utc", "")),
                            "fit_config": json.loads(run.attrs.get("fit_config_json", "{}")),
                            "metadata": json.loads(run.attrs.get("metadata_json", "{}"))})
        return sorted(records, key=lambda item: item["created_utc"], reverse=True)
