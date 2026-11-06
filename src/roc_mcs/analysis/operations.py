"""Adapters from immutable workflow stages to existing ROC-MCS algorithms.

This module intentionally contains orchestration only. Scientific algorithms remain
in roc_mcs.roi, roc_mcs.processing, roc_mcs.peaks and roc_mcs.fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .artifacts import ArtifactBundle


@dataclass(frozen=True, slots=True)
class OperationResult:
    bundle: ArtifactBundle
    output_ports: tuple[str, ...]
    summary: str


@dataclass(frozen=True, slots=True)
class OperationSpec:
    operation_id: str
    label: str
    description: str
    output_ports: tuple[str, ...]
    execute: Callable[[Mapping[str, ArtifactBundle], Mapping[str, Any]], OperationResult]
    validate: Callable[[Mapping[str, Any]], dict[str, Any]]
    parameter_schema_version: int = 1




def _integer_parameter(parameters: Mapping[str, Any], name: str, default: int) -> int:
    raw = parameters.get(name, default)
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{name} должен быть целым числом, не bool")
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} должен быть целым числом") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} должен быть конечным целым числом")
    return int(number)


class OperationRegistry:
    def __init__(self):
        self._specs: dict[str, OperationSpec] = {}

    def register(self, spec: OperationSpec) -> None:
        if spec.operation_id in self._specs:
            raise ValueError(f"Операция уже зарегистрирована: {spec.operation_id}")
        self._specs[spec.operation_id] = spec

    def get(self, operation_id: str) -> OperationSpec:
        try:
            return self._specs[operation_id]
        except KeyError as exc:
            raise KeyError(f"Неизвестная операция: {operation_id}") from exc

    def all(self) -> tuple[OperationSpec, ...]:
        return tuple(self._specs.values())

    def execute(
        self,
        operation_id: str,
        inputs: Mapping[str, ArtifactBundle],
        parameters: Mapping[str, Any],
    ) -> OperationResult:
        spec = self.get(operation_id)
        normalized = spec.validate(parameters)
        result = spec.execute(inputs, normalized)
        if result.output_ports != spec.output_ports:
            raise RuntimeError(
                f"{operation_id}: фактические output_ports {result.output_ports} "
                f"не совпадают с контрактом {spec.output_ports}"
            )
        return result


def _validate_pool(parameters: Mapping[str, Any]) -> dict[str, Any]:
    from roc_mcs.roi.spatial_pooling import PoolSpec

    size = _integer_parameter(parameters, "size", 5)
    stride = _integer_parameter(parameters, "stride", 1)
    spec = PoolSpec(size=size, stride=stride)
    return {"size": int(spec.size), "stride": int(spec.stride)}


def _run_pool(inputs: Mapping[str, ArtifactBundle], parameters: Mapping[str, Any]) -> OperationResult:
    from roc_mcs.roi.spatial_pooling import PoolSpec, pool_centers, pool_yxt

    scan = inputs["scan"]
    data = np.asarray(scan.arrays["data_yxt"])
    theta = np.asarray(scan.arrays["theta"], dtype=float)
    x_coords = np.asarray(scan.arrays["x_coords"], dtype=float)
    y_coords = np.asarray(scan.arrays["y_coords"], dtype=float)
    ny, nx, ntheta = data.shape
    spec = PoolSpec(**parameters)
    pooled = pool_yxt(data, spec)
    x_idx, y_idx = pool_centers((ny, nx), spec)
    # pool_centers returns detector-grid coordinates as float for ROI geometry;
    # here the same integer-valued centers are used as array indices.
    out_x = x_coords[np.rint(x_idx).astype(np.intp)]
    out_y = y_coords[np.rint(y_idx).astype(np.intp)]

    parent_grid = str(scan.metadata["grid_id"])
    grid_id = _grid_signature(parent_grid, spec.signature)
    metadata = dict(scan.metadata)
    metadata.update({
        "grid_id": grid_id,
        "shape_yx": [int(pooled.shape[0]), int(pooled.shape[1])],
        "pool": spec.to_dict(),
        "pool_signature": spec.signature,
        "operation": "spatial count-sum pooling",
        "input_shape_yxt": [int(ny), int(nx), int(ntheta)],
    })
    bundle = ArtifactBundle(
        arrays={
            "data_yxt": pooled,
            "theta": theta,
            "x_coords": out_x,
            "y_coords": out_y,
        },
        metadata=metadata,
    )
    return OperationResult(
        bundle=bundle,
        output_ports=("scan",),
        summary=(
            f"Pooling size={spec.size}, stride={spec.stride}: "
            f"{ny}×{nx} → {pooled.shape[0]}×{pooled.shape[1]} pixels"
        ),
    )


def _validate_spikes(parameters: Mapping[str, Any]) -> dict[str, Any]:
    max_iter = _integer_parameter(parameters, "max_iter", 5)
    tolerance = float(parameters.get("tol", 1e-3))
    if max_iter < 1 or max_iter > 100:
        raise ValueError("max_iter должен быть в диапазоне 1…100")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tol должен быть конечным числом > 0")
    return {
        "max_iter": max_iter,
        "tol": tolerance,
        "collect_kappa_fits": bool(parameters.get("collect_kappa_fits", True)),
    }


def _run_spikes(inputs: Mapping[str, ArtifactBundle], parameters: Mapping[str, Any]) -> OperationResult:
    from roc_mcs.processing.spikes import refine_spike_mask

    scan = inputs["scan"]
    data_yxt = np.asarray(scan.arrays["data_yxt"], dtype=float)
    if data_yxt.ndim != 3 or not np.isfinite(data_yxt).any():
        raise ValueError("Spike detection требует непустой конечный массив Y×X×θ")
    # The existing spike code expects native (theta, Y, X) layout.
    data_tyx = np.moveaxis(data_yxt, -1, 0)
    result = refine_spike_mask(
        data_tyx,
        max_iter=parameters["max_iter"],
        tol=parameters["tol"],
        verbose=False,
        collect_kappa_fits=parameters["collect_kappa_fits"],
    )
    if not np.isfinite(float(result.kappa)):
        raise RuntimeError("Spike detection не смог определить конечное значение κ")
    mask_tyx = np.asarray(result.spike_mask, dtype=bool)
    clean_tyx = np.asarray(result.clean_data, dtype=float)
    if mask_tyx.shape != data_tyx.shape or clean_tyx.shape != data_tyx.shape:
        raise ValueError("refine_spike_mask вернул результат с несовместимой формой")
    valid_yxt = ~np.moveaxis(mask_tyx, 0, -1)
    clean_yxt = np.moveaxis(clean_tyx, 0, -1)
    count_map = np.count_nonzero(mask_tyx, axis=0).astype(np.int32)

    arrays = {
        "spike_mask_tyx": mask_tyx,
        "valid_mask_yxt": valid_yxt,
        "clean_data_yxt": clean_yxt,
        "spike_count_map": count_map,
        "kappa_history": np.asarray([row["kappa"] for row in result.history], dtype=float),
        "spike_count_history": np.asarray([row["n_spikes"] for row in result.history], dtype=np.int64),
    }
    import pandas as pd
    iteration_table = pd.DataFrame(result.history)
    kappa_rows: list[dict[str, Any]] = []
    if result.kappa_fits:
        for iteration_index, fit in enumerate(result.kappa_fits):
            for tested_kappa, n_spikes in zip(fit.kappas_positive, fit.n_spikes_positive):
                kappa_rows.append({
                    "iteration": int(iteration_index),
                    "tested_kappa": float(tested_kappa),
                    "n_spikes": int(n_spikes),
                    "selected_kappa": float(fit.kappa),
                    "n_segments": int(fit.n_segments),
                    "bic_2": float(fit.bic_2),
                    "bic_3": float(fit.bic_3),
                })
    kappa_table = pd.DataFrame(
        kappa_rows,
        columns=["iteration", "tested_kappa", "n_spikes", "selected_kappa", "n_segments", "bic_2", "bic_3"],
    )

    metadata = {
        "grid_id": scan.metadata["grid_id"],
        "kappa": float(result.kappa),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "n_spikes": int(np.count_nonzero(mask_tyx)),
        "spike_fraction": float(np.mean(mask_tyx)),
        "background_median": float(result.background.median),
        "background_sigma": float(result.background.sigma),
        "background_threshold": float(result.background.threshold),
        "mask_semantics": "True means detected spike / invalid sample",
        "clean_data_semantics": "interpolation output from existing refine_spike_mask",
        "diagnostic_tables": ["spike_iterations", "kappa_thresholds"],
    }
    return OperationResult(
        bundle=ArtifactBundle(
            arrays=arrays,
            tables={"spike_iterations": iteration_table, "kappa_thresholds": kappa_table},
            metadata=metadata,
        ),
        output_ports=("spikes",),
        summary=(
            f"Обнаружено {metadata['n_spikes']:,} spike-точек "
            f"({metadata['spike_fraction']:.4%}); κ={metadata['kappa']:.4g}; "
            f"итераций={metadata['iterations']}"
        ),
    )


def _validate_peaks(parameters: Mapping[str, Any]) -> dict[str, Any]:
    from roc_mcs.peaks.detection import PeakDetectionConfig

    mode = str(parameters.get("prominence_mode", "sigma"))
    prominence = float(parameters.get("prominence_value", 3.0))
    config_kwargs: dict[str, Any] = {
        "smooth_window": _integer_parameter(parameters, "smooth_window", 11),
        "smooth_polyorder": _integer_parameter(parameters, "smooth_polyorder", 2),
        "min_distance": _integer_parameter(parameters, "min_distance", 6),
        "min_relative_prominence": (
            None if parameters.get("min_relative_prominence", 0.07) in (None, "")
            else float(parameters.get("min_relative_prominence", 0.07))
        ),
    }
    if not np.isfinite(prominence) or prominence <= 0:
        raise ValueError("prominence_value должен быть конечным числом > 0")
    if mode == "sigma":
        config_kwargs["prominence_sigma"] = prominence
        config_kwargs["prominence_fraction"] = None
    elif mode == "fraction":
        config_kwargs["prominence_fraction"] = prominence
        config_kwargs["prominence_sigma"] = None
    else:
        raise ValueError("prominence_mode должен быть 'sigma' или 'fraction'")
    config = PeakDetectionConfig(**config_kwargs)
    window_size = _integer_parameter(parameters, "window_size", 3)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size должен быть положительным нечётным числом")
    return {
        "window_size": window_size,
        "use_interpolated_data": bool(parameters.get("use_interpolated_data", False)),
        "smooth_window": config.smooth_window,
        "smooth_polyorder": config.smooth_polyorder,
        "min_distance": config.min_distance,
        "prominence_mode": mode,
        "prominence_value": prominence,
        "min_relative_prominence": config.min_relative_prominence,
    }


def _run_peaks(inputs: Mapping[str, ArtifactBundle], parameters: Mapping[str, Any]) -> OperationResult:
    import pandas as pd
    from roc_mcs.peaks.catalog import build_peak_catalog
    from roc_mcs.peaks.detection import PeakDetectionConfig

    scan = inputs["scan"]
    data_yxt = np.asarray(scan.arrays["data_yxt"], dtype=float)
    theta = np.asarray(scan.arrays["theta"], dtype=float)
    # build_peak_catalog / estimate_background_stats contract is canonical
    # theta×Y×X. Do not reconstruct the HDF5-native order here: when the
    # source file stores theta on axis 2, passing Y×X×theta would make the
    # background estimator reduce the wrong dimension.
    source_data_yxt = data_yxt
    used_interpolated_data = bool(parameters["use_interpolated_data"] and "spikes" in inputs)
    if parameters["use_interpolated_data"] and "spikes" not in inputs:
        raise ValueError(
            "Включено использование интерполированных данных, но в выбранной ветви "
            "нет этапа обнаружения спайков. Добавьте этот этап или отключите флажок."
        )
    if used_interpolated_data:
        spike_grid = str(inputs["spikes"].metadata.get("grid_id", ""))
        if spike_grid != str(scan.metadata.get("grid_id", "")):
            raise ValueError("Spike mask и scan относятся к разным spatial-grid")
        source_data_yxt = np.asarray(inputs["spikes"].arrays["clean_data_yxt"], dtype=float)
    if theta.ndim != 1 or not np.isfinite(theta).all() or np.any(np.diff(theta) <= 0):
        raise ValueError("Детектор пиков требует конечную строго возрастающую ось θ")
    if parameters["smooth_window"] > theta.size:
        raise ValueError(
            f"Окно Savitzky–Golay ({parameters['smooth_window']}) больше числа θ-отсчётов ({theta.size})"
        )
    data_tyx = np.moveaxis(source_data_yxt, -1, 0)

    if parameters["prominence_mode"] == "sigma":
        config = PeakDetectionConfig(
            smooth_window=parameters["smooth_window"],
            smooth_polyorder=parameters["smooth_polyorder"],
            min_distance=parameters["min_distance"],
            prominence_sigma=parameters["prominence_value"],
            prominence_fraction=None,
            min_relative_prominence=parameters["min_relative_prominence"],
        )
    else:
        config = PeakDetectionConfig(
            smooth_window=parameters["smooth_window"],
            smooth_polyorder=parameters["smooth_polyorder"],
            min_distance=parameters["min_distance"],
            prominence_sigma=None,
            prominence_fraction=parameters["prominence_value"],
            min_relative_prominence=parameters["min_relative_prominence"],
        )

    # Deliberately call the established catalogue builder instead of
    # duplicating its local-neighbourhood/background scientific logic.
    catalog, background_sigma, background_threshold = build_peak_catalog(
        data_tyx,
        theta,
        window_size=parameters["window_size"],
        peak_detection_config=config,
    )
    if not isinstance(catalog, pd.DataFrame):
        catalog = pd.DataFrame(catalog)
    # Keep a stable schema even when no peaks were detected. Downstream Fit and
    # preview code rely on x/y/theta columns being present for an empty catalog.
    empty_catalog_columns = {
        "peak_id": np.int64,
        "x": np.int64,
        "y": np.int64,
        "theta_index": np.int64,
        "theta": np.float64,
        "prominence": np.float64,
        "width_samples": np.float64,
        "width_theta": np.float64,
        "left_ip": np.float64,
        "right_ip": np.float64,
        "height": np.float64,
    }
    if catalog.empty:
        for column, dtype in empty_catalog_columns.items():
            if column not in catalog.columns:
                catalog[column] = pd.Series(dtype=dtype)
    else:
        missing_columns = set(empty_catalog_columns) - set(catalog.columns)
        if missing_columns:
            raise ValueError(
                "build_peak_catalog вернул неполную схему каталога: "
                f"отсутствуют {sorted(missing_columns)}"
            )
    peak_count = np.zeros(data_yxt.shape[:2], dtype=np.int32)
    if not catalog.empty:
        counts = catalog.groupby(["y", "x"], sort=False).size()
        for (y, x), count in counts.items():
            if 0 <= int(y) < peak_count.shape[0] and 0 <= int(x) < peak_count.shape[1]:
                peak_count[int(y), int(x)] = int(count)
    metadata = {
        "grid_id": scan.metadata["grid_id"],
        "used_interpolated_data": used_interpolated_data,
        "n_peaks": int(len(catalog)),
        "n_pixels_with_peaks": int(np.count_nonzero(peak_count)),
        "background_sigma": float(background_sigma),
        "background_threshold": float(background_threshold),
        "peak_config": {
            key: parameters[key] for key in (
                "window_size", "smooth_window", "smooth_polyorder",
                "min_distance", "prominence_mode", "prominence_value",
                "min_relative_prominence",
            )
        },
    }
    return OperationResult(
        bundle=ArtifactBundle(
            arrays={"peak_count_map": peak_count},
            tables={"peak_catalog": catalog},
            metadata=metadata,
        ),
        output_ports=("peaks",),
        summary=(
            f"Найдено пиков: {metadata['n_peaks']:,}; "
            f"пикселей с пиками: {metadata['n_pixels_with_peaks']:,}"
        ),
    )



def _module_available(module_name: str) -> bool:
    """Return whether an optional runtime dependency can be discovered."""
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def fit_backend_availability() -> dict[str, Any]:
    """Single source of truth for optional fit backends and model support."""
    from roc_mcs.fitting.registry import MODEL_SPECS

    return {
        "lmfit": _module_available("lmfit"),
        "jax": _module_available("jax") and _module_available("jaxlib"),
        "jax_models": {
            name: spec.jax_func is not None for name, spec in MODEL_SPECS.items()
        },
    }


def _validate_fit(parameters: Mapping[str, Any]) -> dict[str, Any]:
    model = str(parameters.get("model", "pvoigt"))
    backend = str(parameters.get("backend", "lmfit")).lower()
    if backend not in {"jax", "lmfit"}:
        raise ValueError("backend должен быть 'jax' или 'lmfit'")
    from roc_mcs.fitting.registry import MODEL_SPECS
    if model not in MODEL_SPECS:
        raise ValueError(f"Неизвестная fit-модель {model!r}")
    availability = fit_backend_availability()
    if not availability[backend]:
        module_name = "jax/jaxlib" if backend == "jax" else "lmfit"
        raise ValueError(
            f"Для backend={backend!r} не установлены необходимые пакеты {module_name!r}. "
            "Установите зависимость или выберите доступный backend."
        )
    if backend == "jax" and not availability["jax_models"].get(model, False):
        raise ValueError(
            f"Для модели {model!r} JAX-backend не реализован; выберите LMFit."
        )
    center_window = float(parameters.get("center_window", 40.0))
    if not np.isfinite(center_window) or center_window <= 0:
        raise ValueError("center_window должен быть конечным числом > 0")
    batch_size = _integer_parameter(parameters, "batch_size", 128) if backend == "jax" else None
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size должен быть >= 1")
    solver_max_iter = _integer_parameter(parameters, "solver_max_iter", 200) if backend == "jax" else None
    if solver_max_iter is not None and solver_max_iter < 1:
        raise ValueError("solver_max_iter должен быть >= 1")
    return {
        "model": model,
        "backend": backend,
        "use_spike_mask": bool(parameters.get("use_spike_mask", True)),
        "allow_center_shift": bool(parameters.get("allow_center_shift", True)),
        "center_window": center_window,
        "add_background": bool(parameters.get("add_background", True)),
        "batch_size": batch_size,
        "solver_max_iter": solver_max_iter,
    }


def _run_fit(inputs: Mapping[str, ArtifactBundle], parameters: Mapping[str, Any]) -> OperationResult:
    from roc_mcs.fitting.pipeline import fit_scan
    from roc_mcs.fitting.registry import MODEL_SPECS

    scan = inputs["scan"]
    peaks = inputs["peaks"]
    catalog = peaks.tables.get("peak_catalog")
    if catalog is None:
        raise ValueError("Во входном артефакте отсутствует peak_catalog")
    scan_grid = str(scan.metadata.get("grid_id", ""))
    if str(peaks.metadata.get("grid_id", "")) != scan_grid:
        raise ValueError("Каталог пиков и scan относятся к разным spatial-grid")
    spike_bundle = inputs.get("spikes") if parameters.get("use_spike_mask", True) else None
    valid_mask = None
    if spike_bundle is not None:
        if str(spike_bundle.metadata.get("grid_id", "")) != scan_grid:
            raise ValueError("Spike mask и scan относятся к разным spatial-grid")
        valid_mask = np.asarray(spike_bundle.arrays["valid_mask_yxt"], dtype=bool)
        if valid_mask.shape != np.asarray(scan.arrays["data_yxt"]).shape:
            raise ValueError("Spike valid_mask имеет форму, несовместимую со scan")
    spec = MODEL_SPECS.get(parameters["model"])
    if spec is None:
        raise ValueError(f"Неизвестная fit-модель {parameters['model']!r}")
    result = fit_scan(
        np.asarray(scan.arrays["data_yxt"], dtype=float),
        np.asarray(scan.arrays["theta"], dtype=float),
        catalog,
        spec,
        backend=parameters["backend"],
        valid_mask=valid_mask,
        allow_center_shift=parameters["allow_center_shift"],
        center_window=parameters["center_window"],
        add_background=parameters["add_background"],
        B=parameters["batch_size"] or 128,
        solver_config=(
            {"max_iter": parameters["solver_max_iter"]}
            if parameters["solver_max_iter"] is not None else None
        ),
        progress=False,
        metadata={"workflow_operation": "fit.scan", "grid_id": scan.metadata["grid_id"]},
    )

    arrays: dict[str, np.ndarray] = {
        "theta": np.asarray(result.theta),
        "theta_valid": np.asarray(result.theta_valid),
    }
    if result.fit_cube is not None:
        arrays["fit_cube"] = np.asarray(result.fit_cube)
    if result.residual_cube is not None:
        arrays["residual_cube"] = np.asarray(result.residual_cube)
    if result.valid_mask is not None:
        arrays["valid_mask"] = np.asarray(result.valid_mask)
    if not result.pixel_fit_catalog.empty and "n_fitted" in result.pixel_fit_catalog:
        nmap = np.zeros(result.image_shape, dtype=np.int32)
        for row in result.pixel_fit_catalog[["x", "y", "n_fitted"]].itertuples(index=False, name=None):
            x, y, count = row
            if not np.isfinite(float(x)) or not np.isfinite(float(y)) or not np.isfinite(float(count)):
                continue
            xi, yi = int(x), int(y)
            if 0 <= yi < nmap.shape[0] and 0 <= xi < nmap.shape[1]:
                nmap[yi, xi] = int(count)
        arrays["n_fitted_map"] = nmap

    tables = {
        "peak_catalog": result.peak_catalog,
        "fitted_peak_catalog": result.fitted_peak_catalog,
        "pixel_fit_catalog": result.pixel_fit_catalog,
    }
    metadata = {
        "grid_id": scan.metadata["grid_id"],
        "model_id": result.model_id,
        "model_version": result.model_version,
        "backend": result.backend,
        "use_spike_mask": valid_mask is not None,
        "run_id": result.run_id,
        "fit_config": result.fit_config,
        "component_param_names": list(result.component_param_names),
        "pixel_param_names": list(result.pixel_param_names),
        "image_shape": list(result.image_shape),
    }
    statuses = (
        result.pixel_fit_catalog["fit_status"].value_counts(dropna=False).to_dict()
        if "fit_status" in result.pixel_fit_catalog else {}
    )
    summary = f"Fit завершён: модель={result.model_id}, backend={result.backend}; статусы={statuses}"
    return OperationResult(
        bundle=ArtifactBundle(arrays=arrays, tables=tables, metadata=metadata),
        output_ports=("fit",),
        summary=summary,
    )


def _grid_signature(parent_grid: str, operation_signature: str) -> str:
    import hashlib

    return hashlib.sha256(
        f"{parent_grid}|{operation_signature}".encode("utf-8")
    ).hexdigest()[:24]


def default_registry() -> OperationRegistry:
    """Реестр поддерживаемых операций первой интеграционной версии."""
    registry = OperationRegistry()
    registry.register(OperationSpec(
        operation_id="preprocess.pool",
        label="Пространственный pooling",
        description="Суммирование пространственных окон; ось θ не агрегируется.",
        output_ports=("scan",),
        execute=_run_pool,
        validate=_validate_pool,
    ))
    registry.register(OperationSpec(
        operation_id="preprocess.spikes",
        label="Обнаружение и маскирование спайков",
        description="Автоматический κ-fit, spike mask и отдельное interpolated представление.",
        output_ports=("spikes",),
        execute=_run_spikes,
        validate=_validate_spikes,
    ))
    registry.register(OperationSpec(
        operation_id="peaks.detect",
        label="Детектор пиков",
        description="Существующий build_peak_catalog с зафиксированной конфигурацией.",
        output_ports=("peaks",),
        execute=_run_peaks,
        validate=_validate_peaks,
    ))
    registry.register(OperationSpec(
        operation_id="fit.scan",
        label="Подгонка скана",
        description="Существующая fit_scan; явные ссылки на peak catalog и valid mask.",
        output_ports=("fit",),
        execute=_run_fit,
        validate=_validate_fit,
    ))
    return registry
