"""Validated, renderer-neutral views assembled from immutable workflow artifacts.

The view model deliberately separates spatial maps from Y×X×theta signals so the
ROI Explorer can offer only physically compatible display choices per node.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PreviewMapLayer:
    key: str
    label: str
    data_yx: np.ndarray
    group: str = "Основные данные"
    unit: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class PreviewCurveSource:
    key: str
    label: str
    data_yxt: np.ndarray
    theta: np.ndarray
    valid_mask_yxt: np.ndarray | None = None
    group: str = "Сигналы"
    description: str = ""


@dataclass(frozen=True, slots=True)
class WorkflowPreviewState:
    node_id: str
    operation_id: str
    label: str
    data_yxt: np.ndarray
    theta: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    valid_mask_yxt: np.ndarray | None
    metadata: dict[str, Any]
    map_layers: tuple[PreviewMapLayer, ...]
    curve_sources: tuple[PreviewCurveSource, ...]
    default_map_key: str
    default_curve_key: str
    summary: str = ""
    analysis_valid_mask_yx: np.ndarray | None = None
    peak_points: Any | None = None
    peak_position_kind: str = "detected"
    branch_edges: Any | None = None
    branch_correlation_preview: Any | None = None
    hkl_identification_preview: Any | None = None

    def map_layer(self, key: str | None = None) -> PreviewMapLayer:
        selected = str(key or self.default_map_key)
        for layer in self.map_layers:
            if layer.key == selected:
                return layer
        for layer in self.map_layers:
            if layer.key == self.default_map_key:
                return layer
        if self.map_layers:
            return self.map_layers[0]
        raise KeyError("В узле отсутствуют пространственные карты для отображения")

    def curve_source(self, key: str | None = None) -> PreviewCurveSource:
        selected = str(key or self.default_curve_key)
        for source in self.curve_sources:
            if source.key == selected:
                return source
        for source in self.curve_sources:
            if source.key == self.default_curve_key:
                return source
        if self.curve_sources:
            return self.curve_sources[0]
        raise KeyError("В узле отсутствуют ROC-источники для отображения")

    @property
    def map_data(self) -> np.ndarray:
        """Compatibility view for callers written for ScanViewState."""
        return self.map_layer().data_yx

    @property
    def map_title(self) -> str:
        return self.map_layer().label


def _validated_axis(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size != int(length) or not np.isfinite(result).all():
        raise ValueError(f"Ось {name} должна быть конечной 1D-осью длины {length}")
    if result.size > 1:
        delta = np.diff(result)
        if not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"Ось {name} должна быть строго монотонной")
    return result


def _valid_mask(value: Any, shape: tuple[int, int, int], name: str) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} имеет форму {mask.shape}, ожидалась {shape}")
    return mask


def _sum_and_mean_maps(data: np.ndarray, valid: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    ny, nx, _ = data.shape
    sums = np.full((ny, nx), np.nan, dtype=np.float64)
    means = np.full((ny, nx), np.nan, dtype=np.float64)
    # Process by rows: keep temporary memory bounded for large detector grids.
    for y0 in range(0, ny, 32):
        y1 = min(ny, y0 + 32)
        block = np.asarray(data[y0:y1])
        mask = np.isfinite(block)
        if valid is not None:
            mask &= valid[y0:y1]
        counts = np.count_nonzero(mask, axis=2)
        total = np.sum(np.where(mask, block, 0.0), axis=2, dtype=np.float64)
        good = counts > 0
        sums[y0:y1][good] = total[good]
        means[y0:y1][good] = total[good] / counts[good]
    return sums, means


def _add_map(
    layers: list[PreviewMapLayer], *, key: str, label: str, value: Any,
    shape_yx: tuple[int, int], group: str = "Основные данные", unit: str = "",
    description: str = "",
) -> None:
    if value is None:
        return
    array = np.asarray(value)
    if array.shape != shape_yx:
        # Never display a map from a different spatial grid.
        return
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).any():
        return
    layers.append(PreviewMapLayer(
        key=key, label=label, data_yx=np.asarray(array, dtype=np.float64),
        group=group, unit=unit, description=description,
    ))


def _table(context: Any, name: str):
    if context is None:
        return None
    tables = getattr(context, "tables", None)
    if not isinstance(tables, dict):
        return None
    return tables.get(name)


def _pixel_metric_map(frame: Any, column: str, shape_yx: tuple[int, int]) -> np.ndarray | None:
    if frame is None or getattr(frame, "empty", True) or not {"x", "y", column}.issubset(frame.columns):
        return None
    ny, nx = shape_yx
    result = np.full((ny, nx), np.nan, dtype=np.float64)
    for x_value, y_value, value in frame[["x", "y", column]].itertuples(index=False, name=None):
        try:
            x_float, y_float, scalar = float(x_value), float(y_value), float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (np.isfinite(x_float) and np.isfinite(y_float) and np.isfinite(scalar)):
            continue
        x, y = int(x_float), int(y_float)
        if 0 <= x < nx and 0 <= y < ny:
            result[y, x] = scalar
    return result if np.isfinite(result).any() else None


def _peak_catalog_map(frame: Any, column: str, shape_yx: tuple[int, int], *, aggregate: str) -> np.ndarray | None:
    if frame is None or getattr(frame, "empty", True) or not {"x", "y", column}.issubset(frame.columns):
        return None
    ny, nx = shape_yx
    buckets: dict[tuple[int, int], list[float]] = {}
    for x_value, y_value, value in frame[["x", "y", column]].itertuples(index=False, name=None):
        try:
            x, y, scalar = int(x_value), int(y_value), float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= x < nx and 0 <= y < ny and np.isfinite(scalar):
            buckets.setdefault((y, x), []).append(scalar)
    if not buckets:
        return None
    result = np.full((ny, nx), np.nan, dtype=np.float64)
    for (y, x), values in buckets.items():
        if aggregate == "max":
            result[y, x] = float(np.max(values))
        elif aggregate == "mean":
            result[y, x] = float(np.mean(values))
        elif aggregate == "count":
            result[y, x] = float(len(values))
    return result


def _residual_summary_maps(residual: Any, valid_mask: np.ndarray | None, shape_yx: tuple[int, int]) -> list[PreviewMapLayer]:
    if residual is None or np.shape(residual)[:2] != shape_yx or np.ndim(residual) != 3:
        return []
    ny, nx, _ = np.shape(residual)
    sums = np.zeros((ny, nx), dtype=np.float64)
    abs_sums = np.zeros((ny, nx), dtype=np.float64)
    squares = np.zeros((ny, nx), dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=np.int64)
    for y0 in range(0, ny, 32):
        y1 = min(ny, y0 + 32)
        block = np.asarray(residual[y0:y1], dtype=np.float64)
        valid = np.isfinite(block)
        if valid_mask is not None:
            valid &= valid_mask[y0:y1]
        counts_block = np.count_nonzero(valid, axis=2)
        sums_block = np.sum(np.where(valid, block, 0.0), axis=2, dtype=np.float64)
        abs_block = np.sum(np.where(valid, np.abs(block), 0.0), axis=2, dtype=np.float64)
        sq_block = np.sum(np.where(valid, np.square(block), 0.0), axis=2, dtype=np.float64)
        sums[y0:y1] = sums_block
        abs_sums[y0:y1] = abs_block
        squares[y0:y1] = sq_block
        counts[y0:y1] = counts_block
    good = counts > 0
    if not np.any(good):
        return []
    mean = np.full((ny, nx), np.nan, dtype=np.float64)
    mae = np.full((ny, nx), np.nan, dtype=np.float64)
    rmse = np.full((ny, nx), np.nan, dtype=np.float64)
    mean[good] = sums[good] / counts[good]
    mae[good] = abs_sums[good] / counts[good]
    rmse[good] = np.sqrt(squares[good] / counts[good])
    return [
        PreviewMapLayer("fit.residual_mean", "Средний знаковый остаток", mean,
                        "Остатки fit", "counts", "Среднее (измерение − модель) по валидным отсчётам θ; знак показывает систематическое смещение."),
        PreviewMapLayer("fit.residual_mae", "Средний модуль остатка", mae,
                        "Остатки fit", "counts", "Среднее абсолютное значение residual_cube по валидным отсчётам θ."),
        PreviewMapLayer("fit.residual_rmse_cube", "RMSE по residual_cube", rmse,
                        "Остатки fit", "counts", "Корень из среднего квадрата residual_cube по валидным отсчётам θ."),
    ]


def _component_parameter_maps(fit: Any, shape_yx: tuple[int, int]) -> list[PreviewMapLayer]:
    frame = _table(fit, "fitted_peak_catalog")
    if frame is None or getattr(frame, "empty", True):
        return []
    required = {"x", "y", "component_id"}
    if not required.issubset(frame.columns):
        return []
    metadata = getattr(fit, "metadata", {}) or {}
    names = list(metadata.get("component_param_names", ()) or ())
    if not names:
        names = [name for name in frame.columns if name not in required and not str(name).endswith("_error")]
    try:
        component_ids = sorted({int(v) for v in frame["component_id"].dropna().tolist() if float(v).is_integer()})
    except (TypeError, ValueError, OverflowError):
        component_ids = []
    if not component_ids:
        return []
    ny, nx = shape_yx
    layers: list[PreviewMapLayer] = []
    model_id = str(metadata.get("model_id", ""))
    try:
        from roc_mcs.fitting.registry import DERIVED_SPECS
        derived = DERIVED_SPECS.get(model_id, {})
    except Exception:
        derived = {}

    component_names = {
        "S": "Площадь", "theta0": "Положение центра θ₀", "H": "Высота",
        "eta": "Параметр смешения η", "sigma": "Ширина σ", "gamma": "Ширина γ",
        "Delta": "Амплитуда углового сдвига Δθₚ",
        "beta_Gl": "Gaussian-ширина слева β_Gl", "beta_Cl": "Lorentz-ширина слева β_Cl",
        "beta_Gr": "Gaussian-ширина справа β_Gr", "beta_Cr": "Lorentz-ширина справа β_Cr",
        "lam": "Параметр экспоненциального хвоста λ",
    }
    for name in names:
        if name not in frame.columns:
            continue
        for component_id in component_ids:
            result = np.full((ny, nx), np.nan, dtype=np.float64)
            subset = frame.loc[frame["component_id"] == component_id, ["x", "y", name]]
            for x_value, y_value, value in subset.itertuples(index=False, name=None):
                try:
                    x, y, scalar = int(x_value), int(y_value), float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if 0 <= x < nx and 0 <= y < ny and np.isfinite(scalar):
                    result[y, x] = scalar
            if np.isfinite(result).any():
                friendly = component_names.get(str(name), str(name))
                if name == "S":
                    unit = "counts·θ"
                elif name in {"theta0", "H", "sigma", "gamma", "Delta", "beta_Gl", "beta_Cl", "beta_Gr", "beta_Cr"}:
                    unit = "θ"
                elif name == "lam":
                    unit = "θ⁻¹"
                elif name == "eta":
                    unit = "—"
                else:
                    unit = ""
                layers.append(PreviewMapLayer(
                    key=f"fit.parameter.{name}.component{component_id}",
                    label=f"{friendly} · компонента {component_id + 1}",
                    data_yx=result, group="Параметры модели", unit=unit,
                    description=f"Параметр {name} модели {model_id}, компонент {component_id + 1}; значение взято из fitted_peak_catalog.",
                ))

    # FWHM is a registered derived quantity only for models whose expression
    # does not require a separately sampled curve (split-Voigt is excluded).
    fwhm_spec = derived.get("FWHM") if isinstance(derived, dict) else None
    if fwhm_spec is not None and not bool(getattr(fwhm_spec, "needs_curve", True)):
        if names and all(p in frame.columns for p in names):
            for component_id in component_ids:
                result = np.full((ny, nx), np.nan, dtype=np.float64)
                subset = frame.loc[frame["component_id"] == component_id, ["x", "y", *names]]
                for row in subset.itertuples(index=False, name=None):
                    x_value, y_value, *parameter_values = row
                    try:
                        x, y = int(x_value), int(y_value)
                        parameters = {p: float(v) for p, v in zip(names, parameter_values)}
                        width = float(fwhm_spec.value_func(parameters))
                    except Exception:
                        continue
                    if 0 <= x < nx and 0 <= y < ny and np.isfinite(width):
                        result[y, x] = width
                if np.isfinite(result).any():
                    layers.append(PreviewMapLayer(
                        key=f"fit.derived.FWHM.component{component_id}",
                        label=f"FWHM · компонента {component_id + 1}",
                        data_yx=result, group="Derived-параметры", unit="θ",
                        description=f"FWHM рассчитана зарегистрированным выражением модели {model_id} из параметров fit; не является отдельным параметром подгонки.",
                    ))
    return layers


def _fit_valid_mask(fit: Any, shape: tuple[int, int, int]) -> np.ndarray | None:
    value = getattr(fit, "arrays", {}).get("valid_mask") if fit is not None else None
    if value is None:
        return None
    mask = np.asarray(value, dtype=bool)
    if mask.shape == shape:
        return mask
    if mask.shape == shape[:2]:
        return np.broadcast_to(mask[:, :, None], shape)
    if mask.shape == (shape[2],):
        return np.broadcast_to(mask[None, None, :], shape)
    return None



def _hkl_alignment_map_layers(
    preview: Any, branch_catalog: Any, shape_yx: tuple[int, int],
) -> tuple[list[PreviewMapLayer], dict[str, Any], str | None]:
    """Build sparse signed post-registration Δθ maps on either branch grid.

    Correlation applies ``theta_partner + theta_shift_final`` in source
    coordinates. Thus the residual is ``(theta_partner + shift) - theta_source``.
    The same event residual is indexed once by source pixels and once by partner
    pixels; no nearest-neighbour coordinate reconstruction is performed.
    Many-to-one assignments are averaged per target pixel and described in the
    layer metadata.
    """
    measurements = getattr(preview, "measurements", None)
    statistics = getattr(preview, "pair_statistics", None)
    required_catalog = {"peak_id", "branch_id", "x", "y"}
    required_measurements = {
        "pair_id", "source_branch_id", "partner_branch_id",
        "source_peak_id", "matched_partner_peak_id",
    }
    required_statistics = {"pair_id", "theta_shift_final_arcsec"}
    if measurements is None or statistics is None or branch_catalog is None:
        return [], {}, "Недостаточно таблиц для построения карт выровненной Δθ."
    if not required_catalog.issubset(getattr(branch_catalog, "columns", ())):
        return [], {}, "branch_catalog не содержит peak_id, branch_id, x, y."
    if not required_measurements.issubset(getattr(measurements, "columns", ())):
        return [], {}, "Таблица HKL measurements не содержит точные идентификаторы назначенных пиков."
    if not required_statistics.issubset(getattr(statistics, "columns", ())):
        return [], {}, "Таблица HKL pair_statistics не содержит итоговый θ-сдвиг регистрации."
    if getattr(measurements, "empty", True) or getattr(statistics, "empty", True):
        return [], {}, "В HKL-результате нет измеренных соответствий для карты Δθ."

    ny, nx = map(int, shape_yx)
    catalog = branch_catalog.loc[:, ["peak_id", "branch_id", "x", "y"]].copy(deep=False)
    if catalog["peak_id"].duplicated().any():
        duplicated = catalog.loc[catalog["peak_id"].duplicated(keep=False), "peak_id"].head(5).tolist()
        return [], {}, f"branch_catalog содержит неуникальные peak_id ({duplicated}); карту нельзя привязать однозначно."

    source_catalog = catalog.rename(columns={
        "peak_id": "source_peak_id", "branch_id": "catalog_source_branch_id",
        "x": "source_x", "y": "source_y",
    })
    partner_catalog = catalog.rename(columns={
        "peak_id": "matched_partner_peak_id", "branch_id": "catalog_partner_branch_id",
        "x": "partner_x", "y": "partner_y",
    })
    merged = measurements.merge(
        source_catalog, on="source_peak_id", how="inner", validate="many_to_one",
    ).merge(
        partner_catalog, on="matched_partner_peak_id", how="inner", validate="many_to_one",
    )
    stats = statistics.loc[:, ["pair_id", "theta_shift_final_arcsec"]].copy(deep=False)
    if stats["pair_id"].duplicated().any():
        return [], {}, "pair_statistics содержит повторные pair_id; θ-сдвиг неоднозначен."
    merged = merged.merge(stats, on="pair_id", how="inner", validate="many_to_one")
    if merged.empty:
        return [], {}, "Ни одно HKL-соответствие не удалось однозначно связать с branch_catalog и θ-сдвигом."

    # Prefer the stored signed measurement; for older compatible artifacts,
    # derive it only from the exact per-event θ values (never from proximity).
    if "delta_theta_signed_arcsec" in merged.columns:
        signed = pd.to_numeric(merged["delta_theta_signed_arcsec"], errors="coerce")
    elif {"theta_source_arcsec", "theta_partner_arcsec"}.issubset(merged.columns):
        signed = (
            pd.to_numeric(merged["theta_partner_arcsec"], errors="coerce")
            - pd.to_numeric(merged["theta_source_arcsec"], errors="coerce")
        )
    else:
        return [], {}, "В measurements отсутствуют знаковая Δθ или точные θ обеих назначенных вершин."
    shifts = pd.to_numeric(merged["theta_shift_final_arcsec"], errors="coerce")
    merged = merged.copy()
    merged["_aligned_delta_theta"] = signed.to_numpy(dtype=np.float64) + shifts.to_numpy(dtype=np.float64)

    layers: list[PreviewMapLayer] = []
    pair_summary: dict[str, Any] = {}
    group_name = "HKL · Δθ после совмещения"

    def aggregate_on_grid(frame: Any, x_column: str, y_column: str) -> tuple[np.ndarray | None, int, int]:
        x = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=np.float64)
        values = pd.to_numeric(frame["_aligned_delta_theta"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        valid &= (x == np.floor(x)) & (y == np.floor(y))
        valid &= (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        if not np.any(valid):
            return None, 0, 0
        xi = x[valid].astype(np.int64)
        yi = y[valid].astype(np.int64)
        vals = values[valid]
        sums = np.zeros((ny, nx), dtype=np.float64)
        counts = np.zeros((ny, nx), dtype=np.int64)
        np.add.at(sums, (yi, xi), vals)
        np.add.at(counts, (yi, xi), 1)
        result = np.full((ny, nx), np.nan, dtype=np.float64)
        occupied = counts > 0
        result[occupied] = sums[occupied] / counts[occupied]
        return result, int(vals.size), int(np.count_nonzero(counts > 1))

    for pair_id, frame in merged.groupby("pair_id", sort=False):
        source_ids = pd.to_numeric(frame["source_branch_id"], errors="coerce").to_numpy(dtype=np.float64)
        partner_ids = pd.to_numeric(frame["partner_branch_id"], errors="coerce").to_numpy(dtype=np.float64)
        source_ids = source_ids[np.isfinite(source_ids)]
        partner_ids = partner_ids[np.isfinite(partner_ids)]
        if source_ids.size == 0 or partner_ids.size == 0:
            continue
        source_value, partner_value = float(source_ids[0]), float(partner_ids[0])
        if not source_value.is_integer() or not partner_value.is_integer():
            continue
        source_branch = int(source_value)
        partner_branch = int(partner_value)
        valid_branch_rows = (
            pd.to_numeric(frame["source_branch_id"], errors="coerce").eq(source_branch)
            & pd.to_numeric(frame["partner_branch_id"], errors="coerce").eq(partner_branch)
            & pd.to_numeric(frame["catalog_source_branch_id"], errors="coerce").eq(source_branch)
            & pd.to_numeric(frame["catalog_partner_branch_id"], errors="coerce").eq(partner_branch)
        )
        pair_frame = frame.loc[valid_branch_rows]
        if pair_frame.empty:
            continue
        delta_source, n_source, duplicate_source = aggregate_on_grid(pair_frame, "source_x", "source_y")
        delta_partner, n_partner, duplicate_partner = aggregate_on_grid(pair_frame, "partner_x", "partner_y")
        source_key = f"hkl.delta_theta.pair_{source_branch}_{partner_branch}.source"
        partner_key = f"hkl.delta_theta.pair_{source_branch}_{partner_branch}.partner"
        formula = "остаток регистрации: (θпартнёр + θ-сдвиг) − θисходной; единицы — arcsec"
        if delta_source is not None:
            layers.append(PreviewMapLayer(
                key=source_key,
                label=f"Δθ · {source_branch}↔{partner_branch} · координаты {source_branch}",
                data_yx=delta_source, group=group_name, unit="″",
                description=(
                    f"{formula}. Значения нанесены в пиксели исходной ветви {source_branch}; "
                    f"N={n_source} точных назначений. "
                    f"При нескольких назначениях в одном пикселе показано среднее "
                    f"(таких пикселей: {duplicate_source})."
                ),
            ))
        if delta_partner is not None:
            layers.append(PreviewMapLayer(
                key=partner_key,
                label=f"Δθ · {source_branch}↔{partner_branch} · координаты {partner_branch}",
                data_yx=delta_partner, group=group_name, unit="″",
                description=(
                    f"{formula}. Значения нанесены в пиксели партнёрской ветви {partner_branch}; "
                    f"знак остатка по-прежнему определён относительно исходной ветви {source_branch}. "
                    f"N={n_partner} точных назначений. "
                    f"При нескольких назначениях в одном пикселе показано среднее "
                    f"(таких пикселей: {duplicate_partner})."
                ),
            ))
        pair_summary[str(pair_id)] = {
            "source_branch_id": source_branch, "partner_branch_id": partner_branch,
            "matched_events": int(len(pair_frame)),
            "source_occupied_pixels": int(np.isfinite(delta_source).sum()) if delta_source is not None else 0,
            "partner_occupied_pixels": int(np.isfinite(delta_partner).sum()) if delta_partner is not None else 0,
        }

    if not layers:
        return [], {}, "Не удалось построить карты: после проверки peak_id, сдвигов и координат не осталось валидных точек."
    return layers, pair_summary, None

def build_workflow_preview_state(
    node: Any,
    context: dict[str, Any],
    source_metadata: dict[str, Any] | None = None,
) -> WorkflowPreviewState:
    """Create node-scoped, grid-validated map and ROC catalogs from its artifacts."""
    scan = context.get("scan")
    if scan is None:
        raise ValueError("В узле отсутствует ScanArtifact")
    arrays = getattr(scan, "arrays", {})
    data = np.asarray(arrays.get("data_yxt"))
    if data.ndim != 3:
        raise ValueError(f"Scan data_yxt должен иметь форму Y×X×θ, получено {data.shape}")
    ny, nx, ntheta = data.shape
    theta = _validated_axis(arrays.get("theta"), ntheta, "θ")
    x_coords = _validated_axis(arrays.get("x_coords"), nx, "X")
    y_coords = _validated_axis(arrays.get("y_coords"), ny, "Y")
    shape = (ny, nx, ntheta)

    scan_valid = _valid_mask(arrays.get("valid_mask_yxt"), shape, "Scan valid_mask_yxt")
    analysis_valid_value = arrays.get("analysis_valid_mask_yx")
    analysis_valid = None
    if analysis_valid_value is not None:
        analysis_valid = np.asarray(analysis_valid_value, dtype=bool)
        if analysis_valid.shape != (ny, nx):
            raise ValueError(
                f"Scan analysis_valid_mask_yx имеет форму {analysis_valid.shape}, "
                f"ожидалась {(ny, nx)}"
            )
    spikes = context.get("spikes")
    peaks = context.get("peaks")
    fit = context.get("fit")
    branches = context.get("branches")

    # At the spike-removal node, the node's primary result is the cleaned
    # signal.  The detector mask (not the post-detection valid mask) is the
    # correct mask for that array: interpolated spike positions are valid
    # values in clean_data_yxt and must remain visible.
    display_data = data
    display_valid = scan_valid
    display_is_interpolated = False
    spike_arrays = getattr(spikes, "arrays", {}) if spikes is not None else {}
    clean_value = spike_arrays.get("clean_data_yxt") if spike_arrays else None
    if clean_value is not None and np.shape(clean_value) == shape:
        detector_mask_value = spike_arrays.get("detector_valid_mask_yxt")
        clean_mask = (
            _valid_mask(detector_mask_value, shape, "Spike detector_valid_mask_yxt")
            if detector_mask_value is not None else scan_valid
        )
        use_clean_as_primary = node.operation_id == "preprocess.spikes"
        if node.operation_id == "peaks.detect":
            peak_metadata = getattr(peaks, "metadata", {}) or {}
            use_clean_as_primary = bool(peak_metadata.get("used_interpolated_data"))
        if use_clean_as_primary:
            display_data = np.asarray(clean_value)
            display_valid = clean_mask
            display_is_interpolated = True

    signal_qualifier = " очищенного сигнала" if display_is_interpolated else " измеренного сигнала"
    layers: list[PreviewMapLayer] = []
    summed, mean = _sum_and_mean_maps(display_data, display_valid)
    _add_map(layers, key="scan.intensity_sum", label=f"Суммарная интенсивность{signal_qualifier} по θ", value=summed,
             shape_yx=(ny, nx), group="Основные данные", unit="counts",
             description=f"Сумма валидных counts{signal_qualifier} вдоль угловой оси θ.")
    _add_map(layers, key="scan.intensity_mean", label=f"Средняя интенсивность{signal_qualifier} по θ", value=mean,
             shape_yx=(ny, nx), group="Основные данные", unit="counts",
             description=f"Среднее по валидным отсчётам{signal_qualifier} вдоль θ.")
    # When configuring a peak stage from the Spike node, keep a measured-count
    # map available as well as the cleaned primary map. This lets the optional
    # real-pixel preview select a location on the same signal source that will
    # be passed to peaks.detect (measured or interpolated), without changing
    # the normal default map for Spike detection.
    if node.operation_id == "preprocess.spikes" and display_is_interpolated:
        measured_sum, _ = _sum_and_mean_maps(data, scan_valid)
        _add_map(
            layers, key="scan.measured_intensity_sum",
            label="Суммарная интенсивность измеренного сигнала по θ",
            value=measured_sum, shape_yx=(ny, nx), group="Основные данные",
            unit="counts",
            description="Сумма исходных измеренных counts Scan-входа вдоль θ, до интерполяции спайков.",
        )
    effective_valid = np.isfinite(display_data)
    if display_valid is not None:
        effective_valid &= display_valid
    if display_valid is not None or not np.isfinite(display_data).all():
        _add_map(layers, key="scan.valid_fraction", label="Доля валидных отсчётов", value=np.mean(effective_valid, axis=2),
                 shape_yx=(ny, nx), group="Контроль качества", unit="доля",
                 description="Доля валидных θ-отсчётов в каждом пространственном пикселе.")

    if spikes is not None:
        spike_arrays = getattr(spikes, "arrays", {})
        _add_map(layers, key="spikes.count", label="Число обнаруженных спайков", value=spike_arrays.get("spike_count_map"),
                 shape_yx=(ny, nx), group="Контроль качества", unit="событий",
                 description="Число точек, отмеченных алгоритмом Spike detection.")

    if peaks is not None:
        peak_arrays = getattr(peaks, "arrays", {})
        _add_map(layers, key="peaks.count", label="Число обнаруженных пиков", value=peak_arrays.get("peak_count_map"),
                 shape_yx=(ny, nx), group="Поиск пиков", unit="пиков",
                 description="Число обнаруженных пиков в пикселе по результату peaks.detect.")

    if fit is not None:
        fit_arrays = getattr(fit, "arrays", {})
        fit_tables = getattr(fit, "tables", {})
        pixel_catalog = fit_tables.get("pixel_fit_catalog") if isinstance(fit_tables, dict) else None
        for key, label, column, unit in (
            ("fit.n_detected", "Число обнаруженных компонент", "n_detected", "компонент"),
            ("fit.n_fitted", "Число успешно fitted-компонент", "n_fitted", "компонент"),
            ("fit.rmse", "RMSE подгонки", "rmse", "counts"),
            ("fit.redchi", "Reduced χ²", "redchi", "—"),
            ("fit.chisqr", "χ² подгонки", "chisqr", "—"),
            ("fit.p95_abs_residual", "95-й процентиль |остатка|", "p95_abs_residual", "counts"),
            ("fit.max_abs_residual", "Максимум |остатка|", "max_abs_residual", "counts"),
            ("fit.rmse_over_noise", "RMSE / оценка шума", "rmse_over_noise", "—"),
            ("fit.aic", "AIC", "aic", "—"),
            ("fit.bic", "BIC", "bic", "—"),
        ):
            _add_map(layers, key=key, label=label,
                     value=_pixel_metric_map(pixel_catalog, column, (ny, nx)),
                     shape_yx=(ny, nx), group="Качество fit" if key not in {"fit.n_detected", "fit.n_fitted"} else "Fit · число компонент",
                     unit=unit, description=f"Значение из pixel_fit_catalog.{column}.")
        if "n_fitted_map" in fit_arrays:
            _add_map(layers, key="fit.n_fitted_map", label="Число fitted-компонент (карта)", value=fit_arrays.get("n_fitted_map"),
                     shape_yx=(ny, nx), group="Fit · число компонент", unit="компонент")
        layers.extend(_component_parameter_maps(fit, (ny, nx)))
        fitted_catalog = fit_tables.get("fitted_peak_catalog") if isinstance(fit_tables, dict) else None
        if fitted_catalog is not None and not getattr(fitted_catalog, "empty", True) and "delta_theta" in fitted_catalog.columns:
            component_ids = sorted({int(v) for v in fitted_catalog["component_id"].dropna().tolist() if float(v).is_integer()}) if "component_id" in fitted_catalog else []
            for component_id in component_ids:
                subset = fitted_catalog.loc[fitted_catalog["component_id"] == component_id]
                values = _peak_catalog_map(subset, "delta_theta", (ny, nx), aggregate="mean")
                _add_map(layers, key=f"fit.derived.delta_theta.component{component_id}",
                         label=f"Смещение центра Δθ · компонента {component_id + 1}",
                         value=values, shape_yx=(ny, nx), group="Derived-параметры", unit="θ",
                         description="Разность θ_fit − θ_detected из fitted_peak_catalog; показывает сдвиг центра после подгонки.")
        residual_mask = _fit_valid_mask(fit, shape)
        layers.extend(_residual_summary_maps(fit_arrays.get("residual_cube"), residual_mask, (ny, nx)))

    # Spatial summaries from the peak catalogue are meaningful even when a
    # downstream fit node is not present in the selected branch.
    peak_catalog = _table(peaks, "peak_catalog")
    if peak_catalog is None and fit is not None:
        peak_catalog = _table(fit, "peak_catalog")
    if peaks is not None or peak_catalog is not None:
        if not any(layer.key == "peaks.count" for layer in layers):
            _add_map(layers, key="peaks.count", label="Число обнаруженных пиков",
                     value=_peak_catalog_map(peak_catalog, "peak_id", (ny, nx), aggregate="count"),
                     shape_yx=(ny, nx), group="Поиск пиков", unit="пиков",
                     description="Число строк peak_catalog на пространственный пиксель.")
        for key, label, column, aggregate, unit in (
            ("peaks.max_height", "Максимальная высота пика", "height", "max", "counts"),
            ("peaks.max_prominence", "Максимальная prominence", "prominence", "max", "counts"),
            ("peaks.mean_width", "Средняя ширина пиков", "width_theta", "mean", "θ"),
        ):
            _add_map(layers, key=key, label=label,
                     value=_peak_catalog_map(peak_catalog, column, (ny, nx), aggregate=aggregate),
                     shape_yx=(ny, nx), group="Поиск пиков", unit=unit,
                     description=f"Пространственная агрегация {aggregate} по найденным пикам из peak_catalog.{column}.")

    branch_catalog = _table(branches, "branch_catalog")
    if branch_catalog is not None and not getattr(branch_catalog, "empty", True):
        if {"x", "y", "branch_id"}.issubset(branch_catalog.columns):
            # A valid detector pixel with no branch is exactly zero branches,
            # not missing data. NaN is reserved for pixels outside the valid
            # analysis mask so the quantitative cbar has a truthful zero end.
            branch_counts = np.zeros((ny, nx), dtype=np.float64)
            if analysis_valid is not None:
                branch_counts[~analysis_valid] = np.nan
            else:
                valid_pixels = np.any(effective_valid, axis=2)
                branch_counts[~valid_pixels] = np.nan
            selection_mask = np.ones(len(branch_catalog), dtype=bool)
            if "analysis_included" in branch_catalog.columns:
                raw_included = branch_catalog["analysis_included"].to_numpy(dtype=object)
                selection_mask = np.asarray([
                    (bool(value) if isinstance(value, (bool, np.bool_))
                     else str(value).strip().lower() in {"true", "1", "yes", "y"})
                    for value in raw_included
                ], dtype=bool)
            selected_catalog = branch_catalog.loc[selection_mask]
            grouped = selected_catalog.groupby(["y", "x"], sort=False)["branch_id"].nunique()
            for (y_value, x_value), count in grouped.items():
                try:
                    x_idx, y_idx = int(x_value), int(y_value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if 0 <= x_idx < nx and 0 <= y_idx < ny:
                    branch_counts[y_idx, x_idx] = float(count)
            _add_map(
                layers, key="branches.count", label="Число ветвей в пикселе",
                value=branch_counts, shape_yx=(ny, nx), group="Дифракционные ветви",
                unit="ветвей", description=(
                    "Число различных branch_id, включённых в дальнейший анализ, "
                    "среди пиков данного пикселя. Исключённые по порогу ветви сохранены в каталоге."
                ),
            )

    peak_points = None
    peak_position_kind = "detected"
    branch_edges = _table(branches, "branch_edges") if branches is not None else None
    if branch_catalog is not None and not getattr(branch_catalog, "empty", True):
        peak_points = branch_catalog
        peak_position_kind = str(getattr(branches, "metadata", {}).get("position_kind", "detected"))
    elif fit is not None:
        try:
            from roc_mcs.peaks.branches import fitted_peak_positions
            peak_points = fitted_peak_positions(fit)
            peak_position_kind = "fitted"
        except (ValueError, KeyError, TypeError):
            # Fitting preview itself remains available even for older fit bundles
            # whose component provenance is insufficient for a safe point overlay.
            peak_points = None
            peak_position_kind = "fitted_unavailable"
    elif peaks is not None:
        peak_points = _table(peaks, "peak_catalog")
        peak_position_kind = "detected"

    curve_sources: list[PreviewCurveSource] = [PreviewCurveSource(
        key="scan.measured", label="Измеренные counts", data_yxt=data,
        theta=theta, valid_mask_yxt=scan_valid, group="Измеренный сигнал",
        description="Исходные counts текущего узла Scan-входа; значения не модифицируются отображением.",
    )]
    if spikes is not None:
        spike_arrays = getattr(spikes, "arrays", {})
        clean = spike_arrays.get("clean_data_yxt")
        if clean is not None and np.shape(clean) == shape:
            # Interpolated samples are valid display values.  valid_mask_yxt
            # intentionally excludes them for some downstream algorithms.
            clean_mask_value = spike_arrays.get("detector_valid_mask_yxt")
            clean_mask = _valid_mask(clean_mask_value, shape, "Spike detector_valid_mask_yxt") if clean_mask_value is not None else scan_valid
            curve_sources.append(PreviewCurveSource(
                key="spikes.interpolated", label="Интерполированная ROC после Spike detection",
                data_yxt=np.asarray(clean), theta=theta, valid_mask_yxt=clean_mask,
                group="Предобработанные сигналы",
                description="clean_data_yxt из Spike detection; обнаруженные спайки заменены алгоритмической интерполяцией.",
            ))

    if fit is not None:
        fit_arrays = getattr(fit, "arrays", {})
        fit_theta_value = fit_arrays.get("theta", theta)
        try:
            fit_theta = _validated_axis(fit_theta_value, ntheta, "θ fit")
        except ValueError:
            fit_theta = None
        fit_mask = _fit_valid_mask(fit, shape)
        for key, label, array_name, group, description in (
            ("fit.model", "Модельная ROC", "fit_cube", "Модель fit", "Вычисленная fit-модель для каждого пикселя."),
            ("fit.residual", "Остаток fit (измерение − модель)", "residual_cube", "Диагностика fit", "Сохранённый residual_cube; знак остатка соответствует соглашению fit-конвейера."),
        ):
            cube = fit_arrays.get(array_name)
            if cube is None or fit_theta is None or np.shape(cube) != shape or not np.isfinite(cube).any():
                continue
            curve_sources.append(PreviewCurveSource(
                key=key, label=label, data_yxt=np.asarray(cube), theta=fit_theta,
                valid_mask_yxt=fit_mask, group=group, description=description,
            ))

    metadata = dict(getattr(scan, "metadata", {}) or {})
    if spikes is not None:
        metadata.update(getattr(spikes, "metadata", {}) or {})
    if peaks is not None:
        metadata.update({f"peaks_{k}": v for k, v in (getattr(peaks, "metadata", {}) or {}).items()})
    if fit is not None:
        metadata.update({f"fit_{k}": v for k, v in (getattr(fit, "metadata", {}) or {}).items()})
    branch_correlation = context.get("branch_correlation")
    branch_correlation_preview = None
    if branch_correlation is not None:
        metadata["branch_correlation"] = dict(
            (getattr(branch_correlation, "metadata", {}) or {}).get("branch_correlation", {})
        )
        try:
            from roc_mcs.analysis.branch_correlation_preview import build_branch_correlation_preview

            branch_correlation_preview = build_branch_correlation_preview(
                branch_catalog, branch_correlation, x_coords, y_coords,
            )
            metadata["branch_correlation_preview"] = {
                "pair_count": len(branch_correlation_preview.pairs),
                "matched_link_count": branch_correlation_preview.total_matched,
                "candidate_link_count": branch_correlation_preview.total_candidates,
                "assignment_counts": dict(branch_correlation_preview.assignment_counts),
            }
        except Exception as exc:
            # The stored scientific result and the ordinary scan/branch preview
            # remain available even if a legacy or malformed correlation table
            # cannot be adapted into display geometry. Report the precise cause.
            metadata["branch_correlation_preview_warning"] = (
                f"3D-визуализация кросс-корреляции недоступна: {type(exc).__name__}: {exc}"
            )
    hkl_identification_preview = None
    hkl_bundle = context.get("hkl_analysis")
    if hkl_bundle is not None:
        try:
            from roc_mcs.analysis.hkl_preview import build_hkl_identification_preview

            hkl_identification_preview = build_hkl_identification_preview(hkl_bundle)
            metadata["hkl_identification"] = {
                "pair_count": int(len(hkl_identification_preview.pair_statistics)),
                "reflection_count": int(len(hkl_identification_preview.theoretical_reflections)),
                "measurement_count": int(len(hkl_identification_preview.measurements)),
                "crystal_name": hkl_identification_preview.metadata.get("crystal_name", ""),
            }
            try:
                alignment_layers, alignment_summary, alignment_warning = _hkl_alignment_map_layers(
                    hkl_identification_preview, branch_catalog, (ny, nx),
                )
                layers.extend(alignment_layers)
                if alignment_layers:
                    metadata["hkl_alignment_maps"] = {
                        "layer_count": len(alignment_layers),
                        "pairs": alignment_summary,
                        "definition": "(theta_partner + theta_shift_final_arcsec) - theta_source",
                        "unit": "arcsec",
                        "aggregation": "mean_per_reference_pixel",
                    }
                elif alignment_warning:
                    metadata["hkl_alignment_map_warning"] = alignment_warning
            except Exception as exc:
                # A spatial-map adapter failure must not hide valid HKL tables.
                metadata["hkl_alignment_map_warning"] = (
                    f"Карта выровненной Δθ недоступна: {type(exc).__name__}: {exc}"
                )
        except Exception as exc:
            # The scientific HDF5 artifact remains available for explicit
            # inspection even if a future schema cannot be rendered.
            metadata["hkl_identification_preview_warning"] = (
                f"Панель HKL недоступна: {type(exc).__name__}: {exc}"
            )
    metadata.update(dict(source_metadata or {}))
    metadata["workflow_node_id"] = str(node.id)
    metadata["workflow_operation_id"] = str(node.operation_id)

    if not layers:
        raise ValueError("Для узла не удалось построить ни одной совместимой карты")
    if not curve_sources:
        raise ValueError("Для узла не удалось построить ни одного совместимого ROC-источника")

    default_map = "scan.intensity_sum"
    if node.operation_id == "preprocess.spikes" and any(layer.key == "spikes.count" for layer in layers):
        default_map = "spikes.count"
    elif node.operation_id == "peaks.detect" and any(layer.key == "peaks.count" for layer in layers):
        default_map = "peaks.count"
    elif node.operation_id == "fit.scan":
        default_map = next((key for key in ("fit.rmse", "fit.redchi", "fit.n_fitted", "fit.n_fitted_map") if any(layer.key == key for layer in layers)), default_map)
    elif node.operation_id == "theory.identify_hkl":
        default_map = next(
            (layer.key for layer in layers if layer.key.startswith("hkl.delta_theta.")),
            "branches.count" if any(layer.key == "branches.count" for layer in layers) else default_map,
        )
    elif node.operation_id in {"peaks.branches", "peaks.branch_correlation"} and any(
        layer.key == "branches.count" for layer in layers
    ):
        default_map = "branches.count"
    default_curve = "scan.measured"
    if node.operation_id == "preprocess.spikes" and any(source.key == "spikes.interpolated" for source in curve_sources):
        default_curve = "spikes.interpolated"
    elif node.operation_id == "fit.scan" and any(source.key == "fit.model" for source in curve_sources):
        default_curve = "fit.model"
    elif node.operation_id == "peaks.detect":
        peak_meta = getattr(peaks, "metadata", {}) or {}
        if bool(peak_meta.get("used_interpolated_data")) and any(source.key == "spikes.interpolated" for source in curve_sources):
            default_curve = "spikes.interpolated"

    return WorkflowPreviewState(
        node_id=str(node.id), operation_id=str(node.operation_id), label=str(node.label),
        data_yxt=display_data, theta=theta, x_coords=x_coords, y_coords=y_coords,
        valid_mask_yxt=display_valid, metadata=metadata,
        map_layers=tuple(layers), curve_sources=tuple(curve_sources),
        default_map_key=default_map, default_curve_key=default_curve,
        summary=str(getattr(node, "summary", "") or ""),
        analysis_valid_mask_yx=analysis_valid,
        peak_points=peak_points,
        peak_position_kind=peak_position_kind,
        branch_edges=branch_edges,
        branch_correlation_preview=branch_correlation_preview,
        hkl_identification_preview=hkl_identification_preview,
    )
