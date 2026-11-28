"""Display-only branch catalog helpers for ROI Explorer.

Branch identity is the connected-component ``branch_id`` created by the
scientific branch operation.  This module never changes workflow artifacts.
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

_BRANCH_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_STATE_SCHEMA_VERSION = 2
_MAX_STORED_CATALOG_PALETTES = 32

_PARAMETER_INFO: dict[str, tuple[str, str]] = {
    # The branch operation records the scan's angular coordinate in arcseconds.
    # Keep units explicit in the map menu/colorbar instead of using a bare θ.
    "theta": ("Положение центра θ", "″"),
    "theta0": ("Положение центра θ₀", "″"),
    "theta_fit": ("Fitted-центр θ", "″"),
    "center_theta": ("Центр θ", "″"),
    "theta_detected": ("Обнаруженный центр θ", "″"),
    "delta_theta": ("Сдвиг центра Δθ", "″"),
    "S": ("Площадь пика S", "counts·″"),
    "H": ("Высота H", "counts"),
    "height": ("Высота пика", "counts"),
    "prominence": ("Prominence", "counts"),
    "width_theta": ("Ширина пика", "″"),
    "sigma": ("Ширина σ", "″"),
    "gamma": ("Ширина γ", "″"),
    "eta": ("Параметр смешения η", "—"),
    "Delta": ("Амплитуда углового сдвига Δθₚ", "″"),
    "beta_Gl": ("Gaussian-ширина слева β_Gl", "″"),
    "beta_Cl": ("Lorentz-ширина слева β_Cl", "″"),
    "beta_Gr": ("Gaussian-ширина справа β_Gr", "″"),
    "beta_Cr": ("Lorentz-ширина справа β_Cr", "″"),
    "lam": ("Параметр экспоненциального хвоста λ", "″⁻¹"),
}

_FORBIDDEN_PARAMETER_COLUMNS = {
    "peak_id", "branch_id", "component_id", "x", "y", "theta_index",
    "analysis_included", "branch_position_kind", "event_id", "pixel_id",
}


def _numeric(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)


def _boolean(values: Any) -> np.ndarray:
    """Parse persisted booleans without treating the string 'False' as true."""
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif value is None or (isinstance(value, float) and not np.isfinite(value)):
            result.append(False)
        else:
            result.append(str(value).strip().lower() in {"true", "1", "yes", "y", "да"})
    return np.asarray(result, dtype=bool)


def default_branch_color(branch_id: int) -> str:
    """Return a stable, high-saturation categorical color for one branch ID."""
    # The golden-angle hue sequence avoids adjacent branch IDs receiving
    # neighbouring colours. HLS keeps lightness fixed for readability in both
    # light and dark application themes.
    hue = (int(branch_id) * 0.6180339887498949 + 0.08) % 1.0
    red, green, blue = colorsys.hls_to_rgb(hue, 0.515, 0.72)
    channels = [int(round(channel * 255.0)) for channel in (red, green, blue)]
    return "#{:02X}{:02X}{:02X}".format(*channels)


def _unused_default_branch_color(branch_id: int, used: set[str]) -> str:
    """Return a stable generated color that is not already assigned."""
    golden_angle = 0.6180339887498949
    for attempt in range(512):
        hue = (int(branch_id) * golden_angle + 0.08 + attempt * 0.071) % 1.0
        red, green, blue = colorsys.hls_to_rgb(hue, 0.515, 0.72)
        channels = [int(round(channel * 255.0)) for channel in (red, green, blue)]
        candidate = "#{:02X}{:02X}{:02X}".format(*channels)
        if candidate not in used:
            return candidate
    raise RuntimeError("Не удалось подобрать уникальный цвет ветви")


def normalize_branch_colors(
    branch_ids: Iterable[int],
    overrides: Mapping[Any, Any] | None = None,
) -> dict[int, str]:
    """Build a complete palette with one distinct color per branch.

    Valid saved overrides are preserved in ascending branch-ID order. A
    duplicate legacy override is replaced by a deterministic generated color,
    so old sidecars cannot reintroduce ambiguous branch identities.
    """
    supplied: dict[int, str] = {}
    for raw_id, raw_color in (overrides or {}).items():
        try:
            branch_id = int(raw_id)
            color = str(raw_color).strip()
        except (TypeError, ValueError, OverflowError):
            continue
        if branch_id >= 0 and _BRANCH_COLOR_RE.fullmatch(color):
            supplied[branch_id] = color.upper()

    valid_ids = sorted({int(value) for value in branch_ids if int(value) >= 0})

    # Reserve valid, distinct user overrides first. Generated defaults for
    # branches without an override must not steal a saved color from a later ID.
    normalized: dict[int, str] = {}
    used: set[str] = set()
    for branch_id in valid_ids:
        candidate = supplied.get(branch_id)
        if candidate is not None and candidate not in used:
            normalized[branch_id] = candidate
            used.add(candidate)

    for branch_id in valid_ids:
        if branch_id in normalized:
            continue
        candidate = _unused_default_branch_color(branch_id, used)
        normalized[branch_id] = candidate
        used.add(candidate)

    return {branch_id: normalized[branch_id] for branch_id in valid_ids}


def _valid_branch_ids(catalog: pd.DataFrame) -> np.ndarray:
    if catalog is None or catalog.empty or "branch_id" not in catalog.columns:
        return np.empty(0, dtype=np.int64)
    values = _numeric(catalog["branch_id"])
    valid = np.isfinite(values) & (values >= 0) & (values == np.rint(values))
    return np.rint(values[valid]).astype(np.int64, copy=False)


def summarize_branch_catalog(catalog: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Summarize event count, occupied detector pixels and theta range per branch."""
    if catalog is None or catalog.empty or not {"branch_id", "x", "y"}.issubset(catalog.columns):
        return []
    branch_ids = _numeric(catalog["branch_id"])
    x_values = _numeric(catalog["x"])
    y_values = _numeric(catalog["y"])
    valid = (
        np.isfinite(branch_ids) & (branch_ids >= 0) & (branch_ids == np.rint(branch_ids))
        & np.isfinite(x_values) & np.isfinite(y_values)
    )
    if not np.any(valid):
        return []
    work = pd.DataFrame({
        "branch_id": np.rint(branch_ids[valid]).astype(np.int64),
        "x": np.rint(x_values[valid]).astype(np.int64),
        "y": np.rint(y_values[valid]).astype(np.int64),
    })
    theta_column = "theta" if "theta" in catalog.columns else (
        "theta_detected" if "theta_detected" in catalog.columns else None
    )
    if theta_column is not None:
        work["theta"] = _numeric(catalog.loc[valid, theta_column])
    else:
        work["theta"] = np.nan
    if "analysis_included" in catalog.columns:
        work["included"] = _boolean(catalog.loc[valid, "analysis_included"])
    else:
        work["included"] = True

    summaries: list[dict[str, Any]] = []
    for branch_id, group in work.groupby("branch_id", sort=True):
        theta = group["theta"].to_numpy(dtype=np.float64, copy=False)
        finite_theta = theta[np.isfinite(theta)]
        summaries.append({
            "branch_id": int(branch_id),
            "event_count": int(len(group)),
            "occupied_pixels": int(group[["x", "y"]].drop_duplicates().shape[0]),
            "median_theta": float(np.median(finite_theta)) if finite_theta.size else None,
            "theta_min": float(np.min(finite_theta)) if finite_theta.size else None,
            "theta_max": float(np.max(finite_theta)) if finite_theta.size else None,
            "analysis_included": bool(group["included"].all()),
        })
    return summaries


def branch_catalog_fingerprint(catalog: pd.DataFrame | None) -> str:
    """Hash stable event identity and branch assignment for display-sidecar safety."""
    if catalog is None or catalog.empty:
        raise ValueError("Нельзя вычислить fingerprint пустого каталога ветвей")
    required = {"branch_id", "x", "y"}
    if not required.issubset(catalog.columns):
        raise ValueError(f"Каталог ветвей не содержит {sorted(required - set(catalog.columns))}")
    # A display palette belongs to event membership, not to mutable fitted
    # values or the size-inclusion policy. Excluding theta and
    # analysis_included preserves a user's colors when they change only the
    # minimum branch-size threshold or refit parameters without changing which
    # peak_ids belong to each branch. A changed partition still changes the
    # peak_id -> branch_id mapping and therefore receives a different key.
    columns = [
        name for name in ("peak_id", "branch_id", "x", "y")
        if name in catalog.columns
    ]
    frame = catalog.loc[:, columns].copy()
    sort_columns = ["peak_id"] if "peak_id" in frame.columns else [
        name for name in ("branch_id", "y", "x") if name in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort", na_position="last")
    digest = hashlib.sha256()
    digest.update(b"roc-mcs-branch-display-membership-v2\0")
    digest.update(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(np.asarray(frame.shape, dtype=np.int64).tobytes())
    hashed = pd.util.hash_pandas_object(frame, index=False, categorize=True)
    digest.update(hashed.to_numpy(dtype=np.uint64, copy=False).tobytes())
    return digest.hexdigest()


def available_branch_parameters(
    catalog: pd.DataFrame | None,
    declared_parameters: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Return physically meaningful numeric branch parameters; never component_id."""
    if catalog is None or catalog.empty:
        return []
    candidates: list[str] = []
    for name in (
        "theta", "theta_detected", "theta0", "theta_fit", "center_theta",
        "delta_theta", "S", "H", "height", "prominence", "width_theta",
        "eta", "sigma", "gamma", "Delta", "beta_Gl", "beta_Cl",
        "beta_Gr", "beta_Cr", "lam",
    ):
        if name in catalog.columns and name not in candidates:
            candidates.append(name)
    for name in declared_parameters:
        value = str(name)
        if value in catalog.columns and value not in candidates:
            candidates.append(value)
    fields: list[dict[str, str]] = []
    for name in candidates:
        if name in _FORBIDDEN_PARAMETER_COLUMNS or name.endswith("_error"):
            continue
        try:
            values = _numeric(catalog[name])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).any():
            continue
        label, unit = _PARAMETER_INFO.get(name, (str(name), ""))
        fields.append({"key": name, "label": label, "unit": unit})
    return fields


def build_branch_parameter_map(
    catalog: pd.DataFrame,
    branch_id: int,
    parameter: str,
    shape_yx: tuple[int, int],
) -> np.ndarray:
    """Rasterize one branch parameter; repeated events in one pixel are averaged.

    ``component_id`` is intentionally not an input to this operation: it is a
    per-pixel fit ordering, not a persistent track identity.
    """
    if catalog is None or catalog.empty:
        raise ValueError("Каталог ветвей пуст")
    if parameter in _FORBIDDEN_PARAMETER_COLUMNS or parameter not in catalog.columns:
        raise ValueError(f"Параметр {parameter!r} недоступен для карты ветви")
    ny, nx = map(int, shape_yx)
    if ny < 1 or nx < 1:
        raise ValueError("Размер карты должен быть положительным")
    required = {"branch_id", "x", "y", parameter}
    if not required.issubset(catalog.columns):
        raise ValueError(f"Каталог не содержит {sorted(required - set(catalog.columns))}")

    ids = _numeric(catalog["branch_id"])
    x_values = _numeric(catalog["x"])
    y_values = _numeric(catalog["y"])
    values = _numeric(catalog[parameter])
    rounded_x = np.rint(np.where(np.isfinite(x_values), x_values, 0.0))
    rounded_y = np.rint(np.where(np.isfinite(y_values), y_values, 0.0))
    valid = (
        np.isfinite(ids) & (ids == int(branch_id))
        & np.isfinite(x_values) & np.isfinite(y_values)
        & (np.abs(x_values - rounded_x) <= 1e-6)
        & (np.abs(y_values - rounded_y) <= 1e-6)
        & (rounded_x >= 0) & (rounded_x < nx)
        & (rounded_y >= 0) & (rounded_y < ny)
        & np.isfinite(values)
    )
    result = np.full((ny, nx), np.nan, dtype=np.float64)
    if not np.any(valid):
        return result
    flat = rounded_y[valid].astype(np.int64) * nx + rounded_x[valid].astype(np.int64)
    sums = np.bincount(flat, weights=values[valid], minlength=ny * nx)
    counts = np.bincount(flat, minlength=ny * nx)
    occupied = counts > 0
    flat_result = result.reshape(-1)
    flat_result[occupied] = sums[occupied] / counts[occupied]
    return result


def branch_footprint_layers(
    catalog: pd.DataFrame,
    branch_ids: Iterable[int],
    shape_yx: tuple[int, int],
    *,
    smooth_sigma: float = 1.2,
) -> Iterator[tuple[int, np.ndarray, float | None]]:
    """Yield normalized detector-occupancy masks ordered by median theta.

    This is a detector-plane XY projection of occupied event pixels, not an
    intensity map or a reconstruction of a grain's 3D physical shape. Gaussian
    smoothing is display-only; catalogue rows and coordinates are untouched.

    The implementation keeps one raster mask alive at a time and uses sorted
    NumPy slices rather than one pandas DataFrame per branch. This bounds the
    additional memory when a user selects many branches.
    """
    if catalog is None or catalog.empty or not {"branch_id", "x", "y"}.issubset(catalog.columns):
        return
    ny, nx = map(int, shape_yx)
    if ny < 1 or nx < 1:
        return
    requested_values: set[int] = set()
    for value in branch_ids:
        if isinstance(value, (bool, np.bool_)):
            continue
        try:
            branch_id = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= branch_id <= np.iinfo(np.int64).max:
            requested_values.add(branch_id)
    requested = sorted(requested_values)
    if not requested:
        return
    sigma = float(smooth_sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("smooth_sigma должен быть конечным неотрицательным числом")

    raw_ids = _numeric(catalog["branch_id"])
    valid_ids = (
        np.isfinite(raw_ids) & (raw_ids >= 0.0)
        & (raw_ids == np.rint(raw_ids))
        & (raw_ids <= np.iinfo(np.int64).max)
    )
    if not np.any(valid_ids):
        return
    integer_ids = np.zeros(raw_ids.shape, dtype=np.int64)
    integer_ids[valid_ids] = np.rint(raw_ids[valid_ids]).astype(np.int64)
    selected_rows = valid_ids & np.isin(integer_ids, np.asarray(requested, dtype=np.int64))
    if not np.any(selected_rows):
        return

    branch_values = integer_ids[selected_rows]
    x_values = _numeric(catalog.loc[selected_rows, "x"])
    y_values = _numeric(catalog.loc[selected_rows, "y"])
    theta_name = "theta" if "theta" in catalog.columns else (
        "theta_detected" if "theta_detected" in catalog.columns else None
    )
    theta_values = (
        _numeric(catalog.loc[selected_rows, theta_name])
        if theta_name is not None else np.full(branch_values.shape, np.nan, dtype=np.float64)
    )

    order = np.argsort(branch_values, kind="stable")
    branch_values = branch_values[order]
    x_values = x_values[order]
    y_values = y_values[order]
    theta_values = theta_values[order]
    starts = np.r_[0, np.flatnonzero(branch_values[1:] != branch_values[:-1]) + 1]
    ends = np.r_[starts[1:], branch_values.size]

    ordered_groups: list[tuple[float, int, int, int]] = []
    for start, end in zip(starts, ends):
        values = theta_values[start:end]
        finite_theta = values[np.isfinite(values)]
        median = float(np.median(finite_theta)) if finite_theta.size else math.inf
        ordered_groups.append((median, int(branch_values[start]), int(start), int(end)))
    ordered_groups.sort(key=lambda item: (item[0], item[1]))

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:  # pragma: no cover - scipy is a core runtime dependency, fallback is safe.
        gaussian_filter = None

    for median, branch_id, start, end in ordered_groups:
        x = x_values[start:end]
        y = y_values[start:end]
        rounded_x = np.rint(np.where(np.isfinite(x), x, 0.0))
        rounded_y = np.rint(np.where(np.isfinite(y), y, 0.0))
        valid = (
            np.isfinite(x) & np.isfinite(y)
            & (np.abs(x - rounded_x) <= 1e-6)
            & (np.abs(y - rounded_y) <= 1e-6)
            & (rounded_x >= 0) & (rounded_x < nx)
            & (rounded_y >= 0) & (rounded_y < ny)
        )
        if not np.any(valid):
            continue
        mask = np.zeros((ny, nx), dtype=np.float32)
        mask[rounded_y[valid].astype(np.int64), rounded_x[valid].astype(np.int64)] = 1.0
        if sigma > 0.0 and gaussian_filter is not None:
            smoothed = gaussian_filter(mask, sigma=sigma).astype(np.float32, copy=False)
            maximum = float(np.max(smoothed))
            if maximum > 0.0:
                smoothed /= maximum
        else:
            smoothed = mask
        yield branch_id, smoothed, (None if not np.isfinite(median) else median)


def composite_footprint_rgba_inplace(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: str,
    *,
    opacity: float = 0.45,
) -> bool:
    """Source-over composite one normalized footprint into an RGBA canvas.

    ``canvas`` is modified in place and must be writable float32 RGBA. Processing
    one branch at a time avoids allocating a full detector-sized RGBA image for
    every selected branch. Painter's order is therefore explicit and bounded.
    """
    target = np.asarray(canvas)
    source_mask = np.asarray(mask, dtype=np.float32)
    if target.ndim != 3 or target.shape[-1] != 4:
        raise ValueError("canvas должен иметь форму (ny, nx, 4)")
    if target.dtype != np.float32 or not target.flags.writeable:
        raise ValueError("canvas должен быть изменяемым массивом float32")
    if source_mask.shape != target.shape[:2]:
        raise ValueError("Форма mask не совпадает с canvas")
    try:
        opacity_value = float(opacity)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("opacity должна быть числом от 0 до 1") from exc
    if not np.isfinite(opacity_value) or not 0.0 <= opacity_value <= 1.0:
        raise ValueError("opacity должна быть числом от 0 до 1")
    if not _BRANCH_COLOR_RE.fullmatch(str(color).strip()):
        raise ValueError(f"Некорректный RGB-цвет ветви: {color!r}")
    if opacity_value == 0.0:
        return False

    finite_mask = np.nan_to_num(source_mask, nan=0.0, posinf=0.0, neginf=0.0)
    source_alpha = np.clip(finite_mask, 0.0, 1.0) * np.float32(opacity_value)
    active = source_alpha > 0.0
    if not np.any(active):
        return False

    rgb = np.asarray([
        int(str(color).strip()[offset:offset + 2], 16) / 255.0
        for offset in (1, 3, 5)
    ], dtype=np.float32)
    destination = target[active]
    if not np.isfinite(destination).all():
        raise ValueError("canvas содержит NaN или бесконечные значения в области наложения")
    src_a = source_alpha[active]
    dst_a = destination[:, 3]
    out_a = src_a + dst_a * (1.0 - src_a)
    numerator = rgb[None, :] * src_a[:, None] + destination[:, :3] * (
        dst_a * (1.0 - src_a)
    )[:, None]
    destination[:, :3] = np.divide(
        numerator, out_a[:, None], out=np.zeros_like(numerator),
        where=out_a[:, None] > 0.0,
    )
    destination[:, 3] = out_a
    target[active] = destination
    return True


def _normalize_saved_colors(colors: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    if not isinstance(colors, dict):
        return result
    for key, value in colors.items():
        try:
            branch_id = int(key)
            color = str(value).strip()
        except (TypeError, ValueError, OverflowError):
            continue
        if branch_id >= 0 and _BRANCH_COLOR_RE.fullmatch(color):
            result[branch_id] = color.upper()
    return result


def load_branch_display_colors(
    path: str | Path,
    fingerprint: str,
) -> dict[int, str]:
    """Load display palette for one exact catalogue, with v1 migration support."""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    version = payload.get("schema_version")
    if version == 1:
        # The former format held exactly one catalogue fingerprint. Reading it
        # remains safe; the next save migrates it into the multi-catalog schema.
        if str(payload.get("catalog_fingerprint", "")) != str(fingerprint):
            return {}
        return _normalize_saved_colors(payload.get("colors", {}))
    if version != _STATE_SCHEMA_VERSION:
        return {}
    catalogs = payload.get("catalogs", {})
    if not isinstance(catalogs, dict):
        return {}
    record = catalogs.get(str(fingerprint), {})
    if not isinstance(record, dict):
        return {}
    return _normalize_saved_colors(record.get("colors", {}))


def save_branch_display_colors(
    path: str | Path,
    fingerprint: str,
    colors: Mapping[Any, Any],
) -> None:
    """Atomically persist up to 32 display palettes without replacing others.

    A new branch assignment (for example after changing theta-gate) receives a
    distinct fingerprint. Keeping a bounded history lets users revisit older
    workflow nodes without losing their previously curated display colours.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, str] = {}
    for raw_id, raw_color in colors.items():
        try:
            branch_id = int(raw_id)
            color = str(raw_color).strip()
        except (TypeError, ValueError, OverflowError):
            continue
        if branch_id >= 0 and _BRANCH_COLOR_RE.fullmatch(color):
            normalized[str(branch_id)] = color.upper()

    catalogs: dict[str, dict[str, Any]] = {}
    try:
        old_payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        old_payload = None
    if isinstance(old_payload, dict):
        if old_payload.get("schema_version") == _STATE_SCHEMA_VERSION:
            old_catalogs = old_payload.get("catalogs", {})
            if isinstance(old_catalogs, dict):
                for key, value in old_catalogs.items():
                    if isinstance(value, dict):
                        catalogs[str(key)] = {
                            "colors": {
                                str(branch_id): color
                                for branch_id, color in sorted(
                                    _normalize_saved_colors(value.get("colors", {})).items()
                                )
                            }
                        }
        elif old_payload.get("schema_version") == 1:
            old_fingerprint = str(old_payload.get("catalog_fingerprint", ""))
            old_colors = _normalize_saved_colors(old_payload.get("colors", {}))
            if old_fingerprint and old_colors:
                catalogs[old_fingerprint] = {
                    "colors": {str(key): value for key, value in sorted(old_colors.items())}
                }

    key = str(fingerprint)
    catalogs.pop(key, None)  # Reinsert to update recency deterministically.
    catalogs[key] = {
        "colors": dict(sorted(normalized.items(), key=lambda item: int(item[0])))
    }
    while len(catalogs) > _MAX_STORED_CATALOG_PALETTES:
        catalogs.pop(next(iter(catalogs)))

    payload = {
        "schema_version": _STATE_SCHEMA_VERSION,
        "catalogs": catalogs,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
