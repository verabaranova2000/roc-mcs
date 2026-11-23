"""Spatially linked peak-event branches on a detector lattice.

A graph node is one peak event (one row in a peak catalog). Edges are built
only between 4-connected detector pixels and use one-to-one minimum-|Δθ|
matching. Connected components define branch IDs.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _validate_catalog(peak_catalog):
    if peak_catalog is None:
        raise ValueError("peak_catalog не задан")
    required = {"peak_id", "x", "y", "theta"}
    missing = required.difference(getattr(peak_catalog, "columns", ()))
    if missing:
        raise ValueError(f"peak_catalog не содержит обязательные столбцы: {sorted(missing)}")
    if peak_catalog.empty:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=float)
    ids_raw = np.asarray(peak_catalog["peak_id"])
    x_raw = np.asarray(peak_catalog["x"], dtype=float)
    y_raw = np.asarray(peak_catalog["y"], dtype=float)
    theta = np.asarray(peak_catalog["theta"], dtype=float)
    try:
        ids_float = np.asarray(ids_raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("peak_id должен содержать целые идентификаторы") from exc
    if not (np.isfinite(ids_float).all() and np.all(ids_float == np.rint(ids_float))):
        raise ValueError("peak_id должен содержать конечные целые идентификаторы")
    if np.unique(ids_float).size != ids_float.size:
        raise ValueError("peak_id должен быть уникальным для каждой строки каталога")
    if not (np.isfinite(x_raw).all() and np.isfinite(y_raw).all() and np.isfinite(theta).all()):
        raise ValueError("Координаты x/y и theta в peak_catalog должны быть конечными")
    if not (np.all(x_raw == np.rint(x_raw)) and np.all(y_raw == np.rint(y_raw))):
        raise ValueError("Координаты x/y в peak_catalog должны быть целочисленными индексами view-grid")
    return ids_float.astype(np.int64), x_raw.astype(np.int64), y_raw.astype(np.int64), theta


def _validate_theta_gate(theta_gate: float, *, allow_infinite: bool = False) -> float:
    try:
        gate = float(theta_gate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("theta_gate должен быть положительным числом в единицах оси θ") from exc
    if np.isnan(gate) or gate <= 0 or (not allow_infinite and not np.isfinite(gate)):
        raise ValueError("theta_gate должен быть конечным положительным числом")
    return gate


def match_neighbor_peaks(peaks_a, peaks_b, theta_gate=60.0):
    """One-to-one matching between events in adjacent pixels."""
    gate = _validate_theta_gate(theta_gate, allow_infinite=True)
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return [], list(peaks_a), list(peaks_b)
    theta_a = np.asarray([float(p["theta"]) for p in peaks_a], dtype=float)
    theta_b = np.asarray([float(p["theta"]) for p in peaks_b], dtype=float)
    if not np.isfinite(theta_a).all() or not np.isfinite(theta_b).all():
        raise ValueError("theta соседних пиков должен быть конечным")
    cost = np.abs(theta_a[:, None] - theta_b[None, :])
    rows, cols = linear_sum_assignment(cost)
    matches = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for i, j in zip(rows, cols):
        delta = float(cost[i, j])
        if delta <= gate:
            matches.append((int(peaks_a[i]["peak_id"]), int(peaks_b[j]["peak_id"]), delta))
            matched_a.add(int(i))
            matched_b.add(int(j))
    unmatched_a = [p for i, p in enumerate(peaks_a) if i not in matched_a]
    unmatched_b = [p for j, p in enumerate(peaks_b) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


def _pixel_peak_lookup(peak_catalog):
    ids, xs, ys, thetas = _validate_catalog(peak_catalog)
    lookup: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for peak_id, x, y, theta in zip(ids, xs, ys, thetas):
        lookup[(int(x), int(y))].append({"peak_id": int(peak_id), "theta": float(theta)})
    for peaks in lookup.values():
        peaks.sort(key=lambda item: (item["theta"], item["peak_id"]))
    return dict(lookup)


def build_peak_branch_graph(peak_catalog, theta_gate=60.0):
    """Build edges between one-to-one matched peaks in right/down neighbours."""
    gate = _validate_theta_gate(theta_gate)
    lookup = _pixel_peak_lookup(peak_catalog)
    edges: list[tuple[int, int]] = []
    deltas: list[float] = []
    for (x, y), peaks_here in lookup.items():
        for neighbour in ((x + 1, y), (x, y + 1)):
            peaks_next = lookup.get(neighbour)
            if not peaks_next:
                continue
            matches, _, _ = match_neighbor_peaks(peaks_here, peaks_next, theta_gate=gate)
            for peak_a, peak_b, delta in matches:
                edges.append((peak_a, peak_b))
                deltas.append(delta)
    return edges, np.asarray(deltas, dtype=np.float64)


def label_peak_branches(peak_catalog, edges):
    """Return ``(n_components, labels)`` in the original catalog row order.

    Isolated events are one-node connected components; therefore a graph with
    no edges has exactly one branch per peak event, not zero branches.
    """
    ids, _, _, _ = _validate_catalog(peak_catalog)
    n_events = int(ids.size)
    if n_events == 0:
        return 0, np.asarray([], dtype=np.int32)
    id_to_row = {int(peak_id): row for row, peak_id in enumerate(ids)}
    rows: list[int] = []
    cols: list[int] = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Некорректное ребро peak graph: {edge!r}")
        a, b = int(edge[0]), int(edge[1])
        if a not in id_to_row or b not in id_to_row:
            raise ValueError(f"Ребро ссылается на неизвестный peak_id: {(a, b)}")
        if a == b:
            continue
        ia, ib = id_to_row[a], id_to_row[b]
        rows.extend((ia, ib))
        cols.extend((ib, ia))
    if not rows:
        return n_events, np.arange(n_events, dtype=np.int32)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(n_events, n_events),
    ).tocsr()
    n_branches, labels = connected_components(graph, directed=False, return_labels=True)
    return int(n_branches), np.asarray(labels, dtype=np.int32)


def summarize_peak_spacing(
    peak_catalog, *, reference_theta_gate: float = 60.0
) -> dict[str, Any]:
    """Per-scan diagnostics for selecting a theta gate.

    ``q90_neighbor_delta`` is the 90th percentile of ``edge_deltas`` produced
    by the same one-to-one matcher at the explicit reference gate. The default
    of 60 arcsec reproduces the diagnostic used in the supplied notebook.
    Internal gaps are computed within each pixel, then reduced to one median
    per pixel before cross-pixel quantiles are calculated.
    """
    ids, xs, ys, theta = _validate_catalog(peak_catalog)
    reference_gate = _validate_theta_gate(reference_theta_gate)
    lookup = _pixel_peak_lookup(peak_catalog)
    internal_by_n: dict[int, list[float]] = defaultdict(list)
    for peaks in lookup.values():
        if len(peaks) < 2:
            continue
        gaps = np.diff(np.sort([item["theta"] for item in peaks]))
        finite_gaps = gaps[np.isfinite(gaps)]
        if finite_gaps.size:
            # Zero is a real zero-separation pair, not a missing value. Keeping
            # it prevents the lower-tail diagnostic from overstating separation.
            internal_by_n[len(peaks)].append(float(np.median(finite_gaps)))

    rows = []
    all_internal = []
    for n in (2, 3, 4, 5):
        values = np.asarray(internal_by_n.get(n, ()), dtype=float)
        if not values.size:
            continue
        all_internal.extend(values.tolist())
        rows.append({
            "n_peaks": int(n),
            "pixel_count": int(values.size),
            "median_gap": float(np.median(values)),
            "q25_gap": float(np.percentile(values, 25)),
            "q75_gap": float(np.percentile(values, 75)),
            "min_pixel_median_gap": float(np.min(values)),
        })

    neighbour_deltas: list[float] = []
    for (x, y), peaks_here in lookup.items():
        for neighbour in ((x + 1, y), (x, y + 1)):
            peaks_next = lookup.get(neighbour)
            if not peaks_next:
                continue
            matches, _, _ = match_neighbor_peaks(
                peaks_here, peaks_next, theta_gate=reference_gate
            )
            neighbour_deltas.extend(delta for _, _, delta in matches)
    q90 = float(np.percentile(neighbour_deltas, 90)) if neighbour_deltas else None
    min_internal = float(np.min(all_internal)) if all_internal else None
    suggestion = 60.0
    status = (
        "Недостаточно данных для автоматической оценки; начальное значение 60″ "
        "задано как рабочая настройка."
    )
    if q90 is not None and min_internal is not None and q90 < min_internal:
        suggestion = float(q90 + 0.25 * (min_internal - q90))
        status = (
            f"Q90 рассчитан по принятым one-to-one связям при опорном gate "
            f"{reference_gate:g}″. Рекомендация находится между этим Q90 и "
            "минимальной пиксельной медианой внутреннего промежутка; это старт "
            "для sensitivity analysis, не физический закон."
        )
    elif q90 is not None and min_internal is not None:
        status = (
            f"При опорном gate {reference_gate:g}″ интервал Q90 соседних связей "
            "< min internal gap не разделён; исходное значение оставлено как старт, "
            "требуется анализ чувствительности."
        )
    return {
        "n_events": int(ids.size),
        "occupied_pixels": int(len(lookup)),
        "reference_theta_gate": float(reference_gate),
        "neighbor_edge_count_reference_gate": int(len(neighbour_deltas)),
        "q90_neighbor_delta": q90,
        "min_internal_gap": min_internal,
        "recommended_theta_gate": suggestion,
        "recommendation_status": status,
        "internal_gap_by_n": rows,
        "theta_unit": "arcsec",
    }


def fitted_peak_positions(fit_bundle):
    """Return fitted peak rows with stable detected ``peak_id`` identity.

    The mapping is validated by (x, y, component_id). Ambiguous or incomplete
    component correspondence is rejected instead of silently assigning a point
    to the wrong detected peak.
    """
    import pandas as pd

    tables = getattr(fit_bundle, "tables", {}) or {}
    detected = tables.get("peak_catalog")
    fitted = tables.get("fitted_peak_catalog")
    if detected is None or fitted is None or detected.empty or fitted.empty:
        raise ValueError("Fit-артефакт должен содержать непустые peak_catalog и fitted_peak_catalog")
    for frame_name, frame in (("peak_catalog", detected), ("fitted_peak_catalog", fitted)):
        missing = {"x", "y"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} не содержит координаты {sorted(missing)}")
    center_column = next(
        (name for name in ("theta0", "theta_fit", "center_theta", "theta") if name in fitted.columns),
        None,
    )
    if center_column is None:
        raise ValueError("В fitted_peak_catalog отсутствует fitted-центр (theta0/theta_fit/center_theta)")
    if "peak_id" not in detected.columns:
        raise ValueError("В исходном peak_catalog отсутствует стабильный peak_id")

    detected_frame = detected.copy()
    fitted_frame = fitted.copy()
    if "component_id" not in detected_frame.columns:
        if "theta" not in detected_frame.columns:
            raise ValueError("Нельзя сопоставить fit-компоненты: в peak_catalog нет theta")
        detected_frame = detected_frame.sort_values(
            ["y", "x", "theta", "peak_id"], kind="mergesort"
        ).copy()
        detected_frame["component_id"] = detected_frame.groupby(
            ["y", "x"], sort=False
        ).cumcount()
    if "component_id" not in fitted_frame.columns:
        raise ValueError(
            "В fitted_peak_catalog отсутствует component_id; безопасное сопоставление невозможно"
        )

    keys = ["x", "y", "component_id"]
    if detected_frame.duplicated(keys).any() or fitted_frame.duplicated(keys).any():
        raise ValueError(
            "Сопоставление detected/fitted пиков неоднозначно: повторяются (x, y, component_id)"
        )
    mapping_columns = [*keys, "peak_id"]
    if "theta" in detected_frame.columns and "theta_detected" not in fitted_frame.columns:
        mapping_columns.append("theta")
    mapping = detected_frame[mapping_columns].copy()
    if "theta" in mapping.columns:
        mapping = mapping.rename(columns={"theta": "theta_detected"})

    # Some fitting implementations may already propagate peak_id. Validate it
    # rather than allowing pandas to create ambiguous peak_id_x/peak_id_y cols.
    if "peak_id" in fitted_frame.columns:
        check = fitted_frame[[*keys, "peak_id"]].merge(
            mapping[[*keys, "peak_id"]],
            on=keys, how="left", validate="one_to_one", sort=False,
            suffixes=("_fitted", "_detected"),
        )
        try:
            fitted_ids = pd.to_numeric(check["peak_id_fitted"], errors="raise").to_numpy(dtype=np.int64)
            detected_ids = pd.to_numeric(check["peak_id_detected"], errors="raise").to_numpy(dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Сохранённые fitted peak_id некорректны") from exc
        if check["peak_id_detected"].isna().any() or not np.array_equal(fitted_ids, detected_ids):
            raise ValueError("peak_id в fitted_peak_catalog не совпадает с идентичностью detected-компонент")
        fitted_frame = fitted_frame.drop(columns=["peak_id"])

    result = fitted_frame.merge(
        mapping, on=keys, how="inner", validate="one_to_one", sort=False
    )
    if len(result) != len(fitted_frame):
        raise ValueError(
            "Не все fitted-компоненты удалось однозначно сопоставить исходным peak_id; "
            "ветвление по Fit остановлено, чтобы не присвоить неверную идентичность."
        )
    centers = pd.to_numeric(result[center_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(centers).all():
        raise ValueError("Fitted-центры содержат NaN или бесконечные значения")
    result["theta"] = centers
    return result
