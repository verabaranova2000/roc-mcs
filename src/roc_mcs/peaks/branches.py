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
    """Build accepted edges between one-to-one matches in right/down neighbours.

    The one-to-one assignment is independent of ``theta_gate``: the gate only
    filters the already assigned candidate edges.  Keeping that distinction
    explicit is what makes an exact, inexpensive gate-sensitivity scan possible.
    """
    gate = _validate_theta_gate(theta_gate)
    candidate_edges, candidate_deltas = build_peak_branch_candidate_graph(peak_catalog)
    accepted = candidate_deltas <= gate
    return (
        [edge for edge, keep in zip(candidate_edges, accepted) if bool(keep)],
        candidate_deltas[accepted],
    )


def build_peak_branch_candidate_graph(peak_catalog):
    """Build all one-to-one candidate edges and their exact ``|Δθ|`` weights.

    For each 4-connected detector-pixel pair the Hungarian assignment is
    calculated once, without applying a finite gate.  The returned graph is
    the complete weighted graph from which every finite-gate branch graph is
    obtained by the exact sublevel condition ``edge_delta <= theta_gate``.
    """
    lookup = _pixel_peak_lookup(peak_catalog)
    edges: list[tuple[int, int]] = []
    deltas: list[float] = []
    for (x, y), peaks_here in lookup.items():
        for neighbour in ((x + 1, y), (x, y + 1)):
            peaks_next = lookup.get(neighbour)
            if not peaks_next:
                continue
            matches, _, _ = match_neighbor_peaks(
                peaks_here, peaks_next, theta_gate=float("inf")
            )
            for peak_a, peak_b, delta in matches:
                edges.append((peak_a, peak_b))
                deltas.append(delta)
    return edges, np.asarray(deltas, dtype=np.float64)


def _candidate_graph_or_build(peak_catalog, candidate_graph=None):
    if candidate_graph is None:
        return build_peak_branch_candidate_graph(peak_catalog)
    edges, deltas = candidate_graph
    edges = [(int(edge[0]), int(edge[1])) for edge in edges]
    deltas = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if len(edges) != deltas.size:
        raise ValueError("Число candidate-рёбер не совпадает с числом весов |Δθ|")
    if not np.isfinite(deltas).all() or np.any(deltas < 0.0):
        raise ValueError("Веса candidate-рёбер должны быть конечными и неотрицательными")
    return edges, deltas


def _branch_sizes_from_candidate_graph(peak_catalog, candidate_graph, theta_gate: float):
    gate = _validate_theta_gate(theta_gate)
    edges, deltas = _candidate_graph_or_build(peak_catalog, candidate_graph)
    selected_edges = [edge for edge, delta in zip(edges, deltas) if delta <= gate]
    count, labels = label_peak_branches(peak_catalog, selected_edges)
    sizes = (
        np.bincount(labels.astype(np.int64), minlength=count).astype(np.int64)
        if count > 0 else np.empty(0, dtype=np.int64)
    )
    return sizes


def scan_theta_gate_stability(
    peak_catalog,
    *,
    candidate_graph=None,
    minimum_gate: float = 0.001,
    minimum_multiplicative_gap: float = 4.0,
    minimum_merges_before: int = 2,
) -> dict[str, Any]:
    """Scan the exact connected-component filtration induced by ``theta_gate``.

    Each Hungarian one-to-one neighbour assignment is fixed before the gate is
    applied.  Increasing the gate therefore only adds weighted edges in order
    of ``|Δθ|``.  A union-find sweep visits all distinct component-merger levels
    exactly, without repeatedly rebuilding the graph for a parameter grid.

    A recommendation is emitted only for a broad *stability window*: either
    the initial interval from the UI's minimum gate to the first later merge,
    or an interval between consecutive merge-causing edge weights.  In each
    case the full connected-component partition is unchanged.  The selected
    point is the geometric midpoint of the widest qualifying multiplicative
    interval.
    This is deliberately not presented as a universal elbow or physical law.
    """
    gate_min = _validate_theta_gate(minimum_gate)
    gap_min = float(minimum_multiplicative_gap)
    if not np.isfinite(gap_min) or gap_min <= 1.0:
        raise ValueError("minimum_multiplicative_gap должен быть конечным числом больше 1")
    min_merges = max(1, int(minimum_merges_before))

    ids, _, _, _ = _validate_catalog(peak_catalog)
    n_events = int(ids.size)
    if n_events == 0:
        return {
            "method": "weighted_candidate_graph_component_filtration_v1",
            "status": "empty",
            "n_events": 0,
            "n_candidate_edges": 0,
            "n_merge_levels": 0,
            "minimum_gate": gate_min,
            "maximum_gate": max(gate_min * 10.0, 0.01),
            "initial_branch_count": 0,
            "final_branch_count": 0,
            "levels": [],
            "stability_window": None,
            "recommended_gate": None,
        }

    edges, deltas = _candidate_graph_or_build(peak_catalog, candidate_graph)
    if not edges:
        return {
            "method": "weighted_candidate_graph_component_filtration_v1",
            "status": "no_candidate_edges",
            "n_events": n_events,
            "n_candidate_edges": 0,
            "n_merge_levels": 0,
            "minimum_gate": gate_min,
            "maximum_gate": max(gate_min * 10.0, 0.01),
            "initial_branch_count": n_events,
            "final_branch_count": n_events,
            "levels": [],
            "stability_window": None,
            "recommended_gate": None,
        }

    id_to_row = {int(peak_id): row for row, peak_id in enumerate(ids)}
    edge_rows = np.empty((len(edges), 2), dtype=np.int64)
    for index, (peak_a, peak_b) in enumerate(edges):
        try:
            edge_rows[index, 0] = id_to_row[int(peak_a)]
            edge_rows[index, 1] = id_to_row[int(peak_b)]
        except KeyError as exc:
            raise ValueError(f"Candidate-ребро ссылается на неизвестный peak_id: {exc.args[0]}") from exc

    order = np.argsort(deltas, kind="mergesort")
    parent = np.arange(n_events, dtype=np.int64)
    component_size = np.ones(n_events, dtype=np.int64)

    def find_root(value: int) -> int:
        root = int(value)
        while parent[root] != root:
            parent[root] = parent[int(parent[root])]
            root = int(parent[root])
        return root

    components = n_events
    accepted_edges_at_minimum = 0
    initial_branch_count = n_events
    levels: list[dict[str, Any]] = []
    cumulative_merges = 0
    index = 0
    while index < order.size:
        delta = float(deltas[order[index]])
        end = index + 1
        while end < order.size and float(deltas[order[end]]) == delta:
            end += 1
        before = components
        for ordered_index in order[index:end]:
            row_a, row_b = edge_rows[int(ordered_index)]
            root_a = find_root(int(row_a))
            root_b = find_root(int(row_b))
            if root_a == root_b:
                continue
            if component_size[root_a] < component_size[root_b]:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            component_size[root_a] += component_size[root_b]
            components -= 1

        merges = before - components
        # Count merges from the beginning of the filtration, including edges
        # already accepted at the minimum UI gate.  They establish that the
        # initial plateau contains actual spatially linked components rather
        # than a collection of isolated peak-events.
        cumulative_merges += merges
        if delta <= gate_min:
            accepted_edges_at_minimum = end
            initial_branch_count = components
        elif merges:
            levels.append({
                "gate": delta,
                "branch_count": int(components),
                "merges_at_level": int(merges),
                "cumulative_merges": int(cumulative_merges),
            })
        index = end

    max_delta = float(np.max(deltas)) if deltas.size else gate_min
    maximum_gate = max(gate_min * 10.0, max_delta * 1.25, gate_min + 1e-9)
    stability_window = None
    candidates: list[dict[str, Any]] = []

    # Include the initial plateau beginning at the smallest gate supported by
    # the UI.  It can be the strongest stability interval when many local
    # links have very small |Δθ| and the first cross-component merger is much
    # farther away.
    if levels:
        merges_before = int(n_events - initial_branch_count)
        merges_after = int(cumulative_merges - merges_before)
        lower = float(gate_min)
        upper = float(levels[0]["gate"])
        ratio = upper / lower if lower > 0.0 else float("inf")
        if merges_before >= min_merges and merges_after >= 1:
            candidates.append({
                "lower_gate": lower,
                "upper_gate": upper,
                "recommended_gate": float(np.sqrt(lower * upper)),
                "branch_count": int(initial_branch_count),
                "multiplicative_gap": float(ratio),
                "merges_before": merges_before,
                "merges_after": merges_after,
            })

    for level_index in range(len(levels) - 1):
        left = levels[level_index]
        right = levels[level_index + 1]
        merges_before = int(left["cumulative_merges"])
        merges_after = int(cumulative_merges - merges_before)
        lower = float(left["gate"])
        upper = float(right["gate"])
        ratio = upper / lower if lower > 0.0 else float("inf")
        if merges_before < min_merges or merges_after < 1:
            continue
        recommendation = float(np.sqrt(lower * upper)) if lower > 0.0 else float(upper)
        candidates.append({
            "lower_gate": lower,
            "upper_gate": upper,
            "recommended_gate": recommendation,
            "branch_count": int(left["branch_count"]),
            "multiplicative_gap": float(ratio),
            "merges_before": merges_before,
            "merges_after": merges_after,
        })

    qualifying = [item for item in candidates if item["multiplicative_gap"] >= gap_min]
    if qualifying:
        stability_window = max(
            qualifying,
            key=lambda item: (item["multiplicative_gap"], item["merges_before"], -item["lower_gate"]),
        )
        status = "stable_window_found"
        recommended_gate = float(stability_window["recommended_gate"])
    else:
        status = (
            "partition_constant" if not levels
            else "no_robust_stability_window"
        )
        recommended_gate = None
        if candidates:
            stability_window = max(
                candidates,
                key=lambda item: (item["multiplicative_gap"], item["merges_before"], -item["lower_gate"]),
            )

    return {
        "method": "weighted_candidate_graph_component_filtration_v1",
        "status": status,
        "n_events": n_events,
        "n_candidate_edges": int(len(edges)),
        "n_merge_levels": int(len(levels)),
        "accepted_edges_at_minimum": int(accepted_edges_at_minimum),
        "minimum_gate": gate_min,
        "maximum_gate": float(maximum_gate),
        "initial_branch_count": int(initial_branch_count),
        "final_branch_count": int(components),
        "levels": levels,
        "stability_window": stability_window,
        "recommended_gate": recommended_gate,
        "minimum_multiplicative_gap": gap_min,
        "minimum_merges_before": min_merges,
    }


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



def recommend_min_branch_events(
    branch_sizes, *, min_multiplicative_gap: float = 4.0,
) -> tuple[int, dict[str, Any]]:
    """Suggest a conservative event-count cutoff from a clear size gap.

    The rule is intentionally descriptive, not a physical law: unique positive
    component sizes are ordered, the largest adjacent multiplicative gap is
    considered, and a cutoff is proposed only when the gap is at least
    ``min_multiplicative_gap`` and at least two components lie below it.
    The threshold is the geometric midpoint (natural on a log-size axis).
    When no clear gap exists, return 1, i.e. retain every branch.
    """
    values = np.asarray(branch_sizes, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 1, {"method": "largest_multiplicative_gap_v1", "status": "empty",
                    "ratio": None, "lower_size": None, "upper_size": None}
    if (not np.isfinite(values).all() or np.any(values < 1)
            or not np.equal(values, np.rint(values)).all()):
        raise ValueError("Размеры ветвей должны быть конечными положительными целыми числами")
    values_int = values.astype(np.int64)
    unique = np.unique(values_int)
    gap_min = float(min_multiplicative_gap)
    if not np.isfinite(gap_min) or gap_min <= 1.0:
        raise ValueError("min_multiplicative_gap должен быть конечным числом больше 1")
    if unique.size < 2:
        return 1, {"method": "largest_multiplicative_gap_v1", "status": "no_size_gap",
                    "ratio": None, "lower_size": int(unique[0]),
                    "upper_size": None, "min_multiplicative_gap": gap_min}

    # Candidate boundaries split *observed* sizes. Branches do not receive a
    # model-derived probability; ties are resolved toward the larger gap.
    candidates = []
    for index in range(unique.size - 1):
        lower = int(unique[index])
        upper = int(unique[index + 1])
        n_lower = int(np.count_nonzero(values_int <= lower))
        n_upper = int(values_int.size - n_lower)
        if n_lower < 2 or n_upper < 1:
            continue
        ratio = float(upper) / float(lower)
        candidates.append((ratio, lower, upper, n_lower, n_upper))
    if not candidates:
        return 1, {"method": "largest_multiplicative_gap_v1", "status": "insufficient_components",
                    "ratio": None, "lower_size": None, "upper_size": None,
                    "min_multiplicative_gap": gap_min}
    ratio, lower, upper, n_lower, n_upper = max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    if ratio < gap_min:
        return 1, {"method": "largest_multiplicative_gap_v1", "status": "no_clear_gap",
                    "ratio": ratio, "lower_size": lower, "upper_size": upper,
                    "branches_below": n_lower, "branches_above": n_upper,
                    "min_multiplicative_gap": gap_min}
    # There are no observed sizes between lower and upper. Any integer in this
    # interval yields the same classification; choose the log-space midpoint.
    threshold = int(np.floor(np.sqrt(float(lower) * float(upper)))) + 1
    threshold = max(lower + 1, min(upper, threshold))
    return threshold, {"method": "largest_multiplicative_gap_v1", "status": "clear_gap",
                      "ratio": ratio, "lower_size": lower, "upper_size": upper,
                      "branches_below": n_lower, "branches_above": n_upper,
                      "min_multiplicative_gap": gap_min,
                      "threshold": threshold}


def branch_analysis_mask(branch_catalog) -> np.ndarray:
    """Return event-aligned inclusion flags; legacy catalogues include all rows."""
    if branch_catalog is None:
        return np.zeros(0, dtype=bool)
    columns = set(getattr(branch_catalog, "columns", ()))
    if "analysis_included" not in columns:
        return np.ones(len(branch_catalog), dtype=bool)
    values = np.asarray(branch_catalog["analysis_included"], dtype=object)
    result = np.zeros(values.size, dtype=bool)
    for index, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            result[index] = bool(value)
        elif value is not None:
            text = str(value).strip().lower()
            result[index] = text in {"true", "1", "yes", "y"}
    return result


def calculate_branch_size_distribution(
    peak_catalog, theta_gate: float, *, candidate_graph=None,
) -> dict[str, Any]:
    """Calculate exact global connected-component sizes for the supplied gate."""
    sizes = _branch_sizes_from_candidate_graph(
        peak_catalog, candidate_graph, theta_gate
    )
    recommended, diagnostics = recommend_min_branch_events(sizes)
    return {
        "sizes": sizes.astype(int).tolist(),
        "recommended_min": int(recommended),
        "diagnostics": diagnostics,
        "n_branches": int(sizes.size),
        "n_events": int(len(peak_catalog)),
    }

def summarize_peak_spacing(
    peak_catalog, *, reference_theta_gate: float = 60.0, candidate_graph=None,
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

    _candidate_edges, candidate_deltas = _candidate_graph_or_build(
        peak_catalog, candidate_graph
    )
    neighbour_deltas = candidate_deltas[candidate_deltas <= reference_gate].tolist()
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
    # Preserve the original discrete scan coordinate alongside the fitted
    # continuous centre. The branch-correlation coarse registration operates
    # on theta_index, while support/refinement operate on fitted theta. This is
    # identity/provenance metadata only; it does not alter fit values or IDs.
    if "theta_index" in detected_frame.columns and "theta_index" not in fitted_frame.columns:
        mapping_columns.append("theta_index")
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
