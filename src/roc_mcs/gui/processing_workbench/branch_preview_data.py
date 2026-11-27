"""Small, scan-derived data and exact graph model for the branch-gate preview.

The UI never invents branch connectivity.  This adapter selects one compact
contiguous detector patch from the parent's peak catalogue and delegates every
candidate/accepted link and connected-component label to the production branch
algorithm.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_REQUIRED_COLUMNS = ("peak_id", "x", "y", "theta")


def select_preview_patch(
    peak_catalog: Any,
    *,
    width: int = 4,
    height: int = 3,
    max_events: int = 48,
) -> tuple[Any, dict[str, Any]]:
    """Choose a compact, contiguous patch without dropping any peak events.

    The preview must not silently thin a busy detector pixel: doing so can
    change the Hungarian one-to-one assignment and therefore show links that
    are not the links the production algorithm would create.  Candidate
    windows are therefore evaluated at 4×3 down to 2×2 and accepted only when
    *all* events in the selected pixels fit the display budget.  Among them,
    the selector prefers complete 2×2 spatial cells, then neighbour pairs,
    occupied pixels and event count.  No source row is modified or discarded.
    """
    import pandas as pd

    if peak_catalog is None or getattr(peak_catalog, "empty", True):
        return pd.DataFrame(columns=_REQUIRED_COLUMNS), {
            "available": False,
            "message": "Каталог пиков пуст: нечего показать в пространственно-угловом фрагменте.",
        }

    missing = set(_REQUIRED_COLUMNS).difference(getattr(peak_catalog, "columns", ()))
    if missing:
        return pd.DataFrame(columns=_REQUIRED_COLUMNS), {
            "available": False,
            "message": f"Для иллюстрации отсутствуют столбцы {sorted(missing)}.",
        }

    source_row_count = int(len(peak_catalog))
    frame = peak_catalog.loc[:, list(_REQUIRED_COLUMNS)].copy()
    for column in _REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame.loc[:, list(_REQUIRED_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[finite].copy()
    if frame.empty:
        return pd.DataFrame(columns=_REQUIRED_COLUMNS), {
            "available": False,
            "message": "В каталоге нет конечных координат peak_id, x, y, θ.",
        }

    integer_xy = (
        (frame["x"].to_numpy(dtype=float) == np.rint(frame["x"].to_numpy(dtype=float)))
        & (frame["y"].to_numpy(dtype=float) == np.rint(frame["y"].to_numpy(dtype=float)))
    )
    integer_ids = frame["peak_id"].to_numpy(dtype=float) == np.rint(
        frame["peak_id"].to_numpy(dtype=float)
    )
    frame = frame.loc[integer_xy & integer_ids].copy()
    valid_event_count = int(len(frame))
    if frame.empty or frame["peak_id"].duplicated().any():
        return pd.DataFrame(columns=_REQUIRED_COLUMNS), {
            "available": False,
            "message": "Невозможно выбрать фрагмент: нужны уникальные целые peak_id и индексы X/Y.",
        }

    frame["peak_id"] = frame["peak_id"].astype(np.int64)
    frame["x"] = frame["x"].astype(np.int64)
    frame["y"] = frame["y"].astype(np.int64)
    frame["theta"] = frame["theta"].astype(np.float64)
    counts = frame.groupby(["x", "y"], sort=True).size()
    count_by_pixel = {
        (int(x), int(y)): int(count)
        for (x, y), count in counts.items()
    }

    width = max(2, int(width))
    height = max(2, int(height))
    max_events = max(4, int(max_events))
    candidate_centres = list(counts.sort_values(ascending=False).head(64).index)
    origins: set[tuple[int, int]] = set()
    for center_x, center_y in candidate_centres:
        for dx in range(width):
            for dy in range(height):
                origins.add((int(center_x) - dx, int(center_y) - dy))

    best: tuple[tuple[int, ...], tuple[int, int], int, int] | None = None
    # Larger windows are considered first but only win when they carry more
    # actual spatial adjacency. This avoids selecting a sparse rectangle just
    # because its bounding box is large.
    for x0, y0 in origins:
        for window_width in range(width, 1, -1):
            for window_height in range(height, 1, -1):
                pixels = [
                    (x, y, count_by_pixel[(x, y)])
                    for x in range(x0, x0 + window_width)
                    for y in range(y0, y0 + window_height)
                    if (x, y) in count_by_pixel
                ]
                if not pixels:
                    continue
                event_count = sum(count for _x, _y, count in pixels)
                if event_count > max_events:
                    continue
                occupied = {(x, y) for x, y, _count in pixels}
                adjacent_pairs = sum(
                    ((x + 1, y) in occupied) + ((x, y + 1) in occupied)
                    for x, y in occupied
                )
                complete_cells = sum(
                    (x + 1, y) in occupied
                    and (x, y + 1) in occupied
                    and (x + 1, y + 1) in occupied
                    for x, y in occupied
                )
                score = (
                    int(complete_cells), int(adjacent_pairs), len(occupied),
                    window_width * window_height, int(event_count),
                )
                if best is None or score > best[0]:
                    best = (score, (x0, y0), window_width, window_height)

    if best is None:
        return pd.DataFrame(columns=_REQUIRED_COLUMNS), {
            "available": False,
            "message": (
                f"Не удалось найти компактный участок: в каждом соседнем окне "
                f"больше {max_events} событий пиков. Иллюстрация не прореживает "
                "данные, чтобы не менять реальные сопоставления."
            ),
        }

    _score, (x0, y0), selected_width, selected_height = best
    local = frame[
        frame["x"].between(x0, x0 + selected_width - 1)
        & frame["y"].between(y0, y0 + selected_height - 1)
    ].copy()
    preview = local.sort_values(
        ["y", "x", "theta", "peak_id"], kind="mergesort"
    ).reset_index(drop=True)
    occupied_pixels = int(preview.groupby(["x", "y"], sort=False).ngroups)
    info = {
        "available": True,
        "x_min": int(x0),
        "x_max": int(x0 + selected_width - 1),
        "y_min": int(y0),
        "y_max": int(y0 + selected_height - 1),
        "width": int(selected_width),
        "height": int(selected_height),
        "occupied_pixels": occupied_pixels,
        "event_count": int(len(preview)),
        "source_event_count": int(len(local)),
        "source_catalog_rows": source_row_count,
        "valid_catalog_rows": valid_event_count,
        "invalid_rows_excluded": int(source_row_count - valid_event_count),
        "truncated": False,
        "all_events_preserved": True,
        "max_events": int(max_events),
        "complete_spatial_cells": int(best[0][0]),
    }
    return preview.loc[:, list(_REQUIRED_COLUMNS)], info


def prepare_branch_preview_graph(peak_catalog: Any) -> dict[str, Any]:
    """Prepare immutable preview events and exact one-to-one candidates once.

    The Hungarian assignment is independent of the finite θ-gate.  The preview
    therefore prepares it only when its parent catalog changes; interactive
    gate edits merely filter these weighted edges and relabel the tiny local
    connected-component graph.
    """
    from roc_mcs.peaks.branches import build_peak_branch_candidate_graph

    if peak_catalog is None or getattr(peak_catalog, "empty", True):
        return {"catalog": None, "events": {}, "weighted_edges": ()}

    ids = np.asarray(peak_catalog["peak_id"], dtype=np.int64)
    xs = np.asarray(peak_catalog["x"], dtype=np.int64)
    ys = np.asarray(peak_catalog["y"], dtype=np.int64)
    thetas = np.asarray(peak_catalog["theta"], dtype=np.float64)
    events = {
        int(peak_id): {
            "peak_id": int(peak_id), "x": int(x), "y": int(y), "theta": float(theta),
        }
        for peak_id, x, y, theta in zip(ids, xs, ys, thetas)
    }

    candidate_pairs, candidate_deltas = build_peak_branch_candidate_graph(peak_catalog)
    weighted_edges = tuple(
        (int(a), int(b), float(delta))
        for (a, b), delta in zip(candidate_pairs, candidate_deltas)
    )
    return {
        # The catalog is only the compact local display patch (at most 48 rows).
        # Keep a private copy so later gate updates label exactly this snapshot.
        "catalog": peak_catalog.copy(deep=False),
        "events": events,
        "weighted_edges": weighted_edges,
    }


def build_branch_preview_graph(
    peak_catalog: Any,
    theta_gate: float,
    *,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize the exact graph at ``theta_gate`` from prepared candidates.

    With ``prepared`` supplied (the interactive path), no Hungarian matching,
    pixel lookup, or event-dictionary construction is repeated per pointer
    movement. The ``delta <= gate`` comparison remains unrounded and exact.
    """
    from roc_mcs.peaks.branches import label_peak_branches

    if prepared is None:
        prepared = prepare_branch_preview_graph(peak_catalog)
    catalog = prepared.get("catalog")
    events = prepared.get("events", {})
    weighted_edges = prepared.get("weighted_edges", ())
    if catalog is None or not events:
        return {
            "events": {}, "candidate_edges": [], "accepted_edges": [],
            "n_branches": 0, "branch_by_peak_id": {},
        }

    gate = float(theta_gate)
    accepted_edges = [
        (a, b, delta) for a, b, delta in weighted_edges if delta <= gate
    ]
    accepted_pairs = [(a, b) for a, b, _delta in accepted_edges]
    n_branches, labels = label_peak_branches(catalog, accepted_pairs)
    ids = np.asarray(catalog["peak_id"], dtype=np.int64)
    branch_by_peak_id = {
        int(peak_id): int(label) for peak_id, label in zip(ids, labels)
    }
    accepted_keys = {(a, b) for a, b, _delta in accepted_edges}
    candidate_edges = [
        (a, b, delta, (a, b) in accepted_keys)
        for a, b, delta in weighted_edges
    ]
    return {
        "events": events,
        "candidate_edges": candidate_edges,
        "accepted_edges": accepted_edges,
        "n_branches": int(n_branches),
        "branch_by_peak_id": branch_by_peak_id,
    }
