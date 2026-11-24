"""Renderer-neutral model for visualising stored 3D branch correlations.

This module is deliberately a read-only adapter over the persisted
``branch_catalog`` and ``branch_correlation_*`` tables.  It does not run or
modify the scientific correlation algorithm.  Coordinates are converted from
catalogue detector indices to the physical X/Y axes supplied by the Scan.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


_DEFAULT_MAX_MATCHED_LINKS = 100_000
_DEFAULT_MAX_CANDIDATE_LINKS = 20_000


def _column(frame: Any, name: str, *, dtype=np.float64, default: Any | None = None) -> np.ndarray:
    if frame is None or name not in getattr(frame, "columns", ()):
        if default is not None:
            return np.full(0, default, dtype=dtype)
        raise ValueError(f"В таблице отсутствует обязательный столбец {name!r}")
    values = frame[name]
    if hasattr(values, "to_numpy"):
        try:
            return np.asarray(values.to_numpy(dtype=dtype, na_value=np.nan), dtype=dtype)
        except (TypeError, ValueError):
            pass
    raw = np.asarray(values, dtype=object).reshape(-1)
    result = np.empty(raw.size, dtype=dtype)
    for index, value in enumerate(raw):
        try:
            result[index] = value
        except (TypeError, ValueError, OverflowError):
            result[index] = np.nan
    return result


def _integer_column(frame: Any, name: str) -> np.ndarray:
    values = _column(frame, name)
    valid = np.isfinite(values) & (values == np.rint(values))
    result = np.full(values.size, -1, dtype=np.int64)
    if np.any(valid):
        safe = values[valid]
        if np.any(np.abs(safe) > np.iinfo(np.int64).max):
            raise ValueError(f"Столбец {name!r} содержит ID вне диапазона int64")
        result[valid] = np.rint(safe).astype(np.int64)
    return result


def _lookup_positions(
    query_ids: np.ndarray,
    sorted_ids: np.ndarray,
    sorted_positions: np.ndarray,
    sorted_branches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ID lookup; return positions, branch IDs and validity mask."""
    queries = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if sorted_ids.size == 0 or queries.size == 0:
        return (
            np.full((queries.size, 3), np.nan, dtype=np.float64),
            np.full(queries.size, -1, dtype=np.int64),
            np.zeros(queries.size, dtype=bool),
        )
    index = np.searchsorted(sorted_ids, queries)
    in_bounds = index < sorted_ids.size
    clipped = np.minimum(index, sorted_ids.size - 1)
    valid = in_bounds & (queries >= 0) & (sorted_ids[clipped] == queries)
    positions = np.full((queries.size, 3), np.nan, dtype=np.float64)
    branches = np.full(queries.size, -1, dtype=np.int64)
    if np.any(valid):
        positions[valid] = sorted_positions[clipped[valid]]
        branches[valid] = sorted_branches[clipped[valid]]
    return positions, branches, valid


def _bounded_indices(count: int, maximum: int) -> np.ndarray:
    count = int(count)
    maximum = int(maximum)
    if count <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    # Deterministic, evenly spaced display subset. The totals remain exact in
    # the model and in the legend; only GPU geometry is capped.
    return np.unique(np.linspace(0, count - 1, maximum, dtype=np.int64))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)
    return int(round(number)) if np.isfinite(number) else int(default)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _safe_int_or_none(value: Any) -> int | None:
    number = _safe_float(value)
    return int(round(number)) if number is not None else None


def _freeze_array(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class BranchCorrelationPairPreview:
    """Display data and audit summary for one source/partner branch pair.

    ``matched_peak_ids`` is the exact set of endpoints in accepted assignments.
    The Qt/VTK adapter intersects it with the bounded set of currently rendered
    glyph IDs before applying the core/residual emphasis.
    """

    source_branch_id: int
    partner_branch_id: int
    matched_segments: np.ndarray
    candidate_segments: np.ndarray
    matched_peak_ids: np.ndarray
    matched_total_count: int
    candidate_total_count: int
    summary: Mapping[str, Any]
    matched_truncated: bool = False
    candidate_truncated: bool = False


@dataclass(frozen=True, slots=True)
class BranchCorrelationPreview:
    """Read-only 3D presentation model built from one stored correlation run."""

    source_branch_id: int
    pairs: tuple[BranchCorrelationPairPreview, ...]
    assignment_counts: Mapping[str, int]
    warnings: tuple[str, ...] = ()

    @property
    def total_matched(self) -> int:
        return int(sum(pair.matched_total_count for pair in self.pairs))

    @property
    def total_candidates(self) -> int:
        return int(sum(pair.candidate_total_count for pair in self.pairs))


def build_branch_correlation_preview(
    branch_catalog: Any,
    correlation_bundle: Any,
    x_coords: Any,
    y_coords: Any,
    *,
    max_matched_links: int = _DEFAULT_MAX_MATCHED_LINKS,
    max_candidate_links: int = _DEFAULT_MAX_CANDIDATE_LINKS,
) -> BranchCorrelationPreview:
    """Build a compact visualization from already-computed correlation tables.

    Accepted links are read only from ``assignment_class == 'partner'`` and
    connect ``source_peak_id`` to ``matched_partner_peak_id``. Alternative
    supported links are taken from ``candidates`` where ``supported`` is true
    and ``assigned`` is false; they are shown only when the user enables the
    optional candidate layer. Peaks not incident to accepted links remain in
    the original branch clouds and can be faded as residual context.
    """
    try:
        max_matched_links = int(max_matched_links)
        max_candidate_links = int(max_candidate_links)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Лимиты отображаемых корреляционных связей должны быть целыми") from exc
    if not 1 <= max_matched_links <= 100_000:
        raise ValueError("max_matched_links должен находиться в диапазоне 1–100000")
    if not 1 <= max_candidate_links <= 20_000:
        raise ValueError("max_candidate_links должен находиться в диапазоне 1–20000")
    if branch_catalog is None or getattr(branch_catalog, "empty", True):
        raise ValueError("Для 3D-корреляции отсутствует непустой branch_catalog")
    tables = getattr(correlation_bundle, "tables", None)
    if not isinstance(tables, Mapping):
        raise ValueError("В артефакте корреляции отсутствует каталог таблиц")
    pairs_frame = tables.get("branch_correlation_pairs")
    assignments = tables.get("branch_correlation_assignments")
    candidates = tables.get("branch_correlation_candidates")
    if pairs_frame is None or getattr(pairs_frame, "empty", True):
        raise ValueError("В результате корреляции отсутствует таблица branch_correlation_pairs")
    if assignments is None:
        raise ValueError("В результате корреляции отсутствует branch_correlation_assignments")

    required_catalog = {"peak_id", "branch_id", "x", "y", "theta"}
    missing = required_catalog.difference(getattr(branch_catalog, "columns", ()))
    if missing:
        raise ValueError(f"branch_catalog не содержит поля {sorted(missing)}")

    x_axis = np.asarray(x_coords, dtype=np.float64)
    y_axis = np.asarray(y_coords, dtype=np.float64)
    if x_axis.ndim != 1 or y_axis.ndim != 1 or not np.isfinite(x_axis).all() or not np.isfinite(y_axis).all():
        raise ValueError("Физические оси X/Y должны быть конечными одномерными массивами")

    peak_ids = _integer_column(branch_catalog, "peak_id")
    branch_ids = _integer_column(branch_catalog, "branch_id")
    x_index = _integer_column(branch_catalog, "x")
    y_index = _integer_column(branch_catalog, "y")
    theta = _column(branch_catalog, "theta")
    valid_catalog = (
        (peak_ids >= 0) & (branch_ids >= 0) & (x_index >= 0) & (x_index < x_axis.size)
        & (y_index >= 0) & (y_index < y_axis.size) & np.isfinite(theta)
    )
    if not np.any(valid_catalog):
        raise ValueError("В branch_catalog нет пиков с корректными ID и координатами")
    ids = peak_ids[valid_catalog]
    branches = branch_ids[valid_catalog]
    positions = np.column_stack((x_axis[x_index[valid_catalog]], y_axis[y_index[valid_catalog]], theta[valid_catalog]))
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_branches = branches[order]
    sorted_positions = positions[order]
    if sorted_ids.size > 1 and np.any(np.diff(sorted_ids) == 0):
        duplicate = int(sorted_ids[np.flatnonzero(np.diff(sorted_ids) == 0)[0]])
        raise ValueError(f"branch_catalog содержит повторяющийся peak_id={duplicate}")

    warnings: list[str] = []
    source_values = _integer_column(assignments, "source_peak_id")
    assignment_class = np.asarray(assignments["assignment_class"], dtype=object)
    assigned_partner = _integer_column(assignments, "assigned_partner_branch_id")
    # Older/partially written artifacts may lack the explicit target peak ID.
    # Keep their registration summary inspectable, but never invent visual
    # links by nearest-coordinate matching.
    if "matched_partner_peak_id" in assignments.columns:
        matched_partner_ids = _integer_column(assignments, "matched_partner_peak_id")
    else:
        matched_partner_ids = np.full(source_values.size, -1, dtype=np.int64)
        warnings.append(
            "В legacy-артефакте отсутствует matched_partner_peak_id: "
            "сводка регистрации доступна, но линии принятых соответствий не восстановлены."
        )
    source_branch_values = (
        _integer_column(assignments, "source_branch_id")
        if "source_branch_id" in assignments.columns else np.full(source_values.size, -1, dtype=np.int64)
    )
    assignment_counts = {
        "partner": int(np.count_nonzero(assignment_class == "partner")),
        "ambiguous": int(np.count_nonzero(assignment_class == "ambiguous")),
        "residual": int(np.count_nonzero(assignment_class == "residual")),
    }

    candidate_source = candidate_partner = candidate_target = None
    candidate_supported = candidate_assigned = None
    if candidates is not None and not getattr(candidates, "empty", True):
        required_candidate = {"source_peak_id", "partner_branch_id", "matched_partner_peak_id", "supported", "assigned"}
        absent = required_candidate.difference(getattr(candidates, "columns", ()))
        if absent:
            raise ValueError(f"branch_correlation_candidates не содержит поля {sorted(absent)}")
        candidate_source = _integer_column(candidates, "source_peak_id")
        candidate_partner = _integer_column(candidates, "partner_branch_id")
        candidate_target = _integer_column(candidates, "matched_partner_peak_id")
        candidate_supported = np.asarray(candidates["supported"], dtype=bool)
        candidate_assigned = np.asarray(candidates["assigned"], dtype=bool)

    pair_previews: list[BranchCorrelationPairPreview] = []
    # The geometry budget is global for the complete scene, not per pair.
    # A deterministic per-pair share prevents a large first pair from hiding
    # every later pair and bounds VTK allocations independently of data size.
    valid_pair_count = max(1, int(len(pairs_frame)))
    matched_base, matched_remainder = divmod(max_matched_links, valid_pair_count)
    candidate_base, candidate_remainder = divmod(max_candidate_links, valid_pair_count)
    for pair_index, (_, pair_row) in enumerate(pairs_frame.iterrows()):
        source_id = _safe_int(pair_row.get("source_branch_id"), -1)
        partner_id = _safe_int(pair_row.get("partner_branch_id"), -1)
        if source_id < 0 or partner_id < 0 or source_id == partner_id:
            warnings.append(f"Пропущена некорректная пара branch_id {source_id} ↔ {partner_id}")
            continue

        assigned_mask = (
            (assignment_class == "partner")
            & (assigned_partner == partner_id)
            & (source_values >= 0)
            & (matched_partner_ids >= 0)
        )
        if np.any(source_branch_values >= 0):
            assigned_mask &= source_branch_values == source_id
        accepted_source_ids = source_values[assigned_mask]
        accepted_target_ids = matched_partner_ids[assigned_mask]
        assigned_row_count = int(np.count_nonzero(assigned_mask))
        source_positions, source_branches, source_valid = _lookup_positions(
            accepted_source_ids, sorted_ids, sorted_positions, sorted_branches
        )
        target_positions, target_branches, target_valid = _lookup_positions(
            accepted_target_ids, sorted_ids, sorted_positions, sorted_branches
        )
        exact_pair = (
            source_valid & target_valid & (source_branches == source_id)
            & (target_branches == partner_id)
        )
        matched_segments_all = np.stack((source_positions[exact_pair], target_positions[exact_pair]), axis=1) if np.any(exact_pair) else np.empty((0, 2, 3), dtype=np.float64)
        matched_source_ids_all = accepted_source_ids[exact_pair]
        matched_target_ids_all = accepted_target_ids[exact_pair]
        invalid_assignment_count = assigned_row_count - int(np.count_nonzero(exact_pair))
        if invalid_assignment_count:
            warnings.append(
                f"Пара {source_id} ↔ {partner_id}: {invalid_assignment_count} принятых "
                "назначений не удалось связать с точками branch_catalog по peak_id/branch_id; "
                "эти линии не нарисованы."
            )
        if matched_source_ids_all.size > 1 and np.unique(matched_source_ids_all).size != matched_source_ids_all.size:
            warnings.append(
                f"Пара {source_id} ↔ {partner_id}: повторяются source_peak_id среди "
                "принятых назначений; повторные строки сохранены без искусственного объединения."
            )

        candidate_mask = np.zeros(0, dtype=bool)
        candidate_segments_all = np.empty((0, 2, 3), dtype=np.float64)
        if candidate_source is not None:
            candidate_mask = (
                (candidate_partner == partner_id) & candidate_supported & ~candidate_assigned
                & (candidate_source >= 0) & (candidate_target >= 0)
            )
            alt_source, alt_source_branch, alt_source_valid = _lookup_positions(
                candidate_source[candidate_mask], sorted_ids, sorted_positions, sorted_branches
            )
            alt_target, alt_target_branch, alt_target_valid = _lookup_positions(
                candidate_target[candidate_mask], sorted_ids, sorted_positions, sorted_branches
            )
            alt_valid = (
                alt_source_valid & alt_target_valid & (alt_source_branch == source_id)
                & (alt_target_branch == partner_id)
            )
            if np.any(alt_valid):
                candidate_segments_all = np.stack((alt_source[alt_valid], alt_target[alt_valid]), axis=1)

        matched_budget = matched_base + int(pair_index < matched_remainder)
        candidate_budget = candidate_base + int(pair_index < candidate_remainder)
        matched_indices = _bounded_indices(len(matched_segments_all), matched_budget)
        candidate_indices = _bounded_indices(len(candidate_segments_all), candidate_budget)
        candidate_row_count = int(np.count_nonzero(candidate_mask))
        invalid_candidate_count = candidate_row_count - int(len(candidate_segments_all))
        if invalid_candidate_count:
            warnings.append(
                f"Пара {source_id} ↔ {partner_id}: {invalid_candidate_count} поддержанных "
                "альтернатив не удалось связать с точками branch_catalog; они не нарисованы."
            )
        # Keep the complete accepted endpoint set for scientifically correct
        # core/residual classification. The GUI adapter intersects this set
        # with the at-most-150k VTK glyph IDs before applying alpha emphasis.
        matched_ids_all = (
            np.unique(np.concatenate((matched_source_ids_all, matched_target_ids_all)))
            if len(matched_source_ids_all) else np.empty(0, dtype=np.int64)
        )
        pair_summary = {
            "source_event_count": _safe_int(pair_row.get("source_event_count")),
            "partner_event_count": _safe_int(pair_row.get("partner_event_count")),
            "assigned_source_count": int(len(matched_segments_all)),
            "assignment_row_count": assigned_row_count,
            "invalid_assignment_count": invalid_assignment_count,
            "supported_count": _safe_int(pair_row.get("final_supported_count")),
            "ambiguous_candidate_count": int(len(candidate_segments_all)),
            "candidate_row_count": candidate_row_count,
            "invalid_candidate_count": invalid_candidate_count,
            "shift_x_px": _safe_int_or_none(pair_row.get("refined_shift_x_px")),
            "shift_y_px": _safe_int_or_none(pair_row.get("refined_shift_y_px")),
            "shift_theta_arcsec": _safe_float(pair_row.get("theta_shift_final_arcsec")),
            "theta_residual_mad_arcsec": _safe_float(pair_row.get("theta_residual_mad_arcsec")),
            "refinement_status": str(pair_row.get("refinement_status", "unknown")),
            "shift_direction": "сдвиг, применяемый к партнёру для совмещения с source",
        }
        pair_previews.append(BranchCorrelationPairPreview(
            source_branch_id=source_id,
            partner_branch_id=partner_id,
            matched_segments=_freeze_array(matched_segments_all[matched_indices], dtype=np.float64),
            candidate_segments=_freeze_array(candidate_segments_all[candidate_indices], dtype=np.float64),
            matched_peak_ids=_freeze_array(matched_ids_all, dtype=np.int64),
            matched_total_count=int(len(matched_segments_all)),
            candidate_total_count=int(len(candidate_segments_all)),
            summary=MappingProxyType(pair_summary),
            matched_truncated=bool(len(matched_segments_all) > len(matched_indices)),
            candidate_truncated=bool(len(candidate_segments_all) > len(candidate_indices)),
        ))

    if not pair_previews:
        raise ValueError("В результате корреляции не найдено ни одной корректной пары ветвей")
    return BranchCorrelationPreview(
        source_branch_id=_safe_int(
            (getattr(correlation_bundle, "metadata", {}) or {}).get("branch_correlation", {}).get(
                "source_branch_id", pair_previews[0].source_branch_id
            ), pair_previews[0].source_branch_id,
        ),
        pairs=tuple(pair_previews),
        assignment_counts=MappingProxyType(assignment_counts),
        warnings=tuple(warnings),
    )


__all__ = [
    "BranchCorrelationPairPreview", "BranchCorrelationPreview",
    "build_branch_correlation_preview",
]
