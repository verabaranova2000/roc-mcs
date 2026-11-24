"""3D cross-correlation and directed branch decomposition.

The implementation extracts the reproducible pipeline from
``Из блокнота кросс-корреляция.ipynb`` (cells 6, 56, 57 and 59):

1. binary 3D cross-correlation on (theta_index, y, x) for coarse registration;
2. directed nearest-neighbour support in continuous (theta, y, x);
3. assignment of a source branch to one or several candidate partner branches;
4. one core-based coarse refinement per partner;
5. a robust continuous theta refinement and final assignment.

No input catalogue is modified. The directed nearest-neighbour rule is kept as
in the notebook: different source events may select the same partner event.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.fft import next_fast_len


_MAX_FFT_VOXELS = 8_000_000
_MAX_PARTNERS = 8
_REQUIRED_COLUMNS = {
    "peak_id", "branch_id", "theta_index", "theta", "x", "y",
}


@dataclass(frozen=True, slots=True)
class BranchCorrelationConfig:
    """Numerical settings, with units matching the branch catalogue."""

    # Required: this must come from the actual scan theta axis, never a guess.
    theta_step_arcsec: float
    xy_tolerance_px: float = 1.5
    theta_tolerance_arcsec: float = 45.0
    assignment_margin: float = 0.20

    def validated(self) -> "BranchCorrelationConfig":
        values = {
            "theta_step_arcsec": self.theta_step_arcsec,
            "xy_tolerance_px": self.xy_tolerance_px,
            "theta_tolerance_arcsec": self.theta_tolerance_arcsec,
            "assignment_margin": self.assignment_margin,
        }
        converted: dict[str, float] = {}
        for name, raw in values.items():
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} должен быть конечным числом") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} должен быть конечным числом")
            converted[name] = value
        if converted["theta_step_arcsec"] == 0.0:
            raise ValueError("theta_step_arcsec не может быть равен нулю")
        if converted["xy_tolerance_px"] <= 0.0:
            raise ValueError("xy_tolerance_px должен быть положительным")
        if converted["theta_tolerance_arcsec"] <= 0.0:
            raise ValueError("theta_tolerance_arcsec должен быть положительным")
        if not 0.0 <= converted["assignment_margin"] < 1.0:
            raise ValueError("assignment_margin должен находиться в диапазоне [0, 1)")
        return BranchCorrelationConfig(**converted)


@dataclass(frozen=True, slots=True)
class BranchCorrelationResult:
    """Tabular, immutable-by-convention result of one source/multi-partner run."""

    pairs: pd.DataFrame
    assignments: pd.DataFrame
    candidates: pd.DataFrame
    metadata: Mapping[str, Any]


def _integer_points(points: Any, *, name: str) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError(f"{name}: ожидался непустой массив N×3 (theta_index, y, x)")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: координаты должны быть конечными числами") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError(f"{name}: координаты должны быть конечными целыми индексами")
    if np.any(numeric < 0) or np.any(numeric > np.iinfo(np.int32).max):
        raise ValueError(f"{name}: индексы должны быть неотрицательными и помещаться в int32")
    return np.asarray(np.rint(numeric), dtype=np.int64)


def points_to_volume(
    points: Any,
    *,
    max_voxels: int = _MAX_FFT_VOXELS,
    name: str = "points",
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize integer (theta_index, y, x) points into a local binary volume.

    Duplicate events at the same grid position share one occupied voxel, exactly
    as in the notebook's assignment ``volume[tuple(local.T)] = 1``.
    """
    grid = _integer_points(points, name=name)
    origin = grid.min(axis=0)
    local = grid - origin
    shape = local.max(axis=0) + 1
    cells = int(prod(int(value) for value in shape))
    if cells > int(max_voxels):
        raise MemoryError(
            f"{name}: ограничивающий 3D-блок содержит {cells:,} ячеек; "
            f"лимит одного бинарного объёма — {int(max_voxels):,}. "
            "Выберите более компактные ветви/ROI или уменьшите диапазон данных."
        )
    volume = np.zeros(tuple(int(value) for value in shape), dtype=np.float32)
    volume[tuple(local.T)] = 1.0
    return volume, origin


def register_branches_3d_coarse(
    points_a: Any,
    points_b: Any,
    *,
    max_fft_voxels: int = _MAX_FFT_VOXELS,
) -> tuple[np.ndarray, float]:
    """Return integer translation for which ``B + shift`` aligns to ``A``.

    The maximum of the full binary 3D cross-correlation is used, preserving the
    notebook's registration definition. A conservative FFT-size guard avoids
    accidental multi-gigabyte temporary allocations for very extended branches.
    """
    volume_a, origin_a = points_to_volume(points_a, name="source branch")
    volume_b, origin_b = points_to_volume(points_b, name="partner branch")
    output_shape = tuple(
        int(a + b - 1) for a, b in zip(volume_a.shape, volume_b.shape)
    )
    fft_shape = tuple(next_fast_len(size) for size in output_shape)
    fft_cells = int(prod(fft_shape))
    if fft_cells > int(max_fft_voxels):
        raise MemoryError(
            "3D cross-correlation потребовала бы FFT-сетку "
            f"{fft_shape} ({fft_cells:,} ячеек); безопасный лимит — "
            f"{int(max_fft_voxels):,}. Разбейте анализ на компактные ROI/ветви."
        )
    correlation = fftconvolve(
        volume_a, volume_b[::-1, ::-1, ::-1], mode="full"
    )
    peak_index = np.asarray(
        np.unravel_index(int(np.argmax(correlation)), correlation.shape),
        dtype=np.int64,
    )
    shift_local = peak_index - (np.asarray(volume_b.shape, dtype=np.int64) - 1)
    shift_global = origin_a - origin_b + shift_local
    peak_value = float(correlation[tuple(peak_index)])
    return shift_global.astype(np.int64), peak_value


def apply_translation(
    points: Any,
    *,
    shift_theta_arcsec: float,
    dy: float = 0.0,
    dx: float = 0.0,
) -> np.ndarray:
    """Apply a continuous (theta, y, x) translation without changing the input."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points должны быть конечным массивом N×3 (theta, y, x)")
    shift = np.asarray([shift_theta_arcsec, dy, dx], dtype=np.float64)
    if not np.isfinite(shift).all():
        raise ValueError("Сдвиг (theta, y, x) должен быть конечным")
    return values + shift[None, :]


def directed_support(
    points_source: Any,
    points_target_aligned: Any,
    *,
    xy_tolerance_px: float = 1.5,
    theta_tolerance_arcsec: float = 45.0,
) -> dict[str, np.ndarray]:
    """Find the nearest partner event to every source event in scaled 3D space.

    ``delta = target_aligned - source`` is deliberately retained for auditing.
    Support requires all three coordinate-wise tolerances, not merely a small
    Euclidean distance. Matching is directed, not one-to-one: many source events
    may point to the same target event, as in the research notebook.
    """
    source = np.asarray(points_source, dtype=np.float64)
    target = np.asarray(points_target_aligned, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or not source.shape[0]:
        raise ValueError("points_source должен быть непустым массивом N×3")
    if target.ndim != 2 or target.shape[1] != 3 or not target.shape[0]:
        raise ValueError("points_target_aligned должен быть непустым массивом M×3")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Координаты для directed_support должны быть конечными")
    xy_tol = float(xy_tolerance_px)
    theta_tol = float(theta_tolerance_arcsec)
    if not np.isfinite(xy_tol) or xy_tol <= 0:
        raise ValueError("xy_tolerance_px должен быть конечным положительным числом")
    if not np.isfinite(theta_tol) or theta_tol <= 0:
        raise ValueError("theta_tolerance_arcsec должен быть конечным положительным числом")

    scale = np.asarray([theta_tol, xy_tol, xy_tol], dtype=np.float64)
    tree = cKDTree(target / scale)
    score, matched_idx = tree.query(source / scale, k=1)
    matched_idx = np.asarray(matched_idx, dtype=np.int64)
    delta = target[matched_idx] - source
    supported = (
        (np.abs(delta[:, 0]) <= theta_tol)
        & (np.abs(delta[:, 1]) <= xy_tol)
        & (np.abs(delta[:, 2]) <= xy_tol)
    )
    return {
        "score": np.asarray(score, dtype=np.float64),
        "matched_idx": matched_idx,
        "delta": np.asarray(delta, dtype=np.float64),
        "supported": np.asarray(supported, dtype=bool),
    }


def _assign_candidates(
    supports: Sequence[Mapping[str, np.ndarray]],
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generalize the notebook's two-partner assignment to N partners.

    Return selected partner index (-1 if none), ambiguous, residual, and the
    number of supported candidate partners per source event. A partner wins
    only if its normalized nearest-neighbour score is strictly better than the
    second supported candidate by the requested relative margin.
    """
    if not supports:
        raise ValueError("Нужна хотя бы одна ветвь-партнёр")
    supported = np.column_stack([item["supported"] for item in supports])
    scores = np.column_stack([item["score"] for item in supports])
    n_source = supported.shape[0]
    selected = np.full(n_source, -1, dtype=np.int64)
    ambiguous = np.zeros(n_source, dtype=bool)
    residual = ~np.any(supported, axis=1)
    n_supported = np.count_nonzero(supported, axis=1).astype(np.int64)

    for row in np.flatnonzero(n_supported):
        choices = np.flatnonzero(supported[row])
        if choices.size == 1:
            selected[row] = int(choices[0])
            continue
        order = choices[np.argsort(scores[row, choices], kind="stable")]
        best, second = int(order[0]), int(order[1])
        if float(scores[row, best]) < (1.0 - margin) * float(scores[row, second]):
            selected[row] = best
        else:
            ambiguous[row] = True
    return selected, ambiguous, residual, n_supported


def _extract_branch_rows(catalog: pd.DataFrame, branch_id: int) -> pd.DataFrame:
    subset = catalog.loc[catalog["branch_id"].to_numpy(dtype=np.int64) == int(branch_id)].copy()
    if subset.empty:
        raise ValueError(f"Ветка branch_id={branch_id} отсутствует в branch_catalog")
    subset = subset.sort_values("peak_id", kind="mergesort").reset_index(drop=True)
    return subset


def _validated_catalog(catalog: Any) -> pd.DataFrame:
    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("branch_catalog должен быть pandas.DataFrame")
    missing = _REQUIRED_COLUMNS.difference(catalog.columns)
    if missing:
        raise ValueError(f"branch_catalog: отсутствуют обязательные столбцы {sorted(missing)}")
    if catalog.empty:
        raise ValueError("branch_catalog пуст: нет событий для кросс-корреляции")

    frame = catalog.copy(deep=False)
    numeric: dict[str, np.ndarray] = {}
    for column in _REQUIRED_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        if not np.isfinite(values).all():
            raise ValueError(f"branch_catalog[{column!r}] содержит NaN/Inf или нечисловые значения")
        numeric[column] = values
    for column in ("peak_id", "branch_id", "theta_index", "x", "y"):
        values = numeric[column]
        if not np.equal(values, np.rint(values)).all():
            raise ValueError(f"branch_catalog[{column!r}] должен содержать целые значения")
    if np.any(numeric["peak_id"] < 0) or np.any(numeric["branch_id"] < 0):
        raise ValueError("peak_id и branch_id должны быть неотрицательными")
    if np.any(numeric["x"] < 0) or np.any(numeric["y"] < 0) or np.any(numeric["theta_index"] < 0):
        raise ValueError("theta_index, x и y должны быть неотрицательными индексами")
    if np.unique(numeric["peak_id"]).size != len(frame):
        raise ValueError("peak_id должен быть уникальным в branch_catalog")
    frame = frame.copy()
    for column in ("peak_id", "branch_id", "theta_index", "x", "y"):
        frame[column] = np.rint(numeric[column]).astype(np.int64)
    frame["theta"] = numeric["theta"].astype(np.float64)
    return frame


def _branch_id(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} должен быть неотрицательным целым branch_id")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} должен быть неотрицательным целым branch_id") from exc
    if not np.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"{name} должен быть неотрицательным целым branch_id")
    return int(number)


def correlate_branch_catalog(
    peak_catalog: pd.DataFrame,
    *,
    source_branch_id: int,
    partner_branch_ids: Sequence[int],
    config: BranchCorrelationConfig,
) -> BranchCorrelationResult:
    """Register one source branch against one or more manually selected partners."""
    if not isinstance(config, BranchCorrelationConfig):
        raise TypeError("config должен быть явным BranchCorrelationConfig с шагом θ из исходного скана")
    cfg = config.validated()
    catalog = _validated_catalog(peak_catalog)
    source_id = _branch_id(source_branch_id, name="source_branch_id")
    partners = [
        _branch_id(value, name=f"partner_branch_ids[{index}]")
        for index, value in enumerate(partner_branch_ids)
    ]
    if not partners:
        raise ValueError("Укажите хотя бы одну ветвь-партнёр")
    if len(partners) > _MAX_PARTNERS:
        raise ValueError(f"За один расчёт допускается не более {_MAX_PARTNERS} ветвей-партнёров")
    if source_id in partners:
        raise ValueError("Исходная ветвь не может одновременно быть партнёром")
    if len(set(partners)) != len(partners):
        raise ValueError("Список partner_branch_ids содержит повторяющиеся branch_id")

    available = set(catalog["branch_id"].unique().tolist())
    missing_ids = [value for value in [source_id, *partners] if value not in available]
    if missing_ids:
        raise ValueError(f"В branch_catalog отсутствуют branch_id: {missing_ids}")

    source_frame = _extract_branch_rows(catalog, source_id)
    source_grid = source_frame[["theta_index", "y", "x"]].to_numpy(dtype=np.int64)
    source_xyz = source_frame[["theta", "y", "x"]].to_numpy(dtype=np.float64)
    partner_frames = [_extract_branch_rows(catalog, branch_id) for branch_id in partners]
    partner_grids = [
        frame[["theta_index", "y", "x"]].to_numpy(dtype=np.int64)
        for frame in partner_frames
    ]
    partner_xyz = [
        frame[["theta", "y", "x"]].to_numpy(dtype=np.float64)
        for frame in partner_frames
    ]

    # 1. Independent coarse 3D registration, following notebook cells 57/59.
    coarse_shifts: list[np.ndarray] = []
    coarse_peaks: list[float] = []
    aligned_coarse: list[np.ndarray] = []
    supports_initial: list[dict[str, np.ndarray]] = []
    for grid, positions in zip(partner_grids, partner_xyz):
        shift, corr_peak = register_branches_3d_coarse(source_grid, grid)
        coarse_shifts.append(shift)
        coarse_peaks.append(corr_peak)
        aligned = apply_translation(
            positions,
            shift_theta_arcsec=float(shift[0]) * cfg.theta_step_arcsec,
            dy=float(shift[1]), dx=float(shift[2]),
        )
        aligned_coarse.append(aligned)
        supports_initial.append(directed_support(
            source_xyz, aligned,
            xy_tolerance_px=cfg.xy_tolerance_px,
            theta_tolerance_arcsec=cfg.theta_tolerance_arcsec,
        ))
    initial_assignment, _, _, _ = _assign_candidates(supports_initial, cfg.assignment_margin)

    # 2. Core-based coarse registration refinement per candidate partner.
    refined_shifts: list[np.ndarray] = []
    refined_peaks: list[float] = []
    refinement_status: list[str] = []
    supports_refined: list[dict[str, np.ndarray]] = []
    aligned_refined: list[np.ndarray] = []
    for partner_index, (grid, positions, initial_support, shift_initial) in enumerate(
        zip(partner_grids, partner_xyz, supports_initial, coarse_shifts)
    ):
        source_core_idx = np.flatnonzero(initial_assignment == partner_index)
        if source_core_idx.size == 0:
            shift_refined = shift_initial.copy()
            corr_refined = coarse_peaks[partner_index]
            status = "coarse_only_no_exclusive_core"
        else:
            target_core_idx = np.unique(initial_support["matched_idx"][source_core_idx])
            if target_core_idx.size == 0:
                shift_refined = shift_initial.copy()
                corr_refined = coarse_peaks[partner_index]
                status = "coarse_only_no_target_core"
            else:
                shift_refined, corr_refined = register_branches_3d_coarse(
                    source_grid[source_core_idx], grid[target_core_idx]
                )
                status = "refined_on_exclusive_core"
        refined_shifts.append(np.asarray(shift_refined, dtype=np.int64))
        refined_peaks.append(float(corr_refined))
        refinement_status.append(status)
        aligned = apply_translation(
            positions,
            shift_theta_arcsec=float(shift_refined[0]) * cfg.theta_step_arcsec,
            dy=float(shift_refined[1]), dx=float(shift_refined[2]),
        )
        aligned_refined.append(aligned)
        supports_refined.append(directed_support(
            source_xyz, aligned,
            xy_tolerance_px=cfg.xy_tolerance_px,
            theta_tolerance_arcsec=cfg.theta_tolerance_arcsec,
        ))

    # 3. Reassign after spatial/integer refinement, then estimate sub-bin theta.
    refined_assignment, _, _, _ = _assign_candidates(supports_refined, cfg.assignment_margin)
    theta_corrections: list[float] = []
    theta_mads: list[float] = []
    theta_shifts_final: list[float] = []
    for partner_index, support in enumerate(supports_refined):
        selected = (refined_assignment == partner_index) & support["supported"]
        delta_theta = support["delta"][selected, 0]
        if delta_theta.size:
            median_residual = float(np.median(delta_theta))
            mad = float(np.median(np.abs(delta_theta - median_residual)))
            # delta = target_aligned - source. We translate TARGET, so the
            # correction must have the opposite sign to cancel this residual.
            correction = -median_residual
        else:
            correction = 0.0
            mad = float("nan")
        theta_corrections.append(float(correction))
        theta_mads.append(float(mad))
        theta_shifts_final.append(
            float(refined_shifts[partner_index][0]) * cfg.theta_step_arcsec + float(correction)
        )

    # 4. Final transforms, final directed support and final partner assignment.
    aligned_final: list[np.ndarray] = []
    supports_final: list[dict[str, np.ndarray]] = []
    for partner_index, positions in enumerate(partner_xyz):
        shift = refined_shifts[partner_index]
        aligned = apply_translation(
            positions,
            shift_theta_arcsec=theta_shifts_final[partner_index],
            dy=float(shift[1]), dx=float(shift[2]),
        )
        aligned_final.append(aligned)
        supports_final.append(directed_support(
            source_xyz, aligned,
            xy_tolerance_px=cfg.xy_tolerance_px,
            theta_tolerance_arcsec=cfg.theta_tolerance_arcsec,
        ))
    final_assignment, ambiguous, residual, n_supported = _assign_candidates(
        supports_final, cfg.assignment_margin
    )

    source_peak_ids = source_frame["peak_id"].to_numpy(dtype=np.int64)
    partner_peak_ids = [frame["peak_id"].to_numpy(dtype=np.int64) for frame in partner_frames]

    # One compact row per source event; partner is nullable for ambiguous/residual events.
    assignment_partner_ids: list[int | None] = []
    assignment_matched_ids: list[int | None] = []
    best_scores = np.full(len(source_frame), np.nan, dtype=np.float64)
    for row, partner_index in enumerate(final_assignment):
        if partner_index >= 0:
            support = supports_final[int(partner_index)]
            target_index = int(support["matched_idx"][row])
            assignment_partner_ids.append(partners[int(partner_index)])
            assignment_matched_ids.append(int(partner_peak_ids[int(partner_index)][target_index]))
            best_scores[row] = float(support["score"][row])
        else:
            assignment_partner_ids.append(None)
            assignment_matched_ids.append(None)
    assignment_class = np.where(
        final_assignment >= 0, "partner", np.where(ambiguous, "ambiguous", "residual")
    )
    assignments = pd.DataFrame({
        "source_peak_id": source_peak_ids,
        "source_branch_id": np.full(len(source_frame), source_id, dtype=np.int64),
        "assignment_class": assignment_class,
        "assigned_partner_branch_id": pd.array(assignment_partner_ids, dtype="Int64"),
        "matched_partner_peak_id": pd.array(assignment_matched_ids, dtype="Int64"),
        "supported_partner_count": n_supported,
        "best_normalized_distance": best_scores,
    })

    candidate_frames: list[pd.DataFrame] = []
    pair_rows: list[dict[str, Any]] = []
    for partner_index, partner_id in enumerate(partners):
        support = supports_final[partner_index]
        matched_ids = partner_peak_ids[partner_index][support["matched_idx"]]
        selected_for_partner = final_assignment == partner_index
        source_status = assignment_class
        candidate_status = np.where(
            ~support["supported"], "not_supported",
            np.where(selected_for_partner, "assigned",
                     np.where(ambiguous, "ambiguous", "supported_other_partner")),
        )
        candidate_frames.append(pd.DataFrame({
            "source_peak_id": source_peak_ids,
            "source_branch_id": np.full(len(source_frame), source_id, dtype=np.int64),
            "partner_branch_id": np.full(len(source_frame), partner_id, dtype=np.int64),
            "matched_partner_peak_id": matched_ids.astype(np.int64),
            "normalized_distance": support["score"],
            "delta_theta_arcsec": support["delta"][:, 0],
            "delta_y_px": support["delta"][:, 1],
            "delta_x_px": support["delta"][:, 2],
            "supported": support["supported"],
            "assigned": selected_for_partner,
            "assignment_class": source_status,
            "candidate_status": candidate_status,
        }))
        pair_rows.append({
            "source_branch_id": source_id,
            "partner_branch_id": partner_id,
            "source_event_count": int(len(source_frame)),
            "partner_event_count": int(len(partner_frames[partner_index])),
            "coarse_shift_theta_index": int(coarse_shifts[partner_index][0]),
            "coarse_shift_y_px": int(coarse_shifts[partner_index][1]),
            "coarse_shift_x_px": int(coarse_shifts[partner_index][2]),
            "refined_shift_theta_index": int(refined_shifts[partner_index][0]),
            "refined_shift_y_px": int(refined_shifts[partner_index][1]),
            "refined_shift_x_px": int(refined_shifts[partner_index][2]),
            "theta_step_arcsec": float(cfg.theta_step_arcsec),
            "theta_shift_coarse_arcsec": float(coarse_shifts[partner_index][0]) * cfg.theta_step_arcsec,
            "theta_shift_refined_arcsec": float(refined_shifts[partner_index][0]) * cfg.theta_step_arcsec,
            "theta_correction_arcsec": theta_corrections[partner_index],
            "theta_shift_final_arcsec": theta_shifts_final[partner_index],
            "theta_residual_mad_arcsec": theta_mads[partner_index],
            "coarse_correlation_peak": coarse_peaks[partner_index],
            "refined_correlation_peak": refined_peaks[partner_index],
            "coarse_overlap_fraction": (
                coarse_peaks[partner_index]
                / max(1, min(np.unique(source_grid, axis=0).shape[0], np.unique(partner_grids[partner_index], axis=0).shape[0]))
            ),
            "initial_supported_count": int(np.count_nonzero(supports_initial[partner_index]["supported"])),
            "refined_supported_count": int(np.count_nonzero(supports_refined[partner_index]["supported"])),
            "final_supported_count": int(np.count_nonzero(support["supported"])),
            "assigned_source_count": int(np.count_nonzero(selected_for_partner)),
            "refinement_status": refinement_status[partner_index],
        })

    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    pairs = pd.DataFrame(pair_rows)
    summary = {
        "source_branch_id": source_id,
        "partner_branch_ids": partners,
        "source_event_count": int(len(source_frame)),
        "partner_count": int(len(partners)),
        "assigned_event_count": int(np.count_nonzero(final_assignment >= 0)),
        "ambiguous_event_count": int(np.count_nonzero(ambiguous)),
        "residual_event_count": int(np.count_nonzero(residual)),
        "theta_step_arcsec": float(cfg.theta_step_arcsec),
        "xy_tolerance_px": float(cfg.xy_tolerance_px),
        "theta_tolerance_arcsec": float(cfg.theta_tolerance_arcsec),
        "assignment_margin": float(cfg.assignment_margin),
        "matching_rule": "directed_nearest_neighbor_many_to_one",
        "assignment_rule": "unique_support_or_relative_best_score_margin",
        "theta_correction_sign": "negative_median(target_aligned_minus_source)",
        "algorithm_version": 1,
    }
    return BranchCorrelationResult(
        pairs=pairs,
        assignments=assignments,
        candidates=candidates,
        metadata=summary,
    )


__all__ = [
    "BranchCorrelationConfig", "BranchCorrelationResult", "points_to_volume",
    "register_branches_3d_coarse", "apply_translation", "directed_support",
    "correlate_branch_catalog",
]
