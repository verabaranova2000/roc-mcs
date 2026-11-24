"""3D cross-correlation and directed branch decomposition.

The production coarse-registration path computes the full linear cross-correlation
of binary occupancy volumes.  It uses a real-valued float64 FFT with an explicit
memory estimate; it does not reject a volume because an arbitrary voxel-count
threshold was crossed.  A bounded exact sparse vote is retained only for tiny
point sets when the FFT estimate does not fit the available memory.

The event-level pipeline (directed nearest-neighbour support, multi-partner
assignment, exclusive-core refinement and robust sub-bin theta correction) is
kept separate from registration and preserves the notebook's matching semantics.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil, prod, sqrt
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.spatial import cKDTree

# Sparse translation voting has O(Na*Nb) work.  It is intentionally a small-data
# fallback, never the path for normal 35k x 35k diffraction branches.
_MAX_SPARSE_CORRELATION_PAIRS = 2_000_000
_SPARSE_CHUNK_PAIRS = 100_000
_DEFAULT_UNKNOWN_MEMORY_BUDGET = 1024**3
_MEMORY_SAFETY_FACTOR = 1.25
_MEMORY_FIXED_OVERHEAD = 32 * 1024**2
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


def _available_memory_bytes() -> int | None:
    """Return currently available physical RAM without making psutil mandatory."""
    try:
        import psutil  # type: ignore[import-not-found]
        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                value = int(status.ullAvailPhys)
                if value > 0:
                    return value
        except Exception:
            pass

    # Linux exposes MemAvailable, which includes reclaimable cache and is a better
    # measure than SC_AVPHYS_PAGES for deciding whether a transform can be run.
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        value = pages * page_size
        return value if value > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def _fft_memory_estimate_bytes(
    shape_a: Sequence[int], shape_b: Sequence[int], fft_shape: Sequence[int]
) -> int:
    """Conservative peak working-set estimate for the float64 real FFT path.

    Counts both dense input volumes, two complex half-spectra, the inverse real
    array, an additional full real-size scratch allowance, fixed FFT/allocator
    overhead, and a 25% safety factor.  This deliberately overestimates the
    measured working set rather than using an arbitrary voxel-count cutoff.
    """
    volume_cells = int(prod(int(v) for v in shape_a)) + int(prod(int(v) for v in shape_b))
    real_cells = int(prod(int(v) for v in fft_shape))
    spectrum_cells = int(prod(int(v) for v in fft_shape[:-1])) * (int(fft_shape[-1]) // 2 + 1)
    volume_bytes = volume_cells * np.dtype(np.float64).itemsize
    real_bytes = real_cells * np.dtype(np.float64).itemsize
    spectrum_bytes = spectrum_cells * np.dtype(np.complex128).itemsize
    live_bytes = volume_bytes + (2 * spectrum_bytes) + (2 * real_bytes)
    return int(ceil(live_bytes * _MEMORY_SAFETY_FACTOR + _MEMORY_FIXED_OVERHEAD))


def _automatic_fft_memory_budget_bytes(available_bytes: int | None = None) -> int:
    available = _available_memory_bytes() if available_bytes is None else int(available_bytes)
    if available is None or available <= 0:
        return _DEFAULT_UNKNOWN_MEMORY_BUDGET
    # Leave 30% of currently available RAM for the OS, GUI, scan arrays and
    # concurrent application tasks.  There is no fixed 8-million-voxel ceiling.
    return max(0, int(available * 0.70))


def points_to_volume(
    points: Any,
    *,
    max_voxels: int | None = None,
    name: str = "points",
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize integer (theta_index, y, x) points into a local binary volume.

    Duplicate events at the same grid position share one occupied voxel, exactly
    as in the research notebook.  ``max_voxels`` is an optional caller-supplied
    guard for standalone use; production registration uses its memory estimator.
    """
    grid = _integer_points(points, name=name)
    origin = grid.min(axis=0)
    local = grid - origin
    shape = local.max(axis=0) + 1
    cells = int(prod(int(value) for value in shape))
    if max_voxels is not None and cells > int(max_voxels):
        raise MemoryError(
            f"{name}: ограничивающий 3D-блок содержит {cells:,} ячеек; "
            f"запрошенный лимит — {int(max_voxels):,}."
        )
    volume = np.zeros(tuple(int(value) for value in shape), dtype=np.float64)
    volume[tuple(local.T)] = 1.0
    return volume, origin


def _sparse_translation_vote(
    local_a: np.ndarray,
    local_b: np.ndarray,
    *,
    shape_b: np.ndarray,
    output_shape: tuple[int, int, int],
    max_pairs: int = _MAX_SPARSE_CORRELATION_PAIRS,
    chunk_pairs: int = _SPARSE_CHUNK_PAIRS,
) -> tuple[np.ndarray, float]:
    """Exact sparse equivalent of binary-volume cross-correlation for tiny clouds."""
    a = np.unique(np.asarray(local_a, dtype=np.int64), axis=0)
    b = np.unique(np.asarray(local_b, dtype=np.int64), axis=0)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Нельзя регистрировать пустую ветвь")
    pair_count = int(len(a)) * int(len(b))
    output_cells = int(prod(int(v) for v in output_shape))
    if output_cells > np.iinfo(np.int64).max:
        raise MemoryError(
            "Размер корреляционного результата превышает диапазон безопасной "
            "индексации int64 для sparse fallback."
        )
    if pair_count > int(max_pairs):
        raise MemoryError(
            f"Плотная FFT не помещается в доступный бюджет, а точный sparse-подсчёт "
            f"потребовал бы {pair_count:,} пар при лимите {int(max_pairs):,}. "
            "Увеличьте доступную память/освободите память и повторите расчёт; "
            "пики не прореживались и результат не приближался."
        )

    chunk_pairs = max(1, int(chunk_pairs))
    block_a = max(1, min(len(a), int(sqrt(chunk_pairs * len(a) / max(1, len(b))))))
    block_b = max(1, min(len(b), chunk_pairs // block_a))
    votes: Counter[int] = Counter()
    offset = np.asarray(shape_b, dtype=np.int64) - 1
    out_y, out_x = int(output_shape[1]), int(output_shape[2])
    for a_start in range(0, len(a), block_a):
        a_block = a[a_start:a_start + block_a]
        for b_start in range(0, len(b), block_b):
            b_block = b[b_start:b_start + block_b]
            differences = a_block[:, None, :] - b_block[None, :, :]
            correlation_indices = differences.reshape(-1, 3) + offset[None, :]
            flat = (
                (correlation_indices[:, 0] * out_y + correlation_indices[:, 1])
                * out_x + correlation_indices[:, 2]
            )
            unique, counts = np.unique(flat, return_counts=True)
            votes.update({int(key): int(count) for key, count in zip(unique, counts)})
    maximum = max(votes.values())
    best_flat = min(key for key, count in votes.items() if count == maximum)
    peak_index = np.asarray(np.unravel_index(best_flat, output_shape), dtype=np.int64)
    return peak_index - offset, float(maximum)


def _register_branches_3d_coarse(
    points_a: Any,
    points_b: Any,
    *,
    max_fft_memory_bytes: int | None = None,
    max_sparse_pairs: int = _MAX_SPARSE_CORRELATION_PAIRS,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Detailed internal registration; the public wrapper preserves the old API."""
    grid_a = _integer_points(points_a, name="source branch")
    grid_b = _integer_points(points_b, name="partner branch")
    origin_a = grid_a.min(axis=0)
    origin_b = grid_b.min(axis=0)
    local_a = np.unique(grid_a - origin_a, axis=0)
    local_b = np.unique(grid_b - origin_b, axis=0)
    shape_a = local_a.max(axis=0) + 1
    shape_b = local_b.max(axis=0) + 1
    output_shape = tuple(int(a + b - 1) for a, b in zip(shape_a, shape_b))
    # Match scipy.signal.fftconvolve's real-transform padding convention.
    fft_shape = tuple(next_fast_len(size, real=True) for size in output_shape)
    estimated_bytes = _fft_memory_estimate_bytes(shape_a, shape_b, fft_shape)
    available_bytes = _available_memory_bytes()
    if max_fft_memory_bytes is None:
        memory_budget = _automatic_fft_memory_budget_bytes(available_bytes)
        if memory_budget <= 0:
            raise MemoryError(
                "Операционная система не сообщает доступную физическую память; "
                "невозможно безопасно выделить рабочие массивы полной 3D FFT."
            )
    else:
        if isinstance(max_fft_memory_bytes, (bool, np.bool_)):
            raise ValueError("max_fft_memory_bytes должен быть положительным целым числом байтов")
        try:
            memory_budget = int(max_fft_memory_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_fft_memory_bytes должен быть положительным целым числом байтов") from exc
        if memory_budget <= 0 or memory_budget != max_fft_memory_bytes:
            raise ValueError("max_fft_memory_bytes должен быть положительным целым числом байтов")
    if int(max_sparse_pairs) < 0:
        raise ValueError("max_sparse_pairs не может быть отрицательным")

    diagnostics: dict[str, Any] = {
        "method": "fft_float64",
        "source_unique_voxels": int(len(local_a)),
        "partner_unique_voxels": int(len(local_b)),
        "source_bbox_theta_y_x": [int(v) for v in shape_a],
        "partner_bbox_theta_y_x": [int(v) for v in shape_b],
        "correlation_shape_theta_y_x": [int(v) for v in output_shape],
        "fft_shape_theta_y_x": [int(v) for v in fft_shape],
        "estimated_peak_memory_bytes": int(estimated_bytes),
        "available_memory_bytes": None if available_bytes is None else int(available_bytes),
        "memory_budget_bytes": int(memory_budget),
        "sparse_pair_count": int(len(local_a)) * int(len(local_b)),
    }

    if estimated_bytes <= memory_budget:
        volume_a = np.zeros(tuple(int(v) for v in shape_a), dtype=np.float64)
        volume_b = np.zeros(tuple(int(v) for v in shape_b), dtype=np.float64)
        volume_a[tuple(local_a.T)] = 1.0
        volume_b[tuple(local_b.T)] = 1.0

        # Explicit real FFT: one complex half-spectrum per input.  Multiply the
        # first spectrum in-place and release the second before the inverse FFT.
        # This bounds transient allocations and keeps the calculation float64.
        try:
            spectrum_a = rfftn(volume_a, s=fft_shape, workers=1)
            spectrum_b = rfftn(volume_b[::-1, ::-1, ::-1], s=fft_shape, workers=1)
            spectrum_a *= spectrum_b
            del spectrum_b, volume_a, volume_b
            correlation = irfftn(
                spectrum_a, s=fft_shape, workers=1, overwrite_x=True
            )
            del spectrum_a
        except MemoryError as exc:
            raise MemoryError(
                "Фактическое выделение памяти для полной float64 FFT не удалось, "
                f"хотя предварительная оценка составляла {estimated_bytes / (1024**3):.2f} GiB "
                f"при бюджете {memory_budget / (1024**3):.2f} GiB. "
                "Возможно, свободная память уменьшилась после проверки. "
                "Закройте другие тяжёлые приложения и повторите расчёт."
            ) from exc

        valid = correlation[tuple(slice(0, size) for size in output_shape)]
        # For binary occupancy, every exact correlation coefficient is an
        # integer. Round only floating FFT noise before deterministic argmax.
        np.rint(valid, out=valid)
        peak_flat = int(np.argmax(valid))
        peak_index = np.asarray(np.unravel_index(peak_flat, output_shape), dtype=np.int64)
        shift_local = peak_index - (shape_b.astype(np.int64) - 1)
        peak_value = float(valid[tuple(peak_index)])
    else:
        pair_count = int(len(local_a)) * int(len(local_b))
        if pair_count > int(max_sparse_pairs):
            available_text = (
                f"{available_bytes / (1024**3):.2f} GiB"
                if available_bytes is not None else "не удалось определить"
            )
            raise MemoryError(
                "Полная 3D-кросс-корреляция не запускалась: расчётная пиковая "
                f"память float64 FFT ≈ {estimated_bytes / (1024**3):.2f} GiB, "
                f"доступная физическая память — {available_text}, "
                f"рабочий бюджет — {memory_budget / (1024**3):.2f} GiB. "
                f"Точный sparse-вариант потребовал бы {pair_count:,} пар "
                f"(лимит {int(max_sparse_pairs):,}). Данные не изменены. "
                "Закройте другие тяжёлые приложения/освободите RAM и повторите расчёт."
            )
        shift_local, peak_value = _sparse_translation_vote(
            local_a,
            local_b,
            shape_b=shape_b,
            output_shape=output_shape,
            max_pairs=max_sparse_pairs,
        )
        diagnostics["method"] = "sparse_exact_vote"

    shift_global = origin_a - origin_b + shift_local
    return shift_global.astype(np.int64), float(peak_value), diagnostics


def register_branches_3d_coarse(
    points_a: Any,
    points_b: Any,
    *,
    max_fft_memory_bytes: int | None = None,
    max_sparse_pairs: int = _MAX_SPARSE_CORRELATION_PAIRS,
) -> tuple[np.ndarray, float]:
    """Return the integer translation to apply to B to align it with A.

    Uses the full binary 3D cross-correlation.  By default the dense float64 FFT
    is selected from a conservative peak-memory estimate and current available
    RAM.  An exact sparse vote is used only when the point-pair count is small.
    ``max_fft_memory_bytes`` is an optional explicit budget for tests and advanced
    callers; no fixed voxel-count ceiling is imposed.
    """
    shift, peak, _ = _register_branches_3d_coarse(
        points_a,
        points_b,
        max_fft_memory_bytes=max_fft_memory_bytes,
        max_sparse_pairs=max_sparse_pairs,
    )
    return shift, peak


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
    coarse_diagnostics: list[dict[str, Any]] = []
    aligned_coarse: list[np.ndarray] = []
    supports_initial: list[dict[str, np.ndarray]] = []
    for grid, positions in zip(partner_grids, partner_xyz):
        shift, corr_peak, registration_info = _register_branches_3d_coarse(source_grid, grid)
        coarse_shifts.append(shift)
        coarse_peaks.append(corr_peak)
        coarse_diagnostics.append(registration_info)
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
    refined_diagnostics: list[dict[str, Any] | None] = []
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
            refinement_info = None
        else:
            target_core_idx = np.unique(initial_support["matched_idx"][source_core_idx])
            if target_core_idx.size == 0:
                shift_refined = shift_initial.copy()
                corr_refined = coarse_peaks[partner_index]
                status = "coarse_only_no_target_core"
                refinement_info = None
            else:
                shift_refined, corr_refined, refinement_info = _register_branches_3d_coarse(
                    source_grid[source_core_idx], grid[target_core_idx]
                )
                status = "refined_on_exclusive_core"
        refined_shifts.append(np.asarray(shift_refined, dtype=np.int64))
        refined_peaks.append(float(corr_refined))
        refinement_status.append(status)
        refined_diagnostics.append(refinement_info)
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
            "coarse_registration_method": coarse_diagnostics[partner_index]["method"],
            "coarse_correlation_shape_theta_y_x": "×".join(
                str(v) for v in coarse_diagnostics[partner_index]["correlation_shape_theta_y_x"]
            ),
            "coarse_fft_shape_theta_y_x": "×".join(
                str(v) for v in coarse_diagnostics[partner_index]["fft_shape_theta_y_x"]
            ),
            "coarse_fft_estimated_peak_memory_mb": (
                coarse_diagnostics[partner_index]["estimated_peak_memory_bytes"] / (1024.0 ** 2)
            ),
            "coarse_fft_available_memory_mb": (
                float("nan") if coarse_diagnostics[partner_index]["available_memory_bytes"] is None
                else coarse_diagnostics[partner_index]["available_memory_bytes"] / (1024.0 ** 2)
            ),
            "coarse_unique_source_voxels": coarse_diagnostics[partner_index]["source_unique_voxels"],
            "coarse_unique_partner_voxels": coarse_diagnostics[partner_index]["partner_unique_voxels"],
            "refined_registration_method": (
                None if refined_diagnostics[partner_index] is None
                else refined_diagnostics[partner_index]["method"]
            ),
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
        "algorithm_version": 2,
        "coarse_registration_backend": "adaptive_float64_real_fft_with_exact_sparse_fallback",
        "fft_precision": "float64",
        "registration_uses_all_unique_occupied_voxels": True,
        "registration_diagnostics": [
            {
                "partner_branch_id": int(partner_id),
                "coarse_method": coarse_diagnostics[index]["method"],
                "correlation_shape_theta_y_x": coarse_diagnostics[index]["correlation_shape_theta_y_x"],
                "fft_shape_theta_y_x": coarse_diagnostics[index]["fft_shape_theta_y_x"],
                "estimated_peak_memory_bytes": coarse_diagnostics[index]["estimated_peak_memory_bytes"],
                "available_memory_bytes": coarse_diagnostics[index]["available_memory_bytes"],
                "source_unique_voxels": coarse_diagnostics[index]["source_unique_voxels"],
                "partner_unique_voxels": coarse_diagnostics[index]["partner_unique_voxels"],
                "refinement_method": (
                    None if refined_diagnostics[index] is None else refined_diagnostics[index]["method"]
                ),
            }
            for index, partner_id in enumerate(partners)
        ],
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
