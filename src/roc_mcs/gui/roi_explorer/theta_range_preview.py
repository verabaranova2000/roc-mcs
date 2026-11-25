"""Чистые численные и семантические операции предпросмотра диапазона θ.

Модуль намеренно не зависит от Qt: распознавание карты, выбор комплементарного
интервала и редукция θ-стека можно проверять отдельно от интерфейса.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np


THETA_RANGE_PREVIEW_INTERVAL_MS = 25

_PROJECTION_ALIASES = {
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "max": "max",
    "maximum": "max",
    "sum": "sum",
}
_FORBIDDEN_TOKENS = {
    "peak", "peaks", "spike", "spikes", "branch", "branches",
    "fit", "fitted", "count", "counts", "residual", "score",
    "correlation", "correspondence", "mask", "valid", "roi",
}


def infer_theta_projection_kind(*identifiers: object) -> str | None:
    """Return ``mean``/``max``/``sum`` only for an unambiguous projection map.

    Accepted identifiers are deliberately conservative: exact reduction names
    (``mean``, ``max``, ``sum``), explicit ``* projection`` labels, and explicit
    scan intensity keys such as ``scan.measured_intensity_sum``. A processing
    result (peak/spike/branch/fit/count/etc.) always vetoes classification.
    Unknown labels stay disabled rather than being guessed.
    """
    found: set[str] = set()
    for identifier in identifiers:
        if identifier is None:
            continue
        text = str(identifier).strip().lower()
        if not text:
            continue
        tokens = re.findall(r"[a-z0-9]+", text)
        token_set = set(tokens)
        if token_set & _FORBIDDEN_TOKENS:
            return None

        normalized = " ".join(tokens)
        exact = _PROJECTION_ALIASES.get(normalized)
        if exact is not None:
            found.add(exact)
            continue

        # "mean projection", "projection max", "sum θ projection", etc.
        if "projection" in token_set:
            reductions = {
                _PROJECTION_ALIASES[token]
                for token in token_set
                if token in _PROJECTION_ALIASES
            }
            if len(reductions) == 1:
                found.update(reductions)
                continue

        # Workflow's explicit input-layer naming convention. Require both
        # "scan" and "intensity" so unrelated processing keys cannot match.
        if "scan" in token_set and "intensity" in token_set:
            reductions = {
                _PROJECTION_ALIASES[token]
                for token in token_set
                if token in _PROJECTION_ALIASES
            }
            if len(reductions) == 1:
                found.update(reductions)

    return next(iter(found)) if len(found) == 1 else None


def normalize_theta_axis(axis: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Validate a finite, strictly monotonic θ-axis and return ascending order.

    Returns ``(ascending_values, source_indices)``; the second array maps the
    ascending order back to the source stack's θ dimension.
    """
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("Для предпросмотра диапазона θ нужны минимум два отсчёта.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Ось θ содержит нечисловые или бесконечные значения.")
    differences = np.diff(values)
    if np.all(differences > 0):
        order = np.arange(values.size, dtype=np.int64)
    elif np.all(differences < 0):
        order = np.arange(values.size - 1, -1, -1, dtype=np.int64)
    else:
        raise ValueError("Ось θ должна быть строго монотонной без повторов.")
    return values[order].copy(), order


def complementary_theta_range(
    low_index: int, high_index: int, count: int
) -> tuple[int, int] | None:
    """Return the opposite one-sided interval, sharing the cut boundary.

    Full range and two-sided interior windows have no single well-defined
    complement in this UI and therefore return ``None``.
    """
    n = int(count)
    low, high = int(low_index), int(high_index)
    if n < 2 or not (0 <= low < high < n):
        return None
    if low == 0 and high < n - 1:
        return high, n - 1
    if high == n - 1 and low > 0:
        return 0, low
    return None


def reduce_theta_stack(
    stack: np.ndarray,
    kind: str,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Reduce a ``(theta, y, x)`` stack with NaN and validity-mask semantics.

    Invalid samples are excluded from all three reductions. Pixels with no
    valid θ samples are NaN (never zero or ``-inf``).
    """
    values = np.asarray(stack)
    if values.ndim != 3:
        raise ValueError(
            f"Ожидался стек (θ, y, x), получена форма {values.shape}."
        )
    if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
        raise TypeError("Для проекции ожидаются вещественные числовые значения.")
    mode = str(kind).strip().lower()
    if mode not in {"mean", "max", "sum"}:
        raise ValueError(f"Неподдерживаемый тип θ-проекции: {kind!r}.")

    data = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(data)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError(
                f"valid_mask имеет форму {mask.shape}, ожидалась {data.shape}."
            )
        valid &= mask

    count = np.sum(valid, axis=0, dtype=np.int64)
    if mode in {"mean", "sum"}:
        total = np.sum(np.where(valid, data, 0.0), axis=0, dtype=np.float64)
        if mode == "sum":
            return np.where(count > 0, total, np.nan)
        result = np.full(total.shape, np.nan, dtype=np.float64)
        np.divide(total, count, out=result, where=count > 0)
        return result

    maximum = np.max(np.where(valid, data, -np.inf), axis=0)
    return np.where(count > 0, maximum, np.nan)


class ThetaProjectionAccumulator:
    """Memory-bounded incremental reducer for interactive θ-range previews.

    ``data`` is kept in its native ``(θ, y, x)`` or ``(y, x, θ)`` layout.
    For mean/sum, a one-sample boundary movement updates only the entering or
    leaving detector plane. For max, the accumulator tracks how many samples
    attain each pixel's current maximum and repairs only pixels whose last
    maximum leaves the interval; dense repairs fall back to a vectorized full
    reduction.

    The accumulator owns only 2D working arrays (sum/count/output, or max/count/
    output) and keeps strong references to the current source arrays so object-id
    reuse cannot accidentally reuse stale results.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._key = None
        self._data = None
        self._valid_mask = None
        self._order = None
        self._axis = None
        self._kind = None
        self._range = None
        self._sum = None
        self._count = None
        self._maximum = None
        self._max_count = None
        self._output = None
        self._incremental_updates = 0

    @staticmethod
    def _plane(data: np.ndarray, axis: int, source_index: int) -> np.ndarray:
        return data[source_index, :, :] if axis == 0 else data[:, :, source_index]

    @staticmethod
    def _plane_mask(
        valid_mask: np.ndarray | None, axis: int, source_index: int
    ) -> np.ndarray | None:
        if valid_mask is None:
            return None
        return (
            valid_mask[source_index, :, :]
            if axis == 0 else valid_mask[:, :, source_index]
        )

    @staticmethod
    def _plane_valid(
        plane: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        valid = np.isfinite(plane)
        if mask is not None:
            np.logical_and(valid, np.asarray(mask, dtype=bool), out=valid)
        return valid

    def _build(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray | None,
        order: np.ndarray,
        axis: int,
        kind: str,
        low: int,
        high: int,
    ) -> None:
        source_indices = order[low:high + 1]
        start = int(np.min(source_indices))
        stop = int(np.max(source_indices)) + 1

        if axis == 0:
            slab = data[start:stop, :, :]
            mask_slab = (
                valid_mask[start:stop, :, :] if valid_mask is not None else None
            )
            reduction_axis = 0
        else:
            slab = data[:, :, start:stop]
            mask_slab = (
                valid_mask[:, :, start:stop] if valid_mask is not None else None
            )
            reduction_axis = 2

        valid = np.isfinite(slab)
        if mask_slab is not None:
            np.logical_and(valid, np.asarray(mask_slab, dtype=bool), out=valid)

        if kind in {"mean", "sum"}:
            self._sum = np.sum(
                slab, axis=reduction_axis, where=valid, dtype=np.float64
            )
            self._count = np.sum(valid, axis=reduction_axis, dtype=np.uint32)
            self._maximum = None
            self._max_count = None
        else:
            initial = (
                -np.inf
                if np.issubdtype(slab.dtype, np.floating)
                else np.iinfo(slab.dtype).min
            )
            self._maximum = np.max(
                slab, axis=reduction_axis, where=valid, initial=initial
            ).astype(np.float64, copy=False)
            expanded_max = (
                self._maximum[np.newaxis, :, :]
                if axis == 0 else self._maximum[:, :, np.newaxis]
            )
            matches = np.equal(slab, expanded_max)
            np.logical_and(matches, valid, out=matches)
            self._max_count = np.sum(
                matches, axis=reduction_axis, dtype=np.uint32
            )
            self._sum = None
            self._count = None

        self._range = (int(low), int(high))
        self._incremental_updates = 0
        self._write_output()

    def _adjust_plane(self, source_index: int, *, add: bool) -> None:
        plane = self._plane(self._data, self._axis, int(source_index))
        mask = self._plane_mask(self._valid_mask, self._axis, int(source_index))
        valid = self._plane_valid(plane, mask)
        if add:
            np.add(self._sum, plane, out=self._sum, where=valid, casting="unsafe")
            np.add(self._count, valid, out=self._count, casting="unsafe")
        else:
            np.subtract(self._sum, plane, out=self._sum, where=valid, casting="unsafe")
            np.subtract(self._count, valid, out=self._count, casting="unsafe")

    def _update_additive_range(self, low: int, high: int) -> bool:
        """Incrementally move range edges; return False when a rebuild is better."""
        old_low, old_high = self._range
        if max(old_low, low) > min(old_high, high):
            return False

        changed_planes = abs(low - old_low) + abs(high - old_high)
        old_size = old_high - old_low + 1
        # For large jumps a single vectorized reduction is faster than many
        # individual 2D plane updates. Small boundary drags stay incremental.
        if changed_planes > max(4, old_size // 3) or self._incremental_updates >= 64:
            return False

        # Remove planes no longer selected, then add newly selected planes.
        for position in range(old_low, min(low, old_high + 1)):
            self._adjust_plane(self._order[position], add=False)
        for position in range(max(high + 1, old_low), old_high + 1):
            self._adjust_plane(self._order[position], add=False)
        for position in range(low, min(old_low, high + 1)):
            self._adjust_plane(self._order[position], add=True)
        for position in range(max(old_high + 1, low), high + 1):
            self._adjust_plane(self._order[position], add=True)

        self._range = (int(low), int(high))
        self._incremental_updates += 1
        self._write_output()
        return True

    def _max_pixel_matrix(self):
        """Return a pixel-major or theta-major view for sparse max repairs.

        The sparse fallback is only used with C-contiguous source arrays.
        Refusing a copy here is important: this optimization must never create
        another full-volume allocation during an interactive drag.
        """
        if not self._data.flags.c_contiguous:
            return None, None
        if self._axis == 0:
            data_matrix = self._data.reshape(self._data.shape[0], -1)
            mask_matrix = (
                self._valid_mask.reshape(self._valid_mask.shape[0], -1)
                if self._valid_mask is not None else None
            )
        else:
            data_matrix = self._data.reshape(-1, self._data.shape[2])
            mask_matrix = (
                self._valid_mask.reshape(-1, self._valid_mask.shape[2])
                if self._valid_mask is not None else None
            )
        if not np.shares_memory(data_matrix, self._data):
            return None, None
        if self._valid_mask is not None and not np.shares_memory(
            mask_matrix, self._valid_mask
        ):
            return None, None
        return data_matrix, mask_matrix

    def _recompute_max_pixels(
        self, affected: np.ndarray, remaining_positions: Iterable[int]
    ) -> bool:
        """Repair maxima only for pixels whose last maximum was removed."""
        flat_indices = np.flatnonzero(np.asarray(affected, dtype=bool).reshape(-1))
        if flat_indices.size == 0:
            return True
        pixel_count = int(affected.size)
        # At this density a single vectorized full-range rebuild is usually
        # faster than sparse pixel gathers; the caller then safely falls back.
        if flat_indices.size > max(64, pixel_count // 4):
            return False
        data_matrix, mask_matrix = self._max_pixel_matrix()
        if data_matrix is None:
            return False

        initial = (
            -np.inf
            if np.issubdtype(self._data.dtype, np.floating)
            else float(np.iinfo(self._data.dtype).min)
        )
        new_max = np.full(flat_indices.size, initial, dtype=np.float64)
        new_count = np.zeros(flat_indices.size, dtype=np.uint32)
        positions = tuple(int(position) for position in remaining_positions)
        if not positions:
            return False

        for position in positions:
            source_index = int(self._order[position])
            if self._axis == 0:
                sample = data_matrix[source_index, flat_indices]
                mask_sample = (
                    mask_matrix[source_index, flat_indices]
                    if mask_matrix is not None else None
                )
            else:
                sample = data_matrix[flat_indices, source_index]
                mask_sample = (
                    mask_matrix[flat_indices, source_index]
                    if mask_matrix is not None else None
                )
            valid = np.isfinite(sample)
            if mask_sample is not None:
                np.logical_and(valid, np.asarray(mask_sample, dtype=bool), out=valid)
            greater = valid & (sample > new_max)
            equal = valid & (sample == new_max)
            new_max[greater] = sample[greater]
            new_count[greater] = 1
            new_count[equal] += 1

        maximum_flat = self._maximum.reshape(-1)
        count_flat = self._max_count.reshape(-1)
        maximum_flat[flat_indices] = new_max
        count_flat[flat_indices] = new_count
        return True

    def _remove_max_plane(
        self, source_index: int, remaining_positions: Iterable[int]
    ) -> bool:
        plane = self._plane(self._data, self._axis, int(source_index))
        mask = self._plane_mask(
            self._valid_mask, self._axis, int(source_index)
        )
        valid = self._plane_valid(plane, mask)
        matches = valid & (plane == self._maximum)
        if not np.any(matches):
            return True
        self._max_count[matches] -= 1
        needs_repair = matches & (self._max_count == 0)
        return self._recompute_max_pixels(needs_repair, remaining_positions)

    def _add_max_plane(self, source_index: int) -> None:
        plane = self._plane(self._data, self._axis, int(source_index))
        mask = self._plane_mask(
            self._valid_mask, self._axis, int(source_index)
        )
        valid = self._plane_valid(plane, mask)
        empty = self._max_count == 0
        greater = valid & (empty | (plane > self._maximum))
        equal = valid & ~empty & (plane == self._maximum)
        self._maximum[greater] = plane[greater]
        self._max_count[greater] = 1
        self._max_count[equal] += 1

    def _update_max_range(self, low: int, high: int) -> bool:
        """Incrementally update max projection; sparse repairs are bounded."""
        old_low, old_high = self._range
        if max(old_low, low) > min(old_high, high):
            return False
        changed_planes = abs(low - old_low) + abs(high - old_high)
        old_size = old_high - old_low + 1
        if changed_planes > max(4, old_size // 3) or self._incremental_updates >= 64:
            return False

        current_positions = list(range(old_low, old_high + 1))
        for position in range(old_low, min(low, old_high + 1)):
            current_positions.remove(position)
            if not self._remove_max_plane(
                self._order[position], current_positions
            ):
                return False
        for position in range(max(high + 1, old_low), old_high + 1):
            current_positions.remove(position)
            if not self._remove_max_plane(
                self._order[position], current_positions
            ):
                return False

        for position in range(low, min(old_low, high + 1)):
            current_positions.append(position)
            self._add_max_plane(self._order[position])
        for position in range(max(old_high + 1, low), high + 1):
            current_positions.append(position)
            self._add_max_plane(self._order[position])

        self._range = (int(low), int(high))
        self._incremental_updates += 1
        self._write_output()
        return True

    def _write_output(self) -> None:
        if self._output is None or self._output.shape != (
            self._sum.shape if self._sum is not None else self._maximum.shape
        ):
            shape = self._sum.shape if self._sum is not None else self._maximum.shape
            self._output = np.empty(shape, dtype=np.float64)

        if self._kind == "max":
            np.copyto(self._output, self._maximum)
            self._output[self._max_count == 0] = np.nan
        elif self._kind == "sum":
            np.copyto(self._output, self._sum)
            self._output[self._count == 0] = np.nan
        else:
            self._output.fill(np.nan)
            np.divide(
                self._sum,
                self._count,
                out=self._output,
                where=self._count > 0,
            )

    def project(
        self,
        data: np.ndarray,
        theta_order: Iterable[int],
        axis: int,
        kind: str,
        low: int,
        high: int,
        valid_mask: np.ndarray | None = None,
        *,
        source_token: object = None,
    ) -> np.ndarray:
        """Return the selected projection, reusing 2D accumulators when safe."""
        values = np.asarray(data)
        if values.ndim != 3 or np.iscomplexobj(values) or not np.issubdtype(
            values.dtype, np.number
        ):
            raise TypeError("Для θ-проекции ожидается вещественный числовой 3D-массив.")
        if axis not in (0, 2):
            raise ValueError("Расположение θ-оси должно быть 0 или 2.")
        mask = None if valid_mask is None else np.asarray(valid_mask)
        if mask is not None and mask.shape != values.shape:
            raise ValueError(
                f"valid_mask имеет форму {mask.shape}, ожидалась {values.shape}."
            )
        order = np.asarray(tuple(theta_order), dtype=np.int64).reshape(-1)
        theta_count = int(values.shape[axis])
        if order.size != theta_count or not np.array_equal(
            np.sort(order), np.arange(theta_count, dtype=np.int64)
        ):
            raise ValueError("theta_order должен быть перестановкой индексов θ-оси.")
        # normalize_theta_axis() returns only contiguous ascending or descending
        # source indices. The accumulator relies on that invariant to use a
        # zero-copy slice of the source stack; reject arbitrary permutations
        # rather than silently including unselected θ planes in a slab.
        if order.size > 1:
            order_steps = np.diff(order)
            if not (
                np.all(order_steps == 1) or np.all(order_steps == -1)
            ):
                raise ValueError(
                    "theta_order должен сохранять непрерывный возрастающий "
                    "или убывающий порядок исходной θ-оси."
                )
        low, high = int(low), int(high)
        if not (0 <= low < high < theta_count):
            raise ValueError("Индексы θ-проекции должны удовлетворять 0 ≤ low < high < N.")
        mode = str(kind).strip().lower()
        if mode not in {"mean", "max", "sum"}:
            raise ValueError(f"Неподдерживаемый тип θ-проекции: {kind!r}.")

        key = (
            id(values), id(mask), source_token, axis, mode,
            values.shape, values.strides, values.dtype.str,
            int(order[0]), int(order[-1]),
        )
        same_source = key == self._key
        if same_source and self._range == (low, high):
            return self._output

        if same_source and mode in {"mean", "sum"}:
            if self._update_additive_range(low, high):
                return self._output
        elif same_source and mode == "max":
            if self._update_max_range(low, high):
                return self._output

        self._data = values
        self._valid_mask = mask
        self._order = order
        self._axis = int(axis)
        self._kind = mode
        self._key = key
        self._build(values, mask, order, axis, mode, low, high)
        return self._output
