"""Чистые численные и семантические операции предпросмотра диапазона θ.

Модуль намеренно не зависит от Qt: распознавание карты, выбор комплементарного
интервала и редукция θ-стека можно проверять отдельно от интерфейса.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np


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
