"""Numerical helpers for the HKL result plots.

This module deliberately has no Qt or Matplotlib dependency. Histogram binning
is a presentation concern, but its edge handling is kept deterministic and
unit-testable independently from the GUI runtime.
"""
from __future__ import annotations

import numpy as np


def build_histogram_edges(
    values: np.ndarray,
    requested_width: float,
    *,
    max_bins: int,
) -> np.ndarray:
    """Build finite, strictly increasing edges covering every finite value.

    Bins are aligned to zero while the requested width fits within ``max_bins``.
    For unusually broad ranges, the bins are widened uniformly rather than
    truncating observations. A constant sample receives one finite-width bin.
    """
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("values must contain at least one finite observation")

    width = float(requested_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("requested_width must be finite and greater than zero")

    try:
        bin_limit = int(max_bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_bins must be a positive integer") from exc
    if bin_limit < 1 or bin_limit != max_bins:
        raise ValueError("max_bins must be a positive integer")

    data_low = float(np.min(sample))
    data_high = float(np.max(sample))

    if data_high == data_low:
        half_width = width * 0.5
        lower = data_low - half_width
        upper = data_high + half_width
        # Preserve a valid interval when the requested width is below the
        # floating-point spacing at the sample's magnitude.
        if not np.isfinite(lower) or lower >= data_low:
            lower = float(np.nextafter(data_low, -np.inf))
        if not np.isfinite(upper) or upper <= data_high:
            upper = float(np.nextafter(data_high, np.inf))
        # At the extreme finite float values, nextafter may point to infinity;
        # the observed endpoint itself is still a valid inclusive histogram edge.
        if not np.isfinite(lower):
            lower = data_low
        if not np.isfinite(upper):
            upper = data_high
        edges = np.asarray([lower, upper], dtype=np.float64)
    else:
        ratio = data_low / width
        aligned_low = float(np.floor(ratio) * width) if np.isfinite(ratio) else np.nan
        span_in_bins = (
            (data_high - aligned_low) / width if np.isfinite(aligned_low) else np.inf
        )

        if not np.isfinite(span_in_bins) or span_in_bins > bin_limit:
            # A display cap must never exclude the largest measured value.
            edges = np.linspace(data_low, data_high, bin_limit + 1, dtype=np.float64)
        else:
            bin_count = max(1, int(np.ceil(span_in_bins)))
            edges = aligned_low + np.arange(bin_count + 1, dtype=np.float64) * width
            if edges[-1] < data_high:
                if bin_count < bin_limit:
                    next_edge = edges[-1] + width
                    if np.isfinite(next_edge) and next_edge > edges[-1]:
                        edges = np.append(edges, next_edge)
                    else:
                        edges = np.asarray([data_low, data_high], dtype=np.float64)
                else:
                    edges = np.linspace(
                        data_low, data_high, bin_limit + 1, dtype=np.float64
                    )

    # Guard against edge collapse at extreme floating-point magnitudes. The
    # two-point fallback still covers both observed extrema without inventing
    # precision that float64 cannot represent.
    valid_edges = (
        edges.size >= 2
        and np.all(np.isfinite(edges))
        and np.all(np.diff(edges) > 0.0)
        and edges[0] <= data_low
        and edges[-1] >= data_high
    )
    if not valid_edges:
        if data_low == data_high:
            lower = float(np.nextafter(data_low, -np.inf))
            upper = float(np.nextafter(data_high, np.inf))
            edges = np.asarray([lower, upper], dtype=np.float64)
        else:
            edges = np.asarray([data_low, data_high], dtype=np.float64)

    return edges


def histogram_view_bounds(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    padding_fraction: float = 0.06,
    nonnegative: bool = True,
) -> tuple[float, float]:
    """Return plot limits based on observations and the actual bin width.

    Bounds are based on the actual first and last bin edges, so the x-axis
    never clips part of a histogram bar. The effective final-bin width is used
    rather than the requested width, which may change when the bin cap is hit.
    """
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]
    bin_edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        raise ValueError("values must contain at least one finite observation")
    if (
        bin_edges.size < 2
        or not np.all(np.isfinite(bin_edges))
        or np.any(np.diff(bin_edges) <= 0.0)
    ):
        raise ValueError("edges must be finite and strictly increasing")

    padding_fraction = float(padding_fraction)
    if not np.isfinite(padding_fraction) or padding_fraction < 0.0:
        raise ValueError("padding_fraction must be finite and non-negative")

    bin_low = float(bin_edges[0])
    bin_high = float(bin_edges[-1])
    effective_width = float(bin_edges[-1] - bin_edges[-2])
    bin_span = bin_high - bin_low
    view_span = max(bin_span, effective_width)
    padding = view_span * padding_fraction

    view_low = bin_low - padding
    view_high = bin_high + padding
    if not np.isfinite(view_low):
        view_low = bin_low
    if not np.isfinite(view_high):
        view_high = bin_high
    if nonnegative:
        view_low = max(0.0, view_low)

    # The padding can round to zero for large magnitudes. Ensure a valid range
    # without changing the measured extrema.
    if not view_high > view_low:
        view_low = min(bin_low, view_low)
        view_high = max(bin_high, view_high)
        if not view_high > view_low:
            view_high = float(np.nextafter(view_low, np.inf))

    return float(view_low), float(view_high)


def zoomed_view_bounds(
    full_bounds: tuple[float, float],
    zoom_factor: float,
    scroll_fraction: float,
) -> tuple[float, float]:
    """Return a movable x-window inside *full_bounds* without altering data.

    ``zoom_factor=1`` returns the full interval. For larger factors the window
    is centred according to ``scroll_fraction`` (0 = left, 1 = right).
    Values are clamped to the physically available interval.
    """
    try:
        low, high = map(float, full_bounds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("full_bounds must contain two finite values") from exc
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise ValueError("full_bounds must be finite and strictly increasing")

    try:
        factor = float(zoom_factor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("zoom_factor must be finite and >= 1") from exc
    if not np.isfinite(factor) or factor < 1.0:
        raise ValueError("zoom_factor must be finite and >= 1")

    try:
        fraction = float(scroll_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scroll_fraction must be finite") from exc
    if not np.isfinite(fraction):
        raise ValueError("scroll_fraction must be finite")
    fraction = float(np.clip(fraction, 0.0, 1.0))

    span = high - low
    if not np.isfinite(span) or span <= 0.0:
        return low, high
    window_span = span / factor
    if not np.isfinite(window_span) or window_span <= 0.0 or factor == 1.0:
        return low, high
    remaining = span - window_span
    view_low = low + remaining * fraction
    view_high = view_low + window_span
    if not view_high > view_low:
        return low, high
    # Floating-point rounding at the right endpoint must not push the viewport
    # beyond the original data range.
    if view_high > high:
        view_high = high
        view_low = high - window_span
    return float(view_low), float(view_high)


def suggest_histogram_bin_width(
    values: np.ndarray,
    *,
    min_width: float = 0.01,
    max_width: float = 100.0,
    fallback_width: float = 0.25,
) -> float:
    """Suggest a compact *display-only* histogram width using FD's rule.

    The Freedman–Diaconis width is robust to outliers. Degenerate samples use
    a bounded range/sqrt(N) fallback; the result is never used by the HKL
    identification calculation.
    """
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]
    lower = float(min_width)
    upper = float(max_width)
    fallback = float(fallback_width)
    if (
        not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0
        or upper < lower or not np.isfinite(fallback) or fallback <= 0.0
    ):
        raise ValueError("bin width limits must be finite, positive and ordered")
    if sample.size < 2:
        return float(np.clip(fallback, lower, upper))

    span = float(np.max(sample) - np.min(sample))
    if not np.isfinite(span) or span <= 0.0:
        return float(np.clip(fallback, lower, upper))

    q25, q75 = np.percentile(sample, [25.0, 75.0])
    iqr = float(q75 - q25)
    fd_width = 2.0 * iqr / np.cbrt(float(sample.size))
    if not np.isfinite(fd_width) or fd_width <= 0.0:
        target_bins = min(50.0, max(5.0, np.sqrt(float(sample.size))))
        fd_width = span / target_bins
    if not np.isfinite(fd_width) or fd_width <= 0.0:
        fd_width = fallback
    return float(np.clip(fd_width, lower, upper))


def comparison_axis_parameters(values: np.ndarray) -> tuple[str, float | None]:
    """Choose a readable linear/symlog axis while retaining distant candidates.

    The scale decision must account for the largest plotted value: one very
    distant theoretical candidate is scientifically meaningful and must not be
    silently ignored merely because it lies above the 90th percentile. The
    linear region is anchored to the lower decile of positive values, which is
    more stable than anchoring it to a single minimum near-zero value.
    """
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]
    positive = sample[sample > 0.0]
    if positive.size < 2:
        return "linear", None
    reference_low = float(np.percentile(positive, 10.0))
    maximum = float(np.max(positive))
    if not (np.isfinite(reference_low) and np.isfinite(maximum) and reference_low > 0.0):
        return "linear", None
    if maximum / reference_low < 100.0:
        return "linear", None
    # Keep the lower measured range linear so near-zero values remain legible,
    # then compress the complete positive tail, including isolated candidates.
    linthresh = max(float(np.min(positive)), reference_low)
    return "symlog", linthresh
