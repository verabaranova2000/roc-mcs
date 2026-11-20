"""Адаптация данных ROI Explorer к контракту VolumePayload без чтения HDF5."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from roc_mcs.roi.roi_types import normalize_rectangular_roi
from .h5_volume_source import (
    SpikeReport, VolumePayload, _display_dtype, _regrid_rotated_roi_to_detector,
    _sum_projection_chunked, _nan_aware_smooth_preserving_invalid, orient_to_ascending,
)



def build_volume_payload(
    raw: Any, theta: Any, axis: int, x_coords: Any, y_coords: Any, *,
    valid_mask: Any | None = None, roi: Any | None = None,
    scan_key: str = "", h5_path: str | Path | None = None,
    operation_id: str = "source.raw", metadata: dict | None = None,
    smoothing_sigma: float = 0.0,
) -> VolumePayload:
    """Build X×Y×θ display payload from the exact selected ROI Explorer view.

    The input is borrowed read-only. No HDF5 access, processing, or in-place
    modification is performed here.
    """
    source = np.asarray(raw)
    if source.ndim != 3 or not np.issubdtype(source.dtype, np.number) or np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError(f"Ожидается вещественный числовой 3D-массив, получено {source.dtype} {source.shape}.")
    try:
        axis = int(axis)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Ось θ должна иметь индекс 0 или 2.") from exc
    if axis not in (0, 2):
        raise ValueError(f"Неподдерживаемая ось θ: {axis}.")
    data_yxt = np.moveaxis(source, axis, -1)
    ny, nx, nt = data_yxt.shape
    if min(ny, nx, nt) < 2:
        raise ValueError(f"Для 3D-объёма каждая ось должна содержать минимум 2 узла; получено {data_yxt.shape}.")
    theta_axis = np.asarray(theta, dtype=np.float64)
    if theta_axis.ndim != 1 or theta_axis.size != nt or not np.isfinite(theta_axis).all():
        raise ValueError("Ось θ не согласована с объёмом или содержит нечисловые значения.")
    x_axis = np.asarray(x_coords, dtype=np.float64)
    y_axis = np.asarray(y_coords, dtype=np.float64)
    if x_axis.shape != (nx,) or y_axis.shape != (ny,) or not np.isfinite(x_axis).all() or not np.isfinite(y_axis).all():
        raise ValueError("Координатные оси X/Y не согласованы с детекторной сеткой.")
    for name, values in (("X", x_axis), ("Y", y_axis), ("θ", theta_axis)):
        if values.size > 1 and not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)):
            raise ValueError(f"Ось {name} должна быть строго монотонной без повторов.")
    mask_yxt = None
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.dtype != np.bool_:
            raise TypeError(f"valid_mask должна иметь bool dtype, получено {mask.dtype}.")
        if mask.shape == source.shape:
            mask_yxt = np.moveaxis(mask, axis, -1)
        elif mask.shape == data_yxt.shape:
            mask_yxt = mask
        else:
            raise ValueError(f"valid_mask shape={mask.shape} не согласована с Y×X×θ={data_yxt.shape}.")

    # Normalize source coordinates once, applying every reversal to data and mask.
    volume_xyt = np.transpose(data_yxt, (1, 0, 2))
    volume_xyt, x_axis, y_axis, theta_axis = orient_to_ascending(volume_xyt, x_axis, y_axis, theta_axis)
    if mask_yxt is not None:
        mask_xyt = np.transpose(mask_yxt, (1, 0, 2))
        if x_coords is not None and np.asarray(x_coords)[0] > np.asarray(x_coords)[-1]:
            mask_xyt = np.flip(mask_xyt, axis=0)
        if y_coords is not None and np.asarray(y_coords)[0] > np.asarray(y_coords)[-1]:
            mask_xyt = np.flip(mask_xyt, axis=1)
        if np.asarray(theta)[0] > np.asarray(theta)[-1]:
            mask_xyt = np.flip(mask_xyt, axis=2)
    else:
        mask_xyt = None

    mode = "full" if roi is None else "roi"
    coord_system = "xy"
    selected_roi = None
    if roi is not None:
        selected_roi = normalize_rectangular_roi(roi)
        from roc_mcs.roi.extraction import extract_roi_volume
        yxt = np.transpose(volume_xyt, (1, 0, 2))
        valid_yxt = None if mask_xyt is None else np.transpose(mask_xyt, (1, 0, 2))
        extracted = extract_roi_volume(
            yxt, theta_axis, selected_roi, axis=2, valid_mask=valid_yxt,
            order=1, x_coords=x_axis, y_coords=y_axis,
        )
        volume_xyt = np.asarray(extracted.volume)
        x_axis = np.asarray(extracted.x, dtype=np.float64)
        y_axis = np.asarray(extracted.y, dtype=np.float64)
        theta_axis = np.asarray(extracted.theta, dtype=np.float64)
        coord_system = str(extracted.coord_system).lower()
        if coord_system == "uv":
            # _regrid_rotated_roi_to_detector has a shared public contract:
            # (x_axis, y_axis, volume_xyt).  The legacy HDF5 loader unpacks
            # that order too.  Reversing it here silently assigned the 1-D X
            # coordinate vector to ``volume_xyt``; orient_to_ascending then
            # correctly rejected e.g. shape (88,) as a non-volume.
            x_axis, y_axis, volume_xyt = _regrid_rotated_roi_to_detector(extracted, selected_roi)
            expected_shape = (x_axis.size, y_axis.size, theta_axis.size)
            if volume_xyt.ndim != 3 or volume_xyt.shape != expected_shape:
                raise ValueError(
                    "После перестройки повёрнутой ROI ожидался объём "
                    f"(x,y,theta)={expected_shape}, получено {volume_xyt.shape}."
                )
            coord_system = "xy"

    dtype = _display_dtype(np.asarray(volume_xyt))
    volume_xyt = np.array(volume_xyt, dtype=dtype, order="C", copy=True)
    # The mask is already represented as NaNs by ROI extraction; for full data
    # apply it explicitly to a display-owned copy.
    if mode == "full" and mask_xyt is not None:
        np.copyto(volume_xyt, np.nan, where=~mask_xyt, casting="unsafe")
    volume_xyt, x_axis, y_axis, theta_axis = orient_to_ascending(volume_xyt, x_axis, y_axis, theta_axis)
    projection = _sum_projection_chunked(volume_xyt)
    sigma = float(smoothing_sigma)
    if not np.isfinite(sigma) or sigma < 0 or sigma > 5:
        raise ValueError("smoothing_sigma должен быть конечным числом в диапазоне [0, 5].")
    if sigma > 0:
        # Отдельные renderer-owned копии: исходные данные Explorer не меняются.
        _nan_aware_smooth_preserving_invalid(projection, sigma)
        _nan_aware_smooth_preserving_invalid(volume_xyt, sigma)
    finite = np.isfinite(volume_xyt)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        raise ValueError("Выбранный источник не содержит конечных данных для 3D.")
    invalid_count = int(volume_xyt.size - finite_count)
    for item in (volume_xyt, projection, x_axis, y_axis, theta_axis):
        item.setflags(write=False)
    meta = dict(metadata or {})
    spike_enabled = str(operation_id) == "preprocess.spikes"
    spike_report = SpikeReport(
        enabled=spike_enabled,
        kappa=_finite_optional(meta.get("kappa")),
        sigma=_finite_optional(meta.get("background_sigma")),
        invalid_count=int(meta.get("n_spikes", 0) or 0),
        invalid_fraction=float(meta.get("spike_fraction", 0.0) or 0.0),
        note="Выбран принятый результат Workbench: удаление спайков." if spike_enabled else "Отображается выбранная версия данных без отдельного spike-этапа.",
    )
    return VolumePayload(
        volume=volume_xyt, projection=projection, x=x_axis, y=y_axis,
        theta=theta_axis, coord_system=coord_system, view_mode=mode,
        scan_key=str(scan_key), h5_path=Path(h5_path).expanduser() if h5_path else Path("."),
        roi=selected_roi, spike_report=spike_report,
        source_shape=tuple(int(v) for v in data_yxt.shape),
        finite_count=finite_count, invalid_count=invalid_count,
        smoothing_sigma=sigma, scalar_dtype=str(volume_xyt.dtype),
    )


def _finite_optional(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None
