"""HDF5 -> VTK volume adapter using ROI Explorer's existing data pipeline.

This module deliberately delegates scan discovery, theta-axis inference, ROI geometry,
and spike detection to the application's existing implementations. It never writes to
the HDF5 source and never modifies arrays owned by H5ScanRepository.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

ViewMode = Literal["full", "roi"]


@dataclass(frozen=True, slots=True)
class SpikeReport:
    enabled: bool
    kappa: float | None = None
    sigma: float | None = None
    invalid_count: int = 0
    invalid_fraction: float = 0.0
    note: str = ""


@dataclass(frozen=True, slots=True)
class VolumePayload:
    """Renderer-oriented volume. ``volume[x, y, theta]`` matches its three axes."""

    volume: np.ndarray
    # NaN-aware sum along theta, shape (len(x), len(y)); derived after spike
    # masking and independently smoothed in 2D (not from the 3D-smoothed volume).
    projection: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    coord_system: str
    view_mode: ViewMode
    scan_key: str
    h5_path: Path
    roi: Any | None
    spike_report: SpikeReport
    source_shape: tuple[int, int, int]
    finite_count: int
    invalid_count: int
    smoothing_sigma: float
    scalar_dtype: str
    # Detector-grid indices, zero-based and expressed as (x, y). These were
    # invalidated in every theta frame before ROI extraction/projection.
    masked_detector_pixels: tuple[tuple[int, int], ...] = ()

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.volume.shape)

    @property
    def nbytes(self) -> int:
        return int(
            self.volume.nbytes + self.projection.nbytes
            + self.x.nbytes + self.y.nbytes + self.theta.nbytes
        )


def _validate_axis(axis: Any, expected: int, name: str) -> np.ndarray:
    result = np.asarray(axis, dtype=np.float64)
    if result.ndim != 1 or result.size != expected:
        raise ValueError(f"Ось {name}: ожидалась 1D длины {expected}, получено {result.shape}.")
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"Ось {name} должна содержать конечные значения.")
    if result.size > 1:
        diff = np.diff(result)
        if not (np.all(diff > 0) or np.all(diff < 0)):
            raise ValueError(f"Ось {name} должна быть строго монотонной без повторов.")
    return result.copy()


def orient_to_ascending(
    volume: Any,
    x: Any,
    y: Any,
    theta: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate (x,y,theta) and reverse descending axes together with their data."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"Объём должен быть 3D в порядке (x,y,theta), получено {array.shape}.")
    axes = [
        _validate_axis(x, array.shape[0], "x"),
        _validate_axis(y, array.shape[1], "y"),
        _validate_axis(theta, array.shape[2], "theta"),
    ]
    for dim, axis in enumerate(axes):
        if axis.size > 1 and axis[0] > axis[-1]:
            axes[dim] = axis[::-1].copy()
            array = np.flip(array, axis=dim)
    return np.asarray(array), axes[0], axes[1], axes[2]


def _display_dtype(source: np.ndarray) -> np.dtype:
    """Keep float64 precision; use float32 only when it cannot lose source precision."""
    dtype = source.dtype
    if dtype == np.dtype(np.float32):
        return np.dtype(np.float32)
    if np.issubdtype(dtype, np.integer) and dtype.itemsize <= 2:
        return np.dtype(np.float32)  # all int8/int16 values are represented exactly
    return np.dtype(np.float64)


def _nan_aware_smooth_preserving_invalid(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Masked Gaussian smoothing in-place, retaining all originally invalid sites.

    This follows ROI Explorer's spatial smoothing convention: XY uses ``sigma``;
    the angular dimension is capped at 0.2 voxel to avoid washing out narrow
    diffraction features along theta. Float32 input stays float32; float64 stays
    float64. Only numerator, weights and one boolean mask are additional arrays.
    """
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Параметр сглаживания sigma должен быть конечным числом >= 0.")
    if sigma == 0:
        return volume
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError("Для сглаживания требуется scipy.") from exc

    valid = np.isfinite(volume)
    if not valid.any():
        raise ValueError("В объёме нет конечных значений для сглаживания.")
    if volume.ndim == 3:
        filter_sigma = (sigma, sigma, min(0.2, sigma))
    elif volume.ndim == 2:
        filter_sigma = (sigma, sigma)
    else:
        raise ValueError(f"Поддерживается только 2D/3D сглаживание, получено ndim={volume.ndim}.")

    work = np.array(volume, copy=True)
    np.copyto(work, 0, where=~valid)
    weights = valid.astype(work.dtype, copy=False)
    gaussian_filter(work, sigma=filter_sigma, mode="reflect", output=work)
    gaussian_filter(weights, sigma=filter_sigma, mode="reflect", output=weights)
    usable = valid & (weights > np.finfo(work.dtype).eps)
    np.divide(work, weights, out=work, where=usable)
    np.copyto(volume, np.nan, where=~valid, casting="unsafe")
    np.copyto(volume, work, where=usable, casting="unsafe")
    return volume


def _sum_projection_chunked(volume: np.ndarray, chunk_x: int = 32) -> np.ndarray:
    """NaN-aware theta sum with bounded temporary memory."""
    nx, ny, _ = volume.shape
    projection = np.empty((nx, ny), dtype=np.float64)
    for x0 in range(0, nx, max(1, int(chunk_x))):
        x1 = min(nx, x0 + max(1, int(chunk_x)))
        block = np.asarray(volume[x0:x1, :, :])
        valid = np.isfinite(block)
        sums = np.sum(np.where(valid, block, 0.0), axis=2, dtype=np.float64)
        sums[np.count_nonzero(valid, axis=2) == 0] = np.nan
        projection[x0:x1, :] = sums
    return projection


def _regrid_rotated_roi_to_detector(
    roi_volume: Any,
    roi: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a local ``(u, v, theta)`` ROI grid to an axis-aligned detector grid.

    ``vtkGPUVolumeRayCastMapper`` is reliable for rectilinear datasets, but a
    rotated ``vtkVolume.SetUserMatrix`` makes its ray-sampling coordinates
    disagree with the rectilinear texture coordinates on some VTK/OpenGL
    builds.  The result is especially misleading for sparse/high-contrast
    data: the XY plane remains visible while the volume is reduced to a fuzzy
    sheet.  Regridding the *display copy* to detector X/Y avoids that failure
    mode without changing the source HDF5 data or the ROI extraction result.

    The ROI is sampled once onto a regular detector-aligned lattice.  Its
    exact rotated footprint is retained by ``roi.contains``; points outside
    that polygon remain NaN and are made transparent by the VTK adapter.  The
    local extraction remains the single authoritative interpolation from the
    detector measurements, so this function is only a coordinate/display
    conversion.  Float precision is preserved from the input array.
    """
    source = np.asarray(roi_volume.volume)
    u_axis = np.asarray(roi_volume.x, dtype=np.float64)
    v_axis = np.asarray(roi_volume.y, dtype=np.float64)
    theta = np.asarray(roi_volume.theta)
    if source.ndim != 3 or source.shape != (u_axis.size, v_axis.size, theta.size):
        raise ValueError(
            "Нельзя перестроить повернутую ROI-сетку: volume и оси (u,v,theta) не согласованы."
        )
    if u_axis.size < 2 or v_axis.size < 2 or theta.size < 2:
        raise ValueError("Для 3D-перестройки повернутой ROI нужны минимум два узла по каждой оси.")
    for name, axis in (("u", u_axis), ("v", v_axis)):
        if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"Локальная ось {name} должна быть конечной и строго возрастающей.")

    corners = np.asarray(roi.corners(), dtype=np.float64).reshape(-1, 2)
    if corners.shape[0] != 4 or not np.all(np.isfinite(corners)):
        raise ValueError("Для перестройки ожидаются четыре конечные вершины прямоугольной ROI.")
    x_min, x_max = float(corners[:, 0].min()), float(corners[:, 0].max())
    y_min, y_max = float(corners[:, 1].min()), float(corners[:, 1].max())
    du = float(np.median(np.diff(u_axis)))
    dv = float(np.median(np.diff(v_axis)))
    step = min(abs(du), abs(dv))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Не удалось определить положительный шаг локальной ROI-сетки.")

    # Use the detector-space AABB only as the rectangular storage lattice.
    # The validity mask below keeps the true rotated rectangle, not this box.
    nx = max(2, int(np.ceil((x_max - x_min) / step)) + 1)
    ny = max(2, int(np.ceil((y_max - y_min) / step)) + 1)
    # Rotation expands the axis-aligned bounding box. Bound the display-only
    # resampling work using the same six-million-voxel budget as ROI extraction.
    # Keep theta resolution exact; reduce only the two display-plane dimensions.
    max_xy = max(4, int(np.ceil(np.sqrt(6_000_000 / max(1, theta.size)))))
    if max(nx, ny) > max_xy:
        reduction = max(nx, ny) / max_xy
        # Match the Plotly ROI display lattice exactly: both renderers then
        # use the same detector-space bounds and the same bounded dimensions.
        nx = max(2, int(np.ceil(nx / reduction)))
        ny = max(2, int(np.ceil(ny / reduction)))
    x_axis = np.linspace(x_min, x_max, nx, dtype=np.float64)
    y_axis = np.linspace(y_min, y_max, ny, dtype=np.float64)
    detector_x, detector_y = np.meshgrid(x_axis, y_axis, indexing="ij")
    local_u, local_v = roi.to_local(detector_x, detector_y)
    iu = np.interp(local_u, u_axis, np.arange(u_axis.size, dtype=np.float64), left=np.nan, right=np.nan)
    iv = np.interp(local_v, v_axis, np.arange(v_axis.size, dtype=np.float64), left=np.nan, right=np.nan)
    inside = np.asarray(roi.contains(detector_x, detector_y), dtype=bool)
    inside &= np.isfinite(iu) & np.isfinite(iv)
    sample_coordinates = [np.where(inside, iu, -1.0), np.where(inside, iv, -1.0)]

    try:
        from scipy.ndimage import map_coordinates
    except ImportError as exc:
        raise ImportError("Для отображения повернутой ROI в VTK требуется scipy.") from exc

    output_dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.dtype(np.float64)
    output = np.full((nx, ny, theta.size), np.nan, dtype=output_dtype)
    for index in range(theta.size):
        layer = np.asarray(source[:, :, index], dtype=output_dtype)
        valid = np.isfinite(layer)
        numerator = map_coordinates(
            np.where(valid, layer, 0.0), sample_coordinates,
            order=1, mode="constant", cval=0.0,
        )
        weights = map_coordinates(
            valid.astype(output_dtype, copy=False), sample_coordinates,
            order=1, mode="constant", cval=0.0,
        )
        usable = inside & (weights > 1e-8)
        np.divide(
            numerator, weights,
            out=output[:, :, index],
            where=usable,
        )
        output[:, :, index][~usable] = np.nan

    return x_axis, y_axis, output


class H5VolumeLoader:
    """Loads full-detector or rectangular ROI volumes directly from HDF5.

    The repository is always configured with PoolSpec(1, 1): no spatial pooling
    is silently applied. ROI extraction is delegated to H5ScanRepository.volume_from_roi,
    which itself calls the same roc_mcs.roi.extraction.extract_roi_volume used by GUI.
    """

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path).expanduser().resolve()
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5-файл не найден: {self.h5_path}")

        try:
            from roc_mcs.roi.h5_scan_repository import H5ScanRepository
            from roc_mcs.roi.spatial_pooling import PoolSpec
        except ImportError as exc:
            raise ImportError(
                "Не удалось импортировать H5ScanRepository. Запускайте viewer из "
                "checkout ROC-MCS, где каталог src доступен в sys.path."
            ) from exc

        self.repository = H5ScanRepository(self.h5_path, pool=PoolSpec(size=1, stride=1))
        from roc_mcs.gui.app_settings import load_detector_bad_pixels
        self.bad_pixels = tuple(load_detector_bad_pixels())

    def set_bad_pixels(self, pixels: Any) -> None:
        """Replace this loader's detector exclusions with validated (x, y) pairs."""
        from roc_mcs.gui.app_settings import normalize_detector_bad_pixels
        self.bad_pixels = tuple(
            (item["x"], item["y"])
            for item in normalize_detector_bad_pixels(pixels)
        )

    def theta_bounds(self, scan_key: str) -> tuple[float, float]:
        """Read only the scan's theta coordinate vector, without loading detector frames."""
        info = self.repository.scans[scan_key]
        raw_theta = self.repository._theta_1d(
            self.repository._load_dataset(info.th_path)
        )
        theta = _validate_axis(raw_theta, raw_theta.size, "theta")
        return float(np.min(theta)), float(np.max(theta))

    @property
    def scan_keys(self) -> list[str]:
        return self.repository.scan_keys()

    def _new_scan_repository(self):
        """Create a per-load repository so switching scans cannot accumulate HDF5 arrays.

        ``self.repository`` is retained only for its lightweight scan metadata and
        key list. A fresh repository owns the actual scan cache for one load call;
        once the payload has been copied, that cache becomes collectible.
        """
        from roc_mcs.roi.h5_scan_repository import H5ScanRepository
        from roc_mcs.roi.spatial_pooling import PoolSpec

        return H5ScanRepository(self.h5_path, pool=PoolSpec(size=1, stride=1))

    def _make_spike_valid_mask(self, repository: Any, scan_key: str) -> tuple[np.ndarray, SpikeReport]:
        """Use exactly the spike-analysis sequence from Palette_Transparency_Lab_v2."""
        try:
            from roc_mcs.processing.background import estimate_background_stats
            from roc_mcs.processing.spikes import find_kappa_threshold, build_spike_mask
        except ImportError as exc:
            raise ImportError(
                "В checkout не найдены roc_mcs.processing.background / spikes. "
                "Режим удаления спайков отключён до установки этих штатных модулей."
            ) from exc

        formal_tyx = np.asarray(repository.data_tyx(scan_key))
        data_yxt = np.asarray(repository.data_yxt(scan_key))
        pooled_background = estimate_background_stats(formal_tyx)
        kappa_fit = find_kappa_threshold(
            formal_tyx,
            background_variance=pooled_background.sigma ** 2,
            kappas=np.arange(0.1, 60.1, 0.1),
            verbose=False,
        )
        spike_mask_tyx = np.asarray(
            build_spike_mask(
                formal_tyx,
                background_variance=pooled_background.sigma ** 2,
                kappa=kappa_fit.kappa,
                verbose=False,
            ),
            dtype=bool,
        )
        if spike_mask_tyx.shape != formal_tyx.shape:
            raise ValueError(
                f"Spike-mask имеет shape={spike_mask_tyx.shape}, "
                f"ожидалось {formal_tyx.shape} (theta,y,x)."
            )
        valid_yxt = ~np.moveaxis(spike_mask_tyx, 0, -1)
        if valid_yxt.shape != data_yxt.shape:
            raise ValueError(
                f"valid mask shape={valid_yxt.shape} не совпадает с Y×X×theta "
                f"данными {data_yxt.shape}."
            )
        invalid_count = int(np.count_nonzero(spike_mask_tyx))
        total = int(spike_mask_tyx.size)
        sigma = float(pooled_background.sigma)
        kappa = float(kappa_fit.kappa)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError(f"Оценка фонового sigma некорректна: {sigma!r}.")
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError(f"Оценка spike kappa некорректна: {kappa!r}.")
        report = SpikeReport(
            enabled=True,
            kappa=kappa,
            sigma=sigma,
            invalid_count=invalid_count,
            invalid_fraction=(invalid_count / total) if total else 0.0,
            note="Штатный estimate_background_stats → find_kappa_threshold → build_spike_mask.",
        )
        return valid_yxt, report

    def load(
        self,
        scan_key: str,
        *,
        mode: ViewMode = "full",
        roi: Any | None = None,
        remove_spikes: bool = False,
        smoothing_sigma: float = 0.0,
        theta_range: tuple[float, float] | None = None,
    ) -> VolumePayload:
        if mode not in ("full", "roi"):
            raise ValueError("mode должен быть 'full' или 'roi'.")
        smoothing_sigma = float(smoothing_sigma)
        if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
            raise ValueError("smoothing_sigma должен быть конечным числом >= 0.")
        if theta_range is not None:
            if len(theta_range) != 2:
                raise ValueError("theta_range должен содержать две границы (min, max).")
            theta_low, theta_high = map(float, theta_range)
            if not (np.isfinite(theta_low) and np.isfinite(theta_high) and theta_low < theta_high):
                raise ValueError("Для диапазона θ необходимо конечное значение min < max.")
            theta_range = (theta_low, theta_high)
        if mode == "roi":
            if roi is None:
                raise ValueError("Для режима ROI необходимо передать объект ROI.")
            normalized = roi.normalized() if callable(getattr(roi, "normalized", None)) else roi
            if str(getattr(normalized, "type", "")).lower() != "rect":
                raise ValueError("3D-объём ROI поддерживает прямоугольную ROI (включая повёрнутую).")
            if not bool(getattr(normalized, "is_valid", True)):
                raise ValueError("Геометрия ROI невалидна.")
            roi = normalized
        else:
            roi = None

        # Keep only scan metadata in self.repository. The per-load repository
        # releases its source/view caches as soon as this function returns.
        repository = self._new_scan_repository()
        scan = repository.load_scan(scan_key)
        x_axis, y_axis = repository.centers_xy(scan_key)
        valid_yxt = None
        if remove_spikes:
            valid_yxt, spike_report = self._make_spike_valid_mask(repository, scan_key)
        else:
            spike_report = SpikeReport(enabled=False, note="Данные загружены без удаления спайков.")

        # Global detector defects are source-grid exclusions, not display
        # filters. Apply them before both full-detector projection and ROI
        # resampling so a dead pixel cannot leak into neighbouring rotated-ROI
        # samples through interpolation.
        ny_source, nx_source = scan.shape_yx
        applicable_bad_pixels = tuple(
            (int(xi), int(yi))
            for xi, yi in self.bad_pixels
            if 0 <= int(xi) < nx_source and 0 <= int(yi) < ny_source
        )
        if applicable_bad_pixels:
            expected_yxt_shape = tuple(int(v) for v in scan.yxt.shape)
            if valid_yxt is None:
                valid_yxt = np.ones(expected_yxt_shape, dtype=bool)
            else:
                if valid_yxt.shape != expected_yxt_shape:
                    raise ValueError(
                        f"Spike validity mask shape={valid_yxt.shape} не совпадает "
                        f"с source Y×X×theta={expected_yxt_shape}."
                    )
                valid_yxt = np.array(valid_yxt, dtype=bool, copy=True)
            for pixel_x, pixel_y in applicable_bad_pixels:
                valid_yxt[pixel_y, pixel_x, :] = False

        if mode == "roi":
            roi_volume = repository.volume_from_roi(
                scan_key,
                roi,
                valid_mask=valid_yxt,
                order=1,
            )
            volume = np.asarray(roi_volume.volume)
            x = np.asarray(roi_volume.x)
            y = np.asarray(roi_volume.y)
            theta = np.asarray(roi_volume.theta)
            coord_system = str(roi_volume.coord_system)
            if coord_system == "uv":
                # Keep the ROI's exact rotated footprint, but hand the VTK GPU
                # mapper an axis-aligned rectilinear grid.  Do not apply an
                # actor rotation to vtkVolume: on affected GPU backends that
                # leaves the bounds transformed while the scalar sampling is
                # evaluated in a mismatched coordinate frame.
                x, y, volume = _regrid_rotated_roi_to_detector(roi_volume, roi)
                coord_system = "xy"
            source_shape = tuple(int(v) for v in scan.data.shape)
        else:
            # ScanData.yxt is always (y,x,theta), irrespective of native HDF5 theta axis.
            data_yxt = np.asarray(scan.yxt)
            if data_yxt.ndim != 3 or data_yxt.shape[:2] != (len(y_axis), len(x_axis)):
                raise ValueError(
                    f"ScanData Y×X×theta={data_yxt.shape} не согласован с "
                    f"центрами x/y {(len(x_axis), len(y_axis))}."
                )
            volume = np.transpose(data_yxt, (1, 0, 2))
            if valid_yxt is not None and valid_yxt.shape != data_yxt.shape:
                raise ValueError("Spike validity mask не согласована с полным объёмом.")
            x = np.asarray(x_axis)
            y = np.asarray(y_axis)
            theta = np.asarray(scan.th)
            coord_system = "xy"
            source_shape = tuple(int(v) for v in scan.data.shape)

        source = np.asarray(volume)
        if not np.issubdtype(source.dtype, np.number) or np.issubdtype(source.dtype, np.complexfloating):
            raise TypeError(f"Ожидались вещественные числовые данные HDF5, получен dtype={source.dtype}.")
        dtype = _display_dtype(source)
        # Make one renderer-owned copy. Apply the validity mask in-place on this
        # copy (rather than np.where + a second full copy) to lower peak RAM.
        volume = np.array(source, dtype=dtype, copy=True, order="C")
        if mode == "full" and valid_yxt is not None:
            valid_xyt = np.transpose(valid_yxt, (1, 0, 2))
            np.copyto(volume, np.nan, where=~valid_xyt, casting="unsafe")
        volume, x, y, theta = orient_to_ascending(volume, x, y, theta)

        # Crop the actual theta samples (never rescale physical coordinates).
        # This happens before integration and smoothing, so the XY projection
        # represents exactly the selected angular interval.
        if theta_range is not None:
            theta_low, theta_high = theta_range
            selected_theta = (theta >= theta_low) & (theta <= theta_high)
            theta_indices = np.flatnonzero(selected_theta)
            if theta_indices.size < 2:
                raise ValueError(
                    f"Диапазон θ [{theta_low:g}, {theta_high:g}] содержит "
                    f"{theta_indices.size} отсчёта; для 3D нужны минимум два. "
                    f"Доступно [{theta[0]:g}, {theta[-1]:g}]."
                )
            start, stop = int(theta_indices[0]), int(theta_indices[-1]) + 1
            volume = volume[:, :, start:stop]
            theta = theta[start:stop].copy()

        # Match the established Plotly data flow: integrate the spike-masked
        # source volume first, then smooth the 2D projection independently.
        # The 3D volume is smoothed separately, with theta sigma capped at 0.2.
        projection = _sum_projection_chunked(volume)
        if smoothing_sigma > 0:
            _nan_aware_smooth_preserving_invalid(projection, smoothing_sigma)
            _nan_aware_smooth_preserving_invalid(volume, smoothing_sigma)

        finite = np.isfinite(volume)
        finite_count = int(np.count_nonzero(finite))
        invalid_count = int(volume.size - finite_count)
        if finite_count == 0:
            raise ValueError("После извлечения/маскирования объём не содержит конечных значений.")
        for name, axis in (("x", x), ("y", y), ("theta", theta)):
            if axis.size > 1 and not np.all(np.diff(axis) > 0):
                raise ValueError(f"Ось {name} не удалось привести к возрастающему порядку.")

        volume.setflags(write=False)
        projection.setflags(write=False)
        for axis in (x, y, theta):
            axis.setflags(write=False)
        return VolumePayload(
            volume=volume,
            projection=projection,
            x=x,
            y=y,
            theta=theta,
            coord_system=coord_system,
            view_mode=mode,
            scan_key=str(scan_key),
            h5_path=self.h5_path,
            roi=roi,
            spike_report=spike_report,
            source_shape=source_shape,
            finite_count=finite_count,
            invalid_count=invalid_count,
            smoothing_sigma=float(smoothing_sigma),
            scalar_dtype=str(volume.dtype),
            masked_detector_pixels=applicable_bad_pixels,
        )
