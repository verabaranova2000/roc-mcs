"""Bounded diagnostic benchmark for the ROI Explorer 3D rendering pipeline.

Run from the installed project environment:
    python -m roc_mcs.gui.roi_explorer.volume_rendering_benchmark --points 25000 --webgl

The benchmark accepts a bounded NPZ export of an already extracted ROIVolume, optionally
with the full-detector XY projection. It never opens HDF5, changes application settings,
or touches the live ROI Explorer widget. Its WebGL run uses an isolated Qt WebEngine window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import threading
import zipfile
from pathlib import Path
from dataclasses import asdict
from typing import Any, Callable

import numpy as np

from roc_mcs.gui.roi_explorer.volume_rendering import (
    MAX_RENDER_POINTS,
    build_volume_figure,
    crop_raw_to_roi,
    effective_extraction_stride,
    effective_volume_stride,
    prepare_volume,
)
from roc_mcs.roi.extraction import extract_roi_volume
from roc_mcs.roi.roi_types import ROI, ROIVolume


DEFAULT_POINTS = 25_000
MAX_SAFE_POINTS = min(int(MAX_RENDER_POINTS), 95_000)
MAX_SYNTHETIC_VOXELS = 6_000_000
# Extraction input export is optional and capped to avoid turning diagnostics
# into an uncontrolled second copy of a very large detector scan.
MAX_RAW_EXTRACTION_VOXELS = 25_000_000


def _timed(call: Callable[[], Any], repeats: int) -> tuple[Any, list[float]]:
    result = None
    samples: list[float] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return result, samples


def _summary(samples: list[float]) -> dict[str, float]:
    values = [float(v) for v in samples]
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
    }


def _memory_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        own_info = process.memory_info()
        vm = psutil.virtual_memory()
        tree_rss = int(own_info.rss)
        children = []
        for child in process.children(recursive=True):
            try:
                child_info = child.memory_info()
                tree_rss += int(child_info.rss)
                children.append({"pid": int(child.pid), "rss_mib": round(child_info.rss / (1024 ** 2), 1)})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        result.update(
            process_rss_mib=round(own_info.rss / (1024 ** 2), 1),
            process_tree_rss_mib=round(tree_rss / (1024 ** 2), 1),
            process_child_count=len(children),
            process_children=children[:20],
            process_peak_working_set_mib=(round(getattr(own_info, "peak_wset") / (1024 ** 2), 1)
                                          if getattr(own_info, "peak_wset", None) else None),
            system_ram_total_gib=round(vm.total / (1024 ** 3), 2),
            system_ram_available_gib=round(vm.available / (1024 ** 3), 2),
            system_ram_percent=round(float(vm.percent), 1),
        )
    except Exception:
        result["psutil"] = "not installed; process/system RAM metrics unavailable"
    return result


class _PeakMemorySampler:
    """Low-frequency process-tree/system RAM sampler for transient CPU peaks."""
    def __init__(self, interval_s: float = 0.10):
        self.interval_s = max(0.05, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_tree_rss: float | None = None
        self._minimum_available: float | None = None
        self._samples = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="roi3d-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval_s):
            snap = _memory_snapshot()
            rss = snap.get("process_tree_rss_mib")
            available = snap.get("system_ram_available_gib")
            if rss is not None:
                self._peak_tree_rss = max(float(rss), self._peak_tree_rss or 0.0)
            if available is not None:
                self._minimum_available = min(float(available), self._minimum_available if self._minimum_available is not None else float(available))
            self._samples += 1

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        return {
            "sample_interval_ms": round(self.interval_s * 1000.0),
            "samples": self._samples,
            "peak_process_tree_rss_mib": round(self._peak_tree_rss, 1) if self._peak_tree_rss is not None else None,
            "min_system_ram_available_gib": round(self._minimum_available, 2) if self._minimum_available is not None else None,
            "note": "100 ms sampling may miss shorter transient peaks; process-tree RSS includes visible child processes when psutil is available.",
        }


def _make_synthetic_roi(target_points: int) -> tuple[ROIVolume, dict[str, Any]]:
    """Create a deterministic compact test volume close to target sampled points."""
    target = max(1_000, min(int(target_points), MAX_SAFE_POINTS))
    n_theta = 32
    n_xy = max(32, 2 * int(math.floor(math.sqrt(target / (n_theta / 2)))))
    while n_xy * n_xy * n_theta > MAX_SYNTHETIC_VOXELS:
        n_xy -= 2
    if n_xy < 8:
        raise RuntimeError("Не удалось сформировать безопасный тестовый объём.")

    x = np.linspace(-1.0, 1.0, n_xy, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, n_xy, dtype=np.float64)
    theta = np.linspace(-1.0, 1.0, n_theta, dtype=np.float64)

    # Broadcasting avoids building several full 3D coordinate grids.
    xx = x.astype(np.float32)[:, None]
    yy = y.astype(np.float32)[None, :]
    tt = theta.astype(np.float32)
    lobe_a = np.exp(-(((xx - 0.22) ** 2 + (yy + 0.12) ** 2) / 0.19)).astype(np.float32)
    lobe_b = 0.72 * np.exp(-(((xx + 0.32) ** 2 + (yy - 0.28) ** 2) / 0.085)).astype(np.float32)
    angular_a = np.exp(-(((tt - 0.15) ** 2) / 0.22)).astype(np.float32)
    angular_b = 0.62 * np.exp(-(((tt + 0.42) ** 2) / 0.10)).astype(np.float32)
    plane = (lobe_a + lobe_b).astype(np.float32)
    volume = (plane[:, :, None] * angular_a[None, None, :] + 0.35 * plane[:, :, None] * angular_b[None, None, :]).astype(np.float32)

    # Constructor order matches the production ROI extraction code:
    # ROIVolume(x, y, theta, volume, coord_system, roi).
    rv = ROIVolume(x, y, theta, volume, "xy", None)
    info = {
        "synthetic_shape": list(volume.shape),
        "synthetic_voxels": int(volume.size),
        "synthetic_array_mib": round(volume.nbytes / (1024 ** 2), 2),
        "target_render_points": target,
    }
    return rv, info


def _load_input_npz(
    path: Path, *, include_extraction: bool = False
) -> tuple[ROIVolume, dict[str, Any], dict[str, np.ndarray] | None, dict[str, Any] | None]:
    """Load extracted ROI, optional detector projection and bounded raw extraction input."""
    with np.load(path, allow_pickle=False) as data:
        required = {"volume", "x", "y", "theta"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ не содержит обязательные массивы: {', '.join(sorted(missing))}")
        volume = np.asarray(data["volume"])
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        theta = np.asarray(data["theta"], dtype=np.float64).reshape(-1)
        metadata: dict[str, Any] = {}
        if "metadata_json" in data.files:
            raw_meta = data["metadata_json"]
            metadata = json.loads(str(raw_meta.item())) if raw_meta.shape == () else {}
        overlay = None
        extraction_input = None
        if include_extraction:
            extraction_keys = {"raw_scan", "scan_x", "scan_y", "scan_theta", "scan_axis"}
            if extraction_keys.issubset(data.files):
                with zipfile.ZipFile(path, "r") as archive:
                    raw_member = next((item for item in archive.infolist() if item.filename == "raw_scan.npy"), None)
                    if raw_member is None:
                        raise ValueError("В NPZ не найден raw_scan.npy")
                    max_raw_bytes = MAX_RAW_EXTRACTION_VOXELS * 8
                    if raw_member.file_size > max_raw_bytes:
                        raise ValueError(
                            f"raw_scan.npy занимает {raw_member.file_size / (1024 ** 2):.1f} MiB без сжатия; "
                            f"предел безопасного экспорта {max_raw_bytes / (1024 ** 2):.1f} MiB."
                        )
                raw_scan = np.asarray(data["raw_scan"])
                scan_x = np.asarray(data["scan_x"], dtype=np.float64).reshape(-1)
                scan_y = np.asarray(data["scan_y"], dtype=np.float64).reshape(-1)
                scan_theta = np.asarray(data["scan_theta"], dtype=np.float64).reshape(-1)
                scan_axis = int(np.asarray(data["scan_axis"]).item())
                if raw_scan.ndim != 3 or scan_axis not in (0, 2) or not np.issubdtype(raw_scan.dtype, np.number):
                    raise ValueError("raw_scan должен быть числовым 3D-массивом, scan_axis — 0 или 2")
                expected_yx = raw_scan.shape[1:] if scan_axis == 0 else raw_scan.shape[:2]
                if expected_yx != (scan_y.size, scan_x.size) or raw_scan.shape[scan_axis] != scan_theta.size:
                    raise ValueError("raw_scan не согласован с scan_x/scan_y/scan_theta/scan_axis")
                if raw_scan.size > MAX_RAW_EXTRACTION_VOXELS:
                    raise ValueError(
                        f"raw_scan содержит {raw_scan.size:,} вокселей; ограничение бенчмарка "
                        f"{MAX_RAW_EXTRACTION_VOXELS:,}. Экспортируйте только допустимый scan-view."
                    )
                if not np.isfinite(scan_theta).all():
                    raise ValueError("scan_theta содержит NaN/Inf")
                extraction_input = {
                    "raw_scan": raw_scan, "scan_x": scan_x, "scan_y": scan_y,
                    "scan_theta": scan_theta, "scan_axis": scan_axis,
                }
        overlay_keys = {"full_projection", "detector_x", "detector_y"}
        if overlay_keys.issubset(data.files):
            full_projection = np.asarray(data["full_projection"], dtype=np.float32)
            detector_x = np.asarray(data["detector_x"], dtype=np.float64).reshape(-1)
            detector_y = np.asarray(data["detector_y"], dtype=np.float64).reshape(-1)
            if full_projection.shape != (detector_x.size, detector_y.size):
                raise ValueError(
                    "full_projection должна иметь форму (len(detector_x), len(detector_y)); "
                    f"получено {full_projection.shape}"
                )
            if not np.isfinite(full_projection).any():
                raise ValueError("full_projection не содержит конечных значений")
            overlay = {
                "full_projection": full_projection,
                "detector_x": detector_x,
                "detector_y": detector_y,
            }
    if volume.ndim != 3 or volume.shape != (x.size, y.size, theta.size):
        raise ValueError(f"Форма массива {volume.shape} не согласована с осями {(x.size, y.size, theta.size)}")
    if volume.size > MAX_SYNTHETIC_VOXELS:
        raise ValueError(f"Экспорт содержит {volume.size:,} вокселей; диагностический предел {MAX_SYNTHETIC_VOXELS:,}. Разбейте ROI или экспортируйте именно используемый приложением ROI.")
    roi_data = metadata.get("roi") or {}
    if not isinstance(roi_data, dict):
        roi_data = {}
    if not roi_data:
        roi_data = {"cx": float(np.nanmean(x)), "cy": float(np.nanmean(y)),
                    "width": float(np.ptp(x)), "height": float(np.ptp(y)),
                    "angle": 0.0, "label": "Benchmark ROI", "color": "#ffb347"}
    roi = ROI(
        cx=float(roi_data.get("cx", np.nanmean(x))),
        cy=float(roi_data.get("cy", np.nanmean(y))),
        width=float(roi_data.get("width", np.ptp(x))),
        height=float(roi_data.get("height", np.ptp(y))),
        angle=float(roi_data.get("angle", 0.0)),
        label=str(roi_data.get("label", "Benchmark ROI")),
        color=str(roi_data.get("color", "#ffb347")),
        mode=str(roi_data.get("mode", "mean")),
        type=str(roi_data.get("type", "rect")),
    )
    coord_system = str(metadata.get("coord_system", "xy"))
    rv = ROIVolume(x, y, theta, volume, coord_system, roi)
    info = {
        "input_kind": "real_roi_npz", "input_path": str(path.resolve()),
        "shape": list(volume.shape), "voxels": int(volume.size),
        "dtype": str(volume.dtype), "array_mib": round(volume.nbytes / (1024 ** 2), 2),
        "finite_voxels": int(np.isfinite(volume).sum()),
        "nan_voxels": int(np.isnan(volume).sum()),
        "coord_system": coord_system, "roi": asdict(roi),
        "full_detector_projection_available": overlay is not None,
        "full_detector_projection_shape": list(overlay["full_projection"].shape) if overlay else None,
        "full_projection_compute_ms": metadata.get("full_projection_compute_ms"),
        "benchmark_settings": metadata.get("benchmark_settings", {}) if isinstance(metadata.get("benchmark_settings", {}), dict) else {},
    }
    return rv, info, overlay, extraction_input


def _benchmark_extraction(
    extraction_input: dict[str, Any], roi: ROI, repeats: int
) -> dict[str, Any]:
    """Repeat the exact crop/stride/extract sequence used by _RenderWorker."""
    raw = extraction_input["raw_scan"]
    axis = int(extraction_input["scan_axis"])
    x = extraction_input["scan_x"]
    y = extraction_input["scan_y"]
    theta = extraction_input["scan_theta"]

    crop_call = lambda: crop_raw_to_roi(raw, axis, x, y, roi)
    cropped_result, crop_samples = _timed(crop_call, repeats)
    cropped, crop_x, crop_y, bounds = cropped_result
    input_stride, estimated_shape, estimated_voxels = effective_extraction_stride(
        roi, crop_x, crop_y, len(theta), requested=1
    )
    raw_view = cropped[::input_stride, ::input_stride, ::input_stride]
    theta_view = theta[::input_stride]
    x_view = crop_x[::input_stride]
    y_view = crop_y[::input_stride]

    def extract_call():
        return extract_roi_volume(
            raw_view, theta_view, roi, axis=axis,
            x_coords=x_view, y_coords=y_view, order=1,
        )

    extracted = extract_call()  # warm-up; not part of the timings
    extracted, extract_samples = _timed(extract_call, repeats)
    return {
        "crop_raw_to_roi": _summary(crop_samples),
        "extract_roi_volume": _summary(extract_samples),
        "worker_cpu_total_median_ms": round(
            statistics.median(crop_samples) + statistics.median(extract_samples), 3
        ),
        "raw_scan_shape": list(raw.shape),
        "raw_scan_dtype": str(raw.dtype),
        "raw_scan_mib": round(raw.nbytes / (1024 ** 2), 2),
        "crop_bounds_xy": list(map(int, bounds)),
        "cropped_raw_shape": list(cropped.shape),
        "extraction_input_stride": int(input_stride),
        "estimated_extraction_shape": list(estimated_shape),
        "estimated_extraction_voxels": int(estimated_voxels),
        "extracted_volume_shape": list(extracted.volume.shape),
        "extracted_volume_dtype": str(extracted.volume.dtype),
        "extracted_volume_mib": round(extracted.volume.nbytes / (1024 ** 2), 2),
        "extracted_coord_system": str(extracted.coord_system),
        "note": "Same crop_raw_to_roi → effective_extraction_stride → extract_roi_volume path as _RenderWorker; HDF5 loading is excluded.",
    }


def _production_settings(theme: str, rv: ROIVolume, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mirror _RenderWorker's quality/stride policy and saved display settings."""
    defaults = {
        "font_family": "Times New Roman", "tick_size": 14, "label_size": 16,
        "axis_tick_length": 3, "show_grid": True, "show_projection": True,
        "show_colorbar": True, "show_legend": False, "volume_colormap": "magma",
        "projection_colorscale": "Greys", "volume_opacity": 0.10,
        "volume_opacity_mode": "uniform", "projection_opacity": 0.75,
        "isomin_fraction": 0.16, "isomax_fraction": 0.95,
        "surface_count": 25, "smooth_sigma": 0.5, "volume_stride": 2,
        "quality": "auto", "dragmode": "orbit",
        "projection_type": "perspective", "rotation_speed": 14.0,
    }
    # Use the app's validated loader when its Qt dependencies are installed.
    # CPU-only benchmarking remains possible without importing Qt widgets.
    try:
        from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings
        settings = dict(load_plot_display_settings()["volume"])
        settings_source = "application validated settings loader"
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("PyQt6"):
            raise
        from roc_mcs.gui.app_settings import load_settings
        raw = load_settings().get("roi_explorer_display", {}).get("volume", {})
        settings = dict(defaults)
        if isinstance(raw, dict):
            settings.update({key: value for key, value in raw.items() if key in defaults})
        settings_source = "settings.json + benchmark defaults (Qt unavailable)"
    allowed_overrides = {
        "font_family", "tick_size", "label_size", "axis_tick_length",
        "show_grid", "show_projection", "show_colorbar", "show_legend",
        "volume_colormap", "projection_colorscale", "volume_opacity",
        "volume_opacity_mode", "projection_opacity", "isomin_fraction",
        "isomax_fraction", "surface_count", "smooth_sigma", "volume_stride",
        "quality", "dragmode", "projection_type",
    }
    applied_overrides = {}
    for key, value in (overrides or {}).items():
        if key in allowed_overrides:
            settings[key] = value
            applied_overrides[key] = value
    settings["_bench_overrides"] = applied_overrides
    settings["_bench_settings_source"] = settings_source
    quality = str(settings.get("quality", "auto"))
    point_cap = {"fast": 25_000, "balanced": 50_000,
                 "auto": MAX_RENDER_POINTS, "full": MAX_RENDER_POINTS}.get(quality, MAX_RENDER_POINTS)
    requested = max(2, int(settings.get("volume_stride", 2)))
    settings["volume_stride"] = effective_volume_stride(rv.volume.shape, requested, point_cap)
    settings["max_render_points"] = point_cap
    settings["surface_count"] = max(2, min(25, int(settings.get("surface_count", 25))))
    settings["_bench_requested_stride"] = requested
    settings["_bench_quality_cap"] = point_cap
    return settings


def _build_cpu_report(points: int, repeats: int, *, rv_input: ROIVolume | None = None,
                      input_info: dict[str, Any] | None = None,
                      overlay_input: dict[str, np.ndarray] | None = None,
                      extraction_input: dict[str, Any] | None = None,
                      theme: str = "dark", webgl_overlay: str = "off") -> tuple[dict[str, Any], str]:
    memory_before = _memory_snapshot()
    if rv_input is None:
        rv, data_info = _make_synthetic_roi(points)
    else:
        rv = rv_input
        data_info = dict(input_info or {})
    memory_after_data = _memory_snapshot()
    settings = _production_settings(theme, rv, data_info.get("benchmark_settings"))
    extraction_report = None
    if extraction_input is not None:
        available_gib = _memory_snapshot().get("system_ram_available_gib")
        if available_gib is not None and available_gib < 1.5:
            extraction_report = {
                "status": "skipped_low_memory",
                "reason": f"Доступно только {available_gib:.2f} GiB RAM; extraction benchmark требует запас >=1.5 GiB.",
            }
        else:
            extraction_report = {"status": "ok", **_benchmark_extraction(extraction_input, rv.roi, repeats)}

    # Один прогрев отделяет холодные импорты/инициализацию библиотек от steady-state.
    prepare_call = lambda: prepare_volume(
            rv, smooth_sigma=float(settings.get("smooth_sigma", 0.5)),
            stride=int(settings["volume_stride"]),
            max_points=int(settings["max_render_points"]),
        )
    prepare_call()  # warm-up; excluded from timings
    sampled_result, prepare_samples = _timed(prepare_call, repeats)
    sampled, projection, _x, _y, _z, values = sampled_result

    import plotly
    import plotly.io as pio
    mode_reports: dict[str, Any] = {}
    payloads: dict[str, str] = {}
    scenarios = {"roi_projection_only": None}
    if overlay_input is not None:
        scenarios["full_detector_projection"] = overlay_input

    for scenario_name, overlay in scenarios.items():
        scenario_modes: dict[str, Any] = {}
        for mode in ("uniform", "soft"):
            mode_settings = dict(settings)
            mode_settings["volume_opacity_mode"] = mode
            figure_call = lambda current=mode_settings, current_overlay=overlay: build_volume_figure(
                rv, current, theme,
                full_projection=current_overlay["full_projection"] if current_overlay else None,
                detector_x=current_overlay["detector_x"] if current_overlay else None,
                detector_y=current_overlay["detector_y"] if current_overlay else None,
                overlay_roi=rv.roi if current_overlay else None,
            )
            figure_call()  # warm-up; excluded from steady-state timings
            figure, figure_samples = _timed(figure_call, repeats)
            json_call = lambda current=figure: pio.to_json(current, validate=False, pretty=False, remove_uids=False)
            json_call()  # serializer warm-up; excluded from steady-state timings
            payload, serialization_samples = _timed(json_call, repeats)
            payloads[f"{scenario_name}:{mode}"] = payload
            volume_trace = next((trace for trace in figure.data if getattr(trace, "type", "") == "volume"), None)
            surface_trace = next((trace for trace in figure.data if getattr(trace, "type", "") == "surface"), None)
            opacity_scale = getattr(volume_trace, "opacityscale", None) if volume_trace is not None else None
            opacity_point_count = len(opacity_scale) if isinstance(opacity_scale, (list, tuple)) else None
            scenario_modes[mode] = {
                "build_volume_figure_including_prepare": _summary(figure_samples),
                "json_serialization": _summary(serialization_samples),
                "json_payload_mib": round(len(payload.encode("utf-8")) / (1024 ** 2), 2),
                "volume_trace_points": int(np.asarray(volume_trace.value).size) if volume_trace is not None else 0,
                "projection_grid_points": int(np.asarray(surface_trace.surfacecolor).size) if surface_trace is not None else 0,
                "opacityscale_type": "uniform" if opacity_scale == "uniform" else "explicit transfer function",
                "opacityscale_control_points": opacity_point_count,
                "trace_types": [str(getattr(trace, "type", "unknown")) for trace in figure.data],
            }
            del figure
        mode_reports[scenario_name] = scenario_modes

    if webgl_overlay == "on" and overlay_input is None:
        raise ValueError("Для --webgl-overlay on NPZ должен содержать full_projection, detector_x и detector_y.")
    selected_scenario = "full_detector_projection" if webgl_overlay == "on" else "roi_projection_only"
    active_mode = str(settings.get("volume_opacity_mode", "uniform"))
    if active_mode not in ("uniform", "soft"):
        active_mode = "uniform"
    payload = payloads[f"{selected_scenario}:{active_mode}"]
    memory_after = _memory_snapshot()
    version_text = str(getattr(plotly, "__version__", "0.0"))
    try:
        version_pair = tuple(int(part) for part in version_text.split(".")[:2])
    except (TypeError, ValueError):
        version_pair = (0, 0)
    plotly_supported = (7, 1) <= version_pair < (8, 0)
    report = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu_count_logical": os.cpu_count(), "plotly_version": version_text,
        "theme": theme, "active_opacity_mode": active_mode,
        "compatibility": {"plotly_required_by_widget": ">=7.1,<8.0", "plotly_supported_by_roi_widget": plotly_supported},
        "data": data_info,
        "application_settings_source": settings.get("_bench_settings_source"),
        "notebook_settings_overrides": settings.get("_bench_overrides", {}),
        "application_settings": {k: v for k, v in settings.items() if not k.startswith("_bench_")},
        "render_policy": {
            "quality": settings.get("quality", "auto"),
            "requested_volume_stride": settings.get("_bench_requested_stride"),
            "effective_volume_stride": settings.get("volume_stride"),
            "max_render_points": settings.get("_bench_quality_cap"),
            "surface_count": settings.get("surface_count"),
        },
        "extraction_pipeline": extraction_report or {
            "status": "not_requested",
            "reason": data_info.get("extraction_skip_reason", "Экспорт raw_scan отсутствует или не указан --include-extraction."),
        },
        "pipeline": {
            "prepare_volume": _summary(prepare_samples),
            "sampled_volume_shape": list(sampled.shape),
            "projection_shape": list(projection.shape),
            "prepared_grid_points": int(np.asarray(values).size),
            "scenarios": mode_reports,
            "webgl_payload_scenario": selected_scenario,
        },
        "memory": {"before": memory_before, "after_data_load": memory_after_data, "after_figures_and_json": memory_after},
        "webgl": {"status": "not_run", "reason": "Добавьте --webgl для проверки Qt WebEngine/WebGL."},
    }
    return report, payload


def _run_webgl_probe(payload: str, timeout_s: int = 120) -> dict[str, Any]:
    """Render the synthetic production Plotly figure in a real Qt WebEngine page."""
    from roc_mcs.gui.qt_webengine_bootstrap import prepare_qt_webengine

    bootstrap_error = prepare_qt_webengine("ROI 3D performance benchmark")
    if bootstrap_error:
        return {"status": "failed", "error": f"Qt WebEngine bootstrap: {bootstrap_error}"}

    try:
        from importlib.metadata import PackageNotFoundError, version as package_version
        from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, QUrl
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        import plotly
    except Exception as exc:
        return {"status": "failed", "error": f"Не удалось импортировать Qt WebEngine: {type(exc).__name__}: {exc}"}

    try:
        webengine_version = package_version("PyQt6-WebEngine")
    except PackageNotFoundError:
        return {"status": "failed", "error": "Не установлен пакет PyQt6-WebEngine."}
    match = re.match(r"^(\d+)\.(\d+)", str(webengine_version))
    webengine_pair = tuple(map(int, match.groups())) if match else (0, 0)
    if not (6, 11) <= webengine_pair < (7, 0):
        return {
            "status": "failed",
            "error": f"PyQt6-WebEngine {webengine_version} несовместим с GUI; требуется >=6.11,<7.0.",
        }

    js_file = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    if not js_file.is_file() or js_file.stat().st_size < 100_000:
        return {"status": "failed", "error": f"Не найден локальный Plotly.js: {js_file}"}

    app = QApplication.instance() or QApplication([sys.argv[0]])
    view = QWebEngineView()
    view.setWindowTitle("ROI 3D diagnostic benchmark — synthetic data")
    view.resize(900, 650)
    web_settings = view.settings()
    for attr_name in ("JavascriptEnabled", "LocalContentCanAccessFileUrls", "WebGLEnabled"):
        attr = getattr(QWebEngineSettings.WebAttribute, attr_name, None)
        if attr is not None:
            web_settings.setAttribute(attr, True)
    remote = getattr(QWebEngineSettings.WebAttribute, "LocalContentCanAccessRemoteUrls", None)
    if remote is not None:
        web_settings.setAttribute(remote, False)

    # Validate JSON on the Python side, then stream it in small sequential IPC chunks.
    # setHtml() has a ~2 MiB percent-encoded limit, and a single very large
    # runJavaScript source string would also create unnecessary IPC pressure.
    json.loads(payload)
    # Match the production widget's 128 KiB chunk size; the transport is
    # runJavaScript-callback IPC here rather than Qt WebChannel ACKs.
    transfer_chunk_chars = 128 * 1024
    transfer_chunks = [payload[i:i + transfer_chunk_chars] for i in range(0, len(payload), transfer_chunk_chars)]
    html = r'''<!doctype html><html><head><meta charset="utf-8">
<style>html,body,#plot{margin:0;width:100%;height:100%;overflow:hidden}#plot{position:absolute;inset:0}</style>
<script src="plotly.min.js"></script></head><body><div id="plot"></div><script>
window.__roi3dBenchResult = null;
window.__roi3dBenchError = null;
let __roi3dTransfer = null;
window.beginRoi3dBenchTransfer = function(count){
  if(!Number.isInteger(count)||count<1||count>2048)throw new Error('Недопустимое число частей Figure');
  __roi3dTransfer={chunks:new Array(count),received:0};
};
window.receiveRoi3dBenchChunk = function(index,count,chunk){
  try{
    if(!__roi3dTransfer||count!==__roi3dTransfer.chunks.length||!Number.isInteger(index)||index<0||index>=count||typeof chunk!=='string')throw new Error('Нарушена последовательность передачи Figure');
    if(__roi3dTransfer.chunks[index]===undefined){__roi3dTransfer.chunks[index]=chunk;__roi3dTransfer.received++;}
    if(__roi3dTransfer.received===count){
      const text=__roi3dTransfer.chunks.join('');__roi3dTransfer=null;
      window.runRoi3dBench(JSON.parse(text),text);
    }
  }catch(e){window.__roi3dBenchError=String(e&&e.stack||e);window.__roi3dBenchResult={status:'failed',error:window.__roi3dBenchError};}
};
const __frameIntervals=[];let __frameLast=performance.now(),__frameMonitor=true;
function __monitorFrames(now){__frameIntervals.push(now-__frameLast);__frameLast=now;if(__frameMonitor)requestAnimationFrame(__monitorFrames);}
requestAnimationFrame(__monitorFrames);
window.addEventListener('error', e => { window.__roi3dBenchError = String(e.message || e.error || 'JavaScript error'); });
window.addEventListener('unhandledrejection', e => { window.__roi3dBenchError = String(e.reason && e.reason.message || e.reason || 'Promise rejection'); });
function median(a){const b=[...a].sort((x,y)=>x-y);return b[Math.floor(b.length/2)]||0;}
function percentile(a,p){const b=[...a].sort((x,y)=>x-y);return b.length?b[Math.min(b.length-1,Math.max(0,Math.ceil((b.length-1)*p)))]:0;}
function rendererInfo(g){
  const canvases=[...g.querySelectorAll('canvas')];
  for(const canvas of canvases){
    let gl=null, kind='';
    try{gl=canvas.getContext('webgl2');if(gl)kind='WebGL 2';}catch(e){}
    if(!gl){try{gl=canvas.getContext('webgl')||canvas.getContext('experimental-webgl');if(gl)kind='WebGL 1';}catch(e){}}
    if(!gl)continue;
    let renderer='unknown', vendor='unknown', totalGpuMiB=null;
    try{
      const ext=gl.getExtension('WEBGL_debug_renderer_info');
      renderer=String(ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER));
      vendor=String(ext?gl.getParameter(ext.UNMASKED_VENDOR_WEBGL):gl.getParameter(gl.VENDOR));
    }catch(e){}
    try{
      const ext=gl.getExtension('GL_NVX_gpu_memory_info');
      if(ext){const value=gl.getParameter(ext.GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX);if(Number.isFinite(value)&&value>0)totalGpuMiB=Math.round(value/1024);}
    }catch(e){}
    return {available:true,version:kind,renderer,vendor,total_gpu_memory_mib:totalGpuMiB,canvas_count:canvases.length};
  }
  return {available:false,version:'none',renderer:'none',vendor:'none',canvas_count:canvases.length};
}
window.runRoi3dBench = async function(figure,serializedFigure){
  try{
    if(!figure || !Array.isArray(figure.data) || !figure.layout) throw new Error('Переданная Plotly Figure повреждена');
    const graph=document.getElementById('plot');
    const started=performance.now();
    await Plotly.react(graph,figure.data,figure.layout,{responsive:false,displayModeBar:false,scrollZoom:false});
    const initial=(performance.now()-started);
    const rebuildTimes=[];
    if(typeof serializedFigure==='string'){
      for(let i=0;i<2;i++){
        const fresh=JSON.parse(serializedFigure);
        const rebuildStart=performance.now();
        await Plotly.react(graph,fresh.data,fresh.layout,{responsive:false,displayModeBar:false,scrollZoom:false});
        rebuildTimes.push(performance.now()-rebuildStart);
      }
    }
    const vi=figure.data.findIndex(t=>t.type==='volume');
    const pi=figure.data.findIndex(t=>t.type==='surface'&&t.name==='Интегральная проекция');
    if(vi<0)throw new Error('Volume trace не найден в диагностической фигуре');
    const thresholdBase=Number(figure.layout&&figure.layout.meta&&figure.layout.meta.roi3d&&figure.layout.meta.roi3d.threshold_base)||1;
    const contrastTimes=[];
    for(let i=0;i<3;i++){
      const t=performance.now();
      await Plotly.restyle(graph,{isomin:[thresholdBase*(0.14+i*0.01)],isomax:[thresholdBase*0.95]},[vi]);
      contrastTimes.push(performance.now()-t);
    }
    let projectionPaletteBefore=null, projectionPaletteAfter=null;
    if(pi>=0)projectionPaletteBefore=JSON.stringify(graph.data[pi].colorscale);
    const paletteTimes=[];
    for(let i=0;i<3;i++){
      const t=performance.now();
      await Plotly.restyle(graph,{colorscale:[i%2?'Magma':'Viridis']},[vi]);
      paletteTimes.push(performance.now()-t);
    }
    if(pi>=0)projectionPaletteAfter=JSON.stringify(graph.data[pi].colorscale);
    const softScale=[[0,0],[0.10,0],[0.22,0.015],[0.36,0.055],[0.52,0.16],[0.68,0.36],[0.84,0.68],[1,1]];
    const opacityTimes=[];
    for(let i=0;i<3;i++){
      const t=performance.now();
      await Plotly.restyle(graph,{opacityscale:[i%2?softScale:[[0,1],[1,1]]]},[vi]);
      opacityTimes.push(performance.now()-t);
    }
    const widthBefore=graph.style.width;
    graph.style.width='97%';
    const resizeStart=performance.now();
    await Plotly.Plots.resize(graph);
    const resizeMs=performance.now()-resizeStart;
    graph.style.width=widthBefore||'100%';
    await Plotly.Plots.resize(graph);
    const baseCamera=(graph.layout&&graph.layout.scene&&graph.layout.scene.camera)||{eye:{x:1.6,y:-1.9,z:1.25},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}};
    const cameraTimes=[];
    for(let i=0;i<3;i++){
      const shift=(i%2?1:-1)*0.035;
      const moved={eye:{x:baseCamera.eye.x+shift,y:baseCamera.eye.y,z:baseCamera.eye.z},center:{x:baseCamera.center.x+shift,y:baseCamera.center.y,z:baseCamera.center.z},up:baseCamera.up};
      const t=performance.now();await Plotly.relayout(graph,{'scene.camera':moved});cameraTimes.push(performance.now()-t);
    }
    await Plotly.relayout(graph,{'scene.camera':baseCamera});
    await new Promise(resolve=>setTimeout(resolve,250));
    __frameMonitor=false;
    const frameIntervals=__frameIntervals.slice();
    const gl=rendererInfo(graph);
    window.__roi3dBenchResult={
      status:'ok',initial_render_ms:initial,
      repeat_rebuild_ms:{median:median(rebuildTimes),samples:rebuildTimes},
      contrast_restyle_ms:{median:median(contrastTimes),samples:contrastTimes},
      volume_palette_restyle_ms:{median:median(paletteTimes),samples:paletteTimes},
      opacity_mode_restyle_ms:{median:median(opacityTimes),samples:opacityTimes},
      resize_ms:resizeMs,
      camera_relayout_ms:{median:median(cameraTimes),samples:cameraTimes},
      animation_frame_ms:{median:median(frameIntervals),p95:percentile(frameIntervals,0.95),max:Math.max(...frameIntervals),over_50ms:frameIntervals.filter(v=>v>50).length,samples:frameIntervals.length},
      webgl:gl,
      projection_palette_preserved:projectionPaletteBefore===projectionPaletteAfter,
      user_agent:navigator.userAgent
    };
  }catch(e){window.__roi3dBenchError=String(e&&e.stack||e);window.__roi3dBenchResult={status:'failed',error:window.__roi3dBenchError};}
};
</script></body></html>'''

    loop = QEventLoop()
    poll_timer = QTimer()
    poll_timer.setInterval(150)
    deadline = QTimer()
    deadline.setSingleShot(True)
    memory_timer = QTimer()
    memory_timer.setInterval(250)
    memory_samples: list[dict[str, Any]] = []
    memory_timer.timeout.connect(lambda: memory_samples.append(_memory_snapshot()))
    state: dict[str, Any] = {"result": None, "load_error": None, "started": False}
    app = QApplication.instance()
    quit_on_last_window = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)

    def finish_with_error(message: str) -> None:
        if state["result"] is None:
            state["result"] = {"status": "failed", "error": message}
        loop.quit()

    def read_result(raw: Any) -> None:
        if not raw:
            return
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return
        if isinstance(parsed, dict) and parsed.get("status") in {"ok", "failed"}:
            state["result"] = parsed
            loop.quit()

    def poll() -> None:
        view.page().runJavaScript(
            "window.__roi3dBenchResult ? JSON.stringify(window.__roi3dBenchResult) : "
            "(window.__roi3dBenchError ? JSON.stringify({status:'failed',error:window.__roi3dBenchError}) : '')",
            read_result,
        )

    def loaded(ok: bool) -> None:
        if not ok:
            finish_with_error("Qt WebEngine не смог загрузить локальную диагностическую страницу.")
            return
        if state["started"]:
            return
        state["started"] = True
        transfer_state = {"index": 0}

        def send_next_chunk(_result: Any = None) -> None:
            if state["result"] is not None:
                return
            index = transfer_state["index"]
            if index >= len(transfer_chunks):
                poll_timer.start()
                return
            chunk_literal = json.dumps(transfer_chunks[index], ensure_ascii=False, separators=(",", ":"))
            script = f"window.receiveRoi3dBenchChunk({index},{len(transfer_chunks)},{chunk_literal})"
            transfer_state["index"] += 1
            view.page().runJavaScript(script, send_next_chunk)

        def transfer_started(_result: Any = None) -> None:
            send_next_chunk()

        view.page().runJavaScript(
            f"window.beginRoi3dBenchTransfer({len(transfer_chunks)})",
            transfer_started,
        )

    poll_timer.timeout.connect(poll)
    memory_timer.start()
    deadline.timeout.connect(lambda: finish_with_error(f"Превышен лимит ожидания {timeout_s} с."))
    view.loadFinished.connect(loaded)
    base_url = QUrl.fromLocalFile(str(js_file.parent / "index.html"))
    view.setHtml(html, base_url)
    view.show()
    deadline.start(int(timeout_s * 1000))
    try:
        loop.exec()
    finally:
        poll_timer.stop()
        memory_timer.stop()
        deadline.stop()
        try:
            page = view.page()
            page.triggerAction(page.WebAction.Stop)
            view.close()
            view.deleteLater()
            # Explicitly flush DeferredDelete before QApplication tears down its
            # default QWebEngineProfile; this avoids the profile/page lifetime warning.
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        except Exception:
            pass
        app.setQuitOnLastWindowClosed(quit_on_last_window)
    result = state["result"] or {"status": "failed", "error": "Нет результата от диагностической страницы."}
    if memory_samples:
        process_tree_values = [item.get("process_tree_rss_mib") for item in memory_samples if item.get("process_tree_rss_mib") is not None]
        available_values = [item.get("system_ram_available_gib") for item in memory_samples if item.get("system_ram_available_gib") is not None]
        result["memory_sampling"] = {
            "sample_interval_ms": 250,
            "samples": len(memory_samples),
            "peak_process_tree_rss_mib": max(process_tree_values) if process_tree_values else None,
            "min_system_ram_available_gib": min(available_values) if available_values else None,
            "note": "process tree includes this Python process and visible Qt WebEngine child processes; 250 ms polling can miss shorter peaks",
        }
    return result


def _classify(report: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not report.get("compatibility", {}).get("plotly_supported_by_roi_widget", True):
        return "FAIL", [
            "Установленная версия Plotly не соответствует ограничению самого ROI 3D widget (требуется >=7.1,<8.0).",
            "Проверьте активный Python interpreter и зависимости именно этого окружения.",
        ]
    webgl = report.get("webgl", {})
    if webgl.get("status") == "not_run":
        return "НЕ ПРОВЕРЕНО: CPU-only", ["GPU/WebGL и фактическая отрисовка Qt WebEngine не проверялись."]
    if webgl.get("status") != "ok":
        return "FAIL", [str(webgl.get("error", "WebGL benchmark failed"))]
    info = webgl.get("webgl", {})
    if not info.get("available"):
        return "FAIL", ["WebGL-контекст не создан; 3D Plotly в этом Qt WebEngine не подтверждён."]
    renderer = str(info.get("renderer", "")).lower()
    software_tokens = ("swiftshader", "llvmpipe", "softpipe", "swrast", "software rasterizer", "warp")
    if any(token in renderer for token in software_tokens):
        notes.append("Обнаружен программный рендерер, а не аппаратный GPU backend: " + str(info.get("renderer")))
    elif renderer.strip() in {"", "unknown", "none"}:
        notes.append("WebGL доступен, но модель GPU/рендерера не удалось достоверно определить.")
    if webgl.get("projection_palette_preserved") is False:
        notes.append("Проверка выявила изменение палитры 2D-проекции после обновления палитры 3D-объёма.")
        return "FAIL", notes

    contrast = float(webgl.get("contrast_restyle_ms", {}).get("median", math.inf))
    palette = float(webgl.get("volume_palette_restyle_ms", {}).get("median", math.inf))
    initial = float(webgl.get("initial_render_ms", math.inf))
    camera_ms = float(webgl.get("camera_relayout_ms", {}).get("median", math.inf))
    frame_p95 = float(webgl.get("animation_frame_ms", {}).get("p95", math.inf))
    frame_max = float(webgl.get("animation_frame_ms", {}).get("max", math.inf))
    if notes:
        return "WARN", notes
    rebuild_ms = float(webgl.get("repeat_rebuild_ms", {}).get("median", 0.0))
    if initial > 5_000 or rebuild_ms > 5_000 or contrast > 500 or palette > 500 or camera_ms > 500 or frame_max > 1_000:
        notes.append("Есть измеримые задержки/длинные блокировки главного потока; сравните тайминги с целевым лабораторным компьютером.")
        return "WARN", notes
    if initial > 2_000 or rebuild_ms > 2_000 or contrast > 150 or palette > 150 or camera_ms > 100 or frame_p95 > 50 or frame_max > 250:
        notes.append("Рендеринг работоспособен, но плавность не соответствует строгим интерактивным целям (контраст/палитра <150 мс, камера <100 мс, p95 кадра <50 мс).")
        return "WARN", notes
    notes.append("Синтетический тест прошёл заданные пороги. Это не гарантия отсутствия сбоев на реальном эксперименте.")
    return "PASS", notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Измерение production ROI 3D pipeline: Python, JSON, Qt WebEngine и WebGL."
    )
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS,
                        help=f"Целевое число точек только для синтетики (1000..{MAX_SAFE_POINTS}).")
    parser.add_argument("--input-npz", type=Path, default=None,
                        help="Экспорт реального ROIVolume (volume,x,y,theta,metadata_json) из ноутбука.")
    parser.add_argument("--include-extraction", action="store_true",
                        help="Дополнительно измерить crop_raw_to_roi + extract_roi_volume по raw_scan из NPZ.")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark", help="Тема сцены для WebGL-прогона.")
    parser.add_argument("--repeats", type=int, default=2, help="Повторы CPU-измерений (1..5).")
    parser.add_argument("--webgl", action="store_true", help="Открыть окно Qt WebEngine и проверить реальную GPU/WebGL-отрисовку.")
    parser.add_argument("--webgl-overlay", choices=("off", "on"), default="off",
                        help="Для WebGL выбрать ROI-проекцию или полный detector XY overlay (нужен экспорт full_projection в NPZ).")
    parser.add_argument("--json-out", type=Path, default=None, help="Сохранить подробный отчёт JSON.")
    args = parser.parse_args(argv)

    if args.input_npz is None and not 1_000 <= args.points <= MAX_SAFE_POINTS:
        parser.error(f"Для синтетики --points должен быть от 1000 до {MAX_SAFE_POINTS}.")
    if not 1 <= args.repeats <= 5:
        parser.error("--repeats должен быть от 1 до 5.")
    if args.input_npz is not None and not args.input_npz.is_file():
        parser.error(f"Файл не найден: {args.input_npz}")
    if args.webgl_overlay == "on" and not args.webgl:
        parser.error("--webgl-overlay применяется только вместе с --webgl")
    if args.include_extraction and args.input_npz is None:
        parser.error("--include-extraction требует --input-npz с экспортированным raw_scan.")

    memory = _memory_snapshot()
    available_gib = memory.get("system_ram_available_gib")
    if available_gib is not None and available_gib < 0.75:
        print("ОТМЕНА: доступно менее 0.75 GiB RAM; измерение не запускается.", file=sys.stderr)
        return 2
    include_extraction = bool(args.include_extraction)
    extraction_skip_reason = None
    if include_extraction and available_gib is not None and available_gib < 1.5:
        include_extraction = False
        extraction_skip_reason = (
            f"Extraction benchmark безопасно пропущен: доступно {available_gib:.2f} GiB RAM, "
            "нужно не менее 1.5 GiB до загрузки raw_scan."
        )
        print(extraction_skip_reason, file=sys.stderr, flush=True)

    rv_input = None
    input_info = None
    overlay_input = None
    extraction_input = None
    label = "синтетический объём"
    try:
        if args.input_npz is not None:
            rv_input, input_info, overlay_input, extraction_input = _load_input_npz(
                args.input_npz, include_extraction=include_extraction
            )
            if extraction_skip_reason:
                input_info["extraction_skip_reason"] = extraction_skip_reason
            label = "реальный экспорт ROIVolume"
    except Exception as exc:
        print(f"INPUT NPZ: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    print("ROI 3D PERFORMANCE BENCHMARK", flush=True)
    print(f"Вход: {label}; тема: {args.theme}; CPU-повторы: {args.repeats}; Qt/WebGL: {'да' if args.webgl else 'нет'}", flush=True)
    print("Используются сохранённые настройки приложения: качество, volume_stride, surface_count, opacity, контраст и палитра.", flush=True)
    print("Полная XY-проекция: " + ("доступна; CPU сравнивает OFF и ON" if overlay_input is not None else "не экспортирована; измеряется только ROI-проекция"), flush=True)
    if rv_input is None:
        print(f"Внимание: synthetic target points={args.points:,}; это только smoke-тест, не оценка реального ROI.", flush=True)

    started = time.perf_counter()
    cpu_memory_sampler = _PeakMemorySampler().start()
    try:
        report, payload = _build_cpu_report(
            args.points, args.repeats, rv_input=rv_input, input_info=input_info,
            overlay_input=overlay_input, extraction_input=extraction_input,
            theme=args.theme, webgl_overlay=args.webgl_overlay,
        )
    except Exception as exc:
        cpu_peak = cpu_memory_sampler.stop()
        print(f"CPU PIPELINE: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("CPU memory sampling:", json.dumps(cpu_peak, ensure_ascii=False), file=sys.stderr)
        return 1
    report["memory"]["cpu_peak_sampling"] = cpu_memory_sampler.stop()
    report["total_cpu_benchmark_seconds"] = round(time.perf_counter() - started, 3)

    print("\nDATA / RENDER POLICY")
    print(f"  Shape: {report['data'].get('shape', report['data'].get('synthetic_shape'))}; voxels: {report['data'].get('voxels', report['data'].get('synthetic_voxels')):,}; array: {report['data'].get('array_mib', report['data'].get('synthetic_array_mib'))} MiB")
    print(f"  Quality: {report['render_policy']['quality']}; requested/effective stride: {report['render_policy']['requested_volume_stride']}/{report['render_policy']['effective_volume_stride']}; cap: {report['render_policy']['max_render_points']:,}; surface_count: {report['render_policy']['surface_count']}")
    print(f"  Plotly: {report['plotly_version']}; supported by ROI widget: {report['compatibility']['plotly_supported_by_roi_widget']}")
    extraction_report = report.get("extraction_pipeline", {})
    if extraction_report.get("status") == "ok":
        print(f"  crop_raw_to_roi: median {extraction_report['crop_raw_to_roi']['median_ms']:.1f} ms")
        print(f"  extract_roi_volume: median {extraction_report['extract_roi_volume']['median_ms']:.1f} ms; output {extraction_report['extracted_volume_shape']}")
        print(f"  extraction input stride: {extraction_report['extraction_input_stride']}; raw crop: {extraction_report['cropped_raw_shape']}")
    elif extraction_report.get("status") == "skipped_low_memory":
        print(f"  ROI extraction benchmark skipped: {extraction_report.get('reason')}")
    print(f"  prepare_volume: median {report['pipeline']['prepare_volume']['median_ms']:.1f} ms")
    print(f"  Prepared lattice: {report['pipeline']['prepared_grid_points']:,} points")
    for scenario, modes in report["pipeline"]["scenarios"].items():
        print(f"  Scenario: {scenario}")
        for mode, item in modes.items():
            opacity_points = item.get("opacityscale_control_points")
            opacity_description = "uniform" if opacity_points is None else f"{opacity_points} transfer points"
            print(f"    {mode}: figure {item['build_volume_figure_including_prepare']['median_ms']:.1f} ms; JSON {item['json_serialization']['median_ms']:.1f} ms; payload {item['json_payload_mib']:.2f} MiB; volume points {item['volume_trace_points']:,}; projection points {item['projection_grid_points']:,}; opacityscale {opacity_description}")
    memory_cpu = report["memory"].get("after_figures_and_json", {})
    if "process_rss_mib" in memory_cpu:
        print(f"  Python RSS after CPU pipeline: {memory_cpu['process_rss_mib']:.1f} MiB")
    if "system_ram_available_gib" in memory_cpu:
        print(f"  Available system RAM: {memory_cpu['system_ram_available_gib']:.2f} GiB")

    if args.webgl:
        if not report["compatibility"]["plotly_supported_by_roi_widget"]:
            webgl_result = {"status": "skipped", "error": "Plotly не соответствует версии, требуемой виджетом."}
        else:
            print("\nЗапуск Qt WebEngine/WebGL; будет открыто диагностическое окно…", flush=True)
            webgl_result = _run_webgl_probe(payload)
        report["webgl"] = webgl_result
        report["memory"]["after_webgl"] = _memory_snapshot()
        if webgl_result.get("status") == "ok":
            print("QT WEBENGINE / WEBGL")
            for key, title in (("initial_render_ms", "Initial Plotly.react render"), ("repeat_rebuild_ms", "Repeated Plotly.react rebuild"), ("contrast_restyle_ms", "Contrast restyle"), ("volume_palette_restyle_ms", "Palette restyle"), ("opacity_mode_restyle_ms", "Opacity mode restyle"), ("resize_ms", "Resize"), ("camera_relayout_ms", "Camera relayout")):
                value = webgl_result.get(key)
                if isinstance(value, dict): value = value.get("median")
                if value is not None: print(f"  {title}: {float(value):.1f} ms")
            frames = webgl_result.get("animation_frame_ms", {})
            print(f"  Frame interval median/p95/max: {frames.get('median', 0):.1f}/{frames.get('p95', 0):.1f}/{frames.get('max', 0):.1f} ms; >50 ms: {frames.get('over_50ms', 0)}")
            gl = webgl_result.get("webgl", {})
            print(f"  WebGL: {gl.get('version')}; renderer: {gl.get('renderer')}; vendor: {gl.get('vendor')}")
            print(f"  2D projection palette invariant: {webgl_result.get('projection_palette_preserved')}")
        else:
            print(f"QT WEBENGINE / WEBGL: {webgl_result.get('status')} — {webgl_result.get('error', 'unknown error')}")

    verdict, notes = _classify(report)
    report["verdict"] = verdict
    report["verdict_notes"] = notes
    print(f"\nИТОГ: {verdict}")
    for note in notes: print(f"  - {note}")
    print("Это измерение конкретной конфигурации, но не стресс-тест длительной сессии и не гарантия для всех драйверов/VRAM.")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Полный JSON-отчёт: {args.json_out.resolve()}")
    return 0 if verdict == "PASS" else (2 if verdict.startswith("НЕ ПРОВЕРЕНО") else 1)


if __name__ == "__main__":
    raise SystemExit(main())
