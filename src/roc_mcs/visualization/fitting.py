import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from roc_mcs.fitting.result_normalization import pixel_valid_mask, validate_valid_mask
from roc_mcs.fitting.results import FitScanResult


ACADEMIC_PEAK_PALETTE = ["#2CA02C", "#9467BD", "#17BECF", "#E377C2", "#8C564B", "#BCBD22", "#FF7F00", "#33A02C", "#6A3D9A", "#B15928"]


def get_pixel_plot_data(result: FitScanResult, data_yxt, *, x, y, spec=None, theta=None, valid_mask=None):
    """Готовит кривые для графика из общего FitScanResult."""
    if not isinstance(result, FitScanResult):
        raise TypeError("result должен быть FitScanResult; используйте fit_scan()")
    data_yxt = np.asarray(data_yxt, dtype=float); x, y = int(x), int(y)
    if data_yxt.ndim != 3 or data_yxt.shape[:2] != result.image_shape:
        raise ValueError("data_yxt не согласован с размером FitScanResult")
    if not (0 <= x < result.image_shape[1] and 0 <= y < result.image_shape[0]):
        raise IndexError(f"Пиксель ({x},{y}) вне изображения")
    theta = result.theta if theta is None else np.asarray(theta, dtype=float)
    if theta.shape != result.theta.shape or not np.allclose(theta, result.theta, equal_nan=True):
        raise ValueError("theta должен совпадать с theta, использованным при подгонке")
    if spec is None:
        from roc_mcs.fitting.registry import MODEL_SPECS
        spec = MODEL_SPECS[result.model_id]
    mask = result.valid_mask if valid_mask is None else validate_valid_mask(valid_mask, data_yxt.shape)
    intensity, valid = pixel_valid_mask(data_yxt, result.theta_valid, mask, x=x, y=y)
    rows = result.fitted_peak_catalog
    fitted_peaks = rows.loc[(rows["x"] == x) & (rows["y"] == y)].sort_values("component_id").reset_index(drop=True)
    components = {}
    for _, row in fitted_peaks.iterrows():
        missing = [name for name in result.component_param_names if name not in row.index]
        if missing:
            raise ValueError(f"В fitted_peak_catalog отсутствуют параметры модели: {missing}")
        params = [float(row[name]) for name in result.component_param_names]
        curve = np.asarray(spec.func(theta, *params), dtype=float)
        if curve.shape != theta.shape:
            raise ValueError(f"Модель {result.model_id!r} вернула кривую формы {curve.shape}, ожидалась {theta.shape}")
        components[f"peak_{int(row['component_id'])}_"] = curve
    pixel_rows = result.pixel_fit_catalog
    pixel_row = pixel_rows.loc[(pixel_rows.x == x) & (pixel_rows.y == y)]
    if len(pixel_row) != 1:
        raise ValueError(f"В pixel_fit_catalog нет единственной строки для ({x},{y})")
    pixel_row = pixel_row.iloc[0]
    best_fit = np.full(theta.shape, np.nan, dtype=float)
    if components:
        best_fit = np.sum(np.stack(list(components.values()), axis=0), axis=0)
        if result.pixel_param_names:
            pixel_params = {name: float(pixel_row.get(name, np.nan)) for name in result.pixel_param_names}
            if not all(np.isfinite(value) for value in pixel_params.values()):
                best_fit[:] = np.nan
            elif tuple(result.pixel_param_names) == ("bg_background",):
                best_fit += pixel_params["bg_background"]
            else:
                pixel_func = getattr(spec, "pixel_func", None)
                if pixel_func is None:
                    raise ValueError(f"Для pixel-параметров {result.pixel_param_names} модель {result.model_id!r} должна определить spec.pixel_func(theta, **params)")
                contribution = np.asarray(pixel_func(theta, **pixel_params), dtype=float)
                if contribution.shape != theta.shape:
                    raise ValueError("spec.pixel_func вернула массив неверной формы")
                best_fit += contribution
    return {"theta": theta, "intensity": intensity, "valid": valid, "best_fit": best_fit,
            "components": components, "fitted_peaks": fitted_peaks, "fit_status": str(pixel_row.fit_status)}


def plot_pixel_fit(plot_data, *, x, y, normalize=True, show_diagnostics=False, figsize=(5.5, 4.0),
                   legend_loc="upper left", palette=ACADEMIC_PEAK_PALETTE, ax=None):
    """Строит профиль сигнала, общий фит и отдельные компоненты."""
    theta = np.asarray(plot_data["theta"], dtype=float)
    intensity = np.asarray(plot_data["intensity"], dtype=float)
    valid = np.asarray(plot_data["valid"], dtype=bool)
    best_fit = np.asarray(plot_data["best_fit"], dtype=float)
    components, fitted_peaks = plot_data["components"], plot_data["fitted_peaks"]
    finite_intensity = np.isfinite(intensity)
    scale = float(np.nanmax(np.abs(intensity[finite_intensity]))) if finite_intensity.any() else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    if not normalize:
        scale = 1.0
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
        ax.clear()
    if isinstance(palette, str) or palette is None:
        cmap = plt.get_cmap(palette or "tab10")
        get_color = lambda i: cmap(i % cmap.N)
    else:
        if not palette:
            raise ValueError("palette не может быть пустой")
        get_color = lambda i: palette[i % len(palette)]
    observed = intensity / scale
    observed_line = observed.copy(); observed_line[~(valid & finite_intensity)] = np.nan
    ax.plot(theta, observed_line, color="black", linewidth=1.2, label="Experimental", zorder=3)
    masked = ~valid & finite_intensity & np.isfinite(theta)
    if masked.any():
        ax.scatter(theta[masked], observed[masked], facecolor="none", edgecolor="red", marker="o", s=40,
                   linewidth=1.5, label="Masked points", zorder=4)
    best_line = best_fit / scale
    best_line[~np.isfinite(best_fit)] = np.nan
    if np.isfinite(best_line).any():
        ax.plot(theta, best_line, color="crimson", linewidth=1.2, label="Total fit", zorder=3)
    for i, (_, row) in enumerate(fitted_peaks.reset_index(drop=True).iterrows()):
        color = get_color(i); key = f"peak_{int(row['component_id'])}_"
        if key not in components:
            continue
        curve = np.asarray(components[key], dtype=float) / scale
        ax.fill_between(theta, 0, curve, facecolor=color, edgecolor=color, alpha=0.35, linewidth=0.8,
                        label=f"Peak {int(row['peak_id'])}")
        ax.plot(theta, curve, color=color, linestyle="--", linewidth=0.9, alpha=0.85)
        if show_diagnostics:
            ax.axvline(float(row["theta_detected"]), color=color, linestyle=":", linewidth=1.0, alpha=0.7)
            ax.axvline(float(row["theta"]), color=color, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r"$\theta$ (arcsec)", fontsize=11)
    ax.set_ylabel(r"$I/I_{\max}$ (rel. u.)" if normalize else "Intensity (counts)", fontsize=11)
    ax.set_title(f"Pixel ({int(x)}, {int(y)}) Profile Fit — {plot_data.get('fit_status', '')}", loc="left", fontsize=11, fontweight="bold")
    finite_theta = theta[np.isfinite(theta)]
    if finite_theta.size:
        ax.set_xlim(finite_theta.min(), finite_theta.max())
    ax.set_ylim(bottom=0)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, linestyle=":", alpha=0.5)
    if ax.has_data():
        ax.legend(loc=legend_loc, frameon=False, fontsize=9.5, handletextpad=0.5, columnspacing=1.0)
    return fig, ax
