import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


from roc_mcs.fitting.backends.lmfit_backend import ACADEMIC_PEAK_PALETTE


# Палитра из 10 контрастных, высокополиграфических цветов для компонентов (Colorblind-friendly)
ACADEMIC_PEAK_PALETTE = [
    "#2CA02C",  # Зеленый
    "#9467BD",  # Фиолетовый
    "#17BECF",  # Голубой / Циан
    "#E377C2",  # Розовый
    "#8C564B",  # Коричневый
    "#BCBD22",  # Оливковый
    "#FF7F00",  # Оранжевый
    "#33A02C",  # Темно-зеленый
    "#6A3D9A",  # Темно-фиолетовый
    "#B15928",  # Терракотовый
]


def get_jax_pixel_plot_data(jax_results, data_yxt, theta, peak_catalog, spec, *, x, y, prepared_fit, valid_mask=None):
    """ JAX → данные для графика с учетом маски спайков """
    x, y = int(x), int(y)
    theta = np.asarray(theta, dtype=float)
    intensity = np.asarray(data_yxt[y, x, :], dtype=float)
    valid = prepared_fit.theta_valid & np.isfinite(intensity)   # базовая маска
    if valid_mask is not None:                                  # интеграция маски спайков
        mask_slice = np.asarray(valid_mask[y, x], dtype=bool)
        valid = valid & mask_slice                              # если mask_slice скаляр (2D маска), broadcasting NumPy сам всё сделает правильно
    for batch_result in jax_results:                            # поиск результатов для конкретного пикселя
        mask = (np.asarray(batch_result["x_pixel"]) == x) & (np.asarray(batch_result["y_pixel"]) == y)
        if np.any(mask):
            i = int(np.flatnonzero(mask)[0])
            K = int(batch_result["K"])
            params = np.asarray(batch_result["x"][i], dtype=float)
            break
    else:
        raise ValueError(f"JAX result not found for pixel ({x}, {y})")

    n_params = len(spec.param_names)
    active = params[:K * n_params].reshape(K, n_params)
    components = {}
    rows = []

    pixel_peaks = peak_catalog[(peak_catalog["x"] == x) & (peak_catalog["y"] == y)].sort_values("theta").reset_index(drop=True)
    if len(pixel_peaks) != K:
        raise ValueError(f"Peak catalog has {len(pixel_peaks)} peaks at ({x}, {y}), JAX result has K={K}")

    for k in range(K):
        p = active[k]
        components[f"peak_{k}_"] = np.asarray(spec.jax_func(theta, *p), dtype=float)
        peak = pixel_peaks.iloc[k]
        rows.append({
            "peak_id": int(peak["peak_id"]),
            "component_id": k,
            "theta_detected": float(peak["theta"]),         # В каталоге начальное положение лежит в колонке "theta"
            "theta": float(active[k, spec.param_names.index("theta0")]),
        })
    best_fit = np.sum(list(components.values()), axis=0)    # суммарный фит на всей оси theta
    if params.size > K * n_params:
        best_fit += params[-1]                              # добавляем фон
    return {
        "theta": theta,
        "intensity": intensity,
        "valid": valid,
        "best_fit": best_fit,
        "components": components,
        "fitted_peaks": pd.DataFrame(rows),
    }



def get_lmfit_pixel_plot_data(pixel_result, data_yxt, theta, spec, *, prepared_fit, valid_mask=None):
    """
    LMFIT → данные для графика с учетом маски спайков.
    Архитектура 1-в-1 с JAX: отвязана от объектов lmfit.
    """
    x, y = int(pixel_result["x"]), int(pixel_result["y"])
    theta = np.asarray(theta, dtype=float)
    intensity = np.asarray(data_yxt[y, x, :], dtype=float)
    valid = prepared_fit.theta_valid & np.isfinite(intensity)
    if valid_mask is not None:
        mask_slice = np.asarray(valid_mask[y, x], dtype=bool)
        if mask_slice.ndim > 1 or (mask_slice.ndim == 1 and mask_slice.shape != valid.shape):
            raise ValueError(f"Invalid mask shape at pixel ({x}, {y}): {mask_slice.shape}")
        valid &= mask_slice

    fitted_peaks = pixel_result["fitted_peaks"]
    components, rows = {}, []
    for _, row in fitted_peaks.iterrows():
        k = int(row["component_id"])
        params = [row[name] for name in spec.param_names]
        components[f"peak_{k}_"] = np.asarray(spec.jax_func(theta, *params), dtype=float)
        rows.append({
            "peak_id": int(row["peak_id"]),
            "component_id": k,
            "theta_detected": float(row["theta_detected"]),
            "theta": float(row["theta"]),
        })

    best_fit = np.sum(list(components.values()), axis=0) if components else np.zeros_like(theta)
    params = pixel_result["result"].params
    if "bg_background" in params:
        best_fit += float(params["bg_background"].value)

    return {
        "theta": theta,
        "intensity": intensity,
        "valid": valid,
        "best_fit": best_fit,
        "components": components,
        "fitted_peaks": pd.DataFrame(rows),
    }
    


def plot_pixel_fit(
    plot_data, *, x, y, normalize=True, show_diagnostics=False,
    figsize=(5.5, 4.0), legend_loc="upper left", palette=ACADEMIC_PEAK_PALETTE,
):
    """Строит single-pixel fit."""
    theta = np.asarray(plot_data["theta"], dtype=float)
    intensity = np.asarray(plot_data["intensity"], dtype=float)
    valid = np.asarray(plot_data["valid"], dtype=bool)
    best_fit = np.asarray(plot_data["best_fit"], dtype=float)
    components = plot_data["components"]
    fitted_peaks = plot_data["fitted_peaks"]

    norm_factor = np.max(intensity) if normalize else 1.0
    intensity = intensity / norm_factor
    best_fit = best_fit / norm_factor

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    if isinstance(palette, str) or palette is None:
        cmap = plt.get_cmap(palette or "tab10")
        get_color = lambda i: cmap(i % cmap.N)
    else:
        get_color = lambda i: palette[i % len(palette)]

    # --- Эксперимент без выбросов ---
    invalid = ~valid
    intensity_valid_line = intensity.copy()
    intensity_valid_line[invalid] = np.nan 
    ax.plot(theta, intensity_valid_line, color="black", linewidth=1.2, label="Experimental", zorder=3)
    # --- Выбросы (спайки) ---
    if np.any(invalid):
        ax.scatter(theta[invalid], intensity[invalid], facecolor="none", edgecolor="red", marker="o", s=40, linewidth=1.5, label="Masked Outliers (Spikes)", zorder=4)
    ax.plot(theta, best_fit, color="crimson", linewidth=1.2, label="Total fit", zorder=3)

    for i, (_, row) in enumerate(fitted_peaks.reset_index(drop=True).iterrows()):
        color = get_color(i)
        component_name = f"peak_{int(row['component_id'])}_"
        if component_name not in components:
            continue
        comp_y = components[component_name] / norm_factor
        ax.fill_between(theta, 0, comp_y, facecolor=color, edgecolor=color, alpha=0.35, linewidth=0.8, label=f"Peak {int(row['peak_id'])}")
        ax.plot(theta, comp_y, color=color, linestyle="--", linewidth=0.9, alpha=0.85)
        if show_diagnostics:
            ax.axvline(row["theta_detected"], color=color, linestyle=":", linewidth=1.0, alpha=0.7)
            ax.axvline(row["theta"], color=color, linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel(r"$\theta$ (arcsec)", fontsize=11)
    ax.set_ylabel(r"$I/I_{\max}$ (rel. u.)" if normalize else "Intensity (counts)", fontsize=11)
    ax.set_title(f"Pixel ({x}, {y}) Profile Fit", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlim(theta.min(), theta.max())
    ax.set_ylim(bottom=0)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc=legend_loc, frameon=False, fontsize=9.5, handletextpad=0.5, columnspacing=1.0)
    return fig, ax