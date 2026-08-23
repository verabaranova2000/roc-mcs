import matplotlib.pyplot as plt
import math
import numpy as np

from roc_mcs.utils import resolve_output_folder
from roc_mcs.processing.alignment import extract_branch
from roc_mcs.fitting.analysis import MODEL_ANALYSIS
from roc_mcs.fitting.trajectory.types import ScalarTrajectory

PLOT_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

plt.rcParams.update(PLOT_STYLE)


def style_line_axes(ax):   
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(True, alpha=0.25, linewidth=0.6)

def style_map_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(direction="out", length=3, width=0.8)  

def plot_roc_map(roc_map, secondary_curves=None):    
    time_s = roc_map["time_s"]         # X
    # theta = roc_map["theta_axis"]    # Y
    Z = roc_map["intensity"].T         # Z     shape: (s, time)
    
    if "theta_axis" in roc_map:        # Y
        y = roc_map["theta_axis"]
        ylabel = "Angle (arcsec)"
    else:
        y = roc_map["s_axis"]
        ylabel = r"$\sin(\omega t + \varphi)$"

    ax_load = None
    if secondary_curves is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, (ax, ax_load) = plt.subplots(2, 1, figsize=(10, 6),
                                          sharex=True, height_ratios=[4, 1],
                                          constrained_layout=True)    
        
    # показываем границы между соседними рок-кривыми, т.е. рисуем не центры, а полуинтервалы
    dt = np.diff(time_s)
    edges = np.empty(len(time_s) + 1)
    edges[1:-1] = 0.5 * (time_s[:-1] + time_s[1:])
    edges[0] = time_s[0] - 0.5 * dt[0]
    edges[-1] = time_s[-1] + 0.5 * dt[-1]   
  
    # im = ax.imshow(
    #     Z,
    #     aspect="auto",
    #     origin="lower",
    #     cmap="viridis",
    #     #extent=[time_s[0], time_s[-1], y.min(), y.max()]  # x_min, x_max, y_min, y_max
    #     extent=[
    #         edges[0],     # time_s[0] - half_step,
    #         edges[-1],    # time_s[-1] + half_step,
    #         y.min(),
    #         y.max(),
    #     ],    
    # )
    # --- ИСПРАВЛЕНИЕ: Заменяем imshow на pcolormesh для корректной отрисовки неравномерной сетки y
    im = ax.pcolormesh(
        time_s, 
        y, 
        Z, 
        cmap="viridis",
        shading="auto",
        rasterized=True  # Ускоряет рендер и предотвращает артефакты сетки в векторных форматах
    )
    # -------------------------
    fig.colorbar(im, ax=ax, pad=0.01, label="Counts")
    
    for t_edge in edges:   #time_s:
        ax.axvline(t_edge, color="white", linestyle=":", linewidth=0.4, alpha=0.4)
        if ax_load is not None:
            ax_load.axvline(t_edge,color="0.7", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title("Rocking curve dynamics")
    style_map_axes(ax)

    if ax_load is not None:
        for label, curve in secondary_curves.items():
            ax_load.plot(curve["x"], curve["y"], lw=1.2, alpha=0.8, label=label)
        # ax_load.set_ylabel("Load")
        ax_load.legend(loc="upper left", frameon=False)
        style_line_axes(ax_load)
        ax_load.set_xlabel("Time (s)")
        ax.margins(x=0)
        ax_load.margins(x=0)        
    else:
        ax.set_xlabel("Time (s)")
    return fig



def plot_phase_diagnostics(qc):
    counts = qc["counts"]
    n_channels = qc["n_channels"]
    best_phi = qc["best_phi"]
    scores = qc["scores"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.2), constrained_layout=True)
    
    # -------------------------
    # (a) Phase criterion
    # -------------------------
    ax[0].plot(scores, lw=1.2, color="#4c72b0")
    ax[0].axvline(best_phi, color="crimson", lw=1.2)
    
    ax[0].set_title( r"Зависимость дисперсии остатка от фазового сдвига")
    ax[0].set_xlabel(r"Фазовый сдвиг, $\varphi$")
    ax[0].set_ylabel(r"Дисперсия остатка, $\mathcal{C}(\varphi)=\mathrm{Var}(I-I_{\mathrm{smooth}})$")
    
    ymin, ymax = ax[0].get_ylim()
    ax[0].annotate(
        rf"$\varphi^*={best_phi:.0f}$",
        xy=(best_phi, ymax), xytext=(5, 0),
        textcoords="offset points",
        fontsize=10, color="red", ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="0.8", linewidth=0.5, alpha=0.8, pad=1.5),  # edgecolor="none"
    )
    
    ax[0].text(0.02, 0.95, "(a)", transform=ax[0].transAxes, fontweight="bold")
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].grid(True, alpha=0.25, linewidth=0.6)
    
    # -------------------------
    # (b) Phase-aligned intensity
    # -------------------------
    s_up, y_up = extract_branch(counts, best_phi, n_channels, branch="up")
    s_dn, y_dn = extract_branch(counts, best_phi, n_channels, branch="down")
    
    ax[1].plot(s_dn, y_dn, ".", ms=2, label=r"$\cos(\varphi) < 0$", color="#2ca02c")   
    ax[1].plot(s_up, y_up, ".", ms=3, label=r"$\cos(\varphi) > 0$", color="#6a3d9a")
    
    ax[1].set_title(r"Интенсивность после фазового сдвига (ветви $\cos(\varphi)>0$ и $\cos(\varphi)<0$)")
    ax[1].set_xlabel(r"$\sin(\varphi)$")
    ax[1].set_ylabel("Интенсивность, counts")
    
    ax[1].text(0.02, 0.95, "(b)", transform=ax[1].transAxes, fontweight="bold")
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].grid(True, alpha=0.25, linewidth=0.6)
    ax[1].legend(frameon=False)
    return fig


COLORS = {
    "nature": ("#3C5488", "#B24745"),      # сине-фиолетовый + приглушённый красный
    "classic": ("#1f3b73", "#8b1e3f"),     # тёмно-синий + бордовый
    "green": ("#1b5e20", "#8e244d"),       # тёмно-зелёный + бордовый
    "okabe": ("#0072B2", "#D55E00"),       # синий + киноварь
    "bw": ("black", "#a11d33"),            # чёрный + тёмно-красный
}

COLOR_LEFT, COLOR_RIGHT = COLORS["nature"]


def plot_model_evolution(df_fit, x="time", max_cols=2, 
                         figsize_per_panel=(5.4, 3.6), 
                         y_dual=False):  # y_mode="shared" | "dual"
                        
    """
    Графики эволюции параметров модели.

    Parameters
    ----------
    df_fit : pandas.DataFrame
        Таблица результатов подгонки.
    x : str, default="time"
        Название столбца, используемого по оси X.
    max_cols : int, default=2
        Максимальное число панелей в одном ряду.
    figsize_per_panel : tuple, default=(5.4, 3.6)
        Размер одной панели (ширина, высота).
    y_dual : bool, default=False
        Если True, для панелей с двумя кривыми используется
        дополнительная правая ось Y с независимым масштабом.
        Если False, все кривые отображаются на общей оси Y.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Объект Figure с построенными графиками.
    """
    if df_fit.empty:
        raise ValueError("df_fit is empty")

    model_name = df_fit["model"].iloc[0]
    spec = MODEL_ANALYSIS.get(model_name, {})
    groups = spec.get("plot_groups", [])

    if not groups:
        raise ValueError(f"No plot_groups defined for model '{model_name}'")

    n_panels = len(groups)
    ncols = min(max_cols, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                             constrained_layout=True)

    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for ax, group in zip(axes, groups):
        plotted_any = False
        available = [(key, label) for key, label in group if key in df_fit.columns]
        if not available:
            ax.set_visible(False)
            continue
    
        # Если ровно 2 кривые — делаем две оси Y
        if y_dual and len(available) == 2:    # if y_mode == "dual" and len(available) == 2:
            (key1, label1), (key2, label2) = available
            ax2 = ax.twinx()
            l1 = ax.plot(df_fit[x], df_fit[key1], marker="o", 
                         lw=1.2, ms=3.5, color=COLOR_LEFT, label=label1)
            l2 = ax2.plot(df_fit[x], df_fit[key2], marker="s", 
                          lw=1.2, ms=3.5, color=COLOR_RIGHT, label=label2)
    
            ax.set_title(f"{label1} / {label2}")
            ax.set_xlabel(x)
            ax.set_ylabel(label1) #, color=COLOR_LEFT)
            ax.tick_params(axis="y", labelcolor=COLOR_LEFT)
            ax2.set_ylabel(label2) #,  color=COLOR_RIGHT)
            ax2.tick_params(axis="y", labelcolor=COLOR_RIGHT)
    
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            # общий legend для обеих осей
            lines = l1 + l2
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, frameon=False, loc="best")
    
        else:
            for key, label in available:
                ax.plot(df_fit[x], df_fit[key], marker="o", lw=1.2, ms=3.5, 
                        color=COLOR_LEFT, label=label)
                plotted_any = True
    
            panel_title = ", ".join(label for _, label in available)
            ax.set_title(panel_title)
            ax.set_xlabel(x)
            ax.set_ylabel(available[0][1] if len(available) == 1 else panel_title) #, color=COLOR_LEFT)
            ax.tick_params(axis="y", labelcolor=COLOR_LEFT)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if len(available) > 1:
                ax.legend(frameon=False)
        plotted_any = True

    for ax in axes[n_panels:]:      # скрыть лишние пустые оси
        ax.set_visible(False)
    return fig



def plot_residual_maps(
    time,
    theta,
    diff_maps,
    metrics=None,
    title="Residual maps",   # title="Разностные карты",
    max_cols=2,
):
    """
    Грид разностных ROC-карт.
    Максимум max_cols графиков в строке.
    """
    models = list(diff_maps.keys())
    n = len(models)

    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    # Общая цветовая шкала
    all_vals = np.concatenate([diff_maps[m].ravel() for m in models])
    vmax = np.nanmax(np.abs(all_vals))
    vmin = -vmax
    
    labels = "abcdefghijklmnopqrstuvwxyz"

    for i, (ax, model) in enumerate(zip(axes, models)):
        im = ax.imshow(diff_maps[model].T,
                       aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax,
                       extent=[time[0], time[-1], theta.min(), theta.max()])
        if metrics is not None and model in metrics:
            metric_dict = metrics[model]
            txt = []
            for key in ("RMSE", "MAE", "BIAS"):
                if key in metric_dict:
                    txt.append(f"{key} = {metric_dict[key]:.4f}")
            ax.text(
                0.02, 0.98,
                "\n".join(txt),
                transform=ax.transAxes,
                va="top", ha="left", multialignment="left", fontsize=8, 
                bbox=dict(facecolor="white", edgecolor="0.8", linewidth=0.5, alpha=0.9, pad=1.5)
            )       
        ax.set_title(f"({labels[i]}) {model}", fontsize=11, fontweight="normal")
        ax.set_xlabel("Time (s)")
        style_map_axes(ax)

    for ax in axes[n:]:         # скрыть пустые оси
        ax.set_visible(False)
    for row in range(nrows):    # подписи только по левому краю
        axes[row * ncols].set_ylabel(r"$\theta$")      

    cbar = fig.colorbar(im, ax=axes[:n], fraction=0.03, pad=0.02)
    cbar.set_label("Residual intensity (Exp − Calc)")     # cbar.set_label("Разность интенсивностей (эксп. − расч.)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig





def plot_scalar_trajectory(
    ax_main,
    traj: ScalarTrajectory,
    time,
    ylabel=None,
    show_uncertainty=True,
):
    if ylabel is None:
        ylabel = traj.name
    t = np.asarray(time)

    # ---- Local fit ---    
    local = np.asarray(traj.local)
    if local.shape[0] != t.shape[0]:
        raise ValueError(f"Длина time ({t.shape[0]}) не совпадает с длиной local ({local.shape[0]}) для траектории '{traj.name}'")      
    ax_main.errorbar(t, local, 
                     yerr=traj.sigma_fit if traj.sigma_fit is not None else None,
                     fmt=".:", alpha=0.9, capsize=2,
                     linewidth=2.2, label="local fit", zorder=3)
    # ---- Ridge ---   
    if traj.ridge is not None:
        ridge = np.asarray(traj.ridge)
        if ridge.shape[0] != t.shape[0]:
            raise ValueError(f"Длина time ({t.shape[0]}) не совпадает с длиной ridge ({ridge.shape[0]}) для траектории '{traj.name}'")        
        ax_main.errorbar(t, ridge, 
                         yerr=traj.sigma_ridge if traj.sigma_ridge is not None else None,
                         fmt="o--", alpha=0.9, capsize=3,
                         linewidth=1.5, label="ridge observation", zorder=4)
    # ---- Smooth ---
    if traj.smooth is not None:
        smooth = np.asarray(traj.smooth)
        if smooth.shape[0] != t.shape[0]:
            raise ValueError(f"Длина time ({t.shape[0]}) не совпадает с длиной smooth ({smooth.shape[0]}) для траектории '{traj.name}'")   
        ax_main.plot(t, smooth, "-", color="black", lw=2.5,
                     label="Kalman-RTS", zorder=5)
        if show_uncertainty and traj.sigma_smooth is not None:
            sigma_smooth = np.asarray(traj.sigma_smooth)
            if sigma_smooth.shape[0] != t.shape[0]:
                raise ValueError(f"Длина time ({t.shape[0]}) не совпадает с длиной sigma_smooth ({sigma_smooth.shape[0]}) "
                                 f"для траектории '{traj.name}'")            
            ax_main.fill_between(t, smooth - 2 * sigma_smooth, smooth + 2 * sigma_smooth,
                                 color="gray", alpha=0.2, label="Kalman ±2σ", zorder=0)
    ax_main.set_ylabel(ylabel)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()


def plot_trajectories(time,                     # временная ось
                      trajectories,             # список ScalarTrajectory
                      secondary_curves=None,     # сила, давление и прочие внешние кривые
                      model_name=None,
):
    # если передали dict[name -> ScalarTrajectory], берём значения
    if isinstance(trajectories, dict):
        trajectories = list(trajectories.values())
    
    n = len(trajectories)
    t = np.asarray(time)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.5 * n), sharex=True, constrained_layout=True,)
    if n == 1:
        axes = [axes]
    for ax, traj in zip(axes, trajectories):
        plot_scalar_trajectory(ax, traj, t, ylabel=traj.name)
        ax.legend(loc="upper left")
        if secondary_curves is not None:
            ax2 = ax.twinx()   
            for label, curve in secondary_curves.items():
                ax2.plot(curve["x"],  curve["y"], "--", linewidth=1.2, alpha=0.7, color=curve["color"], label=label)
            ax2.legend(loc="upper right")
    axes[-1].set_xlabel("time")
    if model_name is not None:
        fig.suptitle(model_name)
    return fig




# =============== Моменты ==================

PANEL_SPECS = {
    "intensity": {
        "title": "Нулевой момент (интегральная интенсивность)",
        "formula": r"$A=\int I(\theta)\,d\theta$",
        "meaning": "Суммарная отражённая интенсивность; чувствительна к перераспределению профиля.",
    },
    "position": {
        "title": "Первый момент (центр масс профиля)",
        "formula": r"$\theta_{\max},\ \mu=\frac{\int \theta I(\theta)\,d\theta}{\int I(\theta)\,d\theta}$",
        "meaning": r"Сдвиг профиля и развитие асимметрии; $\mu-\theta_{\max}$ показывает появление плеча.",
    },
    "width": {
        "title": "Второй центральный момент и интегральные ширины",
        "formula": r"$\sigma^2=\frac{\int(\theta-\mu)^2 I(\theta)\,d\theta}{\int I(\theta)\,d\theta},\ FW_{50/80/90}$",
        "meaning": "Уширение, расплывание и начало расщепления профиля.",
    },
    "shape": {
        "title": "Третий и четвёртый стандартизованные моменты",
        "formula": r"$\gamma_1,\ \gamma_2-3$",
        "meaning": "Асимметрия, плечо, уплощение и многомодальность.",
    },
}
 


def add_panel_caption(ax, spec):
    # основной заголовок слева
    ax.set_title( spec["title"], loc="left", fontsize=11, pad=14)
    # формула справа
    ax.set_title(spec["formula"], loc="right", fontsize=10, pad=14)
    # краткое физическое значение под заголовком
    ax.text(0.0, 1.01, spec["meaning"], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="0.35")    


def plot_roc_moment_evolution(
    df,
    x_col="pressure_MPa",
    x_label="Pressure, MPa",
    normalize=True,
    title_prefix="ROC moment evolution",
    show_shape_stats=False,
):
    """
    Строит компактную эволюцию моментных характеристик рок-кривых.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица moment_table, дополненная столбцами force/pressure.
    x_col : str
        Столбец оси X (например, 'pressure_MPa' или 'force_N').
    x_label : str
        Подпись оси X.
    normalize : bool
        Если True, нормирует area, peak_intensity, sigma и FWxx к первому скану.
    title_prefix : str
        Префикс заголовков.
    show_shape_stats : bool
        Если True, добавляет skewness/kurtosis в отдельную нижнюю панель.

    Returns
    -------
    matplotlib.figure.Figure
    """

    df = df.copy().sort_values(x_col).reset_index(drop=True)
    x = df[x_col].to_numpy(dtype=float)

    area = df["area"].to_numpy(dtype=float)
    peak_intensity = df["peak_intensity"].to_numpy(dtype=float)

    peak_theta = df["peak_theta"].to_numpy(dtype=float)
    centroid = df["centroid"].to_numpy(dtype=float)
    centroid_shift = df["centroid_shift"].to_numpy(dtype=float)

    sigma = df["sigma"].to_numpy(dtype=float)
    fw50 = df["fw50_int"].to_numpy(dtype=float)
    fw80 = df["fw80_int"].to_numpy(dtype=float)
    fw90 = df["fw90_int"].to_numpy(dtype=float)

    skewness = df["skewness"].to_numpy(dtype=float)
    kurtosis = df["kurtosis"].to_numpy(dtype=float)

    def _norm(y):
        if not normalize:
            return y
        y0 = y[0]
        if not np.isfinite(y0) or y0 == 0:
            return y
        return y / y0

    area_n = _norm(area)
    peak_intensity_n = _norm(peak_intensity)

    sigma_n = _norm(sigma)
    fw50_n = _norm(fw50)
    fw80_n = _norm(fw80)
    fw90_n = _norm(fw90)

    nrows = 4 if show_shape_stats else 3
    fig, axes = plt.subplots(nrows=nrows,  ncols=1,  figsize=(8.2, 2.8 * nrows),
                             sharex=True, constrained_layout=True)

    if nrows == 3:
        ax1, ax2, ax3 = axes
        ax4 = None
    else:
        ax1, ax2, ax3, ax4 = axes

    # --- 1) Integral intensity ---
    ax1.plot(x, area_n, marker="o", ms=3, lw=1.4, label=r"$A/A_0$")
    ax1.plot(x, peak_intensity_n, marker="o", ms=3, lw=1.4, alpha=0.4, label=r"$I_{\max}/I_{\max,0}$")
    # ax1.set_ylabel("normalized intensity")
    # ax1.set_title(f"{title_prefix}: intensity")
    #style_line_axes(ax1)
    ax1.set_ylabel(r"$A/A_0$")
    style_line_axes(ax1)
    add_panel_caption(ax1, PANEL_SPECS["intensity"])    
    ax1.legend(frameon=False)

    # --- 2) Peak position / asymmetry shift ---
    ax2.plot(x, centroid, marker="o", ms=3, lw=1.4, label=r"centroid $\mu$")
    ax2.plot(x, peak_theta, marker="o", ms=3, lw=1.4, alpha=0.4, label=r"$\theta_{\max}$")
    ax2.plot(x, centroid_shift, marker="o", ms=3, lw=1.4, alpha=0.4, label=r"$\mu-\theta_{\max}$")
    # ax2.set_ylabel(r"$\theta$")
    # ax2.set_title(f"{title_prefix}: position and asymmetry shift")
    # style_line_axes(ax2)
    ax2.set_ylabel("angle")
    style_line_axes(ax2)
    add_panel_caption(ax2, PANEL_SPECS["position"])    
    ax2.legend(frameon=False)

    # --- 3) Broadening / shape ---
    ax3.plot(x, sigma_n, marker="o", ms=3, lw=1.4, label=r"$\sigma/\sigma_0$")
    ax3.plot(x, fw50_n, marker="o", ms=3, lw=1.4, label=r"FW50/FW50$_0$")
    ax3.plot(x, fw80_n, marker="o", ms=3, lw=1.4, label=r"FW80/FW80$_0$")
    ax3.plot(x, fw90_n, marker="o", ms=3, lw=1.4, label=r"FW90/FW90$_0$")
    # ax3.set_ylabel("normalized width")
    # ax3.set_title(f"{title_prefix}: profile broadening")
    # style_line_axes(ax3)
    ax3.set_ylabel("relative width")
    style_line_axes(ax3)
    add_panel_caption(ax3, PANEL_SPECS["width"])    
    ax3.legend(frameon=False)

    if ax4 is not None:
        color1 = "tab:blue"
        color2 = "tab:orange"
    
        ax4.plot(x, skewness, marker="o", ms=3, lw=1.4, color=color1, label="skewness")
        ax4.set_ylabel("skewness") #, color=color1)
        ax4.tick_params(axis="y") #, labelcolor=color1)
        # ax4.set_title(f"{title_prefix}: asymmetry and flatness")
        add_panel_caption(ax4,  PANEL_SPECS["shape"])
        style_line_axes(ax4)
    
        ax4b = ax4.twinx()
        ax4b.plot(x, kurtosis, marker="s", ms=3, lw=1.4, color=color2, label="excess kurtosis")
        ax4b.set_ylabel("excess kurtosis") #, color=color2)
        ax4b.tick_params(axis="y") #, labelcolor=color2)
    
        # объединённая легенда
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")
    
    axes[-1].set_xlabel(x_label)
    return fig




def save_figure(fig, output_folder=None, filename="roc.png", dpi=300):
    output_folder = resolve_output_folder(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    outpath = output_folder / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return outpath