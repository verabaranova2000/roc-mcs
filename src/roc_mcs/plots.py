import matplotlib.pyplot as plt
import math
import numpy as np

from roc_mcs.utils import resolve_output_folder
from roc_mcs.processing.alignment import extract_branch
from roc_mcs.fitting.analysis import MODEL_ANALYSIS

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


# def style_line_axes(ax):    # Ломает всё! или нет?...
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.tick_params(direction="out", length=3, width=0.8)
#     ax.grid(True, alpha=0.25, linewidth=0.6)

def style_map_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(direction="out", length=3, width=0.8)  


def plot_roc_map(roc_map):
    time_s = roc_map["time_s"]         # X
    # theta = roc_map["theta_axis"]      # Y
    Z = roc_map["intensity"].T         # Z     shape: (s, time)
    if "theta_axis" in roc_map:        # Y
        y = roc_map["theta_axis"]
        ylabel = "Angle (arcsec)"
    else:
        y = roc_map["s_axis"]
        ylabel = r"$\sin(\omega t + \varphi)$"

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[time_s[0], time_s[-1], y.min(), y.max()]  # x_min, x_max, y_min, y_max
    )

    fig.colorbar(im, ax=ax, label="Counts")

    for ti in time_s:
        ax.axvline(ti, color="white", linestyle=":", linewidth=0.4, alpha=0.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title("Rocking curve dynamics")
    style_map_axes(ax)
    plt.tight_layout()
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



def save_figure(fig, output_folder=None, filename="roc.png", dpi=300):
    output_folder = resolve_output_folder(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    outpath = output_folder / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return outpath