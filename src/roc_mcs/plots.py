import matplotlib.pyplot as plt
from roc_mcs.utils import resolve_output_folder


from roc_mcs.processing.alignment import extract_branch

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

    plt.tight_layout()
    return fig



def plot_phase_diagnostics(qc):
    counts = qc["counts"]
    n_channels = qc["n_channels"]
    best_phi = qc["best_phi"]
    scores = qc["scores"]
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.2), constrained_layout=True)
    
    # -------------------------
    # (a) Phase criterion
    # -------------------------
    ax[0].plot(scores, lw=1.2, color="#4c72b0")
    ax[0].axvline(best_phi, color="crimson", lw=1.2)
    
    ax[0].set_title( r"Зависимость дисперсии остатка от фазового сдвига")
    ax[0].set_xlabel(r"Фазовый сдвиг, $\varphi$")
    ax[0].set_ylabel(r"Дисперсия остатка $\mathcal{C}(\varphi)=\mathrm{Var}(I-I_{\mathrm{smooth}})$")
    
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



def save_figure(fig, output_folder=None, filename="roc.png", dpi=300):
    output_folder = resolve_output_folder(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    outpath = output_folder / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return outpath