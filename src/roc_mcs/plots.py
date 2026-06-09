from pathlib import Path
import matplotlib.pyplot as plt


def plot_roc_map(roc_map):
    time_s = roc_map["time_s"]         # X
    theta = roc_map["theta_axis"]      # Y
    Z = roc_map["intensity"].T         # Z     shape: (s, time)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[time_s[0], time_s[-1], theta.min(), theta.max()]  # x_min, x_max, y_min, y_max
    )

    fig.colorbar(im, ax=ax, label="Counts")

    for ti in time_s:
        ax.axvline(ti, color="white", linestyle=":", linewidth=0.4, alpha=0.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (arcsec)" if "theta_axis" in roc_map else r"$\sin(\omega t + \varphi)$")
    ax.set_title("Rocking curve dynamics")

    plt.tight_layout()
    return fig


def save_figure(fig, folder, filename="roc.png", dpi=300):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    outpath = folder / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    print(f"Figure saved: {outpath.resolve()}")
    return outpath