import numpy as np
import matplotlib.pyplot as plt


def _component_curves(X, K, theta, spec):
    q = len(spec.param_names)
    params = np.asarray(X)[:K * q].reshape(K, q)
    return [np.asarray(spec.func(theta, *p)) for p in params]


def _plot_candidate_row(ax_fit, ax_res, r, data_yxt, theta, spec, evaluate_composite, valid_yxt=None):
    X0, X1 = np.asarray(r.M0_x), np.asarray(r.M1_x)
    y, x = int(r.y_pixel), int(r.x_pixel)
    obs = np.asarray(data_yxt[y, x], dtype=float)

    m0 = evaluate_composite(spec, theta, X0[None, :], int(r.K0))[0]
    m1 = evaluate_composite(spec, theta, X1[None, :], int(r.K1))[0]
    comps = _component_curves(X1, int(r.K1), theta, spec)
    c1, c2 = comps[int(r.parent_index)], comps[-1]

    valid = np.isfinite(obs)
    if valid_yxt is not None:
        valid &= valid_yxt[y, x]
    invalid = ~valid
    excluded = invalid & np.isfinite(obs)

    obs_line = obs.copy()
    obs_line[invalid] = np.nan
    ax_fit.plot(theta, obs_line, color="black", lw=1.2, label="data", zorder=3)
    if np.any(excluded):
        ax_fit.scatter(theta[excluded], obs[excluded], facecolor="none", edgecolor="red",
                       marker="o", s=30, zorder=4, label="Outliers")

    ax_fit.plot(theta, m0, color="royalblue", lw=1.2, alpha=0.8, label="M0 total")
    ax_fit.plot(theta, m1, color="crimson", lw=1.5, label="M1 total")
    ax_fit.plot(theta, c1, "--", color="darkred", lw=1, alpha=0.7, label="child 1")
    ax_fit.plot(theta, c2, "--", color="darkorange", lw=1, alpha=0.7, label="child 2")
    ax_fit.axvspan(r.theta_left, r.theta_right, color="gray", alpha=0.12)
    ax_fit.set_title(
        f"pix=({x},{y}), peak={int(r.parent_peak_id)}, "
        f"ΔRSS={r.delta_rss_global:.3g}, f={r.best_fraction:g}"
    )

    res0, res1 = obs - m0, obs - m1
    res0_line, res1_line = res0.copy(), res1.copy()
    res0_line[invalid] = np.nan
    res1_line[invalid] = np.nan
    ax_res.plot(theta, res0_line, color="royalblue", lw=1, alpha=0.8, label="M0 residual")
    ax_res.plot(theta, res1_line, color="crimson", lw=1, label="M1 residual")
    ax_res.axhline(0, color="black", lw=0.8, ls=":")
    ax_res.axvspan(r.theta_left, r.theta_right, color="gray", alpha=0.12)
    if np.any(excluded):
        ax_res.scatter(theta[excluded], res0[excluded], color="royalblue", marker="x", s=20, alpha=0.7)
        ax_res.scatter(theta[excluded], res1[excluded], color="crimson", marker="x", s=20)


def plot_peak_splitting_diagnostics(stats_df, data_yxt, theta, spec, evaluate_composite, valid_yxt=None, top_n=6):
    """
    Сравнение M0/M1 и остатков для сильнейших гипотез расщепления.
    показывает, насколько модель M1 с двумя компонентами улучшает фит M0, и сравнивает остатки.
    """
    if stats_df.empty:
        return None, None
    if top_n < 1:
        raise ValueError("top_n должен быть положительным")

    data_yxt = np.asarray(data_yxt, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if data_yxt.ndim != 3 or data_yxt.shape[-1] != len(theta):
        raise ValueError("Ожидаются data_yxt с формой (y, x, theta) и согласованный theta")

    if valid_yxt is not None:
        valid_yxt = np.asarray(valid_yxt, dtype=bool)
        if valid_yxt.shape != data_yxt.shape:
            raise ValueError("valid_yxt и data_yxt должны иметь одинаковую форму")

    required = {"delta_rss_global", "M0_x", "M1_x", "K0", "K1", "parent_index",
                "x_pixel", "y_pixel", "theta_left", "theta_right", "parent_peak_id", "best_fraction"}
    missing = required - set(stats_df.columns)
    if missing:
        raise ValueError(f"В stats_df отсутствуют столбцы: {sorted(missing)}")

    top = stats_df.nlargest(min(top_n, len(stats_df)), "delta_rss_global")
    fig, axes = plt.subplots(len(top), 2, figsize=(12, 2.7 * len(top)), squeeze=False)

    for i, r in enumerate(top.itertuples()):
        _plot_candidate_row(axes[i, 0], axes[i, 1], r, data_yxt, theta, spec, evaluate_composite, valid_yxt)

    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[-1, 0].set_xlabel("theta")
    axes[-1, 1].set_xlabel("theta")
    fig.tight_layout()
    return fig, axes