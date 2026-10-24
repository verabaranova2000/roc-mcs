from __future__ import annotations
import numpy as np,pandas as pd

VERSION="peak_splitting_visualization_v1"

# ==================== e-BH monitoring ====================

def plot_survival(d):
    import matplotlib.pyplot as plt
    from roc_mcs.peak_splitting.results import reconstruct_trajectory
    tr=reconstruct_trajectory(d); m=len(np.asarray(d["n_rep"])); fig,ax=plt.subplots(figsize=(8.8,5.2))
    ax.plot(tr.t,tr.survivors,label="Наблюдаемые survivors")
    ax.plot(tr.t,m/(tr.t+1),ls="--",label="Ориентир: all-null exchangeable, m/(t+1)")
    ax.set_yscale("log"); ax.set_xlabel("Число MC-реплик t"); ax.set_ylabel("Число survivors"); ax.set_title("Sequential survival"); ax.grid(alpha=.25); ax.legend(frameon=False); return fig,ax

def plot_ebh_margin(d):
    import matplotlib.pyplot as plt
    from roc_mcs.peak_splitting.results import reconstruct_trajectory
    tr=reconstruct_trajectory(d); fig,ax=plt.subplots(figsize=(8.8,5.2))
    ax.plot(tr.t,tr.ebh_margin,label=r"$\rho_t=qS_t(t+1)/m$")
    ax.axhline(1,ls="--",label="e-BH boundary")
    ax.set_xlabel("Число MC-реплик t"); ax.set_ylabel("e-BH margin"); ax.set_title("Приближение к e-BH границе"); ax.grid(alpha=.25); ax.legend(frameon=False); return fig,ax

def plot_compute_savings(d):
    import matplotlib.pyplot as plt
    from roc_mcs.peak_splitting.results import reconstruct_trajectory
    tr=reconstruct_trajectory(d); fig,ax=plt.subplots(figsize=(8.8,5.2))
    ax.plot(tr.t,tr.hyprep,label="Фактические hyp-reps")
    ax.plot(tr.t,tr.full_hyprep,ls="--",label="Без early stopping: m·t")
    ax.set_xlabel("Число MC-реплик t"); ax.set_ylabel("Накопленные hyp-reps"); ax.set_title("Вычислительная экономия"); ax.grid(alpha=.25); ax.legend(frameon=False); return fig,ax

def plot_statistic_by_stop(df):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.8,5.2)); frozen=df.loc[~df.survivor]; surv=df.loc[df.survivor]
    if len(frozen): ax.scatter(frozen.n_rep,frozen.T_obs,s=9,alpha=.25,label="Первый exceedance уже был")
    if len(surv): ax.scatter(surv.n_rep,surv.T_obs,s=14,alpha=.65,label="Survivors")
    ax.set_xlabel("Реплика первого loss / текущий t"); ax.set_ylabel(r"$T_{obs}=\Delta RSS$"); ax.set_title("Сила observed statistic и время survival"); ax.grid(alpha=.25); ax.legend(frameon=False); return fig,ax

def plot_spatial_survivors(df):
    import matplotlib.pyplot as plt
    need={"x_pixel","y_pixel","survivor"}
    if not need.issubset(df.columns): raise ValueError(f"Для spatial plot нужны поля {sorted(need)}")
    z=df.groupby(["y_pixel","x_pixel"],as_index=False).agg(n_hyp=("hypothesis_id","size"),n_survivor=("survivor","sum"))
    z["survivor_fraction"]=z.n_survivor/z.n_hyp
    fig,ax=plt.subplots(figsize=(7.2,6.0)); sc=ax.scatter(z.x_pixel,z.y_pixel,c=z.survivor_fraction,s=18)
    fig.colorbar(sc,ax=ax,label="Доля survivor-гипотез в пикселе"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Пространственная концентрация surviving hypotheses"); ax.invert_yaxis(); ax.set_aspect("equal",adjustable="box"); return fig,ax

def print_ebh_monitor(d):
    from roc_mcs.peak_splitting.results import summarize_ebh_snapshot
    s=summarize_ebh_snapshot(d)
    print("=== e-BH LIVE MONITOR ===")
    print(f"rep={s['rep']}; maxB={s['maxB']}; m={s['m']:,}; survivors={s['survivors']:,}; frozen={s['frozen']:,}")
    print(f"hyp-reps={s['hyprep']:,}; экономия vs m*t={s['throughput_saving']:.1%}")
    print(f"e-BH margin rho={s['ebh_margin']:.4f}; boundary=1")
    print(f"при неизменном числе survivors формально нужен t≈{s['required_t_at_current_survivors']:.0f}; фактически S_t будет меняться")
    print(f"заданный horizon B={s['B']:,}; q={s['q']:g}")

# ==================== e-BH result overview ====================

def _result_grid(pix,col):
    xs=np.arange(int(pix.x.min()),int(pix.x.max())+1); ys=np.arange(int(pix.y.min()),int(pix.y.max())+1); xi={v:i for i,v in enumerate(xs)}; yi={v:i for i,v in enumerate(ys)}
    A=np.full((len(ys),len(xs)),np.nan,float); A[pix.y.map(yi).to_numpy(int),pix.x.map(xi).to_numpy(int)]=pd.to_numeric(pix[col],errors="coerce").to_numpy(float)
    return A,xs,ys

def _result_map(ax,pix,col,title,label,vmin=None,vmax=None):
    import matplotlib.pyplot as plt
    A,xs,ys=_result_grid(pix,col); im=ax.imshow(A,interpolation="nearest",aspect="equal",extent=(xs.min()-.5,xs.max()+.5,ys.max()+.5,ys.min()-.5),vmin=vmin,vmax=vmax)
    ax.set(title=title,xlabel="x, pixel",ylabel="y, pixel"); plt.colorbar(im,ax=ax,fraction=.046,pad=.04,label=label); return im

def plot_publication_atlas(pix,physical_map=None,physical_map_label="Physical map",title=None):
    """Publication atlas: evidence, discoveries, burden и physical overlay."""
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12.2,10.0),constrained_layout=True)
    _result_map(axs[0,0],pix,"max_T_obs",r"A  Maximum observed evidence",r"max $\Delta RSS$")
    _result_map(axs[0,1],pix,"n_rejected","B  e-BH discoveries","Rejected hypotheses / pixel",0,None)
    _result_map(axs[1,0],pix,"rejection_fraction","C  Local rejection fraction","Rejected / tested hypotheses",0,1)
    if physical_map is None: _result_map(axs[1,1],pix,"max_n_rep","D  Sequential persistence","Maximum MC survival time",0,None)
    else:
        A=np.asarray(physical_map,float)
        if A.ndim!=2: raise ValueError("physical_map должен быть 2D")
        im=axs[1,1].imshow(A,interpolation="nearest",aspect="equal"); plt.colorbar(im,ax=axs[1,1],fraction=.046,pad=.04,label=physical_map_label)
        d=pix[pix.detected]; axs[1,1].scatter(d.x,d.y,s=12,facecolors="none",edgecolors="k",linewidths=.55)
        axs[1,1].set(title="D  Discoveries over physical map",xlabel="x, pixel",ylabel="y, pixel"); axs[1,1].invert_yaxis()
    if title: fig.suptitle(title)
    return fig,axs

def plot_diagnostic_atlas(hyp,pix):
    """Diagnostic atlas: stopping, evidence separation, spatial survival, convergence."""
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12.2,9.4),constrained_layout=True)
    f=hyp[~hyp.survivor]; s=hyp[hyp.survivor]
    if len(f): axs[0,0].scatter(f.n_rep,f.T_obs,s=7,alpha=.18,label="first exceedance")
    if len(s): axs[0,0].scatter(s.n_rep,s.T_obs,s=10,alpha=.55,label="survivor")
    axs[0,0].set(xlabel="Stopping / censoring time",ylabel=r"$\Delta RSS$",title="A  Evidence vs sequential stopping"); axs[0,0].grid(alpha=.2); axs[0,0].legend(frameon=False)
    vals=hyp.T_obs.to_numpy(float); bins=np.histogram_bin_edges(vals,bins="fd") if len(np.unique(vals))>1 else 10
    for mask,label,lw in ((~hyp.rejected,"non-rejected",1.3),(hyp.rejected,"rejected",1.5)):
        z=hyp.loc[mask,"T_obs"].to_numpy(float)
        if len(z): axs[0,1].hist(z,bins=bins,density=True,histtype="step",linewidth=lw,label=label)
    axs[0,1].set(xlabel=r"$\Delta RSS$",ylabel="Density",title="B  Observed statistic distribution"); axs[0,1].legend(frameon=False)
    _result_map(axs[1,0],pix,"survivor_fraction","C  Survivor fraction","Survivors / tested hypotheses",0,1)
    t=int(hyp.n_rep.max()) if len(hyp) else 0; m=len(hyp); q=float(pix.attrs.get("q",np.nan)); tt=np.arange(t+1); loss=hyp.loss.to_numpy(bool); n=hyp.n_rep.to_numpy(np.int64)
    S=np.array([m if j==0 else np.sum((~loss)|(loss&(n>j))) for j in tt],float); rho=q*S*(tt+1)/m if m and np.isfinite(q) else np.full(len(tt),np.nan)
    axs[1,1].plot(tt,rho,label=r"$\rho_t=qS_t(t+1)/m$"); axs[1,1].axhline(1,ls="--",label="e-BH boundary")
    axs[1,1].set(xlabel="MC replicate t",ylabel="e-BH margin",title="D  e-BH convergence"); axs[1,1].grid(alpha=.2); axs[1,1].legend(frameon=False)
    return fig,axs

def plot_spatial_qc(pix):
    """Дополнительные spatial QC-карты."""
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    _result_map(axs[0],pix,"max_T_obs",r"max $\Delta RSS$",r"max $\Delta RSS$")
    _result_map(axs[1],pix,"max_e","max e-value","e-value",0,None)
    _result_map(axs[2],pix,"n_hyp","Tested hypotheses","Hypotheses / pixel",0,None)
    return fig,axs

def print_summary(hyp,pix):
    from roc_mcs.peak_splitting.results import summary_table
    tab=summary_table(hyp,pix); s=dict(zip(tab.metric,tab.value))
    print("=== REAL e-BH STAGE F ===")
    print(f"hypotheses={int(s['hypotheses']):,}; rejected={int(s['rejected hypotheses']):,} ({s['rejected fraction']:.2%})")
    print(f"pixels={int(s['pixels']):,}; detected={int(s['detected pixels']):,} ({s['detected pixel fraction']:.2%})")
    print(f"max MC reps={int(s['max MC reps']):,}; hyp-reps={int(s['actual hyp-reps']):,}; saving={s['saving vs m*t']:.1%}")
    print(f"e-BH margin rho={s['e-BH margin rho']:.4f}")

# ==================== RCI decomposition ====================

def _map_grid(df, col, *, xcol="x", ycol="y", quality_col=None, shape_xy=None):
    if shape_xy is not None:
        ny, nx = map(int, shape_xy)
        if ny < 1 or nx < 1: raise ValueError("shape_xy должен быть (Ny,Nx) положительных размеров")
        full_xs = np.arange(nx); full_ys = np.arange(ny)
    else: full_xs = full_ys = None
    if df.empty or not {xcol, ycol, col}.issubset(df.columns):
        if shape_xy is not None: return np.full((ny,nx),np.nan),full_xs,full_ys
        return np.empty((0, 0)), np.array([0]), np.array([0])
    z = df.copy()
    if quality_col and quality_col in z.columns: z = z.loc[z[quality_col].astype(bool)]
    if z.empty:
        if shape_xy is not None: return np.full((ny,nx),np.nan),full_xs,full_ys
        return np.empty((0, 0)), np.array([0]), np.array([0])
    if z.duplicated([xcol, ycol]).any(): raise ValueError(f"Карта {col}: несколько событий на пиксель; задайте K0 и parent_index для стратификации")
    if shape_xy is None:
        xs = np.arange(int(z[xcol].min()), int(z[xcol].max()) + 1); ys = np.arange(int(z[ycol].min()), int(z[ycol].max()) + 1)
    else:
        xs, ys = full_xs, full_ys
        if (z[xcol].min() < 0) or (z[xcol].max() >= nx) or (z[ycol].min() < 0) or (z[ycol].max() >= ny): raise ValueError("Координаты map points выходят за shape_xy")
    xi = {v: i for i, v in enumerate(xs)}; yi = {v: i for i, v in enumerate(ys)}; a = np.full((len(ys), len(xs)), np.nan)
    vals = pd.to_numeric(z[col], errors="coerce").to_numpy(float); a[z[ycol].map(yi).to_numpy(int), z[xcol].map(xi).to_numpy(int)] = vals
    return a, xs, ys

def _plot_map(ax, df, col, title, cbar, *, quality_col=None, vmin=None, vmax=None, cmap=None, shape_xy=None):
    import matplotlib.pyplot as plt
    a, xs, ys = _map_grid(df, col, quality_col=quality_col, shape_xy=shape_xy)
    if a.size == 0: ax.text(.5, .5, "No valid estimates", ha="center", va="center", transform=ax.transAxes); ax.set_title(title); ax.set_axis_off(); return None
    finite = a[np.isfinite(a)]
    if finite.size and (vmin is None or vmax is None):
        lo, hi = np.nanquantile(finite, [.01, .99])
        if not lo < hi: lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        vmin = lo if vmin is None else vmin; vmax = hi if vmax is None else vmax
    kw = {"interpolation": "nearest", "aspect": "equal", "extent": (xs.min()-.5, xs.max()+.5, ys.max()+.5, ys.min()-.5), "vmin": vmin, "vmax": vmax}
    if cmap is not None: kw["cmap"] = cmap
    im = ax.imshow(a, **kw); ax.set(title=title, xlabel="x, pixel", ylabel="y, pixel"); plt.colorbar(im, ax=ax, fraction=.046, pad=.04, label=cbar); return im

def plot_m0_rci_atlas(m0_components, *, K0=1, component_index=0, shape_xy=None, title=None):
    """Классические RCI-карты M0: площадь окна, центр и FWHM для заданного локального ранга."""
    import matplotlib.pyplot as plt
    z = m0_components.loc[(m0_components.K0 == int(K0)) & (m0_components.component_index == int(component_index))].copy()
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.5), constrained_layout=True)
    _plot_map(axs[0], z, "area_window", "A  Component integrated intensity", "Area over measured θ window", quality_col="profile_ok", shape_xy=shape_xy)
    _plot_map(axs[1], z, "theta0", "B  Peak position theta0", "theta0", shape_xy=shape_xy)
    _plot_map(axs[2], z, "FWHM", "C  Peak width", "FWHM in θ units", shape_xy=shape_xy)
    _plot_map(axs[3], z, "S_parameter", "D  Model scale S", "Fitted S parameter", shape_xy=shape_xy)
    if title: fig.suptitle(title)
    else: fig.suptitle(f"M0 RCI atlas · K={K0}, local component rank={component_index}")
    return fig, axs

def plot_split_rci_atlas(split_events, *, K0=1, parent_index=0, quality_only=True, shape_xy=None, title=None):
    """Девять карт split-пар; ветви универсально маркируются lower/higher θ, не main/shoulder."""
    import matplotlib.pyplot as plt
    z = split_events.loc[(split_events.K0 == int(K0)) & (split_events.parent_index == int(parent_index))].copy()
    if quality_only: z = z.loc[z.map_quality_ok]
    fig, axs = plt.subplots(3, 3, figsize=(13.2, 12.0), constrained_layout=True)
    fields = [("left_area_window", "A  Lower-θ child intensity", "Window-integrated intensity"), ("left_theta0", "B  Lower-θ position theta0", "theta0"),
              ("left_FWHM", "C  Lower-θ child width", "FWHM"), ("right_area_window", "D  Higher-θ child intensity", "Window-integrated intensity"),
              ("right_theta0", "E  Higher-θ position theta0", "theta0"), ("right_FWHM", "F  Higher-θ child width", "FWHM"),
              ("separation", "G  Split separation theta_H - theta_L", "theta_H - theta_L"),
              ("right_area_fraction", "H  Higher-θ area fraction", "Higher-θ area / pair area"), ("T_obs", "I  Observed statistic delta RSS", "delta RSS")]
    pair_limits = {}
    for cols in (("left_area_window", "right_area_window"), ("left_theta0", "right_theta0"), ("left_FWHM", "right_FWHM")):
        vv = pd.concat([pd.to_numeric(z[c], errors="coerce") for c in cols]).dropna().to_numpy(float) if len(z) else np.array([])
        if len(vv): pair_limits[cols] = tuple(np.nanquantile(vv, [.02, .98]))
    for j, (ax, (col, ttl, cbar)) in enumerate(zip(axs.ravel(), fields)):
        lim = None
        for cols, bounds in pair_limits.items():
            if col in cols: lim = bounds
        _plot_map(ax, z, col, ttl, cbar, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None, shape_xy=shape_xy)
    fig.suptitle(title or f"e-BH split-component RCI atlas · K0={K0}, parent rank={parent_index} · n={len(z)} map-eligible events")
    return fig, axs

def plot_split_qc_atlas(split_events, *, K0=1, parent_index=0, shape_xy=None, title=None):
    """Fit-QC карты отдельно от inference decision."""
    import matplotlib.pyplot as plt
    z=split_events.loc[(split_events.K0==int(K0))&(split_events.parent_index==int(parent_index))].copy()
    fig,axs=plt.subplots(2,3,figsize=(13.2,8.2),constrained_layout=True)
    fields=[("map_quality_ok","A  Parameter-map QC","QC flag"),("delta_identity_error","B  ΔRSS identity error","absolute error"),
            ("M1_optimality","C  Solver optimality","optimality"),("relative_rss_gain","D  Relative RSS gain","relative gain"),
            ("locality_ratio","E  Locality ratio","ratio"),("M1_n_iter","F  Solver iterations","iterations")]
    for ax,(col,ttl,lab) in zip(axs.ravel(),fields):
        if col not in z.columns: ax.text(.5,.5,f"{col} unavailable",ha="center",va="center",transform=ax.transAxes); ax.set_title(ttl); ax.set_axis_off()
        else: _plot_map(ax,z,col,ttl,lab,shape_xy=shape_xy)
    fig.suptitle(title or f"Stage H fit QC · K0={K0}, parent rank={parent_index} · diagnostic only")
    return fig,axs

# ==================== Synthetic validation ====================

def plot_synthetic_truth_atlas(truth_components, m0_components, matched_components, truth_pixels, *, K0=1, component_index=0, title=None):
    """Synthetic comparison: truth, baseline M0 and selected fitted main/shoulder layers."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 3, figsize=(13.2, 16.4), constrained_layout=True)
    shape_xy = (int(truth_pixels.y.max())+1, int(truth_pixels.x.max())+1) if len(truth_pixels) else None
    truth_main = truth_components.loc[truth_components.role == "main"].copy(); truth_sh = truth_components.loc[truth_components.role == "shoulder"].copy()
    m0 = m0_components.loc[(m0_components.K0 == int(K0)) & (m0_components.component_index == int(component_index))].copy()
    fit_main = matched_components.loc[(matched_components.truth_role_match == "main") & matched_components.truth_present].copy() if len(matched_components) else matched_components
    fit_sh = matched_components.loc[(matched_components.truth_role_match == "shoulder") & matched_components.truth_present].copy() if len(matched_components) else matched_components
    rows = [(truth_main, "True main"), (m0, "M0 one-component baseline"), (truth_sh, "True shoulder"), (fit_main, "Selected M1 child matched to main"), (fit_sh, "Selected M1 child matched to shoulder")]
    cols = [("area_window", "Integrated intensity", "Window area"), ("theta0", "Peak position theta0", "theta0"), ("FWHM", "Peak width", "FWHM")]
    for j, (col, _, _) in enumerate(cols):
        vals = []
        for df, _ in rows:
            if len(df) and col in df: vals.extend(pd.to_numeric(df[col], errors="coerce").dropna().to_list())
        lo, hi = (np.nanquantile(vals, [.02, .98]) if len(vals) else (None, None))
        if lo == hi: lo, hi = None, None
        for i, (df, label) in enumerate(rows):
            cmap = "viridis" if "True" in label or "M0" in label else "viridis"
            _plot_map(axs[i, j], df, col, f"{chr(65+i*3+j)}  {label}: {cols[j][1]}", cols[j][2], vmin=lo, vmax=hi, cmap=cmap, shape_xy=shape_xy)
    if title: fig.suptitle(title)
    else: fig.suptitle("Synthetic RCI decomposition: ground truth vs fitted component maps")
    return fig, axs

def plot_synthetic_detection_profile(truth_pixels, *, bins=20, title=None):
    """Описательная detection fraction по w в одной реализации; не формальная power/CI."""
    import matplotlib.pyplot as plt
    z = truth_pixels.copy(); w = pd.to_numeric(z.split_weight, errors="coerce").to_numpy(float); d = z.detected.to_numpy(bool); y = z.truth_split.to_numpy(bool)
    upper = max(float(np.nanmax(w)) if len(w) else 1.0, 1e-12); edges = np.linspace(0, upper + 1e-12, int(bins) + 1); ix = np.clip(np.digitize(w, edges[1:-1], right=False), 0, int(bins)-1)
    rows = []
    for b in range(int(bins)):
        mask = (ix == b) & y; n = int(mask.sum())
        if not n: continue
        k = int(d[mask].sum()); rows.append({"weight_mid": float(np.mean(w[mask])), "n_truth_split": n, "n_detected": k, "detection_fraction": k/n})
    prof = pd.DataFrame(rows); fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    if len(prof):
        ax.plot(prof.weight_mid, prof.detection_fraction, marker="o", label="Single-scan detection fraction")
        for r in prof.itertuples(): ax.annotate(f"n={r.n_truth_split}", (r.weight_mid, r.detection_fraction), xytext=(3,5), textcoords="offset points", fontsize=7)
    ax.set(xlabel="True synthetic split weight w", ylabel="Detection fraction among true-split pixels", ylim=(-.02, 1.02), title=title or "Single-scan detection profile (descriptive, not formal power)")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    return fig, ax, prof

def plot_synthetic_confusion_map(truth_pixels, title="Synthetic pixel-level confusion map"):
    """Truth-based pixel map: only for synthetic data, never experimental data."""
    import matplotlib.pyplot as plt
    code = {"TN": 0, "TP": 1, "FP": 2, "FN": 3}; z = truth_pixels.copy(); z["_code"] = z.detection_class.map(code)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True); shape_xy = (int(z.y.max())+1, int(z.x.max())+1) if len(z) else None; a, xs, ys = _map_grid(z, "_code", shape_xy=shape_xy)
    if a.size:
        cmap = plt.get_cmap("viridis", 4); im = ax.imshow(a, interpolation="nearest", aspect="equal", extent=(xs.min()-.5,xs.max()+.5,ys.max()+.5,ys.min()-.5), cmap=cmap, vmin=-.5, vmax=3.5)
        cb=plt.colorbar(im,ax=ax,ticks=[0,1,2,3],fraction=.046,pad=.04); cb.ax.set_yticklabels(["TN","TP","FP","FN"])
    ax.set(title=title, xlabel="x, pixel", ylabel="y, pixel")
    return fig, ax

# ==================== Reporting ====================

def print_stage_summary(stage_g, stage_h):
    a = stage_g["audit"]; h = stage_h["audit"]
    print("=== G/e-BH DECISION ===")
    print(f"m={a['m']:,}; q={a['q']:g}; rejected hypotheses={a['n_rejected']:,}; pixels with ≥1 rejected={a['n_rejected_pixels']:,}")
    print("G does not re-run BY/BH: formal candidate set comes from Stage F e-BH.")
    if a.get("snapshot_only", False): print("WARNING: this is a checkpoint snapshot, not a completed formal Stage F result.")
    print("=== H / RCI PARAMETER DECOMPOSITION ===")
    print(f"split events={h['n_split_events']:,}; map-quality OK={h['n_map_quality_ok']:,}; flagged for review={h['n_map_quality_failed']:,}")
    print(h["parameter_source"])

