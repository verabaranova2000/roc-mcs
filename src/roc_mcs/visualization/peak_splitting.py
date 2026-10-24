from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


VERSION="peak_splitting_visualization_v2"


# ==================== D1 split hypothesis diagnostics ====================

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



# ==================== Pre-F noise validation ====================

def _pref_audit(r):
    a=r.get("audit") if isinstance(r,dict) else None
    if a is None: raise ValueError("pre-F result не содержит audit")
    return a

def _pref_get(obj,name,default=np.nan): return getattr(obj,name,default)

def _pref_model(e1_results):
    if not isinstance(e1_results,dict) or e1_results.get("noise_model") is None: raise ValueError("e1_results не содержит noise_model")
    m=e1_results["noise_model"]
    if not callable(getattr(m,"simulate_null",None)): raise TypeError("noise_model не реализует simulate_null")
    return m

def _pref_finite(x):
    a=np.asarray(x,float).ravel()
    return a[np.isfinite(a)]

def _pref_outer_arrays(r):
    tab=r.get("outer_table"); vals=r.get("outer_raw_p"); done=r.get("outer_done"); names=tuple(map(str,r.get("anchors",{}).keys()))
    if isinstance(tab,pd.DataFrame) and "anchor" in tab: names=tuple(tab["anchor"].astype(str))
    if vals is None: return names,None,np.zeros(len(names),int)
    vals=np.asarray(vals,float)
    if vals.ndim!=2: raise ValueError("outer_raw_p должен быть 2D")
    done=np.sum(np.isfinite(vals),axis=1) if done is None else np.asarray(done,int)
    if done.shape!=(vals.shape[0],): raise ValueError("outer_done несовместим с outer_raw_p")
    if len(names)!=vals.shape[0]: names=tuple(f"anchor {i}" for i in range(vals.shape[0]))
    return names,vals,done

def _pref_mu_range(e1_results,pref_result=None):
    mx=[]
    for t in (e1_results.get("tables") or {}).values():
        a=np.asarray(t,float)
        if a.ndim==2 and a.shape[1]>=1:
            z=a[:,0]; z=z[np.isfinite(z)&(z>0)]
            if len(z): mx.append(float(np.max(z)))
    if pref_result is not None:
        mu=np.asarray(pref_result.get("mu_hat",[]),float); z=mu[np.isfinite(mu)&(mu>0)]
        if len(z): mx.append(float(np.quantile(z,.995)))
    return 1.08*max(mx) if mx else 800.0

def _pref_simulate_noise_law(model,mu_grid,n_draws,seed):
    mu=np.asarray(mu_grid,float)
    if mu.ndim!=1 or len(mu)<3 or np.any(~np.isfinite(mu)) or np.any(mu<0): raise ValueError("Некорректный mu_grid")
    n_draws=int(n_draws)
    if n_draws<500: raise ValueError("n_draws должен быть >=500")
    rng=np.random.default_rng(int(seed)); out={k:np.empty(len(mu)) for k in ("mean","var","p0")}
    for i,m in enumerate(mu):
        left=n_draws; chunks=[]
        while left:
            b=min(left,20000); y=np.asarray(model.simulate_null(rng,np.full((b,1),m,float)),float).ravel()
            if len(y)!=b or np.any(~np.isfinite(y)): raise RuntimeError("noise_model.simulate_null вернул некорректный sample")
            chunks.append(y); left-=b
        y=np.concatenate(chunks); out["mean"][i]=y.mean(); out["var"][i]=y.var(ddof=1); out["p0"][i]=np.mean(y==0)
    out["mu"]=mu; out["fano"]=np.divide(out["var"],mu,out=np.full(len(mu),np.nan),where=mu>0)
    return out

def _pref_selected_data(pref_result,data_yxt):
    if "mu_hat" not in pref_result: raise ValueError("pref_result не содержит mu_hat; нужен результат pre-F v4")
    d=pref_result.get("design") or {}; coords=np.asarray(d.get("coords_yx",[])); valid=np.asarray(d.get("valid",[]),bool); mu=np.asarray(pref_result["mu_hat"],float); data=np.asarray(data_yxt,float)
    if coords.ndim!=2 or coords.shape[1]!=2: raise ValueError("pref_result['design']['coords_yx'] отсутствует или имеет неверную форму")
    if mu.ndim!=2 or mu.shape[0]!=len(coords): raise ValueError("mu_hat несовместим с coords_yx")
    if valid.shape!=mu.shape: raise ValueError("design valid несовместим с mu_hat")
    yy=coords[:,0].astype(int); xx=coords[:,1].astype(int)
    if np.any(yy<0)|np.any(xx<0)|np.any(yy>=data.shape[0])|np.any(xx>=data.shape[1]): raise ValueError("coords_yx выходят за data_yxt")
    y=np.asarray(data[yy,xx],float)
    if y.shape!=mu.shape: raise ValueError(f"selected data shape {y.shape} != mu_hat {mu.shape}")
    mask=valid&np.isfinite(y)&np.isfinite(mu)&(mu>=0)
    return y,mu,mask,coords

def _pref_quantile_edges(mu,n_bins):
    z=np.asarray(mu,float); z=z[np.isfinite(z)&(z>=0)]
    if len(z)<max(20,int(n_bins)*4): raise ValueError("Недостаточно значений μ для binning")
    q=np.linspace(0,1,int(n_bins)+1); e=np.unique(np.quantile(z,q))
    if len(e)<3: raise ValueError("Слишком мало уникальных μ для binning")
    e[0]-=np.finfo(float).eps; e[-1]+=np.finfo(float).eps
    return e

def _pref_wilson(k,n,alpha=.05):
    from scipy.stats import norm
    if n<=0: return np.nan,np.nan
    z=float(norm.ppf(1-alpha/2)); p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)

def _pre_f_data_model_bins(pref_result,e1_results,data_yxt,*,n_bins=12,n_sim=30000,seed=20261103):
    """Observed-vs-model calibration на тех же non-overlap curves и fitted μ, что использовал pre-F."""
    model=_pref_model(e1_results); y,mu,mask,_=_pref_selected_data(pref_result,data_yxt); yf=y[mask]; mf=mu[mask]; edges=_pref_quantile_edges(mf,n_bins); bid=np.digitize(mf,edges[1:-1]); rng=np.random.default_rng(int(seed)); rows=[]
    for b in range(len(edges)-1):
        take=bid==b
        if not np.any(take): continue
        yy=yf[take]; mm=mf[take]; n=len(yy); k0=int(np.sum(yy==0)); lo0,hi0=_pref_wilson(k0,n); draw_mu=rng.choice(mm,size=int(n_sim),replace=True)
        ys=np.asarray(model.simulate_null(rng,draw_mu),float).ravel()
        if len(ys)!=len(draw_mu) or np.any(~np.isfinite(ys)): raise RuntimeError("Некорректный model simulation")
        obs_mse=float(np.mean((yy-mm)**2)); sim_mse=float(np.mean((ys-draw_mu)**2)); rows.append({"bin":b,"mu_median":float(np.median(mm)),"mu_mean":float(np.mean(mm)),"n":n,"zero_obs":k0/n,"zero_lo":lo0,"zero_hi":hi0,"zero_model":float(np.mean(ys==0)),"mse_obs":obs_mse,"mse_model":sim_mse,"fano_obs":obs_mse/float(np.mean(mm)) if np.mean(mm)>0 else np.nan,"fano_model":sim_mse/float(np.mean(draw_mu)) if np.mean(draw_mu)>0 else np.nan})
    return pd.DataFrame(rows)

def make_pre_f_summary_table(pref_result):
    a=_pref_audit(pref_result)
    rows=[("decision",str(_pref_get(a,"decision",""))),("reason",str(_pref_get(a,"reason",""))),("raw p",float(_pref_get(a,"raw_p",np.nan))),("calibrated p estimate",float(_pref_get(a,"calibrated_p_estimate",np.nan))),("simultaneous CP lower",float(_pref_get(a,"calibrated_p_lower",np.nan))),("simultaneous CP upper",float(_pref_get(a,"calibrated_p_upper",np.nan))),("alpha",float(_pref_get(a,"family_alpha",np.nan))),("MC risk bound",float(_pref_get(a,"outer_mc_alpha",np.nan))),("curves",int(_pref_get(a,"n_curves",0))),("anchors",int(_pref_get(a,"n_anchors",0))),("unsupported fraction",float(_pref_get(a,"unsupported_fraction",np.nan))),("extrapolated fraction",float(_pref_get(a,"extrapolated_fraction",np.nan))),("boundary-selection fraction",float(_pref_get(a,"boundary_selection_fraction",np.nan))),("outer trials total",int(_pref_get(a,"outer_trials_total",0))),("model kind",str(_pref_get(a,"model_kind","")))]
    return pd.DataFrame(rows,columns=["quantity","value"])

def plot_pre_f_noise_model(e1_results,*,pref_result=None,mu_max=None,n_mu=40,n_draws=30000,seed=20261101,figsize=(10.8,7.2)):
    """Закон генератора Stage F + E1 calibration points."""
    m=_pref_model(e1_results); mu_max=_pref_mu_range(e1_results,pref_result) if mu_max is None else float(mu_max); grid=np.geomspace(max(.05,mu_max/600),mu_max,int(n_mu)); law=_pref_simulate_noise_law(m,grid,n_draws,seed)
    fig,axs=plt.subplots(2,2,figsize=figsize,constrained_layout=True)
    ax=axs[0,0]; ax.plot(law["mu"],law["mean"],marker="o",ms=3,lw=1); ax.plot([0,mu_max],[0,mu_max],ls="--",lw=1); ax.set(xlabel="Target mean μ",ylabel="Simulated mean",title="A  Generator mean calibration"); ax.grid(alpha=.2)
    ax=axs[0,1]; ax.plot(law["mu"],law["var"],marker="o",ms=3,lw=1,label="generator"); ax.plot(law["mu"],law["mu"],ls=":",lw=1,label="Poisson Var=μ")
    for o,t in sorted((e1_results.get("tables") or {}).items()):
        a=np.asarray(t,float)
        if a.ndim==2 and a.shape[1]>=4:
            ok=np.isfinite(a[:,0])&np.isfinite(a[:,3])&(a[:,0]>0)&(a[:,3]>=0)
            if np.any(ok): ax.scatter(a[ok,0],a[ok,3]**2,s=18,alpha=.7,label=f"E1 order {o}")
    ax.set(xscale="log",yscale="log",xlabel="Mean μ",ylabel="Variance",title="B  Conditional variance law"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    ax=axs[1,0]; ax.plot(law["mu"],law["fano"],marker="o",ms=3,lw=1,label="generator")
    for o,t in sorted((e1_results.get("tables") or {}).items()):
        a=np.asarray(t,float)
        if a.ndim==2 and a.shape[1]>=4:
            ok=np.isfinite(a[:,0])&np.isfinite(a[:,3])&(a[:,0]>0)&(a[:,3]>=0)
            if np.any(ok): ax.scatter(a[ok,0],a[ok,3]**2/a[ok,0],s=18,alpha=.7,label=f"E1 order {o}")
    ax.axhline(1,ls=":",lw=1,label="Poisson F=1"); ax.set(xscale="log",xlabel="Mean μ",ylabel="Var(Y|μ)/μ",title="C  Conditional Fano curve"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    ax=axs[1,1]; ax.plot(law["mu"],law["p0"],marker="o",ms=3,lw=1); ax.set(xscale="log",yscale="log",xlabel="Mean μ",ylabel="P(Y=0 | μ)",title="D  Zero mass implied by generator"); ax.grid(alpha=.2)
    fig.suptitle(f"Count-noise generator: {getattr(m,'kind',type(m).__name__)}",fontsize=12)
    return fig,axs,law

def plot_pre_f_data_model_calibration(pref_result,e1_results,data_yxt,*,n_bins=12,n_sim=30000,seed=20261103,figsize=(10.8,4.5)):
    """Zero-rate и residual-variance calibration на реальном pre-F design."""
    z=_pre_f_data_model_bins(pref_result,e1_results,data_yxt,n_bins=n_bins,n_sim=n_sim,seed=seed); fig,axs=plt.subplots(1,2,figsize=figsize,constrained_layout=True)
    ax=axs[0]; x=z["mu_median"].to_numpy(); y=z["zero_obs"].to_numpy(); lo=z["zero_lo"].to_numpy(); hi=z["zero_hi"].to_numpy(); ax.errorbar(x,y,yerr=np.vstack((y-lo,hi-y)),fmt="o",capsize=2,label="observed ±95% Wilson"); ax.plot(x,z["zero_model"].to_numpy(),marker="o",lw=1,label="model"); ax.set(xscale="log",yscale="log",xlabel="Fitted μ",ylabel="P(Y=0 | μ-bin)",title="A  Zero-rate calibration"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    ax=axs[1]; ax.plot(x,z["fano_obs"].to_numpy(),marker="o",lw=1,label="observed residual Fano"); ax.plot(x,z["fano_model"].to_numpy(),marker="o",lw=1,label="model simulation"); ax.axhline(1,ls=":",lw=1,label="Poisson F=1"); ax.set(xscale="log",yscale="log",xlabel="Fitted μ",ylabel="Mean squared residual / mean μ",title="B  Conditional dispersion calibration"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    return fig,axs,z

def plot_pre_f_calibration(pref_result,*,figsize=(11.0,7.2)):
    """Outer-prepivot calibration: zoom на реально достижимые raw p* и явные unevaluated anchors."""
    a=_pref_audit(pref_result); names,vals,done=_pref_outer_arrays(pref_result); pobs=float(_pref_get(a,"raw_p",np.nan)); alpha=float(_pref_get(a,"family_alpha",np.nan)); tab=pref_result.get("outer_table"); fig,axs=plt.subplots(2,2,figsize=figsize,constrained_layout=True)
    ax=axs[0,0]
    allz=[]
    if vals is not None:
        for i,name in enumerate(names):
            z=vals[i,:done[i]]; z=z[np.isfinite(z)]
            if len(z):
                allz.append(z); xs=np.sort(z); yy=np.arange(1,len(xs)+1)/len(xs); ax.step(xs,yy,where="post",lw=1.2,label=f"{name} (n={len(z)})")
        ax.axvline(pobs,ls="--",lw=1.2,label=f"observed={pobs:.4g}")
        if allz:
            zall=np.concatenate(allz); hi=max(np.max(zall),pobs); ax.set_xlim(max(0,min(np.min(zall),pobs)-.15*max(hi,1e-4)),hi*1.25)
        ax.set(xlabel="Inner raw p*",ylabel="Empirical CDF",title="A  Outer null distribution (zoom)"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    else: ax.text(.5,.5,"outer_raw_p unavailable",ha="center",va="center",transform=ax.transAxes); ax.set_axis_off()
    ax=axs[0,1]
    if vals is not None:
        items=[]
        for i,name in enumerate(names):
            z=vals[i,:done[i]]; z=z[np.isfinite(z)]
            if len(z):
                u,c=np.unique(z,return_counts=True)
                for uu,cc in zip(u,c): items.append((name,float(uu),int(cc)))
        if items:
            dd=pd.DataFrame(items,columns=["anchor","p","count"]); pu=np.sort(dd["p"].unique()); width=max((pu[1]-pu[0])*.65 if len(pu)>1 else max(pobs*.25,1e-4),1e-5)
            for name,g in dd.groupby("anchor",sort=False): ax.bar(g["p"],g["count"],width=width,alpha=.65,label=name)
            ax.axvline(pobs,ls="--",lw=1.2,label=f"observed={pobs:.4g}"); ax.set_xlim(max(0,min(dd.p.min(),pobs)-width),max(dd.p.max(),pobs)+2*width); ax.set(xlabel="Discrete inner raw p*",ylabel="Outer trial count",title="B  Discrete p* spectrum"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2,axis="y")
        else: ax.text(.5,.5,"Нет завершённых outer trials",ha="center",va="center",transform=ax.transAxes); ax.set_axis_off()
    else: ax.set_axis_off()
    ax=axs[1,0]
    if isinstance(tab,pd.DataFrame) and len(tab):
        t=tab.copy().reset_index(drop=True); x=np.arange(len(t)); ax.axhline(alpha,ls="--",lw=1,label=f"α={alpha:g}")
        for i,row in t.iterrows():
            n=int(row.get("n_outer",0))
            if n>0 and np.isfinite(row.get("calibrated_p_estimate",np.nan)):
                est=float(row["calibrated_p_estimate"]); lo=float(row["CP_lower"]); hi=float(row["CP_upper"]); ax.errorbar([i],[est],yerr=[[max(est-lo,0)],[max(hi-est,0)]],fmt="o",capsize=3,lw=1)
            else: ax.text(i,.5,"not\nevaluated",ha="center",va="center",fontsize=8)
        ax.set_xticks(x,labels=t["anchor"].astype(str),rotation=18,ha="right"); ax.set_ylim(-.03,1.03); ax.set(ylabel="Calibrated probability",title="C  Anchor-wise simultaneous bounds"); ax.legend(frameon=False); ax.grid(alpha=.2)
    else: ax.text(.5,.5,"outer_table unavailable",ha="center",va="center",transform=ax.transAxes); ax.set_axis_off()
    ax=axs[1,1]; ax.axis("off"); decision=str(_pref_get(a,"decision","")); lower=float(_pref_get(a,"calibrated_p_lower",np.nan)); upper=float(_pref_get(a,"calibrated_p_upper",np.nan)); pest=float(_pref_get(a,"calibrated_p_estimate",np.nan)); reason=str(_pref_get(a,"reason","")).replace("np.str_('","'").replace("')","'")
    txt=(f"Decision: {decision}\n\nraw p = {pobs:.4g}\ncalibrated estimate = {pest:.4g}\nsimultaneous CP = [{lower:.4g}, {upper:.4g}]\nα = {alpha:g}; MC-risk ≤ {float(_pref_get(a,'outer_mc_alpha',np.nan)):g}\n\nunsupported = {float(_pref_get(a,'unsupported_fraction',np.nan)):.2%}\nextrapolated = {float(_pref_get(a,'extrapolated_fraction',np.nan)):.2%}\nboundary-selected = {float(_pref_get(a,'boundary_selection_fraction',np.nan)):.2%}\nouter trials = {int(_pref_get(a,'outer_trials_total',0)):,}\n\n{reason}")
    ax.text(.02,.98,txt,va="top",ha="left",fontsize=9,linespacing=1.35); ax.set_title("D  Formal pre-F audit",loc="left")
    fig.suptitle("Pre-F calibrated composite-null check",fontsize=12)
    return fig,axs

def plot_pre_f_rootogram(pref_result,e1_results,data_yxt,*,max_count=None,n_sim=200,seed=20261104,figsize=(10.2,4.6)):
    """Simulation-based hanging rootogram на exact pre-F design."""
    model=_pref_model(e1_results); y,mu,mask,_=_pref_selected_data(pref_result,data_yxt); yy=y[mask]; mm=mu[mask]
    if not np.allclose(yy,np.rint(yy),atol=1e-8): raise ValueError("Rootogram требует integer-like counts")
    yy=np.rint(yy).astype(int); max_count=int(np.quantile(yy,.995)) if max_count is None else int(max_count); max_count=max(10,max_count)
    obs=np.bincount(np.clip(yy,0,max_count),minlength=max_count+1).astype(float); rng=np.random.default_rng(int(seed)); exp=np.zeros_like(obs)
    for _ in range(int(n_sim)):
        z=np.asarray(model.simulate_null(rng,mm),float).ravel()
        if len(z)!=len(mm) or np.any(~np.isfinite(z)): raise RuntimeError("Некорректный model simulation")
        zz=np.rint(np.maximum(z,0)).astype(int); exp+=np.bincount(np.clip(zz,0,max_count),minlength=max_count+1)
    exp/=float(n_sim); res=np.sqrt(obs)-np.sqrt(np.maximum(exp,0)); fig,ax=plt.subplots(figsize=figsize,constrained_layout=True); ax.axhline(0,lw=1); ax.bar(np.arange(len(res)),res,width=.9,alpha=.65); ax.set(xlabel=f"Count y (last bin ≥{max_count})",ylabel="√observed − √expected",title="Simulation-based hanging rootogram"); ax.grid(alpha=.18,axis="y"); return fig,ax,{"observed":obs,"expected":exp,"residual":res}

def plot_pre_f_pit_diagnostics(pref_result,*,figsize=(10.5,4.4)):
    raw=pref_result.get("raw_result")
    fig,axs=plt.subplots(1,2,figsize=figsize,constrained_layout=True)
    if raw is None:
        for ax,label in zip(axs,("Randomized PIT","RQR Q-Q")): ax.text(.5,.5,f"{label} arrays не сохранены\nв resumed pre-F result",ha="center",va="center",transform=ax.transAxes); ax.set_axis_off()
        return fig,axs
    def find(obj,names):
        names={x.lower() for x in names}
        if isinstance(obj,dict):
            for k,v in obj.items():
                if str(k).lower() in names: return v
            for v in obj.values():
                z=find(v,names)
                if z is not None:return z
        return None
    pit=find(raw,("pit","randomized_pit","pit_values")); rqr=find(raw,("rqr","quantile_residuals","randomized_quantile_residuals"))
    if pit is not None:
        u=_pref_finite(pit); axs[0].hist(u,bins=30,density=True,histtype="stepfilled",alpha=.35); axs[0].axhline(1,ls="--",lw=1); axs[0].set(xlabel="Randomized PIT",ylabel="Density",title="PIT uniformity"); axs[0].grid(alpha=.2)
    else: axs[0].text(.5,.5,"PIT array не сохранён",ha="center",va="center",transform=axs[0].transAxes); axs[0].set_axis_off()
    if rqr is not None:
        from scipy.stats import norm
        z=np.sort(_pref_finite(rqr)); q=norm.ppf((np.arange(len(z))+.5)/len(z)); axs[1].plot(q,z,lw=1); lo=min(q.min(),z.min()); hi=max(q.max(),z.max()); axs[1].plot([lo,hi],[lo,hi],ls="--",lw=1); axs[1].set(xlabel="N(0,1) quantile",ylabel="RQR quantile",title="Randomized quantile residual Q-Q"); axs[1].grid(alpha=.2)
    else: axs[1].text(.5,.5,"RQR array не сохранён",ha="center",va="center",transform=axs[1].transAxes); axs[1].set_axis_off()
    return fig,axs

def plot_pre_f_publication_atlas(pref_result,e1_results,data_yxt,*,n_bins=12,n_sim_bins=30000,n_draws_law=30000,seed=20261110,figsize=(12.2,12.2)):
    """Главная 6-panel figure: закон шума, fit к данным и формальная outer calibration."""
    model=_pref_model(e1_results); mu_max=_pref_mu_range(e1_results,pref_result); law=_pref_simulate_noise_law(model,np.geomspace(max(.05,mu_max/600),mu_max,40),n_draws_law,seed); cal=_pre_f_data_model_bins(pref_result,e1_results,data_yxt,n_bins=n_bins,n_sim=n_sim_bins,seed=seed+1); a=_pref_audit(pref_result); names,vals,done=_pref_outer_arrays(pref_result); tab=pref_result.get("outer_table"); pobs=float(_pref_get(a,"raw_p",np.nan)); alpha=float(_pref_get(a,"family_alpha",np.nan))
    fig,axs=plt.subplots(3,2,figsize=figsize,constrained_layout=True)
    ax=axs[0,0]; ax.plot(law["mu"],law["var"],lw=1.2,label="generator"); ax.plot(law["mu"],law["mu"],ls=":",lw=1,label="Poisson")
    for o,t in sorted((e1_results.get("tables") or {}).items()):
        x=np.asarray(t,float)
        if x.ndim==2 and x.shape[1]>=4:
            ok=np.isfinite(x[:,0])&np.isfinite(x[:,3])&(x[:,0]>0)&(x[:,3]>=0)
            if np.any(ok): ax.scatter(x[ok,0],x[ok,3]**2,s=18,alpha=.7,label=f"E1 order {o}")
    ax.set(xscale="log",yscale="log",xlabel="Mean μ",ylabel="Variance",title="A  Noise variance law"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    ax=axs[0,1]; ax.plot(law["mu"],law["fano"],lw=1.2,label="generator")
    for o,t in sorted((e1_results.get("tables") or {}).items()):
        x=np.asarray(t,float)
        if x.ndim==2 and x.shape[1]>=4:
            ok=np.isfinite(x[:,0])&np.isfinite(x[:,3])&(x[:,0]>0)&(x[:,3]>=0)
            if np.any(ok): ax.scatter(x[ok,0],x[ok,3]**2/x[ok,0],s=18,alpha=.7,label=f"E1 order {o}")
    ax.axhline(1,ls=":",lw=1,label="Poisson F=1"); ax.set(xscale="log",xlabel="Mean μ",ylabel="Fano factor",title="B  Intensity-dependent overdispersion"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    ax=axs[1,0]; x=cal["mu_median"].to_numpy(); y=cal["zero_obs"].to_numpy(); lo=cal["zero_lo"].to_numpy(); hi=cal["zero_hi"].to_numpy(); ax.errorbar(x,y,yerr=np.vstack((y-lo,hi-y)),fmt="o",capsize=2,label="observed"); ax.plot(x,cal["zero_model"],marker="o",lw=1,label="model"); ax.set(xscale="log",yscale="log",xlabel="Fitted μ",ylabel="Zero fraction",title="C  Zero-rate calibration"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    ax=axs[1,1]; ax.plot(x,cal["fano_obs"],marker="o",lw=1,label="observed residual"); ax.plot(x,cal["fano_model"],marker="o",lw=1,label="model"); ax.axhline(1,ls=":",lw=1); ax.set(xscale="log",yscale="log",xlabel="Fitted μ",ylabel="Residual MSE / mean μ",title="D  Dispersion calibration on real curves"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    ax=axs[2,0]
    if vals is not None:
        items=[]
        for i,name in enumerate(names):
            z=vals[i,:done[i]]; z=z[np.isfinite(z)]
            if len(z):
                u,c=np.unique(z,return_counts=True)
                for uu,cc in zip(u,c): items.append((name,float(uu),int(cc)))
        dd=pd.DataFrame(items,columns=["anchor","p","count"])
        if len(dd):
            pu=np.sort(dd.p.unique()); width=max((pu[1]-pu[0])*.65 if len(pu)>1 else max(pobs*.25,1e-4),1e-5)
            for name,g in dd.groupby("anchor",sort=False): ax.bar(g.p,g["count"],width=width,alpha=.65,label=name)
            ax.axvline(pobs,ls="--",lw=1.2,label=f"observed={pobs:.4g}"); ax.set_xlim(max(0,min(dd.p.min(),pobs)-width),max(dd.p.max(),pobs)+2*width); ax.legend(frameon=False,fontsize=8)
    ax.set(xlabel="Inner raw p*",ylabel="Outer trial count",title="E  Prepivot calibration spectrum"); ax.grid(alpha=.2,axis="y")
    ax=axs[2,1]; ax.axhline(alpha,ls="--",lw=1,label=f"α={alpha:g}")
    if isinstance(tab,pd.DataFrame):
        t=tab.reset_index(drop=True); pos=np.arange(len(t))
        for i,row in t.iterrows():
            n=int(row.get("n_outer",0))
            if n>0 and np.isfinite(row.get("calibrated_p_estimate",np.nan)):
                est=float(row["calibrated_p_estimate"]); lo=float(row["CP_lower"]); hi=float(row["CP_upper"]); ax.errorbar([i],[est],yerr=[[max(est-lo,0)],[max(hi-est,0)]],fmt="o",capsize=3)
            else: ax.text(i,.5,"not\nevaluated",ha="center",va="center",fontsize=8)
        ax.set_xticks(pos,labels=t["anchor"].astype(str),rotation=18,ha="right")
    ax.set_ylim(-.03,1.03); ax.set(ylabel="Calibrated probability",title=f"F  Formal gate: {_pref_get(a,'decision','')}"); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
    fig.suptitle("Pre-F validation of the count-noise model",fontsize=13)
    return fig,axs,{"noise_law":law,"calibration_bins":cal}

def save_pre_f_publication_report(pref_result,e1_results,data_yxt,*,output_pdf="pre_f_noise_report_v2.pdf",n_bins=12,n_sim_bins=30000,n_draws_law=30000,seed=20261110):
    from matplotlib.backends.backend_pdf import PdfPages
    out=Path(output_pdf)
    with PdfPages(out) as pdf:
        fig,_,_=plot_pre_f_publication_atlas(pref_result,e1_results,data_yxt,n_bins=n_bins,n_sim_bins=n_sim_bins,n_draws_law=n_draws_law,seed=seed); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        fig,_,_=plot_pre_f_noise_model(e1_results,pref_result=pref_result,n_draws=n_draws_law,seed=seed+2); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        fig,_,_=plot_pre_f_data_model_calibration(pref_result,e1_results,data_yxt,n_bins=n_bins,n_sim=n_sim_bins,seed=seed+3); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        fig,_,_=plot_pre_f_rootogram(pref_result,e1_results,data_yxt,n_sim=200,seed=seed+4); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        fig,_=plot_pre_f_calibration(pref_result); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        fig,_=plot_pre_f_pit_diagnostics(pref_result); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
    return out

def print_pre_f_report(pref_result):
    a=_pref_audit(pref_result); print("=== PRE-F PUBLICATION SUMMARY v2 ==="); print(f"decision={_pref_get(a,'decision','')}; reason={_pref_get(a,'reason','')}"); print(f"raw p={float(_pref_get(a,'raw_p',np.nan)):.4g}; calibrated p≈{float(_pref_get(a,'calibrated_p_estimate',np.nan)):.4g}; CP=[{float(_pref_get(a,'calibrated_p_lower',np.nan)):.4g},{float(_pref_get(a,'calibrated_p_upper',np.nan)):.4g}]"); print(f"unsupported={float(_pref_get(a,'unsupported_fraction',np.nan)):.2%}; extrapolated={float(_pref_get(a,'extrapolated_fraction',np.nan)):.2%}; boundary={float(_pref_get(a,'boundary_selection_fraction',np.nan)):.2%}")

# ==================== E2 observed split statistic ====================

def make_t_obs_pixel_map(df_obs,shape_xy,*,value_col="delta_rss_global",aggregation="max"):
    """2D-карта новой T_obs=ΔRSS; по пикселю берётся max по hypotheses."""
    if value_col not in df_obs or "pixel_id" not in df_obs: raise KeyError(f"Нужны {value_col!r} и pixel_id")
    h,w=map(int,shape_xy); d=df_obs[["pixel_id",value_col]+[c for c in ("x_pixel","y_pixel","x","y") if c in df_obs.columns]].copy()
    xcol,ycol=("x_pixel","y_pixel") if {"x_pixel","y_pixel"}.issubset(d.columns) else ("x","y") if {"x","y"}.issubset(d.columns) else (None,None)
    if xcol is None: raise KeyError("Для карты нужны x_pixel/y_pixel или x/y")
    d=d.loc[np.isfinite(d[value_col].to_numpy(float))].copy();
    if aggregation!="max": raise ValueError("Поддерживается только aggregation='max'")
    out=np.full((h,w),np.nan,float)
    if d.empty: return out
    z=d.groupby("pixel_id",sort=False).agg({value_col:"max",xcol:"first",ycol:"first"}); xi=z[xcol].to_numpy(int); yi=z[ycol].to_numpy(int)
    if np.any((xi<0)|(xi>=w)|(yi<0)|(yi>=h)): raise ValueError("Координаты вне карты")
    out[yi,xi]=z[value_col].to_numpy(float); return out


def plot_test_statistic_overview(df_obs,*,figsize=(10,4.2)):
    """Распределение T_obs=ΔRSS без noise-rescaling."""
    req={"delta_rss_global","relative_rss_gain"}; miss=sorted(req.difference(df_obs.columns))
    if miss: raise ValueError(f"В df_obs отсутствуют: {miss}")
    d=np.asarray(df_obs["delta_rss_global"],float); r=np.asarray(df_obs["relative_rss_gain"],float)
    good=np.isfinite(d)&(d>0)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=figsize,constrained_layout=True)

    if np.any(good):
        z=np.log10(d[good]); ax1.hist(z,bins=80,alpha=.75)
        p90,p99=np.quantile(d[good],[.9,.99]); ax1.axvline(np.log10(p90),ls="--",lw=1,label=f"P90={p90:.3g}"); ax1.axvline(np.log10(p99),ls=":",lw=1,label=f"P99={p99:.3g}"); ax1.legend(frameon=False)
    else:
        ax1.text(.5,.5,"нет положительных ΔRSS",ha="center",va="center",transform=ax1.transAxes)

    ax1.set_xlabel(r"$\log_{10}(\Delta RSS)$"); ax1.set_ylabel("Count"); ax1.set_title(r"Observed statistic: $T_{obs}=\Delta RSS$")

    ok=np.isfinite(d)&np.isfinite(r)&(d>0)
    if np.any(ok):
        hb=ax2.hexbin(np.log10(d[ok]),r[ok],gridsize=45,mincnt=1,bins="log")
        fig.colorbar(hb,ax=ax2,label="Hypotheses count (log)")

    n_zero=np.count_nonzero(np.isfinite(d)&(d==0))
    if n_zero: ax2.text(.02,.98,f"ΔRSS=0: {n_zero}",transform=ax2.transAxes,ha="left",va="top")

    ax2.set_xlabel(r"$\log_{10}(\Delta RSS)$"); ax2.set_ylabel("relative RSS gain"); ax2.set_title("Fit improvement without sigma rescaling")
    return fig,(ax1,ax2)

def plot_t_obs_map(df_obs,shape_xy,*,figsize=(6.2,5.2),aggregation="max"):
    """Карта T_obs=ΔRSS."""
    tmap = make_t_obs_pixel_map(df_obs,shape_xy,aggregation=aggregation)
    fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True)
    v = tmap[np.isfinite(tmap)&(tmap>0)]
    if len(v):
        vmin = float(np.quantile(v,.05))
        vmax = max(float(np.quantile(v,.995)),vmin*(1+1e-12))
        show = np.where(tmap>0,tmap,np.nan)
        sc = ax.imshow(show,origin="lower",aspect="equal",norm=mcolors.LogNorm(vmin=max(vmin,np.finfo(float).tiny),vmax=vmax),cmap="viridis")
        fig.colorbar(sc,ax=ax,label=r"$T_{obs}=\Delta RSS$")
    else: 
        ax.text(.5,.5,"нет положительных ΔRSS",ha="center",va="center",transform=ax.transAxes)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Предварительная карта $T_{obs}=\\Delta RSS$\nmax по гипотезам в пикселе"); return fig,ax,tmap

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
    fig.colorbar(sc,ax=ax,label="Доля survivor-гипотез в пикселе"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Пространственная концентрация surviving hypotheses"); ax.set_aspect("equal",adjustable="box"); return fig,ax

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
    A,xs,ys=_result_grid(pix,col); im=ax.imshow(A,interpolation="nearest",aspect="equal",origin="lower",extent=(xs.min()-.5,xs.max()+.5,ys.min()-.5,ys.max()+.5),vmin=vmin,vmax=vmax)
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
        ny,nx=A.shape; im=axs[1,1].imshow(A,origin="lower",interpolation="nearest",aspect="equal",extent=(-.5,nx-.5,-.5,ny-.5)); plt.colorbar(im,ax=axs[1,1],fraction=.046,pad=.04,label=physical_map_label)
        d=pix[pix.detected]; axs[1,1].scatter(d.x,d.y,s=12,facecolors="none",edgecolors="k",linewidths=.55)
        axs[1,1].set(title="D  Discoveries over physical map",xlabel="x, pixel",ylabel="y, pixel")
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
    kw = {"interpolation": "nearest", "aspect": "equal", "origin": "lower", "extent": (xs.min()-.5, xs.max()+.5, ys.min()-.5, ys.max()+.5), "vmin": vmin, "vmax": vmax}
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
        cmap = plt.get_cmap("viridis", 4); im = ax.imshow(a, interpolation="nearest", aspect="equal", origin="lower", extent=(xs.min()-.5,xs.max()+.5,ys.min()-.5,ys.max()+.5), cmap=cmap, vmin=-.5, vmax=3.5)
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

