from __future__ import annotations
import numpy as np
import pandas as pd

VERSION="peak_splitting_pipeline_v1"

# ==================== Internal contracts ====================

def _noise_backend():
    from . import noise
    need=("fit_count_noise","print_count_noise_fit","plot_count_noise_fit")
    miss=[name for name in need if not callable(getattr(noise,name,None))]
    if miss: raise ImportError(f"roc_mcs.peak_splitting.noise не содержит {miss}; перенесите актуальный count-noise calibration backend в noise.py")
    return noise

def _validate_stage_inputs(d1_results,e1_results,e2_results):
    for name,obj,keys in (("d1_results",d1_results,("hypotheses",)),("e1_results",e1_results,("noise_model",)),("e2_results",e2_results,("df_D1","df_obs"))):
        if not isinstance(obj,dict) or any(k not in obj for k in keys): raise ValueError(f"{name} не содержит {keys}")
    h=d1_results["hypotheses"]; ids=np.sort(np.asarray(h.hypothesis_id,np.int64))
    for name,df in (("df_D1",e2_results["df_D1"]),("df_obs",e2_results["df_obs"])):
        got=np.sort(df["hypothesis_id"].to_numpy(np.int64))
        if len(got)!=len(ids) or not np.array_equal(got,ids) or df["hypothesis_id"].duplicated().any(): raise AssertionError(f"{name}: hypothesis set mismatch")
    model=e1_results["noise_model"]
    if not bool(getattr(model,"assumption_pass",False)): raise ValueError("Noise model не прошла E1 audit")
    return h,model,e2_results["df_D1"],e2_results["df_obs"]

# ==================== D1 · split-hypothesis family ====================

def run_split_hypothesis_stage(*,data_yxt,theta,peak_catalog,m0_results,spec,valid_yxt=None,B=128,fractions=(.25,.5,.75),progress=True,plot_top_n=6,table_top_n=12,make_plot=True,show=True):
    """D1: M0 → split hypotheses → M1 multistart → ΔRSS → QC."""
    from IPython.display import display
    from . import hypotheses as split_engine
    data_yxt=np.asarray(data_yxt,float); theta=np.asarray(theta,float)
    if data_yxt.ndim!=3 or data_yxt.shape[-1]!=len(theta): raise ValueError("data_yxt должен иметь форму (y,x,theta)")
    if valid_yxt is not None:
        valid_yxt=np.asarray(valid_yxt,bool)
        if valid_yxt.shape!=data_yxt.shape: raise ValueError("valid_yxt и data_yxt должны иметь одинаковую форму")
    if not isinstance(peak_catalog,pd.DataFrame): raise TypeError("peak_catalog должен быть pandas.DataFrame")
    miss={"peak_id","x","y","theta_index","theta"}-set(peak_catalog.columns)
    if miss: raise ValueError(f"В peak_catalog отсутствуют: {sorted(miss)}")
    if not m0_results: raise ValueError("m0_results пуст")
    if tuple(spec.param_names)!=("S","theta0","H","eta"): raise ValueError("D1 требует PVoigt-параметры ('S','theta0','H','eta')")
    pc_map={(int(x),int(y)):g.sort_values("theta_index")["peak_id"].to_numpy(np.int64) for (x,y),g in peak_catalog.groupby(["x","y"],sort=False)}
    m0_n=sum(len(g["pixel_id"]) for g in m0_results); success_parts=[]
    for g in m0_results:
        success=np.asarray(g["success"],bool).ravel(); success_parts.append(success)
        if len(success)!=len(g["pixel_id"]): raise ValueError("M0: success/pixel_id length mismatch")
        ids=np.asarray(g["peak_id"])
        for i,(xp,yp) in enumerate(zip(g["x_pixel"],g["y_pixel"])):
            if not np.array_equal(pc_map.get((int(xp),int(yp))),ids[i]): raise AssertionError(f"M0 peak provenance mismatch at {(int(xp),int(yp))}")
    m0_success=np.concatenate(success_parts); print(f"M0 pixels = {m0_n:,}; success = {m0_success.sum():,}/{m0_n:,}; provenance = OK")
    hypotheses=split_engine.prepare_split_hypotheses(m0_results,theta,param_names=spec.param_names); n=len(hypotheses.hypothesis_id); print(f"D1 hypotheses = {n:,}")
    if not n: return {"hypotheses":hypotheses,"split_fit":None,"best_m1":[],"status_df":pd.DataFrame(),"starts":None,"stats":[],"df":pd.DataFrame(),"fig":None,"axes":None}
    split_fit=split_engine.fit_split_jax(data_yxt=data_yxt,theta=theta,hypotheses=hypotheses,spec=spec,B=B,fractions=fractions,progress=progress,valid_yxt=valid_yxt)
    best_m1=split_fit["best"]; status_df=pd.DataFrame(split_fit["status"]); starts=split_fit["starts"]
    if len(best_m1)!=n: raise RuntimeError(f"M1 дал {len(best_m1)}/{n} решений")
    if "n_success" not in status_df: raise ValueError("В split_fit['status'] нет n_success")
    print("successful starts / hypothesis:",status_df["n_success"].value_counts().sort_index().to_dict())
    stats=split_engine.compute_split_statistics(data_yxt,theta,spec,m0_results,best_m1,sigma2_fn=None,valid_yxt=valid_yxt); df=pd.DataFrame(stats); q=len(spec.param_names)
    if len(df):
        df["rss0"]=2*df["M0_cost"]; df["rss1"]=2*df["M1_cost"]; df["relative_rss_gain"]=df["delta_rss_global"]/np.maximum(df["rss0"],np.finfo(float).tiny)
        df["locality_ratio"]=np.divide(df["delta_rss_local"],df["delta_rss_global"],out=np.full(len(df),np.nan),where=np.abs(df["delta_rss_global"].to_numpy())>0)
        df["child_sep"]=[abs(np.asarray(r.M1_x)[int(r.parent_index)*q+1]-np.asarray(r.M1_x)[int(r.K0)*q+1]) for r in df.itertuples()]
        m0_cost={(int(pid),int(g["K"])):float(np.asarray(g["cost"])[i]) for g in m0_results for i,pid in enumerate(np.asarray(g["pixel_id"]))}; m1_cost={int(r["hypothesis_id"]):float(r["cost"]) for r in best_m1}
        e0=np.array([abs(r.M0_cost-m0_cost[(int(r.pixel_id),int(r.K0))])/max(1,abs(r.M0_cost)) for r in df.itertuples()]); e1=np.array([abs(r.M1_cost-m1_cost[int(r.hypothesis_id)])/max(1,abs(r.M1_cost)) for r in df.itertuples()])
        identity=np.max(np.abs(df["delta_rss_global"]-2*(df["M0_cost"]-df["M1_cost"]))/np.maximum(1,df["rss0"])); tol=1e-8*np.maximum(1,df["rss0"].to_numpy()); neg=df["delta_rss_global"].to_numpy()<-tol; center_bad=[]; param_bad=[]
        for r in df.itertuples():
            X=np.asarray(r.M1_x); a,b=int(r.parent_index)*q,int(r.K0)*q; c1,c2=X[a:a+q],X[b:b+q]; center_bad.append(not (r.theta_left<=c1[1]<=c2[1]<=r.theta_right)); param_bad.append(not (np.isfinite(c1).all() and np.isfinite(c2).all() and c1[0]>=0 and c2[0]>=0 and c1[2]>0 and c2[2]>0 and -1e-10<=c1[3]<=1+1e-10 and -1e-10<=c2[3]<=1+1e-10))
        print(f"M0 rel.err={e0.max():.3e}; M1 rel.err={e1.max():.3e}; ΔRSS identity={identity:.3e}; negative ΔRSS={neg.sum()}; center bad={np.count_nonzero(center_bad)}; param bad={np.count_nonzero(param_bad)}")
        if np.any(center_bad) or np.any(param_bad) or identity>=1e-10: raise AssertionError("D1 numerical validation failed")
        cols=["hypothesis_id","pixel_id","parent_peak_id","K0","best_fraction","delta_rss_global","delta_rss_local","relative_rss_gain","locality_ratio","child_sep","M1_optimality","M1_n_iter"]; display(df.nlargest(table_top_n,"delta_rss_global")[cols])
    fig=axes=None
    if make_plot:
        import matplotlib.pyplot as plt
        from roc_mcs.visualization.peak_splitting import plot_peak_splitting_diagnostics
        fig,axes=plot_peak_splitting_diagnostics(stats_df=df,data_yxt=data_yxt,theta=theta,spec=spec,evaluate_composite=split_engine.evaluate_composite,valid_yxt=valid_yxt,top_n=plot_top_n)
        if fig is not None and show: plt.show()
    print("D1 finished: canonical observed statistic is ΔRSS.")
    return {"hypotheses":hypotheses,"split_fit":split_fit,"best_m1":best_m1,"status_df":status_df,"starts":starts,"stats":stats,"df":df,"fig":fig,"axes":axes}

# ==================== E1 · count-noise model ====================

def run_count_noise_model_stage(*,data_yxt,theta,valid_yxt=None,orders=(2,3,4),primary_order=4,n_bins=12,min_bin=1000,max_contrasts=2_000_000,dependence_limit=.10,fano_cv_limit=.25,curve_order_cv_limit=.35,order_fano_spread_limit=.25,poisson_tol=.30,intercept_ratio_limit=.10,pool_size=1,pool_stride=1,allow_fano_curve=True,require_pass=True,make_plot=True,show=True):
    """E1: маргинальная count-модель с учётом spatial overlap."""
    noise=_noise_backend(); out=noise.fit_count_noise(data_yxt,theta,valid_yxt,orders=orders,primary_order=primary_order,n_bins=n_bins,min_bin=min_bin,max_contrasts=max_contrasts,dependence_limit=dependence_limit,fano_cv_limit=fano_cv_limit,curve_order_cv_limit=curve_order_cv_limit,order_fano_spread_limit=order_fano_spread_limit,poisson_tol=poisson_tol,intercept_ratio_limit=intercept_ratio_limit,pool_size=pool_size,pool_stride=pool_stride,allow_fano_curve=allow_fano_curve)
    noise.print_count_noise_fit(out); fig=axes=None
    if make_plot:
        fig,axes=noise.plot_count_noise_fit(out)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    if require_pass and not out["audit"].assumption_pass: raise RuntimeError(out["audit"].reason)
    return {"noise_model":out["model"],"audit":out["audit"],"tables":out["tables"],"fano_by_order":out["fano_by_order"],"fano_order_spread":out["fano_order_spread"],"rho_intensity_abscontrast":out["rho_intensity_abscontrast"],"intercept_ratio":out["intercept_ratio"],"assumption_pass":bool(out["audit"].assumption_pass),"fig":fig,"axes":axes}

# ==================== E2 · observed statistic ====================

def run_split_noise_bridge(*,d1_results,m0_results,data_yxt,theta,spec,shape_xy=None,valid_yxt=None,top_n=12,make_plot=True,show=True):
    """E2: canonical bridge D1→F; T_obs=ΔRSS без sigma-rescaling."""
    from IPython.display import display
    from . import hypotheses as split_engine
    if not isinstance(d1_results,dict) or any(k not in d1_results for k in ("hypotheses","best_m1")): raise ValueError("d1_results не содержит hypotheses/best_m1")
    hypotheses,best_m1=d1_results["hypotheses"],d1_results["best_m1"]; expected=np.sort(np.asarray(hypotheses.hypothesis_id,np.int64))
    stats=split_engine.compute_split_statistics(data_yxt,theta,spec,m0_results,best_m1,sigma2_fn=None,valid_yxt=valid_yxt); df=pd.DataFrame(stats)
    if len(df)!=len(expected) or not np.array_equal(np.sort(df["hypothesis_id"].to_numpy(np.int64)),expected): raise AssertionError("E2 hypothesis set mismatch")
    if df["hypothesis_id"].duplicated().any(): raise AssertionError("E2 duplicate hypothesis_id")
    df["rss0"]=2*df["M0_cost"]; df["rss1"]=2*df["M1_cost"]; df["relative_rss_gain"]=df["delta_rss_global"]/np.maximum(df["rss0"],np.finfo(float).tiny); df["T_obs"]=df["delta_rss_global"].astype(float)
    T=df["T_obs"].to_numpy(float)
    if np.any(~np.isfinite(T)) or np.any(T<0): raise ValueError("T_obs=ΔRSS должен быть конечным и неотрицательным")
    top=df.nlargest(top_n,"T_obs"); print(f"\nTop {min(top_n,len(df))} hypotheses by ΔRSS:"); display(top[["hypothesis_id","pixel_id","parent_peak_id","K0","delta_rss_global","relative_rss_gain"]])
    fig_overview=axes_overview=fig_map=ax_map=T_obs_map=None
    if make_plot:
        import matplotlib.pyplot as plt
        from roc_mcs.visualization import peak_splitting as rv
        fig_overview,axes_overview=rv.plot_test_statistic_overview(df)
        if show: plt.show()
        if shape_xy is not None:
            fig_map,ax_map,T_obs_map=rv.plot_t_obs_map(df,shape_xy)
            if show: plt.show()
    print("E2 finished: T_obs = ΔRSS; noise model does not rescale the observed statistic.")
    return {"stats":stats,"df_D1":df.copy(),"df_obs":df,"fig_overview":fig_overview,"axes_overview":axes_overview,"fig_map":fig_map,"ax_map":ax_map,"T_obs_map":T_obs_map}

# ==================== Pre-F · noise validation ====================

def run_pre_f_validation_stage(*,e1_results,data_yxt,theta,valid_yxt=None,checkpoint_path="E1_pref_v4.npz",spatial_phase=(0,0),nonoverlap_lag=None,max_pixels=None,bandwidth_points=(.6,.9,1.3,1.8,2.5,3.5,5.0),min_neighbors=2,n_bins=4,tail_probs=(.10,.05,.02,.01),n_reference=199,n_bootstrap=399,family_alpha=.01,max_extrapolated_fraction=.05,max_unsupported_fraction=.05,outer_initial=32,outer_max=128,outer_multiplier=2.,outer_mc_alpha=1e-3,seed=20261030,batch_size=8,run_generator_certification=True,generator_n_draws=200_000,generator_family_alpha=1e-4,progress=True):
    """Pre-F: matched-design prepivot validation перед Stage F."""
    from . import validation
    r=validation.validate_noise_pre_f(data_yxt,theta,e1_results,valid_yxt,spatial_phase=spatial_phase,nonoverlap_lag=nonoverlap_lag,max_pixels=max_pixels,bandwidth_points=bandwidth_points,min_neighbors=min_neighbors,n_bins=n_bins,tail_probs=tail_probs,n_reference=n_reference,n_bootstrap=n_bootstrap,family_alpha=family_alpha,max_extrapolated_fraction=max_extrapolated_fraction,max_unsupported_fraction=max_unsupported_fraction,outer_initial=outer_initial,outer_max=outer_max,outer_multiplier=outer_multiplier,outer_mc_alpha=outer_mc_alpha,checkpoint_path=checkpoint_path,seed=seed,batch_size=batch_size,run_generator_certification=run_generator_certification,generator_n_draws=generator_n_draws,generator_family_alpha=generator_family_alpha,progress=progress)
    validation.print_pre_f(r); return r

# ==================== F · e-process + e-BH ====================

def plan_ebh_horizon(d1_results,*,q=.01,target_discoveries=500,ranks=(50,100,200,500,1000,2000,5000)):
    """Планирует MC horizon B по минимальному числу достижимых discoveries."""
    from . import ebh
    h=d1_results.get("hypotheses") if isinstance(d1_results,dict) else None
    if h is None: raise ValueError("d1_results не содержит hypotheses")
    m=len(np.asarray(h.hypothesis_id)); q=float(q); k0=int(target_discoveries)
    if m<1 or not(0<q<1) or not(1<=k0<=m): raise ValueError("Некорректные m/q/target_discoveries")
    rows=[(int(k),int(ebh.required_B_for_ebh(m,q,int(k)))) for k in ranks if 1<=int(k)<=m]; B=int(ebh.required_B_for_ebh(m,q,k0))
    table=pd.DataFrame(rows,columns=["target_discoveries","B_min"])
    print(f"e-BH family: m={m:,}; q={q:g}; target discoveries={k0:,}; required B={B:,}")
    if len(table): print(table.to_string(index=False))
    return {"m":m,"q":q,"target_discoveries":k0,"B":B,"table":table}

def validate_stage_f_prerequisites(*,e1_results,pre_f_results):
    """Строгий gate перед formal Stage F."""
    model=e1_results.get("noise_model") if isinstance(e1_results,dict) else None; audit=pre_f_results.get("audit") if isinstance(pre_f_results,dict) else None
    if model is None: raise ValueError("e1_results не содержит noise_model")
    if audit is None: raise ValueError("pre_f_results не содержит audit")
    if not bool(getattr(model,"assumption_pass",False)): raise RuntimeError("Stage F запрещён: noise plugin assumption_pass=False")
    if not bool(getattr(audit,"assumption_pass",False)): raise RuntimeError("Stage F запрещён: pre-F noise validation не дала PASS")
    print("PASS: Stage F prerequisites satisfied"); return True

def run_split_empirical_null_ebh(*,d1_results,e1_results,e2_results,m0_results,data_yxt,theta,spec,valid_yxt=None,checkpoint_path="stage_F_ebh.npz",seed=20261004,B=50000,B_fit=128,checkpoint_every=10,fractions=(.25,.5,.75),fdr_q=.01,progress=True):
    """Stage F: aggressive e-process + e-BH."""
    from . import ebh
    hypotheses,noise_model,df_D1,_=_validate_stage_inputs(d1_results,e1_results,e2_results)
    r=ebh.run_empirical_null_ebh(data_yxt,theta,spec,m0_results,hypotheses,df_D1,noise_model,valid_yxt=valid_yxt,B=B,B_fit=B_fit,seed=seed,checkpoint_path=checkpoint_path,checkpoint_every=checkpoint_every,fractions=fractions,fdr_q=fdr_q,progress=progress)
    print("=== STAGE F / e-BH ==="); print(f"controller: {r['mc_controller']}"); print(f"q: {r['fdr_q']:g}"); print(f"survivors: {r['survivor_count']:,}"); print(f"rejected: {len(r['rejected_ids']):,}"); print(f"hypothesis-replicates: {r['total_hypothesis_replicates']:,}"); print("failures:",r["failure_counts"]); return r
