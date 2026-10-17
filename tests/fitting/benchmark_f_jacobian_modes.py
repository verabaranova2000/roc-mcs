from __future__ import annotations
from contextlib import contextmanager
from time import perf_counter
import importlib,numpy as np,pandas as pd

# @contextmanager
# def _patched_factory(enf,base_factory,analytic_factory,jacobian_mode,microbatch_by_K):
#     old=enf.make_jax_lm_solver_v4
#     factory=base_factory if jacobian_mode=="autodiff" else analytic_factory
#     def wrapped(spec,K,*a,**kw):
#         kw["microbatch_size"]=int(microbatch_by_K.get(int(K),kw.get("microbatch_size",4)))
#         return factory(spec,K,*a,**kw)
#     enf.make_jax_lm_solver_v4=wrapped
#     try: yield
#     finally: enf.make_jax_lm_solver_v4=old
    
@contextmanager
def _patched_factory(enf,factory,jacobian_mode,microbatch_by_K):
    old=enf.make_jax_lm_solver_v4
    def wrapped(spec,K,*a,**kw):
        kw["microbatch_size"]=int(microbatch_by_K.get(int(K),kw.get("microbatch_size",4)))
        kw["jacobian_mode"]=str(jacobian_mode)
        return factory(spec,K,*a,**kw)
    enf.make_jax_lm_solver_v4=wrapped
    try: yield
    finally: enf.make_jax_lm_solver_v4=old

def _noise_contract(model):
    if not callable(getattr(model,"simulate_null",None)) or not bool(getattr(getattr(model,"audit",None),"assumption_pass",False)): raise TypeError("Нужен прошедший audit noise model с simulate_null()")
    return model

def _df_e2(e2):
    for k in ("df_D1","df","df_obs"):
        if isinstance(e2,dict) and k in e2:return e2[k]
    raise KeyError("E2 dataframe not found")

def _subset_h(h,idx):
    idx=np.asarray(idx,np.int64); names=("hypothesis_id","K0","pixel_id","x_pixel","y_pixel","parent_index","parent_peak_id","theta_left","theta_right","m0_x","m0_lb","m0_ub","m0_cost","n_valid")
    return type(h)(*(np.asarray(getattr(h,n))[idx].copy() for n in names))

def _select_ids(df,n_ids):
    out=[]; groups=[g for _,g in df.groupby("K0",sort=True)]; per=max(1,int(np.ceil(n_ids/max(1,len(groups)))))
    for g in groups:
        z=g[np.isfinite(g["T_obs"].to_numpy(float))].sort_values(["T_obs","hypothesis_id"])
        if len(z): out.extend(z.iloc[np.unique(np.linspace(0,len(z)-1,min(per,len(z)),dtype=int))]["hypothesis_id"].astype(int).tolist())
    return np.asarray(sorted(set(out))[:int(n_ids)],np.int64)

def _align_h(h,ids):
    pos={int(v):i for i,v in enumerate(np.asarray(h.hypothesis_id,np.int64))}; return _subset_h(h,[pos[int(v)] for v in ids])

def _simulate(rng,data_star,state,mu,noise):
    for K in sorted(state):
        g=state[K]; data_star[g.y,g.x]=np.asarray(noise.simulate_null(rng,mu[K]),float)

def _fit_m0(enf,data_star,theta,state,spec,valid,B,cfg,solvers):
    cfg0=dict(enf.DEFAULT_SOLVER if cfg is None else cfg); cfg1=enf._cfg_retry(cfg0); cfg2=dict(cfg1); cfg2["max_iter"]=2*int(cfg1["max_iter"])
    base=enf.fit_m0_star(data_star,theta,state,spec,valid_yxt=valid,B=B,solver_config=cfg0,solvers=solvers.setdefault("base",{}))
    for tag,cc in (("retry1",cfg1),("retry2",cfg2)):
        bad={int(g["K"]):np.flatnonzero(~np.asarray(g["success"],bool)) for g in base if np.any(~np.asarray(g["success"],bool))}
        if not bad:break
        xo={}
        for K,rows in bad.items():
            gg=next(z for z in base if int(z["K"])==K); cand=np.asarray(gg["x"],float)[rows]; ref=np.asarray(state[K].Xhat,float)[rows]; lb=np.asarray(gg["lb"],float)[rows]; ub=np.asarray(gg["ub"],float)[rows]; feasible=np.all(np.isfinite(cand),1)&np.all(cand>=lb,1)&np.all(cand<=ub,1); xo[K]=np.where(feasible[:,None],cand,ref)
        enf._merge_m0_rows(base,enf.fit_m0_star(data_star,theta,state,spec,valid_yxt=valid,B=B,solver_config=cc,solvers=solvers.setdefault(tag,{}),row_indices=bad,x0_overrides=xo))
    return base

def _one_config(*,enf,diag,d1,e1,e2,m0,data,theta,spec,valid,n_ids,n_replicates,B_fit,seed,fractions,m0_cfg,m1_cfg,factory,jacobian_mode,microbatch_by_K):
    ids=_select_ids(_df_e2(e2),n_ids); hobs=_align_h(d1["hypotheses"],ids); noise=_noise_contract(e1["noise_model"]); state=enf.build_bootstrap_state(m0,target_pixel_ids=np.unique(hobs.pixel_id)); target_source=enf._prepare_target_source(state,hobs); fast=enf._can_use_vectorized_responsibility(hobs,theta,len(spec.param_names)); mu={K:enf.evaluate_composite(spec,theta,g.Xhat) for K,g in state.items()}; data_star=np.empty_like(data,float); m0sol={}; m1sol={}; rows=[]; times=[]
    with _patched_factory(enf,factory,jacobian_mode,microbatch_by_K):
        for rep in range(int(n_replicates)):
            rng=np.random.default_rng(np.random.SeedSequence([int(seed),rep])); _simulate(rng,data_star,state,mu,noise); t0=perf_counter(); m0s=_fit_m0(enf,data_star,theta,state,spec,valid,int(B_fit),m0_cfg,m0sol); t1=perf_counter(); bad=sum(int(np.sum(~np.asarray(g["success"],bool))) for g in m0s)
            if bad: raise RuntimeError(f"M0* hard failures={bad}")
            hs=enf._rebuild_hypotheses_star(hobs,m0s,theta,q0=len(spec.param_names),target_source=target_source,fast_responsibility=fast); t2=perf_counter(); r=enf._fit_m1_best(data_star,theta,hs,spec,valid_yxt=valid,B=int(B_fit),fractions=tuple(fractions),solver_config=m1_cfg,solvers=m1sol,x0_overrides=None,return_x=False); t3=perf_counter(); times.append((t1-t0,t2-t1,t3-t2,t3-t0)); cost=np.asarray(r["cost"],float)
            for hid,c in zip(np.asarray(hs.hypothesis_id,np.int64),cost):rows.append((rep,int(hid),float(c)))
    return {"table":pd.DataFrame(rows,columns=["replicate","hypothesis_id","cost"]),"times":np.asarray(times,float),"n_ids":len(ids)}

def benchmark_f_device_matrix(*,d1_results,e1_results,e2_results,m0_results,data_yxt,theta,spec,valid_yxt=None,configs=None,n_ids=256,n_replicates=3,seed=20261015,fractions=(.25,.5,.75),m0_solver_config=None,m1_solver_config=None,ef_module="empirical_null_EF_v8_noise_plugins",cost_rtol=1e-9,cost_atol=1e-8):
    """CPU/GPU benchmark actual F M0*+M1* workload с проверкой cost-equivalence."""
    import jax
    enf=importlib.import_module(ef_module); base_factory=enf.make_jax_lm_solver_v4
    factory=enf.make_jax_lm_solver_v4
    # analytic_factory=importlib.import_module("jax_pvoigt_analytic_v6").make_jax_lm_solver_pvoigt_analytic_v6
    diag=importlib.import_module("m1_bootstrap_efficiency_diagnostic_v3")
    if configs is None: configs=[{"name":"baseline","B_fit":128,"jacobian":"autodiff","mb":{1:128,2:4,3:4}},{"name":"gpu_mb16","B_fit":256,"jacobian":"autodiff","mb":{1:256,2:16,3:16}},{"name":"gpu_mb64","B_fit":512,"jacobian":"autodiff","mb":{1:512,2:64,3:64}},{"name":"analytic_mb64","B_fit":512,"jacobian":"analytic","mb":{1:512,2:64,3:64}}]
    results={}; summary=[]; ref=None
    print("=== F DEVICE MATRIX ==="); print("backend=",jax.default_backend(),"devices=",jax.devices())
    for cfg in configs:
        name=str(cfg["name"]); out=_one_config(enf=enf,diag=diag,d1=d1_results,e1=e1_results,e2=e2_results,m0=m0_results,data=data_yxt,theta=theta,spec=spec,valid=valid_yxt,n_ids=int(n_ids),n_replicates=int(n_replicates),B_fit=int(cfg["B_fit"]),seed=int(seed),fractions=fractions,m0_cfg=m0_solver_config,m1_cfg=m1_solver_config,factory=factory,jacobian_mode=str(cfg.get("jacobian","autodiff")),microbatch_by_K=dict(cfg["mb"]))
        results[name]=out
        steady=out["times"][1:] if len(out["times"])>1 else out["times"]; med=np.median(steady,axis=0); exact=True; maxdc=0.0
        if ref is None: ref=out
        else:
            a=ref["table"].sort_values(["replicate","hypothesis_id"]).reset_index(drop=True); b=out["table"].sort_values(["replicate","hypothesis_id"]).reset_index(drop=True); d=np.abs(a.cost.to_numpy()-b.cost.to_numpy()); maxdc=float(np.max(d)); exact=bool(np.all(d<=cost_atol+cost_rtol*np.maximum(1.0,np.abs(a.cost.to_numpy()))))
        summary.append({"name":name,"B_fit":int(cfg["B_fit"]),"jacobian":cfg.get("jacobian","autodiff"),"mb1":cfg["mb"].get(1),"mb2":cfg["mb"].get(2),"mb3":cfg["mb"].get(3),"M0_s":med[0],"rebuild_s":med[1],"M1_s":med[2],"total_s":med[3],"speedup_vs_baseline":np.nan,"cost_equiv":exact,"max_abs_dcost":maxdc})
        print(f"{name}: total={med[3]:.4f}s M0={med[0]:.4f}s M1={med[2]:.4f}s equiv={exact} max|dcost|={maxdc:.3e}")
    df=pd.DataFrame(summary); base=float(df.iloc[0].total_s); df["speedup_vs_baseline"]=base/df.total_s; print("\n"+df.to_string(index=False)); return {"summary":df,"runs":results,"backend":jax.default_backend(),"devices":jax.devices()}

def recommended_gpu_configs():
    """Небольшая матрица: сначала искать крупный K2 microbatch, затем B_fit."""
    return [
        {"name":"base_128_mb4","B_fit":128,"jacobian":"autodiff","mb":{1:128,2:4,3:4}},
        {"name":"B256_mb16","B_fit":256,"jacobian":"autodiff","mb":{1:256,2:16,3:16}},
        {"name":"B256_mb64","B_fit":256,"jacobian":"autodiff","mb":{1:256,2:64,3:64}},
        {"name":"B512_mb64","B_fit":512,"jacobian":"autodiff","mb":{1:512,2:64,3:64}},
        {"name":"B512_mb128","B_fit":512,"jacobian":"autodiff","mb":{1:512,2:128,3:128}},
        {"name":"B1024_mb128","B_fit":1024,"jacobian":"autodiff","mb":{1:1024,2:128,3:128}},
        {"name":"B512_mb64_analytic","B_fit":512,"jacobian":"analytic","mb":{1:512,2:64,3:64}},
    ]
