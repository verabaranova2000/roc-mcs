from __future__ import annotations
from pathlib import Path
import os,numpy as np
import roc_mcs.peak_splitting.bootstrap as core

VERSION="EF_v8_ebh_aggressive_v1"

def _require_core():
    v=str(getattr(core,"EF_VERSION","")); need=("_noise_signature","_canonicalize_m0_star","fit_m0_star","_fit_m1_best","_merge_m0_rows","_merge_m1","_subset_hypotheses","_prepare_target_source","_can_use_vectorized_responsibility","_rebuild_hypotheses_star","build_bootstrap_state","evaluate_composite","_simulate_bootstrap_into","_adaptive_run_signature","_cfg_retry")
    miss=[x for x in need if not callable(getattr(core,x,None))]
    if "m0canon" not in v or miss: raise RuntimeError(f"Несовместимый core EF: version={v!r}, missing={miss}")
    return v

def required_B_for_ebh(m,q,min_rejections):
    """Минимальный horizon B, чтобы k одинаковых e=B+1 могли пройти e-BH."""
    m=int(m); k=int(min_rejections); q=float(q)
    if m<1 or k<1 or k>m or not(0<q<1): raise ValueError("Некорректные m/q/min_rejections")
    return int(np.ceil(m/(q*k))-1)

def aggressive_evalue(n,loss):
    """Aggressive e-process: E_t=(t+1) до первого loss, затем 0."""
    n=np.asarray(n,np.int64); loss=np.asarray(loss,np.int64)
    if np.any(n<0) or np.any(loss<0) or np.any(loss>n): raise ValueError("Некорректные e-process counts")
    return np.where(loss==0,n.astype(float)+1.0,0.0)

def ebh_rejected(e,q):
    """Base e-BH; FDR-valid under arbitrary dependence for valid e-values."""
    e=np.asarray(e,float); q=float(q); m=len(e)
    if e.ndim!=1 or m<1 or np.any(~np.isfinite(e)) or np.any(e<0) or not(0<q<1): raise ValueError("Некорректные e-values/q")
    order=np.argsort(-e,kind="mergesort"); es=e[order]; thr=m/(q*np.arange(1,m+1,dtype=float)); ok=es>=thr
    if not np.any(ok): return np.zeros(m,bool)
    k=int(np.flatnonzero(ok)[-1])+1; cutoff=m/(q*k); return e>=cutoff

def _subset_state(state,pixels):
    pix=np.asarray(pixels,np.int64); out={}
    for K,g in state.items():
        m=np.isin(g.pixel_id,pix)
        if np.any(m): out[K]=core.BootstrapGroup(int(K),g.pixel_id[m],g.x[m],g.y[m],g.peak_id[m],g.Xhat[m],g.lb[m],g.ub[m])
    if not out: raise RuntimeError("e-BH: пустой bootstrap state")
    return out

def _save(path,*,ids,T,loss,n_rep,B,q,seed,next_attempt,noise_signature,run_signature,failure_counts):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+f".tmp-{os.getpid()}")
    payload={"version":np.asarray(VERSION),"core_version":np.asarray(str(core.EF_VERSION)),"controller":np.asarray("ebh_aggressive"),"noise_signature":np.asarray(noise_signature),"run_signature":np.asarray(run_signature),"hypothesis_id":np.asarray(ids,np.int64),"T_obs":np.asarray(T,float),"loss_count":np.asarray(loss,np.int64),"n_rep":np.asarray(n_rep,np.int64),"B":np.int64(B),"fdr_q":np.float64(q),"seed":np.int64(seed),"next_attempt":np.int64(next_attempt)}
    payload.update({f"failure_{k}":np.int64(v) for k,v in failure_counts.items()})
    try:
        with open(tmp,"wb") as f: np.savez_compressed(f,**payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass

def _load(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.array(z[k],copy=True) for k in z.files}

def _sample_once(data_star,theta,spec,state,mu_state,target_h,batch,T,noise_model,loss,n_rep,failure_counts,*,B_fit,fractions,valid_yxt,cfg0,cfg0r,cfg0r2,cfg1,cfg1r,m0_solvers,m0_retry_solvers,m0_retry2_solvers,m1_solvers,m1_retry_solvers,rng):
    core._simulate_bootstrap_into(rng,data_star,state,mu_state,noise_model,theta=theta,valid_yxt=valid_yxt)
    m0_star=core.fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0,solvers=m0_solvers)
    bad0={int(g["K"]):np.flatnonzero(~np.asarray(g["success"],bool)) for g in m0_star if np.any(~np.asarray(g["success"],bool))}
    if bad0:
        failure_counts["M0_retry"]+=sum(len(v) for v in bad0.values()); xover={}
        for K,rows in bad0.items():
            gg=next(z for z in m0_star if int(z["K"])==K); cand=np.asarray(gg["x"],float)[rows]; ref=np.asarray(state[K].Xhat,float)[rows]
            feasible=np.all(np.isfinite(cand),1)&np.all(cand>=np.asarray(gg["lb"],float)[rows],1)&np.all(cand<=np.asarray(gg["ub"],float)[rows],1); xover[K]=np.where(feasible[:,None],cand,ref)
        core._merge_m0_rows(m0_star,core.fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0r,solvers=m0_retry_solvers,row_indices=bad0,x0_overrides=xover))
        bad1={int(g["K"]):np.flatnonzero(~np.asarray(g["success"],bool)) for g in m0_star if np.any(~np.asarray(g["success"],bool))}
        if bad1:
            failure_counts["M0_retry2"]+=sum(len(v) for v in bad1.values()); xover2={}
            for K,rows in bad1.items():
                gg=next(z for z in m0_star if int(z["K"])==K); cand=np.asarray(gg["x"],float)[rows]; ref=np.asarray(state[K].Xhat,float)[rows]
                feasible=np.all(np.isfinite(cand),1)&np.all(cand>=np.asarray(gg["lb"],float)[rows],1)&np.all(cand<=np.asarray(gg["ub"],float)[rows],1); xover2[K]=np.where(feasible[:,None],cand,ref)
            core._merge_m0_rows(m0_star,core.fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0r2,solvers=m0_retry2_solvers,row_indices=bad1,x0_overrides=xover2))
    failure_counts["M0_label_swap"]+=core._canonicalize_m0_star(m0_star,len(spec.param_names))
    batch_h=core._subset_hypotheses(target_h,batch); source_all=core._prepare_target_source(state,batch_h); fast_all=core._can_use_vectorized_responsibility(batch_h,theta,len(spec.param_names))
    bad_keys={(int(g["K"]),int(g["pixel_id"][r])) for g in m0_star for r in np.flatnonzero(~np.asarray(g["success"],bool))}
    m0_bad=np.fromiter(((int(target_h.K0[j]),int(target_h.pixel_id[j])) in bad_keys for j in batch),dtype=bool,count=len(batch))
    if np.any(m0_bad): failure_counts["M0_hard"]+=int(np.count_nonzero(m0_bad)); loss[batch[m0_bad]]=1
    good_local=~m0_bad
    if np.any(good_local):
        good=batch[good_local]
        if np.all(good_local): active_h=batch_h; source=source_all; fast=fast_all
        else: active_h=core._subset_hypotheses(target_h,good); source=core._prepare_target_source(state,active_h); fast=core._can_use_vectorized_responsibility(active_h,theta,len(spec.param_names))
        hstar=core._rebuild_hypotheses_star(active_h,m0_star,theta,q0=len(spec.param_names),target_source=source,fast_responsibility=fast)
        m1=core._fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=fractions,solver_config=cfg1,solvers=m1_solvers,return_x=False)
        bad=np.flatnonzero(~np.asarray(m1["success"],bool))
        if len(bad):
            failure_counts["M1_retry"]+=len(bad); cont={int(j):m1["continuation_x"][int(j)] for j in bad if m1["continuation_x"][int(j)] is not None}
            if cont: core._merge_m1(m1,core._fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=(fractions[0],),solver_config=cfg1r,solvers=m1_retry_solvers,hypothesis_idx=np.asarray(sorted(cont),np.int64),x0_overrides=cont,return_x=False))
            bad=np.flatnonzero(~np.asarray(m1["success"],bool))
            if len(bad): core._merge_m1(m1,core._fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=fractions,solver_config=cfg1r,solvers=m1_retry_solvers,hypothesis_idx=bad,include_null_start=False,return_x=False)); bad=np.flatnonzero(~np.asarray(m1["success"],bool))
            if len(bad): failure_counts["M1_hard"]+=len(bad); loss[good[bad]]=1
        ok=np.flatnonzero(np.asarray(m1["success"],bool))
        if len(ok):
            delta=2.0*(np.asarray(hstar.m0_cost,float)[ok]-np.asarray(m1["cost"],float)[ok]); finite=np.isfinite(delta)&(delta>=0)
            if np.any(~finite): failure_counts["delta_unknown"]+=int(np.count_nonzero(~finite)); loss[good[ok[~finite]]]=1
            if np.any(finite):
                loc=ok[finite]; glob=good[loc]; Ts=delta[finite]; loss[glob]|=(Ts>=T[glob])
    n_rep[batch]+=1

def run_empirical_null_ebh(data_yxt,theta,spec,m0_results,hypotheses_obs,df_obs,noise_model,*,valid_yxt=None,B=50000,B_fit=128,seed=20261004,checkpoint_path=None,checkpoint_every=10,m0_solver_config=None,m1_solver_config=None,fractions=core.DEFAULT_FRACTIONS,fdr_q=.01,progress=True):
    """Aggressive e-process + e-BH. Unknown solver outcome считается loss."""
    _require_core(); core._validate_noise_plugin(noise_model)
    data_yxt=np.asarray(data_yxt,float); theta=np.asarray(theta,float); core._validate_valid_mask(valid_yxt,data_yxt.shape); B=int(B); B_fit=int(B_fit); q=float(fdr_q)
    if B<1 or B_fit<1 or not(0<q<1): raise ValueError("Некорректные B/B_fit/fdr_q")
    noise_sig=core._noise_signature(noise_model); df=df_obs.reset_index(drop=True).copy()
    if not {"hypothesis_id","delta_rss_global"}.issubset(df.columns): raise ValueError("df_obs должен содержать hypothesis_id и delta_rss_global")
    ids0=np.asarray(df["hypothesis_id"],np.int64); T0=np.asarray(df["delta_rss_global"],float); hids=np.asarray(hypotheses_obs.hypothesis_id,np.int64)
    if len(np.unique(ids0))!=len(ids0) or len(np.unique(hids))!=len(hids) or not np.array_equal(np.sort(ids0),np.sort(hids)): raise ValueError("e-BH разрешён только для полного D1-семейства")
    if np.any(~np.isfinite(T0)) or np.any(T0<0): raise ValueError("delta_rss_global должен быть конечным и неотрицательным")
    o=np.argsort(ids0,kind="mergesort"); ids=ids0[o]; T=T0[o]; pos={int(x):i for i,x in enumerate(hids)}; idx=np.array([pos[int(x)] for x in ids],np.int64); target_h=core._subset_hypotheses(hypotheses_obs,idx); n=len(ids)
    loss=np.zeros(n,np.int8); n_rep=np.zeros(n,np.int64); next_attempt=0; cp=Path(checkpoint_path) if checkpoint_path else None
    failure_counts={"M0_retry":0,"M0_retry2":0,"M0_hard":0,"M0_label_swap":0,"M1_retry":0,"M1_hard":0,"delta_unknown":0}
    run_sig=core._adaptive_run_signature(data_yxt,theta,valid_yxt,m0_results,hypotheses_obs,spec,B_fit=B_fit,fractions=fractions,m0_solver_config=m0_solver_config,m1_solver_config=m1_solver_config,noise_signature=noise_sig)
    if cp and cp.exists():
        d=_load(cp); req={"version","controller","noise_signature","run_signature","hypothesis_id","T_obs","loss_count","n_rep","B","fdr_q","seed","next_attempt"}; miss=sorted(req.difference(d))
        if miss: raise ValueError(f"e-BH checkpoint неполный: {miss}")
        if str(d["version"])!=VERSION or str(d["controller"])!="ebh_aggressive" or str(d["noise_signature"])!=noise_sig or str(d["run_signature"])!=run_sig: raise ValueError("e-BH checkpoint signature mismatch")
        if not np.array_equal(np.asarray(d["hypothesis_id"],np.int64),ids) or not np.allclose(np.asarray(d["T_obs"],float),T,rtol=1e-12,atol=0): raise ValueError("e-BH checkpoint hypothesis/T mismatch")
        if int(d["B"])>B or not np.isclose(float(d["fdr_q"]),q) or int(d["seed"])!=int(seed): raise ValueError("e-BH checkpoint configuration mismatch")
        loss=np.asarray(d["loss_count"],np.int8); n_rep=np.asarray(d["n_rep"],np.int64); next_attempt=int(d["next_attempt"])
        if loss.shape!=(n,) or n_rep.shape!=(n,) or np.any((loss<0)|(loss>1)) or np.any(n_rep<0): raise ValueError("e-BH checkpoint state invalid")
        for k in failure_counts:
            if f"failure_{k}" in d: failure_counts[k]=int(d[f"failure_{k}"])
    cfg0=core.DEFAULT_SOLVER if m0_solver_config is None else dict(m0_solver_config); cfg1=core.DEFAULT_SOLVER if m1_solver_config is None else dict(m1_solver_config); cfg0r=core._cfg_retry(cfg0); cfg0r2=dict(cfg0r); cfg0r2["max_iter"]=2*int(cfg0r["max_iter"]); cfg1r=core._cfg_retry(cfg1)
    m0_solvers={}; m0_retry_solvers={}; m0_retry2_solvers={}; m1_solvers={}; m1_retry_solvers={}
    full_state=core.build_bootstrap_state(m0_results,target_pixel_ids=np.unique(target_h.pixel_id)); data_star=np.empty_like(data_yxt,float)
    pbar=None
    if progress:
        from tqdm.auto import tqdm
        pbar=tqdm(total=None,initial=next_attempt,desc="e-BH empirical null v8",unit="rep",mininterval=.5)
    try:
        while True:
            active=np.flatnonzero((loss==0)&(n_rep<B))
            if not len(active): break
            state=_subset_state(full_state,np.unique(target_h.pixel_id[active])); mu_state={K:core.evaluate_composite(spec,theta,g.Xhat) for K,g in state.items()}
            attempt=next_attempt; rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(attempt)]))
            _sample_once(data_star,theta,spec,state,mu_state,target_h,active,T,noise_model,loss,n_rep,failure_counts,B_fit=B_fit,fractions=fractions,valid_yxt=valid_yxt,cfg0=cfg0,cfg0r=cfg0r,cfg0r2=cfg0r2,cfg1=cfg1,cfg1r=cfg1r,m0_solvers=m0_solvers,m0_retry_solvers=m0_retry_solvers,m0_retry2_solvers=m0_retry2_solvers,m1_solvers=m1_solvers,m1_retry_solvers=m1_retry_solvers,rng=rng)
            next_attempt=attempt+1
            if cp and checkpoint_every>0 and next_attempt%int(checkpoint_every)==0: _save(cp,ids=ids,T=T,loss=loss,n_rep=n_rep,B=B,q=q,seed=seed,next_attempt=next_attempt,noise_signature=noise_sig,run_signature=run_sig,failure_counts=failure_counts)
            if pbar is not None:
                e=aggressive_evalue(n_rep,loss); r=ebh_rejected(e,q); pbar.update(1); pbar.set_postfix(active=int(np.sum((loss==0)&(n_rep<B))),survivors=int(np.sum(loss==0)),reject=int(np.sum(r)),maxB=int(n_rep.max()),hyprep=int(n_rep.sum()))
    finally:
        if pbar is not None: pbar.close()
    e=aggressive_evalue(n_rep,loss); rejected=ebh_rejected(e,q); status=np.where(rejected,1,2).astype(np.int8); labels=np.where(rejected,"rejected","nonreject")
    if cp: _save(cp,ids=ids,T=T,loss=loss,n_rep=n_rep,B=B,q=q,seed=seed,next_attempt=next_attempt,noise_signature=noise_sig,run_signature=run_sig,failure_counts=failure_counts)
    return {"B":B,"attempts":next_attempt,"test_statistic":"delta_rss","noise_model_kind":noise_model.kind,"noise_signature":noise_sig,"run_signature":run_sig,"hypothesis_id":ids,"family_ids":ids.copy(),"T_obs":T,"delta_rss_obs":T.copy(),"loss_count":loss,"n_bootstrap":n_rep,"e_value":e,"status":status,"status_label":labels,"rejected_ids":ids[rejected],"nonreject_ids":ids[~rejected],"fdr_q":q,"fdr_method":"e-BH","mc_controller":"ebh_aggressive","complete":True,"family_complete":True,"failure_counts":failure_counts,"family_m":n,"survivor_count":int(np.sum(loss==0)),"total_hypothesis_replicates":int(np.sum(n_rep)),"mean_hypothesis_replicates":float(np.mean(n_rep)),"max_hypothesis_replicates":int(np.max(n_rep)),"formal_fdr_bound":q,"mc_resampling_risk_bound":None}
