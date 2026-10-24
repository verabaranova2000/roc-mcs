"""Stage E/F v8: parametric empirical-null bootstrap with explicit noise calibration.

Каноническая тестовая статистика: T = ΔRSS = RSS(M0)-RSS(M1).
Noise model оценивается отдельно и используется только для генерации bootstrap data.
"""
from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import os, json, hashlib
import numpy as np
import jax.numpy as jnp
from scipy.stats import beta as beta_dist

from roc_mcs.fitting.backends.jax_backend import make_jax_lm_solver_v4
from roc_mcs.peak_splitting.hypotheses import SplitHypotheses,responsibility_interval,iter_split_batches,pad_split_batch,canonicalize_split_x,make_split_null_x
# Noise plugin используется через явный runtime-contract; class identity после reload не требуется.

DEFAULT_FRACTIONS=(0.25,0.50,0.75)
DEFAULT_SOLVER={"max_iter":300,"ftol":1e-9,"xtol":1e-9,"gtol":1e-8}
EF_VERSION="EF_v8_deltaRSS_noise_plugins_sequential_m0canon_v3_mmctest"



def _validate_noise_plugin(model):
    kind=getattr(model,"kind",None); scope=getattr(model,"scope",None)
    if not isinstance(kind,str) or not kind or scope not in {"independent","marginal"} or not callable(getattr(model,"simulate_null",None)) or not callable(getattr(model,"signature",None)): raise TypeError("noise_model должен реализовывать plugin contract: kind/scope/simulate_null/signature")
    if not bool(getattr(model,"assumption_pass",False)): raise ValueError("noise_model не прошёл calibration audit")

def _noise_signature(model):
    _validate_noise_plugin(model); sig=model.signature()
    if not isinstance(sig,str) or not sig: raise TypeError("noise_model.signature() должен возвращать непустую строку")
    return sig

def _simulate_bootstrap_into(rng,data_star,state,mu_state,model,*,theta=None,valid_yxt=None):
    """Генерация null через единый plugin-contract."""
    _validate_noise_plugin(model)
    for K in sorted(state):
        g=state[K]; mu=np.asarray(mu_state[K],float); y=np.asarray(model.simulate_null(rng,mu),float)
        if y.shape!=mu.shape or np.any(~np.isfinite(y)): raise RuntimeError(f"noise_model вернул некорректный sample для K={K}: shape={y.shape}, expected={mu.shape}")
        data_star[g.y,g.x]=y
    return data_star

def _validate_valid_mask(valid_yxt, shape):
    if valid_yxt is None:
        return
    v = np.asarray(valid_yxt, bool)
    Ny, Nx, M = shape
    if v.shape not in ((Ny, Nx), (Ny, Nx, M)):
        raise ValueError(f"valid_yxt shape {v.shape} incompatible with {(Ny, Nx, M)}")

def _rows_valid(obs,y,x,theta,valid_yxt):
    v=np.isfinite(obs)&np.isfinite(theta)[None,:]
    if valid_yxt is None: return v
    vm=np.asarray(valid_yxt,bool)
    if vm.ndim==2: v&=vm[y,x,None]
    elif vm.ndim==3: v&=vm[y,x]
    else: raise ValueError("valid_yxt must be 2D or 3D")
    return v

def evaluate_composite(spec, theta, X):
    """Векторизованное NumPy-предсказание composite model."""
    X = np.asarray(X, float); q = len(spec.param_names); K = (X.shape[1] - 1) // q
    comp = X[:, :K*q].reshape(len(X), K, q); out = np.zeros((len(X), len(theta)), float)
    for k in range(K): out += np.asarray(spec.batch_func(theta, *[comp[:, k, j] for j in range(q)]))
    return out + X[:, -1, None]

@dataclass(frozen=True, slots=True)
class BootstrapGroup:
    K:int; pixel_id:np.ndarray; x:np.ndarray; y:np.ndarray; peak_id:np.ndarray; Xhat:np.ndarray; lb:np.ndarray; ub:np.ndarray

def build_bootstrap_state(m0_results,*,target_pixel_ids):
    """Собирает fitted M0 для пикселей активных гипотез."""
    wanted=set(map(int,np.asarray(target_pixel_ids,np.int64))); groups={}; seen=set()
    if not wanted: raise ValueError("target_pixel_ids пуст")
    for g in m0_results:
        K=int(g["K"]); pid=np.asarray(g["pixel_id"],np.int64); ok=np.asarray(g["success"],bool); cost=np.asarray(g["cost"],float)
        X=np.asarray(g["x"],float); lb=np.asarray(g["lb"],float); ub=np.asarray(g["ub"],float); xp=np.asarray(g["x_pixel"],int); yp=np.asarray(g["y_pixel"],int); pk=np.asarray(g["peak_id"],np.int64)
        keep=ok&np.isfinite(cost)&np.asarray([int(p) in wanted for p in pid])
        if not np.any(keep): continue
        selected=set(map(int,pid[keep])); overlap=seen.intersection(selected)
        if overlap: raise ValueError(f"pixel_id duplicated in bootstrap state: {sorted(overlap)[:10]}")
        seen.update(selected); groups.setdefault(K,[]).append(BootstrapGroup(K,pid[keep].copy(),xp[keep].copy(),yp[keep].copy(),pk[keep].copy(),X[keep].copy(),lb[keep].copy(),ub[keep].copy()))
    missing=wanted-seen
    if missing: raise ValueError(f"Нет успешного M0 fit для {len(missing)} target pixels; первые={sorted(missing)[:10]}")
    out={}
    for K,pp in groups.items():
        out[K]=BootstrapGroup(K,np.concatenate([p.pixel_id for p in pp]),np.concatenate([p.x for p in pp]),np.concatenate([p.y for p in pp]),
                              np.concatenate([p.peak_id for p in pp]),np.vstack([p.Xhat for p in pp]),np.vstack([p.lb for p in pp]),np.vstack([p.ub for p in pp]))
    if not out: raise ValueError("Не найдено M0-пикселей для bootstrap")
    for g in out.values():
        if len(np.unique(g.pixel_id))!=len(g.pixel_id): raise ValueError("pixel_id duplicated in bootstrap state")
    return out


def _pad_rows(obs,valid,x0,lb,ub,target_B):
    n=len(obs)
    if n==target_B: return obs,valid,x0,lb,ub,n
    if n>target_B: raise ValueError("rows > target_B")
    p=target_B-n; rep=lambda a:np.repeat(a[-1:],p,axis=0)
    return np.vstack((obs,np.zeros((p,obs.shape[1])))),np.vstack((valid,np.zeros((p,valid.shape[1]),bool))),np.vstack((x0,rep(x0))),np.vstack((lb,rep(lb))),np.vstack((ub,rep(ub))),n

def fit_m0_star(data_star,theta,state,spec,*,valid_yxt=None,B=128,solver_config=None,solvers=None,row_indices=None,x0_overrides=None):
    """M0*; можно считать только выбранные строки, при retry — с continuation-x."""
    cfg=DEFAULT_SOLVER if solver_config is None else solver_config; solvers={} if solvers is None else solvers; theta=np.asarray(theta,float); theta_j=jnp.asarray(theta,jnp.float64); parts={}
    for K in sorted(state):
        g=state[K]; N=len(g.pixel_id); ridx=np.arange(N,dtype=np.int64) if row_indices is None else np.asarray(row_indices.get(K,()),np.int64)
        if ridx.ndim!=1 or (len(ridx) and (ridx.min()<0 or ridx.max()>=N or len(np.unique(ridx))!=len(ridx))): raise ValueError(f"Некорректные row_indices для K={K}")
        if not len(ridx): continue
        parts[K]=[]
        if K not in solvers: solvers[K],_,_,_=make_jax_lm_solver_v4(spec,K,max_iter=cfg["max_iter"],ftol=cfg["ftol"],xtol=cfg["xtol"],gtol=cfg["gtol"])
        xo=None if x0_overrides is None else np.asarray(x0_overrides[int(K)],float)
        if xo is not None and xo.shape!=(len(ridx),g.Xhat.shape[1]): raise ValueError(f"Некорректный x0_overrides для K={K}")
        for start in range(0,len(ridx),B):
            jj=ridx[start:min(start+B,len(ridx))]; obs=np.asarray(data_star[g.y[jj],g.x[jj]],float); valid=_rows_valid(obs,g.y[jj],g.x[jj],theta,valid_yxt); x0=g.Xhat[jj].copy() if xo is None else xo[start:min(start+B,len(ridx))].copy(); lb=g.lb[jj]; ub=g.ub[jj]; obs,valid,x0,lb,ub,n=_pad_rows(obs,valid,x0,lb,ub,B); o=solvers[K](jnp.asarray(obs),jnp.asarray(valid),jnp.asarray(x0),jnp.asarray(lb),jnp.asarray(ub),theta_j); z=slice(0,n)
            parts[K].append({"K":K,"x":np.asarray(o["x"])[z],"cost":np.asarray(o["cost"])[z],"success":np.asarray(o["success"])[z],"optimality":np.asarray(o["optimality"])[z],"n_iter":np.asarray(o["n_iter"])[z],"convergence_reason":np.asarray(o["convergence_reason"])[z],"pixel_id":g.pixel_id[jj],"x_pixel":g.x[jj],"y_pixel":g.y[jj],"peak_id":g.peak_id[jj],"lb":g.lb[jj],"ub":g.ub[jj],"n_valid":np.asarray(valid.sum(axis=1)[:n],np.int32),"row_index":jj})
    out=[]
    for K in sorted(parts):
        ps=parts[K]; z={k:np.concatenate([p[k] for p in ps],axis=0) if k not in ("K",) else K for k in ps[0]}; g=state[K]
        if row_indices is None and (len(z["pixel_id"])!=len(g.pixel_id) or not np.array_equal(z["pixel_id"],g.pixel_id)): raise RuntimeError(f"M0* group K={K} order/length mismatch")
        if not np.array_equal(z["pixel_id"],g.pixel_id[z["row_index"]]): raise RuntimeError(f"M0* selected order mismatch K={K}")
        out.append(z)
    return out

def _merge_m0_rows(base,extra):
    byK={int(g["K"]):g for g in base}
    for eg in extra:
        bg=byK[int(eg["K"])]
        rows=np.asarray(eg["row_index"],np.int64)
        for key in ("x","cost","success","optimality","n_iter","convergence_reason","pixel_id","x_pixel","y_pixel","peak_id","lb","ub","n_valid"): bg[key][rows]=eg[key]
    return base


def _canonicalize_m0_star(m0_star,q0):
    """Канонизирует M0* по возрастанию центра; fitted curve/cost не меняются."""
    q0=int(q0); swapped=0
    if q0<2: raise ValueError("Для канонизации M0* нужен параметр центра с индексом 1")
    for g in m0_star:
        K=int(g["K"]); X=np.asarray(g["x"],float); LB=np.asarray(g["lb"],float); UB=np.asarray(g["ub"],float); ok=np.asarray(g["success"],bool)
        need=K*q0+1
        if X.ndim!=2 or X.shape[1]!=need or LB.shape!=X.shape or UB.shape!=X.shape or ok.shape!=(len(X),): raise RuntimeError(f"Некорректная M0* group shape для K={K}")
        if K<=1: continue
        for r in np.flatnonzero(ok):
            centers=X[r,1:K*q0:q0]
            if np.any(~np.isfinite(centers)): ok[r]=False; continue
            order=np.argsort(centers,kind="mergesort")
            if np.array_equal(order,np.arange(K)): continue
            Xc=X[r,:K*q0].reshape(K,q0).copy(); LBc=LB[r,:K*q0].reshape(K,q0).copy(); UBc=UB[r,:K*q0].reshape(K,q0).copy()
            X[r,:K*q0]=Xc[order].reshape(-1); LB[r,:K*q0]=LBc[order].reshape(-1); UB[r,:K*q0]=UBc[order].reshape(-1); swapped+=1
        g["x"]=X; g["lb"]=LB; g["ub"]=UB; g["success"]=ok
        good=np.flatnonzero(ok)
        if len(good):
            C=X[good,1:K*q0:q0]
            if np.any(np.diff(C,axis=1)<0) or np.any(X[good]<LB[good]) or np.any(X[good]>UB[good]): raise RuntimeError(f"M0* canonicalization invariant failed for K={K}")
    return swapped

def _prepare_target_source(state,target_h):
    out={}
    for K in np.unique(target_h.K0):
        K=int(K); rows=np.flatnonzero(target_h.K0==K); g=state.get(K)
        if g is None: raise RuntimeError(f"Bootstrap M0 group K={K} missing")
        order=np.argsort(g.pixel_id,kind="mergesort"); sp=np.asarray(g.pixel_id)[order]; want=np.asarray(target_h.pixel_id)[rows]; loc=np.searchsorted(sp,want)
        if np.any(loc>=len(sp)) or np.any(sp[np.minimum(loc,len(sp)-1)]!=want): raise RuntimeError("Bootstrap M0 missing target pixel")
        out[K]=(rows,order[loc])
    return out

def _responsibility_midpoints(X,parent,K,theta,q0):
    X=np.asarray(X,float); parent=np.asarray(parent,np.int64); K=int(K); q0=int(q0)
    if X.ndim!=2 or X.shape[1]<K*q0+1 or parent.shape!=(len(X),) or np.any(parent<0) or np.any(parent>=K): raise ValueError("Некорректные параметры для интервалов ответственности")
    centers=X[:,1:K*q0:q0]; j=np.arange(len(X)); c=centers[j,parent]; lo=np.full(len(X),float(theta[0])); hi=np.full(len(X),float(theta[-1])); m=parent>0; lo[m]=0.5*(centers[j[m],parent[m]-1]+c[m]); m=parent<K-1; hi[m]=0.5*(c[m]+centers[j[m],parent[m]+1]); return lo,hi

def _can_use_vectorized_responsibility(h,theta,q0):
    for K in np.unique(h.K0):
        K=int(K); rows=np.flatnonzero(h.K0==K); X=np.asarray(h.m0_x[rows],float); parent=np.asarray(h.parent_index[rows],np.int64)
        if np.any(parent<0)|np.any(parent>=K): return False
        lo,hi=_responsibility_midpoints(X,parent,K,theta,q0)
        if not np.allclose(lo,h.theta_left[rows],rtol=1e-12,atol=1e-12) or not np.allclose(hi,h.theta_right[rows],rtol=1e-12,atol=1e-12): return False
    return True

def _rebuild_hypotheses_star(h_obs,m0_star,theta,*,q0,target_source=None,fast_responsibility=True):
    theta=np.asarray(theta,float); q0=int(q0); groups={int(g["K"]):g for g in m0_star}; n=len(h_obs.hypothesis_id); Xout=np.full_like(h_obs.m0_x,np.nan,dtype=float); lbout=np.full_like(h_obs.m0_lb,np.nan,dtype=float); ubout=np.full_like(h_obs.m0_ub,np.nan,dtype=float); cout=np.empty(n,float); nvout=np.empty(n,np.int32); loout=np.empty(n,float); hiout=np.empty(n,float)
    if target_source is None: target_source=_prepare_target_source({K:g for K,g in groups.items()},h_obs)
    for K in sorted(target_source):
        K=int(K); rows,src=target_source[K]; g=groups[K]; success=np.asarray(g["success"],bool)[src]; need=K*q0+1
        if not np.all(success): raise RuntimeError("Bootstrap M0 failed for target pixel")
        X=np.asarray(g["x"],float)[src]; lb=np.asarray(g["lb"],float)[src]; ub=np.asarray(g["ub"],float)[src]
        if X.shape!=(len(rows),need) or lb.shape!=X.shape or ub.shape!=X.shape or need>Xout.shape[1]: raise RuntimeError(f"M0* shapes mismatch for K={K}: x={X.shape}, expected {(len(rows),need)}")
        parent=np.asarray(h_obs.parent_index[rows],np.int64); Xout[rows,:need]=X; lbout[rows,:need]=lb; ubout[rows,:need]=ub; cout[rows]=np.asarray(g["cost"],float)[src]; nvout[rows]=np.asarray(g["n_valid"],np.int32)[src]
        if fast_responsibility: loout[rows],hiout[rows]=_responsibility_midpoints(X,parent,K,theta,q0)
        else:
            for j,r in enumerate(rows):
                centers=X[j,1:K*q0:q0]; loout[r],hiout[r]=responsibility_interval(centers,int(parent[j]),float(theta[0]),float(theta[-1]))
    return SplitHypotheses(h_obs.hypothesis_id.copy(),h_obs.K0.copy(),h_obs.pixel_id.copy(),h_obs.x_pixel.copy(),h_obs.y_pixel.copy(),h_obs.parent_index.copy(),h_obs.parent_peak_id.copy(),loout,hiout,Xout,lbout,ubout,cout,nvout)

def _fit_m1_best(data_star,theta,h,spec,*,valid_yxt=None,B=128,fractions=DEFAULT_FRACTIONS,solver_config=None,solvers=None,hypothesis_idx=None,x0_overrides=None,include_null_start=False,return_x=True):
    """M1 multistart; retry может брать только unresolved-гипотезы и continuation-x."""
    cfg=DEFAULT_SOLVER if solver_config is None else solver_config; solvers={} if solvers is None else solvers; N=len(h.hypothesis_id); q0=len(spec.param_names); theta_j=jnp.asarray(theta,jnp.float64); sel=np.arange(N,dtype=np.int64) if hypothesis_idx is None else np.asarray(hypothesis_idx,np.int64)
    if sel.ndim!=1 or (len(sel) and (sel.min()<0 or sel.max()>=N or len(np.unique(sel))!=len(sel))): raise ValueError("Некорректный hypothesis_idx")
    best_cost=np.asarray(h.m0_cost,float).copy(); best_success=np.zeros(N,bool); best_opt=np.full(N,np.nan); best_iter=np.zeros(N,np.int32); best_reason=np.full(N,-1,np.int8); best_frac=np.full(N,np.nan); selected_null=np.ones(N,bool); groups={}; best_x_by_K={}; continuation_x=[None]*N; cont_cost=np.full(N,np.inf); last_cost=np.full(N,np.nan); last_opt=np.full(N,np.nan); last_iter=np.zeros(N,np.int32); last_reason=np.full(N,-1,np.int8); reason_counts=np.zeros((N,5),np.int16)
    for K0 in sorted(np.unique(h.K0[sel]).tolist()):
        K0=int(K0); idx_all=np.flatnonzero(h.K0==K0); idx=idx_all[np.isin(idx_all,sel)]; groups[K0]=idx; sub=SplitHypotheses(np.arange(len(idx),dtype=np.int64),h.K0[idx],h.pixel_id[idx],h.x_pixel[idx],h.y_pixel[idx],h.parent_index[idx],h.parent_peak_id[idx],h.theta_left[idx],h.theta_right[idx],h.m0_x[idx],h.m0_lb[idx],h.m0_ub[idx],h.m0_cost[idx],h.n_valid[idx]); best_x_by_K[K0]=(np.stack([make_split_null_x(sub,local,spec.param_names) for local in range(len(idx))]) if len(idx) else np.empty((0,(K0+1)*q0+1))) if return_x else None
        for batch in iter_split_batches(data_star,theta,sub,param_names=spec.param_names,fractions=fractions,max_batch=B,valid_yxt=valid_yxt,include_null_start=include_null_start):
            bp,n=pad_split_batch(batch,B); K1=int(bp.K1); 
            if K1 not in solvers: solvers[K1],_,_,_=make_jax_lm_solver_v4(spec,K1,max_iter=cfg["max_iter"],ftol=cfg["ftol"],xtol=cfg["xtol"],gtol=cfg["gtol"])
            hid=np.asarray(bp.hypothesis_id)[:n]; glob=idx[hid]; x0=np.asarray(bp.x0,float).copy()
            if x0_overrides is not None:
                for j,gidx in enumerate(glob):
                    xx=x0_overrides.get(int(gidx)) if hasattr(x0_overrides,"get") else None
                    if xx is not None: x0[j]=np.asarray(xx,float)
            o=solvers[K1](jnp.asarray(bp.obs),jnp.asarray(bp.valid),jnp.asarray(x0),jnp.asarray(bp.lb),jnp.asarray(bp.ub),theta_j); s=slice(0,n); success=np.asarray(o["success"])[s]; cost=np.asarray(o["cost"])[s]; X1=np.asarray(o["x"])[s]; opt=np.asarray(o["optimality"])[s]; it=np.asarray(o["n_iter"])[s]; reason=np.asarray(o["convergence_reason"])[s]; last_cost[glob]=cost; last_opt[glob]=opt; last_iter[glob]=it; last_reason[glob]=reason
            lb1=np.asarray(bp.lb)[s]; ub1=np.asarray(bp.ub)[s]; feasible=np.all(np.isfinite(X1),axis=1)&np.all(X1>=lb1,axis=1)&np.all(X1<=ub1,axis=1)&np.isfinite(cost)
            for j,gidx in enumerate(glob):
                if feasible[j] and cost[j]<cont_cost[int(gidx)]: cont_cost[int(gidx)]=cost[j]; continuation_x[int(gidx)]=X1[j].copy()
            for rr in range(5):
                m=reason==rr
                if np.any(m): np.add.at(reason_counts[:,rr],glob[m],1)
            ok=success & feasible
            if not np.any(ok): continue
            glob_all=glob[ok]; best_success[glob_all]=True; p=np.flatnonzero(ok); order=np.lexsort((cost[p],hid[p])); p=p[order]; p=p[np.r_[True,hid[p[1:]]!=hid[p[:-1]]]]; globp=idx[hid[p]]; improved=cost[p]<best_cost[globp]
            if np.any(improved):
                p=p[improved]; globp=globp[improved]; best_cost[globp]=cost[p];
                if return_x: best_x_by_K[K0][hid[p]]=X1[p]
                best_opt[globp]=opt[p]; best_iter[globp]=it[p]; best_reason[globp]=reason[p]; best_frac[globp]=np.asarray(bp.fraction)[s][p]; selected_null[globp]=False
    best_x=[None]*N
    if return_x:
        for K0,idx in groups.items():
            for local,glob in enumerate(idx):
                if best_success[glob]: best_x[int(glob)]=canonicalize_split_x(best_x_by_K[K0][local],K0,int(h.parent_index[glob]),spec.param_names)
    return {"success":best_success,"cost":best_cost,"x":best_x,"optimality":best_opt,"n_iter":best_iter,"convergence_reason":best_reason,"fraction":best_frac,"selected_null":selected_null,"reason_counts":reason_counts,"last_cost":last_cost,"last_optimality":last_opt,"last_n_iter":last_iter,"last_reason":last_reason,"continuation_x":continuation_x}

def _merge_m1(base,extra):
    take=np.flatnonzero(np.asarray(extra["success"],bool))
    for key in ("success","cost","optimality","n_iter","convergence_reason","fraction","selected_null"): base[key][take]=extra[key][take]
    for i in take: base["x"][int(i)]=extra["x"][int(i)]
    base["reason_counts"]+=extra["reason_counts"]; base["last_cost"][take]=extra["last_cost"][take]; base["last_optimality"][take]=extra["last_optimality"][take]; base["last_n_iter"][take]=extra["last_n_iter"][take]; base["last_reason"][take]=extra["last_reason"][take]
    for i in range(len(base["continuation_x"])):
        if extra["continuation_x"][i] is not None: base["continuation_x"][i]=extra["continuation_x"][i]
    return base

def empirical_pvalues(T_obs,ge_count,B):
    """p=(1+число превышений)/(B+1)."""
    T_obs=np.asarray(T_obs,float); ge_count=np.asarray(ge_count,np.int64); B=np.asarray(B,np.int64)
    if np.any(B<1) or np.any(ge_count<0) or np.any(ge_count>B): raise ValueError("Некорректные bootstrap counts")
    return (1.0+ge_count)/(B+1.0)

def empirical_pvalue_bounds(ge_count,unknown_count,n_attempt):
    """Границы p при неизвестных исходах отдельных MC-реплик."""
    ge=np.asarray(ge_count,np.int64); unk=np.asarray(unknown_count,np.int64); n=np.asarray(n_attempt,np.int64)
    if np.any(n<1) or np.any(ge<0) or np.any(unk<0) or np.any(ge+unk>n): raise ValueError("Некорректные MC counts")
    return (1.0+ge)/(n+1.0),(1.0+ge+unk)/(n+1.0)

def _cfg_retry(cfg):
    """Фиксированный retry-бюджет без изменения критериев сходимости."""
    c=dict(DEFAULT_SOLVER if cfg is None else cfg); c["max_iter"]=int(4*c["max_iter"]); return c

def _look_schedule(max_B,initial_B,look_multiplier):
    max_B=int(max_B); b=min(max_B,int(initial_B)); out=[]
    while b<=max_B:
        out.append(b)
        if b==max_B: break
        b=min(max_B,max(b+1,int(np.ceil(b*look_multiplier))))
    return tuple(out)

def _binomial_cp_bounds(k,n,alpha):
    k=np.asarray(k,np.int64); n=np.asarray(n,np.int64); a=np.broadcast_to(np.asarray(alpha,float),k.shape)
    if np.any(n<1) or np.any(k<0) or np.any(k>n) or np.any(~np.isfinite(a)) or np.any((a<=0)|(a>=.5)): raise ValueError("Некорректные binomial counts/alpha")
    lo=np.zeros(k.shape,float); hi=np.ones(k.shape,float); m=k>0; lo[m]=beta_dist.ppf(a[m],k[m],n[m]-k[m]+1); m=k<n; hi[m]=beta_dist.ppf(1.0-a[m],k[m]+1,n[m]-k[m]); return lo,hi

def _mc_cp_bounds(ge,unknown,n,alpha):
    """CP-границы, если unknown исходов дают K∈[ge, ge+unknown]."""
    ge=np.asarray(ge,np.int64); unknown=np.asarray(unknown,np.int64); n=np.asarray(n,np.int64)
    if np.any(n<1) or np.any(ge<0) or np.any(unknown<0) or np.any(ge+unknown>n): raise ValueError("Некорректные MC counts")
    lo,_=_binomial_cp_bounds(ge,n,alpha); _,hi=_binomial_cp_bounds(ge+unknown,n,alpha); return lo,hi

def _stepup_rejected(p,m,q,method):
    p=np.asarray(p,float); m=int(m); q=float(q); method=str(method).upper()
    if p.ndim!=1 or len(p)!=m or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)): raise ValueError("Некорректные p-values")
    if not (0<q<1): raise ValueError("q must be in (0,1)")
    factor=1.0 if method=="BH" else float(np.sum(1.0/np.arange(1,m+1,dtype=float))) if method=="BY" else None
    if factor is None: raise ValueError("fdr_method должен быть BH или BY")
    order=np.argsort(p,kind="mergesort"); thr=q*np.arange(1,m+1,dtype=float)/(m*factor); ok=p[order]<=thr
    if not np.any(ok): return np.zeros(m,bool)
    k=int(np.flatnonzero(ok)[-1])+1
    return p<=q*k/(m*factor)

def _mc_fdr_sets(p_lower,p_upper,m,q,method):
    p_lower=np.asarray(p_lower,float); p_upper=np.asarray(p_upper,float)
    if len(p_lower)!=m or len(p_upper)!=m or np.any(p_lower>p_upper): raise ValueError("Некорректные MC confidence bounds")
    return _stepup_rejected(p_lower,m,q,method),_stepup_rejected(p_upper,m,q,method)


def _load_npz_closed(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.array(z[k],copy=True) for k in z.files}

def _checkpoint_save(path,target_ids,family_ids,T_obs,ge,unknown,n_rep,status,p_lower,p_upper,look_index,next_attempt,*,B,initial_B,look_multiplier,
                     mc_alpha,fdr_q,fdr_method,seed,noise_signature,traces,look_schedule,failure_counts):
    """Атомарный checkpoint; npz не остаётся открытым в Windows."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
    payload={"algorithm_version":np.asarray(EF_VERSION),"test_statistic":np.asarray("delta_rss"),"noise_signature":np.asarray(noise_signature),
             "target_ids":np.asarray(target_ids,np.int64),"family_ids":np.asarray(family_ids,np.int64),"T_obs":np.asarray(T_obs,float),
             "ge_count":np.asarray(ge,np.int64),"unknown_count":np.asarray(unknown,np.int64),"n_rep":np.asarray(n_rep,np.int64),
             "status":np.asarray(status,np.int8),"p_lower":np.asarray(p_lower,float),"p_upper":np.asarray(p_upper,float),
             "look_index":np.int64(look_index),"next_attempt":np.int64(next_attempt),"B":np.int64(B),"initial_B":np.int64(initial_B),
             "look_multiplier":np.float64(look_multiplier),"mc_alpha":np.float64(mc_alpha),"fdr_q":np.float64(fdr_q),
             "fdr_method":np.asarray(str(fdr_method).upper()),"seed":np.int64(seed),"look_schedule":np.asarray(look_schedule,np.int64)}
    payload.update({f"failure_{k}":np.int64(v) for k,v in failure_counts.items()})
    payload.update({f"trace_{int(h)}":np.asarray(v,float) for h,v in traces.items()})
    try:
        with open(tmp,"wb") as f:
            np.savez_compressed(f,**payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass

def save_checkpoint_from_result(result,path,*,initial_B=64,look_multiplier=2.0,mc_alpha=1e-6,fdr_q=0.01,fdr_method="BY",seed=20261004):
    required=("hypothesis_id","family_ids","T_obs","ge_count","unknown_count","n_bootstrap","status","p_lower","p_upper","B","attempts","noise_signature")
    miss=[k for k in required if k not in result]
    if miss: raise ValueError(f"В result отсутствуют поля: {miss}")
    B=int(result["B"]); schedule=tuple(np.asarray(result.get("look_schedule",_look_schedule(B,int(initial_B),float(look_multiplier))),np.int64).tolist())
    if not schedule or schedule[-1]!=B: raise ValueError("Некорректный look_schedule")
    _checkpoint_save(path,result["hypothesis_id"],result["family_ids"],result["T_obs"],result["ge_count"],result["unknown_count"],result["n_bootstrap"],
                     result["status"],result["p_lower"],result["p_upper"],len(schedule),int(result["attempts"]),B=B,initial_B=min(int(initial_B),B),
                     look_multiplier=look_multiplier,mc_alpha=mc_alpha,fdr_q=fdr_q,fdr_method=fdr_method,seed=seed,
                     noise_signature=result["noise_signature"],traces=result.get("trace",{}),look_schedule=schedule,
                     failure_counts=result.get("failure_counts",{}))


def _run_empirical_null_union(data_yxt,theta,spec,m0_results,hypotheses_obs,df_obs,noise_model,*,valid_yxt=None,B=32768,B_fit=128,seed=20261004,
                       checkpoint_path=None,checkpoint_every=32,target_hypothesis_ids=None,trace_hypothesis_ids=(),m0_solver_config=None,
                       m1_solver_config=None,fractions=DEFAULT_FRACTIONS,progress=True,initial_B=64,look_multiplier=2.0,
                       mc_alpha=1e-6,fdr_q=0.01,fdr_method="BY"):
    """Sequential parametric bootstrap under fitted M0; T=ΔRSS."""
    _validate_noise_plugin(noise_model)
    data_yxt=np.asarray(data_yxt,float); theta=np.asarray(theta,float); _validate_valid_mask(valid_yxt,data_yxt.shape)
    B=int(B); B_fit=int(B_fit); initial_B=int(initial_B); look_multiplier=float(look_multiplier); mc_alpha=float(mc_alpha); fdr_q=float(fdr_q); fdr_method=str(fdr_method).upper()
    if B<1 or B_fit<1 or initial_B<1 or look_multiplier<=1: raise ValueError("Некорректные параметры sequential bootstrap")
    if not (0<mc_alpha<fdr_q<1): raise ValueError("Требуется 0 < mc_alpha < fdr_q < 1")
    if fdr_method not in {"BH","BY"}: raise ValueError("fdr_method должен быть BH или BY")
    initial_B=min(initial_B,B); noise_sig=_noise_signature(noise_model)
    df=df_obs.reset_index(drop=True).copy()
    if not {"hypothesis_id","delta_rss_global"}.issubset(df.columns): raise ValueError("df_obs должен содержать hypothesis_id и delta_rss_global")
    ids0=np.asarray(df["hypothesis_id"],np.int64); T0=np.asarray(df["delta_rss_global"],float)
    if len(np.unique(ids0))!=len(ids0): raise ValueError("df_obs содержит повторяющиеся hypothesis_id")
    hids_all=np.asarray(hypotheses_obs.hypothesis_id,np.int64)
    if len(np.unique(hids_all))!=len(hids_all) or not np.array_equal(np.sort(ids0),np.sort(hids_all)): raise ValueError("df_obs должен содержать ровно полное D1-семейство")
    if np.any(~np.isfinite(T0)) or np.any(T0<0): raise ValueError("delta_rss_global должен быть конечным и неотрицательным")
    order0=np.argsort(ids0,kind="mergesort"); family_ids=ids0[order0]; T_family=T0[order0]; family_m=len(family_ids)
    if family_m<1: raise ValueError("Пустое семейство hypotheses")
    if target_hypothesis_ids is None: ids=family_ids.copy(); formal_family=True
    else:
        ids=np.sort(np.intersect1d(family_ids,np.asarray(tuple(target_hypothesis_ids),np.int64)),kind="mergesort"); formal_family=False
        if not len(ids): raise ValueError("Нет target hypotheses в полном семействе")
    pos_family={int(h):i for i,h in enumerate(family_ids)}; target_pos=np.array([pos_family[int(h)] for h in ids],np.int64); T=T_family[target_pos]
    hids=np.asarray(hypotheses_obs.hypothesis_id,np.int64); pos={int(h):i for i,h in enumerate(hids)}
    idx=np.array([pos[int(h)] for h in ids],np.int64); h=hypotheses_obs
    target_h=SplitHypotheses(h.hypothesis_id[idx].copy(),h.K0[idx].copy(),h.pixel_id[idx].copy(),h.x_pixel[idx].copy(),h.y_pixel[idx].copy(),
                             h.parent_index[idx].copy(),h.parent_peak_id[idx].copy(),h.theta_left[idx].copy(),h.theta_right[idx].copy(),
                             h.m0_x[idx].copy(),h.m0_lb[idx].copy(),h.m0_ub[idx].copy(),h.m0_cost[idx].copy(),h.n_valid[idx].copy())
    n=len(ids); ge=np.zeros(n,np.int64); unknown=np.zeros(n,np.int64); n_rep=np.zeros(n,np.int64); status=np.zeros(n,np.int8)
    p_lower=np.zeros(n,float); p_upper=np.ones(n,float); target_set=set(map(int,ids)); traces={int(hid):np.full(B,np.nan,float) for hid in trace_hypothesis_ids if int(hid) in target_set}
    schedule=_look_schedule(B,initial_B,look_multiplier); look_index=0; next_attempt=0; cp=Path(checkpoint_path) if checkpoint_path else None
    failure_counts={"M0_retry":0,"M0_retry2":0,"M0_hard":0,"M0_label_swap":0,"M1_retry":0,"M1_hard":0,"delta_unknown":0}
    if cp and cp.exists():
        d=_load_npz_closed(cp); version=str(d.get("algorithm_version",""))
        if version!=EF_VERSION: raise ValueError(f"Checkpoint incompatible with {EF_VERSION}")
        required={"target_ids","family_ids","T_obs","ge_count","unknown_count","n_rep","status","p_lower","p_upper","look_index","next_attempt","B",
                  "initial_B","look_multiplier","mc_alpha","fdr_q","fdr_method","seed","noise_signature","look_schedule","test_statistic"}
        miss=sorted(required.difference(d))
        if miss: raise ValueError(f"Checkpoint неполный: отсутствуют {miss}")
        if str(d["test_statistic"])!="delta_rss" or str(d["noise_signature"])!=noise_sig: raise ValueError("Checkpoint statistic/noise-model mismatch")
        if not np.array_equal(np.asarray(d["target_ids"],np.int64),ids) or not np.array_equal(np.asarray(d["family_ids"],np.int64),family_ids): raise ValueError("Checkpoint hypothesis set mismatch")
        if int(d["B"])>B or int(d["initial_B"])!=initial_B or not np.isclose(float(d["look_multiplier"]),look_multiplier) or not np.isclose(float(d["mc_alpha"]),mc_alpha) or not np.isclose(float(d["fdr_q"]),fdr_q) or str(d["fdr_method"]).upper()!=fdr_method or int(d["seed"])!=seed: raise ValueError("Checkpoint configuration mismatch")
        if not np.allclose(np.asarray(d["T_obs"],float),T,rtol=1e-12,atol=0): raise ValueError("Checkpoint observed statistic mismatch")
        ge=np.asarray(d["ge_count"],np.int64); unknown=np.asarray(d["unknown_count"],np.int64); n_rep=np.asarray(d["n_rep"],np.int64)
        status=np.asarray(d["status"],np.int8); p_lower=np.asarray(d["p_lower"],float); p_upper=np.asarray(d["p_upper"],float)
        look_index=int(d["look_index"]); next_attempt=int(d["next_attempt"]); old_B=int(d["B"])
        if any(a.shape!=(n,) for a in (ge,unknown,n_rep,status,p_lower,p_upper)): raise ValueError("Checkpoint array shape mismatch")
        if np.any(n_rep<0) or np.any(ge<0) or np.any(unknown<0) or np.any(ge+unknown>n_rep) or np.any((status<0)|(status>2)) or np.any(p_lower>p_upper): raise ValueError("Checkpoint state invalid")
        old_schedule=tuple(np.asarray(d["look_schedule"],np.int64).tolist()); full_schedule=_look_schedule(B,initial_B,look_multiplier)
        if not old_schedule or old_schedule[-1]!=old_B or tuple(full_schedule[:len(old_schedule)])!=old_schedule: raise ValueError("Checkpoint look schedule incompatible")
        schedule=old_schedule if B==old_B else full_schedule
        for k in failure_counts:
            key=f"failure_{k}"
            if key in d: failure_counts[k]=int(d[key])
        for hid in traces:
            key=f"trace_{hid}"
            if key in d:
                z=np.asarray(d[key],float); traces[hid][:min(len(z),B)]=z[:min(len(z),B)]
    m0_solvers={}; m1_solvers={}; m0_retry_solvers={}; m0_retry2_solvers={}; m1_retry_solvers={}
    cfg0=DEFAULT_SOLVER if m0_solver_config is None else dict(m0_solver_config); cfg1=DEFAULT_SOLVER if m1_solver_config is None else dict(m1_solver_config)
    cfg0r=_cfg_retry(cfg0); cfg0r2=dict(cfg0r); cfg0r2["max_iter"]=2*int(cfg0r["max_iter"]); cfg1r=_cfg_retry(cfg1); pbar=None
    def classify(active_idx,look_no):
        if not len(active_idx): return
        alpha_look=mc_alpha*(6.0/(np.pi**2*(look_no+1)**2))/(2.0*family_m)
        lo,hi=_mc_cp_bounds(ge[active_idx],unknown[active_idx],n_rep[active_idx],alpha_look)
        p_lower[active_idx]=np.maximum(p_lower[active_idx],lo); p_upper[active_idx]=np.minimum(p_upper[active_idx],hi)
        if np.any(p_lower[active_idx]>p_upper[active_idx]+1e-15): raise RuntimeError("MC confidence interval became empty")
        if formal_family:
            possible,certain=_mc_fdr_sets(p_lower,p_upper,family_m,fdr_q,fdr_method)
            status[(status==0)&certain]=1; status[(status==0)&~possible]=2
    if progress:
        from tqdm.auto import tqdm
        pbar=tqdm(total=B,initial=int(np.max(n_rep) if len(n_rep) else 0),desc="Sequential empirical null v8",unit="replicate",mininterval=.5)
    try:
        while look_index<len(schedule):
            active_idx=np.flatnonzero(status==0)
            if not len(active_idx): break
            current=int(n_rep[active_idx[0]])
            if np.any(n_rep[active_idx]!=current): raise RuntimeError("Active hypotheses имеют разные n_rep")
            target=int(schedule[look_index]); active_all=active_idx
            state=build_bootstrap_state(m0_results,target_pixel_ids=np.unique(target_h.pixel_id[active_all]))
            active_h_all=SplitHypotheses(target_h.hypothesis_id[active_all],target_h.K0[active_all],target_h.pixel_id[active_all],target_h.x_pixel[active_all],
                                         target_h.y_pixel[active_all],target_h.parent_index[active_all],target_h.parent_peak_id[active_all],
                                         target_h.theta_left[active_all],target_h.theta_right[active_all],target_h.m0_x[active_all],target_h.m0_lb[active_all],
                                         target_h.m0_ub[active_all],target_h.m0_cost[active_all],target_h.n_valid[active_all])
            target_source_all=_prepare_target_source(state,active_h_all); fast_all=_can_use_vectorized_responsibility(active_h_all,theta,len(spec.param_names))
            mu_state={K:evaluate_composite(spec,theta,g.Xhat) for K,g in state.items()}; data_star=np.empty_like(data_yxt,float)
            while current<target:
                attempt=next_attempt; rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(attempt)]))
                _simulate_bootstrap_into(rng,data_star,state,mu_state,noise_model,theta=theta,valid_yxt=valid_yxt)
                m0_star=fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0,solvers=m0_solvers)
                bad0={int(g["K"]):np.flatnonzero(~np.asarray(g["success"],bool)) for g in m0_star if np.any(~np.asarray(g["success"],bool))}
                if bad0:
                    failure_counts["M0_retry"]+=sum(len(v) for v in bad0.values()); xover={}
                    for K,rows in bad0.items():
                        gg=next(z for z in m0_star if int(z["K"])==K); cand=np.asarray(gg["x"],float)[rows]; ref=np.asarray(state[K].Xhat,float)[rows]
                        feasible=np.all(np.isfinite(cand),axis=1)&np.all(cand>=np.asarray(gg["lb"],float)[rows],axis=1)&np.all(cand<=np.asarray(gg["ub"],float)[rows],axis=1)
                        xover[K]=np.where(feasible[:,None],cand,ref)
                    _merge_m0_rows(m0_star,fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0r,solvers=m0_retry_solvers,row_indices=bad0,x0_overrides=xover))
                    bad1={int(g["K"]):np.flatnonzero(~np.asarray(g["success"],bool)) for g in m0_star if np.any(~np.asarray(g["success"],bool))}
                    if bad1:
                        failure_counts["M0_retry2"]+=sum(len(v) for v in bad1.values()); xover2={}
                        for K,rows in bad1.items():
                            gg=next(z for z in m0_star if int(z["K"])==K); cand=np.asarray(gg["x"],float)[rows]; ref=np.asarray(state[K].Xhat,float)[rows]
                            feasible=np.all(np.isfinite(cand),axis=1)&np.all(cand>=np.asarray(gg["lb"],float)[rows],axis=1)&np.all(cand<=np.asarray(gg["ub"],float)[rows],axis=1)
                            xover2[K]=np.where(feasible[:,None],cand,ref)
                        _merge_m0_rows(m0_star,fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0r2,solvers=m0_retry2_solvers,row_indices=bad1,x0_overrides=xover2))
                failure_counts["M0_label_swap"]+=_canonicalize_m0_star(m0_star,len(spec.param_names))
                bad_keys={(int(g["K"]),int(g["pixel_id"][r])) for g in m0_star for r in np.flatnonzero(~np.asarray(g["success"],bool))}
                m0_bad=np.fromiter(((int(target_h.K0[j]),int(target_h.pixel_id[j])) in bad_keys for j in active_all),dtype=bool,count=len(active_all))
                if np.any(m0_bad): failure_counts["M0_hard"]+=int(np.count_nonzero(m0_bad)); unknown[active_all[m0_bad]]+=1
                good_local=~m0_bad
                if np.any(good_local):
                    good_active=active_all[good_local]
                    if np.all(good_local): active_h=active_h_all; target_source=target_source_all; fast=fast_all
                    else:
                        active_h=SplitHypotheses(target_h.hypothesis_id[good_active],target_h.K0[good_active],target_h.pixel_id[good_active],target_h.x_pixel[good_active],
                                                 target_h.y_pixel[good_active],target_h.parent_index[good_active],target_h.parent_peak_id[good_active],
                                                 target_h.theta_left[good_active],target_h.theta_right[good_active],target_h.m0_x[good_active],target_h.m0_lb[good_active],
                                                 target_h.m0_ub[good_active],target_h.m0_cost[good_active],target_h.n_valid[good_active])
                        target_source=_prepare_target_source(state,active_h); fast=_can_use_vectorized_responsibility(active_h,theta,len(spec.param_names))
                    hstar=_rebuild_hypotheses_star(active_h,m0_star,theta,q0=len(spec.param_names),target_source=target_source,fast_responsibility=fast)
                    m1=_fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=fractions,solver_config=cfg1,solvers=m1_solvers,return_x=False)
                    bad=np.flatnonzero(~np.asarray(m1["success"],bool))
                    if len(bad):
                        failure_counts["M1_retry"]+=len(bad); cont={int(j):m1["continuation_x"][int(j)] for j in bad if m1["continuation_x"][int(j)] is not None}
                        if cont:
                            _merge_m1(m1,_fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=(fractions[0],),solver_config=cfg1r,
                                                      solvers=m1_retry_solvers,hypothesis_idx=np.asarray(sorted(cont),np.int64),x0_overrides=cont,return_x=False))
                        bad=np.flatnonzero(~np.asarray(m1["success"],bool))
                        if len(bad):
                            _merge_m1(m1,_fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=fractions,solver_config=cfg1r,
                                                      solvers=m1_retry_solvers,hypothesis_idx=bad,include_null_start=False,return_x=False))
                            bad=np.flatnonzero(~np.asarray(m1["success"],bool))
                        if len(bad): failure_counts["M1_hard"]+=len(bad); unknown[good_active[bad]]+=1
                    okidx=np.flatnonzero(np.asarray(m1["success"],bool))
                    if len(okidx):
                        delta=2.0*(np.asarray(hstar.m0_cost,float)[okidx]-np.asarray(m1["cost"],float)[okidx]); good=np.isfinite(delta)&(delta>=0)
                        if np.any(~good): failure_counts["delta_unknown"]+=int(np.count_nonzero(~good)); unknown[good_active[okidx[~good]]]+=1
                        if np.any(good):
                            loc=okidx[good]; Ts=delta[good]; glob=good_active[loc]; ge[glob]+=(Ts>=T[glob]); b0=current
                            for jj,gidx in enumerate(glob):
                                hid=int(ids[gidx])
                                if hid in traces: traces[hid][b0]=float(Ts[jj])
                n_rep[active_all]+=1; current+=1; next_attempt=attempt+1
                if pbar is not None: pbar.update(1)
                if cp and checkpoint_every>0 and next_attempt%int(checkpoint_every)==0:
                    _checkpoint_save(cp,ids,family_ids,T,ge,unknown,n_rep,status,p_lower,p_upper,look_index,next_attempt,B=B,initial_B=initial_B,
                                     look_multiplier=look_multiplier,mc_alpha=mc_alpha,fdr_q=fdr_q,fdr_method=fdr_method,seed=seed,
                                     noise_signature=noise_sig,traces=traces,look_schedule=schedule,failure_counts=failure_counts)
            classify(active_all,look_index); look_index+=1
            if cp: _checkpoint_save(cp,ids,family_ids,T,ge,unknown,n_rep,status,p_lower,p_upper,look_index,next_attempt,B=B,initial_B=initial_B,
                                    look_multiplier=look_multiplier,mc_alpha=mc_alpha,fdr_q=fdr_q,fdr_method=fdr_method,seed=seed,
                                    noise_signature=noise_sig,traces=traces,look_schedule=schedule,failure_counts=failure_counts)
            if pbar is not None: pbar.refresh()
    finally:
        if pbar is not None: pbar.close()
    p_emp_lo,p_emp_hi=empirical_pvalue_bounds(ge,unknown,n_rep); p_emp=np.where(unknown==0,p_emp_lo,np.nan); unresolved=np.flatnonzero(status==0)
    labels=np.asarray(["unresolved","certified_reject","certified_nonreject"],object)[status]; cert=ids[status==1]; non=ids[status==2]
    return {"B":B,"look_schedule":np.asarray(schedule,np.int64),"attempts":next_attempt,"invalid_attempts":int(np.sum(unknown)),
            "unknown_events":int(np.sum(unknown)),"unknown_hypotheses":int(np.count_nonzero(unknown)),"failure_counts":failure_counts,
            "complete":len(unresolved)==0,"family_complete":formal_family,"unresolved_count":int(len(unresolved)),"test_statistic":"delta_rss",
            "noise_model_kind":noise_model.kind,"noise_signature":noise_sig,"hypothesis_id":ids,"T_obs":T,"delta_rss_obs":T.copy(),
            "ge_count":ge,"unknown_count":unknown,"n_bootstrap":n_rep,"p_emp":p_emp,"p_emp_lower":p_emp_lo,"p_emp_upper":p_emp_hi,
            "p_lower":p_lower,"p_upper":p_upper,"status":status,"status_label":labels,"bh_rejected_ids":cert,"certified_reject_ids":cert,
            "certified_nonreject_ids":non,"fdr_q":fdr_q,"fdr_method":fdr_method,"mc_alpha":mc_alpha,"trace":traces,
            "family_m":family_m,"family_ids":family_ids,"target_hypothesis_ids":ids.copy()}

# ===================== Adaptive MC controller =====================
def _mc_alpha_at(look_no,m,mc_alpha):
    j=np.asarray(look_no,np.int64); return float(mc_alpha)*(6.0/(np.pi**2*(j+1.0)**2))/(2.0*int(m))

def _mmctest_alpha_increment(n_old,n_new,m,epsilon,r):
    """Односторонний CP-бюджет MMCTest для перехода n_old→n_new."""
    n0=np.asarray(n_old,float); n1=np.asarray(n_new,float); m=int(m); epsilon=float(epsilon); r=float(r)
    if m<1 or not(0<epsilon<1) or not(r>0) or np.any(n0<0) or np.any(n1<=n0): raise ValueError("Некорректные MMCTest spending-параметры")
    a=(n1/(n1+r)-n0/(n0+r))*(epsilon/m)/2.0
    if np.any(a<=0) or np.any(a>=.5): raise RuntimeError("Некорректный MMCTest alpha increment")
    return a

def _subset_hypotheses(h,idx):
    idx=np.asarray(idx,np.int64); return SplitHypotheses(h.hypothesis_id[idx],h.K0[idx],h.pixel_id[idx],h.x_pixel[idx],h.y_pixel[idx],h.parent_index[idx],h.parent_peak_id[idx],h.theta_left[idx],h.theta_right[idx],h.m0_x[idx],h.m0_lb[idx],h.m0_ub[idx],h.m0_cost[idx],h.n_valid[idx])

def _hash_array(h,a):
    a=np.ascontiguousarray(np.asarray(a)); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,np.int64).tobytes()); h.update(a.view(np.uint8))

def _adaptive_run_signature(data_yxt,theta,valid_yxt,m0_results,hypotheses_obs,spec,*,B_fit,fractions,m0_solver_config,m1_solver_config,noise_signature):
    h=hashlib.sha256(); cfg={"B_fit":int(B_fit),"fractions":list(map(float,fractions)),"m0_solver":dict(DEFAULT_SOLVER if m0_solver_config is None else m0_solver_config),"m1_solver":dict(DEFAULT_SOLVER if m1_solver_config is None else m1_solver_config),"param_names":list(spec.param_names),"spec_type":f"{type(spec).__module__}.{type(spec).__qualname__}","batch_func":f"{getattr(spec.batch_func,'__module__','')}.{getattr(spec.batch_func,'__qualname__',getattr(spec.batch_func,'__name__',''))}"}; h.update(str(noise_signature).encode()); h.update(json.dumps(cfg,sort_keys=True,separators=(",",":"),default=str).encode()); _hash_array(h,data_yxt); _hash_array(h,theta); _hash_array(h,np.asarray([],bool) if valid_yxt is None else np.asarray(valid_yxt,bool))
    for g in m0_results:
        for k in ("K","pixel_id","x_pixel","y_pixel","x","lb","ub","success","cost"):
            if k in g: _hash_array(h,g[k])
    for k in ("hypothesis_id","K0","pixel_id","parent_index","parent_peak_id","theta_left","theta_right","m0_x","m0_lb","m0_ub","m0_cost","n_valid"): _hash_array(h,getattr(hypotheses_obs,k))
    return h.hexdigest()

def _adaptive_priority(p_lower,p_upper,m,q,method,look_pos,ids):
    lo=np.asarray(p_lower,float); hi=np.asarray(p_upper,float); eps=np.finfo(float).tiny; mid=np.where(lo>0,np.sqrt(lo*np.maximum(hi,eps)),.5*hi); order=np.argsort(mid,kind="mergesort"); rank=np.empty(len(mid),np.int64); rank[order]=np.arange(1,len(mid)+1); factor=1.0 if str(method).upper()=="BH" else float(np.sum(1.0/np.arange(1,int(m)+1,dtype=float))); thr=float(q)*rank/(int(m)*factor); width=np.maximum(hi-lo,0); prox=1.0+np.abs(np.log(np.maximum(mid,eps)/np.maximum(thr,eps))); return np.log1p(width/np.maximum(thr,eps))/prox

def _adaptive_checkpoint_save(path,*,ids,family_ids,T,ge,unknown,n_rep,status,p_lower,p_upper,look_pos,allocation_mask,next_attempt,B,initial_B,look_multiplier,mc_alpha,fdr_q,fdr_method,seed,noise_signature,run_signature,traces,schedule,failure_counts,adaptive_fraction,adaptive_min,adaptive_max_lead,controller_version="cp_adaptive_v2_m0canon",mmctest_r=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".tmp-{os.getpid()}"); payload={"algorithm_version":np.asarray(EF_VERSION),"controller_version":np.asarray(controller_version),"test_statistic":np.asarray("delta_rss"),"noise_signature":np.asarray(noise_signature),"run_signature":np.asarray(run_signature),"target_ids":np.asarray(ids,np.int64),"family_ids":np.asarray(family_ids,np.int64),"T_obs":np.asarray(T,float),"ge_count":np.asarray(ge,np.int64),"unknown_count":np.asarray(unknown,np.int64),"n_rep":np.asarray(n_rep,np.int64),"status":np.asarray(status,np.int8),"p_lower":np.asarray(p_lower,float),"p_upper":np.asarray(p_upper,float),"look_pos":np.asarray(look_pos,np.int16),"allocation_mask":np.asarray(allocation_mask,bool),"next_attempt":np.int64(next_attempt),"B":np.int64(B),"initial_B":np.int64(initial_B),"look_multiplier":np.float64(look_multiplier),"mc_alpha":np.float64(mc_alpha),"fdr_q":np.float64(fdr_q),"fdr_method":np.asarray(str(fdr_method).upper()),"seed":np.int64(seed),"look_schedule":np.asarray(schedule,np.int64),"adaptive_fraction":np.float64(adaptive_fraction),"adaptive_min":np.int64(adaptive_min),"adaptive_max_lead":np.int64(adaptive_max_lead)};
    if mmctest_r is not None: payload["mmctest_r"]=np.float64(mmctest_r)
    payload.update({f"failure_{k}":np.int64(v) for k,v in failure_counts.items()}); payload.update({f"trace_{int(h)}":np.asarray(v,float) for h,v in traces.items()})
    try:
        with open(tmp,"wb") as f: np.savez_compressed(f,**payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass

def _run_empirical_null_adaptive(data_yxt,theta,spec,m0_results,hypotheses_obs,df_obs,noise_model,*,valid_yxt=None,B=32768,B_fit=128,seed=20261004,checkpoint_path=None,checkpoint_every=32,target_hypothesis_ids=None,trace_hypothesis_ids=(),m0_solver_config=None,m1_solver_config=None,fractions=DEFAULT_FRACTIONS,progress=True,initial_B=64,look_multiplier=2.0,mc_alpha=1e-6,fdr_q=0.01,fdr_method="BY",adaptive_fraction=.25,adaptive_min=64,adaptive_max_lead=2,mc_scheme="cp_adaptive",mmctest_r=10000.0):
    """Adaptive engine: cp_adaptive либо MMCTest spending; bootstrap-ядро общее."""
    mc_scheme=str(mc_scheme).lower(); mmctest_r=float(mmctest_r); controller_version="cp_adaptive_v2_m0canon" if mc_scheme=="cp_adaptive" else "mmctest_v1_m0canon"
    if mc_scheme not in {"cp_adaptive","mmctest"}: raise ValueError("Неизвестная adaptive MC-схема")
    if mc_scheme=="mmctest" and not(mmctest_r>0): raise ValueError("mmctest_r должен быть > 0")
    if target_hypothesis_ids is not None: raise ValueError(f"{mc_scheme} разрешён только для полного formal family; pilot используйте mc_controller='cp_union'")
    _validate_noise_plugin(noise_model)
    data_yxt=np.asarray(data_yxt,float); theta=np.asarray(theta,float); _validate_valid_mask(valid_yxt,data_yxt.shape); B=int(B); B_fit=int(B_fit); initial_B=min(int(initial_B),B); look_multiplier=float(look_multiplier); mc_alpha=float(mc_alpha); fdr_q=float(fdr_q); fdr_method=str(fdr_method).upper(); adaptive_fraction=float(adaptive_fraction); adaptive_min=int(adaptive_min); adaptive_max_lead=int(adaptive_max_lead)
    if B<1 or B_fit<1 or initial_B<1 or look_multiplier<=1 or not(0<mc_alpha<fdr_q<1) or fdr_method not in {"BH","BY"}: raise ValueError("Некорректные sequential/FDR параметры")
    if not(0<adaptive_fraction<=1) or adaptive_min<1 or adaptive_max_lead<0: raise ValueError("Некорректные adaptive-параметры")
    noise_sig=_noise_signature(noise_model); df=df_obs.reset_index(drop=True).copy()
    if not {"hypothesis_id","delta_rss_global"}.issubset(df.columns): raise ValueError("df_obs должен содержать hypothesis_id и delta_rss_global")
    ids0=np.asarray(df["hypothesis_id"],np.int64); T0=np.asarray(df["delta_rss_global"],float); hids_all=np.asarray(hypotheses_obs.hypothesis_id,np.int64)
    if len(np.unique(ids0))!=len(ids0) or len(np.unique(hids_all))!=len(hids_all) or not np.array_equal(np.sort(ids0),np.sort(hids_all)): raise ValueError("df_obs должен содержать ровно полное D1-семейство")
    if np.any(~np.isfinite(T0)) or np.any(T0<0): raise ValueError("delta_rss_global должен быть конечным и неотрицательным")
    order0=np.argsort(ids0,kind="mergesort"); ids=ids0[order0]; T=T0[order0]; family_ids=ids.copy(); family_m=len(ids); pos={int(h):i for i,h in enumerate(hids_all)}; idx=np.array([pos[int(h)] for h in ids],np.int64); target_h=_subset_hypotheses(hypotheses_obs,idx); n=len(ids)
    ge=np.zeros(n,np.int64); unknown=np.zeros(n,np.int64); n_rep=np.zeros(n,np.int64); status=np.zeros(n,np.int8); p_lower=np.zeros(n,float); p_upper=np.ones(n,float); look_pos=np.zeros(n,np.int16); allocation_mask=np.zeros(n,bool); traces={int(hid):np.full(B,np.nan,float) for hid in trace_hypothesis_ids if int(hid) in set(map(int,ids))}; schedule=_look_schedule(B,initial_B,look_multiplier); next_attempt=0; cp=Path(checkpoint_path) if checkpoint_path else None; failure_counts={"M0_retry":0,"M0_retry2":0,"M0_hard":0,"M0_label_swap":0,"M1_retry":0,"M1_hard":0,"delta_unknown":0}; run_sig=_adaptive_run_signature(data_yxt,theta,valid_yxt,m0_results,hypotheses_obs,spec,B_fit=B_fit,fractions=fractions,m0_solver_config=m0_solver_config,m1_solver_config=m1_solver_config,noise_signature=noise_sig)
    if cp and cp.exists():
        d=_load_npz_closed(cp); req={"algorithm_version","controller_version","test_statistic","noise_signature","run_signature","target_ids","family_ids","T_obs","ge_count","unknown_count","n_rep","status","p_lower","p_upper","look_pos","allocation_mask","next_attempt","B","initial_B","look_multiplier","mc_alpha","fdr_q","fdr_method","seed","look_schedule","adaptive_fraction","adaptive_min","adaptive_max_lead"};
        if mc_scheme=="mmctest": req.add("mmctest_r")
        miss=sorted(req.difference(d))
        if miss: raise ValueError(f"Adaptive checkpoint неполный: {miss}")
        if str(d["algorithm_version"])!=EF_VERSION or str(d["controller_version"])!=controller_version or str(d["test_statistic"])!="delta_rss" or str(d["noise_signature"])!=noise_sig or str(d["run_signature"])!=run_sig: raise ValueError("Adaptive checkpoint signature mismatch")
        if not np.array_equal(np.asarray(d["target_ids"],np.int64),ids) or not np.array_equal(np.asarray(d["family_ids"],np.int64),family_ids) or not np.allclose(np.asarray(d["T_obs"],float),T,rtol=1e-12,atol=0): raise ValueError("Adaptive checkpoint hypothesis/T mismatch")
        if int(d["B"])>B or int(d["initial_B"])!=initial_B or not np.isclose(float(d["look_multiplier"]),look_multiplier) or not np.isclose(float(d["mc_alpha"]),mc_alpha) or not np.isclose(float(d["fdr_q"]),fdr_q) or str(d["fdr_method"]).upper()!=fdr_method or int(d["seed"])!=seed or not np.isclose(float(d["adaptive_fraction"]),adaptive_fraction) or int(d["adaptive_min"])!=adaptive_min or int(d["adaptive_max_lead"])!=adaptive_max_lead or (mc_scheme=="mmctest" and not np.isclose(float(d["mmctest_r"]),mmctest_r)): raise ValueError("Adaptive checkpoint configuration mismatch")
        ge=np.asarray(d["ge_count"],np.int64); unknown=np.asarray(d["unknown_count"],np.int64); n_rep=np.asarray(d["n_rep"],np.int64); status=np.asarray(d["status"],np.int8); p_lower=np.asarray(d["p_lower"],float); p_upper=np.asarray(d["p_upper"],float); look_pos=np.asarray(d["look_pos"],np.int16); allocation_mask=np.asarray(d["allocation_mask"],bool); next_attempt=int(d["next_attempt"]); old_B=int(d["B"])
        if any(a.shape!=(n,) for a in (ge,unknown,n_rep,status,p_lower,p_upper,look_pos,allocation_mask)) or np.any(ge+unknown>n_rep) or np.any(p_lower>p_upper) or np.any((status<0)|(status>2)): raise ValueError("Adaptive checkpoint state invalid")
        old_schedule=tuple(np.asarray(d["look_schedule"],np.int64).tolist()); full_schedule=_look_schedule(B,initial_B,look_multiplier)
        if not old_schedule or old_schedule[-1]!=old_B or tuple(full_schedule[:len(old_schedule)])!=old_schedule: raise ValueError("Adaptive checkpoint look schedule incompatible")
        schedule=old_schedule if B==old_B else full_schedule
        for k in failure_counts:
            if f"failure_{k}" in d: failure_counts[k]=int(d[f"failure_{k}"])
        for hid in traces:
            if f"trace_{hid}" in d:
                z=np.asarray(d[f"trace_{hid}"],float); traces[hid][:min(len(z),B)]=z[:min(len(z),B)]
    cfg0=DEFAULT_SOLVER if m0_solver_config is None else dict(m0_solver_config); cfg1=DEFAULT_SOLVER if m1_solver_config is None else dict(m1_solver_config); cfg0r=_cfg_retry(cfg0); cfg0r2=dict(cfg0r); cfg0r2["max_iter"]=2*int(cfg0r["max_iter"]); cfg1r=_cfg_retry(cfg1); m0_solvers={}; m1_solvers={}; m0_retry_solvers={}; m0_retry2_solvers={}; m1_retry_solvers={}
    def classify(batch,level):
        if not len(batch): return
        if mc_scheme=="cp_adaptive": a=_mc_alpha_at(np.full(len(batch),int(level)),family_m,mc_alpha)
        else:
            n1=n_rep[batch]; n0=np.zeros(len(batch),np.int64) if level==0 else np.full(len(batch),int(schedule[level-1]),np.int64); a=_mmctest_alpha_increment(n0,n1,family_m,mc_alpha,mmctest_r)
        lo,hi=_mc_cp_bounds(ge[batch],unknown[batch],n_rep[batch],a); p_lower[batch]=np.maximum(p_lower[batch],lo); p_upper[batch]=np.minimum(p_upper[batch],hi)
        if np.any(p_lower[batch]>p_upper[batch]+1e-15): raise RuntimeError("MC confidence interval became empty")
        possible,certain=_mc_fdr_sets(p_lower,p_upper,family_m,fdr_q,fdr_method); status[(status==0)&certain]=1; status[(status==0)&~possible]=2
    def choose_batch():
        eligible=np.flatnonzero((status==0)&(look_pos<len(schedule)))
        if not len(eligible): return np.empty(0,np.int64),None
        if np.all(n_rep[eligible]==0): return eligible,0
        min_level=int(np.min(look_pos[eligible])); cand=eligible[look_pos[eligible]<=min_level+adaptive_max_lead]; score=_adaptive_priority(p_lower,p_upper,family_m,fdr_q,fdr_method,look_pos,ids); seed_idx=cand[np.lexsort((ids[cand],look_pos[cand],-score[cand]))[0]]; level=int(look_pos[seed_idx]); pool=eligible[look_pos[eligible]==level]; ordp=np.lexsort((ids[pool],-score[pool])); take=min(len(pool),max(adaptive_min,int(np.ceil(adaptive_fraction*len(pool))))); chosen=pool[ordp[:take]]; pix=np.unique(target_h.pixel_id[chosen]); chosen=pool[np.isin(target_h.pixel_id[pool],pix)]; return np.sort(chosen),level
    pbar=None
    if progress:
        from tqdm.auto import tqdm
        pbar=tqdm(total=n*B,initial=int(np.sum(n_rep)),desc="Adaptive empirical null v8",unit="hyp-rep",mininterval=.5)
    try:
        while True:
            if np.any(allocation_mask):
                batch=np.flatnonzero(allocation_mask); levels=np.unique(look_pos[batch]);
                if len(levels)!=1: raise RuntimeError("Adaptive allocation_mask содержит разные look levels")
                level=int(levels[0])
            else:
                batch,level=choose_batch()
                if not len(batch): break
                allocation_mask[batch]=True
            target=int(schedule[level]); current=int(n_rep[batch[0]])
            if np.any(n_rep[batch]!=current) or current>target: raise RuntimeError("Adaptive batch имеет несогласованный n_rep")
            state=build_bootstrap_state(m0_results,target_pixel_ids=np.unique(target_h.pixel_id[batch])); batch_h=_subset_hypotheses(target_h,batch); target_source_all=_prepare_target_source(state,batch_h); fast_all=_can_use_vectorized_responsibility(batch_h,theta,len(spec.param_names)); mu_state={K:evaluate_composite(spec,theta,g.Xhat) for K,g in state.items()}; data_star=np.empty_like(data_yxt,float)
            while current<target:
                attempt=next_attempt; rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(attempt)])); _simulate_bootstrap_into(rng,data_star,state,mu_state,noise_model,theta=theta,valid_yxt=valid_yxt); m0_star=fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0,solvers=m0_solvers); bad0={int(g["K"]):np.flatnonzero(~np.asarray(g["success"],bool)) for g in m0_star if np.any(~np.asarray(g["success"],bool))}
                if bad0:
                    failure_counts["M0_retry"]+=sum(len(v) for v in bad0.values()); xover={}
                    for K,rows in bad0.items():
                        gg=next(z for z in m0_star if int(z["K"])==K); candx=np.asarray(gg["x"],float)[rows]; ref=np.asarray(state[K].Xhat,float)[rows]; feasible=np.all(np.isfinite(candx),axis=1)&np.all(candx>=np.asarray(gg["lb"],float)[rows],axis=1)&np.all(candx<=np.asarray(gg["ub"],float)[rows],axis=1); xover[K]=np.where(feasible[:,None],candx,ref)
                    _merge_m0_rows(m0_star,fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0r,solvers=m0_retry_solvers,row_indices=bad0,x0_overrides=xover)); bad1={int(g["K"]):np.flatnonzero(~np.asarray(g["success"],bool)) for g in m0_star if np.any(~np.asarray(g["success"],bool))}
                    if bad1:
                        failure_counts["M0_retry2"]+=sum(len(v) for v in bad1.values()); xover2={}
                        for K,rows in bad1.items():
                            gg=next(z for z in m0_star if int(z["K"])==K); candx=np.asarray(gg["x"],float)[rows]; ref=np.asarray(state[K].Xhat,float)[rows]; feasible=np.all(np.isfinite(candx),axis=1)&np.all(candx>=np.asarray(gg["lb"],float)[rows],axis=1)&np.all(candx<=np.asarray(gg["ub"],float)[rows],axis=1); xover2[K]=np.where(feasible[:,None],candx,ref)
                        _merge_m0_rows(m0_star,fit_m0_star(data_star,theta,state,spec,valid_yxt=valid_yxt,B=B_fit,solver_config=cfg0r2,solvers=m0_retry2_solvers,row_indices=bad1,x0_overrides=xover2))
                failure_counts["M0_label_swap"]+=_canonicalize_m0_star(m0_star,len(spec.param_names))
                bad_keys={(int(g["K"]),int(g["pixel_id"][r])) for g in m0_star for r in np.flatnonzero(~np.asarray(g["success"],bool))}; m0_bad=np.fromiter(((int(target_h.K0[j]),int(target_h.pixel_id[j])) in bad_keys for j in batch),dtype=bool,count=len(batch))
                if np.any(m0_bad): failure_counts["M0_hard"]+=int(np.count_nonzero(m0_bad)); unknown[batch[m0_bad]]+=1
                good_local=~m0_bad
                if np.any(good_local):
                    good_active=batch[good_local]
                    if np.all(good_local): active_h=batch_h; target_source=target_source_all; fast=fast_all
                    else: active_h=_subset_hypotheses(target_h,good_active); target_source=_prepare_target_source(state,active_h); fast=_can_use_vectorized_responsibility(active_h,theta,len(spec.param_names))
                    hstar=_rebuild_hypotheses_star(active_h,m0_star,theta,q0=len(spec.param_names),target_source=target_source,fast_responsibility=fast); m1=_fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=fractions,solver_config=cfg1,solvers=m1_solvers,return_x=False); bad=np.flatnonzero(~np.asarray(m1["success"],bool))
                    if len(bad):
                        failure_counts["M1_retry"]+=len(bad); cont={int(j):m1["continuation_x"][int(j)] for j in bad if m1["continuation_x"][int(j)] is not None}
                        if cont: _merge_m1(m1,_fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=(fractions[0],),solver_config=cfg1r,solvers=m1_retry_solvers,hypothesis_idx=np.asarray(sorted(cont),np.int64),x0_overrides=cont,return_x=False))
                        bad=np.flatnonzero(~np.asarray(m1["success"],bool))
                        if len(bad): _merge_m1(m1,_fit_m1_best(data_star,theta,hstar,spec,valid_yxt=valid_yxt,B=B_fit,fractions=fractions,solver_config=cfg1r,solvers=m1_retry_solvers,hypothesis_idx=bad,include_null_start=False,return_x=False)); bad=np.flatnonzero(~np.asarray(m1["success"],bool))
                        if len(bad): failure_counts["M1_hard"]+=len(bad); unknown[good_active[bad]]+=1
                    okidx=np.flatnonzero(np.asarray(m1["success"],bool))
                    if len(okidx):
                        delta=2.0*(np.asarray(hstar.m0_cost,float)[okidx]-np.asarray(m1["cost"],float)[okidx]); good=np.isfinite(delta)&(delta>=0)
                        if np.any(~good): failure_counts["delta_unknown"]+=int(np.count_nonzero(~good)); unknown[good_active[okidx[~good]]]+=1
                        if np.any(good):
                            loc=okidx[good]; Ts=delta[good]; glob=good_active[loc]; ge[glob]+=(Ts>=T[glob])
                            for jj,gidx in enumerate(glob):
                                hid=int(ids[gidx]); b0=int(n_rep[gidx]);
                                if hid in traces and b0<B: traces[hid][b0]=float(Ts[jj])
                n_rep[batch]+=1; current+=1; next_attempt=attempt+1
                if pbar is not None: pbar.update(len(batch))
                if cp and checkpoint_every>0 and next_attempt%int(checkpoint_every)==0: _adaptive_checkpoint_save(cp,ids=ids,family_ids=family_ids,T=T,ge=ge,unknown=unknown,n_rep=n_rep,status=status,p_lower=p_lower,p_upper=p_upper,look_pos=look_pos,allocation_mask=allocation_mask,next_attempt=next_attempt,B=B,initial_B=initial_B,look_multiplier=look_multiplier,mc_alpha=mc_alpha,fdr_q=fdr_q,fdr_method=fdr_method,seed=seed,noise_signature=noise_sig,run_signature=run_sig,traces=traces,schedule=schedule,failure_counts=failure_counts,adaptive_fraction=adaptive_fraction,adaptive_min=adaptive_min,adaptive_max_lead=adaptive_max_lead,controller_version=controller_version,mmctest_r=(mmctest_r if mc_scheme=="mmctest" else None))
            classify(batch,level); look_pos[batch]+=1; allocation_mask[:]=False
            if cp: _adaptive_checkpoint_save(cp,ids=ids,family_ids=family_ids,T=T,ge=ge,unknown=unknown,n_rep=n_rep,status=status,p_lower=p_lower,p_upper=p_upper,look_pos=look_pos,allocation_mask=allocation_mask,next_attempt=next_attempt,B=B,initial_B=initial_B,look_multiplier=look_multiplier,mc_alpha=mc_alpha,fdr_q=fdr_q,fdr_method=fdr_method,seed=seed,noise_signature=noise_sig,run_signature=run_sig,traces=traces,schedule=schedule,failure_counts=failure_counts,adaptive_fraction=adaptive_fraction,adaptive_min=adaptive_min,adaptive_max_lead=adaptive_max_lead,controller_version=controller_version,mmctest_r=(mmctest_r if mc_scheme=="mmctest" else None))
            if pbar is not None: pbar.refresh()
    finally:
        if pbar is not None: pbar.close()
    p_emp_lo,p_emp_hi=empirical_pvalue_bounds(ge,unknown,np.maximum(n_rep,1)); zero=n_rep==0; p_emp_lo[zero]=0; p_emp_hi[zero]=1; p_emp=np.where((unknown==0)&~zero,p_emp_lo,np.nan); unresolved=np.flatnonzero(status==0); labels=np.asarray(["unresolved","certified_reject","certified_nonreject"],object)[status]; cert=ids[status==1]; non=ids[status==2]
    return {"B":B,"look_schedule":np.asarray(schedule,np.int64),"attempts":next_attempt,"invalid_attempts":int(np.sum(unknown)),"unknown_events":int(np.sum(unknown)),"unknown_hypotheses":int(np.count_nonzero(unknown)),"failure_counts":failure_counts,"complete":len(unresolved)==0,"family_complete":True,"unresolved_count":int(len(unresolved)),"test_statistic":"delta_rss","noise_model_kind":noise_model.kind,"noise_signature":noise_sig,"run_signature":run_sig,"hypothesis_id":ids,"T_obs":T,"delta_rss_obs":T.copy(),"ge_count":ge,"unknown_count":unknown,"n_bootstrap":n_rep,"p_emp":p_emp,"p_emp_lower":p_emp_lo,"p_emp_upper":p_emp_hi,"p_lower":p_lower,"p_upper":p_upper,"status":status,"status_label":labels,"bh_rejected_ids":cert,"certified_reject_ids":cert,"certified_nonreject_ids":non,"fdr_q":fdr_q,"fdr_method":fdr_method,"mc_alpha":mc_alpha,"trace":traces,"family_m":family_m,"family_ids":family_ids,"target_hypothesis_ids":ids.copy(),"mc_controller":mc_scheme,"controller_version":controller_version,"mmctest_r":(mmctest_r if mc_scheme=="mmctest" else np.nan),"look_pos":look_pos,"allocation_mask":allocation_mask,"adaptive_fraction":adaptive_fraction,"adaptive_min":adaptive_min,"adaptive_max_lead":adaptive_max_lead,"total_hypothesis_replicates":int(np.sum(n_rep)),"mean_hypothesis_replicates":float(np.mean(n_rep)),"max_hypothesis_replicates":int(np.max(n_rep))}

def run_empirical_null(data_yxt,theta,spec,m0_results,hypotheses_obs,df_obs,noise_model,*,valid_yxt=None,B=32768,B_fit=128,seed=20261004,checkpoint_path=None,checkpoint_every=32,target_hypothesis_ids=None,trace_hypothesis_ids=(),m0_solver_config=None,m1_solver_config=None,fractions=DEFAULT_FRACTIONS,progress=True,initial_B=64,look_multiplier=2.0,mc_alpha=1e-6,fdr_q=0.01,fdr_method="BY",mc_controller="cp_union",adaptive_fraction=.25,adaptive_min=64,adaptive_max_lead=2,fdr_budget_mode="separate",mmctest_r=10000.0):
    """F-v8: прежний cp_union или строгая adaptive allocation; optional joint FDR/MC budget."""
    ctrl=str(mc_controller).lower(); budget=str(fdr_budget_mode).lower(); q_target=float(fdr_q); eps=float(mc_alpha)
    if budget not in {"separate","joint"}: raise ValueError("fdr_budget_mode должен быть 'separate' или 'joint'")
    if not(0<eps<q_target<1): raise ValueError("Требуется 0 < mc_alpha < fdr_q < 1")
    q_internal=q_target if budget=="separate" else q_target-eps
    kw=dict(valid_yxt=valid_yxt,B=B,B_fit=B_fit,seed=seed,checkpoint_path=checkpoint_path,checkpoint_every=checkpoint_every,target_hypothesis_ids=target_hypothesis_ids,trace_hypothesis_ids=trace_hypothesis_ids,m0_solver_config=m0_solver_config,m1_solver_config=m1_solver_config,fractions=fractions,progress=progress,initial_B=initial_B,look_multiplier=look_multiplier,mc_alpha=eps,fdr_q=q_internal,fdr_method=fdr_method)
    if ctrl=="cp_union": out=_run_empirical_null_union(data_yxt,theta,spec,m0_results,hypotheses_obs,df_obs,noise_model,**kw); out["mc_controller"]="cp_union"
    elif ctrl in {"cp_adaptive","mmctest"}: out=_run_empirical_null_adaptive(data_yxt,theta,spec,m0_results,hypotheses_obs,df_obs,noise_model,adaptive_fraction=adaptive_fraction,adaptive_min=adaptive_min,adaptive_max_lead=adaptive_max_lead,mc_scheme=ctrl,mmctest_r=mmctest_r,**kw)
    else: raise ValueError("mc_controller должен быть 'cp_union', 'cp_adaptive' или 'mmctest'")
    out["mc_controller"]=ctrl; out["total_hypothesis_replicates"]=int(np.sum(out["n_bootstrap"])); out["mean_hypothesis_replicates"]=float(np.mean(out["n_bootstrap"])); out["max_hypothesis_replicates"]=int(np.max(out["n_bootstrap"])); out["fdr_budget_mode"]=budget; out["fdr_q_target"]=q_target; out["fdr_q_internal"]=q_internal; out["mc_resampling_risk_bound"]=eps; out["formal_fdr_bound_if_complete"]=(q_internal+eps if bool(out.get("complete",False)) and bool(out.get("family_complete",False)) else np.nan)
    return out
