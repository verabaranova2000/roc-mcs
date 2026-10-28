from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist

VERSION="profile-nb-lifecycle-v1.0"

@dataclass(slots=True)
class ProfileNBResult:
    selected_J:int;selected_alpha:float;model_table:pd.DataFrame;alpha_trace:pd.DataFrame
    fits:dict[int,Any];candidates:dict[int,dict[str,Any]];fit_kwargs:dict[str,Any];metadata:dict[str,Any]

@dataclass(slots=True)
class BootstrapOrderResult:
    J_null:int;J_alt:int;T_obs:float;p_value:float;p_lower:float;p_upper:float;mc_ci:tuple[float,float]
    n_bootstrap:int;n_valid:int;n_failed:int;table:pd.DataFrame;observed:ProfileNBResult;metadata:dict[str,Any]

def _core():
    from roc_mcs.fitting import dynamic_roc_decomposition as m
    return m

def _clean_kwargs(kwargs):
    out=dict(kwargs or {})
    for k in ("noise","nb_alpha","J_max","seed","n_starts","progress","init_result"):out.pop(k,None)
    return out

def _row_for_J(r,J):
    q=r.model_table.loc[r.model_table["J"].astype(int)==int(J)]
    if len(q)!=1:raise RuntimeError(f"model_table не содержит ровно одну строку J={J}")
    return q.iloc[0]

def _fit_exact_J_fixed_alpha(roc_map,J,alpha,*,seeds,n_starts,fit_kwargs):
    core=_core();runs=[]
    for seed in tuple(map(int,seeds)):
        r=core.fit_lifecycle_dynamic_rocking(roc_map,noise="nb",nb_alpha=float(alpha),J_max=int(J),n_starts=int(n_starts),seed=seed,progress=False,init_result=None,**_clean_kwargs(fit_kwargs))
        row=_row_for_J(r,J);cand=r.candidates[int(J)];ll=float(cand.get("loglik",row["loglik"]))
        runs.append({"fit":r,"candidate":cand,"row":row,"seed":seed,"loglik":ll,"success":bool(row.get("success",True)),"start_spread":float(row.get("start_spread",np.nan))})
    if not runs:raise RuntimeError(f"Не получен fit для J={J}, alpha={alpha}")
    best=max(runs,key=lambda z:z["loglik"]);vals=np.asarray([z["loglik"] for z in runs],float);best["seed_loglik_spread"]=float(np.ptp(vals));best["n_seed_success"]=int(sum(z["success"] for z in runs));return best

def fit_lifecycle_profile_nb(roc_map,*,J_max=4,J_values=None,alpha_bounds=(1e-3,3.),alpha_grid_points=11,alpha_log_tol=2e-3,max_profile_iter=24,seeds=(20261019,20261020,20261021),n_starts=8,fit_kwargs=None,progress=True,require_interior=True):
    """NB2 profile-MLE: для каждого J профилирует nuisance dispersion alpha."""
    Js=tuple(range(1,int(J_max)+1)) if J_values is None else tuple(sorted(set(map(int,J_values))))
    if not Js or min(Js)<1 or max(Js)>int(J_max):raise ValueError("Некорректные J_values")
    alo,ahi=map(float,alpha_bounds)
    if not(0<alo<ahi) or int(alpha_grid_points)<5:raise ValueError("Нужны 0<alpha_bounds[0]<alpha_bounds[1] и alpha_grid_points>=5")
    log_grid=np.linspace(np.log(alo),np.log(ahi),int(alpha_grid_points));trace=[];fits={};cands={};rows=[];iterator=Js
    if progress:
        from tqdm.auto import tqdm
        iterator=tqdm(Js,desc="Profile NB2 lifecycle",unit="J")
    for J in iterator:
        cache={}
        def ev(loga,stage):
            a=float(np.exp(loga));key=round(float(loga),12)
            if key not in cache:
                z=_fit_exact_J_fixed_alpha(roc_map,J,a,seeds=seeds,n_starts=n_starts,fit_kwargs=fit_kwargs);cache[key]=z
                trace.append({"J":J,"alpha":a,"log_alpha":float(loga),"loglik":z["loglik"],"seed":z["seed"],"start_spread":z["start_spread"],"seed_loglik_spread":z["seed_loglik_spread"],"n_seed_success":z["n_seed_success"],"success":z["success"],"stage":stage})
            return cache[key]
        grid=[ev(x,"grid") for x in log_grid];ll=np.array([z["loglik"] for z in grid]);ib=int(np.nanargmax(ll));boundary=ib in (0,len(log_grid)-1);opts=[(log_grid[ib],grid[ib])]
        if not boundary:
            lo,hi=float(log_grid[ib-1]),float(log_grid[ib+1]);opt=minimize_scalar(lambda x:-ev(x,"refine")["loglik"],bounds=(lo,hi),method="bounded",options={"xatol":float(alpha_log_tol),"maxiter":int(max_profile_iter)})
            opts.append((float(opt.x),ev(float(opt.x),"refine_final")))
        loga,z=max(opts,key=lambda q:q[1]["loglik"]);a=float(np.exp(loga));row=z["row"];nobs=int(np.asarray(z["fit"].valid,bool).sum());p_mean=int(row["npar"]);p_total=p_mean+1;bic=-2*z["loglik"]+p_total*np.log(nobs)
        rows.append({"J":J,"alpha_hat":a,"loglik":z["loglik"],"nobs":nobs,"npar_mean":p_mean,"npar_total":p_total,"bic":bic,"seed":z["seed"],"start_spread":z["start_spread"],"seed_loglik_spread":z["seed_loglik_spread"],"n_seed_success":z["n_seed_success"],"success":z["success"],"alpha_boundary":boundary});fits[J]=z["fit"];cands[J]=z["candidate"]
    tab=pd.DataFrame(rows).sort_values("J").reset_index(drop=True);tab["delta_bic"]=tab["bic"]-tab["bic"].min();w=np.exp(-.5*np.clip(tab["delta_bic"].to_numpy(float),0,1400));tab["bic_weight"]=w/w.sum();bad=tab["alpha_boundary"].to_numpy(bool)
    if require_interior and np.any(bad):raise RuntimeError("Profile alpha достиг границы поиска для J="+",".join(map(str,tab.loc[bad,"J"].tolist()))+"; расширьте alpha_bounds")
    q=tab.loc[tab["bic"].idxmin()];J=int(q["J"]);a=float(q["alpha_hat"]);status="OK" if bool(tab["success"].all()) else "INCONCLUSIVE_OPTIMIZATION"
    meta={"version":VERSION,"status":status,"alpha_bounds":(alo,ahi),"alpha_grid_points":int(alpha_grid_points),"alpha_log_tol":float(alpha_log_tol),"seeds":tuple(map(int,seeds)),"n_starts":int(n_starts),"selection":"profile NB2 maximum likelihood + BIC; alpha counted as one nuisance parameter","core_version":getattr(_core(),"VERSION","unknown")}
    return ProfileNBResult(J,a,tab,pd.DataFrame(trace).sort_values(["J","alpha"]).reset_index(drop=True),fits,cands,_clean_kwargs(fit_kwargs),meta)

def get_selected_core_fit(r:ProfileNBResult,*,require_exact=True):
    """Возвращает core-result при alpha_hat выбранной J."""
    z=r.fits[int(r.selected_J)]
    if require_exact and int(z.selected_J)!=int(r.selected_J):raise RuntimeError(f"Core при alpha_hat_J выбрал J={z.selected_J}, profile-selection выбрал J={r.selected_J}; используйте profile result/candidate, не core selected fields")
    return z

def print_profile_summary(r:ProfileNBResult):
    print(f"=== PROFILE NB2 LIFECYCLE {VERSION} ===");print(f"selected J={r.selected_J}; alpha_hat={r.selected_alpha:.6g}; status={r.metadata['status']}; core={r.metadata['core_version']}")
    print(r.model_table[["J","alpha_hat","loglik","bic","delta_bic","bic_weight","start_spread","seed_loglik_spread","alpha_boundary"]].to_string(index=False,float_format=lambda x:f"{x:.6g}"))

def plot_alpha_profiles(r:ProfileNBResult,figsize=(7.5,4.5)):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=figsize,constrained_layout=True)
    for J,g in r.alpha_trace.groupby("J"):
        g=g.sort_values("alpha");m=g["loglik"].max();ax.plot(g["alpha"],2*(m-g["loglik"]),marker="o",ms=3,label=f"J={J}")
    ax.set_xscale("log");ax.set(xlabel=r"NB2 dispersion $\alpha$",ylabel=r"$2[\ell_{\max,J}-\ell_J(\alpha)]$",title="NB2 profile likelihood");ax.grid(alpha=.2);ax.legend(frameon=False);return fig,ax

def plot_profile_selection(r:ProfileNBResult,figsize=(6.5,4.)):
    import matplotlib.pyplot as plt
    d=r.model_table;fig,ax=plt.subplots(figsize=figsize,constrained_layout=True);ax.plot(d["J"],d["delta_bic"],"o-");ax.axhline(0,color="0.5",ls="--",lw=.8);ax.set(xlabel="J",ylabel=r"Profiled $\Delta$BIC",xticks=d["J"],title="Lifecycle model selection with profiled NB2 dispersion");ax.grid(alpha=.2);return fig,ax

def _simulate_nb2(rng,mu,alpha):
    mu=np.maximum(np.asarray(mu,float),1e-12);a=float(alpha)
    if not np.isfinite(a) or a<=0:raise ValueError("alpha должен быть >0")
    size=1./a;p=size/(size+mu);return rng.negative_binomial(size,p).astype(float)

def _full_valid_from_fit(roc_map,fit,source):
    y=np.asarray(roc_map[source],float);v=np.isfinite(y);mask=np.asarray(fit.metadata.get("theta_mask"),bool)
    if mask.shape!=(y.shape[1],):raise RuntimeError("Core result не содержит совместимый theta_mask")
    if np.asarray(fit.valid).shape!=(y.shape[0],int(mask.sum())):raise RuntimeError("Core valid несовместим с theta_mask")
    v[:,mask]=np.asarray(fit.valid,bool);return v,mask

def bootstrap_lifecycle_order(roc_map,*,J_null,J_alt,observed:ProfileNBResult|None=None,B=199,seed=20261101,seeds_fit=(20261019,20261020,20261021),n_starts=6,profile_kwargs=None,fit_kwargs=None,progress=True):
    """Parametric-bootstrap LRT J_null→J_alt с полным NB2 profiling alpha."""
    J0,J1=int(J_null),int(J_alt)
    if J1<=J0 or int(B)<1:raise ValueError("Требуется J_alt>J_null и B>=1")
    pk=dict(profile_kwargs or {});pk.update({"J_max":max(J0,J1),"J_values":(J0,J1),"seeds":tuple(map(int,seeds_fit)),"n_starts":int(n_starts),"fit_kwargs":fit_kwargs,"progress":False})
    if observed is None:observed=fit_lifecycle_profile_nb(roc_map,**pk)
    t=observed.model_table.set_index("J")
    if J0 not in t.index or J1 not in t.index:raise ValueError("observed profile не содержит обе модели")
    Tobs=2*(float(t.loc[J1,"loglik"])-float(t.loc[J0,"loglik"]));fit0=observed.fits[J0];cand0=observed.candidates[J0];mu0=np.asarray(cand0["mu"],float);a0=float(t.loc[J0,"alpha_hat"]);source=str((fit_kwargs or {}).get("source","intensity"));full_valid,mask=_full_valid_from_fit(roc_map,fit0,source);raw=np.asarray(roc_map[source],float);rows=[];iterator=range(int(B))
    if progress:
        from tqdm.auto import tqdm
        iterator=tqdm(iterator,total=int(B),desc=f"Bootstrap J{J0}→J{J1}",unit="rep")
    for b in iterator:
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(b)]));sim=np.array(raw,copy=True);sim[:,mask]=_simulate_nb2(rng,mu0,a0);rm=dict(roc_map);rm[source]=sim;fk=dict(fit_kwargs or {});fk["valid"]=full_valid;fk["spike_filter"]=False
        try:
            rr=fit_lifecycle_profile_nb(rm,**{**pk,"fit_kwargs":fk});tt=rr.model_table.set_index("J");T=2*(float(tt.loc[J1,"loglik"])-float(tt.loc[J0,"loglik"]))
            rows.append({"replicate":b,"success":True,"T":T,"alpha_null":float(tt.loc[J0,"alpha_hat"]),"alpha_alt":float(tt.loc[J1,"alpha_hat"]),"error":""})
        except Exception as e:rows.append({"replicate":b,"success":False,"T":np.nan,"alpha_null":np.nan,"alpha_alt":np.nan,"error":f"{type(e).__name__}: {e}"})
    tab=pd.DataFrame(rows);ok=tab["success"].to_numpy(bool);nv=int(ok.sum());nf=int((~ok).sum());ge=int(np.sum(tab.loc[ok,"T"].to_numpy(float)>=Tobs));B=int(B);p=np.nan if nf else (1+ge)/(B+1);pl=(1+ge)/(B+1);pu=(1+ge+nf)/(B+1)
    if nv:
        lo=0. if ge==0 else float(beta_dist.ppf(.025,ge,nv-ge+1));hi=1. if ge==nv else float(beta_dist.ppf(.975,ge+1,nv-ge))
    else:lo,hi=0.,1.
    meta={"version":VERSION,"null_alpha":a0,"simulation":"NB2 conditional on fitted J_null mean; observed valid/spike mask fixed in bootstrap","p_definition":"(1+#(T*>=Tobs))/(B+1); failed refits produce conservative p bounds","status":"OK" if nf==0 else "INCONCLUSIVE_COMPUTATIONAL_FAILURES"}
    return BootstrapOrderResult(J0,J1,Tobs,float(p),float(pl),float(pu),(lo,hi),B,nv,nf,tab,observed,meta)

def print_bootstrap_summary(r:BootstrapOrderResult):
    print(f"=== PARAMETRIC BOOTSTRAP J={r.J_null} -> J={r.J_alt} ===");print(f"T_obs={r.T_obs:.6g}; B={r.n_bootstrap}; valid={r.n_valid}; failed={r.n_failed}; status={r.metadata['status']}")
    if np.isfinite(r.p_value):print(f"p_boot={r.p_value:.6g}; MC 95% CI for exceedance probability=[{r.mc_ci[0]:.6g},{r.mc_ci[1]:.6g}]")
    else:print(f"p_boot INCONCLUSIVE; conservative bounds=[{r.p_lower:.6g},{r.p_upper:.6g}]")
