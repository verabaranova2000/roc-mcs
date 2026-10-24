from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,math,os,numpy as np
from scipy.special import gammaln
from scipy.stats import beta as beta_dist,nbinom,poisson,kstest

VERSION="peak_splitting_validation_v1"

# Стабильный контракт старых pre-F checkpoint/hash; не версии Python-модуля.
_CHECKPOINT_SCHEMA="noise_validation_pref_v4_resume2phase"
_CHECKPOINT_BASE_SCHEMA="noise_model_certification_v1"

@dataclass(frozen=True,slots=True)
class GeneratorAudit:
    assumption_pass:bool; reason:str; n_draws:int; n_mu:int; family_alpha:float; max_abs_mean_z:float; max_abs_var_z:float; tail_rejections:int; model_kind:str; model_signature:str

@dataclass(frozen=True,slots=True)
class RefitPITAudit:
    assumption_pass:bool; reason:str; n_curves:int; n_obs:int; nonoverlap_lag:int; spatial_phase:tuple; n_cells:int; n_reference:int; n_bootstrap:int; family_alpha:float; max_t_observed:float; max_t_critical:float; global_p:float; model_kind:str; model_signature:str; bandwidth_points:tuple; selected_bandwidth_median:float; unsupported_fraction:float; extrapolated_fraction:float; fano_support:tuple; residual_spatial_corr:float; boundary_selection_fraction:float=0.0

@dataclass(frozen=True,slots=True)
class ProcedureAudit:
    assumption_pass:bool; reason:str; n_trials:int; n_reject:int; rejection_rate:float; cp_upper:float; size_limit:float; n_scenarios:int; model_signature:str; n_inconclusive:int=0

def _kind(model):
    k=str(getattr(model,"kind",""))
    if k not in {"negative_binomial_fano","negative_binomial_fano_curve"}: raise TypeError(f"Требуется NB-Fano plugin, получено kind={k!r}")
    return k

def _fano(mu,model):
    mu=np.asarray(mu,float); k=_kind(model)
    if k=="negative_binomial_fano": F=np.full(mu.shape,float(getattr(model,"fano",np.nan)))
    else:
        fn=getattr(model,"fano_for_mean",None)
        if not callable(fn): raise TypeError("NB-Fano-curve plugin должен иметь fano_for_mean(mu)")
        F=np.asarray(fn(mu),float)
        if F.shape!=mu.shape:
            try: F=np.broadcast_to(F,mu.shape)
            except ValueError as e: raise ValueError(f"fano_for_mean shape={F.shape}, ожидалось {mu.shape}") from e
    if np.any(~np.isfinite(F)) or np.any(F<1): raise ValueError("F(mu) должен быть конечным и >=1")
    return F

def _nb_params(mu,model):
    mu=np.maximum(np.asarray(mu,float),0.0); F=_fano(mu,model); near=F<=1+1e-10; r=np.full(mu.shape,np.inf); p=np.ones(mu.shape); far=(~near)&(mu>0); r[far]=mu[far]/(F[far]-1.0); p[far]=1.0/F[far]; return mu,F,near,r,p

def _cdf(y,mu,model):
    y=np.asarray(y,float); mu,F,near,r,p=_nb_params(mu,model); out=np.empty(np.broadcast_shapes(y.shape,mu.shape),float); y=np.broadcast_to(y,out.shape); mu=np.broadcast_to(mu,out.shape); near=np.broadcast_to(near,out.shape); r=np.broadcast_to(r,out.shape); p=np.broadcast_to(p,out.shape); z=mu<=0; m=near&~z; out[m]=poisson.cdf(y[m],mu[m]); m=(~near)&~z; out[m]=nbinom.cdf(y[m],r[m],p[m]); out[z]=np.where(y[z]>=0,1.0,0.0); return out

def _logpmf(y,mu,model):
    y=np.asarray(y,float); mu,F,near,r,p=_nb_params(mu,model); out=np.full(np.broadcast_shapes(y.shape,mu.shape),-np.inf,float); y=np.broadcast_to(y,out.shape); mu=np.broadcast_to(mu,out.shape); near=np.broadcast_to(near,out.shape); r=np.broadcast_to(r,out.shape); p=np.broadcast_to(p,out.shape); z=mu<=0; out[z&(y==0)]=0.0; m=near&~z; out[m]=poisson.logpmf(y[m],mu[m]); m=(~near)&~z; yy=y[m]; rr=r[m]; pp=p[m]; out[m]=gammaln(yy+rr)-gammaln(rr)-gammaln(yy+1)+rr*np.log(pp)+yy*np.log1p(-pp); return out

def _soft_tail(y,mu,model,a,side):
    y=np.rint(np.asarray(y,float)); Flo=_cdf(y-1,mu,model); Fhi=_cdf(y,mu,model); w=Fhi-Flo; eps=1e-15; good=w>eps; out=np.empty_like(w,float)
    if side=="lower":
        out[~good]=(Fhi[~good]<=float(a)).astype(float); np.divide(float(a)-Flo,w,out=out,where=good); out[good]=np.clip(out[good],0,1); return out
    c=1-float(a); out[~good]=(Flo[~good]>=c).astype(float); np.divide(Fhi-c,w,out=out,where=good); out[good]=np.clip(out[good],0,1); return out

def _cp_upper(k,n,alpha):
    if n<1: return np.nan
    return 1.0 if k>=n else float(beta_dist.ppf(1-float(alpha),k+1,n-k))

def _default_mu_grid(model,n=12):
    c=np.asarray(getattr(model,"centers",[]),float); c=c[np.isfinite(c)&(c>0)]
    if len(c)>=2: lo,hi=float(np.quantile(c,.02)),float(np.quantile(c,.98)); lo=max(lo,.25); hi=max(hi,lo*4)
    else: lo,hi=.5,500.0
    return np.geomspace(lo,hi,int(n))

def certify_nb_generator(noise_model,*,mu_grid=None,n_draws=200_000,family_alpha=1e-4,seed=20261014,batch_size=20_000):
    """Точная проверка реализации simulate_null против NB(mean,F(mu))."""
    _kind(noise_model); mu=_default_mu_grid(noise_model) if mu_grid is None else np.asarray(mu_grid,float); mu=mu[np.isfinite(mu)&(mu>=0)]
    if len(mu)<3 or int(n_draws)<10_000: raise ValueError("Нужны >=3 mu и n_draws>=10000")
    rng=np.random.default_rng(int(seed)); rows=[]; tail_rej=0; zmean=[]; zvar=[]; probs=(.001,.01,.05,.5,.95,.99,.999); alpha_cell=float(family_alpha)/(len(mu)*(2+len(probs)))
    for m in mu:
        chunks=[]; left=int(n_draws)
        while left:
            b=min(int(batch_size),left); z=np.asarray(noise_model.simulate_null(rng,np.full(b,m,float)),float); chunks.append(z); left-=b
        y=np.concatenate(chunks); F=float(_fano(np.asarray([m]),noise_model)[0]); var=F*m; em=float(np.mean(y)); ev=float(np.var(y,ddof=1)); mean_se=math.sqrt(max(var,1e-300)/len(y)); zm=(em-m)/max(mean_se,1e-300)
        if F<=1+1e-10: sk=float(poisson.stats(m,moments="s")); ku=float(poisson.stats(m,moments="k"))
        else:
            r=m/(F-1); p=1/F; sk=float(nbinom.stats(r,p,moments="s")); ku=float(nbinom.stats(r,p,moments="k"))
        mu4=(ku+3)*var*var; var_se=math.sqrt(max(mu4-((len(y)-3)/(len(y)-1))*var*var,1e-300)/len(y)); zv=(ev-var)/max(var_se,1e-300); zmean.append(abs(zm)); zvar.append(abs(zv))
        p0_theo=float(_cdf(np.asarray([0.0]),np.asarray([m]),noise_model)[0]); p0_obs=float(np.mean(y==0)); k0=int(np.sum(y==0)); lo0=0 if k0==0 else float(beta_dist.ppf(alpha_cell/2,k0,len(y)-k0+1)); hi0=1 if k0==len(y) else float(beta_dist.ppf(1-alpha_cell/2,k0+1,len(y)-k0)); reject0=not(lo0<=p0_theo<=hi0); tail_rej+=int(reject0)
        qr=[]
        for pr in probs:
            if F<=1+1e-10: q=int(poisson.ppf(pr,m)); c=float(poisson.cdf(q,m))
            else: q=int(nbinom.ppf(pr,m/(F-1),1/F)); c=float(nbinom.cdf(q,m/(F-1),1/F))
            obs=float(np.mean(y<=q)); k=int(np.sum(y<=q)); lo=0 if k==0 else float(beta_dist.ppf(alpha_cell/2,k,len(y)-k+1)); hi=1 if k==len(y) else float(beta_dist.ppf(1-alpha_cell/2,k+1,len(y)-k)); rej=not(lo<=c<=hi); tail_rej+=int(rej); qr.append((pr,q,c,obs,rej))
        rows.append((m,F,em,ev,zm,zv,p0_theo,p0_obs,reject0,qr))
    # |z|<6 — инженерная проверка моментов; хвосты контролируются exact binomial intervals.
    passed=max(zmean)<6 and max(zvar)<6 and tail_rej==0; reason="PASS: NB generator matches theoretical moments/CDF" if passed else f"FAIL: generator mismatch; max|z_mean|={max(zmean):.2f}, max|z_var|={max(zvar):.2f}, tail rejects={tail_rej}"
    import pandas as pd
    flat=[]
    for m,F,em,ev,zm,zv,p0t,p0o,r0,qr in rows:
        flat.append({"mu":m,"F":F,"mean_emp":em,"mean_theory":m,"mean_z":zm,"var_emp":ev,"var_theory":F*m,"var_z":zv,"p0_emp":p0o,"p0_theory":p0t,"p0_reject":r0})
    audit=GeneratorAudit(passed,reason,int(n_draws),len(mu),float(family_alpha),float(max(zmean)),float(max(zvar)),int(tail_rej),_kind(noise_model),str(noise_model.signature()))
    return {"audit":audit,"table":pd.DataFrame(flat),"mu_grid":mu}

def _select_data(data_yxt,theta,valid_yxt,lag,phase,max_pixels):
    d=np.asarray(data_yxt,float); th=np.asarray(theta,float)
    if d.ndim!=3 or th.ndim!=1 or d.shape[-1]!=len(th): raise ValueError("Ожидаются data_yxt=(y,x,theta) и theta")
    if len(th)<12 or np.any(~np.isfinite(th)) or np.any(np.diff(th)<=0): raise ValueError("theta должна быть конечной, возрастающей, len>=12")
    v=np.isfinite(d) if valid_yxt is None else np.asarray(valid_yxt,bool)
    if v.shape==d.shape[:2]: v=np.broadcast_to(v[...,None],d.shape)
    if v.shape!=d.shape: raise ValueError("valid_yxt shape mismatch")
    vals=d[v&np.isfinite(d)]; integer=float(np.mean(np.abs(vals-np.rint(vals))<=1e-10*np.maximum(1,np.abs(vals)))) if vals.size else 0
    if integer<1-1e-10 or np.any(vals<0): raise ValueError(f"Нужны неотрицательные integer counts; integer-like={integer:.6%}")
    lag=int(lag); py,px=map(int,phase)
    if lag<1 or not(0<=py<lag and 0<=px<lag): raise ValueError("Некорректные lag/phase")
    yy,xx=np.indices(d.shape[:2]); coords=np.column_stack(np.where((yy%lag==py)&(xx%lag==px)))
    if max_pixels is not None and len(coords)>int(max_pixels): coords=coords[np.linspace(0,len(coords)-1,int(max_pixels),dtype=int)]
    y=d[coords[:,0],coords[:,1]]; vm=v[coords[:,0],coords[:,1]]&np.isfinite(y); keep=vm.sum(1)>=max(8,int(.6*len(th))); coords,y,vm=coords[keep],y[keep],vm[keep]
    if len(coords)<30: raise ValueError("После spatial skeleton осталось <30 curves")
    return d.shape,th,coords,y,vm,integer

def _kernel(theta,h,truncate=4.0):
    th=np.asarray(theta,float); step=float(np.median(np.diff(th))); sigma=float(h)*step; dist=th[:,None]-th[None,:]; W=np.exp(-.5*(dist/sigma)**2); W[np.abs(dist)>float(truncate)*sigma]=0; np.fill_diagonal(W,0.0); return W

def _loo_candidates(y,valid,theta,model,bandwidth_points,min_neighbors):
    """LOO candidates; exact boundary events are preserved, not epsilon-clipped."""
    y=np.asarray(y,float); v=np.asarray(valid,bool); ncur,nth=y.shape; preds=[]; supports=[]; losses=[]; impossible=[]; admissible=[]; min_support=max(6,int(.5*nth))
    for h in bandwidth_points:
        W=_kernel(theta,h); den=v.astype(float)@W.T; num=np.where(v,y,0.0)@W.T; pred=np.divide(num,den,out=np.zeros_like(num),where=den>1e-12); nn=v.astype(int)@(W>0).T; ok=v&(nn>=int(min_neighbors))&(den>1e-12)&np.isfinite(pred)&(pred>=0)
        imp=ok&(pred<=0)&(y>0); lp=_logpmf(y,np.maximum(pred,0),model); fin=ok&~imp&np.isfinite(lp); nfin=np.sum(fin,axis=1); loss=np.divide(np.sum(np.where(fin,-lp,0.0),axis=1),nfin,out=np.full(ncur,np.inf),where=nfin>0); support=np.sum(ok,axis=1)
        preds.append(np.maximum(pred,0)); supports.append(ok); losses.append(loss); impossible.append(np.sum(imp,axis=1)); admissible.append(support>=min_support)
    return np.stack(preds),np.stack(supports),np.stack(losses,axis=1),np.stack(impossible,axis=1),np.stack(admissible,axis=1)

def _choose_loo(loss,imp,adm):
    """Lexicographic selector: impossible events, finite NB loss, wider bandwidth."""
    loss=np.asarray(loss,float); imp=np.asarray(imp,int); adm=np.asarray(adm,bool)
    if loss.shape!=imp.shape or loss.shape!=adm.shape: raise ValueError("LOO selector shape mismatch")
    n,b=loss.shape; choice=np.full(n,-1,np.int16); best_imp=np.full(n,np.iinfo(np.int32).max,np.int32); best_loss=np.full(n,np.inf,float)
    for j in range(b):
        same_loss=(np.isinf(loss[:,j])&np.isinf(best_loss))|(np.isfinite(loss[:,j])&np.isfinite(best_loss)&(np.abs(loss[:,j]-best_loss)<=1e-15)); better=adm[:,j]&((imp[:,j]<best_imp)|((imp[:,j]==best_imp)&((loss[:,j]<best_loss-1e-15)|(same_loss&(j>choice)))))
        choice[better]=j; best_imp[better]=imp[better,j]; best_loss[better]=loss[better,j]
    if np.any(choice<0): raise RuntimeError(f"LOO smoother has no admissible bandwidth for {np.sum(choice<0)} curves")
    return choice,best_imp,best_loss

def _fit_mean(y,valid,theta,model,bandwidth_points,min_neighbors):
    P,O,L,I,A=_loo_candidates(y,valid,theta,model,bandwidth_points,min_neighbors); choice,bimp,_=_choose_loo(L,I,A); j=np.arange(len(y)); return P[choice,j,:],O[choice,j,:],choice,bimp

def _fit_mean_batch(yb,valid,theta,model,bandwidth_points,min_neighbors):
    """Batch refit with the identical boundary-aware selector."""
    yb=np.asarray(yb,float); v=np.asarray(valid,bool)
    if yb.ndim!=3 or yb.shape[1:]!=v.shape: raise ValueError("yb должен иметь shape=(batch,curve,theta)")
    B,N,T=yb.shape; bws=tuple(bandwidth_points); H=len(bws); P=np.empty((H,B,N,T),float); O=np.empty((H,B,N,T),bool); L=np.empty((B,N,H),float); I=np.empty((B,N,H),np.int32); A=np.empty((B,N,H),bool); vv=v.astype(float); min_support=max(6,int(.5*T))
    for ih,h in enumerate(bws):
        W=_kernel(theta,h); den=vv@W.T; nn=v.astype(int)@(W>0).T; ok0=v&(nn>=int(min_neighbors))&(den>1e-12); num=np.matmul(np.where(v[None,:,:],yb,0.0),W.T); pred=np.divide(num,den[None,:,:],out=np.zeros_like(num),where=den[None,:,:]>1e-12); pred=np.maximum(pred,0.0); ok=np.broadcast_to(ok0[None,:,:],pred.shape); imp=ok&(pred<=0)&(yb>0); lp=_logpmf(yb,pred,model); fin=ok&~imp&np.isfinite(lp); nfin=np.sum(fin,axis=2); loss=np.divide(np.sum(np.where(fin,-lp,0.0),axis=2),nfin,out=np.full((B,N),np.inf),where=nfin>0); support=np.sum(ok,axis=2)
        P[ih]=pred; O[ih]=ok; L[:,:,ih]=loss; I[:,:,ih]=np.sum(imp,axis=2); A[:,:,ih]=support>=min_support
    fc,fi,_=_choose_loo(L.reshape(B*N,H),I.reshape(B*N,H),A.reshape(B*N,H)); choice=fc.reshape(B,N); bimp=fi.reshape(B,N); bi=np.arange(B)[:,None]; ni=np.arange(N)[None,:]; return P[choice,bi,ni,:],O[choice,bi,ni,:],choice,bimp

def _bin_ids(mu,ok,n_bins):
    x=np.asarray(mu,float); m=np.asarray(ok,bool); vals=x[m]; idx=np.flatnonzero(m.ravel()); order=np.argsort(vals,kind="mergesort"); bid=np.full(x.size,-1,np.int16)
    for b,ii in enumerate(np.array_split(order,int(n_bins))): bid[idx[ii]]=b
    return bid.reshape(x.shape)

def _cell_rates(y,mu,ok,model,n_bins,tail_probs,schema=None):
    bid=_bin_ids(mu,ok,n_bins); rows=[]
    if schema is None:
        schema=[]
        for label in [-1]+list(range(int(n_bins))):
            m=ok if label==-1 else ok&(bid==label); n=int(np.sum(m))
            if n<100: continue
            for a in tail_probs:
                if n*float(a)<8: continue
                for side in ("lower","upper"): schema.append((label,float(a),side))
    for label,a,side in schema:
        m=ok if label==-1 else ok&(bid==int(label)); n=int(np.sum(m))
        if n<50 or n*float(a)<3: raise RuntimeError(f"Bootstrap cell unsupported: bin={label}, tail={a:g}, side={side}, n={n}")
        score=_soft_tail(y[m],mu[m],model,float(a),side); rows.append((label,float(a),side,float(np.mean(score)),n))
    return rows,bid,tuple(schema)

def _simulate_refit_rates(rng,model,mu_gen,valid,theta,bandwidth_points,min_neighbors,n_bins,tail_probs,schema,n_rep,*,batch_size=8,progress=False,label="bootstrap"):
    out=[]; choices=[]; done=0
    while done<int(n_rep):
        b=min(int(batch_size),int(n_rep)-done); ys=np.asarray(model.simulate_null(rng,np.broadcast_to(mu_gen[None,:,:],(b,*mu_gen.shape))),float); mus,oks,chs,bimp=_fit_mean_batch(ys,valid,theta,model,bandwidth_points,min_neighbors)
        for r in range(b):
            rows,_,_=_cell_rates(ys[r],mus[r],oks[r],model,n_bins,tail_probs,schema=schema); out.append([x[3] for x in rows]); choices.append(chs[r])
        done+=b
        if progress and (done%max(1,int(n_rep)//10)<=b-1 or done==int(n_rep)): print(f"{label}: {done}/{n_rep}")
    return np.asarray(out,float),np.asarray(choices,np.int16)

def validate_nb_refit_pit(data_yxt,theta,noise_model,valid_yxt=None,*,nonoverlap_lag=None,spatial_phase=(0,0),bandwidth_points=(.6,.9,1.3,1.8,2.5,3.5,5.0),min_neighbors=2,n_bins=5,tail_probs=(.10,.05,.02,.01,.005),n_reference=399,n_bootstrap=1999,family_alpha=.01,seed=20261014,max_pixels=None,batch_size=8,max_extrapolated_fraction=.05,progress=True):
    """Рефит-bootstrap tail calibration: LOO mean переоценивается в каждой реплике."""
    _kind(noise_model); lag=int(nonoverlap_lag if nonoverlap_lag is not None else getattr(noise_model.audit,"nonoverlap_lag",1)); bws=tuple(map(float,bandwidth_points))
    if any(h<=0 for h in bws) or tuple(sorted(bws))!=bws: raise ValueError("bandwidth_points должны быть положительными и возрастающими")
    if int(n_reference)<199 or int(n_bootstrap)<399: raise ValueError("Для formal run нужны n_reference>=199, n_bootstrap>=399")
    if int(batch_size)<1: raise ValueError("batch_size должен быть >=1")
    shape,th,coords,y,valid,integer=_select_data(data_yxt,theta,valid_yxt,lag,spatial_phase,max_pixels); mu,ok,choice,bimp=_fit_mean(y,valid,th,noise_model,bws,min_neighbors); boundary_fraction=float(np.mean(bimp>0)); unsupported=1-float(np.sum(ok))/float(np.sum(valid)); centers=np.asarray(getattr(noise_model,"centers",[]),float); centers=centers[np.isfinite(centers)&(centers>0)]; fano_support=(float(np.min(centers)),float(np.max(centers))) if len(centers) else (0.0,float("inf")); extrap=float(np.mean((mu[ok]<fano_support[0])|(mu[ok]>fano_support[1]))) if np.any(ok) else 1.0; rows,bid,schema=_cell_rates(y,mu,ok,noise_model,n_bins,tail_probs); rate_obs=np.asarray([x[3] for x in rows],float)
    if len(rows)<10: raise ValueError("Слишком мало пригодных tail cells")
    rng=np.random.default_rng(int(seed)); ref,choice_ref=_simulate_refit_rates(rng,noise_model,mu,valid,th,bws,min_neighbors,n_bins,tail_probs,schema,int(n_reference),batch_size=batch_size,progress=progress,label="reference")
    if ref.shape[1]!=len(rows): raise RuntimeError("Bootstrap cell schema mismatch")
    center=np.mean(ref,0); scale=np.std(ref,0,ddof=1)
    if np.any(~np.isfinite(scale)|(scale<=np.finfo(float).eps)): raise RuntimeError("Degenerate bootstrap scale")
    t_obs=np.abs(rate_obs-center)/scale; max_obs=float(np.max(t_obs)); test,choice_test=_simulate_refit_rates(rng,noise_model,mu,valid,th,bws,min_neighbors,n_bins,tail_probs,schema,int(n_bootstrap),batch_size=batch_size,progress=progress,label="maxT"); tnull=np.max(np.abs(test-center)/scale,axis=1); critical=float(np.quantile(tnull,1-float(family_alpha),method="higher")); p=float((1+np.sum(tnull>=max_obs))/(len(tnull)+1)); passed=bool(p>family_alpha and unsupported<=.05 and extrap<=float(max_extrapolated_fraction))
    reason="PASS: refit-PIT tails compatible with NB-F(mu)" if passed else ("INCONCLUSIVE: >5% observations lack supported LOO mean" if unsupported>.05 else (f"INCONCLUSIVE: F(mu) extrapolation fraction={extrap:.2%} exceeds limit={float(max_extrapolated_fraction):.2%}" if extrap>float(max_extrapolated_fraction) else "FAIL: refit-PIT tails depart from NB-F(mu)"))
    import pandas as pd
    tab=[]
    for j,(label,a,side,_,n) in enumerate(rows): tab.append({"mu_bin":"all" if label==-1 else int(label),"side":side,"tail_prob":a,"n_obs":n,"observed_rate":rate_obs[j],"null_rate":center[j],"null_sd":scale[j],"rate_ratio":rate_obs[j]/center[j] if center[j]>0 else np.inf,"maxT_component":t_obs[j],"reject_cell":bool(t_obs[j]>critical)})
    # descriptive profiles
    bw=np.asarray(bws)[choice]; bw_counts=pd.Series(bw).value_counts().sort_index().rename_axis("bandwidth_points").reset_index(name="n_curves")
    theta_rows=[]
    for j,t in enumerate(th):
        m=ok[:,j]
        if np.sum(m)>=30:
            theta_rows.append({"theta":t,"n":int(np.sum(m)),"mean_mu":float(np.mean(mu[m,j])),"lower01":float(np.mean(_soft_tail(y[m,j],mu[m,j],noise_model,.01,"lower"))),"upper01":float(np.mean(_soft_tail(y[m,j],mu[m,j],noise_model,.01,"upper")))})
    audit=RefitPITAudit(passed,reason,len(y),int(np.sum(ok)),lag,tuple(map(int,spatial_phase)),len(rows),int(n_reference),int(n_bootstrap),float(family_alpha),max_obs,critical,p,_kind(noise_model),str(noise_model.signature()),bws,float(np.median(bw)),unsupported,extrap,fano_support,float(getattr(noise_model.audit,"spatial_corr",np.nan)),boundary_fraction)
    return {"audit":audit,"cell_table":pd.DataFrame(tab),"bandwidth_table":bw_counts,"theta_table":pd.DataFrame(theta_rows),"mu_hat":mu,"support_mask":ok,"bandwidth_choice":bw,"coords_yx":coords,"max_t_null":tnull,"reference_rates":ref,"test_rates":test,"data_shape":shape,"theta":th,"integer_like":integer}

def build_m0_templates(m0_results,spec,theta,*,max_templates=256):
    """Детерминированная библиотека реалистичных mean-профилей из успешных M0 fits."""
    th=np.asarray(theta,float); q=len(spec.param_names); curves=[]
    for g in m0_results:
        K=int(g.get("K",0)); X=np.asarray(g["x"],float); ok=np.asarray(g["success"],bool)&np.isfinite(np.asarray(g["cost"],float))
        if K<1 or X.ndim!=2 or X.shape[1]!=K*q+1: raise ValueError(f"Некорректная M0 group K={K}")
        X=X[ok]
        if not len(X): continue
        comp=X[:,:K*q].reshape(len(X),K,q); mu=np.zeros((len(X),len(th)),float)
        for k in range(K): mu+=np.asarray(spec.batch_func(th,*[comp[:,k,j] for j in range(q)]),float)
        mu+=X[:,-1,None]; good=np.all(np.isfinite(mu),axis=1)&np.all(mu>=0,axis=1); curves.append(mu[good])
    if not curves: raise ValueError("Нет успешных M0 templates")
    C=np.vstack(curves); score=np.max(C,axis=1)+.1*np.mean(C,axis=1); o=np.argsort(score,kind="mergesort")
    if len(o)>int(max_templates): o=o[np.linspace(0,len(o)-1,int(max_templates),dtype=int)]
    return C[o]

def _scenario_mu(theta,n_curves,scenario,model,rng,templates=None):
    th=np.asarray(theta,float); z=(th-th.mean())/max(float(np.ptp(th)),1e-12); c=np.asarray(getattr(model,"centers",[2,10,50,200]),float); c=c[np.isfinite(c)&(c>0)]; lo=float(np.quantile(c,.15)) if len(c) else 2; mid=float(np.quantile(c,.5)) if len(c) else 20; hi=float(np.quantile(c,.85)) if len(c) else 200; base=np.full((n_curves,len(th)),max(.5,lo))
    amp=rng.uniform(.6,1.4,size=(n_curves,1)); shift=rng.normal(0,.08,size=(n_curves,1)); zz=z[None,:]-shift
    if scenario=="templates":
        if templates is None: raise ValueError("scenario=templates требует templates")
        T=np.asarray(templates,float); idx=rng.integers(0,len(T),size=n_curves); scale=rng.uniform(.85,1.15,size=(n_curves,1)); return np.maximum(.1,T[idx]*scale)
    if scenario=="flat_low": return np.full_like(base,max(.5,lo))
    if scenario=="flat_mid": return np.full_like(base,max(2,mid))
    if scenario=="slope": return np.maximum(.1,max(.5,lo)+amp*(mid-lo)*(z[None,:]+1)/2)
    if scenario=="single_broad": return np.maximum(.1,max(.5,lo)+amp*(hi-lo)*np.exp(-.5*(zz/.18)**2))
    if scenario=="single_narrow": return np.maximum(.1,max(.5,lo)+amp*(hi-lo)*np.exp(-.5*(zz/.055)**2))
    if scenario=="shoulder": return np.maximum(.1,max(.5,lo)+amp*(hi-lo)*(np.exp(-.5*(zz/.09)**2)+.38*np.exp(-.5*((zz-.11)/.055)**2)))
    if scenario=="multi_peak": return np.maximum(.1,max(.5,lo)+amp*(hi-lo)*(.7*np.exp(-.5*((zz+.22)/.07)**2)+np.exp(-.5*(zz/.055)**2)+.45*np.exp(-.5*((zz-.27)/.09)**2)))
    raise ValueError(f"Неизвестный scenario={scenario!r}")

def validate_refit_pit_size(noise_model,theta,*,scenarios=("flat_low","flat_mid","slope","single_broad","single_narrow","shoulder","multi_peak"),n_trials_per_scenario=12,n_curves=64,bandwidth_points=(.6,.9,1.3,1.8,2.5,3.5,5.0),n_bins=4,tail_probs=(.10,.05,.02,.01),inner_reference=199,inner_bootstrap=399,family_alpha=.01,size_limit=.05,seed=20261015,templates=None,progress=True):
    """Outer-MC проверка размера самого refit-PIT теста на известных NB-нулевых профилях."""
    _kind(noise_model); th=np.asarray(theta,float); rng=np.random.default_rng(int(seed)); rows=[]; pvals=[]
    for s,sc in enumerate(scenarios):
        for tr in range(int(n_trials_per_scenario)):
            mu=_scenario_mu(th,int(n_curves),sc,noise_model,rng,templates=templates); y=np.asarray(noise_model.simulate_null(rng,mu),float); data=y.reshape(int(n_curves),1,len(th)); valid=np.ones_like(data,bool)
            r=validate_nb_refit_pit(data,th,noise_model,valid,nonoverlap_lag=1,spatial_phase=(0,0),bandwidth_points=bandwidth_points,n_bins=n_bins,tail_probs=tail_probs,n_reference=inner_reference,n_bootstrap=inner_bootstrap,family_alpha=family_alpha,seed=int(seed)+100000*s+tr,max_pixels=None,max_extrapolated_fraction=1.0,progress=False)
            stat_reject=bool(r["audit"].global_p<=float(family_alpha)); inconclusive=bool(r["audit"].reason.startswith("INCONCLUSIVE")); pvals.append(r["audit"].global_p); rows.append((sc,tr,r["audit"].global_p,stat_reject,inconclusive,r["audit"].unsupported_fraction,r["audit"].extrapolated_fraction,r["audit"].boundary_selection_fraction,r["audit"].selected_bandwidth_median))
            if progress: print(f"size {sc}: {tr+1}/{n_trials_per_scenario}; p={r['audit'].global_p:.4g}; reject={stat_reject}; inconclusive={inconclusive}")
    n=len(rows); k=int(sum(x[3] for x in rows)); nincon=int(sum(x[4] for x in rows)); upper=_cp_upper(k,n,.05); rate=k/n; passed=upper<=float(size_limit) and nincon==0; reason=(f"PASS: outer-MC size upper95={upper:.3g} <= {size_limit:g}" if passed else (f"INCONCLUSIVE: {nincon}/{n} outer trials were not evaluable" if nincon else f"FAIL: outer-MC size not certified; rejects={k}/{n}, upper95={upper:.3g}"))
    import pandas as pd
    df=pd.DataFrame(rows,columns=["scenario","trial","global_p","reject","inconclusive","unsupported_fraction","extrapolated_fraction","boundary_selection_fraction","median_bandwidth"]); by=df.groupby("scenario",as_index=False).agg(n=("reject","size"),rejects=("reject","sum"),inconclusive=("inconclusive","sum"),rate=("reject","mean"),median_p=("global_p","median"),boundary_fraction=("boundary_selection_fraction","mean"))
    audit=ProcedureAudit(passed,reason,n,k,rate,upper,float(size_limit),len(tuple(scenarios)),str(noise_model.signature()),nincon)
    return {"audit":audit,"trials":df,"by_scenario":by,"pvalues":np.asarray(pvals,float),"ks_uniform":kstest(np.asarray(pvals,float),"uniform")}

def print_generator_audit(r):
    a=r["audit"]; print("=== NB GENERATOR CERTIFICATION ==="); print(f"PASS={a.assumption_pass}; {a.reason}"); print(f"mu points={a.n_mu}; draws/mu={a.n_draws:,}; max|mean z|={a.max_abs_mean_z:.3g}; max|var z|={a.max_abs_var_z:.3g}; exact-CDF rejects={a.tail_rejections}")

def print_refit_pit_audit(r):
    a=r["audit"]; print("=== NB REFIT-PIT CALIBRATION ==="); print(f"PASS={a.assumption_pass}; {a.reason}"); print(f"curves={a.n_curves:,}; usable obs={a.n_obs:,}; lag={a.nonoverlap_lag}; cells={a.n_cells}; unsupported={a.unsupported_fraction:.2%}; F(mu) extrapolated={a.extrapolated_fraction:.2%} outside {a.fano_support}"); print(f"reference={a.n_reference}; bootstrap={a.n_bootstrap}; alpha={a.family_alpha:g}; maxT={a.max_t_observed:.3g}; critical={a.max_t_critical:.3g}; global p={a.global_p:.4g}"); print(f"bandwidth median={a.selected_bandwidth_median:g}; candidates={a.bandwidth_points}; boundary-selected curves={a.boundary_selection_fraction:.2%}"); print(r["cell_table"].assign(dev=lambda d:np.abs(np.log2(np.maximum(d.rate_ratio,1e-300)))).sort_values("dev",ascending=False).drop(columns="dev").head(12).to_string(index=False))

def print_procedure_audit(r):
    a=r["audit"]; print("=== REFIT-PIT PROCEDURE SIZE CERTIFICATION ==="); print(f"PASS={a.assumption_pass}; {a.reason}"); print(f"outer trials={a.n_trials}; rejects={a.n_reject}; rate={a.rejection_rate:.3%}; CP upper95={a.cp_upper:.3%}; inconclusive={a.n_inconclusive}; scenarios={a.n_scenarios}"); print(r["by_scenario"].to_string(index=False))

def plot_certification(generator=None,refit=None,size=None,*,figsize=(12,8)):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=figsize,constrained_layout=True); A=ax.ravel()
    if generator is not None:
        t=generator["table"]; A[0].plot(t.mu,t.mean_emp/t.mean_theory,"o-",label="mean ratio"); A[0].plot(t.mu,t.var_emp/t.var_theory,"s-",label="variance ratio"); A[0].axhline(1,ls=":",lw=1); A[0].set_xscale("log"); A[0].set(xlabel="mu",ylabel="empirical/theory",title="Generator moments"); A[0].legend(frameon=False,fontsize=8)
    if refit is not None:
        t=refit["cell_table"]; A[1].scatter(t.null_rate,t.observed_rate,s=24); lim=max(.01,float(max(t.null_rate.max(),t.observed_rate.max()))); A[1].plot([0,lim],[0,lim],"k:"); A[1].set(xlabel="bootstrap-null tail rate",ylabel="observed tail rate",title=f"Refit-PIT tails; p={refit['audit'].global_p:.3g}")
        bt=refit["bandwidth_table"]; A[2].bar(bt.bandwidth_points.astype(str),bt.n_curves); A[2].set(xlabel="LOO bandwidth, theta steps",ylabel="curves",title="Selected mean-smoother bandwidth")
    if size is not None:
        p=np.asarray(size["pvalues"],float); A[3].hist(p,bins=np.linspace(0,1,11),histtype="step",density=True); A[3].axhline(1,ls=":",lw=1); A[3].set(xlabel="outer null p-value",ylabel="density",title=f"Procedure null p-values; rejects={size['audit'].n_reject}/{size['audit'].n_trials}")
    return fig,ax

# ==================== Calibrated pre-F gate ====================

@dataclass(frozen=True,slots=True)
class PreFNoiseAudit:
    assumption_pass:bool; decision:str; reason:str; raw_p:float; calibrated_p_estimate:float; calibrated_p_lower:float; calibrated_p_upper:float
    family_alpha:float; outer_mc_alpha:float; n_curves:int; n_anchors:int; model_kind:str; model_signature:str; design_signature:str
    unsupported_fraction:float; extrapolated_fraction:float; boundary_selection_fraction:float; outer_trials_total:int

@dataclass(frozen=True,slots=True)
class RawAuditSnapshot:
    global_p:float; n_curves:int; unsupported_fraction:float; extrapolated_fraction:float; boundary_selection_fraction:float


def _scalar(x):
    a=np.asarray(x)
    if a.ndim!=0: raise ValueError("Ожидался scalar")
    return a.item()

def _plugin(model):
    if str(getattr(model,"kind","")) not in {"negative_binomial_fano","negative_binomial_fano_curve"}: raise TypeError("Нужен NB-Fano noise plugin")
    if not callable(getattr(model,"simulate_null",None)) or not callable(getattr(model,"signature",None)): raise TypeError("noise_model не реализует simulate_null/signature")
    if not bool(getattr(model,"assumption_pass",False)): raise ValueError("noise_model не прошёл E1 audit")
    sig=model.signature()
    if not isinstance(sig,str) or not sig: raise TypeError("noise_model.signature() должен вернуть непустую строку")
    return sig

def _hash_array(h,a):
    a=np.ascontiguousarray(np.asarray(a)); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,np.int64).tobytes()); h.update(a.view(np.uint8))

def _hash_inputs(model_sig,data_yxt,theta,valid_yxt,cfg):
    h=hashlib.sha256(); h.update(_CHECKPOINT_SCHEMA.encode()); h.update(_CHECKPOINT_BASE_SCHEMA.encode()); h.update(model_sig.encode()); h.update(json.dumps(cfg,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
    _hash_array(h,np.asarray(data_yxt)); _hash_array(h,np.asarray(theta)); _hash_array(h,np.asarray([] if valid_yxt is None else valid_yxt,bool)); return h.hexdigest()

def _hash_design(input_sig,theta,coords,valid,anchors):
    h=hashlib.sha256(); h.update(input_sig.encode()); _hash_array(h,theta); _hash_array(h,coords); _hash_array(h,valid)
    for k in sorted(anchors): h.update(k.encode()); _hash_array(h,anchors[k])
    return h.hexdigest()

def _atomic_save(path,payload):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+f".tmp-{os.getpid()}")
    try:
        with open(tmp,"wb") as f: np.savez_compressed(f,**payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass

def _load(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.array(z[k],copy=True) for k in z.files}

def _paths(path):
    if path is None: return None,None
    p=Path(path); suffix=p.suffix if p.suffix else ".npz"; stem=p.with_suffix("")
    return stem.with_name(stem.name+".prelude"+suffix),stem.with_name(stem.name+".outer"+suffix)

def _schedule(max_n,initial,mult):
    max_n=int(max_n); initial=min(int(initial),max_n); mult=float(mult)
    if max_n<1 or initial<1 or mult<=1: raise ValueError("Некорректный outer schedule")
    out=[]; n=initial
    while True:
        out.append(n)
        if n==max_n: return tuple(out)
        n=min(max_n,max(n+1,int(np.ceil(n*mult))))

def _cp(k,n,a):
    k=int(k); n=int(n); a=float(a)
    if not(0<a<.5) or not(0<=k<=n) or n<1: raise ValueError("Некорректные CP параметры")
    lo=0. if k==0 else float(beta_dist.ppf(a,k,n-k+1)); hi=1. if k==n else float(beta_dist.ppf(1-a,k+1,n-k)); return lo,hi

def _alpha_side(look,S,mc_alpha): return float(mc_alpha)*(6.0/(np.pi**2*(int(look)+1)**2))/(2.0*int(S))

def _pass_inevitable(k,n,target,a,alpha):
    """PASS неизбежен к target даже если все будущие trials неуспешны."""
    k,n,target=int(k),int(n),int(target)
    if not(0<=k<=n<=target): raise ValueError("Некорректное состояние look-ahead")
    lo,_=_cp(k,target,a)
    return bool(lo>float(alpha)),float(lo)

def _real_design(data_yxt,theta,valid_yxt,lag,phase,max_pixels):
    shape,th,coords,y,v,integer=_select_data(data_yxt,theta,valid_yxt,int(lag),tuple(map(int,phase)),max_pixels)
    return {"shape":shape,"theta":th,"coords":coords,"y":y,"valid":v,"integer_like":integer}

def _mean_anchors(design,model,bandwidth_points,min_neighbors,raw_mu):
    y,v,th=design["y"],design["valid"],design["theta"]; P,O,L,I,A=_loo_candidates(y,v,th,model,bandwidth_points,min_neighbors); choice,_,_=_choose_loo(L,I,A); j=np.arange(len(y)); selected=P[choice,j,:]
    if not np.allclose(selected,np.asarray(raw_mu,float),rtol=1e-12,atol=1e-12): raise RuntimeError("LOO mean mismatch между raw validator и anchor builder")
    first=np.argmax(A,axis=1); last=A.shape[1]-1-np.argmax(A[:,::-1],axis=1); candidates={"selected":selected,"rough":P[first,j,:],"smooth":P[last,j,:]}; out={}
    for name,mu in candidates.items():
        mu=np.maximum(np.asarray(mu,float),0.0)
        if np.any(~np.isfinite(mu)): raise RuntimeError(f"Нефинитный nuisance mean: {name}")
        if not any(np.allclose(mu,z,rtol=1e-13,atol=1e-13) for z in out.values()): out[name]=mu
    if not out: raise RuntimeError("Не построено ни одного nuisance anchor")
    return out

def _raw(data_yxt,theta,model,valid_yxt,*,bandwidth_points,min_neighbors,n_bins,tail_probs,n_reference,n_bootstrap,family_alpha,seed,max_extrapolated_fraction,progress,nonoverlap_lag=1,spatial_phase=(0,0),max_pixels=None,batch_size=8):
    return validate_nb_refit_pit(data_yxt,theta,model,valid_yxt,nonoverlap_lag=nonoverlap_lag,spatial_phase=spatial_phase,bandwidth_points=bandwidth_points,min_neighbors=min_neighbors,n_bins=n_bins,tail_probs=tail_probs,n_reference=n_reference,n_bootstrap=n_bootstrap,family_alpha=family_alpha,seed=seed,max_pixels=max_pixels,batch_size=batch_size,max_extrapolated_fraction=max_extrapolated_fraction,progress=progress)

def _audit_snapshot(a):
    return RawAuditSnapshot(float(a.global_p),int(a.n_curves),float(a.unsupported_fraction),float(a.extrapolated_fraction),float(a.boundary_selection_fraction))

def _save_prelude(path,input_sig,design_sig,msig,design,anchors,a,mu_hat,generator):
    names=tuple(anchors); payload={"version":np.asarray(_CHECKPOINT_SCHEMA),"base_version":np.asarray(_CHECKPOINT_BASE_SCHEMA),"input_signature":np.asarray(input_sig),"design_signature":np.asarray(design_sig),"model_signature":np.asarray(msig),"anchor_names":np.asarray(names),"theta":np.asarray(design["theta"]),"coords":np.asarray(design["coords"]),"valid":np.asarray(design["valid"],bool),"integer_like":np.asarray(bool(design["integer_like"])),"mu_hat":np.asarray(mu_hat,float),"raw_global_p":np.float64(a.global_p),"raw_n_curves":np.int64(a.n_curves),"unsupported_fraction":np.float64(a.unsupported_fraction),"extrapolated_fraction":np.float64(a.extrapolated_fraction),"boundary_selection_fraction":np.float64(a.boundary_selection_fraction),"generator_ran":np.asarray(generator is not None),"generator_pass":np.asarray(True if generator is None else bool(generator["audit"].assumption_pass))}
    for i,name in enumerate(names): payload[f"anchor_{i}"]=np.asarray(anchors[name],float)
    _atomic_save(path,payload)

def _load_prelude(path,input_sig,msig):
    d=_load(path)
    if str(_scalar(d.get("version","")))!=_CHECKPOINT_SCHEMA or str(_scalar(d.get("base_version","")))!=_CHECKPOINT_BASE_SCHEMA: raise ValueError("Pre-F prelude version mismatch")
    if str(_scalar(d.get("input_signature","")))!=input_sig or str(_scalar(d.get("model_signature","")))!=msig: raise ValueError("Pre-F prelude signature mismatch")
    names=tuple(np.asarray(d["anchor_names"]).astype(str)); anchors={name:np.asarray(d[f"anchor_{i}"],float) for i,name in enumerate(names)}
    design={"theta":np.asarray(d["theta"],float),"coords":np.asarray(d["coords"]),"valid":np.asarray(d["valid"],bool),"integer_like":bool(_scalar(d["integer_like"]))}
    a=RawAuditSnapshot(float(_scalar(d["raw_global_p"])),int(_scalar(d["raw_n_curves"])),float(_scalar(d["unsupported_fraction"])),float(_scalar(d["extrapolated_fraction"])),float(_scalar(d["boundary_selection_fraction"])))
    return d,design,anchors,a,np.asarray(d["mu_hat"],float)

def _outer_payload(msig,input_sig,dsig,names,vals,done,completed_look,pobs,terminal_decision="",terminal_reason="",terminal_lower=np.nan,terminal_upper=np.nan,terminal_target=-1,terminal_anchor=""):
    width=max(1,int(np.max(done)))
    return {"version":np.asarray(_CHECKPOINT_SCHEMA),"base_version":np.asarray(_CHECKPOINT_BASE_SCHEMA),"model_signature":np.asarray(msig),"input_signature":np.asarray(input_sig),"design_signature":np.asarray(dsig),"anchor_names":np.asarray(names),"outer_raw_p":np.asarray(vals[:,:width],float),"done":np.asarray(done,np.int64),"completed_look":np.int64(completed_look),"raw_p_observed":np.float64(pobs),"terminal_decision":np.asarray(terminal_decision),"terminal_reason":np.asarray(terminal_reason),"terminal_lower":np.float64(terminal_lower),"terminal_upper":np.float64(terminal_upper),"terminal_target":np.int64(terminal_target),"terminal_anchor":np.asarray(terminal_anchor)}

def inspect_checkpoint(checkpoint_path):
    pre,outer=_paths(checkpoint_path); out={"prelude":None,"outer":None}
    if pre and pre.exists():
        d=_load(pre); out["prelude"]={"path":str(pre),"raw_p":float(_scalar(d["raw_global_p"])),"n_curves":int(_scalar(d["raw_n_curves"])),"anchors":tuple(np.asarray(d["anchor_names"]).astype(str)),"design_signature":str(_scalar(d["design_signature"]))}
    if outer and outer.exists():
        d=_load(outer); names=tuple(np.asarray(d["anchor_names"]).astype(str)); done=np.asarray(d["done"],np.int64); out["outer"]={"path":str(outer),"completed_look":int(_scalar(d["completed_look"])),"done":dict(zip(names,map(int,done))),"total":int(done.sum())}
    return out

def inspect_lookahead(checkpoint_path,*,family_alpha=.01,outer_mc_alpha=1e-3,outer_initial=32,outer_max=128,outer_multiplier=2.0):
    """Показывает, можно ли уже строго завершить текущий fixed look."""
    pre,outer=_paths(checkpoint_path)
    if pre is None or outer is None or not pre.exists() or not outer.exists(): raise FileNotFoundError("Нужны prelude и outer checkpoint")
    pd=_load(pre); od=_load(outer); names=tuple(np.asarray(od["anchor_names"]).astype(str)); vals=np.asarray(od["outer_raw_p"],float); done=np.asarray(od["done"],np.int64); pobs=float(_scalar(od["raw_p_observed"])); completed=int(_scalar(od["completed_look"])); sched=_schedule(int(outer_max),int(outer_initial),float(outer_multiplier)); look=completed+1
    if look>=len(sched): return {"terminal":True,"reason":"schedule exhausted","done":dict(zip(names,map(int,done)))}
    target=int(sched[look]); a=_alpha_side(look,len(names),float(outer_mc_alpha)); rows=[]
    for si,name in enumerate(names):
        n=int(done[si]); k=int(np.sum(vals[si,:n]<=pobs)) if n else 0; inevitable,lo=_pass_inevitable(k,n,target,a,float(family_alpha)) if n<=target else (False,np.nan); rows.append({"anchor":name,"n":n,"k_le_obs":k,"target":target,"worst_case_CP_lower":lo,"PASS_inevitable":inevitable})
    return {"raw_p_observed":pobs,"look":look,"alpha_side":a,"rows":rows}

def validate_noise_pre_f(data_yxt,theta,e1_results,valid_yxt=None,*,spatial_phase=(0,0),nonoverlap_lag=None,max_pixels=None,bandwidth_points=(.6,.9,1.3,1.8,2.5,3.5,5.0),min_neighbors=2,n_bins=4,tail_probs=(.10,.05,.02,.01),n_reference=199,n_bootstrap=399,family_alpha=.01,max_extrapolated_fraction=.05,max_unsupported_fraction=.05,outer_initial=32,outer_max=512,outer_multiplier=2.0,outer_mc_alpha=1e-3,checkpoint_path=None,seed=20261030,batch_size=8,run_generator_certification=True,generator_n_draws=200_000,generator_family_alpha=1e-4,progress=True):
    """Pre-F gate с двухфазным атомарным resume."""
    if not isinstance(e1_results,dict) or "noise_model" not in e1_results: raise ValueError("e1_results должен содержать noise_model")
    model=e1_results["noise_model"]; msig=_plugin(model); alpha=float(family_alpha); eps=float(outer_mc_alpha)
    if not(0<eps<alpha<1): raise ValueError("Требуется 0 < outer_mc_alpha < family_alpha < 1")
    lag=int(nonoverlap_lag if nonoverlap_lag is not None else getattr(getattr(model,"audit",None),"nonoverlap_lag",1)); phase=tuple(map(int,spatial_phase)); bws=tuple(map(float,bandwidth_points)); tails=tuple(map(float,tail_probs))
    if lag<1 or len(phase)!=2 or int(n_reference)<1 or int(n_bootstrap)<1 or int(outer_initial)<1 or int(outer_max)<1: raise ValueError("Некорректные параметры pre-F")
    cfg={"lag":lag,"phase":list(phase),"max_pixels":None if max_pixels is None else int(max_pixels),"bandwidth_points":list(bws),"min_neighbors":int(min_neighbors),"n_bins":int(n_bins),"tail_probs":list(tails),"n_reference":int(n_reference),"n_bootstrap":int(n_bootstrap),"family_alpha":alpha,"max_extrapolated_fraction":float(max_extrapolated_fraction),"max_unsupported_fraction":float(max_unsupported_fraction),"batch_size":int(batch_size),"outer_initial":int(outer_initial),"outer_multiplier":float(outer_multiplier),"outer_mc_alpha":eps,"seed":int(seed),"run_generator_certification":bool(run_generator_certification),"generator_n_draws":int(generator_n_draws),"generator_family_alpha":float(generator_family_alpha)}
    input_sig=_hash_inputs(msig,data_yxt,theta,valid_yxt,cfg); pre_cp,outer_cp=_paths(checkpoint_path); generator=None; raw=None; resumed_prelude=False
    if pre_cp and pre_cp.exists():
        _,design,anchors,a,mu_hat=_load_prelude(pre_cp,input_sig,msig); resumed_prelude=True
        dsig=_hash_design(input_sig,design["theta"],design["coords"],design["valid"],anchors)
        if progress: print(f"pre-F resume: observed/prelude loaded; raw p={a.global_p:.4g}; anchors={len(anchors)}")
    else:
        if run_generator_certification:
            generator=certify_nb_generator(model,n_draws=int(generator_n_draws),family_alpha=float(generator_family_alpha),seed=int(seed)-1)
            if not generator["audit"].assumption_pass: raise RuntimeError("NB generator certification failed: "+generator["audit"].reason)
        design=_real_design(data_yxt,theta,valid_yxt,lag,phase,max_pixels)
        raw=_raw(data_yxt,theta,model,valid_yxt,bandwidth_points=bws,min_neighbors=min_neighbors,n_bins=n_bins,tail_probs=tails,n_reference=n_reference,n_bootstrap=n_bootstrap,family_alpha=alpha,seed=int(seed),max_extrapolated_fraction=max_extrapolated_fraction,progress=progress,nonoverlap_lag=lag,spatial_phase=phase,max_pixels=max_pixels,batch_size=batch_size)
        a=_audit_snapshot(raw["audit"]); mu_hat=np.asarray(raw["mu_hat"],float)
        if a.unsupported_fraction<=float(max_unsupported_fraction) and a.extrapolated_fraction<=float(max_extrapolated_fraction): anchors=_mean_anchors(design,model,bws,int(min_neighbors),mu_hat)
        else: anchors={}
        dsig=_hash_design(input_sig,design["theta"],design["coords"],design["valid"],anchors)
        if pre_cp: _save_prelude(pre_cp,input_sig,dsig,msig,design,anchors,a,mu_hat,generator)
    if a.unsupported_fraction>float(max_unsupported_fraction) or a.extrapolated_fraction>float(max_extrapolated_fraction):
        reason=(f"INCONCLUSIVE: unsupported={a.unsupported_fraction:.2%} > {float(max_unsupported_fraction):.2%}" if a.unsupported_fraction>float(max_unsupported_fraction) else f"INCONCLUSIVE: F(mu) extrapolation={a.extrapolated_fraction:.2%} > {float(max_extrapolated_fraction):.2%}")
        audit=PreFNoiseAudit(False,"INCONCLUSIVE",reason,a.global_p,np.nan,np.nan,np.nan,alpha,eps,a.n_curves,0,str(model.kind),msig,dsig,a.unsupported_fraction,a.extrapolated_fraction,a.boundary_selection_fraction,0)
        return {"audit":audit,"generator_result":generator,"raw_result":raw,"raw_audit":a,"mu_hat":mu_hat,"anchors":{},"outer_table":None,"checkpoint_paths":{"prelude":str(pre_cp) if pre_cp else None,"outer":str(outer_cp) if outer_cp else None},"resumed_prelude":resumed_prelude}
    names=tuple(anchors); S=len(names); sched=_schedule(int(outer_max),int(outer_initial),float(outer_multiplier)); vals=np.full((S,int(outer_max)),np.nan); done=np.zeros(S,np.int64); completed_look=-1; pobs=float(a.global_p)
    if outer_cp and outer_cp.exists():
        d=_load(outer_cp)
        checks=(str(_scalar(d.get("version","")))==_CHECKPOINT_SCHEMA,str(_scalar(d.get("base_version","")))==_CHECKPOINT_BASE_SCHEMA,str(_scalar(d.get("model_signature","")))==msig,str(_scalar(d.get("input_signature","")))==input_sig,str(_scalar(d.get("design_signature","")))==dsig,tuple(np.asarray(d["anchor_names"]).astype(str))==names)
        if not all(checks): raise ValueError("Pre-F outer checkpoint signature mismatch")
        if float(_scalar(d["raw_p_observed"]))!=pobs: raise ValueError("Pre-F checkpoint observed raw-p mismatch")
        old=np.asarray(d["outer_raw_p"],float); done=np.asarray(d["done"],np.int64); completed_look=int(_scalar(d["completed_look"]))
        if old.ndim!=2 or old.shape[0]!=S or done.shape!=(S,) or old.shape[1]>int(outer_max) or np.any(done<0) or np.any(done>old.shape[1]) or completed_look>=len(sched): raise ValueError("Pre-F outer checkpoint state invalid")
        vals[:,:old.shape[1]]=old
        for si,n in enumerate(done):
            if n and np.any(~np.isfinite(vals[si,:n])): raise ValueError("Pre-F checkpoint содержит пропуски среди завершённых trials")
        if completed_look>=0 and np.any(done<int(sched[completed_look])): raise ValueError("Pre-F completed_look inconsistent")
        terminal_decision=str(_scalar(d.get("terminal_decision",""))); terminal_reason=str(_scalar(d.get("terminal_reason",""))); terminal_lower=float(_scalar(d.get("terminal_lower",np.nan))); terminal_upper=float(_scalar(d.get("terminal_upper",np.nan))); terminal_target=int(_scalar(d.get("terminal_target",-1))); terminal_anchor=str(_scalar(d.get("terminal_anchor","")))
        if progress: print(f"pre-F resume: outer loaded; trials={int(done.sum())}; completed_look={completed_look}; terminal={terminal_decision or 'no'}")
    else: terminal_decision=terminal_reason=terminal_anchor=""; terminal_lower=terminal_upper=np.nan; terminal_target=-1
    decision="INCONCLUSIVE"; reason="INCONCLUSIVE: outer calibration unresolved"; bounds=None; valid3=design["valid"].reshape(len(design["valid"]),1,len(design["theta"]))
    if terminal_decision in {"PASS","FAIL"}:
        decision,reason=terminal_decision,terminal_reason; rows=[]; est=[]
        if terminal_decision=="PASS" and np.isfinite(terminal_lower) and terminal_anchor in names:
            for si,name in enumerate(names):
                n=int(done[si]); k=int(np.sum(vals[si,:n]<=pobs)) if n else 0; pe=(1+k)/(n+1) if n else np.nan; est.append(pe); rows.append((name,n,k,pe,float(terminal_lower) if name==terminal_anchor else 0.0,float(terminal_upper) if name==terminal_anchor else 1.0))
            bounds=(rows,float(np.nanmax(est)) if np.any(np.isfinite(est)) else np.nan,float(terminal_lower),float(terminal_upper))
        else:
            look_for_bounds=max(0,min(completed_look if completed_look>=0 else 0,len(sched)-1)); los=[]; his=[]
            for si,name in enumerate(names):
                n=int(done[si]); k=int(np.sum(vals[si,:n]<=pobs)) if n else 0; lo,hi=_cp(k,n,_alpha_side(look_for_bounds,S,eps)) if n else (0.,1.); pe=(1+k)/(n+1) if n else np.nan; rows.append((name,n,k,pe,lo,hi)); los.append(lo); his.append(hi); est.append(pe)
            bounds=(rows,float(np.nanmax(est)) if np.any(np.isfinite(est)) else np.nan,float(max(los)),float(max(his)))
    for look,target in (() if terminal_decision in {"PASS","FAIL"} else enumerate(sched)):
        if look<=completed_look: continue
        pass_now=False
        for si,name in enumerate(names):
            while done[si]<target:
                n0=int(done[si]); k0=int(np.sum(vals[si,:n0]<=pobs)) if n0 else 0; a_side=_alpha_side(look,S,eps); inevitable,lo_worst=_pass_inevitable(k0,n0,target,a_side,alpha)
                if inevitable:
                    decision="PASS"; reason=f"PASS: anchor {name!r}; fixed-look decision inevitable from {n0}/{target} trials (worst-case CP lower={lo_worst:.4g} > alpha)"; pass_now=True
                    pe=(1+k0)/(n0+1) if n0 else np.nan; rows=[]
                    for sj,nm in enumerate(names):
                        nj=int(done[sj]); kj=int(np.sum(vals[sj,:nj]<=pobs)) if nj else 0
                        if sj==si: rows.append((nm,nj,kj,pe,lo_worst,1.0))
                        elif nj: rows.append((nm,nj,kj,(1+kj)/(nj+1),0.0,1.0))
                        else: rows.append((nm,0,0,np.nan,0.0,1.0))
                    bounds=(rows,float(pe),float(lo_worst),1.0)
                    if outer_cp: _atomic_save(outer_cp,_outer_payload(msig,input_sig,dsig,names,vals,done,completed_look,pobs,decision,reason,lo_worst,1.0,target,name))
                    if progress: print(f"outer {name}: early certified at {n0}/{target}; worst-case CP lower={lo_worst:.4g} > alpha={alpha:g}")
                    break
                tr=int(done[si]); rg=np.random.default_rng(np.random.SeedSequence([int(seed),si,tr,0])); y=np.asarray(model.simulate_null(rg,anchors[name]),float)
                if y.shape!=anchors[name].shape or np.any(~np.isfinite(y)): raise RuntimeError(f"noise_model вернул некорректный outer sample для anchor={name}")
                data=y.reshape(len(y),1,y.shape[1]); inner_seed=int(np.random.SeedSequence([int(seed),si,tr,1]).generate_state(1,dtype=np.uint32)[0])
                rr=_raw(data,design["theta"],model,valid3,bandwidth_points=bws,min_neighbors=min_neighbors,n_bins=n_bins,tail_probs=tails,n_reference=n_reference,n_bootstrap=n_bootstrap,family_alpha=alpha,seed=inner_seed,max_extrapolated_fraction=1.0,progress=False,nonoverlap_lag=1,spatial_phase=(0,0),max_pixels=None,batch_size=batch_size)
                val=float(rr["audit"].global_p)
                if not np.isfinite(val) or not(0<=val<=1): raise RuntimeError("Outer raw p некорректен")
                vals[si,tr]=val; done[si]+=1
                if outer_cp: _atomic_save(outer_cp,_outer_payload(msig,input_sig,dsig,names,vals,done,completed_look,pobs))
                if progress: print(f"outer {name}: {done[si]}/{target}; raw p*={val:.4g}")
            if pass_now: break
            n=int(done[si]); k=int(np.sum(vals[si,:n]<=pobs)); lo,hi=_cp(k,n,_alpha_side(look,S,eps))
            if lo>alpha:
                decision="PASS"; reason=f"PASS: composite-null not rejected; anchor {name!r} has calibrated p lower bound > alpha"; pass_now=True; break
        if pass_now and bounds is not None:
            break
        rows=[]; los=[]; his=[]; est=[]
        for si,name in enumerate(names):
            n=int(done[si])
            if n: k=int(np.sum(vals[si,:n]<=pobs)); lo,hi=_cp(k,n,_alpha_side(look,S,eps)); pe=(1+k)/(n+1)
            else: k=0; lo,hi,pe=0.,1.,np.nan
            rows.append((name,n,k,pe,lo,hi)); los.append(lo); his.append(hi); est.append(pe)
        lower=float(max(los)); upper=float(max(his)); pest=float(np.nanmax(est)) if np.any(np.isfinite(est)) else np.nan; bounds=(rows,pest,lower,upper)
        if pass_now:
            if outer_cp: _atomic_save(outer_cp,_outer_payload(msig,input_sig,dsig,names,vals,done,completed_look,pobs,decision,reason))
            break
        completed_look=look
        if outer_cp: _atomic_save(outer_cp,_outer_payload(msig,input_sig,dsig,names,vals,done,completed_look,pobs))
        if upper<=alpha:
            decision="FAIL"; reason="FAIL: prepivot-calibrated refit-PIT rejects NB-F(mu) for all nuisance-mean anchors"
            if outer_cp: _atomic_save(outer_cp,_outer_payload(msig,input_sig,dsig,names,vals,done,completed_look,pobs,decision,reason))
            break
    if bounds is None: raise RuntimeError("Outer calibration не сформировала bounds")
    rows,pest,lower,upper=bounds
    import pandas as pd
    tab=pd.DataFrame(rows,columns=["anchor","n_outer","count_raw_p_le_obs","calibrated_p_estimate","CP_lower","CP_upper"])
    audit=PreFNoiseAudit(decision=="PASS",decision,reason,pobs,pest,lower,upper,alpha,eps,a.n_curves,S,str(model.kind),msig,dsig,a.unsupported_fraction,a.extrapolated_fraction,a.boundary_selection_fraction,int(np.sum(done)))
    return {"audit":audit,"generator_result":generator,"raw_result":raw,"raw_audit":a,"mu_hat":mu_hat,"anchors":anchors,"outer_table":tab,"outer_raw_p":vals,"outer_done":done,"completed_look":completed_look,"look_schedule":np.asarray(sched,np.int64),"design":{"coords_yx":design["coords"],"valid":design["valid"],"theta":design["theta"],"integer_like":design["integer_like"]},"checkpoint_paths":{"prelude":str(pre_cp) if pre_cp else None,"outer":str(outer_cp) if outer_cp else None},"resumed_prelude":resumed_prelude}

def print_pre_f(r):
    a=r["audit"]; print("=== PRE-F NOISE VALIDATION v4 ==="); print(f"{a.decision}; {a.reason}"); print(f"raw p={a.raw_p:.4g}; calibrated p≈{a.calibrated_p_estimate:.4g}; simultaneous CP=[{a.calibrated_p_lower:.4g},{a.calibrated_p_upper:.4g}]; alpha={a.family_alpha:g}; MC-risk≤{a.outer_mc_alpha:g}"); print(f"curves={a.n_curves:,}; anchors={a.n_anchors}; outer trials total={a.outer_trials_total:,}; unsupported={a.unsupported_fraction:.2%}; extrapolated={a.extrapolated_fraction:.2%}; boundary-selected={a.boundary_selection_fraction:.2%}; resumed_prelude={bool(r.get('resumed_prelude',False))}")
    if r.get("outer_table") is not None: print(r["outer_table"].to_string(index=False))
