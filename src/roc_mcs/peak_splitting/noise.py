from __future__ import annotations
from abc import ABC,abstractmethod
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
from scipy.stats import spearmanr

VERSION="noise_plugins_v3_overlap"

@dataclass(frozen=True,slots=True)
class NoiseAudit:
    assumption_pass:bool; reason:str; theta_corr:float=np.nan; spatial_corr:float=np.nan; integer_like:float=np.nan; zero_fraction:float=np.nan
    slope:float=np.nan; intercept:float=np.nan; fano:float=np.nan; fano_cv:float=np.nan; n_bins:int=0
    adjacent_spatial_corr:float=np.nan; expected_overlap_corr:float=np.nan; nonoverlap_lag:int=1; curve_order_cv:float=np.nan

class NoiseModel(ABC):
    kind="base"; scope="marginal"; version=VERSION
    def __init__(self,audit:NoiseAudit): self.audit=audit
    @property
    def assumption_pass(self): return bool(self.audit.assumption_pass)
    @abstractmethod
    def simulate_null(self,rng,mu): ...
    @abstractmethod
    def payload(self): ...
    def signature(self):
        p={"version":self.version,"kind":self.kind,**self.payload()}; return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

class GaussianHomoskedastic(NoiseModel):
    kind="gaussian_homoskedastic"
    def __init__(self,sigma,audit,lower_bound=None): super().__init__(audit); self.sigma=float(sigma); self.lower_bound=lower_bound
    def simulate_null(self,rng,mu):
        y=np.asarray(mu,float)+rng.normal(size=np.shape(mu))*self.sigma; return np.maximum(y,self.lower_bound) if self.lower_bound is not None else y
    def payload(self): return {"sigma":self.sigma,"lower_bound":self.lower_bound}

class GaussianMeanVariance(NoiseModel):
    kind="gaussian_mean_variance"
    def __init__(self,centers,sigmas,audit,lower_bound=None): super().__init__(audit); self.centers=np.asarray(centers,float); self.sigmas=np.asarray(sigmas,float); self.lower_bound=lower_bound
    def sigma_for_mean(self,mu):
        o=np.argsort(self.centers); x=self.centers[o]; s=self.sigmas[o]; xu,i=np.unique(x,return_index=True); su=s[i]
        return np.exp(np.interp(np.asarray(mu,float),xu,np.log(su),left=np.log(su[0]),right=np.log(su[-1])))
    def simulate_null(self,rng,mu):
        y=np.asarray(mu,float)+rng.normal(size=np.shape(mu))*self.sigma_for_mean(mu); return np.maximum(y,self.lower_bound) if self.lower_bound is not None else y
    def payload(self): return {"centers":np.round(self.centers,14).tolist(),"sigmas":np.round(self.sigmas,14).tolist(),"lower_bound":self.lower_bound}

class Poisson(NoiseModel):
    kind="poisson"
    def simulate_null(self,rng,mu): return rng.poisson(np.maximum(np.asarray(mu,float),0.0)).astype(float)
    def payload(self): return {}

class NegativeBinomialFano(NoiseModel):
    """NB1: E[Y]=mu, Var[Y]=F*mu, F>=1."""
    kind="negative_binomial_fano"
    def __init__(self,fano,audit):
        super().__init__(audit); self.fano=float(fano)
        if not np.isfinite(self.fano) or self.fano<1: raise ValueError("fano должен быть >=1")
    def simulate_null(self,rng,mu):
        mu=np.maximum(np.asarray(mu,float),0.0); out=np.zeros(mu.shape,float); pos=mu>0
        if self.fano<=1+1e-10: out[pos]=rng.poisson(mu[pos]); return out
        r=mu[pos]/(self.fano-1.0); p=1.0/self.fano; out[pos]=rng.negative_binomial(r,p); return out
    def payload(self): return {"fano":self.fano}

class NegativeBinomialFanoCurve(NoiseModel):
    """Маргинальная NB: E[Y]=mu, Var[Y]=F(mu)*mu."""
    kind="negative_binomial_fano_curve"
    def __init__(self,centers,fanos,audit):
        super().__init__(audit); x=np.asarray(centers,float); f=np.asarray(fanos,float); good=np.isfinite(x)&np.isfinite(f)&(x>0)&(f>=1)
        x,f=x[good],f[good]
        if len(x)<3: raise ValueError("Для F(mu) нужны минимум 3 опорные точки")
        o=np.argsort(x); x,f=x[o],f[o]; xu,idx=np.unique(x,return_index=True); self.centers=xu; self.fanos=f[idx]
    def fano_for_mean(self,mu):
        mu=np.asarray(mu,float); tiny=np.finfo(float).tiny; lx=np.log(self.centers); lf=np.log(self.fanos)
        z=np.log(np.maximum(mu,tiny)); return np.exp(np.interp(z,lx,lf,left=lf[0],right=lf[-1]))
    def simulate_null(self,rng,mu):
        mu=np.maximum(np.asarray(mu,float),0.0); out=np.zeros(mu.shape,float); pos=mu>0
        if not np.any(pos): return out
        F=np.maximum(self.fano_for_mean(mu[pos]),1.0)
        near=F<=1+1e-10
        if np.any(near): out_pos=np.empty_like(F); out_pos[near]=rng.poisson(mu[pos][near])
        else: out_pos=np.empty_like(F)
        far=~near
        if np.any(far):
            m=mu[pos][far]; ff=F[far]; out_pos[far]=rng.negative_binomial(m/(ff-1.0),1.0/ff)
        out[pos]=out_pos; return out
    def payload(self): return {"centers":np.round(self.centers,14).tolist(),"fanos":np.round(self.fanos,14).tolist()}

class PoissonGaussian(NoiseModel):
    """Y=g*Pois(mu/g)+N(0,sigma_read^2); Var=g*mu+sigma_read^2."""
    kind="poisson_gaussian"
    def __init__(self,gain,sigma_read,audit,clip_zero=False):
        super().__init__(audit); self.gain=float(gain); self.sigma_read=float(sigma_read); self.clip_zero=bool(clip_zero)
        if self.gain<=0 or self.sigma_read<0: raise ValueError("gain>0, sigma_read>=0")
    def simulate_null(self,rng,mu):
        mu=np.maximum(np.asarray(mu,float),0.0); y=self.gain*rng.poisson(mu/self.gain)+rng.normal(0.0,self.sigma_read,mu.shape)
        return np.maximum(y,0.0) if self.clip_zero else y
    def payload(self): return {"gain":self.gain,"sigma_read":self.sigma_read,"clip_zero":self.clip_zero}

def noise_signature(model):
    if not isinstance(model,NoiseModel): raise TypeError("Ожидался NoiseModel")
    return model.signature()

def simulate_bootstrap_into(rng,data_star,state,mu_state,model,*,theta=None,valid_yxt=None):
    if not isinstance(model,NoiseModel): raise TypeError("Ожидался NoiseModel")
    if model.scope not in {"independent","marginal"}: raise NotImplementedError("Для whole-field dependent model нужен отдельный генератор")
    for K in sorted(state):
        g=state[K]; data_star[g.y,g.x]=model.simulate_null(rng,mu_state[K])
    return data_star

# ==================== Count-noise calibration ====================

IQR_NORMAL=1.3489795003921634

def _validate(data_yxt,theta,valid_yxt):
    d=np.asarray(data_yxt,float); th=np.asarray(theta,float)
    if d.ndim!=3 or th.ndim!=1 or d.shape[-1]!=len(th): raise ValueError("Ожидаются data_yxt=(y,x,theta) и theta")
    if len(th)<7 or not np.all(np.isfinite(th)) or not np.all(np.diff(th)>0): raise ValueError("theta должна быть конечной и строго возрастающей")
    v=np.isfinite(d) if valid_yxt is None else np.asarray(valid_yxt,bool)
    if v.shape==d.shape[:2]: v=np.broadcast_to(v[...,None],d.shape)
    if v.shape!=d.shape: raise ValueError("Некорректная valid_yxt")
    return d,th,v&np.isfinite(d)

def _w(x,order):
    z=(np.asarray(x,float)-np.mean(x))/max(float(np.ptp(x)),np.finfo(float).tiny); V=np.vstack([z**k for k in range(order)]); _,_,vh=np.linalg.svd(V,full_matrices=True); w=vh[-1]/np.linalg.norm(vh[-1]); return -w if w[-1]<0 else w

def _collect(d,th,v,order,max_n):
    cs=[]; ls=[]
    for j in range(len(th)-order):
        sl=slice(j,j+order+1); w=_w(th[sl],order); b=d[...,sl]; ok=np.all(v[...,sl],axis=-1); c=np.tensordot(b,w,axes=([-1],[0])); level=np.mean(b,axis=-1); good=ok&np.isfinite(c)&np.isfinite(level)
        if np.any(good): cs.append(c[good]); ls.append(level[good])
    if not cs: raise ValueError(f"Нет contrast observations для order={order}")
    c=np.concatenate(cs); l=np.concatenate(ls)
    if len(c)>max_n:
        i=np.linspace(0,len(c)-1,max_n,dtype=np.int64); c=c[i]; l=l[i]
    return c,l

def _corr(a,b,min_n=1000):
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel(); ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<min_n: return np.nan
    x=a[ok]-np.mean(a[ok]); y=b[ok]-np.mean(b[ok]); den=np.sqrt(np.dot(x,x)*np.dot(y,y)); return float(np.dot(x,y)/den) if den>0 else np.nan

def _contrast_cube(d,th,v,order):
    out=[]; width=order+1
    for j in range(0,len(th)-order,width):
        sl=slice(j,j+width); w=_w(th[sl],order); b=d[...,sl]; ok=np.all(v[...,sl],axis=-1); c=np.tensordot(b,w,axes=([-1],[0])); out.append(np.where(ok,c,np.nan))
    return np.stack(out,axis=-1)

def _spatial_corr(cube,lag):
    sc=[]; lag=int(max(1,lag))
    if cube.shape[1]>lag: sc.append(_corr(cube[:,:-lag,:],cube[:,lag:,:]))
    if cube.shape[0]>lag: sc.append(_corr(cube[:-lag,:,:],cube[lag:,:,:]))
    return float(np.nanmax(np.abs(sc))) if sc and np.any(np.isfinite(sc)) else np.nan

def _dependence(d,th,v,order,nonoverlap_lag):
    cube=_contrast_cube(d,th,v,order); tc=_corr(cube[...,:-1],cube[...,1:]) if cube.shape[-1]>1 else np.nan
    return tc,_spatial_corr(cube,1),_spatial_corr(cube,nonoverlap_lag)

def _bins(level,c,n_bins,min_bin):
    idx=np.argsort(level,kind="mergesort"); rows=[]
    for ii in np.array_split(idx,n_bins):
        if len(ii)<min_bin: continue
        z=c[ii]; q1,q3=np.quantile(z,[.25,.75]); rows.append((float(np.median(level[ii])),len(ii),float((q3-q1)/IQR_NORMAL),float(np.std(z)),float(np.mean(z==np.median(z)))))
    return np.asarray(rows,float)

def _fano_global(c,level):
    mu=float(np.mean(level)); var=float(np.var(c)); return (var/mu if mu>0 else np.nan),mu,var

def _linear_diag(tab,F):
    good=(tab[:,0]>0)&(tab[:,3]>0)&np.isfinite(tab[:,3]); x=tab[good,0]; var=tab[good,3]**2
    if len(x)<4: return np.nan,np.nan,np.inf,len(x)
    slope,intercept=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),var,rcond=None)[0]; keep=x>=np.quantile(x,.25); pred=F*x[keep]; rel=float(np.sqrt(np.mean(((var[keep]-pred)/np.maximum(pred,np.finfo(float).tiny))**2)))
    return float(slope),float(intercept),rel,len(x)

def _fano_curve(tab):
    good=(tab[:,0]>0)&(tab[:,3]>0)&np.isfinite(tab[:,3]); x=tab[good,0]; f=(tab[good,3]**2)/x
    good=np.isfinite(f)&(f>=1); return x[good],f[good]

def _curve_order_cv(tables,primary_order):
    xp,fp=_fano_curve(tables[primary_order])
    if len(xp)<3: return np.inf
    vals=[]; lx=np.log(xp)
    for o,t in tables.items():
        if o==primary_order: continue
        x,f=_fano_curve(t)
        if len(x)<3: continue
        lo=max(x.min(),xp.min()); hi=min(x.max(),xp.max()); use=(xp>=lo)&(xp<=hi)
        if use.sum()<3: continue
        fi=np.exp(np.interp(lx[use],np.log(x),np.log(f))); vals.append(np.sqrt(np.mean(((fi-fp[use])/np.maximum(fp[use],1.0))**2)))
    return float(np.max(vals)) if vals else np.inf

def fit_count_noise(data_yxt,theta,valid_yxt=None,*,orders=(2,3,4),primary_order=4,n_bins=12,min_bin=1000,max_contrasts=2_000_000,dependence_limit=.10,fano_cv_limit=.25,curve_order_cv_limit=.35,order_fano_spread_limit=.25,poisson_tol=.30,intercept_ratio_limit=.10,pool_size=1,pool_stride=1,allow_fano_curve=True):
    """Калибрует маргинальную count-модель; overlap проверяется на неперекрывающемся spatial lag."""
    d,th,v=_validate(data_yxt,theta,valid_yxt); orders=tuple(sorted(set(map(int,orders))))
    if primary_order not in orders: raise ValueError("primary_order должен входить в orders")
    pool_size=int(pool_size); pool_stride=int(pool_stride)
    if pool_size<1 or pool_stride<1: raise ValueError("pool_size и pool_stride должны быть >=1")
    nonoverlap_lag=max(1,int(math.ceil(pool_size/pool_stride)))
    expected_overlap=max(0.0,1.0-pool_stride/pool_size) if pool_stride<pool_size else 0.0
    tables={}; fs=[]; primary=None
    for o in orders:
        c,l=_collect(d,th,v,o,int(max_contrasts)); tab=_bins(l,c,int(n_bins),int(min_bin)); tables[o]=tab; F,_,_=_fano_global(c,l); fs.append(F)
        if o==primary_order: primary=(c,l,tab,F)
    c,l,tab,F=primary; finite=np.asarray([x for x in fs if np.isfinite(x)],float); fspread=float((finite.max()-finite.min())/np.median(finite)) if len(finite)>1 else np.inf
    slope,intercept,fcv,n=_linear_diag(tab,F); curve_cv=_curve_order_cv(tables,primary_order)
    z=d[v]; scale=np.maximum(1.0,np.abs(z)); integer=float(np.mean(np.abs(z-np.rint(z))<=1e-10*scale)); zero=float(np.mean(z==0)); tc,adj_sc,sc=_dependence(d,th,v,int(primary_order),nonoverlap_lag)
    dep_ok=(not np.isfinite(tc) or abs(tc)<=dependence_limit) and (not np.isfinite(sc) or abs(sc)<=dependence_limit)
    medvar=float(np.median((tab[(tab[:,0]>0)&(tab[:,3]>0),3])**2)) if np.any((tab[:,0]>0)&(tab[:,3]>0)) else np.nan; int_ratio=max(intercept,0.0)/max(medvar,np.finfo(float).tiny) if np.isfinite(medvar) else np.inf
    common=integer>.99 and dep_ok and np.isfinite(F) and F>=max(0.0,1-poisson_tol) and fspread<=order_fano_spread_limit
    base=dict(theta_corr=tc,spatial_corr=sc,integer_like=integer,zero_fraction=zero,slope=slope,intercept=intercept,fano=F,fano_cv=fcv,n_bins=n,adjacent_spatial_corr=adj_sc,expected_overlap_corr=expected_overlap,nonoverlap_lag=nonoverlap_lag,curve_order_cv=curve_cv)
    if common and abs(F-1)<=poisson_tol and fcv<=fano_cv_limit:
        audit=NoiseAudit(True,"Целочисленный маргинальный шум; Fano≈1",**base); model=Poisson(audit)
    elif common and F>1+poisson_tol and fcv<=fano_cv_limit and int_ratio<=intercept_ratio_limit:
        audit=NoiseAudit(True,"NB1: Var≈F·mu; spatial overlap учтён отдельно",**base); model=NegativeBinomialFano(F,audit)
    elif common and F>1+poisson_tol and allow_fano_curve and curve_cv<=curve_order_cv_limit:
        x,f=_fano_curve(tab)
        audit=NoiseAudit(True,"Маргинальная NB с F(mu); constant-F недостаточен, spatial overlap учтён",**base); model=NegativeBinomialFanoCurve(x,f,audit)
    elif dep_ok and slope>0 and max(intercept,0)>0 and integer<.99:
        audit=NoiseAudit(False,"Кандидат Poisson-Gaussian: нужна отдельная dark/replicate validation",**base); model=PoissonGaussian(slope,np.sqrt(max(intercept,0.0)),audit)
    else:
        dep=max(abs(tc) if np.isfinite(tc) else 0,abs(sc) if np.isfinite(sc) else 0)
        why=f"auto-audit FAIL: Fano-CV={fcv:.3f}, curve-order-CV={curve_cv:.3f}, order-spread={fspread:.3f}, dep(nonoverlap)={dep:.3f}, integer={integer:.3f}"
        audit=NoiseAudit(False,why,**base); model=None
    return {"model":model,"audit":audit,"tables":tables,"fano_by_order":dict(zip(orders,fs)),"fano_order_spread":fspread,"rho_intensity_abscontrast":float(spearmanr(l,np.abs(c-np.median(c))).statistic),"intercept_ratio":int_ratio}

def print_count_noise_fit(r):
    a=r["audit"]; print("=== COUNT NOISE MODEL AUDIT ==="); print(f"PASS={a.assumption_pass}; {a.reason}")
    print(f"integer-like={a.integer_like:.3%}; zeros={a.zero_fraction:.3%}; theta_corr={a.theta_corr:+.4f}; spatial_corr(nonoverlap,lag={a.nonoverlap_lag})={a.spatial_corr:+.4f}")
    print(f"adjacent spatial corr={a.adjacent_spatial_corr:+.4f}; overlap reference≈{a.expected_overlap_corr:+.4f}")
    print("Fano by order =",", ".join(f"{k}:{v:.4g}" for k,v in r["fano_by_order"].items()))
    print(f"primary Fano={a.fano:.4g}; constant-F CV={a.fano_cv:.3f}; curve-order-CV={a.curve_order_cv:.3f}; order spread={r['fano_order_spread']:.3f}; Var~a*mu+b: a={a.slope:.4g}, b={a.intercept:.4g}; intercept ratio={r['intercept_ratio']:.3f}")
    print("model =",None if r["model"] is None else r["model"].kind)


def plot_count_noise_fit(r,*,figsize=(9.6,4.0)):
    """Mean–variance и локальный Fano для count-noise audit."""
    import matplotlib.pyplot as plt
    tables=r["tables"]; a=r["audit"]; F=float(a.fano); orders=sorted(tables)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=figsize,constrained_layout=True)
    markers=("o","s","^","D","v")

    xmax=0.0
    for j,o in enumerate(orders):
        t=np.asarray(tables[o],float); ok=np.isfinite(t[:,0])&np.isfinite(t[:,3])&(t[:,0]>0)&(t[:,3]>0)
        x=t[ok,0]; var=t[ok,3]**2
        if not len(x): continue
        xmax=max(xmax,float(np.max(x))); ax1.plot(x,var,marker=markers[j%len(markers)],lw=1,label=f"order {o}"); ax2.plot(x,var/x,marker=markers[j%len(markers)],lw=1)

    if xmax>0:
        x=np.linspace(0,xmax*1.05,300); ax1.plot(x,x,ls=":",lw=1,label="Poisson: Var=μ"); ax1.plot(x,F*x,ls="--",lw=1.2,label=f"model: Fμ, F={F:.3g}")
        if np.isfinite(a.slope) and np.isfinite(a.intercept): ax1.plot(x,a.slope*x+a.intercept,ls="-.",lw=1,label=f"diag: {a.slope:.3g}μ+{a.intercept:.3g}")
        ax2.axhline(1,ls=":",lw=1,label="Poisson F=1"); ax2.axhline(F,ls="--",lw=1.2,label=f"global F={F:.3g}")

    ax1.set(xlabel="Local intensity μ",ylabel="Contrast variance",title="Mean–variance law"); ax1.legend(frameon=False,fontsize=8)
    ax2.set(xlabel="Local intensity μ",ylabel="Variance / μ",title="Local Fano factor"); ax2.legend(frameon=False,fontsize=8)
    ax2.text(.98,.98,f"zeros={a.zero_fraction:.1%}\ninteger-like={a.integer_like:.1%}\nρᵩ={a.theta_corr:+.3f}; ρspace={a.spatial_corr:+.3f}",transform=ax2.transAxes,ha="right",va="top",fontsize=8)
    return fig,(ax1,ax2)
