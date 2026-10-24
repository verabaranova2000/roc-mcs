from __future__ import annotations
from abc import ABC,abstractmethod
from dataclasses import dataclass
import hashlib,json
import numpy as np

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
