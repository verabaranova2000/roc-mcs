from dataclasses import dataclass
import numpy as np
from typing import Any


from roc_mcs.fitting.dynamic_roc_decomposition import detect_isolated_spikes, _basis


@dataclass(frozen=True, slots=True)
class BackgroundStats:
    median: float
    sigma: float
    threshold: float


def estimate_background_stats(data_3d: np.ndarray) -> BackgroundStats:
    """Оценивает фон, шум и порог отделения сигнала от фона."""
    data_3d = np.asarray(data_3d, dtype=float)
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d должен быть 3D, получено ndim={data_3d.ndim}")

    max_map = np.max(data_3d, axis=0)
    finite_max = max_map[np.isfinite(max_map)]
    if finite_max.size == 0:
        raise ValueError("Не найдено ни одного конечного значения для оценки фона.")

    q30 = np.percentile(finite_max, 30)
    bg_pixels = finite_max[finite_max <= q30]
    bg_median = float(np.median(bg_pixels))
    bg_mad = float(np.median(np.abs(bg_pixels - bg_median)))
    bg_sigma = 1.4826 * bg_mad if bg_mad > 0 else 1.0
    bg_threshold = bg_median + 5.0 * bg_sigma

    return BackgroundStats(
        median=bg_median,
        sigma=bg_sigma,
        threshold=float(bg_threshold),
    )






@dataclass(frozen=True,slots=True)
class BackgroundOnlyResult:
    theta: np.ndarray; time_s: np.ndarray; data: np.ndarray; valid: np.ndarray; spike_mask: np.ndarray
    background_map: np.ndarray; corrected_map: np.ndarray; corrected_nonnegative: np.ndarray; weight_map: np.ndarray
    standardized_residual_map: np.ndarray; noise: str; nb_alpha: float|None; converged: bool; n_iter: int; relative_change: float; metadata: dict[str,Any]

def estimate_background_only(roc_map,*,source='intensity',theta_window=None,valid=None,noise='nb',nb_alpha=None,n_basis_bg=4,spline_degree=3,background_mask=None,background_ranges=None,signal_window=None,spike_filter=True,spike_window=5,spike_z=10.,spike_isolation_ratio=.35,max_iter=1000):
    """Фон без модели пиков: exact Poisson/NB2 fit только по заданной background-support."""
    from scipy.optimize import minimize
    from scipy.special import gammaln
    th0=np.asarray(roc_map.get('theta_axis',roc_map.get('theta')),float);t=np.asarray(roc_map['time_s'],float);z0=np.asarray(roc_map[source],float)
    if z0.shape!=(len(t),len(th0)) or not np.all(np.diff(th0)>0) or not np.all(np.diff(t)>=0):raise ValueError('Некорректные theta/time/data')
    keep=np.ones(len(th0),bool) if theta_window is None else (th0>=min(theta_window))&(th0<=max(theta_window));th=th0[keep];z=z0[:,keep];vv=np.isfinite(z) if valid is None else np.asarray(valid,bool)[:,keep]&np.isfinite(z)
    if keep.sum()<12 or np.any(z[vv]<0):raise ValueError('Нужны raw nonnegative counts и >=12 угловых точек')
    if noise not in {'poisson','nb'}:raise ValueError("noise должен быть 'poisson' или 'nb'")
    if noise=='nb' and (nb_alpha is None or not np.isfinite(nb_alpha) or nb_alpha<=0):raise ValueError('Для NB задайте фиксированный nb_alpha > 0')
    modes=sum(x is not None for x in (background_mask,background_ranges,signal_window))
    if modes!=1:raise ValueError('Задайте ровно одно: background_mask, background_ranges или signal_window')
    if background_mask is not None:
        m=np.asarray(background_mask,bool);m=m[keep] if m.shape==(len(th0),) else m[:,keep] if m.shape==z0.shape else m
        support=np.broadcast_to(m,z.shape).copy() if m.ndim==1 else m.copy()
    elif background_ranges is not None:
        m=np.zeros(len(th),bool)
        for a,b in background_ranges:m|=(th>=min(a,b))&(th<=max(a,b))
        support=np.broadcast_to(m,z.shape).copy()
    else:
        a,b=map(float,signal_window);support=np.broadcast_to(~((th>=min(a,b))&(th<=max(a,b))),z.shape).copy()
    sm=detect_isolated_spikes(z,vv,window=spike_window,z=spike_z,isolation_ratio=spike_isolation_ratio) if spike_filter else np.zeros_like(vv);vv&=~sm;support&=vv
    if np.any(support.sum(1)<max(6,3*int(n_basis_bg))):raise ValueError('Слишком мало background-support точек хотя бы в одном scan')
    left=support[:,th<=np.median(th)].sum(1);right=support[:,th>=np.median(th)].sum(1)
    if np.any(left<2)|np.any(right<2):raise ValueError('Для интерполяции фона через diffraction region нужны background-support точки с обеих сторон')
    Bt=_basis(t,int(n_basis_bg),int(spline_degree));u=(th-th[0])/(th[-1]-th[0]);Q=np.column_stack(((1-u)**2,2*u*(1-u),u*u));X=np.einsum('im,kq->ikqm',Bt,Q).reshape(len(t)*len(th),3*Bt.shape[1]);y=z.ravel();mask=support.ravel();Xm=X[mask];ym=y[mask];eps=1e-9
    c0=np.full(X.shape[1],max(float(np.median(ym)),1e-3))
    def fg(c):
        mu=np.maximum(Xm@c,eps)
        if noise=='poisson':f=float(np.sum(mu-ym*np.log(mu)+gammaln(ym+1)));d=1-ym/mu
        else:
            a=float(nb_alpha);r=1/a;f=float(-np.sum(gammaln(ym+r)-gammaln(r)-gammaln(ym+1)+r*(np.log(r)-np.log(r+mu))+ym*(np.log(mu)-np.log(r+mu))));d=(mu-ym)/(mu*(1+a*mu))
        return f,Xm.T@d
    o=minimize(lambda c:fg(c),c0,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(c0),options={'maxiter':int(max_iter),'ftol':1e-12,'gtol':1e-7,'maxls':50})
    coef=np.maximum(o.x,0);bg=np.maximum((X@coef).reshape(z.shape),0);var=bg if noise=='poisson' else bg+float(nb_alpha)*bg*bg;sr=(z-bg)/np.sqrt(np.maximum(var,1e-12));sr[~vv]=np.nan;corr=z-bg;w=np.where(support,1.,0.);w[~vv]=np.nan;rs=sr[support]
    meta={'method':'background-support exact count-likelihood Bernstein×B-spline','theta_mask':keep,'n_basis_bg':int(n_basis_bg),'spline_degree':int(spline_degree),'support_fraction':float(support.sum()/vv.sum()),'support_points':int(support.sum()),'support_residual_mean':float(np.mean(rs)),'support_residual_sd':float(np.std(rs,ddof=1)),'objective':float(o.fun),'message':str(o.message),'note':'Peak profiles are not fitted; only declared signal-free support enters the likelihood.'}
    return BackgroundOnlyResult(th,t,z,vv,sm,bg,corr,np.clip(corr,0,None),w,sr,noise,None if nb_alpha is None else float(nb_alpha),bool(o.success),int(o.nit),float(np.linalg.norm(o.jac,np.inf)),meta)





# ==================== НОВОЕ =============

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom,norm,poisson

@dataclass(frozen=True,slots=True)
class AutoBackgroundSupport:
    support_mask: np.ndarray; signal_mask: np.ndarray; ambiguous_mask: np.ndarray; pilot_map: np.ndarray; pvalue_map: np.ndarray; scales: tuple[int,...]; metadata: dict



def _background_design(theta,time_s,n_basis_bg,spline_degree):
    Bt=_basis(time_s,int(n_basis_bg),int(spline_degree));u=(theta-theta[0])/(theta[-1]-theta[0]);Q=np.column_stack(((1-u)**2,2*u*(1-u),u*u));return np.einsum('im,kq->ikqm',Bt,Q).reshape(len(time_s)*len(theta),3*Bt.shape[1])


# Замена auto-support блока в roc_mcs.processing.background
# Требует уже существующие: AutoBackgroundSupport, estimate_background_only

def _by_reject(p,q):
    """BY-FDR для зависимых p-values; NaN не участвуют."""
    import numpy as np
    a=np.asarray(p,float);f=a.ravel();idx=np.flatnonzero(np.isfinite(f));out=np.zeros(f.shape,bool)
    if not len(idx):return out.reshape(a.shape)
    v=f[idx];o=np.argsort(v);s=v[o];m=len(s);c=np.sum(1./np.arange(1,m+1));h=np.flatnonzero(s<=float(q)*np.arange(1,m+1)/(m*c))
    if len(h):out[idx]=v<=s[h[-1]]
    return out.reshape(a.shape)

def _finite_min_stack(P):
    """Минимум по первому измерению без All-NaN warnings."""
    import numpy as np
    P=np.asarray(P,float);out=np.full(P.shape[1:],np.nan);ok=np.any(np.isfinite(P),axis=0)
    if np.any(ok):out[ok]=np.min(np.where(np.isfinite(P[:,ok]),P[:,ok],np.inf),axis=0)
    return out

def _multiscale_positive_test(data,mu,valid,noise,nb_alpha,fdr,max_scale_fraction):
    """Односторонний multiscale screening положительного excess относительно fitted background."""
    import numpy as np
    from scipy.stats import nbinom,norm,poisson
    y=np.asarray(data,float);m=np.maximum(np.asarray(mu,float),1e-12);v=np.asarray(valid,bool)&np.isfinite(y)&np.isfinite(m);T,N=y.shape
    var=m if noise=='poisson' else m+float(nb_alpha)*m*m;lim=max(3,int(np.floor(float(max_scale_fraction)*N)));scales=[1];w=3
    while w<=lim:scales.append(w);w=2*w+1
    ps=[]
    for w in scales:
        if w==1:
            if noise=='poisson':p=poisson.sf(np.floor(y)-1,m)
            else:r=1/float(nb_alpha);pp=r/(r+m);p=nbinom.sf(np.floor(y)-1,r,pp)
            ps.append(np.where(v,p,np.nan));continue
        k=np.ones(w,float);rr=np.stack([np.convolve(np.where(v[i],y[i]-m[i],0.),k,'same') for i in range(T)]);vv=np.stack([np.convolve(np.where(v[i],var[i],0.),k,'same') for i in range(T)]);nn=np.stack([np.convolve(v[i].astype(float),k,'same') for i in range(T)])
        p=norm.sf(rr/np.sqrt(np.maximum(vv,1e-12)));ps.append(np.where((nn>=max(3,.8*w))&v,p,np.nan))
    P=np.stack(ps);rej=_by_reject(P,float(fdr));hits=rej.sum(0);return hits>0,_finite_min_stack(P),tuple(scales),hits

def detect_background_support_auto(roc_map,*,source='intensity',theta_window=None,valid=None,noise='nb',nb_alpha=None,n_basis_bg=4,spline_degree=3,fdr=.01,max_scale_fraction=.25,refine_iter=4,spike_filter=True,spike_window=5,spike_z=10.,spike_isolation_ratio=.35,max_iter=1000):
    """Автоматический support: exact count background -> conservative multiscale peeling."""
    import numpy as np
    z0=np.asarray(roc_map[source],float);base_mask=np.ones_like(z0,bool) if valid is None else np.asarray(valid,bool).copy()
    b=estimate_background_only(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,background_mask=base_mask,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter)
    support=b.valid.copy();all_signal=np.zeros_like(support,bool);pmin=np.full(support.shape,np.nan);hits=np.zeros(support.shape,int);scales=()
    for it in range(max(1,int(refine_iter))):
        signal,pm,scales,h=_multiscale_positive_test(b.data,b.background_map,b.valid,noise,nb_alpha,fdr,max_scale_fraction);new=support&~signal
        pmin=np.fmin(pmin,pm) if np.any(np.isfinite(pmin)) else pm.copy();hits=np.maximum(hits,h);all_signal|=signal
        need=max(6,3*int(n_basis_bg))
        if np.any(new.sum(1)<need):raise RuntimeError(f'INCONCLUSIVE automatic background support: screening оставил <{need} background-точек хотя бы в одном scan')
        left=new[:,b.theta<=np.median(b.theta)].sum(1);right=new[:,b.theta>=np.median(b.theta)].sum(1)
        if np.any(left<2)|np.any(right<2):raise RuntimeError('INCONCLUSIVE automatic background support: после screening нет background-support с обеих сторон хотя бы в одном scan')
        if np.array_equal(new,support):support=new;break
        support=new;b=estimate_background_only(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,background_mask=support,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter)
    ambiguous=(hits>0)&(hits<len(scales))&b.valid
    meta={'method':'iterative exact-count background + multiscale BY screening','fdr':float(fdr),'scales':scales,'support_fraction':float(support.sum()/max(b.valid.sum(),1)),'signal_fraction':float(all_signal.sum()/max(b.valid.sum(),1)),'ambiguous_fraction':float(ambiguous.sum()/max(b.valid.sum(),1)),'refine_iter':int(it+1),'note':'BY is conservative for dependent tests; support is an automatic screening result, not proof of zero physical signal.'}
    return AutoBackgroundSupport(support,all_signal,ambiguous,b.background_map.copy(),pmin,scales,meta)

def estimate_background_auto(roc_map,*,source='intensity',theta_window=None,valid=None,noise='nb',nb_alpha=None,n_basis_bg=4,spline_degree=3,fdr=.01,max_scale_fraction=.25,refine_iter=4,spike_filter=True,spike_window=5,spike_z=10.,spike_isolation_ratio=.35,max_iter=1000,return_support=False):
    """Auto-support + существующий exact Poisson/NB2 background fit."""
    a=detect_background_support_auto(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,fdr=fdr,max_scale_fraction=max_scale_fraction,refine_iter=refine_iter,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter)
    b=estimate_background_only(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,background_mask=a.support_mask,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter)
    b.metadata.update({'support_mode':'auto','auto_support_method':a.metadata['method'],'auto_fdr':float(fdr),'auto_scales':a.scales,'auto_support_fraction':a.metadata['support_fraction'],'auto_signal_fraction':a.metadata['signal_fraction'],'auto_ambiguous_fraction':a.metadata['ambiguous_fraction']})
    return (b,a) if return_support else b


def background_time_variation_audit(b,*,n_folds=4):
    """Сравнивает constant и smooth B(t,theta) по scan-fold predictive count NLL."""
    from scipy.special import gammaln
    z=np.asarray(b.data,float);v=np.asarray(b.valid,bool)&(np.asarray(b.weight_map,float)>.5);th=np.asarray(b.theta,float);t=np.asarray(b.time_s,float);K=max(2,min(int(n_folds),len(t)));fold=np.arange(len(t))%K
    def Xof(mode):
        Bt=np.ones((len(t),1)) if mode=='constant' else _basis(t,int(b.metadata.get('n_basis_bg',4)),int(b.metadata.get('spline_degree',3)));u=(th-th[0])/(th[-1]-th[0]);Q=np.column_stack(((1-u)**2,2*u*(1-u),u*u));return np.einsum('im,kq->ikqm',Bt,Q).reshape(len(t),len(th),-1)
    def nll(y,mu):
        mu=np.maximum(mu,1e-12)
        if b.noise=='poisson':return np.sum(mu-y*np.log(mu)+gammaln(y+1))
        a=float(b.nb_alpha);r=1/a;return -np.sum(gammaln(y+r)-gammaln(r)-gammaln(y+1)+r*(np.log(r)-np.log(r+mu))+y*(np.log(mu)-np.log(r+mu)))
    scores={}
    for mode in ('constant','smooth'):
        X=Xof(mode);d=[]
        for k in range(K):
            tr=(fold!=k)[:,None]&v;te=(fold==k)[:,None]&v;A=X[tr];y=z[tr];c0=np.full(X.shape[-1],max(float(np.median(y)),1e-3))
            def fg(c):
                mu=np.maximum(A@c,1e-12)
                if b.noise=='poisson':g=1-y/mu
                else:a=float(b.nb_alpha);g=(mu-y)/(mu*(1+a*mu))
                return float(nll(y,mu)),A.T@g
            o=minimize(lambda c:fg(c),c0,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(c0),options={'maxiter':1000});d.append(float(nll(z[te],np.maximum(X[te]@np.maximum(o.x,0),1e-12))))
        scores[mode]=np.asarray(d,float)
    delta=scores['constant']-scores['smooth'];mean=float(np.mean(delta));se=float(np.std(delta,ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan;decision='smooth' if np.isfinite(se) and mean>2*se else 'constant' if np.isfinite(se) and mean< -2*se else 'inconclusive'
    return {'constant_nll':scores['constant'],'smooth_nll':scores['smooth'],'delta_nll_constant_minus_smooth':delta,'mean_delta_nll':mean,'se_delta_nll':se,'decision':decision,'note':'Positive delta favors smooth time-varying background; 2-SE rule is a predictive diagnostic, not a physical law.'}
