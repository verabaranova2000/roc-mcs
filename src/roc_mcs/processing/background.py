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
