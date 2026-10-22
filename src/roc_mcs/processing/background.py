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

def estimate_background_only(roc_map,*,source='intensity',theta_window=None,valid=None,noise='nb',nb_alpha=None,n_basis_bg=4,spline_degree=3,background_mask=None,background_ranges=None,signal_window=None,support_policy='strict',spike_filter=True,spike_window=5,spike_z=10.,spike_isolation_ratio=.35,max_iter=1000):
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
    if support_policy not in {'strict','temporal'}:raise ValueError("support_policy должен быть 'strict' или 'temporal'")
    left=support[:,th<=np.median(th)].sum(1);right=support[:,th>=np.median(th)].sum(1);nscan=support.sum(1)
    if support_policy=='strict':
        if np.any(nscan<max(6,3*int(n_basis_bg))):raise ValueError('Слишком мало background-support точек хотя бы в одном scan')
        if np.any(left<2)|np.any(right<2):raise ValueError('Для strict-режима нужны background-support точки с обеих сторон каждого scan')
    Bt=_basis(t,int(n_basis_bg),int(spline_degree));u=(th-th[0])/(th[-1]-th[0]);Q=np.column_stack(((1-u)**2,2*u*(1-u),u*u));X=np.einsum('im,kq->ikqm',Bt,Q).reshape(len(t)*len(th),3*Bt.shape[1]);y=z.ravel();mask=support.ravel();Xm=X[mask];ym=y[mask];eps=1e-9
    if len(ym)<max(12,2*X.shape[1]) or np.linalg.matrix_rank(Xm)<X.shape[1]:raise ValueError('Background globally неидентифицируем: недостаточно независимых support-наблюдений')
    c0=np.full(X.shape[1],max(float(np.median(ym)),1e-3))
    def fg(c):
        mu=np.maximum(Xm@c,eps)
        if noise=='poisson':f=float(np.sum(mu-ym*np.log(mu)+gammaln(ym+1)));d=1-ym/mu
        else:
            a=float(nb_alpha);r=1/a;f=float(-np.sum(gammaln(ym+r)-gammaln(r)-gammaln(ym+1)+r*(np.log(r)-np.log(r+mu))+ym*(np.log(mu)-np.log(r+mu))));d=(mu-ym)/(mu*(1+a*mu))
        return f,Xm.T@d
    o=minimize(lambda c:fg(c),c0,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(c0),options={'maxiter':int(max_iter),'ftol':1e-12,'gtol':1e-7,'maxls':50})
    coef=np.maximum(o.x,0);bg=np.maximum((X@coef).reshape(z.shape),0);var=bg if noise=='poisson' else bg+float(nb_alpha)*bg*bg;sr=(z-bg)/np.sqrt(np.maximum(var,1e-12));sr[~vv]=np.nan;corr=z-bg;w=np.where(support,1.,0.);w[~vv]=np.nan;rs=sr[support]
    mu_s=np.maximum(Xm@coef,eps);iw=1/np.maximum(mu_s,eps) if noise=='poisson' else 1/np.maximum(mu_s*(1+float(nb_alpha)*mu_s),eps);info=Xm.T@(iw[:,None]*Xm);cov=np.linalg.pinv(info,rcond=1e-10);pred_se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0)).reshape(z.shape)
    observed=nscan>0;status=np.full(len(t),'bilateral_observed',dtype=object);status[(left<2)^(right<2)]='unilateral_temporal';status[(nscan>0)&(left<2)&(right<2)]='sparse_temporal';status[nscan==0]='temporal_only'
    idx=np.flatnonzero(observed)
    for i in np.flatnonzero(nscan==0):
        status[i]='temporal_interpolation' if np.any(idx<i) and np.any(idx>i) else 'temporal_extrapolation'
    provisional=status!='bilateral_observed';rel_se=pred_se/np.maximum(bg,1.)
    meta={'method':'background-support exact count-likelihood Bernstein×B-spline','theta_mask':keep,'n_basis_bg':int(n_basis_bg),'spline_degree':int(spline_degree),'support_policy':support_policy,'support_fraction':float(support.sum()/vv.sum()),'support_points':int(support.sum()),'support_residual_mean':float(np.mean(rs)),'support_residual_sd':float(np.std(rs,ddof=1)),'objective':float(o.fun),'message':str(o.message),'support_left_points':left,'support_right_points':right,'support_points_per_scan':nscan,'support_status':status,'provisional_scans':np.flatnonzero(provisional),'prediction_se_map':pred_se,'prediction_relative_se_map':rel_se,'information_rank':int(np.linalg.matrix_rank(info)),'information_condition':float(np.linalg.cond(info)),'note':'Temporal mode treats absent support as missing observations inside the joint likelihood; interpolation/extrapolation is model-based and explicitly flagged.'}
    return BackgroundOnlyResult(th,t,z,vv,sm,bg,corr,np.clip(corr,0,None),w,sr,noise,None if nb_alpha is None else float(nb_alpha),bool(o.success),int(o.nit),float(np.linalg.norm(o.jac,np.inf)),meta)

# ==================== AUTO BACKGROUND SUPPORT ====================

from scipy.optimize import minimize
from scipy.stats import nbinom,norm,poisson

@dataclass(frozen=True,slots=True)
class AutoBackgroundSupport:
    support_mask: np.ndarray; signal_mask: np.ndarray; ambiguous_mask: np.ndarray; pilot_map: np.ndarray; pvalue_map: np.ndarray
    scales: tuple[int,...]; history: tuple[dict[str,Any],...]; metadata: dict[str,Any]

def _by_reject(p,q):
    """BY-FDR для зависимых p-values; NaN исключаются."""
    a=np.asarray(p,float);f=a.ravel();idx=np.flatnonzero(np.isfinite(f));out=np.zeros(f.shape,bool)
    if not len(idx):return out.reshape(a.shape)
    v=f[idx];s=np.sort(v);m=len(s);c=np.sum(1/np.arange(1,m+1,dtype=float));h=np.flatnonzero(s<=float(q)*np.arange(1,m+1)/(m*c))
    if len(h):out[idx]=v<=s[h[-1]]
    return out.reshape(a.shape)

def _finite_min_stack(a):
    """Минимум по стеку без All-NaN warnings."""
    a=np.asarray(a,float);out=np.full(a.shape[1:],np.nan);ok=np.any(np.isfinite(a),axis=0)
    if np.any(ok):out[ok]=np.min(np.where(np.isfinite(a[:,ok]),a[:,ok],np.inf),axis=0)
    return out

def _multiscale_positive_test(data,mu,valid,noise,nb_alpha,fdr,max_scale_fraction):
    """Односторонний multiscale screening положительного excess."""
    y=np.asarray(data,float);m=np.maximum(np.asarray(mu,float),1e-12);v=np.asarray(valid,bool)&np.isfinite(y)&np.isfinite(m);T,N=y.shape
    if not 0<float(fdr)<1:raise ValueError('fdr должен лежать в (0,1)')
    if not 0<float(max_scale_fraction)<=.5:raise ValueError('max_scale_fraction должен лежать в (0,0.5]')
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

def _support_guard(mask,theta,n_basis_bg,*,policy='temporal'):
    if policy not in {'strict','temporal'}:raise ValueError("policy должен быть 'strict' или 'temporal'")
    need=max(6,3*int(n_basis_bg));n=mask.sum(1);med=np.median(theta);left=mask[:,theta<=med].sum(1);right=mask[:,theta>=med].sum(1)
    if policy=='strict':
        if np.any(n<need):raise RuntimeError(f'INCONCLUSIVE automatic background support: осталось <{need} background-точек хотя бы в одном scan')
        if np.any(left<2)|np.any(right<2):raise RuntimeError('INCONCLUSIVE automatic background support: нет background-support с обеих сторон хотя бы в одном scan')
    elif mask.sum()<max(12,2*3*int(n_basis_bg)):raise RuntimeError('INCONCLUSIVE automatic background support: глобально слишком мало support-точек')


def detect_background_support_auto(roc_map,*,source='intensity',theta_window=None,valid=None,noise='nb',nb_alpha=None,n_basis_bg=4,spline_degree=3,fdr=.01,max_scale_fraction=.25,max_refine_iter=30,mask_change_tol=0.,support_policy='temporal',spike_filter=True,spike_window=5,spike_z=10.,spike_isolation_ratio=.35,max_iter=1000):
    """Автоматический support: count-background + монотонный multiscale screening до сходимости."""
    if int(max_refine_iter)<1:raise ValueError('max_refine_iter должен быть >=1')
    if not 0<=float(mask_change_tol)<1:raise ValueError('mask_change_tol должен лежать в [0,1)')
    z0=np.asarray(roc_map[source],float);base=np.ones_like(z0,bool) if valid is None else np.asarray(valid,bool).copy()
    b=estimate_background_only(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,background_mask=base,support_policy=support_policy,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter)
    support=b.valid.copy();history=[];scales=();pmin=np.full(support.shape,np.nan);hits=np.zeros(support.shape,int);converged=False;valid_n=max(int(b.valid.sum()),1)
    history.append({'iteration':0,'support_fraction':float(support.sum()/valid_n),'newly_rejected_fraction':0.,'mask_change_fraction':np.nan,'support_residual_mean':float(b.metadata['support_residual_mean']),'support_residual_sd':float(b.metadata['support_residual_sd']),'objective':float(b.metadata['objective'])})
    for it in range(1,int(max_refine_iter)+1):
        signal,pm,scales,h=_multiscale_positive_test(b.data,b.background_map,b.valid,noise,nb_alpha,fdr,max_scale_fraction);new=support&~signal;_support_guard(new,b.theta,n_basis_bg,policy=support_policy)
        changed=int(np.count_nonzero(new!=support));frac=changed/valid_n;pmin=pm.copy();hits=h.copy()
        if changed:
            support=new;b=estimate_background_only(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,background_mask=support,support_policy=support_policy,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter)
        history.append({'iteration':it,'support_fraction':float(support.sum()/valid_n),'newly_rejected_fraction':float(frac),'mask_change_fraction':float(frac),'support_residual_mean':float(b.metadata['support_residual_mean']),'support_residual_sd':float(b.metadata['support_residual_sd']),'objective':float(b.metadata['objective'])})
        if changed==0 or (mask_change_tol>0 and frac<=float(mask_change_tol)):converged=True;break
    signal_mask=b.valid&~support;ambiguous=(hits>0)&(hits<len(scales))&signal_mask
    meta={'method':'iterative exact-count background + multiscale BY screening','fdr':float(fdr),'scales':scales,'support_fraction':float(support.sum()/valid_n),'signal_fraction':float(signal_mask.sum()/valid_n),'ambiguous_fraction':float(ambiguous.sum()/valid_n),'converged':bool(converged),'n_refine':int(history[-1]['iteration']),'last_mask_change_fraction':float(history[-1]['mask_change_fraction']),'max_refine_iter':int(max_refine_iter),'mask_change_tol':float(mask_change_tol),'support_policy':support_policy,'note':'BY применяется внутри screening-шага; iterative data reuse не означает глобальный FDR всей процедуры. Сходимость — numerical, robustness проверяется sensitivity-аудитом.'}
    return AutoBackgroundSupport(support,signal_mask,ambiguous,b.background_map.copy(),pmin,scales,tuple(history),meta)

def estimate_background_auto(roc_map,*,source='intensity',theta_window=None,valid=None,noise='nb',nb_alpha=None,n_basis_bg=4,spline_degree=3,fdr=.01,max_scale_fraction=.25,max_refine_iter=30,mask_change_tol=0.,support_policy='temporal',spike_filter=True,spike_window=5,spike_z=10.,spike_isolation_ratio=.35,max_iter=1000,require_convergence=True,return_support=False):
    """Автоматический support + финальный exact Poisson/NB2 background fit."""
    a=detect_background_support_auto(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,fdr=fdr,max_scale_fraction=max_scale_fraction,max_refine_iter=max_refine_iter,mask_change_tol=mask_change_tol,support_policy=support_policy,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter)
    if require_convergence and not a.metadata['converged']:raise RuntimeError(f"INCONCLUSIVE automatic background support: не достигнута сходимость за {max_refine_iter} итераций; last mask change={a.metadata['last_mask_change_fraction']:.3%}")
    b=estimate_background_only(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,background_mask=a.support_mask,support_policy=support_policy,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter)
    b.metadata.update({'support_mode':'auto','auto_support_method':a.metadata['method'],'auto_fdr':float(fdr),'auto_scales':a.scales,'auto_support_fraction':a.metadata['support_fraction'],'auto_signal_fraction':a.metadata['signal_fraction'],'auto_ambiguous_fraction':a.metadata['ambiguous_fraction'],'auto_converged':a.metadata['converged'],'auto_n_refine':a.metadata['n_refine'],'auto_last_mask_change_fraction':a.metadata['last_mask_change_fraction']})
    return (b,a) if return_support else b

def run_background_cii_auto(roc_map,*,levels=(.05,.10,.25,.50,.75,.90,.95),time_variation_audit=True,**kwargs):
    """Production-вызов: auto-support -> background -> CII -> optional time-variation audit."""
    from roc_mcs.processing.moments import compute_cii
    b,a=estimate_background_auto(roc_map,return_support=True,**kwargs);c=compute_cii(b,levels=levels);tv=background_time_variation_audit(b) if time_variation_audit else None;q=background_support_quality_table(b)
    return {'background':b,'support':a,'cii':c,'time_variation':tv,'quality':q,'latest_quality':q.iloc[-1].to_dict()}

def audit_background_cii_sensitivity(roc_map,*,fdr_values=(.005,.01,.02),levels=(.05,.10,.25,.50,.75,.90,.95),source='intensity',theta_window=None,valid=None,noise='nb',nb_alpha=None,n_basis_bg=4,spline_degree=3,max_scale_fraction=.25,max_refine_iter=30,mask_change_tol=0.,support_policy='temporal',spike_filter=True,spike_window=5,spike_z=10.,spike_isolation_ratio=.35,max_iter=1000,require_convergence=True):
    """Sensitivity CII и фоновых метрик к screening-level FDR."""
    import pandas as pd
    from roc_mcs.processing.moments import compute_cii,compute_roc_map_moments
    fs=tuple(float(x) for x in fdr_values)
    if len(fs)<2 or any(not 0<x<1 for x in fs):raise ValueError('Нужно >=2 корректных fdr_values')
    runs={};rows=[];run_rows=[]
    for fdr in fs:
        b,a=estimate_background_auto(roc_map,source=source,theta_window=theta_window,valid=valid,noise=noise,nb_alpha=nb_alpha,n_basis_bg=n_basis_bg,spline_degree=spline_degree,fdr=fdr,max_scale_fraction=max_scale_fraction,max_refine_iter=max_refine_iter,mask_change_tol=mask_change_tol,support_policy=support_policy,spike_filter=spike_filter,spike_window=spike_window,spike_z=spike_z,spike_isolation_ratio=spike_isolation_ratio,max_iter=max_iter,require_convergence=require_convergence,return_support=True);c=compute_cii(b,levels=levels);pm=compute_roc_map_moments(c['roc_map']);runs[fdr]=(b,a,c)
        run_rows.append({'fdr':fdr,'converged':bool(a.metadata['converged']),'n_refine':int(a.metadata['n_refine']),'last_mask_change_fraction':float(a.metadata['last_mask_change_fraction']),'support_fraction':float(a.metadata['support_fraction']),'ambiguous_fraction':float(a.metadata['ambiguous_fraction']),'residual_mean':float(b.metadata['support_residual_mean']),'residual_sd':float(b.metadata['support_residual_sd']),'background_min':float(np.nanmin(b.background_map)),'background_max':float(np.nanmax(b.background_map))})
        q=c['quantiles'];qkeys={f'q{int(round(100*p)):02d}':np.asarray(v,float) for p,v in q.items()};metrics={**qkeys,'w50_cii':q[.75]-q[.25] if .75 in q and .25 in q else np.full(len(b.time_s),np.nan),'w80_cii':q[.90]-q[.10] if .90 in q and .10 in q else np.full(len(b.time_s),np.nan),'w90_cii':q[.95]-q[.05] if .95 in q and .05 in q else np.full(len(b.time_s),np.nan),'centroid':pm['centroid'].to_numpy(float),'area':pm['area'].to_numpy(float),'signed_area_ratio':np.divide(c['signed_area'],c['positive_area'],out=np.full_like(c['signed_area'],np.nan),where=c['positive_area']>0)}
        for name,val in metrics.items():
            for i,x in enumerate(np.asarray(val,float)):rows.append({'fdr':fdr,'scan_id':i,'time_s':float(b.time_s[i]),'metric':name,'value':float(x)})
    values=pd.DataFrame(rows);grp=values.groupby(['scan_id','time_s','metric'],as_index=False)['value'];ranges=grp.agg(value_min='min',value_max='max',value_median='median');ranges['value_range']=ranges.value_max-ranges.value_min;dtheta=float(np.median(np.diff(next(iter(runs.values()))[0].theta)));angular=~ranges.metric.isin(['area','signed_area_ratio']);ranges['range_bins']=np.where(angular,ranges.value_range/max(abs(dtheta),np.finfo(float).eps),np.nan)
    summary=ranges.groupby('metric',as_index=False).agg(median_range=('value_range','median'),p95_range=('value_range',lambda x:float(np.nanpercentile(x,95))),max_range=('value_range','max'),median_range_bins=('range_bins','median'),max_range_bins=('range_bins','max'))
    return {'runs':runs,'values':values,'ranges':ranges,'summary':summary,'run_table':pd.DataFrame(run_rows),'fdr_values':fs,'levels':tuple(map(float,levels)),'dtheta':dtheta,'note':'Sensitivity ranges quantify dependence on the declared screening levels; they are robustness diagnostics, not confidence intervals.'}


def background_support_quality_table(b):
    """Таблица качества background-support по scan; пригодна для live-индикации."""
    import pandas as pd
    m=b.metadata;status=np.asarray(m.get('support_status',['unknown']*len(b.time_s)),object);left=np.asarray(m.get('support_left_points',np.zeros(len(b.time_s))),int);right=np.asarray(m.get('support_right_points',np.zeros(len(b.time_s))),int);n=np.asarray(m.get('support_points_per_scan',left+right),int);rse=np.asarray(m.get('prediction_relative_se_map',np.full_like(b.background_map,np.nan)),float)
    rows=[]
    for i in range(len(b.time_s)):
        rows.append({'scan_id':i,'time_s':float(b.time_s[i]),'status':str(status[i]),'provisional':bool(status[i]!='bilateral_observed'),'support_points':int(n[i]),'left_points':int(left[i]),'right_points':int(right[i]),'median_background_relative_se':float(np.nanmedian(rse[i]))})
    return pd.DataFrame(rows)

def background_time_variation_audit(b,*,n_folds=4):
    """Сравнивает constant и smooth B(t,theta) по scan-fold predictive count NLL."""
    from scipy.special import gammaln
    z=np.asarray(b.data,float);v=np.asarray(b.valid,bool)&(np.asarray(b.weight_map,float)>.5);th=np.asarray(b.theta,float);t=np.asarray(b.time_s,float);K=max(2,min(int(n_folds),len(t)));fold=np.arange(len(t))%K
    def Xof(mode):
        Bt=np.ones((len(t),1)) if mode=='constant' else _basis(t,int(b.metadata.get('n_basis_bg',4)),int(b.metadata.get('spline_degree',3)));u=(th-th[0])/(th[-1]-th[0]);Q=np.column_stack(((1-u)**2,2*u*(1-u),u*u));return np.einsum('im,kq->ikqm',Bt,Q).reshape(len(t),len(th),-1)
    def nll(y,mu):
        from scipy.special import gammaln
        mu=np.maximum(mu,1e-12)
        if b.noise=='poisson':return np.sum(mu-y*np.log(mu)+gammaln(y+1))
        a=float(b.nb_alpha);r=1/a;return -np.sum(gammaln(y+r)-gammaln(r)-gammaln(y+1)+r*(np.log(r)-np.log(r+mu))+y*(np.log(mu)-np.log(r+mu)))
    scores={}
    for mode in ('constant','smooth'):
        X=Xof(mode);d=[]
        for k in range(K):
            tr=(fold!=k)[:,None]&v;te=(fold==k)[:,None]&v;A=X[tr];y=z[tr];c0=np.full(X.shape[-1],max(float(np.median(y)),1e-3))
            def fg(c):
                mu=np.maximum(A@c,1e-12);g=1-y/mu if b.noise=='poisson' else (mu-y)/(mu*(1+float(b.nb_alpha)*mu));return float(nll(y,mu)),A.T@g
            o=minimize(lambda c:fg(c),c0,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(c0),options={'maxiter':1000});d.append(float(nll(z[te],np.maximum(X[te]@np.maximum(o.x,0),1e-12))))
        scores[mode]=np.asarray(d,float)
    delta=scores['constant']-scores['smooth'];mean=float(np.mean(delta));se=float(np.std(delta,ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan;decision='smooth' if np.isfinite(se) and mean>2*se else 'constant' if np.isfinite(se) and mean< -2*se else 'inconclusive'
    return {'constant_nll':scores['constant'],'smooth_nll':scores['smooth'],'delta_nll_constant_minus_smooth':delta,'mean_delta_nll':mean,'se_delta_nll':se,'decision':decision,'note':'Positive delta favors smooth time-varying background; 2-SE rule is a predictive diagnostic, not a physical law.'}
