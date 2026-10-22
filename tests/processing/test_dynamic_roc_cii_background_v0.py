import numpy as np
import roc_mcs.fitting.dynamic_roc_decomposition as drx
from roc_mcs.processing.moments import compute_cii

def _synthetic(seed=7):
    rng=np.random.default_rng(seed);T,N=24,181;t=np.linspace(0,360,T);th=np.linspace(-280,220,N);u=(th-th[0])/(th[-1]-th[0]);tt=np.linspace(0,1,T)
    c=np.column_stack([7+1.5*tt,9+1.2*np.sin(np.pi*tt),11-1.2*tt]);bg=c[:,0,None]*(1-u)**2+2*c[:,1,None]*u*(1-u)+c[:,2,None]*u*u;peak=np.zeros_like(bg)
    for i,x in enumerate(tt):
        ctr=-100+150*x;peak[i]+=150*np.exp(-4*np.log(2)*((th-ctr)/(38+24*x))**2);peak[i]+=55*x*np.exp(-4*np.log(2)*((th-(ctr-55))/80)**2)
    alpha=.16;mu=bg+peak;r=1/alpha;p=r/(r+mu);y=rng.negative_binomial(r,p)
    return {'time_s':t,'theta_axis':th,'intensity':y},bg,peak,alpha

def test_background_only_invariants():
    roc,bg,peak,alpha=_synthetic();r=drx.estimate_background_only(roc,noise='nb',nb_alpha=alpha,n_basis_bg=5,max_iter=80,tol=1e-3)
    assert r.converged and r.background_map.shape==roc['intensity'].shape
    assert np.all(np.isfinite(r.background_map)) and np.all(r.background_map>=0)
    assert np.allclose(r.corrected_map,r.data-r.background_map)
    assert np.all(r.corrected_nonnegative>=0)
    assert np.nanmedian(r.weight_map[peak>25])<np.nanmedian(r.weight_map[peak<1])
    assert np.sqrt(np.mean((r.background_map-bg)**2))/np.mean(bg)<.25

def test_cii_is_monotone_after_clipping():
    roc,_,_,alpha = _synthetic()
    b = drx.estimate_background_only(roc,noise='nb',nb_alpha=alpha,n_basis_bg=5,max_iter=80,tol=1e-3)
    c = compute_cii(b)
    assert np.all(np.diff(c['cii'],axis=1)>=-1e-12)
    assert np.allclose(c['cii'][:,-1],1,atol=1e-12,equal_nan=False)
    assert set(c['quantiles'])=={.05,.10,.25,.50,.75,.90,.95}
    for q in c['quantiles'].values():assert np.all(np.isfinite(q))
