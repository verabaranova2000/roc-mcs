import numpy as np
from roc_mcs.fitting import dynamic_roc_decomposition as drx
from roc_mcs.processing.moments import compute_cii

def test_background_support_nb_recovers_scale():
    T,N=24,181;t=np.linspace(0,300,T);th=np.linspace(-250,250,N);u=(th-th[0])/(th[-1]-th[0]);tt=t/t[-1];bg=12+4*tt[:,None]+3*u[None,:]+2*u[None,:]**2;c=-100+180*tt;peak=180*np.exp(-.5*((th[None,:]-c[:,None])/(45/2.355))**2);mu=bg+peak;a=.08;r=1/a;rng=np.random.default_rng(3);y=rng.negative_binomial(r,r/(r+mu));roc={'time_s':t,'theta_axis':th,'intensity':y}
    q=drx.estimate_background_only(roc,noise='nb',nb_alpha=a,signal_window=(-180,160),n_basis_bg=5,spline_degree=3,spike_filter=False)
    assert q.converged;assert np.median(np.abs(q.background_map-bg)/bg)<.08;assert abs(q.metadata['support_residual_mean'])<.15;assert .7<q.metadata['support_residual_sd']<1.3;assert np.median(q.background_map)>5

def test_cii_diagnostic_keeps_background_error_sign():
    from types import SimpleNamespace
    th=np.linspace(-5,5,101);signal=np.exp(-.5*(th/.7)**2)*100;true_bg=np.full_like(th,10.);data=(true_bg+signal)[None,:];base=dict(theta=th,time_s=np.array([0.]),data=data)
    lo=SimpleNamespace(**base,background_map=np.full_like(data,8.),corrected_map=data-8.);hi=SimpleNamespace(**base,background_map=np.full_like(data,12.),corrected_map=data-12.)
    dlo=compute_cii(lo)['cii_diagnostic'][0];dhi=compute_cii(hi)['cii_diagnostic'][0]
    assert dlo[-1]>1;assert dhi[-1]<1
