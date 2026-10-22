import numpy as np

def _synthetic(seed=7):
    rng=np.random.default_rng(seed);T,N=20,101;t=np.linspace(0,360,T);th=np.linspace(-240,240,N);u=(th-th[0])/(th[-1]-th[0]);tu=(t-t[0])/(t[-1]-t[0]);bg=(1.7+.45*tu[:,None])*(.9+.22*u[None,:]+.08*u[None,:]**2);c=-115+190*tu;w=22+28*tu;sig=50*np.exp(-.5*((th[None,:]-c[:,None])/w[:,None])**2)+10*np.exp(-.5*((th[None,:]-(c[:,None]+70))/42)**2)*(tu[:,None]>.4);mu=bg+sig;alpha=.12;r=1/alpha;p=r/(r+mu);z=rng.negative_binomial(r,p).astype(float);return {'time_s':t,'theta_axis':th,'intensity':z},bg,alpha

def test_auto_background_converges_and_is_reasonable():
    from roc_mcs.processing.background import estimate_background_auto
    roc,true_bg,alpha=_synthetic();b,a=estimate_background_auto(roc,noise='nb',nb_alpha=alpha,n_basis_bg=5,max_refine_iter=30,return_support=True)
    assert a.metadata['converged'];assert a.metadata['last_mask_change_fraction']==0;assert 0.2<a.metadata['support_fraction']<0.8
    assert abs(b.metadata['support_residual_mean'])<.1;assert .7<b.metadata['support_residual_sd']<1.3
    rel=np.median(np.abs(b.background_map[a.support_mask]-true_bg[a.support_mask])/np.maximum(true_bg[a.support_mask],1e-9));assert rel<.2

def test_production_fails_closed_without_convergence():
    import pytest
    from roc_mcs.processing.background import estimate_background_auto
    roc,_,alpha=_synthetic()
    with pytest.raises(RuntimeError,match='не достигнута сходимость'):estimate_background_auto(roc,noise='nb',nb_alpha=alpha,n_basis_bg=5,max_refine_iter=1,require_convergence=True)

def test_sensitivity_contract():
    from roc_mcs.processing.background import audit_background_cii_sensitivity
    roc,_,alpha=_synthetic();a=audit_background_cii_sensitivity(roc,noise='nb',nb_alpha=alpha,n_basis_bg=5,fdr_values=(.005,.01,.02),max_refine_iter=30)
    assert set(a)=={'runs','values','ranges','summary','run_table','fdr_values','levels','dtheta','note'};assert set(a['run_table']['fdr'])=={.005,.01,.02};assert a['run_table']['converged'].all();assert {'q50','q95','w80_cii','w90_cii','centroid','area'}<=set(a['summary']['metric'])

def test_visualization_smoke():
    import matplotlib;matplotlib.use('Agg')
    from roc_mcs.processing.background import run_background_cii_auto,audit_background_cii_sensitivity
    from roc_mcs.visualization.processing import plot_background_auto_convergence,plot_background_support_map,plot_background_support_slices,plot_background_only_diagnostics,plot_cii_sensitivity,plot_cii_sensitivity_ranges,plot_cii_robustness_publication
    roc,_,alpha=_synthetic();r=run_background_cii_auto(roc,noise='nb',nb_alpha=alpha,n_basis_bg=5,max_refine_iter=30,time_variation_audit=False);b,a,c=r['background'],r['support'],r['cii'];s=audit_background_cii_sensitivity(roc,noise='nb',nb_alpha=alpha,n_basis_bg=5,fdr_values=(.005,.01,.02),max_refine_iter=30)
    figs=[plot_background_auto_convergence(a)[0],plot_background_support_map(b,autos=a)[0],plot_background_support_slices(b,(0,5,10,19),autos=a)[0],plot_background_only_diagnostics(b,c,(0,10,19))[0],plot_cii_sensitivity(s)[0],plot_cii_sensitivity_ranges(s)[0],plot_cii_robustness_publication(s)[0]]
    assert all(f is not None for f in figs)
