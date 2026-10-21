import numpy as np
import jax.numpy as jnp
import pytest
from roc_mcs.fitting import dynamic_roc_decomposition as dr

class PseudoVoigtSpec:
    name='pvoigt_test';param_names=('S','theta0','H','eta')
    @staticmethod
    def func(th,S,theta0,H,eta):
        th=np.asarray(th,float);x=th-theta0;sig=max(H/2.354820045,1e-12);g=np.exp(-.5*(x/sig)**2);g/=np.trapezoid(g,th);gam=max(H/2,1e-12);l=gam/np.pi/(x*x+gam*gam);l/=np.trapezoid(l,th);return S*((1-eta)*g+eta*l)
    @staticmethod
    def jax_func(th,S,theta0,H,eta):
        x=th-theta0;sig=jnp.maximum(H/2.354820045,1e-12);g=jnp.exp(-.5*(x/sig)**2);g=g/jnp.trapezoid(g,th);gam=jnp.maximum(H/2,1e-12);l=gam/jnp.pi/(x*x+gam*gam);l=l/jnp.trapezoid(l,th);return S*((1-eta)*g+eta*l)

def make_synthetic(seed=7):
    rng=np.random.default_rng(seed);th=np.linspace(-90,90,91);t=np.linspace(0,120,16);u=(t-t[0])/(t[-1]-t[0]);spec=PseudoVoigtSpec();mu=[]
    for q in u:
        b=3.0+0.7*(th-th[0])/(th[-1]-th[0]);m=b+spec.func(th,1200*(1-.35*q),-25+18*q,18+5*q,.30);g=1/(1+np.exp(-(q-.38)/.045));m+=g*spec.func(th,850,6+42*q,22+3*q,.45);mu.append(m)
    mu=np.asarray(mu);alpha=.12;r=1/alpha;p=r/(r+mu);y=rng.negative_binomial(r,p);return {'theta_axis':th,'time_s':t,'intensity':y},alpha

@pytest.fixture(scope='module')
def fit2():
    roc,alpha=make_synthetic();return dr.fit_lifecycle_dynamic_rocking(roc,spec=PseudoVoigtSpec(),model_name='pvoigt_test',J_max=2,theta_window=(-90,90),noise='nb',nb_alpha=alpha,criterion='bic',background='quadratic',n_basis_area=5,n_basis_center=5,n_basis_width=4,n_basis_bg=4,spline_degree=3,activation_width_scans=1.5,min_fwhm_points=2,max_fwhm_fraction=.75,n_starts=2,max_iter=700,spike_filter=False,progress=False,seed=11)

def test_basis_partition_of_unity():
    t=np.linspace(0,1,17);B=dr._basis(t,6,3);assert B.shape==(17,6);np.testing.assert_allclose(B.sum(1),1.,atol=1e-12)

def test_activation_is_bounded_and_has_tails():
    u=np.linspace(0,1,101);g=dr._activation(u,.35,.80,.04);assert np.all((g>=0)&(g<=1));assert g[50]>g[0] and g[50]>g[-1];assert g[0]>0 and g[-1]>0

def test_isolated_spike_detector():
    y=np.full((2,21),10.);y[0,10]=1000.;m=dr.detect_isolated_spikes(y,window=5,z=8.);assert m[0,10];assert m.sum()==1

def test_synthetic_selects_two_branches(fit2):
    assert fit2.selected_J==2;assert list(fit2.model_table.J)==[1,2];assert float(fit2.model_table.loc[fit2.model_table.J==2,'delta_criterion'].iloc[0])==pytest.approx(0.)

def test_reconstruction_identity(fit2):
    best=fit2.candidates[fit2.selected_J];recon=fit2.background_map+best['components'].sum(1);np.testing.assert_allclose(fit2.fitted_map,recon,rtol=2e-10,atol=2e-10)

def test_branch_accounting(fit2):
    b=fit2.branches.sort_values(['scan_id','branch_id']);np.testing.assert_allclose(b.effective_area,b.S_intrinsic*b.activation,rtol=2e-10,atol=2e-10);s=b.groupby('scan_id').area_fraction.sum().to_numpy();np.testing.assert_allclose(s,1.,rtol=1e-12,atol=1e-12);assert set(b.lifecycle_active.unique())<={True,False}

def test_result_contract_and_diagnostics(fit2):
    T,M=fit2.data.shape;J=fit2.selected_J;assert fit2.fitted_map.shape==(T,M)==fit2.background_map.shape==fit2.residual_map.shape==fit2.rq_residual_map.shape;assert fit2.candidates[J]['components'].shape==(T,J,M);assert np.all(np.isfinite(fit2.fitted_map));assert np.all(fit2.fitted_map>0);assert np.isfinite(fit2.noise_audit['pearson_dispersion']);assert np.isfinite(fit2.noise_audit['rq_mean']);assert np.isfinite(fit2.noise_audit['rq_sd']);assert float(fit2.model_table.model_equiv_err.max())<1e-8

def test_negative_counts_are_rejected():
    roc,alpha=make_synthetic();roc['intensity']=roc['intensity'].astype(float);roc['intensity'][0,0]=-1
    with pytest.raises(ValueError,match='raw nonnegative counts'):dr.fit_lifecycle_dynamic_rocking(roc,spec=PseudoVoigtSpec(),model_name='x',J_max=1,noise='nb',nb_alpha=alpha,progress=False)
