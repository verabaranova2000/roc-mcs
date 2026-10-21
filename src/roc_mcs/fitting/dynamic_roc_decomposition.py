from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Mapping
import numpy as np,pandas as pd
from scipy.interpolate import BSpline
from scipy.ndimage import median_filter
from scipy.special import gammaln
from scipy.stats import nbinom,norm
import jax,jax.numpy as jnp
from jax import lax
jax.config.update('jax_enable_x64',True)



VERSION='dynamic-roc-v1.0'
_EPS=np.finfo(float).eps

@dataclass(frozen=True,slots=True)
class LifecycleConfig:
    model_name: str = 'pvoigt'
    J_max: int = 4
    theta_window: tuple[float,float] | None = None
    noise: str = 'nb'
    criterion: str = 'bic'
    background: str = 'quadratic'
    n_basis_area: int = 6
    n_basis_center: int = 6
    n_basis_width: int = 5
    n_basis_bg: int = 4
    spline_degree: int = 3
    activation_width_scans: float = 1.5
    min_fwhm_points: float = 2.
    instrument_fwhm: float | None = None
    max_fwhm_fraction: float = .75
    nb_alpha: float | None = None
    n_starts: int = 3
    max_iter: int = 2200
    seed: int = 20261019
    spike_filter: bool = True
    spike_window: int = 5
    spike_z: float = 10.
    spike_isolation_ratio: float = .35
    lambda0: float = 1e-3
    lambda_up: float = 10.
    lambda_down: float = .3
    rho_good: float = .75
    rho_bad: float = .25

@dataclass(slots=True)
class LifecycleResult:
    config: LifecycleConfig
    theta: np.ndarray
    time_s: np.ndarray
    data: np.ndarray
    valid: np.ndarray
    spike_mask: np.ndarray
    selected_J: int
    model_table: pd.DataFrame
    branches: pd.DataFrame
    summary: pd.DataFrame
    fitted_map: np.ndarray
    background_map: np.ndarray
    residual_map: np.ndarray
    rq_residual_map: np.ndarray
    candidates: dict[int,dict[str,Any]]
    noise_audit: dict[str,Any]
    metadata: dict[str,Any]

def _basis(t,n,p):
    t = np.asarray(t,float)
    u = np.zeros_like(t) if t[-1]==t[0] else (t-t[0])/(t[-1]-t[0])
    n, p = int(n), int(p)
    if n<p+1:
        raise ValueError('n_basis должен быть >= spline_degree+1')
    inner = np.linspace(0,1,n-p+1)[1:-1]
    knots = np.r_[np.zeros(p+1),inner,np.ones(p+1)]
    B = BSpline.design_matrix(u,knots,p,extrapolate=True).toarray()
    if B.shape!=(len(t),n) or not np.allclose(B.sum(1),1,atol=1e-10):
        raise RuntimeError('Некорректный B-spline basis')
    return B

def detect_isolated_spikes(data,valid=None,*,window=5,z=10.,isolation_ratio=.35):
    y = np.asarray(data,float)
    v = np.isfinite(y) if valid is None else np.asarray(valid,bool)&np.isfinite(y)
    w = int(window)
    if y.ndim!=2 or v.shape!=y.shape or w<3 or w%2!=1 or z<=0:
        raise ValueError('Некорректные параметры spike detector')
    fill = np.nanmedian(np.where(v,y,np.nan),axis=1,keepdims=True)
    work = np.where(v,y,fill)
    med = median_filter(work,size=(1,w),mode='nearest')
    excess = work-med
    sigma = np.sqrt(np.maximum(med,1.))
    left, right = np.roll(work,1,1), np.roll(work,-1,1)
    left[:,0], right[:,-1] = work[:,0], work[:,-1]
    out = (excess>z*sigma)&(np.maximum(left-med,right-med)<=isolation_ratio*np.maximum(excess,0))&v;out[:,[0,-1]]=False
    return out

def _load_spec(model_name,spec):
    if spec is None:
        from roc_mcs.fitting.registry import MODEL_SPECS
        spec =MODEL_SPECS[model_name]
    names=tuple(spec.param_names)
    for n in ('S','theta0','H'):
        if n not in names:raise NotImplementedError(f'Lifecycle-v5 exact-JAX требует area-profile с S, theta0, H; param_names={names}')
    if not callable(getattr(spec,'func',None)) or not callable(getattr(spec,'jax_func',None)):raise ValueError('ModelSpec должен иметь согласованные func и jax_func')
    return spec,names

def _roles(names,fixed):
    fixed=dict(fixed or {});unknown=set(fixed)-set(names)
    if unknown:raise ValueError(f'Неизвестные fixed_params: {sorted(unknown)}')
    extra=set(names)-{'S','theta0','H','eta'}-set(fixed)
    if extra:raise NotImplementedError(f'Для параметров {sorted(extra)} задайте fixed_params; v5-STABLE меняет только S/theta0/H и, опционально, eta')
    return fixed,'eta' in names and 'eta' not in fixed

def _bg(theta,b):
    u=(theta-theta[0])/(theta[-1]-theta[0]);return b[0]*(1-u)**2+2*b[1]*u*(1-u)+b[2]*u*u

def _sigmoid(x):
    x=np.asarray(x,float);out=np.empty_like(x);m=x>=0;out[m]=1/(1+np.exp(-x[m]));e=np.exp(x[~m]);out[~m]=e/(1+e);return out

def _activation(u,on,off,tau):return _sigmoid((u-on)/tau)*_sigmoid((off-u)/tau)

def _loglik(y,mu,noise,alpha):
    y=np.asarray(y,float);mu=np.maximum(np.asarray(mu,float),1e-12)
    if noise=='poisson':return float(np.sum(y*np.log(mu)-mu-gammaln(y+1)))
    if noise=='nb':
        r=1/float(alpha);p=r/(r+mu);return float(np.sum(gammaln(y+r)-gammaln(r)-gammaln(y+1)+r*np.log(p)+y*np.log1p(-p)))
    raise ValueError("noise должен быть 'poisson' или 'nb'")

def _criterion(ll,p,n,kind):
    k=str(kind).lower()
    if k=='bic':return float(-2*ll+p*np.log(n))
    if k=='aic':return float(-2*ll+2*p)
    raise ValueError('criterion должен быть BIC/AIC')

def _layout(J,na,nc,nw,nbg,fit_eta):
    o=0;br=[]
    for j in range(J):
        d={'S':slice(o,o+na)};o+=na;d['theta0']=slice(o,o+nc);o+=nc;d['H']=slice(o,o+nw);o+=nw;d['eta']=slice(o,o+1) if fit_eta else None;o+=1 if fit_eta else 0
        if j:d['on']=slice(o,o+1);o+=1;d['off']=slice(o,o+1);o+=1
        else:d['on']=d['off']=None
        br.append(d)
    bg=[]
    for _ in range(3):bg.append(slice(o,o+nbg));o+=nbg
    return {'branch':br,'bg':bg,'npar':o}

def _fit_coef(B,y):return np.linalg.lstsq(B,np.asarray(y,float),rcond=None)[0]

def _initial(init_result,J,theta,time,data,valid,Ba,Bc,Bw,Bg,Hmin,Hmax,fit_eta,fixed):
    T=len(time)
    S=np.empty((T,J))
    C=np.empty((T,J))
    H=np.empty((T,J))
    E=np.full((T,J),float(fixed.get('eta',.5)))
    bg=np.empty((T,3))
    ok=init_result is not None and hasattr(init_result,'models')
    if ok:
        for i in range(T):
            m=init_result.models.get((i,J))
            if m is None or not bool(m.get('success',True)):ok=False;break
            vn=tuple(m['var_names']);cp=np.asarray(m['components'],float);order=np.argsort(cp[:,vn.index('theta0')]);cp=cp[order]
            if len(cp)!=J or not {'S','theta0','H'}.issubset(vn):ok=False;break
            S[i]=cp[:,vn.index('S')];C[i]=cp[:,vn.index('theta0')];H[i]=cp[:,vn.index('H')]
            if fit_eta and 'eta' in vn:E[i]=cp[:,vn.index('eta')]
            th=np.asarray(m['theta_used'],float);b=np.asarray(m.get('background',[]),float);kind=m.get('bg_kind','linear');u=(th-th[0])/(th[-1]-th[0])
            if kind=='fixed':curve=np.asarray(m['fixed_bg'],float)
            elif kind=='none':curve=np.zeros_like(th)
            elif kind=='constant':curve=np.full_like(th,b[0])
            elif kind=='linear':curve=(1-u)*b[0]+u*b[1]
            elif kind=='quadratic':curve=(1-u)**2*b[0]+2*u*(1-u)*b[1]+u*u*b[2]
            else:curve=np.zeros_like(th)
            bg[i]=[max(float(curve[0]),0),max(float(np.interp(.5*(theta[0]+theta[-1]),th,curve)),0),max(float(curve[-1]),0)]
    if not ok:
        edge=max(2,len(theta)//10)
        for i in range(T):
            y=np.where(valid[i],data[i],np.nan);left=max(float(np.nanmedian(y[:edge])),0);right=max(float(np.nanmedian(y[-edge:])),0);mid=max(float(np.nanmedian(np.r_[y[:edge],y[-edge:]])),0);bg[i]=[left,mid,right];base=_bg(theta,bg[i]);r=np.clip(np.nan_to_num(y,nan=0)-base,0,None);q=np.cumsum(r);C[i]=np.linspace(theta[0]+.2*(theta[-1]-theta[0]),theta[-1]-.2*(theta[-1]-theta[0]),J) if q[-1]<=0 else np.interp((np.arange(J)+.5)/J,q/q[-1],theta);total=max(float(np.trapezoid(r,theta)),1.);S[i]=total/J;H[i]=np.clip((theta[-1]-theta[0])/(6*max(J,1)),Hmin,Hmax)
    return np.maximum(S,1e-8),np.clip(C,theta[0],theta[-1]),np.clip(H,Hmin,Hmax),E,np.maximum(bg,1e-8)

def _pack(J,lay,Ba,Bc,Bw,Bg,S,C,H,E,bg,fit_eta):
    x=np.zeros(lay['npar'])
    for j,d in enumerate(lay['branch']):
        x[d['S']]=_fit_coef(Ba,S[:,j]);x[d['theta0']]=_fit_coef(Bc,C[:,j]);x[d['H']]=_fit_coef(Bw,H[:,j])
        if fit_eta:x[d['eta']]=np.clip(np.nanmedian(E[:,j]),.02,.98)
        if j:x[d['on']]=-.25 if j==1 else min(.8,.2+.18*j);x[d['off']]=1.25
    for k,s in enumerate(lay['bg']):x[s]=_fit_coef(Bg,bg[:,k])
    return x

def _bounds(J,lay,theta,data,Hmin,Hmax,fit_eta):
    area=max(float(np.nanmax(np.trapezoid(np.maximum(data,0),theta,axis=1))),1.);edge=max(float(np.nanmax(data)),1.);lo=np.full(lay['npar'],-np.inf);hi=np.full(lay['npar'],np.inf)
    for j,d in enumerate(lay['branch']):
        lo[d['S']]=0.;hi[d['S']]=10*area;lo[d['theta0']]=theta[0];hi[d['theta0']]=theta[-1];lo[d['H']]=Hmin;hi[d['H']]=Hmax
        if fit_eta:lo[d['eta']]=0.;hi[d['eta']]=1.
        if j:lo[d['on']]=-1.;hi[d['on']]=2.;lo[d['off']]=-1.;hi[d['off']]=2.
    for s in lay['bg']:lo[s]=0.;hi[s]=3*edge
    return lo,hi

def _predict_np(theta,time,x,J,lay,Ba,Bc,Bw,Bg,spec,names,fixed,fit_eta,tau):
    T=len(time);u=np.zeros(T) if time[-1]==time[0] else (time-time[0])/(time[-1]-time[0]);mu=np.empty((T,len(theta)));comps=np.empty((T,J,len(theta)));bgmap=np.empty_like(mu);S=np.empty((T,J));C=np.empty((T,J));H=np.empty((T,J));E=np.empty((T,J));G=np.ones((T,J))
    for j,d in enumerate(lay['branch']):
        S[:,j]=Ba@x[d['S']];C[:,j]=Bc@x[d['theta0']];H[:,j]=Bw@x[d['H']];E[:,j]=float(x[d['eta']][0]) if fit_eta else float(fixed.get('eta',.5))
        if j:G[:,j]=_activation(u,float(x[d['on']][0]),float(x[d['off']][0]),tau)
    bgcoef=np.column_stack([Bg@x[s] for s in lay['bg']])
    for i in range(T):
        bgmap[i]=_bg(theta,bgcoef[i]);mu[i]=bgmap[i]
        for j in range(J):
            p=dict(fixed);p.update(S=float(S[i,j]),theta0=float(C[i,j]),H=float(H[i,j]))
            if 'eta' in names:p['eta']=float(E[i,j])
            raw=np.asarray(spec.func(theta,**{n:p[n] for n in names}),float);comps[i,j]=G[i,j]*raw;mu[i]+=comps[i,j]
    return np.maximum(mu,1e-12),comps,bgmap,S,C,H,E,G,bgcoef

def _resid_and_drdmu(y,mu,noise,alpha):
    mu=jnp.maximum(mu,1e-12);d=y-mu
    if noise=='poisson':
        t=jnp.where(y>0,y*jnp.log(jnp.maximum(y,1e-300)/mu)-(y-mu),mu);t=jnp.maximum(t,0.);root=jnp.sqrt(2*t);r=jnp.sign(d)*root;raw=-jnp.abs(d)/(mu*jnp.maximum(root,1e-300));lim=-1/jnp.sqrt(mu);dr=jnp.where(root>1e-7*jnp.sqrt(jnp.maximum(mu,1.)),raw,lim);return r,dr
    rr=1./float(alpha);t=jnp.where(y>0,y*jnp.log(jnp.maximum(y,1e-300)/mu),0.)-(y+rr)*jnp.log((y+rr)/(mu+rr));t=jnp.maximum(t,0.);root=jnp.sqrt(2*t);r=jnp.sign(d)*root;raw=-rr*jnp.abs(d)/(mu*(mu+rr)*jnp.maximum(root,1e-300));lim=-jnp.sqrt(rr/(mu*(mu+rr)));dr=jnp.where(root>1e-7*jnp.sqrt(jnp.maximum(mu/(1+mu/rr),1e-12)),raw,lim);return r,dr

def _make_jax_eval(theta,time,data,valid,J,spec,names,fixed,fit_eta,cfg,Ba,Bc,Bw,Bg,lay,tau):
    th=jnp.asarray(theta,jnp.float64);tt=jnp.asarray(time,jnp.float64);yy=jnp.asarray(data,jnp.float64);vm=jnp.asarray(valid,bool);Ba=jnp.asarray(Ba);Bc=jnp.asarray(Bc);Bw=jnp.asarray(Bw);Bg=jnp.asarray(Bg);u=jnp.zeros_like(tt) if len(time)<2 or time[-1]==time[0] else (tt-tt[0])/(tt[-1]-tt[0]);uth=(th-th[0])/(th[-1]-th[0]);fixed={k:float(v) for k,v in fixed.items()};npos={n:i for i,n in enumerate(names)};analytic=getattr(spec,'jax_jac_func',None)
    if callable(analytic):
        comp_vj=jax.vmap(lambda *a:analytic(th,*a),in_axes=tuple(0 for _ in names),out_axes=(0,0))
    else:
        def one(*a):
            pv=jnp.stack(a);f=lambda p:spec.jax_func(th,*tuple(p));pred=f(pv);return pred,jax.jacfwd(f)(pv)
        comp_vj=jax.vmap(one,in_axes=tuple(0 for _ in names),out_axes=(0,0))
    phi=jnp.stack(((1-uth)**2,2*uth*(1-uth),uth**2),axis=1)
    def predict_jac(x):
        S=[];C=[];H=[];E=[];G=[];s1s=[];s2s=[]
        for j,d in enumerate(lay['branch']):
            Sj=Ba@x[d['S']];Cj=Bc@x[d['theta0']];Hj=Bw@x[d['H']];Ej=jnp.full(len(time),x[d['eta']][0]) if fit_eta else jnp.full(len(time),float(fixed.get('eta',.5)))
            if j:
                s1=jax.nn.sigmoid((u-x[d['on']][0])/tau);s2=jax.nn.sigmoid((x[d['off']][0]-u)/tau);Gj=s1*s2
            else:s1=jnp.ones(len(time));s2=jnp.ones(len(time));Gj=jnp.ones(len(time))
            S.append(Sj);C.append(Cj);H.append(Hj);E.append(Ej);G.append(Gj);s1s.append(s1);s2s.append(s2)
        S=jnp.stack(S,1);C=jnp.stack(C,1);H=jnp.stack(H,1);E=jnp.stack(E,1);G=jnp.stack(G,1);bgc=jnp.stack([Bg@x[s] for s in lay['bg']],axis=1);bg=bgc@phi.T;Jm=jnp.zeros((len(time),len(theta),lay['npar']),dtype=th.dtype);comps=[]
        for k,s in enumerate(lay['bg']):Jm=Jm.at[:,:,s].set(Bg[:,None,:]*phi[None,:,k,None])
        for j,d in enumerate(lay['branch']):
            vals=[]
            for n in names:
                if n=='S':vals.append(S[:,j])
                elif n=='theta0':vals.append(C[:,j])
                elif n=='H':vals.append(H[:,j])
                elif n=='eta':vals.append(E[:,j])
                else:vals.append(jnp.full(len(time),fixed[n]))
            pred,jp=comp_vj(*vals);Gj=G[:,j];comps.append(Gj[:,None]*pred)
            Jm=Jm.at[:,:,d['S']].set((Gj[:,None]*jp[:,:,npos['S']])[:,:,None]*Ba[:,None,:]);Jm=Jm.at[:,:,d['theta0']].set((Gj[:,None]*jp[:,:,npos['theta0']])[:,:,None]*Bc[:,None,:]);Jm=Jm.at[:,:,d['H']].set((Gj[:,None]*jp[:,:,npos['H']])[:,:,None]*Bw[:,None,:])
            if fit_eta:Jm=Jm.at[:,:,d['eta']].set((Gj[:,None]*jp[:,:,npos['eta']])[:,:,None])
            if j:
                s1=s1s[j];s2=s2s[j];dgon=-(s1*(1-s1)*s2)/tau;dgoff=(s1*s2*(1-s2))/tau;Jm=Jm.at[:,:,d['on']].set((dgon[:,None]*pred)[:,:,None]);Jm=Jm.at[:,:,d['off']].set((dgoff[:,None]*pred)[:,:,None])
        cc=jnp.stack(comps,1);mu=jnp.maximum(bg+jnp.sum(cc,axis=1),1e-12);return mu,cc,bg,S,C,H,E,G,bgc,Jm
    yflat=yy.reshape(-1);vflat=vm.reshape(-1)
    def eval_rj(x):
        out=predict_jac(x);m=out[0].reshape(-1);r,dr=_resid_and_drdmu(yflat,m,cfg.noise,cfg.nb_alpha);r=jnp.where(vflat,r,0.);Jm=out[-1].reshape((-1,lay['npar']));J=jnp.where(vflat[:,None],dr[:,None]*Jm,0.);return r,J,out[:-1]
    def eval_r(x):
        out=predict_jac(x);m=out[0].reshape(-1);r,_=_resid_and_drdmu(yflat,m,cfg.noise,cfg.nb_alpha);return jnp.where(vflat,r,0.),out[:-1]
    def predict(x):return predict_jac(x)[:-1]
    return jax.jit(eval_rj),jax.jit(eval_r),jax.jit(predict)

def _lm_solve(eval_rj,eval_r,x0,lo,hi,*,max_iter=2200,ftol=1e-9,xtol=1e-9,gtol=1e-9,lambda0=1e-3,lambda_up=10.,lambda_down=.3,rho_good=.75,rho_bad=.25,bound_tol=1e-10,ridge=1e-12):
    x=jnp.asarray(x0,jnp.float64);lo=jnp.asarray(lo);hi=jnp.asarray(hi);q=x.size;eye=jnp.eye(q,dtype=x.dtype);finite_lb=jnp.isfinite(lo);finite_ub=jnp.isfinite(hi);fixed=finite_lb&finite_ub&(hi<=lo);eps_lb=bound_tol*jnp.maximum(1.,jnp.abs(jnp.where(finite_lb,lo,0.)));eps_ub=bound_tol*jnp.maximum(1.,jnp.abs(jnp.where(finite_ub,hi,0.)));x=jnp.clip(x,lo,hi);r,J,_=eval_rj(x);cost=.5*jnp.sum(r*r);A=J.T@J;g=J.T@r
    def opt(x,g,A):
        diag=jnp.maximum(jnp.diag(A),ridge);scale=jnp.sqrt(diag);at_lb=finite_lb&(x<=lo+eps_lb);at_ub=finite_ub&(x>=hi-eps_ub);outward=fixed|(at_lb&(g>0))|(at_ub&(g<0));pg=jnp.where(outward,0.,g);return jnp.max(jnp.abs(pg)/scale)
    optimality=opt(x,g,A);done=optimality<=gtol;reason=jnp.where(done,1,0).astype(jnp.int8);it=jnp.array(0,jnp.int32);lam=lambda0*jnp.maximum(jnp.max(jnp.diag(A)),1.)
    def cond(st):return (~st[6])&(st[7]<max_iter)
    def body(st):
        x,cost,J,A,g,lam,done,it,reason=st;diag=jnp.maximum(jnp.diag(A),ridge);at_lb=finite_lb&(x<=lo+eps_lb);at_ub=finite_ub&(x>=hi-eps_ub);free=~(fixed|(at_lb&(g>0))|(at_ub&(g<0)));fo=free[:,None]&free[None,:];Areg=jnp.where(fo,A,0.)+jnp.diag(lam*diag+ridge)+jnp.diag((~free).astype(x.dtype));gf=jnp.where(free,g,0.);delta=jnp.linalg.solve(Areg,-gf);xt=jnp.clip(x+delta,lo,hi);d=xt-x;rt,_=eval_r(xt);ct=.5*jnp.sum(rt*rt);actual=cost-ct;pred=-(jnp.dot(g,d)+.5*jnp.dot(d,A@d));rho=actual/jnp.maximum(pred,1e-30);finite=jnp.isfinite(ct)&jnp.isfinite(pred)&jnp.isfinite(rho);accept=(~done)&finite&(actual>0)&(pred>0)&(rho>0)
        def accept_branch(_):
            rn,Jn,_=eval_rj(xt);An=Jn.T@Jn;gn=Jn.T@rn;return xt,ct,Jn,An,gn
        xn,cn,Jn,An,gn=lax.cond(accept,accept_branch,lambda _: (x,cost,J,A,g),operand=None);factor=jnp.where(rho>rho_good,lambda_down,jnp.where(rho<rho_bad,lambda_up,1.));lamn=jnp.where(done,lam,jnp.clip(lam*jnp.where(accept,factor,lambda_up),1e-15,1e15));step_norm=jnp.max(jnp.abs(d)/jnp.maximum(1.,jnp.abs(x)));cost_ok=jnp.abs(actual)<=ftol*jnp.maximum(cost,1.);step_ok=step_norm<=xtol;optn=opt(xn,gn,An);grad_now=optn<=gtol;step_now=(~grad_now)&accept&step_ok;cost_now=(~grad_now)&(~step_now)&accept&cost_ok;conv=grad_now|step_now|cost_now;itn=it+(~done).astype(jnp.int32);max_now=(~conv)&(itn>=max_iter);reasonn=jnp.where(done,reason,jnp.where(grad_now,1,jnp.where(step_now,2,jnp.where(cost_now,3,jnp.where(max_now,4,0)))));donen=done|conv|max_now;return xn,cn,Jn,An,gn,lamn,donen,itn,reasonn
    st=lax.while_loop(cond,body,(x,cost,J,A,g,lam,done,it,reason));x,cost,J,A,g,lam,done,it,reason=st;success=jnp.isfinite(cost)&jnp.all(jnp.isfinite(x))&(reason>0)&(reason<4);return x,cost,opt(x,g,A),it,reason,success

def _check_model_equivalence(spec,names,theta,S,C,H,E,fixed,*,rtol=2e-9,atol=1e-10):
    ids=np.unique(np.linspace(0,len(S)-1,min(4,len(S)),dtype=int));errs=[]
    for i in ids:
        for j in range(S.shape[1]):
            p=dict(fixed);p.update(S=float(S[i,j]),theta0=float(C[i,j]),H=float(H[i,j]));
            if 'eta' in names:
                p['eta']=float(E[i,j])
            a=np.asarray(spec.func(theta,**{n:p[n] for n in names}),float)
            b=np.asarray(spec.jax_func(jnp.asarray(theta),*[p[n] for n in names]),float)
            scale=max(float(np.max(np.abs(a))),1.)
            errs.append(float(np.max(np.abs(a-b))/scale))
    e=max(errs,default=0.)
    if not np.isfinite(e) or e>rtol+atol:
        raise RuntimeError(f'spec.func и spec.jax_func неэквивалентны: max relative discrepancy={e:.3g}')
    return e

def _fit_J(theta,time,data,valid,J,spec,names,fixed,fit_eta,cfg,init_result,rng):
    Ba=_basis(time,cfg.n_basis_area,cfg.spline_degree)
    Bc=_basis(time,cfg.n_basis_center,cfg.spline_degree)
    Bw=_basis(time,cfg.n_basis_width,cfg.spline_degree)
    Bg=_basis(time,cfg.n_basis_bg,cfg.spline_degree)
    lay=_layout(J,Ba.shape[1],Bc.shape[1],Bw.shape[1],Bg.shape[1],fit_eta)
    dth=float(np.median(np.diff(theta)))
    Hmin=max(cfg.min_fwhm_points*dth,cfg.instrument_fwhm or 0.,_EPS)
    Hmax=max(Hmin*1.01,cfg.max_fwhm_fraction*(theta[-1]-theta[0]))
    tau=max(float(cfg.activation_width_scans)/max(len(time)-1,1),1e-6)
    S,C,H,E,bg=_initial(init_result,J,theta,time,data,valid,Ba,Bc,Bw,Bg,Hmin,Hmax,fit_eta,fixed)
    equiv=_check_model_equivalence(spec,names,theta,S,C,H,E,fixed)
    x0=_pack(J,lay,Ba,Bc,Bw,Bg,S,C,H,E,bg,fit_eta);lo,hi=_bounds(J,lay,theta,data,Hmin,Hmax,fit_eta)
    x0=np.minimum(np.maximum(x0,lo+1e-9),hi-1e-9)
    eval_rj,eval_r,predict=_make_jax_eval(theta,time,data,valid,J,spec,names,fixed,fit_eta,cfg,Ba,Bc,Bw,Bg,lay,tau)
    best=None;scores=[]
    solve=jax.jit(lambda x:_lm_solve(eval_rj,eval_r,x,jnp.asarray(lo),jnp.asarray(hi),max_iter=int(cfg.max_iter),lambda0=cfg.lambda0,lambda_up=cfg.lambda_up,lambda_down=cfg.lambda_down,rho_good=cfg.rho_good,rho_bad=cfg.rho_bad))
    for s in range(cfg.n_starts):
        xx=x0.copy()
        if s:
            for j,d in enumerate(lay['branch']):
                xx[d['theta0']]+=rng.normal(0,1.5*dth,size=Bc.shape[1]);xx[d['S']]*=np.exp(rng.normal(0,.10,size=Ba.shape[1]));xx[d['H']]*=np.exp(rng.normal(0,.07,size=Bw.shape[1]))
                if j:xx[d['on']]+=rng.normal(0,.04);xx[d['off']]+=rng.normal(0,.04)
            xx=np.minimum(np.maximum(xx,lo+1e-10),hi-1e-10)
        xo,cost,opt,nit,reason,success=solve(jnp.asarray(xx));xo=np.asarray(xo);mu,cc,bgm,Sv,Cv,Hv,Ev,Gv,bgc=predict(jnp.asarray(xo));mu=np.asarray(mu);cc=np.asarray(cc);bgm=np.asarray(bgm);Sv=np.asarray(Sv);Cv=np.asarray(Cv);Hv=np.asarray(Hv);Ev=np.asarray(Ev);Gv=np.asarray(Gv);ll=_loglik(data[valid],mu[valid],cfg.noise,cfg.nb_alpha);ic=_criterion(ll,lay['npar'],int(valid.sum()),cfg.criterion);scores.append(ic);rec={'J':J,'criterion':ic,'loglik':ll,'success':bool(success),'optimality':float(opt),'nfev':int(nit),'convergence_reason':int(reason),'message':{1:'gradient',2:'step',3:'cost',4:'max_iter'}.get(int(reason),'unknown'),'x':xo.copy(),'mu':mu,'components':cc,'background':bgm,'S_intrinsic':Sv,'effective_area':Sv*Gv,'theta0':Cv,'H':Hv,'eta':Ev,'activation':Gv,'layout':lay,'start_criteria':None,'tau':tau,'model_equivalence_error':equiv}
        if best is None or ic<best['criterion']:
            best=rec
    best['start_criteria']=np.asarray(scores,float)
    return best

def _conditional_support(data,valid,fit,noise,alpha):
    T,J,_=fit['components'].shape;out=np.zeros((T,J));snr=np.zeros((T,J));neff=np.zeros((T,J))
    for i in range(T):
        v=valid[i]
        y=data[i,v]
        full=fit['mu'][i,v]
        for j in range(J):
            c=np.maximum(fit['components'][i,j,v],0)
            red=np.maximum(full-c,1e-12)
            out[i,j]=max(0.,2*(_loglik(y,full,noise,alpha)-_loglik(y,red,noise,alpha)))
            var=full if noise=='poisson' else full+alpha*full*full
            snr[i,j]=float(np.sqrt(np.sum(c*c/np.maximum(var,1e-12))))
            q=float(np.sum(c*c))
            neff[i,j]=float(c.sum()**2/q) if q>0 else 0.
    return out,snr,neff

def _rq(data,mu,valid,noise,alpha,seed):
    rng=np.random.default_rng(seed)
    out=np.full_like(data,np.nan,float)
    y=np.rint(data[valid]).astype(int)
    m=mu[valid]
    if noise=='poisson':
        from scipy.stats import poisson
        lo=poisson.cdf(y-1,m)
        hi=poisson.cdf(y,m)
    else:
        r=1/alpha
        p=r/(r+m)
        lo=nbinom.cdf(y-1,r,p)
        hi=nbinom.cdf(y,r,p)
    u=lo+rng.random(len(y))*np.maximum(hi-lo,1e-15)
    out[valid]=norm.ppf(np.clip(u,1e-10,1-1e-10))
    return out

def estimate_nb_alpha(data,mu,valid=None,*,trim_quantile=.98):
    y=np.asarray(data,float)
    m=np.asarray(mu,float)
    v=np.isfinite(y)&np.isfinite(m)&(m>0) if valid is None else np.asarray(valid,bool)&np.isfinite(y)&np.isfinite(m)&(m>0)
    yy=y[v]
    mm=m[v]
    a=((yy-mm)**2-mm)/np.maximum(mm*mm,1e-12)
    a=a[np.isfinite(a)]
    if not len(a):
        return 0.
    hi=np.quantile(a,float(trim_quantile))
    return float(max(np.mean(np.clip(a,None,hi)),0.))

def fit_lifecycle_dynamic_rocking(roc_map,*,model_name='pvoigt',spec=None,J_max=4,theta_window=None,source='intensity',valid=None,noise='nb',nb_alpha=None,criterion='bic',background='quadratic',n_basis_area=6,n_basis_center=6,n_basis_width=5,n_basis_bg=4,spline_degree=3,activation_width_scans=1.5,min_fwhm_points=2.,instrument_fwhm=None,max_fwhm_fraction=.75,n_starts=3,max_iter=2200,seed=20261019,spike_filter=True,spike_window=5,spike_z=10.,spike_isolation_ratio=.35,fixed_params=None,init_result=None,progress=True,lambda0=1e-3,lambda_up=10.,lambda_down=.3,rho_good=.75,rho_bad=.25):
    """JAX-реплика v5-STABLE: те же x, bounds, starts и deviance-LS; JAX только ускоряет GN/LM."""
    if background!='quadratic':raise ValueError('Поддерживается неотрицательный quadratic Bernstein background')
    if noise not in {'poisson','nb'}:raise ValueError("noise должен быть 'poisson' или 'nb'")
    if noise=='nb' and (nb_alpha is None or not np.isfinite(nb_alpha) or nb_alpha<=0):raise ValueError('Для NB задайте фиксированный nb_alpha > 0')
    spec,names=_load_spec(model_name,spec);fixed,fit_eta=_roles(names,fixed_params);th0=np.asarray(roc_map.get('theta_axis',roc_map.get('theta')),float);t=np.asarray(roc_map['time_s'],float);z0=np.asarray(roc_map[source],float)
    if z0.shape!=(len(t),len(th0)) or not np.all(np.diff(th0)>0) or not np.all(np.diff(t)>=0):raise ValueError('Некорректные theta/time/data')
    keep=np.ones(len(th0),bool) if theta_window is None else (th0>=min(theta_window))&(th0<=max(theta_window));th=th0[keep];z=z0[:,keep];vv=np.isfinite(z) if valid is None else np.asarray(valid,bool)[:,keep]&np.isfinite(z)
    if keep.sum()<12 or np.any(z[vv]<0):raise ValueError('Count likelihood требует >=12 угловых точек и raw nonnegative counts')
    sm=detect_isolated_spikes(z,vv,window=spike_window,z=spike_z,isolation_ratio=spike_isolation_ratio) if spike_filter else np.zeros_like(vv);vv&=~sm;cfg=LifecycleConfig(model_name,int(J_max),theta_window,noise,criterion,background,int(n_basis_area),int(n_basis_center),int(n_basis_width),int(n_basis_bg),int(spline_degree),float(activation_width_scans),float(min_fwhm_points),instrument_fwhm,float(max_fwhm_fraction),None if nb_alpha is None else float(nb_alpha),int(n_starts),int(max_iter),int(seed),bool(spike_filter),int(spike_window),float(spike_z),float(spike_isolation_ratio),float(lambda0),float(lambda_up),float(lambda_down),float(rho_good),float(rho_bad));rng=np.random.default_rng(seed);cand={};it=range(1,J_max+1)
    if progress:
        from tqdm.auto import tqdm
        it=tqdm(it,desc='Lifecycle exact-JAX models',unit='model')
    for J in it:cand[J]=_fit_J(th,t,z,vv,J,spec,names,fixed,fit_eta,cfg,init_result,rng)
    tab=pd.DataFrame([{'J':J,'criterion':r['criterion'],'loglik':r['loglik'],'npar':r['layout']['npar'],'success':r['success'],'optimality':r['optimality'],'nfev':r['nfev'],'convergence_reason':r['message'],'start_spread':float(np.ptp(r['start_criteria'])),'model_equiv_err':r['model_equivalence_error']} for J,r in cand.items()]).sort_values('J').reset_index(drop=True);tab['delta_criterion']=tab.criterion-tab.criterion.min();w=np.exp(-.5*np.clip(tab.delta_criterion.to_numpy(),0,1400));tab['weight']=w/w.sum();J=int(tab.loc[tab.criterion.idxmin(),'J']);best=cand[J];order=np.argsort(np.median(best['theta0'],axis=0),kind='mergesort');best['components']=best['components'][:,order,:]
    for k in ('S_intrinsic','effective_area','theta0','H','eta','activation'):best[k]=best[k][:,order]
    on_u=np.full(J,-np.inf);off_u=np.full(J,np.inf)
    for j,d in enumerate(best['layout']['branch']):
        if j:on_u[j]=float(best['x'][d['on']][0]);off_u[j]=float(best['x'][d['off']][0])
    on_u=on_u[order];off_u=off_u[order];span=float(t[-1]-t[0]) if len(t)>1 else 0.;on_s=np.where(np.isfinite(on_u),t[0]+on_u*span,-np.inf);off_s=np.where(np.isfinite(off_u),t[0]+off_u*span,np.inf);best['onset_u']=on_u;best['offset_u']=off_u;best['onset_s']=on_s;best['offset_s']=off_s
    sup,snr,neff=_conditional_support(z,vv,best,noise,cfg.nb_alpha);rows=[];u=np.zeros(len(t)) if span<=0 else (t-t[0])/span
    for i in range(len(t)):
        A=best['effective_area'][i];total=max(float(A.sum()),1e-300)
        for j in range(J):
            nominal=bool((u[i]>=on_u[j]) and (u[i]<=off_u[j]))
            rows.append({'scan_id':i,'time_s':t[i],'branch_id':j,'S_intrinsic':float(best['S_intrinsic'][i,j]),'effective_area':float(A[j]),'area_fraction':float(A[j]/total),'theta0':float(best['theta0'][i,j]),'H':float(best['H'][i,j]),'eta':float(best['eta'][i,j]),'activation':float(best['activation'][i,j]),'onset_s':float(on_s[j]),'offset_s':float(off_s[j]),'lifecycle_active':nominal,'conditional_delta_deviance':float(sup[i,j]),'profile_snr':float(snr[i,j]),'support_neff':float(neff[i,j])})
    br=pd.DataFrame(rows);summary=[]
    for i,g in br.groupby('scan_id'):
        g=g.sort_values('branch_id');f=g.area_fraction.to_numpy();c=g.theta0.to_numpy();H=g.H.to_numpy();m=float(np.sum(f*c));summary.append({'scan_id':i,'time_s':t[i],'effective_branch_number':float(np.exp(-np.sum(f*np.log(np.maximum(f,1e-300))))),'weighted_center':m,'between_branch_sigma':float(np.sqrt(np.sum(f*(c-m)**2))),'area_weighted_fwhm':float(np.sum(f*H)),'max_separation':float(c.max()-c.min()) if len(c)>1 else 0.,'dominant_fraction':float(f.max())})
    rq=_rq(z,best['mu'],vv,noise,cfg.nb_alpha,seed+991);var=best['mu'] if noise=='poisson' else best['mu']+cfg.nb_alpha*best['mu']**2;resid=(z-best['mu'])/np.sqrt(np.maximum(var,1e-12));resid[~vv]=np.nan;rr=rq[np.isfinite(rq)];pear=float(np.nansum(resid[vv]**2)/max(vv.sum()-best['layout']['npar'],1));audit={'alpha':cfg.nb_alpha,'pearson_dispersion':pear,'rq_mean':float(np.mean(rr)),'rq_sd':float(np.std(rr,ddof=1))};meta={'version':VERSION,'backend':'jax-projected-gn-lm-deviance','spec_name':getattr(spec,'name',model_name),'param_names':names,'theta_mask':keep,'spike_count':int(sm.sum()),'spike_fraction':float(sm.mean()),'interpretation':'Физическая intensity trajectory = effective_area = activation*S_intrinsic. on/off и S_intrinsic отдельно не интерпретируются без идентификационного анализа.'};return LifecycleResult(cfg,th,t,z,vv,sm,J,tab,br.sort_values(['branch_id','scan_id']).reset_index(drop=True),pd.DataFrame(summary),best['mu'],best['background'],resid,rq,cand,audit,meta)

def print_summary(r):
    print('=== DYNAMIC ROCKING LIFECYCLE v5-STABLE EXACT-JAX v1.1 ===')
    print(f"model={r.metadata['spec_name']}; slices={len(r.time_s)}; selected J={r.selected_J}; noise={r.config.noise}; criterion={r.config.criterion.upper()}")
    print(f"masked spikes={r.metadata['spike_count']} ({r.metadata['spike_fraction']:.3%}); alpha={r.noise_audit['alpha']}; Pearson dispersion={r.noise_audit['pearson_dispersion']:.3f}; RQ mean/sd={r.noise_audit['rq_mean']:.3f}/{r.noise_audit['rq_sd']:.3f}")
    print(r.model_table[['J','criterion','delta_criterion','weight','npar','success','optimality','nfev','convergence_reason','start_spread','model_equiv_err']].to_string(index=False,float_format=lambda x:f'{x:.4g}'));q=r.branches.groupby('branch_id').agg(onset_s=('onset_s','first'),offset_s=('offset_s','first'),max_fraction=('area_fraction','max'),median_fraction=('area_fraction','median'),max_deltaD=('conditional_delta_deviance','max'),median_snr=('profile_snr','median'),median_fwhm=('H','median'))
    print('\nBranches:')
    print(q.to_string(float_format=lambda x:f'{x:.4g}'))


