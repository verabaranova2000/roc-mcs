"""Stage D1: H1 split-гипотезы и batched M1-fit поверх готового M0."""
# split_engine_D1_maskfixed.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from roc_mcs.fitting.backends.jax_backend import make_jax_lm_solver_v4

DEFAULT_SPLIT_FRACTIONS = (0.25, 0.50, 0.75)

DEFAULT_SOLVER = {"max_iter": 300, "ftol": 1e-9, "xtol": 1e-9, "gtol": 1e-8}


@dataclass(frozen=True, slots=True)
class SplitHypotheses:
    """Одна строка = одна H1 для одного компонента M0."""
    hypothesis_id: np.ndarray; K0: np.ndarray; pixel_id: np.ndarray; x_pixel: np.ndarray; y_pixel: np.ndarray
    parent_index: np.ndarray; parent_peak_id: np.ndarray; theta_left: np.ndarray; theta_right: np.ndarray
    m0_x: np.ndarray; m0_lb: np.ndarray; m0_ub: np.ndarray; m0_cost: np.ndarray; n_valid: np.ndarray

@dataclass(frozen=True, slots=True)
class SplitFitBatch:
    """Пачка M1-задач для одного K0 и набора multistart."""
    hypothesis_id: np.ndarray; K0: int; K1: int; pixel_id: np.ndarray; x_pixel: np.ndarray; y_pixel: np.ndarray
    parent_index: np.ndarray; parent_peak_id: np.ndarray; theta_left: np.ndarray; theta_right: np.ndarray
    fraction: np.ndarray; obs: np.ndarray; valid: np.ndarray; x0: np.ndarray; lb: np.ndarray; ub: np.ndarray

def responsibility_interval(theta0, parent_index, theta_min, theta_max):
    """Интервал ответственности по серединам соседних центров."""
    theta0 = np.asarray(theta0, float); m = int(parent_index)
    if theta0.ndim != 1 or not (0 <= m < len(theta0)) or not np.all(np.isfinite(theta0)): raise ValueError("Некорректные центры M0")
    order = np.argsort(theta0); centers = theta0[order]
    if len(centers) > 1 and np.any(np.diff(centers) <= 0): raise ValueError("Центры M0 не образуют строгого порядка")
    pos = int(np.flatnonzero(order == m)[0]); mu = float(theta0[m])
    lo = float(theta_min) if pos == 0 else 0.5 * (float(centers[pos - 1]) + mu)
    hi = float(theta_max) if pos == len(order) - 1 else 0.5 * (mu + float(centers[pos + 1]))
    if not lo < hi: raise ValueError("Пустой интервал ответственности")
    return lo, hi

def _validate_m0_group(group,q0):
    """Проверяет формы одной M0-группы до построения split-гипотез."""
    required=("pixel_id","x_pixel","y_pixel","peak_id","success","cost")
    missing=[name for name in required if name not in group]
    if missing: raise ValueError(f"M0 result missing fields: {missing}")
    K0=int(group["K"])
    if K0<1: raise ValueError("M0 K must be >= 1")
    X=np.asarray(group["x"],float); LB=np.asarray(group["lb"],float); UB=np.asarray(group["ub"],float)
    if X.ndim!=2 or X.shape[1]!=K0*q0+1: raise ValueError(f"Некорректная форма M0 x для K={K0}: {X.shape}")
    if LB.shape!=X.shape or UB.shape!=X.shape: raise ValueError("M0 lb/ub должны иметь форму x")
    n=len(X)
    for name in ("pixel_id","x_pixel","y_pixel","success","cost"):
        if np.asarray(group[name]).shape!=(n,): raise ValueError(f"M0 {name} должен иметь форму (N,)")
    if np.asarray(group["peak_id"]).shape!=(n,K0): raise ValueError("M0 peak_id должен иметь форму (N,K)")
    if "n_valid" in group and np.asarray(group["n_valid"]).shape!=(n,): raise ValueError("M0 n_valid должен иметь форму (N,)")
    if np.any(np.isnan(LB)) or np.any(np.isnan(UB)) or np.any(LB>UB): raise ValueError("M0 bounds содержат NaN или lb > ub")
    return K0,X,LB,UB

def prepare_split_hypotheses(m0_results, theta, *, param_names=("S", "theta0", "H", "eta"), theta_min=None, theta_max=None):
    """Строит split-гипотезы для разных K0; параметры M0 дополняются NaN справа."""
    theta=np.asarray(theta,float); names=tuple(param_names); q0=len(names)
    if names != ("S","theta0","H","eta"): raise NotImplementedError("D1 поддерживает PVoigt (S, theta0, H, eta)")
    if theta.ndim!=1 or len(theta)<2 or not np.all(np.isfinite(theta)) or not np.all(np.diff(theta)>0): raise ValueError("theta должна быть конечной, 1D и строго возрастающей")
    tmin=float(theta[0] if theta_min is None else theta_min); tmax=float(theta[-1] if theta_max is None else theta_max)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or not tmin<tmax: raise ValueError("theta_min/theta_max должны быть конечными и theta_min < theta_max")
    rows=[]; hid=0; max_width=q0+1
    for group in m0_results:
        K0,X,LB,UB=_validate_m0_group(group,q0); max_width=max(max_width,K0*q0+1)
        success=np.asarray(group["success"],bool); cost=np.asarray(group["cost"],float); n_valid=np.asarray(group.get("n_valid",np.full(len(X),len(theta))),int)
        pixel_id=np.asarray(group["pixel_id"],np.int64); xp=np.asarray(group["x_pixel"],np.int32); yp=np.asarray(group["y_pixel"],np.int32); peak_ids=np.asarray(group["peak_id"],np.int64)
        if n_valid.shape!=(len(X),): raise ValueError("M0 n_valid должен иметь форму (N,)")
        for b in range(len(X)):
            if not success[b] or not np.isfinite(cost[b]) or n_valid[b]<=0 or not np.all(np.isfinite(X[b])): continue
            centers=X[b,1:K0*q0:q0]
            for m in range(K0):
                p=X[b,m*q0:(m+1)*q0]; S,mu,H=map(float,p[:3])
                if not (np.isfinite(S) and np.isfinite(mu) and np.isfinite(H) and S>0 and H>0): continue
                try: lo,hi=responsibility_interval(centers,m,tmin,tmax)
                except ValueError: continue
                if not (np.isfinite(lo) and np.isfinite(hi) and lo<hi and lo<=mu<=hi): continue
                rows.append((hid,K0,pixel_id[b],xp[b],yp[b],m,peak_ids[b,m],lo,hi,X[b].copy(),LB[b].copy(),UB[b].copy(),cost[b],n_valid[b])); hid+=1
    z=lambda dtype:np.empty(0,dtype=dtype)
    if not rows:
        e=np.empty((0,max_width),float)
        return SplitHypotheses(z(np.int64),z(np.int32),z(np.int64),z(np.int32),z(np.int32),z(np.int32),z(np.int64),z(float),z(float),e,e.copy(),e.copy(),z(float),z(np.int32))
    def pack(field):
        out=np.full((len(rows),max_width),np.nan,float)
        for i,row in enumerate(rows):
            a=np.asarray(row[field],float)
            if a.ndim!=1 or len(a)>max_width: raise ValueError("Некорректная длина параметров M0")
            out[i,:len(a)]=a
        return out
    meta_dtypes=(np.int64,np.int32,np.int64,np.int32,np.int32,np.int32,np.int64,float,float)
    meta=tuple(np.asarray([r[i] for r in rows],dtype=dtype) for i,dtype in enumerate(meta_dtypes))
    return SplitHypotheses(*meta,m0_x=pack(9),m0_lb=pack(10),m0_ub=pack(11),m0_cost=np.asarray([r[12] for r in rows],float),n_valid=np.asarray([r[13] for r in rows],np.int32))

def _make_split_base(h,row,param_names):
    """Строит общие M1 bounds из валидного префикса M0 для данного K0."""
    names=tuple(param_names); q0=len(names)
    if names != ("S","theta0","H","eta"): raise NotImplementedError("D1 поддерживает PVoigt (S, theta0, H, eta)")
    K0=int(h.K0[row]); parent=int(h.parent_index[row]); need=K0*q0+1
    X=np.asarray(h.m0_x[row],float); LB=np.asarray(h.m0_lb[row],float); UB=np.asarray(h.m0_ub[row],float)
    if any(a.ndim!=1 or a.size<need for a in (X,LB,UB)): raise ValueError(f"Некорректная длина M0-параметров: row={row}, K0={K0}, требуется {need}")
    X=X[:need]; LB=LB[:need]; UB=UB[:need]
    if not 0<=parent<K0: raise ValueError(f"Некорректный parent_index={parent} для K0={K0}")
    q1=(K0+1)*q0+1; iS,iMu=0,1; a,b=parent*q0,K0*q0; p=X[a:a+q0].copy(); S,mu,H=map(float,p[:3]); lo,hi=float(h.theta_left[row]),float(h.theta_right[row])
    if not (np.all(np.isfinite(X)) and np.all(~np.isnan(LB)) and np.all(~np.isnan(UB)) and np.all(LB<=UB) and np.all(X>=LB) and np.all(X<=UB) and np.isfinite(S) and S>0 and np.isfinite(mu) and np.isfinite(H) and H>0 and np.isfinite(lo) and np.isfinite(hi) and lo<hi and lo<=mu<=hi):
        raise ValueError(f"Некорректные M0-параметры/bounds: row={row}, K0={K0}, parent={parent}, S={S}, mu={mu}, H={H}, interval=({lo},{hi})")
    x=np.empty(q1,float); lb=np.empty(q1,float); ub=np.empty(q1,float); n0=K0*q0
    x[:n0]=X[:n0]; lb[:n0]=LB[:n0]; ub[:n0]=UB[:n0]; x[-1]=X[n0]; lb[-1]=LB[n0]; ub[-1]=UB[n0]
    parent_lb=LB[a:a+q0]; parent_ub=UB[a:a+q0]; x[b:b+q0]=p; lb[b:b+q0]=parent_lb; ub[b:b+q0]=parent_ub
    if np.isfinite(parent_ub[iS]) and parent_ub[iS]<=0: raise ValueError("Верхняя граница S parent должна быть > 0")
    lb[b+iS]=0.0
    for j in (parent,K0): lb[j*q0+iMu],ub[j*q0+iMu]=lo,hi
    if not (np.all(lb<=ub) and np.all(np.isfinite(x)) and np.all(x>=lb) and np.all(x<=ub)): raise ValueError("Некорректные M1 bounds/initialization")
    return x,lb,ub,K0,parent,q0,a,b,p,S,mu,H,lo,hi

def make_split_null_initialization(h,row,param_names):
    """Exact H0 старт для M1: возвращает x и bounds."""
    x,lb,ub,K0,parent,q0,a,b,p,*_=_make_split_base(h,row,param_names)
    x[a:a+q0]=p
    x[b:b+q0]=p
    x[b]=0.0
    return x,lb,ub
    
def make_split_initialization(h, row, param_names, fraction):
    """Старт H1: parent → 2 children."""
    x,lb,ub,K0,parent,q0,a,b,p,S,mu,H,lo,hi=_make_split_base(h,row,param_names); iS,iMu=0,1; f=float(fraction)
    if not np.isfinite(f) or f<=0: raise ValueError("fraction must be positive and finite")
    eps=1e-8*max(1.0,hi-lo); d=min(f*H,max(0.0,mu-lo-eps),max(0.0,hi-mu-eps)); d=max(0.0,d)
    pL=p.copy(); pR=p.copy(); pL[iS]=0.5*S; pR[iS]=0.5*S; pL[iMu]=mu-d; pR[iMu]=mu+d
    x[a:a+q0]=pL; x[b:b+q0]=pR
    x[a:a+q0]=np.clip(x[a:a+q0],lb[a:a+q0],ub[a:a+q0]); x[b:b+q0]=np.clip(x[b:b+q0],lb[b:b+q0],ub[b:b+q0])
    return x,lb,ub
    
def make_split_null_x(h,row,param_names):
    """Exact H0 embedding; возвращает только x. (Как старая функция, но через новый хелпер)"""
    x,_,_=make_split_null_initialization(h,row,param_names)
    return x

    
def iter_split_batches(data_yxt,theta,hypotheses,*,param_names,fractions=DEFAULT_SPLIT_FRACTIONS,max_batch=128,valid_yxt=None,include_null_start=False):
    """Генерирует M1 multistart-батчи отдельно для каждого K0."""
    data_yxt=np.asarray(data_yxt); theta=np.asarray(theta,float)
    if data_yxt.ndim!=3 or theta.ndim!=1 or data_yxt.shape[-1]!=len(theta): raise ValueError("Ожидаются data_yxt=(Ny,Nx,Ntheta) и theta=(Ntheta,)")
    if valid_yxt is not None and np.asarray(valid_yxt).shape not in (data_yxt.shape[:2],data_yxt.shape): raise ValueError("valid_yxt должен иметь форму (Ny,Nx) или data_yxt")
    fractions=tuple(float(f) for f in fractions); F=len(fractions); nstarts=F+int(include_null_start)
    if F<1 or any(not np.isfinite(f) or f<=0 for f in fractions): raise ValueError("fractions must be positive finite")
    if int(max_batch)<nstarts: raise ValueError("max_batch must be >= number of starts")
    max_h=int(max_batch)//nstarts; Ny,Nx,M=data_yxt.shape
    for K0 in sorted(np.unique(hypotheses.K0).tolist()):
        idx=np.flatnonzero(hypotheses.K0==K0)
        for start in range(0,len(idx),max_h):
            base=idx[start:start+max_h]
            rows=([(int(r),None) for r in base] if include_null_start else [])+[(int(r),f) for r in base for f in fractions]
            n=len(rows); q0=len(param_names); q1=(int(K0)+1)*q0+1
            obs=np.empty((n,M),float); valid=np.zeros((n,M),bool); x0=np.empty((n,q1),float); lb=np.empty_like(x0); ub=np.empty_like(x0)
            hid=np.empty(n,np.int64); pid=np.empty(n,np.int64); xp=np.empty(n,np.int32); yp=np.empty(n,np.int32); pidx=np.empty(n,np.int32); pk=np.empty(n,np.int64); lo=np.empty(n,float); hi=np.empty(n,float); frac=np.empty(n,float)
            for j,(r,f) in enumerate(rows):
                y,x=int(hypotheses.y_pixel[r]),int(hypotheses.x_pixel[r])
                if not (0<=y<Ny and 0<=x<Nx): raise IndexError(f"Координаты hypothesis_id={int(hypotheses.hypothesis_id[r])} вне data_yxt: (y,x)=({y},{x})")
                obs[j]=np.asarray(data_yxt[y,x],float); v=np.isfinite(obs[j])&np.isfinite(theta)
                if valid_yxt is not None:
                    mask=np.asarray(valid_yxt[y,x],bool); mask=mask if mask.ndim else np.full(M,bool(mask))
                    if mask.shape!=(M,): raise ValueError("valid_yxt[y,x] должен быть scalar или длины Ntheta")
                    v &= mask
                valid[j]=v
                if f is None: x0[j],lb[j],ub[j]=make_split_null_initialization(hypotheses,r,param_names); frac[j]=np.nan
                else: x0[j],lb[j],ub[j]=make_split_initialization(hypotheses,r,param_names,f); frac[j]=f
                hid[j]=hypotheses.hypothesis_id[r]; pid[j]=hypotheses.pixel_id[r]; xp[j]=x; yp[j]=y; pidx[j]=hypotheses.parent_index[r]; pk[j]=hypotheses.parent_peak_id[r]; lo[j]=hypotheses.theta_left[r]; hi[j]=hypotheses.theta_right[r]
            yield SplitFitBatch(hid,int(K0),int(K0+1),pid,xp,yp,pidx,pk,lo,hi,frac,obs,valid,x0,lb,ub)

def pad_split_batch(batch, target_B):
    """Дополняет split-batch до фиксированного размера."""
    n = len(batch.hypothesis_id)
    if n > target_B: raise ValueError("split batch exceeds target_B")
    if n == target_B: return batch, n
    pad = target_B-n; rep=lambda a: np.repeat(a[-1:],pad,axis=0); zeros=lambda a: np.zeros((pad,*a.shape[1:]),dtype=a.dtype)
    return SplitFitBatch(np.r_[batch.hypothesis_id,rep(batch.hypothesis_id)],batch.K0,batch.K1,np.r_[batch.pixel_id,rep(batch.pixel_id)],np.r_[batch.x_pixel,rep(batch.x_pixel)],np.r_[batch.y_pixel,rep(batch.y_pixel)],np.r_[batch.parent_index,rep(batch.parent_index)],np.r_[batch.parent_peak_id,rep(batch.parent_peak_id)],np.r_[batch.theta_left,rep(batch.theta_left)],np.r_[batch.theta_right,rep(batch.theta_right)],np.r_[batch.fraction,rep(batch.fraction)],np.vstack([batch.obs,zeros(batch.obs)]),np.vstack([batch.valid,zeros(batch.valid)]),np.vstack([batch.x0,rep(batch.x0)]),np.vstack([batch.lb,rep(batch.lb)]),np.vstack([batch.ub,rep(batch.ub)])), n

def canonicalize_split_x(x, K0, parent_index, param_names):
    """Устраняет label-switching двух дочерних компонент."""
    names=tuple(param_names); q0=len(names); iMu=names.index("theta0"); parent=int(parent_index); a=parent*q0; b=K0*q0
    if x[a+iMu] <= x[b+iMu]: return x
    y=np.array(x,copy=True); ca=y[a:a+q0].copy(); y[a:a+q0]=y[b:b+q0]; y[b:b+q0]=ca; return y

    
def fit_split_jax(data_yxt,theta,hypotheses,spec,*,B=128,fractions=DEFAULT_SPLIT_FRACTIONS,solver_config=None,progress=False,valid_yxt=None,retry_multipliers=(2,4)):
    """Fit M1; budget растёт только для гипотез без успешного multistart."""
    cfg=dict(DEFAULT_SOLVER) if solver_config is None else dict(solver_config); fractions=tuple(float(f) for f in fractions)
    if not fractions or any(not np.isfinite(f) or f<=0 for f in fractions): raise ValueError("fractions must be positive finite")
    base_iter=int(cfg.get("max_iter",0))
    if base_iter<1: raise ValueError("solver_config['max_iter'] должен быть >= 1")
    retry_multipliers=tuple(int(v) for v in retry_multipliers)
    if any(v<=1 for v in retry_multipliers) or any(b<=a for a,b in zip(retry_multipliers,retry_multipliers[1:])):
        raise ValueError("retry_multipliers должны строго возрастать и быть > 1")

    def subset(h,idx):
        idx=np.asarray(idx,np.int64)
        if idx.ndim!=1 or (len(idx) and (idx.min()<0 or idx.max()>=len(h.hypothesis_id))): raise ValueError("Некорректные индексы SplitHypotheses")
        return SplitHypotheses(*(np.asarray(getattr(h,name))[idx].copy() for name in ("hypothesis_id","K0","pixel_id","x_pixel","y_pixel","parent_index","parent_peak_id","theta_left","theta_right","m0_x","m0_lb","m0_ub","m0_cost","n_valid")))

    def run_pass(hset,cfg_pass,desc):
        solvers={}; starts=[]; pbar=None
        if progress:
            from tqdm.auto import tqdm
            pbar=tqdm(total=len(hset.hypothesis_id)*len(fractions),desc=desc,unit="start",mininterval=0.5)
        try:
            for batch in iter_split_batches(data_yxt,theta,hset,param_names=spec.param_names,fractions=fractions,max_batch=B,valid_yxt=valid_yxt):
                bp,n_actual=pad_split_batch(batch,B); K1=int(bp.K1)
                if K1 not in solvers: solvers[K1],_,_,_=make_jax_lm_solver_v4(spec,K1,max_iter=cfg_pass["max_iter"],ftol=cfg_pass["ftol"],xtol=cfg_pass["xtol"],gtol=cfg_pass["gtol"])
                o=solvers[K1](jnp.asarray(bp.obs),jnp.asarray(bp.valid),jnp.asarray(bp.x0),jnp.asarray(bp.lb),jnp.asarray(bp.ub),jnp.asarray(theta)); sl=slice(0,n_actual)
                X=np.asarray(o["x"])[sl]; cost=np.asarray(o["cost"])[sl]; success=np.asarray(o["success"])[sl]; reason=np.asarray(o["convergence_reason"])[sl]
                starts.append({"K0":batch.K0,"K":K1,"hypothesis_id":batch.hypothesis_id[sl],"fraction":batch.fraction[sl],"x":X,"cost":cost,"success":success,"optimality":np.asarray(o["optimality"])[sl],"n_iter":np.asarray(o["n_iter"])[sl],"convergence_reason":reason,"pixel_id":batch.pixel_id[sl],"x_pixel":batch.x_pixel[sl],"y_pixel":batch.y_pixel[sl],"parent_index":batch.parent_index[sl],"parent_peak_id":batch.parent_peak_id[sl],"theta_left":batch.theta_left[sl],"theta_right":batch.theta_right[sl]})
                if pbar is not None: pbar.update(n_actual)
        finally:
            if pbar is not None: pbar.close()
        return starts

    def summarize(starts):
        best = {}; status = {}
        for g in starts:
            for i, hid0 in enumerate(g["hypothesis_id"]):
                hid = int(hid0)
                s = status.setdefault(hid, {"hypothesis_id": hid, "n_attempts": 0, "n_success": 0, "n_max_iter": 0})
                s["n_attempts"] += 1
                reason = int(g["convergence_reason"][i])
                success = bool(g["success"][i])
                cost = float(g["cost"][i])
                x = g["x"][i]
                s["n_max_iter"] += int(reason == 4)
                # ВАЖНО: success засчитываем только если весь результат действительно пригоден к использованию.
                valid = success and np.isfinite(cost) and np.all(np.isfinite(x))
                if not valid: continue
                s["n_success"] += 1
                cand = {k: (v[i] if isinstance(v, np.ndarray) else v) for k, v in g.items()}
                cand["selected_null"] = False
                if (hid not in best or float(cand["cost"]) < float(best[hid]["cost"])):
                    best[hid] = cand
        return best, status

    all_starts=run_pass(hypotheses,cfg,"M1 split fitting")
    best,status=summarize(all_starts)
    for factor in retry_multipliers:
        unresolved=np.array([hid for hid in hypotheses.hypothesis_id if status.get(int(hid),{}).get("n_success",0)==0],np.int64)
        if not unresolved.size: break
        retry_idx=np.flatnonzero(np.isin(hypotheses.hypothesis_id,unresolved)); retry_h=subset(hypotheses,retry_idx)
        retry_cfg=dict(cfg); retry_cfg["max_iter"]=base_iter*factor
        all_starts.extend(run_pass(retry_h,retry_cfg,f"M1 retry ×{factor}"))
        best,status=summarize(all_starts)

    unresolved=np.array([hid for hid in hypotheses.hypothesis_id if status.get(int(hid),{}).get("n_success",0)==0],np.int64)
    if unresolved.size:
        details={int(h):{k:status.get(int(h),{}).get(k,0) for k in ("n_attempts","n_success","n_max_iter")} for h in unresolved[:10]}
        raise RuntimeError(f"M1: нет валидного converged multistart после escalating retries для {len(unresolved)} гипотез; первые={unresolved[:10].tolist()}, диагностика={details}")

    for r,hid0 in enumerate(hypotheses.hypothesis_id):
        hid=int(hid0); c=best[hid]; null_cost=float(hypotheses.m0_cost[r])
        if null_cost<=float(c["cost"]):
            c["x"]=make_split_null_x(hypotheses,r,spec.param_names); c["cost"]=null_cost; c["fraction"]=np.nan; c["optimality"]=np.nan; c["n_iter"]=0; c["convergence_reason"]=-1; c["selected_null"]=True

    for c in best.values(): c["x"]=canonicalize_split_x(c["x"],int(c["K0"]),int(c["parent_index"]),spec.param_names)
    status=[status[k] for k in sorted(status)]
    return {"best":[best[k] for k in sorted(best)],"starts":all_starts,"status":status}



def evaluate_composite(spec, theta, X, K):
    """NumPy composite prediction для T_obs."""
    X=np.asarray(X,float); q0=len(spec.param_names); comp=X[:,:K*q0].reshape(len(X),K,q0); out=np.zeros((len(X),len(theta)),float)
    for k in range(K): out += np.asarray(spec.batch_func(theta,*[comp[:,k,j] for j in range(q0)]))
    return out+X[:,-1,None]

def compute_split_statistics(data_yxt, theta, spec, m0_results, split_best, *, sigma2_fn=None, valid_yxt=None):
    """Global/local ΔRSS и T=ΔRSS/σ²; p-value вычисляется отдельным bootstrap-модулем."""
    theta=np.asarray(theta,float); m0={}
    for g in m0_results:
        for i,pid in enumerate(np.asarray(g["pixel_id"])): m0[(int(pid),int(g["K"]))]=np.asarray(g["x"])[i]
    rows=[]
    for r in split_best:
        K0=int(r["K0"]); pid=int(r["pixel_id"]); x=int(r["x_pixel"]); y=int(r["y_pixel"]); obs=np.asarray(data_yxt[y,x],float); valid=np.isfinite(obs)&np.isfinite(theta);
        if valid_yxt is not None:
            mask=np.asarray(valid_yxt[y,x],bool); mask=np.asarray(mask if mask.ndim else np.full(len(theta),bool(mask)),bool)
            if mask.shape != (len(theta),): raise ValueError("valid_yxt[y,x] must be scalar or length-M vector")
            valid &= mask
        X0=m0[(pid,K0)]; X1=np.asarray(r["x"])
        mu0=evaluate_composite(spec,theta,X0[None],K0)[0]; mu1=evaluate_composite(spec,theta,X1[None],K0+1)[0]; r0=obs-mu0; r1=obs-mu1
        delta_global=float(np.sum(r0[valid]**2-r1[valid]**2)); local=valid&(theta>=float(r["theta_left"]))&(theta<=float(r["theta_right"])); delta_local=float(np.sum(r0[local]**2-r1[local]**2))
        sigma2=np.nan if sigma2_fn is None else float(sigma2_fn(r0,valid,local)); Tg=delta_global/sigma2 if np.isfinite(sigma2) and sigma2>0 else np.nan; Tl=delta_local/sigma2 if np.isfinite(sigma2) and sigma2>0 else np.nan
        rows.append({"hypothesis_id":int(r["hypothesis_id"]),"pixel_id":pid,"x_pixel":x,"y_pixel":y,"parent_peak_id":int(r["parent_peak_id"]),"parent_index":int(r["parent_index"]),"K0":K0,"K1":K0+1,"best_fraction":float(r["fraction"]),"theta_left":float(r["theta_left"]),"theta_right":float(r["theta_right"]),"delta_rss_global":delta_global,"delta_rss_local":delta_local,"sigma2":sigma2,"T_global":Tg,"T_local":Tl,"local_points":int(local.sum()),"M0_cost":0.5*float(np.sum(r0[valid]**2)),"M1_cost":0.5*float(np.sum(r1[valid]**2)),"M1_optimality":float(r["optimality"]),"M1_n_iter":int(r["n_iter"]),"M1_convergence_reason":int(r["convergence_reason"]),"M1_selected_null":bool(r.get("selected_null",False)),"M0_x":X0.copy(),"M1_x":X1.copy()})
    return rows
