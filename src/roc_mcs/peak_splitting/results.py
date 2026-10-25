from __future__ import annotations
from pathlib import Path
import time,numpy as np,pandas as pd

VERSION="peak_splitting_results_v1"
_STAGE_GH_SCHEMA="stage_GH_ebh_rci_v1"

# ==================== Stage G / e-BH decision ====================

def _vec(a, name, dtype=None):
    z = np.asarray(a, dtype=dtype)
    if z.ndim != 1:
        raise ValueError(f"{name}: ожидался 1D-массив, shape={z.shape}")
    return z

def _evalues(n_rep, loss):
    n = np.asarray(n_rep, np.int64); loss = np.asarray(loss, np.int8)
    if n.shape != loss.shape or np.any(n < 0) or np.any((loss < 0) | (loss > 1)):
        raise ValueError("Некорректные n_rep/loss_count")
    return np.where(loss == 0, n.astype(float) + 1.0, 0.0)

def _ebh_rule(e, q):
    e = np.asarray(e, float); q = float(q); m = len(e)
    if e.ndim != 1 or not m or np.any(~np.isfinite(e)) or np.any(e < 0) or not 0 < q < 1:
        raise ValueError("Некорректные e-values/q")
    order = np.argsort(-e, kind="mergesort"); es = e[order]
    ok = es >= m / (q * np.arange(1, m + 1, dtype=float))
    rejected = np.zeros(m, bool)
    if np.any(ok):
        k = int(np.flatnonzero(ok)[-1]) + 1
        rejected = e >= m / (q * k)
    return rejected

def _field(h, name, dtype=None):
    if not hasattr(h, name):
        return None
    return np.asarray(getattr(h, name), dtype=dtype)

def stage_g_ebh(ebh_result, d1_results, e2_results, *, q=None, allow_snapshot=False):
    """G: проверяет и оформляет F/e-BH решение; не применяет второй FDR."""
    terminal = bool(ebh_result.get("complete", False))
    snapshot_only = bool(ebh_result.get("snapshot_only", False))
    if not bool(ebh_result.get("family_complete", False)) or (not terminal and not (allow_snapshot and snapshot_only)):
        raise ValueError("Stage G требует завершённый F на полном семействе; snapshot разрешён только явно для диагностики")
    req = ("hypothesis_id", "T_obs", "n_bootstrap", "loss_count", "e_value", "rejected_ids", "fdr_q")
    miss = [k for k in req if k not in ebh_result]
    if miss: raise ValueError(f"F-result не содержит полей: {miss}")
    h = d1_results.get("hypotheses")
    if h is None: raise ValueError("d1_results не содержит hypotheses")
    ids = _vec(ebh_result["hypothesis_id"], "F.hypothesis_id", np.int64)
    T = _vec(ebh_result["T_obs"], "F.T_obs", float)
    n = _vec(ebh_result["n_bootstrap"], "F.n_bootstrap", np.int64)
    loss = _vec(ebh_result["loss_count"], "F.loss_count", np.int8)
    e = _vec(ebh_result["e_value"], "F.e_value", float)
    if len({len(ids), len(T), len(n), len(loss), len(e)}) != 1 or pd.Index(ids).has_duplicates:
        raise ValueError("Несогласованные F arrays/duplicate ID")
    if np.any(~np.isfinite(T)) or np.any(T < 0): raise ValueError("F.T_obs должен быть конечным и неотрицательным")
    qq = float(ebh_result["fdr_q"] if q is None else q)
    if not np.isclose(qq, float(ebh_result["fdr_q"]), rtol=0, atol=0): raise ValueError("q отличается от значения Stage F")
    ee = _evalues(n, loss)
    if not np.allclose(ee, e, rtol=0, atol=0): raise ValueError("e_value не согласован с (n_rep, loss_count)")
    reject = _ebh_rule(e, qq)
    returned = np.sort(np.asarray(ebh_result["rejected_ids"], np.int64))
    if not np.array_equal(np.sort(ids[reject]), returned): raise ValueError("rejected_ids не совпадает с повторной проверкой e-BH rule")
    hid_all = _vec(h.hypothesis_id, "D1.hypothesis_id", np.int64)
    if pd.Index(hid_all).has_duplicates or not np.array_equal(np.sort(hid_all), np.sort(ids)):
        raise ValueError("F не соответствует полному семейству D1")
    if "family_ids" in ebh_result and not np.array_equal(np.sort(np.asarray(ebh_result["family_ids"], np.int64)), np.sort(ids)):
        raise ValueError("family_ids не совпадает с F.hypothesis_id")
    obs = e2_results.get("df_obs", e2_results.get("df_D1"))
    if not isinstance(obs, pd.DataFrame) or "hypothesis_id" not in obs.columns:
        raise ValueError("e2_results должен содержать df_obs/df_D1 с hypothesis_id")
    tcol = "T_obs" if "T_obs" in obs.columns else "delta_rss_global" if "delta_rss_global" in obs.columns else None
    if tcol is None or obs.hypothesis_id.duplicated().any() or not np.array_equal(np.sort(obs.hypothesis_id.to_numpy(np.int64)), np.sort(ids)):
        raise ValueError("E2 hypothesis family/T column mismatch")
    tobs = obs.set_index("hypothesis_id")[tcol].reindex(ids).to_numpy(float)
    if not np.allclose(tobs, T, rtol=1e-12, atol=1e-12): raise ValueError("T_obs в F не совпадает с E2")
    meta = {"hypothesis_id": hid_all}
    for src, dst, dtype in (("pixel_id", "pixel_id", np.int64), ("x_pixel", "x", np.int64), ("y_pixel", "y", np.int64),
                            ("K0", "K0", np.int64), ("parent_index", "parent_index", np.int64),
                            ("parent_peak_id", "parent_peak_id", np.int64), ("theta_left", "theta_left", float), ("theta_right", "theta_right", float)):
        a = _field(h, src, dtype)
        if a is not None:
            if a.shape != hid_all.shape: raise ValueError(f"D1.{src} shape mismatch")
            meta[dst] = a
    tab = pd.DataFrame(meta).merge(pd.DataFrame({"hypothesis_id": ids, "T_obs": T, "n_rep": n, "loss": loss.astype(bool),
                                                "survivor": loss == 0, "e_value": e, "rejected": reject}), on="hypothesis_id", validate="one_to_one")
    scalar_cols = [c for c in ("delta_rss_global", "delta_rss_local", "relative_rss_gain", "locality_ratio", "child_sep", "M0_cost", "M1_cost", "M1_optimality", "M1_n_iter", "M1_convergence_reason", "M1_selected_null") if c in obs.columns and c not in tab.columns]
    if scalar_cols:
        tab = tab.merge(obs[["hypothesis_id"] + scalar_cols], on="hypothesis_id", how="left", validate="one_to_one")
    selected_ids = ids[reject]
    stats = obs.loc[obs.hypothesis_id.isin(selected_ids)].copy().sort_values("hypothesis_id", kind="mergesort").reset_index(drop=True)
    pix = tab.groupby([c for c in ("x", "y") if c in tab], as_index=False).agg(n_hyp=("hypothesis_id", "size"), n_rejected=("rejected", "sum"), detected=("rejected", "max")) if {"x", "y"}.issubset(tab.columns) else pd.DataFrame()
    audit = {"version": _STAGE_GH_SCHEMA, "m": len(ids), "q": qq, "n_rejected": int(reject.sum()), "n_rejected_pixels": int(pix.detected.sum()) if len(pix) else 0,
             "complete": terminal, "snapshot_only": snapshot_only, "decision_source": "Stage F e-BH; no second multiple-testing correction"}
    return {"all": tab.sort_values("hypothesis_id", kind="mergesort").reset_index(drop=True), "selected": tab.loc[tab.rejected].sort_values("hypothesis_id", kind="mergesort").reset_index(drop=True),
            "selected_ids": selected_ids.copy(), "selected_stats": stats, "pixel_summary": pix, "audit": audit}

def stage_g_ebh_snapshot(snapshot, d1_results, e2_results):
    """G для checkpoint snapshot: только текущий диагностический набор, не терминальный результат."""
    req = ("hypothesis_id", "T_obs", "n_rep", "loss_count", "fdr_q")
    miss = [k for k in req if k not in snapshot]
    if miss: raise ValueError(f"Checkpoint F не содержит полей: {miss}")
    ids = _vec(snapshot["hypothesis_id"], "snapshot.hypothesis_id", np.int64)
    T = _vec(snapshot["T_obs"], "snapshot.T_obs", float); n = _vec(snapshot["n_rep"], "snapshot.n_rep", np.int64); loss = _vec(snapshot["loss_count"], "snapshot.loss_count", np.int8)
    if not (len(ids) == len(T) == len(n) == len(loss)): raise ValueError("Checkpoint F arrays mismatch")
    e = _evalues(n, loss); q = float(snapshot["fdr_q"]); rej = _ebh_rule(e, q)
    adapted = {"complete": False, "family_complete": True, "snapshot_only": True, "hypothesis_id": ids, "family_ids": ids.copy(), "T_obs": T,
               "n_bootstrap": n, "loss_count": loss, "e_value": e, "rejected_ids": ids[rej], "fdr_q": q}
    out = stage_g_ebh(adapted, d1_results, e2_results, allow_snapshot=True)
    out["audit"]["decision_source"] = "Diagnostic e-BH snapshot from checkpoint; NOT terminal inference"
    return out

# ==================== Checkpoint reconstruction ====================

def load_ebh_snapshot(path,retries=6,delay=.08):
    """Быстро читает атомарный checkpoint в память."""
    p=Path(path); err=None
    for _ in range(int(retries)):
        try:
            with np.load(p,allow_pickle=False) as z: d={k:np.array(z[k],copy=True) for k in z.files}
            req={"version","controller","hypothesis_id","T_obs","loss_count","n_rep","B","fdr_q","next_attempt"}
            miss=sorted(req.difference(d))
            if miss: raise ValueError(f"Checkpoint неполный: {miss}")
            ids=np.asarray(d["hypothesis_id"],np.int64); T=np.asarray(d["T_obs"],float); loss=np.asarray(d["loss_count"],np.int8); n=np.asarray(d["n_rep"],np.int64)
            if any(x.shape!=(len(ids),) for x in (T,loss,n)) or np.any((loss<0)|(loss>1)) or np.any(n<0): raise ValueError("Некорректное состояние checkpoint")
            return d
        except (OSError,EOFError,ValueError) as e:
            err=e; time.sleep(float(delay))
    raise RuntimeError(f"Не удалось прочитать snapshot: {err}") from err

def summarize_ebh_snapshot(d):
    ids=np.asarray(d["hypothesis_id"],np.int64); loss=np.asarray(d["loss_count"],np.int8); n=np.asarray(d["n_rep"],np.int64); q=float(d["fdr_q"]); m=len(ids); t=int(n.max(initial=0)); S=int(np.sum(loss==0)); H=int(n.sum()); rho=float(q*S*(t+1)/m) if m else np.nan
    full=m*t; saving=1-H/full if full>0 else 0.; req_t=float(m/(q*S)-1) if S else np.inf
    return {"m":m,"rep":int(d["next_attempt"]),"maxB":t,"survivors":S,"frozen":m-S,"hyprep":H,"throughput_saving":saving,"ebh_margin":rho,"required_t_at_current_survivors":req_t,"B":int(d["B"]),"q":q}

def make_ebh_snapshot_df(d,hypotheses=None,df_d1=None):
    ids=np.asarray(d["hypothesis_id"],np.int64); T=np.asarray(d["T_obs"],float); loss=np.asarray(d["loss_count"],np.int8); n=np.asarray(d["n_rep"],np.int64); cur=int(n.max(initial=0))
    df=pd.DataFrame({"hypothesis_id":ids,"T_obs":T,"n_rep":n,"loss":loss.astype(bool),"survivor":loss==0,"censored":(loss==0)&(n==cur)})
    if df_d1 is not None:
        z=df_d1.copy()
        if "hypothesis_id" not in z.columns: raise ValueError("df_d1 должен содержать hypothesis_id")
        keep=[c for c in ("hypothesis_id","pixel_id","x_pixel","y_pixel","parent_index","parent_peak_id","delta_rss_global") if c in z.columns]
        df=df.merge(z[keep].drop_duplicates("hypothesis_id"),on="hypothesis_id",how="left",validate="one_to_one")
    elif hypotheses is not None:
        h=hypotheses; pos={int(x):i for i,x in enumerate(np.asarray(h.hypothesis_id,np.int64))}; ix=np.array([pos[int(x)] for x in ids],np.int64)
        for name in ("pixel_id","x_pixel","y_pixel","parent_index","parent_peak_id"):
            if hasattr(h,name): df[name]=np.asarray(getattr(h,name))[ix]
    return df

def reconstruct_trajectory(d):
    loss=np.asarray(d["loss_count"],np.int8); n=np.asarray(d["n_rep"],np.int64); m=len(n); q=float(d["fdr_q"]); tmax=int(n.max(initial=0)); rows=[]
    H=0
    for t in range(tmax+1):
        S=int(np.sum((loss==0)|((loss==1)&(n>t)))) if t>0 else m
        hyp=int(np.sum(np.minimum(n,t)))
        rho=float(q*S*(t+1)/m) if m else np.nan
        req=float(m/(q*S)-1) if S else np.inf
        rows.append((t,S,S/m if m else np.nan,hyp,m*t,hyp/(m*t) if t else 1.,rho,req))
    return pd.DataFrame(rows,columns=["t","survivors","survivor_fraction","hyprep","full_hyprep","cost_ratio","ebh_margin","required_t"])

# ==================== e-BH result tables ====================

def _v(x,dtype=None):
    a=np.asarray(x,dtype=dtype)
    if a.ndim!=1: raise ValueError(f"Ожидался 1D-массив, shape={a.shape}")
    return a

def _state(snap):
    req=("hypothesis_id","T_obs","n_rep","loss_count","fdr_q")
    miss=[k for k in req if k not in snap]
    if miss: raise ValueError(f"В checkpoint отсутствуют поля: {miss}")
    ids=_v(snap["hypothesis_id"],np.int64); T=_v(snap["T_obs"],float); n=_v(snap["n_rep"],np.int64); loss=_v(snap["loss_count"],np.int8)
    if len({len(ids),len(T),len(n),len(loss)})!=1: raise ValueError("Несогласованные длины checkpoint arrays")
    if pd.Index(ids).has_duplicates: raise ValueError("Повторяющиеся hypothesis_id")
    if np.any(~np.isfinite(T)) or np.any(n<0) or np.any((loss<0)|(loss>1)): raise ValueError("Некорректные T_obs/n_rep/loss_count")
    e=np.where(loss==0,n+1,0).astype(float); q=float(np.asarray(snap["fdr_q"])); m=len(ids)
    order=np.argsort(-e,kind="mergesort"); es=e[order]; k=np.arange(1,m+1); ok=es>=m/(q*k)
    kr=int(np.max(np.flatnonzero(ok))+1) if np.any(ok) else 0; rejected=np.zeros(m,bool)
    if kr: rejected[order[:kr]]=True
    return ids,T,n,loss.astype(bool),e,rejected,q,m

def build_ebh_results(snap,hypotheses,df_d1=None):
    """Собирает согласованные hypothesis- и pixel-level результаты."""
    ids,T,n,loss,e,rejected,q,m=_state(snap); hids=_v(hypotheses.hypothesis_id,np.int64)
    if pd.Index(hids).has_duplicates: raise ValueError("Повторяющиеся hypothesis_id в D1")
    pos={int(v):i for i,v in enumerate(hids)}; miss=[int(v) for v in ids if int(v) not in pos]
    if miss: raise ValueError(f"{len(miss)} checkpoint ID отсутствуют в D1; примеры: {miss[:10]}")
    ix=np.fromiter((pos[int(v)] for v in ids),dtype=np.int64,count=len(ids))
    d={"hypothesis_id":ids,"T_obs":T,"n_rep":n,"loss":loss,"survivor":~loss,"e_value":e,"rejected":rejected}
    for src,dst in (("pixel_id","pixel_id"),("x_pixel","x"),("y_pixel","y"),("parent_index","parent_index"),("parent_peak_id","parent_peak_id"),("theta_left","theta_left"),("theta_right","theta_right")):
        if hasattr(hypotheses,src): d[dst]=np.asarray(getattr(hypotheses,src))[ix]
    hyp=pd.DataFrame(d)
    if not {"x","y"}<=set(hyp.columns): raise ValueError("Hypotheses должны содержать x_pixel/y_pixel")
    if df_d1 is not None:
        z=df_d1.copy()
        if "hypothesis_id" not in z.columns: raise ValueError("df_d1 должен содержать hypothesis_id")
        extra=[c for c in z.columns if c not in hyp.columns and c!="hypothesis_id"]
        if extra: hyp=hyp.merge(z[["hypothesis_id"]+extra],on="hypothesis_id",how="left",validate="one_to_one")
    pix=(hyp.groupby(["x","y"],as_index=False).agg(n_hyp=("hypothesis_id","size"),n_rejected=("rejected","sum"),detected=("rejected","max"),max_T_obs=("T_obs","max"),median_T_obs=("T_obs","median"),max_e=("e_value","max"),max_n_rep=("n_rep","max"),median_n_rep=("n_rep","median"),survivor_fraction=("survivor","mean")))
    pix["rejection_fraction"]=pix["n_rejected"]/pix["n_hyp"]
    pix.attrs.update({"m":m,"q":q,"n_rejected_hyp":int(rejected.sum()),"n_detected_pixels":int(pix.detected.sum())})
    return hyp,pix

def summary_table(hyp,pix):
    m=len(hyp); R=int(hyp.rejected.sum()); npix=len(pix); Rp=int(pix.detected.sum()); t=int(hyp.n_rep.max()) if len(hyp) else 0
    actual=int(hyp.n_rep.sum()); full=m*t; saving=1-actual/full if full else np.nan; q=float(pix.attrs.get("q",np.nan)); S=int(hyp.survivor.sum())
    rho=q*S*(t+1)/m if m and np.isfinite(q) else np.nan
    return pd.DataFrame({"metric":["hypotheses","rejected hypotheses","rejected fraction","pixels","detected pixels","detected pixel fraction","max MC reps","actual hyp-reps","saving vs m*t","e-BH margin rho"],"value":[m,R,R/m if m else np.nan,npix,Rp,Rp/npix if npix else np.nan,t,actual,saving,rho]})

def export_tables(hyp,pix,prefix="real_ebh"):
    """Экспортирует численные результаты для воспроизводимости."""
    hp=f"{prefix}_hypotheses.csv"; pp=f"{prefix}_pixels.csv"; hyp.to_csv(hp,index=False); pix.to_csv(pp,index=False); return hp,pp



# ==================== Synthetic evaluation ====================

def build_synthetic_truth(synth, theta, spec):
    """Truth-компоненты и пиксельные поля из SyntheticScan; только для validation."""
    from .decomposition import _profile_metrics
    theta = np.asarray(theta, float); names = tuple(spec.param_names); rows = []
    cc = getattr(synth, "component_catalog", None)
    if isinstance(cc, pd.DataFrame) and len(cc):
        for r in cc.to_dict("records"):
            p = {n: float(r[n]) for n in names if n in r and pd.notna(r[n])}
            met = _profile_metrics(spec, theta, names, np.asarray([p.get(n, np.nan) for n in names], float))
            rows.append({"x": int(r["x"]), "y": int(r["y"]), "component_id": int(r.get("component_id", -1)), "truth_peak_id": int(r.get("peak_id", -1)),
                         "role": "shoulder" if bool(r.get("is_shoulder", False)) else "main", "is_shoulder": bool(r.get("is_shoulder", False)),
                         "split_weight": float(r.get("split_weight", 0.0)), "split_separation": float(r.get("split_separation", 0.0)), **met,
                         **{f"param_{n}": p.get(n, np.nan) for n in names}})
    else:
        truth = getattr(synth, "truth", None)
        if not isinstance(truth, dict): raise ValueError("synth должен содержать component_catalog либо truth dict")
        for (x, y), rec in truth.items():
            for c in rec.get("components", []):
                p = {n: float(c[n]) for n in names if n in c and pd.notna(c[n])}; met = _profile_metrics(spec, theta, names, np.asarray([p.get(n, np.nan) for n in names], float))
                shoulder = bool(c.get("is_shoulder", int(c.get("component_id", 0)) > 0))
                rows.append({"x": int(x), "y": int(y), "component_id": int(c.get("component_id", -1)), "truth_peak_id": int(c.get("peak_id", -1)), "role": "shoulder" if shoulder else "main",
                             "is_shoulder": shoulder, "split_weight": float(rec.get("split_weight", 0.0)), "split_separation": float(rec.get("split_separation", 0.0)), **met,
                             **{f"param_{n}": p.get(n, np.nan) for n in names}})
    comp = pd.DataFrame(rows)
    pixrows = []
    for (x, y), rec in getattr(synth, "truth", {}).items():
        pixrows.append({"x": int(x), "y": int(y), "split_weight": float(rec.get("split_weight", 0.0)),
                        "split_separation": float(rec.get("split_separation", 0.0)),
                        "truth_split": any(bool(c.get("is_shoulder", int(c.get("component_id", 0)) > 0)) for c in rec.get("components", [])),
                        "n_truth_components": int(rec.get("n_components", len(rec.get("components", []))))})
    pix = pd.DataFrame(pixrows)
    if len(pix) and pix.duplicated(["x", "y"]).any(): raise ValueError("Повторные synthetic truth pixels")
    return comp, pix

def match_synthetic_truth(stage_h, synth, theta, spec):
    """Связывает fitted split children с synthetic main/shoulder только для оценки; решение G не меняется."""
    truth, truth_pix = build_synthetic_truth(synth, theta, spec); children = stage_h["split_components"]
    if children.empty:
        truth_pix["detected"] = False; truth_pix["detection_class"] = np.where(truth_pix.truth_split, "FN", "TN")
        return {"truth_components": truth, "truth_pixels": truth_pix, "matched_components": pd.DataFrame()}
    rows = []
    for hid, g in children.groupby("hypothesis_id", sort=False):
        g = g.sort_values("side_theta", key=lambda s: s.map({"lower_theta": 0, "higher_theta": 1})); x, y = int(g.iloc[0].x), int(g.iloc[0].y)
        tr = truth[(truth.x == x) & (truth.y == y)] if len(truth) else truth
        pid = int(g.iloc[0].parent_peak_id)
        mains = tr[(tr.role == "main") & (tr.truth_peak_id == pid)] if "truth_peak_id" in tr else tr.iloc[0:0]
        if len(mains) == 0: mains = tr[tr.role == "main"]
        if len(mains) > 1: raise ValueError(f"Synthetic truth неоднозначен для пикселя {(x,y)}, parent_peak_id={pid}; требуется provenance-сопоставление")
        main = mains.iloc[0] if len(mains) else None
        shoulders = tr[tr.role == "shoulder"]
        fits = [g.iloc[0], g.iloc[1]]
        best_pair = None
        if main is not None and len(shoulders):
            for _, sh in shoulders.iterrows():
                c_a = abs(float(fits[0].theta0) - float(main.theta0)) + abs(float(fits[1].theta0) - float(sh.theta0))
                c_b = abs(float(fits[0].theta0) - float(sh.theta0)) + abs(float(fits[1].theta0) - float(main.theta0))
                cand = (min(c_a, c_b), ("main", "shoulder") if c_a <= c_b else ("shoulder", "main"), sh)
                if best_pair is None or cand[0] < best_pair[0]: best_pair = cand
            roles, shoulder = best_pair[1], best_pair[2]
        elif main is not None:
            idx = int(np.argmin([abs(float(z.theta0) - float(main.theta0)) for z in fits])); roles = tuple("main" if j == idx else "extra_no_truth" for j in range(2)); shoulder = None
        else:
            roles = ("extra_no_truth", "extra_no_truth"); shoulder = None
        for f, role in zip(fits, roles):
            tv = main if role == "main" else shoulder if role == "shoulder" else None
            row = {k: f[k] for k in f.index}; row["truth_role_match"] = role; row["truth_present"] = tv is not None
            row["truth_component_id"] = int(tv.component_id) if tv is not None else -1
            for key in ("S_parameter", "theta0", "FWHM", "area_window", "eta"):
                row[f"truth_{key}"] = float(tv[key]) if tv is not None and key in tv and pd.notna(tv[key]) else np.nan
                row[f"error_{key}"] = float(row[key] - row[f"truth_{key}"]) if np.isfinite(row.get(key, np.nan)) and np.isfinite(row[f"truth_{key}"]) else np.nan
                den = abs(row[f"truth_{key}"])
                row[f"relative_error_{key}"] = row[f"error_{key}"] / den if np.isfinite(row[f"error_{key}"]) and den > 0 else np.nan
            rows.append(row)
    matched = pd.DataFrame(rows)
    det = stage_h["split_events"].groupby(["x", "y"], as_index=False).agg(detected=("hypothesis_id", "size")) if len(stage_h["split_events"]) else pd.DataFrame(columns=["x", "y", "detected"])
    truth_pix = truth_pix.merge(det, on=["x", "y"], how="left", validate="one_to_one")
    truth_pix["detected"] = truth_pix["detected"].fillna(0).gt(0)
    truth_pix["detection_class"] = np.select([truth_pix.truth_split & truth_pix.detected, ~truth_pix.truth_split & truth_pix.detected, truth_pix.truth_split & ~truth_pix.detected], ["TP", "FP", "FN"], default="TN")
    return {"truth_components": truth, "truth_pixels": truth_pix, "matched_components": matched}

def synthetic_pixel_metrics(truth_pixels):
    """Pixel-level realized metrics for one synthetic realization; realized FDP is not FDR."""
    z=truth_pixels; tp=int((z.detection_class=="TP").sum()); fp=int((z.detection_class=="FP").sum()); fn=int((z.detection_class=="FN").sum()); tn=int((z.detection_class=="TN").sum())
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"precision":tp/(tp+fp) if tp+fp else np.nan,"recall":tp/(tp+fn) if tp+fn else np.nan,
            "realized_FDP":fp/(tp+fp) if tp+fp else 0.0,"specificity":tn/(tn+fp) if tn+fp else np.nan}

# ==================== Stage G/H export ====================

def export_stage_tables(stage_g, stage_h, prefix="stage_GH_ebh_rci"):
    """Сохраняет воспроизводимые таблицы; массивы параметров M1 не экспортируются в CSV."""
    all_path=f"{prefix}_G_all.csv"; sel_path=f"{prefix}_G_selected.csv"; m0_path=f"{prefix}_H_M0_components.csv"; ev_path=f"{prefix}_H_split_events.csv"; ch_path=f"{prefix}_H_split_components.csv"
    stage_g["all"].to_csv(all_path, index=False); stage_g["selected"].to_csv(sel_path, index=False); stage_h["m0_components"].to_csv(m0_path, index=False); stage_h["split_events"].to_csv(ev_path, index=False); stage_h["split_components"].to_csv(ch_path, index=False)
    return all_path, sel_path, m0_path, ev_path, ch_path

