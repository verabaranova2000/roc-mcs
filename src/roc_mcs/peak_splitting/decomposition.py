from __future__ import annotations
import numpy as np
import pandas as pd

VERSION="peak_splitting_decomposition_v1"
_STAGE_GH_SCHEMA="stage_GH_ebh_rci_v1"

def _profile_metrics(spec, theta, names, vec):
    theta = np.asarray(theta, float); v = np.asarray(vec, float); q = len(names)
    if v.shape != (q,) or np.any(~np.isfinite(v)) or theta.ndim != 1 or len(theta) < 2 or np.any(~np.isfinite(theta)) or np.any(np.diff(theta) <= 0):
        return {"profile_ok": False, "S_parameter": np.nan, "theta0": np.nan, "FWHM": np.nan, "eta": np.nan, "area_window": np.nan}
    p = dict(zip(names, map(float, v)))
    try:
        y = np.asarray(spec.func(theta, **p), float)
        if y.shape != theta.shape or np.any(~np.isfinite(y)): raise ValueError("bad model profile")
        area = float(np.trapezoid(y, theta) if hasattr(np, "trapezoid") else np.trapz(y, theta))
    except Exception:
        y = None; area = np.nan
    center_name = next((k for k in ("theta0", "mu", "center", "x0") if k in p), None)
    width_name = next((k for k in ("H", "fwhm", "FWHM") if k in p), None)
    eta_name = next((k for k in ("eta", "mix", "mixing", "fraction") if k in p), None)
    width = float(p[width_name]) if width_name else np.nan
    if not np.isfinite(width) and y is not None and np.nanmax(y) > 0:
        # Полуширина по профилю в измеренном диапазоне; NaN при обрезанной вершине/хвосте.
        grid = np.linspace(theta[0], theta[-1], max(1025, 32 * len(theta)))
        try:
            yg = np.asarray(spec.func(grid, **p), float); peak = int(np.nanargmax(yg)); half = float(yg[peak]) / 2
            lefts = np.flatnonzero(yg[:peak] < half); rights = np.flatnonzero(yg[peak + 1:] < half)
            if peak > 0 and len(lefts) and len(rights) and peak < len(grid) - 1:
                il = int(lefts[-1]); ir = int(peak + 1 + rights[0])
                xl = np.interp(half, yg[il:il + 2], grid[il:il + 2]); xr = np.interp(half, yg[ir - 1:ir + 1][::-1], grid[ir - 1:ir + 1][::-1])
                width = float(xr - xl)
        except Exception: pass
    center = float(p[center_name]) if center_name else np.nan
    sparam = float(p["S"]) if "S" in p else np.nan
    eta = float(p[eta_name]) if eta_name else np.nan
    physical = bool(np.all(np.isfinite(v)) and (not np.isfinite(sparam) or sparam > 0) and (not np.isfinite(width) or width > 0) and (not np.isfinite(eta) or 0 <= eta <= 1) and (not np.isfinite(center) or theta[0] <= center <= theta[-1]) and (not np.isfinite(area) or area > 0))
    return {"profile_ok": bool(y is not None and np.isfinite(area)), "S_parameter": sparam, "theta0": center, "FWHM": width, "eta": eta, "area_window": area, "physical_ok": physical}

def _peak_id_at(g, i, j):
    if "peak_id" not in g: return -1
    a = np.asarray(g["peak_id"])
    try: return int(a[i, j] if a.ndim == 2 else a[i])
    except Exception: return -1

def build_m0_components(m0_results, theta, spec):
    """M0 RCI table: all fitted components, stratified by model order K and local rank."""
    names = tuple(spec.param_names); q = len(names); rows = []
    for g in m0_results:
        K = int(g["K"]); Xs = np.asarray(g["x"], float); ids = np.asarray(g["pixel_id"], np.int64); xs = np.asarray(g["x_pixel"], np.int64); ys = np.asarray(g["y_pixel"], np.int64); ok = np.asarray(g.get("success", np.ones(len(ids), bool)), bool)
        if Xs.shape != (len(ids), K * q + 1) or xs.shape != ids.shape or ys.shape != ids.shape or ok.shape != ids.shape:
            raise ValueError(f"M0 group K={K}: shape inconsistency")
        for i in range(len(ids)):
            if not ok[i]: continue
            for j in range(K):
                vec = Xs[i, j*q:(j+1)*q]; met = _profile_metrics(spec, theta, names, vec)
                rows.append({"pixel_id": int(ids[i]), "x": int(xs[i]), "y": int(ys[i]), "K0": K, "component_index": j,
                             "parent_peak_id": _peak_id_at(g, i, j), "background": float(Xs[i, -1]), "fit_success": True, **met,
                             **{f"param_{n}": float(v) for n, v in zip(names, vec)}})
    cols = ["pixel_id", "x", "y", "K0", "component_index", "parent_peak_id", "background", "fit_success", "profile_ok", "physical_ok", "S_parameter", "theta0", "FWHM", "eta", "area_window"] + [f"param_{n}" for n in names]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out: out[c] = pd.Series(dtype="object")
    return out[cols]

def _subset_hypotheses(h, ids):
    from .hypotheses import SplitHypotheses
    ids = np.asarray(ids, np.int64); pos = {int(v): i for i, v in enumerate(np.asarray(h.hypothesis_id, np.int64))}
    if len(np.unique(ids)) != len(ids): raise ValueError("H refit: повторные candidate IDs")
    missing = [int(v) for v in ids if int(v) not in pos]
    if missing: raise ValueError(f"H refit: IDs отсутствуют в D1: {missing[:10]}")
    ix = np.asarray([pos[int(v)] for v in ids], np.int64)
    return SplitHypotheses(*[np.asarray(getattr(h, k))[ix].copy() for k in ("hypothesis_id", "K0", "pixel_id", "x_pixel", "y_pixel", "parent_index", "parent_peak_id", "theta_left", "theta_right", "m0_x", "m0_lb", "m0_ub", "m0_cost", "n_valid")])

def stage_h_decompose(stage_g, e2_results, m0_results, theta, spec, *, data_yxt=None, valid_yxt=None,
                      d1_results=None, refit=False, B_refit=128, fractions_refit=(.15,.25,.35,.50,.65,.75,.85), solver_config=None, progress=False):
    """H: параметризация и QC e-BH split-кандидатов; significance не пересчитывается."""
    ids = np.asarray(stage_g["selected_ids"], np.int64); names = tuple(spec.param_names); q = len(names); theta = np.asarray(theta, float)
    base_obs = e2_results.get("df_obs", e2_results.get("df_D1"))
    if not isinstance(base_obs, pd.DataFrame): raise ValueError("e2_results не содержит df_obs/df_D1")
    obs = base_obs
    target = None; bounds_by_id = {}; split_engine = None
    if d1_results is not None and len(ids):
        from . import hypotheses as split_engine
        target = _subset_hypotheses(d1_results["hypotheses"], ids)
        for j, hid in enumerate(ids):
            _, lb, ub = split_engine.make_split_null_initialization(target, j, names)
            bounds_by_id[int(hid)] = (np.asarray(lb, float), np.asarray(ub, float))
    if refit and len(ids):
        if data_yxt is None or d1_results is None or target is None: raise ValueError("H refit требует data_yxt и d1_results")
        fit = split_engine.fit_split_jax(data_yxt=np.asarray(data_yxt, float), theta=theta, hypotheses=target, spec=spec, B=int(B_refit), fractions=tuple(fractions_refit), progress=bool(progress), valid_yxt=valid_yxt)
        best = fit.get("best", [])
        if len(best) != len(ids): raise RuntimeError(f"H refit вернул {len(best)}/{len(ids)} M1 fits")
        obs = pd.DataFrame(split_engine.compute_split_statistics(data_yxt, theta, spec, m0_results, best, sigma2_fn=None, valid_yxt=valid_yxt))
        if len(obs) != len(ids) or not np.array_equal(np.sort(obs.hypothesis_id.to_numpy(np.int64)), np.sort(ids)): raise RuntimeError("H refit: ID mismatch")
    byid = obs.set_index("hypothesis_id", drop=False)
    T_F = stage_g["selected"].set_index("hypothesis_id")["T_obs"].to_dict()
    m0tab = build_m0_components(m0_results, theta, spec)
    events = []; child_rows = []; all_m1_rows = []
    e_by_id = stage_g["all"].set_index("hypothesis_id")["e_value"].to_dict()
    if len(ids):
        missing = [int(i) for i in ids if int(i) not in byid.index]
        if missing: raise ValueError(f"Отсутствуют E2 stats для rejected IDs: {missing[:10]}")
    for hid in ids:
        r = byid.loc[int(hid)]
        if isinstance(r, pd.DataFrame): raise ValueError(f"Повторный E2 hypothesis_id={hid}")
        K = int(r["K0"]); parent = int(r["parent_index"]); X = np.asarray(r["M1_x"], float)
        expected = (K + 1) * q + 1
        if X.shape != (expected,): raise ValueError(f"hypothesis_id={hid}: M1_x shape={X.shape}, expected={(expected,)}")
        pair = [(parent, X[parent*q:(parent+1)*q]), (K, X[K*q:(K+1)*q])]
        pair.sort(key=lambda z: float(z[1][names.index(next(k for k in ("theta0", "mu", "center", "x0") if k in names))]))
        centers = [float(v[names.index(next(k for k in ("theta0", "mu", "center", "x0") if k in names))]) for _, v in pair]
        lo = float(r.get("theta_left", np.nan)); hi = float(r.get("theta_right", np.nan)); delta = float(r.get("delta_rss_global", r.get("T_obs", np.nan))); t_formal = float(T_F[int(hid)])
        selected_null = bool(r.get("M1_selected_null", False)) if pd.notna(r.get("M1_selected_null", False)) else False
        center_ok = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= centers[0] <= centers[1] <= hi)
        mets = [_profile_metrics(spec, theta, names, v) for _, v in pair]
        pair_ok = all(m["profile_ok"] and m["physical_ok"] and (not np.isfinite(m["S_parameter"]) or m["S_parameter"] > 0) for m in mets)
        bounds_checked = int(hid) in bounds_by_id; bounds_ok = True
        if bounds_checked:
            lb, ub = bounds_by_id[int(hid)]; tol = 1e-8 * np.maximum(1., np.maximum(np.abs(lb), np.abs(ub)))
            bounds_ok = bool(X.shape == lb.shape == ub.shape and np.all(np.isfinite(X)) and np.all(X >= lb-tol) and np.all(X <= ub+tol))
        solver_checked = "M1_convergence_reason" in r.index and "M1_optimality" in r.index
        if solver_checked:
            reason_code = int(r["M1_convergence_reason"]) if pd.notna(r["M1_convergence_reason"]) else -1
            optval = float(r["M1_optimality"]) if pd.notna(r["M1_optimality"]) else np.nan
            solver_ok = bool(reason_code in (1,2,3) and np.isfinite(optval))
        else: solver_ok = True
        identity_ok = True; identity_error = np.nan; delta_expected = np.nan
        if all(c in r.index for c in ("M0_cost", "M1_cost")) and np.isfinite(delta):
            delta_expected = 2 * (float(r["M0_cost"]) - float(r["M1_cost"]))
            identity_error = abs(delta_expected - delta)
            nloc = max(1, int(r["local_points"])) if "local_points" in r.index and pd.notna(r["local_points"]) else max(1, int(r.get("n_valid", 1)))
            tol = 128 * np.finfo(float).eps * nloc * max(1., abs(delta_expected), abs(delta))
            identity_ok = identity_error <= tol
        quality_ok = bool(pair_ok and center_ok and np.isfinite(delta) and delta >= 0 and identity_ok and bounds_ok and solver_ok and not selected_null)
        x = int(r.get("x_pixel", r.get("x", np.nan))); y = int(r.get("y_pixel", r.get("y", np.nan)))
        event = {"hypothesis_id": int(hid), "pixel_id": int(r["pixel_id"]), "x": x, "y": y, "K0": K, "parent_index": parent,
                 "parent_peak_id": int(r.get("parent_peak_id", -1)), "theta_left": lo, "theta_right": hi, "T_obs": t_formal, "T_refit": delta,
                 "e_value": float(e_by_id[int(hid)]), "M0_cost": float(r["M0_cost"]) if "M0_cost" in r.index and pd.notna(r["M0_cost"]) else np.nan,
                 "M1_cost": float(r["M1_cost"]) if "M1_cost" in r.index and pd.notna(r["M1_cost"]) else np.nan,
                 "relative_rss_gain": float(r["relative_rss_gain"]) if "relative_rss_gain" in r.index and pd.notna(r["relative_rss_gain"]) else np.nan,
                 "locality_ratio": float(r["locality_ratio"]) if "locality_ratio" in r.index and pd.notna(r["locality_ratio"]) else np.nan,
                 "M1_optimality": float(r["M1_optimality"]) if "M1_optimality" in r.index and pd.notna(r["M1_optimality"]) else np.nan,
                 "M1_n_iter": int(r["M1_n_iter"]) if "M1_n_iter" in r.index and pd.notna(r["M1_n_iter"]) else -1,
                 "M1_convergence_reason": int(r["M1_convergence_reason"]) if "M1_convergence_reason" in r.index and pd.notna(r["M1_convergence_reason"]) else -1,
                 "delta_identity_error": identity_error, "delta_expected_from_costs": delta_expected,
                 "separation": centers[1] - centers[0], "left_area_window": mets[0]["area_window"], "right_area_window": mets[1]["area_window"],
                 "left_S_parameter": mets[0]["S_parameter"], "right_S_parameter": mets[1]["S_parameter"],
                 "left_theta0": mets[0]["theta0"], "right_theta0": mets[1]["theta0"], "left_FWHM": mets[0]["FWHM"], "right_FWHM": mets[1]["FWHM"],
                 "left_eta": mets[0]["eta"], "right_eta": mets[1]["eta"],
                 "right_area_fraction": mets[1]["area_window"] / sum(m["area_window"] for m in mets) if np.isfinite(sum(m["area_window"] for m in mets)) and sum(m["area_window"] for m in mets) > 0 else np.nan,
                 "selected_null": selected_null, "center_in_parent_interval": center_ok, "component_parameters_ok": pair_ok, "bounds_checked": bounds_checked, "bounds_ok": bounds_ok,
                 "solver_diagnostics_checked": solver_checked, "solver_ok": solver_ok, "delta_identity_ok": identity_ok, "map_quality_ok": quality_ok}
        events.append(event)
        for side, (slot, vec), met in zip(("lower_theta", "higher_theta"), pair, mets):
            child = {"hypothesis_id": int(hid), "pixel_id": int(r["pixel_id"]), "x": x, "y": y, "K0": K, "parent_index": parent,
                     "parent_peak_id": int(r.get("parent_peak_id", -1)), "component_slot_M1": int(slot), "side_theta": side,
                     "T_obs": t_formal, "T_refit": delta, "e_value": event["e_value"], "selected_null": selected_null, "bounds_checked": bounds_checked, "bounds_ok": bounds_ok,
                     "solver_diagnostics_checked": solver_checked, "solver_ok": solver_ok, "map_quality_ok": quality_ok, **met,
                     **{f"param_{n}": float(v) for n, v in zip(names, vec)}}
            child_rows.append(child)
        for j in range(K + 1):
            vec = X[j*q:(j+1)*q]; met = _profile_metrics(spec, theta, names, vec)
            all_m1_rows.append({"hypothesis_id": int(hid), "pixel_id": int(r["pixel_id"]), "x": x, "y": y, "K0": K,
                                "parent_index": parent, "component_slot_M1": j, "is_split_child": bool(j == parent or j == K),
                                "T_obs": t_formal, "T_refit": delta, "bounds_checked": bounds_checked, "bounds_ok": bounds_ok, "solver_diagnostics_checked": solver_checked, "solver_ok": solver_ok, "map_quality_ok": quality_ok, **met,
                                **{f"param_{n}": float(v) for n, v in zip(names, vec)}})
    event_cols = ["hypothesis_id", "pixel_id", "x", "y", "K0", "parent_index", "parent_peak_id", "theta_left", "theta_right", "T_obs", "T_refit", "e_value", "M0_cost", "M1_cost", "relative_rss_gain", "locality_ratio", "M1_optimality", "M1_n_iter", "M1_convergence_reason", "delta_identity_error", "delta_expected_from_costs", "separation", "left_area_window", "right_area_window", "left_S_parameter", "right_S_parameter", "left_theta0", "right_theta0", "left_FWHM", "right_FWHM", "left_eta", "right_eta", "right_area_fraction", "selected_null", "center_in_parent_interval", "component_parameters_ok", "bounds_checked", "bounds_ok", "solver_diagnostics_checked", "solver_ok", "delta_identity_ok", "map_quality_ok"]
    events = pd.DataFrame(events, columns=event_cols); children = pd.DataFrame(child_rows); m1all = pd.DataFrame(all_m1_rows)
    if len(events) and events.duplicated(["hypothesis_id"]).any(): raise RuntimeError("Duplicate split event")
    return {"m0_components": m0tab, "split_events": events, "split_components": children, "m1_components": m1all,
            "audit": {"version": _STAGE_GH_SCHEMA, "n_rejected_hypotheses": len(ids), "n_split_events": len(events), "n_map_quality_ok": int(events.map_quality_ok.sum()) if len(events) else 0,
                      "n_map_quality_failed": int((~events.map_quality_ok).sum()) if len(events) else 0,
                      "n_bounds_checked": int(events.bounds_checked.sum()) if len(events) else 0,
                      "n_solver_diagnostics_checked": int(events.solver_diagnostics_checked.sum()) if len(events) else 0,
                      "parameter_source": ("Expanded observed M1 refit of selected candidates; no new significance test" if refit else "Observed M1 fits from D1/E2; no new significance test or parameter refit"), "refit": bool(refit)}}

