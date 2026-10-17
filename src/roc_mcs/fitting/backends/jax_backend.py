import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax


from roc_mcs.fitting.preparation import iter_fit_batches, pad_batch_arrays, prepare_fit_data
from roc_mcs.fitting.result_normalization import broadcast_valid_mask, pixel_valid_mask, summarize_residuals, validate_valid_mask



# ==============================
# JAX
# ==============================

# ----------- Solver ------------------------

CONV_GRAD, CONV_STEP, CONV_COST, CONV_MAXITER = 1, 2, 3, 4
CONV_REASON_NAMES = {1: "gradient", 2: "step", 3: "cost", 4: "max_iter"}


def make_jax_lm_solver_v4(
    spec,
    K,
    *,
    max_iter=200,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    lambda0=1e-3,
    lambda_up=10.0,
    lambda_down=0.3,
    rho_good=0.75,
    rho_bad=0.25,
    bound_tol=1e-10,
    ridge=1e-12,
    microbatch_size=None,
    add_background=True,
    jacobian_mode="auto",
):
    """
    Batch LM/GN; auto использует аналитический JAX-Jacobian, если он зарегистрирован.
    jacobian_mode :
        Допустимы три режима:
            auto      ->  analytic, если доступен, иначе autodiff
            analytic
            autodiff
    """
    if spec.jax_func is None:
        raise ValueError(f"JAX backend is not available for model {spec.name!r}")
    jacobian_mode=str(jacobian_mode).lower()
    if jacobian_mode not in {"auto","analytic","autodiff"}: raise ValueError("jacobian_mode должен быть 'auto', 'analytic' или 'autodiff'")
    analytic_func=getattr(spec,"jax_jac_func",None); use_analytic=analytic_func is not None and jacobian_mode in {"auto","analytic"}
    if jacobian_mode=="analytic" and analytic_func is None: raise ValueError(f"Для модели {spec.name!r} не зарегистрирован analytic JAX Jacobian")

    K = int(K)
    if K < 1:
        raise ValueError("K must be >= 1")

    if microbatch_size is None:
        microbatch_size = 128 if K == 1 else 4
    microbatch_size = int(microbatch_size)
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be >= 1")

    n_params = len(spec.param_names)
    q_model = K * n_params
    q = q_model + int(add_background)
    eye = jnp.eye(q, dtype=jnp.float64)
    model_func = spec.jax_func

    def component_model(params,theta): return model_func(theta,*params)
    model_components=jax.vmap(component_model,in_axes=(0,None),out_axes=0)

    def model_single(params,theta):
        comp=params[:q_model].reshape(K,n_params); prediction=jnp.sum(model_components(comp,theta),axis=0)
        return prediction+params[-1] if add_background else prediction

    if use_analytic:
        def component_value_jac(params,theta): return analytic_func(theta,*params)
        component_value_jacs=jax.vmap(component_value_jac,in_axes=(0,None),out_axes=(0,0))
        def model_jac_single(params,theta):
            comp=params[:q_model].reshape(K,n_params); pred_c,jac_c=component_value_jacs(comp,theta); pred=jnp.sum(pred_c,axis=0); J=jnp.transpose(jac_c,(1,0,2)).reshape(theta.shape[0],q_model)
            if add_background: pred=pred+params[-1]; J=jnp.concatenate((J,jnp.ones((theta.shape[0],1),dtype=theta.dtype)),axis=1)
            return J,pred
    else:
        def model_with_aux_single(params,theta):
            pred=model_single(params,theta); return pred,pred
        model_jac_single=jax.jacfwd(model_with_aux_single,argnums=0,has_aux=True)

    model_jac_batch=jax.vmap(model_jac_single,in_axes=(0,None),out_axes=(0,0))
    def residual_single(params,y,valid,theta):
        pred=model_single(params,theta); y_safe=jnp.where(valid,y,0.0); return jnp.where(valid,pred-y_safe,0.0)
    model_batch=jax.vmap(model_single,in_axes=(0,None),out_axes=0); residual_batch=jax.vmap(residual_single,in_axes=(0,0,0,None),out_axes=0)
    if use_analytic:
        def jac_single(params,y,valid,theta): return jnp.where(valid[:,None],model_jac_single(params,theta)[0],0.0)
    else:
        jac_single=jax.jacfwd(residual_single,argnums=0)
    jac_batch=jax.vmap(jac_single,in_axes=(0,0,0,None),out_axes=0)

    def solve_group(obs, valid, x0, lb, ub, theta):
        y_safe = jnp.where(valid, obs, 0.0)
        finite_lb = jnp.isfinite(lb)
        finite_ub = jnp.isfinite(ub)
        fixed = finite_lb & finite_ub & (ub <= lb)
        eps_lb = bound_tol * jnp.maximum(1.0, jnp.abs(jnp.where(finite_lb, lb, 0.0)))
        eps_ub = bound_tol * jnp.maximum(1.0, jnp.abs(jnp.where(finite_ub, ub, 0.0)))

        x = jnp.clip(jnp.asarray(x0, dtype=jnp.float64), lb, ub)
        J_model, pred = model_jac_batch(x, theta)
        J = jnp.where(valid[:, :, None], J_model, 0.0)
        r = jnp.where(valid, pred - y_safe, 0.0)
        cost = 0.5 * jnp.sum(r * r, axis=1)
        A = jnp.einsum("bmq,bmr->bqr", J, J)
        g = jnp.einsum("bmq,bm->bq", J, r)

        def projected_optimality(x, g, A):
            diag = jnp.maximum(jnp.diagonal(A, axis1=1, axis2=2), ridge)
            scale = jnp.sqrt(diag)
            at_lb = finite_lb & (x <= lb + eps_lb)
            at_ub = finite_ub & (x >= ub - eps_ub)
            outward = fixed | (at_lb & (g > 0.0)) | (at_ub & (g < 0.0))
            pg = jnp.where(outward, 0.0, g)
            return jnp.max(jnp.abs(pg) / scale, axis=1)

        optimality = projected_optimality(x, g, A)
        done = optimality <= gtol
        reason = jnp.where(done, CONV_GRAD, 0).astype(jnp.int8)
        n_iter = jnp.zeros(x.shape[0], dtype=jnp.int32)
        lam = lambda0 * jnp.maximum(
            jnp.max(jnp.diagonal(A, axis1=1, axis2=2), axis=1), 1.0
        )

        def cond(state):
            _, _, _, _, _, _, done, n_iter, _ = state
            return jnp.any((~done) & (n_iter < max_iter))

        def body(state):
            x, cost, J, A, g, lam, done, n_iter, reason = state

            diag = jnp.maximum(jnp.diagonal(A, axis1=1, axis2=2), ridge)
            at_lb = finite_lb & (x <= lb + eps_lb)
            at_ub = finite_ub & (x >= ub - eps_ub)
            free = ~(fixed | (at_lb & (g > 0.0)) | (at_ub & (g < 0.0)))

            free_outer = free[:, :, None] & free[:, None, :]
            Areg = jnp.where(free_outer, A, 0.0)
            regdiag = lam[:, None] * diag + ridge
            Areg = Areg + regdiag[:, :, None] * eye[None, :, :]
            Areg = Areg + (~free).astype(jnp.float64)[:, :, None] * eye[None, :, :]
            g_free = jnp.where(free, g, 0.0)

            delta = jnp.linalg.solve(Areg, -g_free[..., None])[..., 0]
            delta = jnp.where(done[:, None], 0.0, delta)

            x_trial = jnp.clip(x + delta, lb, ub)
            d = x_trial - x

            J_trial_model, pred_trial = model_jac_batch(x_trial, theta)
            J_trial = jnp.where(valid[:, :, None], J_trial_model, 0.0)
            r_trial = jnp.where(valid, pred_trial - y_safe, 0.0)
            cost_trial = 0.5 * jnp.sum(r_trial * r_trial, axis=1)

            actual = cost - cost_trial
            predicted = -(jnp.sum(g * d, axis=1) + 0.5 * jnp.einsum("bi,bij,bj->b", d, A, d))
            rho = actual / jnp.maximum(predicted, 1e-30)

            finite = jnp.isfinite(cost_trial) & jnp.isfinite(predicted) & jnp.isfinite(rho)
            accept = (~done) & finite & (actual > 0.0) & (predicted > 0.0) & (rho > 0.0)

            x_new = jnp.where(accept[:, None], x_trial, x)
            cost_new = jnp.where(accept, cost_trial, cost)

            factor = jnp.where(
                rho > rho_good,
                lambda_down,
                jnp.where(rho < rho_bad, lambda_up, 1.0),
            )
            lam_candidate = jnp.clip(
                lam * jnp.where(accept, factor, lambda_up), 1e-15, 1e15
            )
            lam_new = jnp.where(done, lam, lam_candidate)

            # Для reject сохраняем старые J/A/g без изменения.
            A_trial = jnp.einsum("bmq,bmr->bqr", J_trial, J_trial)
            g_trial = jnp.einsum("bmq,bm->bq", J_trial, r_trial)
            J_new = jnp.where(accept[:, None, None], J_trial, J)
            A_new = jnp.where(accept[:, None, None], A_trial, A)
            g_new = jnp.where(accept[:, None], g_trial, g)

            step_norm = jnp.max(jnp.abs(d) / jnp.maximum(1.0, jnp.abs(x)), axis=1)
            cost_change_ok = jnp.abs(actual) <= ftol * jnp.maximum(cost, 1.0)
            step_ok = step_norm <= xtol

            optimality_new = projected_optimality(x_new, g_new, A_new)
            grad_now = optimality_new <= gtol
            step_now = (~grad_now) & accept & step_ok
            cost_now = (~grad_now) & (~step_now) & accept & cost_change_ok
            converged_now = grad_now | step_now | cost_now

            n_iter_new = n_iter + (~done).astype(jnp.int32)
            max_now = (~converged_now) & (n_iter_new >= max_iter)

            reason_new = jnp.where(
                done,
                reason,
                jnp.where(
                    grad_now,
                    CONV_GRAD,
                    jnp.where(
                        step_now,
                        CONV_STEP,
                        jnp.where(
                            cost_now,
                            CONV_COST,
                            jnp.where(max_now, CONV_MAXITER, 0),
                        ),
                    ),
                ),
            )
            done_new = done | converged_now | max_now

            return x_new, cost_new, J_new, A_new, g_new, lam_new, done_new, n_iter_new, reason_new

        state = lax.while_loop(
            cond,
            body,
            (x, cost, J, A, g, lam, done, n_iter, reason),
        )
        x, cost, J, A, g, lam, done, n_iter, reason = state

        return {
            "x": x,
            "cost": cost,
            "optimality": projected_optimality(x, g, A),
            "n_iter": n_iter,
            "success": jnp.isfinite(cost) & jnp.all(jnp.isfinite(x), axis=1) & ((reason == CONV_GRAD) | (reason == CONV_STEP) | (reason == CONV_COST)),
            "hit_max_iter": reason == CONV_MAXITER,
            "convergence_reason": reason,
            "lambda": lam,
        }


    @jax.jit
    def solve(obs, valid, x0, lb, ub, theta):
        obs = jnp.asarray(obs, dtype=jnp.float64)
        valid = jnp.asarray(valid, dtype=bool)
        x0 = jnp.asarray(x0, dtype=jnp.float64)
        lb = jnp.asarray(lb, dtype=jnp.float64)
        ub = jnp.asarray(ub, dtype=jnp.float64)
        theta = jnp.asarray(theta, dtype=jnp.float64)

        B = obs.shape[0]
        mb = min(microbatch_size, B)
        n_groups = (B + mb - 1) // mb
        padded_B = n_groups * mb
        pad = padded_B - B

        # Дополняем последний microbatch; невалидные строки не влияют на fit.
        obs = jnp.concatenate([obs, jnp.zeros((pad, obs.shape[1]), dtype=obs.dtype)], axis=0)
        valid = jnp.concatenate([valid, jnp.zeros((pad, valid.shape[1]), dtype=bool)], axis=0)
        x0 = jnp.concatenate([x0, jnp.repeat(x0[-1:], pad, axis=0)], axis=0)
        lb = jnp.concatenate([lb, jnp.repeat(lb[-1:], pad, axis=0)], axis=0)
        ub = jnp.concatenate([ub, jnp.repeat(ub[-1:], pad, axis=0)], axis=0)

        obs_g = obs.reshape(n_groups, mb, -1)
        valid_g = valid.reshape(n_groups, mb, -1)
        x0_g = x0.reshape(n_groups, mb, -1)
        lb_g = lb.reshape(n_groups, mb, -1)
        ub_g = ub.reshape(n_groups, mb, -1)

        def scan_body(_, xs):
            obs_i, valid_i, x0_i, lb_i, ub_i = xs
            return None, solve_group(obs_i, valid_i, x0_i, lb_i, ub_i, theta)

        _, groups = lax.scan(
            scan_body,
            None,
            (obs_g, valid_g, x0_g, lb_g, ub_g),
            unroll=1,
        )

        def flatten_groups(a):
            return a.reshape((padded_B,) + a.shape[2:])[:B]

        return {key: flatten_groups(value) for key, value in groups.items()}

    return solve, model_batch, residual_batch, jac_batch


# --------- Упаковка активных и фиксированных параметров ------------
def _expand_jax_batch_parameters(batch, prepared, spec, add_background, B, n_actual, x0, lb, ub):
    """Разворачивает активные параметры в полный порядок spec.param_names; фиксированные зажимает границами."""
    names = tuple(spec.param_names)
    active = tuple(prepared.active_names)
    group = prepared.groups[int(batch.K)]
    fixed_names = set(group.fixed)
    if len(set(active)) != len(active) or set(active) & fixed_names or set(active) | fixed_names != set(names):
        raise ValueError(f"active_names и fixed параметры не покрывают {spec.name!r}: active={active}, fixed={sorted(fixed_names)}")
    if active == names and not fixed_names:
        return x0, lb, ub
    K, A, P = int(batch.K), len(active), len(names)
    expected_active = K * A + int(add_background)
    if x0.shape[1] != expected_active:
        raise ValueError(f"Ожидалось {expected_active} активных параметров на пиксель, получено {x0.shape[1]}")
    q = K * P + int(add_background)
    full_x = np.empty((B, q), dtype=float); full_lb = np.empty_like(full_x); full_ub = np.empty_like(full_x)
    name_pos = {name: i for i, name in enumerate(names)}
    active_pos = {name: i for i, name in enumerate(active)}
    group_rows = {int(pid): i for i, pid in enumerate(np.asarray(group.pixel_id))}
    for i in range(n_actual):
        pixel_id = int(batch.pixel_id[i])
        if pixel_id not in group_rows:
            raise ValueError(f"pixel_id={pixel_id} отсутствует в PreparedFitGroup K={K}")
        gr = group_rows[pixel_id]
        for k in range(K):
            for name in active:
                dst, src = k * P + name_pos[name], k * A + active_pos[name]
                full_x[i, dst], full_lb[i, dst], full_ub[i, dst] = x0[i, src], lb[i, src], ub[i, src]
            for name, values in group.fixed.items():
                dst = k * P + name_pos[name]
                value = float(values[gr, k])
                full_x[i, dst] = full_lb[i, dst] = full_ub[i, dst] = value
        if add_background:
            src, dst = K * A, K * P
            full_x[i, dst], full_lb[i, dst], full_ub[i, dst] = x0[i, src], lb[i, src], ub[i, src]
    if n_actual and n_actual < B:
        full_x[n_actual:] = full_x[n_actual - 1]
        full_lb[n_actual:] = full_lb[n_actual - 1]
        full_ub[n_actual:] = full_ub[n_actual - 1]
    return full_x, full_lb, full_ub


# --------- Production fitter ------------

JAX_SOLVER_CONFIG = {
    "max_iter": 200,
    "ftol": 1e-8,
    "xtol": 1e-8,
    "gtol": 1e-8,
    "jacobian_mode": "auto",
}

def fit_jax(
    data_yxt,
    theta,
    peak_catalog,
    spec,
    *,
    valid_mask=None,
    B=128,
    solver_config=None,
    allow_center_shift=True,
    center_window=40.0,
    add_background=True,
    progress=False,
    prepared_fit=None,
):
    """Выполняет JAX batch fitting; возвращает прежний список batch-результатов."""
    if solver_config is None:
        solver_config = JAX_SOLVER_CONFIG
    data_yxt = np.asarray(data_yxt, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if data_yxt.ndim != 3 or theta.ndim != 1 or len(theta) != data_yxt.shape[-1]:
        raise ValueError("Ожидаются data_yxt формы (Y,X,T) и theta длины T")
    valid_mask = broadcast_valid_mask(valid_mask, data_yxt.shape)
    prepared = prepared_fit if prepared_fit is not None else prepare_fit_data(
        data_yxt=data_yxt, theta=theta, peak_catalog=peak_catalog, spec=spec, valid_mask=valid_mask,
        allow_center_shift=allow_center_shift, center_window=center_window, add_background=add_background,
    )

    solvers = {}
    theta_j = jnp.asarray(theta, dtype=jnp.float64)

    batches = iter_fit_batches(
        data_yxt=data_yxt, 
        prepared=prepared,
        valid_mask=valid_mask,  # <--- 4. ПРОКИНУЛИ В ГЕНЕРАТОР БАТЧЕЙ 
        max_batch_pixels=B
    )
    if progress:
        from tqdm.auto import tqdm

        total_pixels = sum(len(group.pixel_id) for group in prepared.groups.values())
        pbar = tqdm(total=total_pixels, desc="JAX fitting", unit="pixel", mininterval=0.5)
    else:
        pbar = None
        
    results = []
    try:
        for batch in batches:
            K = int(batch.K)
            if K not in solvers:
                solvers[K], _, _, _ = make_jax_lm_solver_v4(
                    spec,
                    K,
                    max_iter=solver_config["max_iter"],
                    ftol=solver_config["ftol"],
                    xtol=solver_config["xtol"],
                    gtol=solver_config["gtol"],
                    add_background=add_background,
                    jacobian_mode=solver_config.get("jacobian_mode","auto"),
                )

            obs, valid, x0, lb, ub, n_actual = pad_batch_arrays(batch, B)
            x0, lb, ub = _expand_jax_batch_parameters(batch, prepared, spec, add_background, B, n_actual, x0, lb, ub)
            expected_q = K * len(spec.param_names) + int(add_background)
            if x0.shape[1] != expected_q:
                raise ValueError(f"JAX solver expects {expected_q} parameters for {spec.name!r}, K={K}, add_background={add_background}; got {x0.shape[1]}")
            out = solvers[K](
                jnp.asarray(obs, dtype=jnp.float64),
                jnp.asarray(valid, dtype=bool),
                jnp.asarray(x0, dtype=jnp.float64),
                jnp.asarray(lb, dtype=jnp.float64),
                jnp.asarray(ub, dtype=jnp.float64),
                theta_j,
            )

            results.append({
                "K": K,
                "x": np.asarray(out["x"])[:n_actual],
                "cost": np.asarray(out["cost"])[:n_actual],
                "success": np.asarray(out["success"])[:n_actual],
                "optimality": np.asarray(out["optimality"])[:n_actual],
                "n_iter": np.asarray(out["n_iter"])[:n_actual],
                "convergence_reason": np.asarray(out["convergence_reason"])[:n_actual],
                "pixel_id": np.asarray(batch.pixel_id)[:n_actual],
                "peak_id": np.asarray(batch.peak_id)[:n_actual],
                "n_valid": np.asarray(batch.valid.sum(axis=1))[:n_actual],
                "lb": np.asarray(lb)[:n_actual],
                "ub": np.asarray(ub)[:n_actual],
                "x_pixel": np.asarray(batch.x)[:n_actual],
                "y_pixel": np.asarray(batch.y)[:n_actual],
            })
            if pbar is not None:
                pbar.update(n_actual)

    finally:
        if pbar is not None:
            pbar.close()
    return results

# --------- Нормализация результатов JAX ------------
def _decode_jax_results(jax_results, data_yxt, theta, peak_catalog, spec, *, prepared_fit, add_background=True, valid_mask=None, noise_sigma=None):
    """Преобразует нативные batch-результаты JAX в параметры и общие пиксельные QC-записи."""
    data_yxt = np.asarray(data_yxt, dtype=float); theta = np.asarray(theta, dtype=float)
    ny, nx, nt = data_yxt.shape
    if theta.shape != (nt,):
        raise ValueError("theta не согласован с data_yxt")
    valid_mask = broadcast_valid_mask(valid_mask, data_yxt.shape)
    names, active_names = tuple(spec.param_names), tuple(prepared_fit.active_names)
    n_params = len(names); fitted_records = []; pixel_records = []
    fit_cube = np.full((ny, nx, nt), np.nan, dtype=np.float32)
    residual_cube = np.full_like(fit_cube, np.nan)
    seen = set()
    for batch_result in jax_results:
        K = int(batch_result["K"]); xs = np.asarray(batch_result["x"], dtype=float)
        pixel_ids = np.asarray(batch_result["pixel_id"], dtype=np.int64)
        peak_ids = np.asarray(batch_result["peak_id"], dtype=np.int64)
        xp = np.asarray(batch_result["x_pixel"], dtype=int); yp = np.asarray(batch_result["y_pixel"], dtype=int)
        group = prepared_fit.groups.get(K)
        if group is None:
            raise ValueError(f"Нет PreparedFitGroup для K={K}")
        group_rows = {int(pid): i for i, pid in enumerate(np.asarray(group.pixel_id))}
        layout = batch_result.get("parameter_layout", "legacy")
        has_bg = bool(batch_result.get("add_background", add_background))
        expected_full = K * n_params + int(has_bg)
        expected_active = K * len(active_names) + int(has_bg)
        for i, vector in enumerate(xs):
            x, y, pixel_id = int(xp[i]), int(yp[i]), int(pixel_ids[i])
            if not (0 <= x < nx and 0 <= y < ny):
                raise ValueError(f"JAX вернул координату вне изображения: ({x},{y})")
            if (x, y) in seen:
                raise ValueError(f"JAX вернул пиксель ({x},{y}) более одного раза")
            seen.add((x, y))
            if pixel_id not in group_rows:
                raise ValueError(f"pixel_id={pixel_id} отсутствует в PreparedFitGroup K={K}")
            group_row = group_rows[pixel_id]
            finite_params = vector.size in (expected_full, expected_active) and np.isfinite(vector).all()
            reason_value = int(np.asarray(batch_result["convergence_reason"])[i])
            reason_name = CONV_REASON_NAMES.get(reason_value, "unknown")
            success = bool(np.asarray(batch_result["success"])[i]) and finite_params and reason_value in (CONV_GRAD, CONV_STEP, CONV_COST)
            background = float(vector[-1]) if has_bg and vector.size else np.nan
            full_params = []
            if finite_params:
                if layout == "full" or vector.size == expected_full:
                    matrix = vector[:K * n_params].reshape(K, n_params)
                    full_params = [{name: float(matrix[k, j]) for j, name in enumerate(names)} for k in range(K)]
                elif vector.size == expected_active:
                    active_values = vector[:K * len(active_names)].reshape(K, len(active_names))
                    for k in range(K):
                        vals = {name: float(value[group_row, k]) for name, value in group.fixed.items()}
                        vals.update({name: float(active_values[k, j]) for j, name in enumerate(active_names)})
                        if set(vals) != set(names):
                            raise ValueError(f"Нельзя восстановить полный набор параметров {spec.name!r}: {sorted(vals)} != {sorted(names)}")
                        full_params.append(vals)
                # Фиксированные параметры никогда не берём из округлённого результата solver-а.
                for k, values in enumerate(full_params):
                    for name, fixed_values in group.fixed.items():
                        values[name] = float(fixed_values[group_row, k])
                if has_bg and not np.isfinite(background):
                    success = False
            n_varying = K * len(active_names) + int(has_bg)
            diagnostics = {"n_data": 0, "n_varying": n_varying, "dof": -n_varying,
                           **{k: np.nan for k in ("chisqr", "redchi", "rmse", "p95_abs_residual", "max_abs_residual", "rmse_over_noise", "aic", "bic")}}
            message = ""; valid = np.zeros(nt, dtype=bool); decode_exception = False
            try:
                intensity, valid = pixel_valid_mask(data_yxt, prepared_fit.theta_valid, valid_mask, x=x, y=y)
                if not np.any(valid):
                    success = False; message = "Нет валидных точек для подгонки"
                elif finite_params:
                    components = [np.asarray(spec.func(theta, *(full_params[k][name] for name in names)), dtype=float) for k in range(K)]
                    if any(c.shape != theta.shape for c in components):
                        raise ValueError("Модель компоненты вернула массив неверной формы")
                    prediction = np.sum(components, axis=0)
                    if has_bg:
                        prediction = prediction + background
                    if not np.isfinite(prediction[valid]).all():
                        raise ValueError("Модель вернула нечисловые значения на валидных точках")
                    diagnostics, residual_valid = summarize_residuals(intensity[valid], prediction[valid], n_varying=n_varying, noise_sigma=noise_sigma)
                    if success:
                        fit_cube[y, x, valid] = prediction[valid].astype(np.float32)
                        residual_cube[y, x, valid] = (intensity[valid] - prediction[valid]).astype(np.float32)
                        for k, vals in enumerate(full_params):
                            record = {"peak_id": int(peak_ids[i, k]), **vals}
                            record.update({f"{name}_error": np.nan for name in names})
                            fitted_records.append(record)
            except Exception as exc:
                success = False; decode_exception = True; message = f"postprocess: {exc!r}"
            status = "success" if success else ("exception" if decode_exception else "fit_failed")
            if not message and not success:
                message = f"Солвер завершился: {reason_name}"
            pixel = {"x": x, "y": y, "n_detected": K, "n_fitted": K if success else 0,
                     "fit_status": status, "model": str(spec.name), **diagnostics,
                     "n_iter": int(np.asarray(batch_result["n_iter"])[i]),
                     "cost": float(np.asarray(batch_result["cost"])[i]),
                     "optimality": float(np.asarray(batch_result["optimality"])[i]),
                     "convergence_reason": reason_name, "message": message}
            if has_bg:
                pixel["bg_background"] = background if success and finite_params else np.nan
                pixel["bg_background_error"] = np.nan
            pixel_records.append(pixel)
    return {"fitted_params": pd.DataFrame.from_records(fitted_records), "pixel_records": pixel_records,
            "fit_cube": fit_cube, "residual_cube": residual_cube,
            "pixel_param_names": ("bg_background",) if add_background else ()}
