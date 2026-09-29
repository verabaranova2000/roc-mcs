# optimize.py
from dataclasses import dataclass
from scipy.optimize import differential_evolution, curve_fit
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax



from roc_mcs.fitting.params import build_parameters
from roc_mcs.fitting.metrics import compute_fit_metrics
from roc_mcs.fitting.results import FitResult
from roc_mcs.fitting.registry import MODEL_SPECS
from roc_mcs.fitting.batching import iter_fit_batches, pad_batch_arrays, prepare_fit_data


@dataclass(slots=True)
class FitConfig:
    model_name: str = "gauss_us"
    seed: int = 42
    use_global: bool = True
    use_local: bool = True


def fit_curve(theta, intensity, config: FitConfig = FitConfig()) -> FitResult:
    spec = MODEL_SPECS[config.model_name]
    model = spec.func

    p0 = spec.guess_fn(theta, intensity)
    bounds_dict = spec.bounds_fn(p0)
    bounds = [(bounds_dict[k][0], bounds_dict[k][1]) for k in spec.param_names]

    # lower, upper = [], []
    # for k in spec.param_names:
    #     lo, hi = bounds_dict[k]
    #     lower.append(lo)
    #     upper.append(hi)
    # bounds_tuple = (lower, upper)

    if config.use_global:
        def loss_fn(p):
            residual = intensity - model(theta, *p)
            return np.sum(residual ** 2)

        de = differential_evolution(loss_fn, bounds=bounds, seed=config.seed)
        p_start = de.x
        y_fit_global = model(theta, *p_start)
    else:
        p_start = np.array(p0, dtype=float)
        y_fit_global = None

    if config.use_local:
        popt, pcov = curve_fit(model, theta, intensity, p0=p_start)  #, bounds=bounds_tuple)
        perr = np.sqrt(np.diag(pcov))
        y_fit_local = model(theta, *popt)
        metrics = compute_fit_metrics(intensity, y_fit_local, n_params=len(popt))
        params = build_parameters(spec.param_names, popt, perr)
    else:
        popt = p_start
        pcov = None
        y_fit_local = None
        metrics = compute_fit_metrics(intensity, y_fit_global, n_params=len(p_start))
        params = build_parameters(spec.param_names, p_start, None)

    return FitResult(
        model=config.model_name,
        parameters=params,
        metrics=metrics,
        y_fit_global=y_fit_global,
        y_fit_local=y_fit_local,
        covariance=pcov,
        success=True,
    )  

# ========= Использование ============
# from roc_mcs.fitting.optimize import fit_curve, FitConfig

# result = fit_curve(
#     theta_shift,
#     I_exp_prod_us,
#     FitConfig(model_name="gauss_us", seed=42),
# )





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
):
    """Batch LM/GN с независимыми microbatch-циклами."""
    if spec.jax_func is None:
        raise ValueError(f"JAX backend is not available for model {spec.name!r}")

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
    q = q_model + 1
    eye = jnp.eye(q, dtype=jnp.float64)
    model_func = spec.jax_func

    def component_model(params, theta):
        return model_func(theta, *params)

    model_components = jax.vmap(component_model, in_axes=(0, None), out_axes=0)

    def model_single(params, theta):
        comp = params[:q_model].reshape(K, n_params)
        return jnp.sum(model_components(comp, theta), axis=0) + params[-1]

    def model_with_aux_single(params, theta):
        pred = model_single(params, theta)
        return pred, pred

    model_jac_single = jax.jacfwd(model_with_aux_single, argnums=0, has_aux=True)
    model_jac_batch = jax.vmap(model_jac_single, in_axes=(0, None), out_axes=(0, 0))

    def residual_single(params, y, valid, theta):
        pred = model_single(params, theta)
        y_safe = jnp.where(valid, y, 0.0)
        return jnp.where(valid, pred - y_safe, 0.0)

    model_batch = jax.vmap(model_single, in_axes=(0, None), out_axes=0)
    residual_batch = jax.vmap(residual_single, in_axes=(0, 0, 0, None), out_axes=0)

    jac_single = jax.jacfwd(residual_single, argnums=0)
    jac_batch = jax.vmap(jac_single, in_axes=(0, 0, 0, None), out_axes=0)

    def solve_group(obs, valid, x0, lb, ub, theta):
        y_safe = jnp.where(valid, obs, 0.0)
        finite_lb = jnp.isfinite(lb)
        finite_ub = jnp.isfinite(ub)
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
            outward = (at_lb & (g > 0.0)) | (at_ub & (g < 0.0))
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
            free = ~((at_lb & (g > 0.0)) | (at_ub & (g < 0.0)))

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
            "success": reason != CONV_MAXITER,
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
        groups = []
        for start in range(0, B, microbatch_size):
            end = min(start + microbatch_size, B)
            groups.append(solve_group(
                obs[start:end], valid[start:end], x0[start:end],
                lb[start:end], ub[start:end], theta
            ))

        return {key: jnp.concatenate([group[key] for group in groups], axis=0) for key in groups[0]}

    return solve, model_batch, residual_batch, jac_batch




# --------- Production fitter ------------

JAX_SOLVER_CONFIG = {
    "max_iter": 200,
    "ftol": 1e-8,
    "xtol": 1e-8,
    "gtol": 1e-8,
}

def fit_jax(
    data_yxt,
    theta,
    peak_catalog,
    spec,
    *,
    B=128,
    solver_config=None,
    allow_center_shift=True,
    center_window=40.0,
    add_background=True,
    progress=False,
):
    """Выполняет JAX batch fitting всех пикселей с обнаруженными пиками."""
    if solver_config is None:
        solver_config = JAX_SOLVER_CONFIG
    if not allow_center_shift:
        raise NotImplementedError("JAX fitting currently requires allow_center_shift=True")

    prepared = prepare_fit_data(
        data_yxt=data_yxt,
        theta=theta,
        peak_catalog=peak_catalog,
        spec=spec,
        allow_center_shift=allow_center_shift,
        center_window=center_window,
        add_background=add_background,
    )

    solvers = {}
    theta_j = jnp.asarray(theta, dtype=jnp.float64)

    batches = iter_fit_batches(data_yxt=data_yxt, prepared=prepared, max_batch_pixels=B)
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
                )

            obs, valid, x0, lb, ub, n_actual = pad_batch_arrays(batch, B)

            expected_q = K * len(spec.param_names) + 1
            if x0.shape[1] != expected_q:
                raise ValueError(f"JAX solver expects {expected_q} parameters for {spec.name!r}, K={K}; got {x0.shape[1]}")

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
                "x_pixel": np.asarray(batch.x)[:n_actual],
                "y_pixel": np.asarray(batch.y)[:n_actual],
            })
            if pbar is not None:
                pbar.update(n_actual)

    finally:
        if pbar is not None:
            pbar.close()
    return results