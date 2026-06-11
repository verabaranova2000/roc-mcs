# optimize.py
from dataclasses import dataclass
from scipy.optimize import differential_evolution, curve_fit
import numpy as np
from functools import partial


from roc_mcs.fitting.params import estimate_gauss_p0, build_parameters
from roc_mcs.fitting.models import US_MODELS
from roc_mcs.fitting.metrics import compute_fit_metrics
from roc_mcs.fitting.results import FitResult
from roc_mcs.fitting.specs import MODEL_SPECS



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
    # bounds = pack_bounds(spec.param_names, bounds_dict)

    if config.use_global:
        def loss_fn(p):
            residual = intensity - model(theta, *p)
            return np.sum(residual ** 2)

        de = differential_evolution(loss_fn, bounds=list(zip(bounds[0], bounds[1])), seed=config.seed)
        p_start = de.x
        y_fit_global = model(theta, *p_start)
    else:
        p_start = np.array(p0, dtype=float)
        y_fit_global = None

    if config.use_local:
        popt, pcov = curve_fit(model, theta, intensity, p0=p_start, bounds=bounds)
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





from functools import partial
from scipy.optimize import differential_evolution, curve_fit


model_name = "gauss"
model_us = US_MODELS[model_name]

# --- Физически разумные границы ---
#     S — площадь под исходной гауссианой;
#     theta0 — центр исходной гауссианы;
#     sigma — собственная ширина пика;
#     Delta — амплитуда ультразвуковой модуляции.
S0, theta00, sigma_eff = estimate_gauss_p0(theta_shift, I_exp_prod_us)  # сначала оценим моменты
PARAM_BOUNDS = {"S": (0.5*S0, 1.5*S0),
                "theta0": (theta00 - 5, theta00 + 5),
                "sigma": (0.2, sigma_eff),
                "Delta": (0.0, 20.0)}

# ==================================================
# global optimization (DE)
# ==================================================
estimate_gauss_p0(theta_shift, I_exp_prod_us)
loss_fn = partial(loss, theta=theta_shift, I_exp=I_exp_prod_us, model=model_us)
result_de = differential_evolution(loss_fn,  bounds=list(PARAM_BOUNDS.values()),
                                   seed=42)
popt_glob = result_de.x
I_fit_glob = model_us(theta_shift, *popt_glob)


# ==================================================
# local refinement (curve_fit)
# ==================================================
popt_loc, pcov_loc = curve_fit(model_us, theta_shift, I_exp_prod_us, p0=popt_glob)
perr_loc = np.sqrt(np.diag(pcov_loc))
I_fit_loc = model_us(theta_shift, *popt_loc)


# ==================================================
# unpack fitted parameters
# ==================================================
S_fit_glob, theta0_fit_glob, sigma_fit_glob, Delta_fit_glob = popt_glob
S_fit_loc, theta0_fit_loc, sigma_fit_loc, Delta_fit_loc = popt_loc