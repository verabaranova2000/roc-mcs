# optimize.py
from dataclasses import dataclass
from scipy.optimize import differential_evolution, curve_fit
import numpy as np

from roc_mcs.fitting.params import build_parameters
from roc_mcs.fitting.metrics import compute_fit_metrics
from roc_mcs.fitting.results import FitResult
from roc_mcs.fitting.registry import MODEL_SPECS



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
