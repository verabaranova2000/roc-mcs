import numpy as np
from roc_mcs.fitting.trajectory.types import ScalarTrajectory
from roc_mcs.fitting.params import PARAMETER_INFO
from roc_mcs.fitting.derived import estimate_fwhm_from_curve
from roc_mcs.fitting.registry import MODEL_SPECS, DERIVED_SPECS

"""
Структура: 

build_fwhm_trajectory
    ↓
_fwhm_value_and_sigma
    ↓
    analytic → delta method

или

    curve → Monte Carlo
"""


# ==================================================
# Helpers
# ==================================================
def _params_dict(params_vec, param_order):
    return {k: float(v) for k, v in zip(param_order, params_vec)}


def _params_vector_from_fit_result(result, param_order):
    return np.array([result.parameters[k].value for k in param_order], dtype=float)


def _curve_from_params(batch_func, theta, params_vec):
    """
    Batch-модель для одного набора параметров.
    """
    args = [np.asarray([p], dtype=float) for p in params_vec]
    y = batch_func(theta, *args)
    return np.asarray(y[0], dtype=float)



def _sigma_from_delta_method(cov, grad):
    """
    Delta method (закон распространения ошибок).
    Реализует классическую формулу распространения ошибок.
    """
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return np.nan
    cov = 0.5 * (cov + cov.T)
    var = float(grad @ cov @ grad)
    return float(np.sqrt(max(var, 0.0)))



def _sanitize_parameter_samples(samples, param_order):
    """
    Применяет физические ограничения к Monte-Carlo выборке.
    """
    samples = np.asarray(samples, dtype=float).copy()
    for j, name in enumerate(param_order):
        info = PARAMETER_INFO.get(name, {})
        lower = info.get("lower")
        upper = info.get("upper")
        if lower is not None:
            samples[:, j] = np.maximum(samples[:, j], lower)
        if upper is not None:
            samples[:, j] = np.minimum(samples[:, j], upper)
    return samples    

    
def _sigma_from_mc(theta, mean_vec, cov, param_order, batch_func, n_mc=2000, random_state=None):
    """
    Monte-Carlo propagation of uncertainty
    Оценка sigma(FWHM) методом Monte-Carlo по ковариации параметров.
    Используется для моделей без простой аналитической формулы.
    вместо формулы ∇fᵀΣ∇f мы буквально моделируем множество возможных 
    реализаций параметров и смотрим, как гуляет FWHM.
    """
    rng = np.random.default_rng(random_state)
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return np.nan
    cov = 0.5 * (cov + cov.T)
    # --- генерируем случайные параметры (генерируются тысячи наборов, похожих на параметры из фита, причём с учётом корреляций.).
    samples = rng.multivariate_normal(mean=np.asarray(mean_vec, dtype=float), cov=cov, size=n_mc)
    samples = _sanitize_parameter_samples(samples, param_order)
    # --- Для каждого набора параметров строится кривая. То есть получается 2000 разных профилей.
    y = batch_func(theta, *samples.T)
    fwhm_vals = np.array([estimate_fwhm_from_curve(theta, row) for row in y], dtype=float) # Для каждого профиля считается ширина
    return float(np.nanstd(fwhm_vals, ddof=1)) # Получаем распределение FWHM и берем σ_FWHM=std(FWHMi)


def _fwhm_value_and_sigma(
    model_name,
    theta,
    params_vec,
    cov,
    param_order,
    y_fit=None,
    batch_func=None,
    n_mc=2000,
    random_state=None,
):
    """
    Возвращает (FWHM, sigma_FWHM) для одного набора параметров.

    Для lorentz, gauss, voigt, pvoigt:
        value_func аналитическая;
        grad_func аналитическая;
        needs_curve=False.
    Тогда FWHM и ошибка считаются через delta method.

    Для split_voigt:
        value_func = estimate_fwhm_from_curve;
        grad_func = None;
        needs_curve=True.
    Тогда значение берётся по кривой; ошибка берётся по Monte Carlo.    
    """
    spec = DERIVED_SPECS[model_name]["FWHM"]
    params = _params_dict(params_vec, param_order)

    # --- 1. аналитическая формула ---
    if not spec.needs_curve:
        value = spec.value_func(params)
        grad = spec.grad_func(params, param_order)

        sigma = _sigma_from_delta_method(cov, grad)
        return float(value), float(sigma)

    # --- 2. кривая (needs_curve) -> FWHM по профилю ---
    if y_fit is None:
        if batch_func is None:
            raise ValueError(f"batch_func is required for curve-based FWHM: {model_name}")
        y_fit = _curve_from_params(batch_func, theta, params_vec)

    value = spec.value_func(theta, y_fit)

    if cov is None:
        sigma = np.nan
    else:
        sigma = _sigma_from_mc(
            theta=theta,
            mean_vec=params_vec,
            cov=cov,
            param_order=param_order,
            batch_func=batch_func,
            n_mc=n_mc,
            random_state=random_state,
        )

    return float(value), float(sigma)


# ==================================================
# FWHM trajectory
# ==================================================
def build_fwhm_trajectory(
    model_name,
    theta,
    param_order,
    fit_results,
    ridge_observations,
    smooth_states,
    ridge_covariances,
    smooth_covariances,
    # derived_name="FWHM",
    n_mc=2000,
    random_state=None,
):
    """
    Построение траектории FWHM для local fit, ridge и Kalman RTS.
    """
    spec = MODEL_SPECS[model_name]
    batch_func = spec.batch_func

    T = len(fit_results)

    local = np.empty(T, dtype=float)
    ridge = np.empty(T, dtype=float)
    smooth = np.empty(T, dtype=float)

    sigma_fit = np.empty(T, dtype=float)
    sigma_ridge = np.empty(T, dtype=float)
    sigma_smooth = np.empty(T, dtype=float)

    for i, res in enumerate(fit_results):
        p_fit = _params_vector_from_fit_result(res, param_order)
        p_ridge = np.asarray(ridge_observations[i], dtype=float)
        p_smooth = np.asarray(smooth_states[i], dtype=float)

        cov_fit = np.asarray(res.covariance, dtype=float)
        cov_ridge = np.asarray(ridge_covariances[i], dtype=float)
        cov_smooth = np.asarray(smooth_covariances[i], dtype=float)

        # --- local fit ---
        local[i], sigma_fit[i] = _fwhm_value_and_sigma(
            model_name=model_name,
            theta=theta,
            params_vec=p_fit,
            cov=cov_fit,
            param_order=param_order,
            y_fit=getattr(res, "y_fit", None),
            batch_func=batch_func,
            n_mc=n_mc,
            random_state=random_state,
        )

        # --- ridge ---
        ridge[i], sigma_ridge[i] = _fwhm_value_and_sigma(
            model_name=model_name,
            theta=theta,
            params_vec=p_ridge,
            cov=cov_ridge,
            param_order=param_order,
            batch_func=batch_func,
            n_mc=n_mc,
            random_state=random_state,
        )

        # --- smooth ---
        smooth[i], sigma_smooth[i] = _fwhm_value_and_sigma(
            model_name=model_name,
            theta=theta,
            params_vec=p_smooth,
            cov=cov_smooth,
            param_order=param_order,
            batch_func=batch_func,
            n_mc=n_mc,
            random_state=random_state,
        )

    return ScalarTrajectory(
        name="FWHM",
        idx=None,
        kind="derived",
        unit="arcsec",
        local=local,
        ridge=ridge,
        smooth=smooth,
        sigma_fit=sigma_fit,
        sigma_ridge=sigma_ridge,
        sigma_smooth=sigma_smooth,
    )