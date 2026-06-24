import numpy as np
from roc_mcs.fitting.trajectory.types import ScalarTrajectory
from roc_mcs.fitting.registry import MODEL_SPECS


def build_scalar_trajectory(
    name,
    param_order,
    df_fit,
    ridge_observations,
    smooth_states,
    fit_results,
    ridge_covariances,
    smooth_covariances,
    unit=None,
    kind="parameter",
):
    idx = param_order.index(name)

    return ScalarTrajectory(
        name=name,
        idx=idx,
        kind=kind,
        unit=unit,
        local=df_fit[name].to_numpy(),
        ridge=ridge_observations[:, idx],
        smooth=smooth_states[:, idx],
        sigma_fit=np.array([np.sqrt(max(r.covariance[idx, idx], 0.0)) for r in fit_results]),
        sigma_ridge=np.array([np.sqrt(max(R[idx, idx], 0.0)) for R in ridge_covariances]),
        sigma_smooth=np.array([np.sqrt(max(P[idx, idx], 0.0)) for P in smooth_covariances]),
    )



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
        p_ridge = _params_vector_from_state(ridge_observations[i])
        p_smooth = _params_vector_from_state(smooth_states[i])

        cov_fit = np.asarray(res.covariance, dtype=float)
        cov_ridge = np.asarray(ridge_covariances[i], dtype=float)
        cov_smooth = np.asarray(smooth_covariances[i], dtype=float)

        # local fit
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

        # ridge
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

        # smooth
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
        unit="same as theta",
        local=local,
        ridge=ridge,
        smooth=smooth,
        sigma_fit=sigma_fit,
        sigma_ridge=sigma_ridge,
        sigma_smooth=sigma_smooth,
    )