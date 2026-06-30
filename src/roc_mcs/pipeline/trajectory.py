import numpy as np

from roc_mcs.fitting.trajectory.types import TrajectoryResult
from roc_mcs.fitting.trajectory.config import MODEL_TRAJECTORY
from roc_mcs.fitting.trajectory.candidates import generate_time_candidates_nd
from roc_mcs.fitting.registry import MODEL_SPECS
from roc_mcs.fitting.trajectory.derived import build_fwhm_trajectory
from roc_mcs.fitting.trajectory.metrics import  mean_roughness
from roc_mcs.fitting.trajectory.kalman import estimate_Q, kalman_smoother_random_walk
from roc_mcs.fitting.trajectory.ridge import extract_ridge_trajectory
from roc_mcs.fitting.trajectory.types import ScalarTrajectory

# Создать их сразу после локального фиттинга
def build_local_trajectories(df_fit, results, param_order):
    trajectories = {}
    for name in param_order:
        idx = param_order.index(name)
        trajectories[name] = ScalarTrajectory(
            name=name,
            idx=idx,
            local=df_fit[name].to_numpy(),
            sigma_fit=np.array([np.sqrt(max(r.covariance[idx, idx], 0.0)) for r in results]),
        )
    return trajectories


# Потом отдельная функция для обогащения ridge/Kalman
def enrich_trajectories_with_kalman(trajectories, traj_result):
    for name in traj_result.param_keys:
        tr = trajectories[name]
        idx = tr.idx
        tr.ridge = traj_result.trajectory.obs[:, idx]
        tr.smooth = traj_result.kalman.x_smooth[:, idx]
        tr.sigma_ridge = np.array([np.sqrt(max(R[idx, idx], 0.0))  for R in traj_result.trajectory.covs])
        tr.sigma_smooth = np.array([np.sqrt(max(P[idx, idx], 0.0)) for P in traj_result.kalman.P_smooth])
    for name, derived_traj in traj_result.derived.items():
        trajectories[name] = derived_traj
    return trajectories


def run_trajectory_analysis(
    roc_map,
    df_fit,
    results,
    model_name,
    param_keys=None,
    rel_spans=None,
    ns=None,
    rel_keep=None,
    max_keep=None,
    q_alpha=None,
):
    """
    Постобработка траекторий параметров методом ridge + Kalman.

    Parameters
    ----------
    roc_map : dict
        Карта rocking curve.

    df_fit : pandas.DataFrame
        Таблица результатов локального фитирования.

    results : list[FitResult]
        Результаты локального фитирования.

    model_name : str
        Имя модели.

    param_keys : sequence[str]
        Параметры, участвующие в ridge/Kalman-анализе.

    Returns
    -------
    dict
        Результаты ridge-анализа, сглаженные траектории и диагностические данные.
    """
    spec = MODEL_SPECS[model_name]
    traj_cfg = MODEL_TRAJECTORY.get(model_name, {})
    if param_keys is None:
        param_keys = traj_cfg.get("param_keys", spec.param_names)
    if rel_spans is None:
        rel_spans = traj_cfg.get("rel_spans", 0.25)
    if ns is None:
        ns = traj_cfg.get("ns", 11)
    if rel_keep is None:
        rel_keep = traj_cfg.get("rel_keep", 1.05)
    if q_alpha is None:
        q_alpha = traj_cfg.get("q_alpha", 0.01)

    
    # --- 1. Candidate space ---

    all_time_candidates = generate_time_candidates_nd(
        df_fit=df_fit.assign(_result=results),
        roc_map=roc_map,
        model_name=model_name,
        param_keys=param_keys,
        rel_spans=rel_spans,
        ns=ns,
    )

    # --- 2. Ridge extraction ---
    traj = extract_ridge_trajectory(
        all_time_candidates,
        param_keys=param_keys,
        rel_keep=rel_keep,
        max_keep=max_keep,
    )

    # --- 3. Process noise ---
    Q = estimate_Q(df_fit, param_keys=param_keys, alpha=q_alpha)

    # --- 4. Kalman smoothing ---
    kalman = kalman_smoother_random_walk(traj.obs, traj.covs, Q=Q)

    # --- 5. DataFrame ---
    df_kf = df_fit.copy()

    for i, key in enumerate(param_keys):
        df_kf[f"{key}_kf"] = kalman.x_smooth[:, i]

    # --- 6. Diagnostics ---
    roughness_before = {key: mean_roughness(df_fit[key]) for key in param_keys}
    roughness_after = {key: mean_roughness(df_kf[f"{key}_kf"]) for key in param_keys}

    fwhm_traj = build_fwhm_trajectory(
        model_name=model_name,
        theta=roc_map["theta_axis"],
        param_order=param_keys,
        fit_results=results,
        ridge_observations=traj.obs,
        smooth_states=kalman.x_smooth,
        ridge_covariances=traj.covs,
        smooth_covariances=kalman.P_smooth,
        n_mc=1000,
    )
    
    return TrajectoryResult(
        # --- core results ---        
        df_kf=df_kf,
        trajectory=traj,
        kalman=kalman,
        Q=Q,
        # --- diagnostics ---
        roughness_before=roughness_before,
        roughness_after=roughness_after,
        # --- metadata ---
        model_name=model_name,
        param_keys=tuple(param_keys),
        rel_spans=rel_spans,
        ns=ns,
        rel_keep=rel_keep,
        max_keep=max_keep,
        q_alpha=q_alpha,
        derived={"FWHM": fwhm_traj},
    )
    