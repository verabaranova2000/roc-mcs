import numpy as np
from typing import Sequence

from roc_mcs.fitting.trajectory.types import KalmanResult

def kalman_smoother_random_walk(
    zs: np.ndarray,
    Rs: Sequence[np.ndarray],
    Q: np.ndarray,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> KalmanResult:
    """
    Random-walk state-space model:
        x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
        z_t = x_t + v_t,       v_t ~ N(0, R_t)

    Parameters
    ----------
    zs : np.ndarray, shape (T, d)
        Observations
    Rs : list[np.ndarray]
        Observation covariance for each t
    Q : np.ndarray, shape (d, d)
        Process noise covariance
    x0 : np.ndarray, shape (d,), optional
    P0 : np.ndarray, shape (d, d), optional

    Returns
    -------
    x_smooth : np.ndarray, shape (T, d)
    P_smooth : list[np.ndarray]
    """
    zs = np.asarray(zs, dtype=float)
    T, d = zs.shape

    if x0 is None:
        x0 = zs[0].copy()
    if P0 is None:
        P0 = np.eye(d) * 1.0

    # Forward pass
    x_filt = np.zeros((T, d))
    P_filt = [None] * T
    x_pred = np.zeros((T, d))
    P_pred = [None] * T

    x_prev = x0
    P_prev = P0

    I = np.eye(d)

    for t in range(T):
        if t == 0:
            x_pred[t] = x_prev
            P_pred[t] = P_prev
        else:
            x_pred[t] = x_prev
            P_pred[t] = P_prev + Q

        S = P_pred[t] + Rs[t]
        K = P_pred[t] @ np.linalg.solve(S, P_pred[t].T).T       # вместо np.linalg.inv(S)

        y = zs[t] - x_pred[t]
        x_post = x_pred[t] + K @ y
        P_post = (I - K) @ P_pred[t]

        x_filt[t] = x_post
        P_filt[t] = P_post

        x_prev = x_post
        P_prev = P_post

    # RTS smoothing
    x_smooth = x_filt.copy()
    P_smooth = [p.copy() for p in P_filt]

    for t in range(T - 2, -1, -1):
        C = P_filt[t] @ np.linalg.inv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred[t + 1]) @ C.T

    return KalmanResult(
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        x_filt=x_filt,
        P_filt=P_filt,
        x_pred=x_pred,
        P_pred=P_pred,
    )