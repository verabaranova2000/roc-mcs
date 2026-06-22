from dataclasses import dataclass
import numpy as np

@dataclass
class CandidateCloud:
    """
    Облако кандидатов в пространстве параметров модели
    и соответствующие значения целевой функции.
    """
    X: np.ndarray      # (N, d)
    scores: np.ndarray # (N,)


@dataclass
class RidgeTimeSlice:
    """
    Результаты ridge-анализа для одного временного шага.

    Содержит множество кандидатов, ridge-область,
    взвешенную оценку параметров и ковариацию наблюдения.
    """
    scores: np.ndarray
    best_idx: int
    best_score: float
    X_all: np.ndarray
    keep: np.ndarray
    X_keep: np.ndarray
    weights: np.ndarray
    x_best: np.ndarray
    x_hat: np.ndarray
    cov: np.ndarray


@dataclass
class RidgeTrajectory:
    """
    Траектория параметров, полученная методом ridge-анализа.

    Содержит оценки параметров и ковариационные матрицы для всех временных шагов.
    """
    param_keys: tuple[str, ...]
    obs: np.ndarray
    covs: list[np.ndarray]
    best_scores: np.ndarray
    slices: list[RidgeTimeSlice]   


@dataclass
class KalmanResult:
    """
    Результаты фильтрации и сглаживания в модели случайного блуждания.
    """
    x_smooth: np.ndarray
    P_smooth: list[np.ndarray]
    x_filt: np.ndarray
    P_filt: list[np.ndarray]
    x_pred: np.ndarray
    P_pred: list[np.ndarray]    