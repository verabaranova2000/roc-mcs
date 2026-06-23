from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


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




@dataclass(slots=True)
class TrajectoryResult:
    """
    Результаты ridge/Kalman-постобработки траекторий параметров.
    """
    # итоговые данные
    df_kf: pd.DataFrame

    # промежуточные результаты
    trajectory: RidgeTrajectory
    kalman: KalmanResult

    # параметры модели случайного блуждания
    Q: np.ndarray


    # диагностика
    roughness_before: dict[str, float]
    roughness_after: dict[str, float]

    # конфигурация анализа
    model_name: str
    param_keys: tuple[str, ...]

    rel_spans: float | Sequence[float]
    ns: int | Sequence[int]
    rel_keep: float
    max_keep: int | None
    q_alpha: float    