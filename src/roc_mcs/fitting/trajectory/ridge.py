from roc_mcs.fitting.trajectory.types import RidgeTimeSlice, RidgeTrajectory


import numpy as np

# --- 1) Ядро: ridge по одному времени ---
def ridge_from_candidates(cloud, rel_keep=1.05, max_keep=None) -> RidgeTimeSlice:
    """
    Извлечение ridge-точки для одного временного среза.

    Среди кандидатов выбирается ridge-область, включающая решения
    с значением SSE не выше rel_keep относительно минимума.
    Координаты ridge-точки определяются как взвешенный центр
    выбранной области, а ковариационная матрица используется
    как оценка неопределённости наблюдения.

    Параметры
    ----------
    cloud : CandidateCloud
        Облако кандидатов и соответствующих значений SSE.

    param_keys : sequence[str]
        Имена параметров модели.

    rel_keep : float
        Допустимое относительное увеличение SSE относительно минимума.

    max_keep : int or None
        Максимальное число кандидатов в ridge-области.

    Возвращает
    ----------
    RidgeTimeSlice
        Результат анализа одного временного среза.
    """
    scores = cloud.scores
    best_idx = int(np.argmin(scores))
    best_score = float(scores[best_idx])

    X_all = cloud.X
    
    # --- оставляем только кандидатов около минимума ---
    # --- a) оставить ВСЕ точки, у которых SSE не хуже чем на 5% относительно минимума. 
    #        То есть в ridge-область попало 990 кандидатов.
    keep = np.where(scores <= best_score * rel_keep)[0]

    if len(keep) == 0:
        raise RuntimeError("No candidates passed ridge selection.")
    # --- b) Если ridge-область слишком большая, она искусственно обрезается.
    #        тогда из 990 точек останутся только max_keep лучших по SSE.
    if max_keep is not None and len(keep) > max_keep:
        keep = keep[np.argsort(scores[keep])[:max_keep]]

    X_keep = X_all[keep]
    s = scores[keep]
    
    # --- веса: чем лучше score, тем выше вес ---
    tau = np.std(s) + 1e-12
    w = np.exp(-(s - s.min()) / tau)
    w = w / w.sum()

    x_hat = np.sum(w[:, None] * X_keep, axis=0)        # то же самое: x_hat = np.average(X_keep, axis=0, weights=w)
    diff = X_keep - x_hat
    cov = np.sum(w[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0,)

    return RidgeTimeSlice(
        scores=scores,
        best_idx=best_idx,
        best_score=best_score,
        X_all=X_all,
        keep=keep,
        X_keep=X_keep,
        weights=w,
        x_best=X_all[best_idx],
        x_hat=x_hat,
        cov=cov + 1e-9*np.eye(X_all.shape[1]),
    )



def extract_ridge_trajectory(all_time_candidates, param_keys=("H", "eta"),
                             rel_keep=1.05, max_keep=None) -> RidgeTrajectory:
    """
    Построение ridge-траектории по последовательности временных срезов.

    Для каждого момента времени вычисляется ridge-точка и её
    ковариационная матрица. Полученная последовательность наблюдений
    используется на этапе последующего сглаживания фильтром Калмана.

    Параметры
    ----------
    all_time_candidates : sequence[CandidateCloud]
        Наборы кандидатов для всех временных срезов.

    param_keys : sequence[str]
        Имена параметров модели.

    rel_keep : float
        Допустимое относительное увеличение SSE относительно минимума.

    max_keep : int or None
        Максимальное число кандидатов в ridge-области.

    Возвращает
    ----------
    RidgeTrajectory
        Ridge-траектория и связанные статистики.
    """  
    slices = []
    obs = []
    covs = []
    best_scores = []
    
    for cloud in all_time_candidates:
        slice_ = ridge_from_candidates(cloud, rel_keep=rel_keep, max_keep=max_keep)
        slices.append(slice_)
    
        obs.append(slice_.x_hat)
        covs.append(slice_.cov)
        best_scores.append(slice_.best_score)
    
    return RidgeTrajectory(
        param_keys=param_keys,
        obs=np.array(obs),
        covs=covs,
        best_scores=np.array(best_scores),
        slices=slices
    )  