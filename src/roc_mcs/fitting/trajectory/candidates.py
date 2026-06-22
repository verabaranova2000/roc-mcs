import numpy as np
from tqdm import tqdm

from roc_mcs.fitting.registry import MODEL_SPECS
from roc_mcs.fitting.trajectory.types import CandidateCloud


def make_nd_grid_array(centers, param_keys, rel_spans=0.25, ns=11):
    """
    Генерация N-мерной сетки по заданным параметрам.

    Параметры
    ----------
    centers : dict
        Центры варьируемых параметров.

    param_keys : sequence[str]
        Порядок варьируемых параметров в выходной матрице.

    rel_spans : float или sequence[float]
        Относительный диапазон сетки по каждой размерности.

    ns : int или sequence[int]
        Число точек по каждой размерности.

    Возвращает
    ----------
    X : ndarray, shape (N, d)
        Матрица значений варьируемых параметров.
        Столбцы идут в порядке param_keys.
    """
    keys = tuple(param_keys)
    n_dim = len(keys)

    if np.isscalar(rel_spans):
        rel_spans = [rel_spans] * n_dim
    else:
        rel_spans = list(rel_spans)
        if len(rel_spans) != n_dim:
            raise ValueError("len(rel_spans) must match len(param_keys)")        
    if np.isscalar(ns):
        ns = [ns] * n_dim
    else:
        ns = list(ns)
        if len(ns) != n_dim:
            raise ValueError("len(ns) must match len(param_keys)")        

    axes = []
    for key, span, n in zip(keys, rel_spans, ns):
        c = centers[key]
        axes.append(np.linspace(c * (1 - span), c * (1 + span), n))

    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.column_stack([m.ravel() for m in mesh])
    return X



def score_candidates_batch(theta, intensity, batch_model_func, X_full):
    """
    Вычисление SSE для набора кандидатов параметров.

    Параметры
    ----------
    theta : ndarray, shape (M,)
        Ось углов.

    intensity : ndarray, shape (M,)
        Экспериментальная интенсивность.

    model_func : callable
        Векторизованная модель, принимающая параметры в фиксированном порядке.

    X_full : ndarray, shape (N, d)
        Матрица параметров в порядке, ожидаемом  batch_model_func.

    Возвращает
    ----------
    sse : ndarray, shape (N,)
        SSE для каждого кандидата.
    """
    y_fit = batch_model_func(theta, *X_full.T)
    residual = y_fit - intensity[None, :]
    return np.sum(residual**2, axis=1)    



def generate_time_candidates_nd(
    df_fit,
    roc_map,
    model_name,
    param_keys,
    rel_spans=0.25,
    ns=11,
    show_progress=True,
):
    spec = MODEL_SPECS[model_name]
    theta = roc_map["theta_axis"]
    intensity_map = roc_map["intensity"]
    
    full_keys = tuple(spec.param_names)
    vary_keys = tuple(param_keys)
    vary_idx = [full_keys.index(k) for k in vary_keys]

    missing = [k for k in vary_keys if k not in full_keys]
    if missing:
        raise KeyError(f"param_keys contain unknown parameters: {missing}")    
    
    all_time_candidates = []
    iterator = df_fit.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df_fit), desc=f"Candidates: {model_name}")    

    for i, row in iterator:
        result = row["_result"]
        I = intensity_map[i, :]

        base = {k: result.parameters[k].value for k in full_keys}
        centers = {k: base[k] for k in vary_keys}
        X_vary = make_nd_grid_array(centers, vary_keys, rel_spans=rel_spans, ns=ns)

        n = len(X_vary)
        X_full = np.empty((n, len(full_keys)), dtype=float)

        for j, k in enumerate(full_keys):
            X_full[:, j] = base[k]
        for col, idx in enumerate(vary_idx):
            X_full[:, idx] = X_vary[:, col]

        scores = score_candidates_batch(theta=theta, intensity=I, batch_func=spec.batch_func, X_full=X_full)
        all_time_candidates.append(CandidateCloud(X=X_full, scores=scores))

    return all_time_candidates
