import numpy as np

def make_nd_grid_array(centers, rel_spans=0.25, ns=11):
    """
    Генерация N-мерной сетки параметров (candidate space).

    Используется для построения набора гипотез вокруг локального минимума
    параметров модели на каждом временном шаге.

    Параметры
    ----------
    centers : dict
        Центры параметров {"H": ..., "eta": ..., ...}.

    rel_spans : float или list[float]
        Относительный диапазон отклонений от центра.

    ns : int или list[int]
        Число точек сетки по каждой размерности.

    Возвращает
    ----------
    X : ndarray, shape (N, d)
        Матрица кандидатов параметров.
        Каждая строка соответствует одному набору параметров.
    """
    keys = list(centers.keys())
    n_dim = len(keys)

    if np.isscalar(rel_spans):
        rel_spans = [rel_spans] * n_dim
    if np.isscalar(ns):
        ns = [ns] * n_dim

    axes = []
    for key, span, n in zip(keys, rel_spans, ns):
        c = centers[key]
        axis = np.linspace(c * (1 - span), c * (1 + span), n)
        axes.append(axis)
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.column_stack([m.ravel() for m in mesh])
    return X




