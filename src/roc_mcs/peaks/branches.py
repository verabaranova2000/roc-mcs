import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from roc_mcs.peaks.catalog import build_pixel_peak_lookup


def match_neighbor_peaks(
    peaks_a,
    peaks_b,
    theta_gate=60.0,
):
    """
    One-to-one matching
    Однозначно сопоставляет пики двух соседних пикселей
    по минимальному угловому расстоянию |Δθ|.

    Parameters
    ----------
    peaks_a, peaks_b : list of dict
        Пики двух соседних пикселей. Каждый словарь содержит:
        ``peak_id`` и ``theta``; дополнительно могут присутствовать
        ``prominence`` и ``height``.

    theta_gate : float, default=60.0
        Максимально допустимое угловое расстояние |Δθ| между
        сопоставляемыми пиками, в угловых секундах.

    Returns
    -------
    matches : list of tuple
        Принятые пары ``(peak_id_a, peak_id_b, Δθ)``.

    unmatched_a : list of dict
        Пики из ``peaks_a``, которым не найдено допустимое соответствие.

    unmatched_b : list of dict
        Пики из ``peaks_b``, которым не найдено допустимое соответствие.
    """

    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return [], peaks_a, peaks_b

    theta_a = np.asarray([p["theta"] for p in peaks_a], dtype=float)
    theta_b = np.asarray([p["theta"] for p in peaks_b], dtype=float,)

    # Матрица стоимостей |theta_a - theta_b|
    cost = np.abs(theta_a[:, None] - theta_b[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_a = set()
    matched_b = set()
    for i, j in zip(row_ind, col_ind):
        delta = float(cost[i, j])
        if delta <= theta_gate:
            matches.append(
                (
                    peaks_a[i]["peak_id"],
                    peaks_b[j]["peak_id"],
                    delta,
                )
            )
            matched_a.add(i)
            matched_b.add(j)
    unmatched_a = [peaks_a[i] for i in range(len(peaks_a)) if i not in matched_a]
    unmatched_b = [peaks_b[j] for j in range(len(peaks_b)) if j not in matched_b]
    return (
        matches,
        unmatched_a,
        unmatched_b,
    )




def build_peak_branch_graph(peak_catalog, theta_gate=60.0):
    """
    Строит граф peak branches (пространственно связанных локальных пиков ROC-кривых).
    Для каждой пары 4-связных соседних пикселей
    `(x, y) → (x+1, y)` и `(x, y) → (x, y+1)`
    пики сопоставляются по принципу one-to-one с минимальным |Δθ|.
    Принимаются только соответствия, удовлетворяющие условию
        |Δθ| ≤ theta_gate.
    
    Узел:
        один локальный ROC peak.

    Ребро:
        соответствие peaks между соседними pixels,
        найденное one-to-one assignment.

    Соседство:
        4-connected.

    Parameters
    ----------
    peak_catalog : pandas.DataFrame
        Каталог локальных пиков с колонками:
        peak_id, x, y, theta, prominence и height.

    theta_gate : float, default=60.0
        Максимальное допустимое угловое различие
        между соответствующими пиками соседних пикселей,
        в угловых секундах.

    Returns
    -------
    edges : list of tuple
        Рёбра графа в виде (peak_id_a, peak_id_b).

    edge_deltas : numpy.ndarray
        Значения |Δθ| для принятых рёбер, в угловых секундах.
    """
    pixel_peaks = build_pixel_peak_lookup(peak_catalog)
    edges = []
    edge_deltas = []

    # ---------------------------------------------------------
    # Идём по всем существующим pixels
    # ---------------------------------------------------------
    for (x, y), peaks_here in pixel_peaks.items():
        # Только right и down:
        # каждая пара pixels рассматривается ровно один раз.
        neighbours = [(x + 1, y), (x, y + 1)]
        for neighbour in neighbours:
            if neighbour not in pixel_peaks:
                continue
            peaks_next = pixel_peaks[neighbour]
            matches, _, _ = match_neighbor_peaks(
                peaks_here,
                peaks_next,
                theta_gate=theta_gate,
            )
            for peak_id_a, peak_id_b, delta_theta in matches:
                edges.append((peak_id_a, peak_id_b))
                edge_deltas.append(delta_theta)
    return (
        edges,
        np.asarray(edge_deltas),
    )





def label_peak_branches(peak_catalog, edges):
    """
    Выделяет peak branches как связные компоненты
    графа соответствий локальных ROC-пиков.
    
    Присваивает каждому пику branch_id по connected components графа соответствий.
    
    Parameters
    ----------
    peak_catalog : pandas.DataFrame
        Каталог всех detected peaks с уникальным peak_id.

    edges : list of tuple
        Рёбра графа соответствий в формате (peak_id_a, peak_id_b).

    Returns
    -------
    n_branches : int
        Число найденных связных peak branches.

    branch_labels : numpy.ndarray
        Метка branch для каждой строки peak_catalog.
        Индекс массива соответствует порядку строк peak_catalog.
    """
    peak_ids = peak_catalog["peak_id"].to_numpy(dtype=int)
    peak_to_index = {peak_id: i for i, peak_id in enumerate(peak_ids)}
    rows = []
    cols = []
    for peak_a, peak_b in edges:
        i = peak_to_index[peak_a]
        j = peak_to_index[peak_b]
        rows.extend([i, j])
        cols.extend([j, i])

    if len(rows) == 0:
        branch_labels = np.arange(len(peak_catalog))
        return 0, branch_labels

    graph = coo_matrix(
        (
            np.ones(len(rows), dtype=np.uint8),
            (rows, cols),
        ),
        shape=(len(peak_catalog), len(peak_catalog))).tocsr()
    n_branches, branch_labels = connected_components(graph, directed=False)
    return n_branches, branch_labels
