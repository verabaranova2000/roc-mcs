from pathlib import Path
import h5py
import numpy as np


def resolve_output_folder(output_folder=None):
    return Path.cwd() if output_folder is None else Path(output_folder)

def ensure_folder(path):
    path.mkdir(parents=True, exist_ok=True)
    return path    



def show_tree(obj, file_out=None, indent=0):
    """
    Рекурсивно выводит дерево HDF5.

    Parameters
    ----------
    obj : h5py.Group
        Группа или файл HDF5.
    file_out : file-like object | None, optional
        Поток вывода. Если None, выводится в консоль.
    indent : int, optional
        Текущий уровень вложенности.

    Examples
    --------
    В консоль:

        with h5py.File(path, "r") as f:
            show_tree(f)

    В текстовый файл:

        with h5py.File(path, "r") as f:
            with open("tree.txt", "w", encoding="utf-8") as out:
                show_tree(f, file_out=out)
    """
    for key in obj:
        line = "    " * indent + key
        print(line, file=file_out)

        if isinstance(obj[key], h5py.Group):
            show_tree(
                obj[key],
                file_out=file_out,
                indent=indent + 1
            )


def to_yxt(data_3d, th):
    """
    Единый helper.
    Для исходного массива h5:
        raw = scan["raw"]       # (201, 256, 256)
        th = scan["th"]         # (201,)
        axis = scan["axis"]     # 0
    Возвращает
        data → Y X θ
    """
    data = np.asarray(data_3d, dtype=float)
    
    if len(th) not in data.shape:
        raise ValueError(f"len(th)={len(th)} не найден в data.shape={data.shape}")
    theta_axis = data.shape.index(len(th))
    
    if theta_axis != 2:
        data = np.moveaxis(data, theta_axis, -1)
        
    return data



def print_histogram_bins(
    data,
    bins=30,
    *,
    name=None,
    precision=6,
    compact=False,
):
    """
    Печатает численные значения столбиков гистограммы.

    Example
    ------
    print_histogram_bins(
        pool.n_control,
        bins=min(40, max(5, int(np.sqrt(len(pool.n_control))))),
        name="control points / pixel",
        compact=True,
    )
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    counts, edges = np.histogram(data, bins=bins)
    if name is not None:
        print(f"\n{name}")
    if compact:
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            print(f"{left:.{precision}f}–{right:.{precision}f} : {count}")
        return

    centers = 0.5 * (edges[:-1] + edges[1:])
    print(
        f"{'bin':>4} "
        f"{'x_left':>12} "
        f"{'x_right':>12} "
        f"{'x_center':>12} "
        f"{'count':>8}"
    )
    for i, (left, right, center, count) in enumerate(zip(edges[:-1], edges[1:], centers, counts)):
        print(
            f"{i:4d} "
            f"{left:12.{precision}f} "
            f"{right:12.{precision}f} "
            f"{center:12.{precision}f} "
            f"{count:8d}"
        )