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