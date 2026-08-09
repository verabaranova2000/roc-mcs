# sardana_h5.py

import h5py
import numpy as np
from pathlib import Path

TPC_DATA_H5_PATH = Path(r"C:\Users\User\Desktop\For_Sardana\TPC_data.h5")

# ===========================
# Классификация entry
# ===========================

ENTRY_1D = "1d_entry"
ENTRY_2D = "2d_entry"
ENTRY_UNKNOWN = "unknown"


def classify_entry_measurement(measurement_group):
    """
    Определяет тип скана по содержимому группы measurement.

    Parameters
    ----------
    measurement_group : h5py.Group
        Группа measurement из HDF5-файла.

    Returns
    -------
    str
        ENTRY_1D      — entry с одномерными данными;
        ENTRY_2D      — entry с двумерными данными;
        ENTRY_UNKNOWN — тип entry определить не удалось.
    """
    keys = set(measurement_group.keys())
    # --- Признак обычного 1D-скана ---
    if {"th", "ct0"}.issubset(keys):
        th = measurement_group["th"]
        ct0 = measurement_group["ct0"]
        if getattr(th, "ndim", None) == 1 and getattr(ct0, "ndim", None) == 1:
            return ENTRY_1D
    # --- Признаки 2D-скана ---
    if "Frame" in keys or "total" in keys:
        return ENTRY_2D
    return ENTRY_UNKNOWN


def classify_entry(file_path, entry_id):
    """
    Определяет тип указанного entry в HDF5-файле.
    Обертка: занимается открытием файла и передачей данных в classify_entry_measurement(...).

    Parameters
    ----------
    file_path : str | Path
        Путь к HDF5-файлу.

    entry_id : int
        Номер entry.

    Returns
    -------
    str
        ENTRY_1D      — entry с одномерными данными;
        ENTRY_2D      — entry с двумерными данными;
        ENTRY_UNKNOWN — entry отсутствует или его тип определить не удалось.
    """
    with h5py.File(file_path, "r") as f:
        entry_name = f"entry{entry_id}"
        if entry_name not in f:
            return ENTRY_UNKNOWN

        entry = f[entry_name]
        if "measurement" not in entry:
            return ENTRY_UNKNOWN
        return classify_entry_measurement(entry["measurement"])


def load_1d_entry(file_path, entry_id):
    """
    Загружает одномерный скан из HDF5-файла.

    Parameters
    ----------
    file_path : str | Path
        Путь к HDF5-файлу.

    entry_id : int
        Номер entry.

    Returns
    -------
    dict
        {
            "theta": np.ndarray,
            "intensity": np.ndarray
        }

    Raises
    ------
    KeyError
        Если entry отсутствует в файле.

    ValueError
        Если entry не является 1D-сканом.
    """
    with h5py.File(file_path, "r") as f:
        entry_name = f"entry{entry_id}"
        if entry_name not in f:
            raise KeyError(f"Нет entry{entry_id} в файле")

        meas = f[entry_name]["measurement"]
        scan_kind = classify_entry_measurement(meas)
        if scan_kind != ENTRY_1D:
            raise ValueError(f"entry{entry_id} — не 1D моторный скан, а {scan_kind}")
        return {
            "theta": meas["th"][:],
            "intensity": meas["ct0"][:],
        }


def list_1d_entries(file_path):
    """
    Возвращает список entry с одномерными сканами.

    Parameters
    ----------
    file_path : str | Path
        Путь к HDF5-файлу.

    Returns
    -------
    list[int]
        Отсортированный список идентификаторов entry,
        содержащих 1D-сканы.
    """
    ids = []
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            if not key.startswith("entry"):
                continue

            entry_id = int(key[5:])
            try:
                meas = f[key]["measurement"]
            except KeyError:
                continue

            if classify_entry_measurement(meas) == ENTRY_1D:
                ids.append(entry_id)
    return sorted(ids)


# ===========================
# Два простых provider-а
# ===========================
class Dict1DEntryProvider:    
    """ Provider 1D entry, хранящихся в словаре. """
    def __init__(self, curves_dict):
        self.curves_dict = curves_dict
        self.file_path = None  # У словаря нет файла на диске

    def list_ids(self):
        """ Возвращает список доступных ID. """        
        return [int(k) for k in self.curves_dict.keys()]

    def load(self, entry_id):
        """ Загружает данные по указанному ID. """
        return self.curves_dict[str(entry_id)]


class H51DEntryProvider:
    """ Provider 1D entry, загружаемых из HDF5-файла. """
    def __init__(self, file_path):
        self.file_path = file_path

    def list_ids(self):
        """ Возвращает список доступных ID. """
        return list_1d_entries(self.file_path)

    def load(self, entry_id):
        """ Загружает данные по указанному ID. """
        return load_1d_entry(self.file_path, entry_id)
    


# ======================================
# Экспорт из лабораторного ТРС_data.h5
# ======================================
def export_entries_to_h5(
    src_h5_file,
    entry_ids,
    dst_h5_file,
    overwrite_file=False,
    overwrite_entries=False,
):
    """
    Копирует выбранные entry из исходного HDF5-файла в целевой HDF5-файл.

    Если целевой файл уже существует, он открывается в режиме append.
    Можно либо пропускать уже существующие entry, либо перезаписывать их.

    Parameters
    ----------
    src_h5_file : str | Path
        Исходный HDF5-файл.

    entry_ids : int | sequence[int]
        Один или несколько номеров entry для копирования.

    dst_h5_file : str | Path
        Путь к целевому HDF5-файлу.

    overwrite_file : bool
        Если True, полностью пересоздаёт целевой файл.

    overwrite_entries : bool
        Если True, удаляет уже существующий entry в целевом файле
        и записывает его заново.

    Returns
    -------
    Path
        Путь к целевому файлу.

    Examples
    -------
    Создать новый файл заново :
        export_entries_to_h5(src_h5_file, [321, 401], dst_h5_file, overwrite_file=True)
    
    Дописать в существующий файл :
        export_entries_to_h5(src_h5_file, [352, 353], dst_h5_file)
    
    Перезаписать уже существующие entry :
        export_entries_to_h5(
            src_h5_file,
            [321, 401],
            dst_h5_file,
            overwrite_entries=True,
        )
    """
    src_h5_file = Path(src_h5_file)
    dst_h5_file = Path(dst_h5_file)

    if isinstance(entry_ids, int):
        entry_ids = [entry_ids]

    mode = "w" if overwrite_file else "a"

    with h5py.File(src_h5_file, "r") as src, h5py.File(dst_h5_file, mode) as dst:
        for entry_id in entry_ids:
            entry_name = f"entry{entry_id}"
            if entry_name not in src:
                raise KeyError(f"{entry_name} не найден в исходном файле")

            if entry_name in dst:
                if overwrite_entries:
                    del dst[entry_name]
                else:
                    continue

            src.copy(src[entry_name], dst, name=entry_name)

    return dst_h5_file



# ==================================================================
# Возможно, больше не пригодится 
# ==================================================================
def export_motor_scans(entry_ids, h5_file, out_folder):
    """
    Экспортирует моторные сканы из h5 в отдельные txt-файлы.

    Parameters
    ----------
    entry_ids : int | sequence[int]
        Номер(а) entry.
    h5_file : str | Path
        Путь к TPC_data.h5.
    out_folder : str | Path
        Папка для сохранения файлов.
    """
    if np.isscalar(entry_ids):
        entry_ids = [entry_ids]

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_file, "r") as f:
        for entry_id in entry_ids:
            g = f[f"entry{entry_id}"]["measurement"]

            theta = g["th"][:]
            intensity = g["ct0"][:]
            data = np.column_stack([theta, intensity])
            outfile = out_folder / f"scan{entry_id}.txt"
            np.savetxt(
                outfile,
                data,
                header="theta\tintensity",
                fmt="%.8f",
                delimiter="\t",
            )
            print(outfile)


# Использование
# export_motor_scans(
#     entry_ids=[320, 321, 322],
#     h5_file=H5_FILE,
#     out_folder="motor_scans",
# )

# или
# export_motor_scans(
#     entry_ids=321,
#     h5_file=H5_FILE,
#     out_folder="motor_scans",
# )

# Получится
# motor_scans/
#     scan320.txt
#     scan321.txt
#     scan322.txt