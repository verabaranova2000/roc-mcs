# sardana_h5.py

import h5py
import numpy as np
from pathlib import Path


def load_motor_scan(file, entry_id):
    """
    Загружает одномерный motor scan Sardana из HDF5.

    Parameters
    ----------
    file : str or Path
        Путь к TPC_data.h5.
    entry_id : int
        Номер entry со сканом (например 321).

    Returns
    -------
    dict
        {
            "theta": np.ndarray,
            "intensity": np.ndarray
        }
    """
    file = Path(file)

    with h5py.File(file, "r") as f:
        g = f[f"entry{entry_id}"]["measurement"]
        return {
            "theta": g["th"][:],
            "intensity": g["ct0"][:],
        }
    



def export_motor_scans(
    entry_ids,
    h5_file,
    out_folder,
):
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