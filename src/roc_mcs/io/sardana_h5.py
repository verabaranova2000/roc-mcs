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