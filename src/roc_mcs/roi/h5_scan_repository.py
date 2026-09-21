import h5py
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from roc_mcs.roi.extraction import extract_from_roi
from roc_mcs.roi.roi_types import ROI


"""
H5ScanRepository
│
├── Работа с измерением/HDF5
│   ├── поиск scan'ов
│   ├── загрузка и кэширование
│   ├── определение оси θ
│   └── получение 2D-проекций
│
└── Работа с ROI
    └── извлечение ROC-данных для point / line / rect ROI

    
Работа с HDF5-измерением

       HDF5-файл
           │
           ▼
   H5ScanRepository
           │
 ┌─────────┼──────────┐
 ▼         ▼          ▼
scan   raw + th   2D-карты
           │
           ▼
    extract_from_roi()
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
  point   line   rect
    └──────┼──────┘
           ▼
     ROC-данные ROI    
"""


@dataclass
class ScanInfo:
    """Метаданные scan и пути к связанным HDF5-данным."""
    key: str
    kind: str
    raw_path: str
    th_path: str
    map_paths: Dict[str, str]



class H5ScanRepository:
    """Поиск, загрузка и кэширование scan-данных из HDF5."""

    def __init__(self, h5_path: Union[str, Path]):
        """Создаёт repository для указанного HDF5-файла."""
        self.h5_path = Path(h5_path).expanduser().resolve()
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        self.scans: Dict[str, ScanInfo] = self._discover_scans()
        if not self.scans:
            raise ValueError("No usable scan groups were found in the HDF5 file.")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _discover_scans(self) -> Dict[str, ScanInfo]:
        """
        Ищет внутри HDF5 подходящие scan'ы.
        Различает entry*_CALC и entry*.
        Для каждого scan сохраняет пути к исходным данным,
        угловой оси θ и доступным 2D-картам:
            raw_data
            theta
            available maps
        """
        scans: Dict[str, ScanInfo] = {}
        with h5py.File(self.h5_path, "r") as f:
            for name, obj in f.items():
                if not isinstance(obj, h5py.Group):
                    continue

                # Processed scans: entry###_CALC with raw_data_2D and raw_data_th.
                if name.endswith("_CALC") and "raw_data_2D" in obj and "raw_data_th" in obj:
                    map_paths: Dict[str, str] = {}
                    for ds_name, ds_obj in obj.items():
                        if not isinstance(ds_obj, h5py.Dataset):
                            continue
                        if ds_name in {"raw_data_2D", "raw_data_th", "raw_data_total_int"}:
                            continue
                        map_paths[ds_name] = f"{name}/{ds_name}"
                    scans[name] = ScanInfo(
                        key=name,
                        kind="processed",
                        raw_path=f"{name}/raw_data_2D",
                        th_path=f"{name}/raw_data_th",
                        map_paths=map_paths,
                    )
                    continue

                # Raw scans: entry###/measurement/Frame and th.
                if "measurement" in obj and isinstance(obj["measurement"], h5py.Group):
                    meas = obj["measurement"]
                    if all(k in meas for k in ("Frame", "th")):
                        scans[name] = ScanInfo(
                            key=name,
                            kind="raw",
                            raw_path=f"{name}/measurement/Frame",
                            th_path=f"{name}/measurement/th",
                            map_paths={},
                        )
        return scans

    def scan_keys(self) -> List[str]:
        """Возвращает идентификаторы найденных scan'ов."""
        return list(self.scans.keys())

    def available_maps(self, scan_key: str) -> List[str]:
        """Возвращает доступные 2D-карты для указанного scan."""
        info = self.scans[scan_key]
        if info.map_paths:
            return list(info.map_paths.keys())
        return ["Mean projection", "Max projection", "Sum projection"]

    def _load_dataset(self, dataset_path: str) -> np.ndarray:
        """Загружает указанный HDF5 dataset в NumPy-массив."""
        with h5py.File(self.h5_path, "r") as f:
            return np.asarray(f[dataset_path])

    def load_scan(self, scan_key: str) -> Dict[str, Any]:
        """
        Загружает данные scan и сохраняет их в кэше:
            raw
            th
            axis
            maps
        Возвращает raw, th, ось θ и кэш 2D-карт.
        """
        if scan_key in self._cache:
            return self._cache[scan_key]

        info = self.scans[scan_key]
        raw = self._load_dataset(info.raw_path)
        th = self._load_dataset(info.th_path)
        axis = self._infer_angle_axis(raw, th)

        cache = {
            "raw": raw,
            "th": th,
            "axis": axis,
            "maps": {},
        }
        self._cache[scan_key] = cache
        return cache

    @staticmethod
    def _infer_angle_axis(raw: np.ndarray, th: np.ndarray) -> int:
        """
        Определяет, где в 3D массиве находится ось theta.
        Например:
            (raw[theta, y, x]) или (raw[y, x, theta])
        """
        if raw.ndim != 3:
            raise ValueError(f"Expected 3D raw data, got shape {raw.shape}")
        th_len = len(th)
        if raw.shape[0] == th_len:
            return 0
        if raw.shape[-1] == th_len:
            return 2
        return 0

    def get_map(self, scan_key: str, map_name: str) -> np.ndarray:
        """
        Возвращает 2D-карту для указанного scan.
        Поддерживаются карты: 
            Mean projection,
            Max projection,
            Sum projection,
            а также сохранённые в HDF5 карты
        """
        info = self.scans[scan_key]
        cache = self.load_scan(scan_key)
        maps = cache["maps"]
        if map_name in maps:
            return maps[map_name]

        if map_name in info.map_paths:
            arr = self._load_dataset(info.map_paths[map_name])
            maps[map_name] = arr
            return arr

        raw = cache["raw"]
        axis = cache["axis"]
        if map_name == "Mean projection":
            arr = raw.mean(axis=axis)
        elif map_name == "Max projection":
            arr = raw.max(axis=axis)
        elif map_name == "Sum projection":
            arr = raw.sum(axis=axis)
        else:
            arr = raw.max(axis=axis)
        maps[map_name] = arr
        return arr

    # ===========================
    # Извлечение данных из ROI
    # ===========================
    def curve_from_roi(
        self,
        scan_key: str,
        roi: ROI,
        reduction_mode: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Извлекает ROC-данные из ROI выбранного скана.

        Parameters
        ----------
        scan_key : str
            Идентификатор скана.
        roi : ROI
            Описание выбранной области интереса.
        reduction_mode : {"mean", "sum"}, optional
            Способ агрегации кривых ROI.

        Returns
        -------
        th : ndarray
            Угловая ось.
        data : ndarray
            Агрегированная ROC-кривая.
        count : int
            Число пикселей (или отсчётов) в ROI.
        """
        cache = self.load_scan(scan_key)
        raw = cache["raw"]
        th = cache["th"]
        axis = cache["axis"]

        if raw.ndim != 3:
            raise ValueError(f"Expected 3D raw data, got {raw.shape}")

        if axis == 0:
            ny, nx = raw.shape[1], raw.shape[2]
        else:
            ny, nx = raw.shape[0], raw.shape[1]

        return extract_from_roi(
            raw=raw,
            th=th,
            axis=axis,
            roi=roi,
            nx=nx,
            ny=ny,
            reduction_mode=reduction_mode,
        )