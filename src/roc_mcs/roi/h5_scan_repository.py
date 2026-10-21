# from __future__ import annotations
import os,tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Dict,List,Tuple,Union
import h5py
import numpy as np

from roc_mcs.roi.extraction import extract_from_roi,extract_roi_volume
from roc_mcs.roi.roi_types import ROI
from roc_mcs.roi.spatial_pooling import (
    PoolSpec,
    normalize_pool,
    pool_centers,
    pool_native,
    pool_valid_yxt
)



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

BUILTIN_MAPS=("Mean projection","Max projection","Sum projection")

@dataclass(frozen=True,slots=True)
class ScanInfo:
    """Пути и тип scan внутри HDF5."""
    key:str
    kind:str
    raw_path:str
    th_path:str
    map_paths:Dict[str,str]

@dataclass(frozen=True,slots=True)
class ScanData:
    """Данные scan в одном строго определённом spatial-view."""
    key:str
    data:np.ndarray
    th:np.ndarray
    axis:int
    pool:PoolSpec
    source_shape_yx:tuple[int,int]
    maps:Dict[str,np.ndarray]

    @property
    def shape_yx(self)->tuple[int,int]: return self.data.shape[1:] if self.axis==0 else self.data.shape[:2]
    @property
    def raw(self)->np.ndarray: return self.data
    @property
    def yxt(self)->np.ndarray: return np.moveaxis(self.data,self.axis,-1) if self.axis!=2 else self.data
    @property
    def tyx(self)->np.ndarray: return np.moveaxis(self.data,self.axis,0) if self.axis!=0 else self.data
    @property
    def view(self)->str: return self.pool.tag
    def __getitem__(self,key:str)->Any:
        table={"data":self.data,"raw":self.data,"th":self.th,"axis":self.axis,"maps":self.maps,"pool":self.pool,"view":self.view,"shape_yx":self.shape_yx,"source_shape_yx":self.source_shape_yx,"spatial_pool":self.pool,"spatial_pool_signature":self.pool.signature}
        try: return table[key]
        except KeyError: raise KeyError(key) from None

class H5ScanRepository:
    """HDF5-источник с переключаемым raw/count-pool spatial-view."""

    def __init__(self,h5_path:Union[str,Path],*,pool:None|int|PoolSpec|dict=None,spatial_pool=None):
        self.h5_path=Path(h5_path).expanduser().resolve()
        if not self.h5_path.exists(): raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        if pool is not None and spatial_pool is not None: raise ValueError("задайте только pool; spatial_pool оставлен лишь для совместимости")
        self._pool=normalize_pool(pool if pool is not None else spatial_pool)
        self.scans=self._discover_scans()
        if not self.scans: raise ValueError("No usable scan groups were found in the HDF5 file.")
        self._source_cache:Dict[str,ScanData]={}
        self._view_cache:Dict[tuple[str,str],ScanData]={}
        self._map_cache:Dict[tuple[str,str,str],np.ndarray]={}

    @property
    def pool(self)->PoolSpec: return self._pool
    @property
    def view(self)->str: return self._pool.tag
    @property
    def view_signature(self)->str: return self._pool.signature

    def set_pool(self,pool:None|int|PoolSpec|dict=1,*,stride:int|None=None)->"H5ScanRepository":
        """Переключает spatial-view без повторного чтения HDF5."""
        self._pool=normalize_pool(pool,stride=stride)
        return self

    def _discover_scans(self)->Dict[str,ScanInfo]:
        """
        Ищет внутри HDF5 подходящие scan'ы.
        Различает entry*_CALC и entry*.
        Для каждого scan сохраняет пути к исходным данным,
        угловой оси θ и доступным 2D-картам:
            raw_data
            theta
            available maps
        """        
        scans:Dict[str,ScanInfo]={}
        with h5py.File(self.h5_path,"r") as f:
            for name,obj in f.items():
                if not isinstance(obj,h5py.Group): continue
                # Processed scans: entry###_CALC with raw_data_2D and raw_data_th.
                if name.endswith("_CALC") and "raw_data_2D" in obj and "raw_data_th" in obj: 
                    maps={k:f"{name}/{k}" for k,v in obj.items() if isinstance(v,h5py.Dataset) and v.ndim==2 and k not in {"raw_data_2D","raw_data_th","raw_data_total_int"}}
                    scans[name]=ScanInfo(name,"processed",f"{name}/raw_data_2D",f"{name}/raw_data_th",maps)
                    continue
                # Raw scans: entry###/measurement/Frame and th.
                meas=obj.get("measurement")
                if isinstance(meas,h5py.Group) and all(k in meas for k in ("Frame","th")):
                    scans[name]=ScanInfo(name,"raw",f"{name}/measurement/Frame",f"{name}/measurement/th",{})
        return scans

    def scan_keys(self)->List[str]: 
        """Возвращает идентификаторы найденных scan'ов."""
        return list(self.scans)

    def available_maps(self,scan_key:str)->List[str]:
        """Возвращает доступные 2D-карты для указанного scan."""
        self._require_scan(scan_key)
        return list(BUILTIN_MAPS) if self.pool.active else list(self.scans[scan_key].map_paths) or list(BUILTIN_MAPS)

    def _require_scan(self,scan_key:str)->None:
        if scan_key not in self.scans: raise KeyError(f"Unknown scan: {scan_key}")

    def _load_dataset(self,path:str)->np.ndarray:
        """Загружает указанный HDF5 dataset в NumPy-массив."""
        with h5py.File(self.h5_path,"r") as f: return np.asarray(f[path])

    @staticmethod
    def _theta_1d(th:np.ndarray)->np.ndarray:
        th=np.asarray(th).squeeze()
        if th.ndim!=1 or th.size<1: raise ValueError(f"theta должна быть 1D, shape={th.shape}")
        return th

    @staticmethod
    def _infer_angle_axis(raw:np.ndarray,th:np.ndarray)->int:
        """
        Определяет, где в 3D массиве находится ось theta.
        Например:
            (raw[theta, y, x]) или (raw[y, x, theta])
        """
        if raw.ndim!=3: raise ValueError(f"Expected 3D raw data, got shape {raw.shape}")
        candidates=[ax for ax in (0,2) if raw.shape[ax]==th.size]
        if len(candidates)==1: return candidates[0]
        if not candidates: raise ValueError(f"Не найдена theta-axis: raw={raw.shape}, theta={th.size}")
        raise ValueError(f"Theta-axis неоднозначна: raw={raw.shape}, theta={th.size}, candidates={candidates}")

    @staticmethod
    def _readonly(a:np.ndarray)->np.ndarray:
        a=np.asarray(a); a.setflags(write=False); return a

    def load_source(self,scan_key:str)->ScanData:
        """Исходный scan на detector-grid, всегда без pooling."""
        self._require_scan(scan_key)
        if scan_key in self._source_cache: return self._source_cache[scan_key]
        info=self.scans[scan_key]; raw=self._load_dataset(info.raw_path); th=self._theta_1d(self._load_dataset(info.th_path)); axis=self._infer_angle_axis(raw,th)
        raw=self._readonly(raw); th=self._readonly(th); shape_yx=raw.shape[1:] if axis==0 else raw.shape[:2]
        data=ScanData(scan_key,raw,th,axis,PoolSpec(),tuple(map(int,shape_yx)),{})
        self._source_cache[scan_key]=data; return data

    def load_scan_raw(self,scan_key:str)->ScanData: return self.load_source(scan_key)

    def load_scan(self,scan_key:str)->ScanData:
        """
        Scan в текущем spatial-view. Загружает данные scan и сохраняет их в кэше:
            raw
            th
            axis
            maps
        Возвращает raw, th, ось θ и кэш 2D-карт.        
        """
        src=self.load_source(scan_key); key=(scan_key,self.pool.signature)
        if key in self._view_cache: return self._view_cache[key]
        if not self.pool.active: return src
        raw=self._readonly(pool_native(src.data,src.axis,self.pool)); data=ScanData(scan_key,raw,src.th,src.axis,self.pool,src.source_shape_yx,{})
        self._view_cache[key]=data; return data

    def source_yxt(self,scan_key:str)->np.ndarray: return self.load_source(scan_key).yxt
    def source_tyx(self,scan_key:str)->np.ndarray: return self.load_source(scan_key).tyx
    def data_yxt(self,scan_key:str)->np.ndarray: return self.load_scan(scan_key).yxt
    def data_tyx(self,scan_key:str)->np.ndarray: return self.load_scan(scan_key).tyx

    def valid_yxt(self,scan_key:str,source_valid:np.ndarray|None=None)->np.ndarray:
        """Переводит source-grid valid-mask в Y×X×theta текущего view."""
        src=self.load_source(scan_key)
        if source_valid is None: valid=np.ones(src.data.shape,dtype=bool)
        else:
            valid=np.asarray(source_valid,dtype=bool)
            if valid.shape!=src.data.shape: raise ValueError(f"source_valid должен иметь source-shape {src.data.shape}, получено {valid.shape}")
        yxt=np.moveaxis(valid,src.axis,-1) if src.axis!=2 else valid
        out=pool_valid_yxt(yxt,self.pool); out.setflags(write=False); return out

    def valid_from_invalid(self,scan_key:str,source_invalid:np.ndarray)->np.ndarray:
        """Spike/invalid-mask source-grid → valid Y×X×theta текущего view."""
        invalid=np.asarray(source_invalid,dtype=bool); src=self.load_source(scan_key)
        if invalid.shape!=src.data.shape: raise ValueError(f"source_invalid должен иметь source-shape {src.data.shape}, получено {invalid.shape}")
        return self.valid_yxt(scan_key,~invalid)

    def centers_xy(self,scan_key:str)->tuple[np.ndarray,np.ndarray]:
        """Центры текущего view в координатах исходного detector-grid."""
        return pool_centers(self.load_source(scan_key).source_shape_yx,self.pool)

    def grid_info(self,scan_key:str)->dict[str,Any]:
        scan=self.load_scan(scan_key); x,y=self.centers_xy(scan_key)
        return {"view":self.view,"pool":self.pool,"shape_yx":scan.shape_yx,"source_shape_yx":scan.source_shape_yx,"x":x,"y":y}

    def view_key(self,scan_key:str)->tuple[str,str]:
        """Стабильный ключ scan + spatial-view для GUI/session-state."""
        self._require_scan(scan_key); return scan_key,self.view_signature

    def pool_source(self,scan_key:str,source_data:np.ndarray)->np.ndarray:
        """Применяет текущий pooling к массиву source-grid в native layout."""
        src=self.load_source(scan_key); a=np.asarray(source_data)
        if a.shape!=src.data.shape: raise ValueError(f"source_data должен иметь source-shape {src.data.shape}, получено {a.shape}")
        return pool_native(a,src.axis,self.pool)

    def get_map(self,scan_key:str,map_name:str)->np.ndarray:
        """2D-карта строго на grid текущего spatial-view."""
        self._require_scan(scan_key); key=(scan_key,self.pool.signature,map_name)
        if key in self._map_cache: return self._map_cache[key]
        info=self.scans[scan_key]
        if map_name in info.map_paths:
            if self.pool.active: raise ValueError(f"HDF5-карта {map_name!r} относится к raw-grid и недоступна в {self.view}")
            arr=self._load_dataset(info.map_paths[map_name]); expected=self.load_source(scan_key).source_shape_yx
            if arr.shape!=expected: raise ValueError(f"HDF5-карта {map_name!r}: shape={arr.shape}, ожидается source-grid {expected}")
            arr=self._readonly(arr); self._map_cache[key]=arr; self.load_source(scan_key).maps[map_name]=arr; return arr
        scan=self.load_scan(scan_key); raw=scan.data; axis=scan.axis
        if map_name=="Mean projection": arr=raw.mean(axis=axis)
        elif map_name=="Max projection": arr=raw.max(axis=axis)
        elif map_name=="Sum projection": arr=raw.sum(axis=axis)
        else: raise KeyError(f"Unknown map: {map_name}")
        arr=self._readonly(arr); self._map_cache[key]=arr; scan.maps[map_name]=arr; return arr

    # ===========================
    # Извлечение данных из ROI
    # ===========================
    def curve_from_roi(self,scan_key:str,roi:ROI,reduction_mode:str="mean",*,valid_mask=None)->Tuple[np.ndarray,np.ndarray,int]:
        """ROC из ROI; геометрия ROI в source-grid, valid_mask — в координатах текущего view."""
        scan=self.load_scan(scan_key); ny,nx=scan.shape_yx; x_coords,y_coords=self.centers_xy(scan_key)
        return extract_from_roi(raw=scan.data,th=scan.th,axis=scan.axis,roi=roi,nx=nx,ny=ny,reduction_mode=reduction_mode,x_coords=x_coords,y_coords=y_coords,valid_mask=valid_mask)

    def volume_from_roi(self,scan_key:str,roi:ROI,*,valid_mask=None,order=1):
        """Извлекает ROI-объём текущего view; геометрия ROI всегда source-grid."""
        scan=self.load_scan(scan_key); x_coords,y_coords=self.centers_xy(scan_key)
        return extract_roi_volume(scan.data,scan.th,roi,axis=scan.axis,valid_mask=valid_mask,order=order,x_coords=x_coords,y_coords=y_coords)

    def view_indices_to_source_xy(self,scan_key:str,x,y):
        """Преобразует индексы пикселей текущего view в координаты source-grid."""
        scan=self.load_scan(scan_key); x_axis,y_axis=self.centers_xy(scan_key)
        xb,yb=np.broadcast_arrays(np.asarray(x),np.asarray(y))
        if not np.all(np.isfinite(xb)) or not np.all(np.isfinite(yb)): raise ValueError("x/y должны быть конечными индексами")
        xi,yi=np.rint(xb).astype(int),np.rint(yb).astype(int)
        if np.any(xi!=xb) or np.any(yi!=yb): raise ValueError("x/y должны быть целочисленными индексами текущего view")
        if np.any((xi<0)|(xi>=len(x_axis))|(yi<0)|(yi>=len(y_axis))): raise IndexError("x/y выходят за границы текущего view")
        return x_axis[xi],y_axis[yi]

    def export_scan_copy_with_maps(self,scan_key:str,output_path:Union[str,Path],maps:Dict[str,np.ndarray],*,map_attrs:Dict[str,Dict[str,Any]]|None=None,overwrite:bool=False,compression:str|None="gzip",compression_opts:int|None=4)->Path:
        """Копирует scan и добавляет только source-grid 2D-карты."""
        self._require_scan(scan_key)
        if not maps: raise ValueError("maps не должен быть пустым")
        source_shape=self.load_source(scan_key).source_shape_yx
        for name,arr in maps.items():
            if not isinstance(name,str) or not name or "/" in name: raise ValueError(f"Некорректное имя карты: {name!r}")
            a=np.asarray(arr)
            if a.ndim!=2 or not np.issubdtype(a.dtype,np.number): raise ValueError(f"Карта {name!r} должна быть числовым 2D-массивом")
            if a.shape!=source_shape: raise ValueError(f"Карта {name!r}: shape={a.shape}; export принимает только source-grid {source_shape}")
        out=Path(output_path).expanduser().resolve()
        if out==self.h5_path: raise ValueError("output_path совпадает с исходным HDF5")
        if out.exists() and not overwrite: raise FileExistsError(f"Файл уже существует: {out}")
        out.parent.mkdir(parents=True,exist_ok=True); info=self.scans[scan_key]
        fd,tmp_name=tempfile.mkstemp(prefix=out.name+".tmp-",dir=out.parent); os.close(fd); tmp=Path(tmp_name)
        try:
            with h5py.File(self.h5_path,"r") as src,h5py.File(tmp,"w") as dst:
                src.copy(info.key,dst,name=info.key); group=dst[info.key]
                for name,arr in maps.items():
                    if name in group:
                        if not overwrite: raise ValueError(f"Карта уже существует: {scan_key}/{name}")
                        del group[name]
                    kw={} if compression is None else {"compression":compression,"compression_opts":compression_opts}
                    ds=group.create_dataset(name,data=arr,**kw); ds.attrs["source_scan"]=scan_key; ds.attrs["source_h5"]=str(self.h5_path); ds.attrs["grid"]="source"
                    if map_attrs and name in map_attrs:
                        for k,v in map_attrs[name].items():
                            if not isinstance(v,(str,int,float,bool,np.integer,np.floating,np.bool_)): raise TypeError(f"Атрибут {name}/{k} должен быть скаляром")
                            ds.attrs[str(k)]=v.item() if isinstance(v,np.generic) else v
                dst.flush()
            if out.exists(): out.unlink()
            tmp.replace(out)
        finally:
            if tmp.exists(): tmp.unlink()
        return out
