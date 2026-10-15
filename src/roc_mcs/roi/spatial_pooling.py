# from __future__ import annotations
from dataclasses import asdict,dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True,slots=True)
class PoolSpec:
    """Пространственный count-sum pooling без смешивания theta."""
    size:int=1
    stride:int=1

    def __post_init__(self):
        if not isinstance(self.size,(int,np.integer)) or self.size<1 or self.size%2==0: raise ValueError("pool size должен быть положительным нечётным")
        if not isinstance(self.stride,(int,np.integer)) or self.stride<1: raise ValueError("pool stride должен быть >= 1")

    @property
    def active(self)->bool: return self.size!=1 or self.stride!=1
    @property
    def area(self)->int: return self.size*self.size
    @property
    def tag(self)->str: return "raw" if not self.active else f"pool{self.size}s{self.stride}"
    @property
    def signature(self)->str: return f"count-sum|size={self.size}|stride={self.stride}|boundary=valid"
    def to_dict(self)->dict[str,int]: return asdict(self)


def normalize_pool(pool:None|int|PoolSpec|dict[str,Any]=None,*,stride:int|None=None)->PoolSpec:
    """Нормализует пользовательское описание pooling."""
    if pool is None: spec=PoolSpec()
    elif isinstance(pool,PoolSpec): spec=pool
    elif isinstance(pool,(int,np.integer)): spec=PoolSpec(int(pool),1 if stride is None else int(stride))
    elif isinstance(pool,dict): spec=PoolSpec(**pool)
    else: raise TypeError("pool должен быть None, int, dict или PoolSpec")
    if stride is not None and isinstance(pool,PoolSpec) and spec.stride!=stride: raise ValueError("stride уже задан внутри PoolSpec")
    return spec


def pooled_shape(source_shape_yx:tuple[int,int],pool:None|int|PoolSpec=None)->tuple[int,int]:
    """Размер spatial-grid после valid pooling."""
    p=normalize_pool(pool); ny,nx=map(int,source_shape_yx)
    if p.size>ny or p.size>nx: raise ValueError(f"pool {p.size} больше spatial-grid {source_shape_yx}")
    return (1+(ny-p.size)//p.stride,1+(nx-p.size)//p.stride)


def pool_centers(source_shape_yx:tuple[int,int],pool:None|int|PoolSpec=None)->tuple[np.ndarray,np.ndarray]:
    """Координаты центров pooled-grid в индексах исходного detector."""
    p=normalize_pool(pool); py,px=pooled_shape(source_shape_yx,p); h=p.size//2
    y=h+np.arange(py,dtype=float)*p.stride; x=h+np.arange(px,dtype=float)*p.stride
    return x,y


def _sum_windows_yxt(a:np.ndarray,size:int)->np.ndarray:
    a=np.asarray(a)
    if a.ndim!=3: raise ValueError(f"ожидался Y×X×T массив, shape={a.shape}")
    dtype=np.int64 if np.issubdtype(a.dtype,np.integer) or a.dtype==np.bool_ else np.float64
    z=np.asarray(a,dtype=dtype); cs=np.pad(z,((1,0),(1,0),(0,0))).cumsum(0,dtype=dtype).cumsum(1,dtype=dtype)
    return cs[size:,size:]-cs[:-size,size:]-cs[size:,:-size]+cs[:-size,:-size]


def pool_yxt(data_yxt:np.ndarray,pool:None|int|PoolSpec=None)->np.ndarray:
    """Суммирует Y×X count-окна; theta остаётся независимой осью."""
    p=normalize_pool(pool); a=np.asarray(data_yxt)
    if a.ndim!=3: raise ValueError(f"data_yxt должен быть 3D, shape={a.shape}")
    pooled_shape(a.shape[:2],p)
    out=a.copy() if p.size==1 else _sum_windows_yxt(a,p.size)
    return out[::p.stride,::p.stride].copy()


def pool_valid_yxt(valid_yxt:np.ndarray,pool:None|int|PoolSpec=None)->np.ndarray:
    """Pooled sample валиден, только если валидно всё исходное окно."""
    p=normalize_pool(pool); valid=np.asarray(valid_yxt,dtype=bool)
    if valid.ndim!=3: raise ValueError(f"valid_yxt должен быть 3D, shape={valid.shape}")
    pooled_shape(valid.shape[:2],p)
    if p.size==1: out=valid.copy()
    else: out=_sum_windows_yxt(valid,p.size)==p.area
    return out[::p.stride,::p.stride].copy()


def pool_native(data:np.ndarray,theta_axis:int,pool:None|int|PoolSpec=None)->np.ndarray:
    """Pooling 3D-массива с сохранением исходного положения theta-axis."""
    a=np.asarray(data)
    if a.ndim!=3: raise ValueError(f"data должен быть 3D, shape={a.shape}")
    if theta_axis not in (0,2): raise ValueError("theta_axis должен быть 0 или 2")
    yxt=np.moveaxis(a,theta_axis,-1) if theta_axis!=2 else a
    pooled=pool_yxt(yxt,pool)
    return np.moveaxis(pooled,-1,theta_axis) if theta_axis!=2 else pooled
