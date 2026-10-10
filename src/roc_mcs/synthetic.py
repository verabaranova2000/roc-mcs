from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from roc_mcs.fitting.registry import ModelSpec

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticScan:
    frame: FloatArray
    theta: FloatArray
    peak_catalog: pd.DataFrame
    truth: dict[tuple[int, int], dict]
    component_catalog: pd.DataFrame | None = None
    split_weight: FloatArray | None = None
    split_separation: FloatArray | None = None


def make_synthetic_scan(
    spec: ModelSpec,
    theta: ArrayLike,
    Ny: int = 8,
    Nx: int = 8,
    max_components: int = 3,
    noise_sigma: float = 3.0,
    seed: int = 42,
) -> SyntheticScan:
    """
    Генерирует синтетический 2D-скан спектров с добавлением шума и эталонной разметкой.

    Parameters
    ----------
    spec : object
        Объект модели пика, содержащий метод `func(theta, **params)`.
    theta : array_like
        1D-массив или последовательность значений сетки углов theta.
    Ny : int, default 8
        Количество пикселей по пространственной оси Y.
    Nx : int, default 8
        Количество пикселей по пространственной оси X.
    max_components : int, default 3
        Максимальное случайное число пиковых компонент в одном пикселе.
    noise_sigma : float, default 3.0
        Стандартное отклонение аддитивного гауссова шума.
    seed : int, default 42
        Зерно генератора случайных чисел NumPy для воспроизводимости.

    Returns
    -------
    SyntheticScan
        Синтетический скан с полями:
        frame : np.ndarray
            Зашумленный 3D-скан формы (Ntheta, Ny, Nx).
        theta : np.ndarray
            1D-массив углов theta формы (Ntheta,).
        peak_catalog : pd.DataFrame
            Каталог сгенерированных пиков и их параметрами.
        truth : dict
            Ground truth по каждому пикселю: число пиков, фон,
            параметры компонент и чистый спектр.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, float)
    ntheta = len(theta)

    frame_yxt = np.zeros((Ny, Nx, ntheta), float)
    peak_rows, truth = [], {}
    
    global_peak_id = 0        # счетчик

    for y in range(Ny):
        for x in range(Nx):
            n = rng.integers(0, max_components + 1)
            bg = float(rng.uniform(2, 8))
            clean = np.full(ntheta, bg, float)
            components = []
            if n:
                idx = np.sort(rng.choice(np.arange(3, ntheta - 3), n, replace=False))
                for local_k, theta_index in enumerate(idx):     # используем local_k (или component_id) для локального порядка
                    params = {
                        "S": float(rng.uniform(800, 1800)),
                        "theta0": float(theta[theta_index]),
                        "H": float(rng.uniform(6, 12)),
                        "eta": float(rng.uniform(0.2, 0.8)),
                    }
                    component = np.asarray(spec.func(theta, **params), float)
                    clean += component

                    peak_rows.append({
                        "peak_id": global_peak_id,       # <--- Глобальный уникальный ID
                        "x": x,
                        "y": y,
                        "theta_index": int(theta_index),
                        "theta": params["theta0"],
                        "prominence": float(component.max()),
                        "width_samples": params["H"] / np.median(np.diff(theta)),
                        "width_theta": params["H"],
                        "left_ip": np.nan,
                        "right_ip": np.nan,
                        "height": float(clean[theta_index]),
                        "component_id": local_k,         # <--- Локальный номер пика в пикселе
                        "branch_id": np.nan,             # (пока ветвей нет, ставим NaN)
                    })
                    components.append({                  # В truth тоже сохраняем оба ID для удобства проверок
                        "peak_id": global_peak_id, 
                        "component_id": local_k, 
                        **params
                    })
                    global_peak_id += 1                  # Не забываем увеличивать счетчик!

            frame_yxt[y, x] = clean + rng.normal(0, noise_sigma, ntheta)
            truth[(x, y)] = {
                "n_peaks": n,
                "background": bg,
                "components": components,
                "clean_curve": clean,
            }

    frame = np.moveaxis(frame_yxt, -1, 0)  # (Ntheta, Ny, Nx)
    
    # 4. Обновляем список колонок (добавлен component_id)
    peak_catalog = pd.DataFrame(peak_rows, columns=[
        "peak_id", "x", "y", "theta_index", "theta",
        "prominence", "width_samples", "width_theta",
        "left_ip", "right_ip", "height", "branch_id", "component_id"
    ])
    return SyntheticScan(
        frame=frame,
        theta=theta,
        peak_catalog=peak_catalog,
        truth=truth,
    )


def make_synthetic_shoulder_scan(
    theta: ArrayLike, *, spec: ModelSpec | None = None, model_name: str | None = None,
    Ny: int = 256, Nx: int = 256, beam_x: tuple[int, int] | None = None,
    beam_y: tuple[int, int] | None = None, n_curves: int = 4,
    curve_y_end: float | None = None, curve_spacing: float | None = None,
    curve_depth: float | None = None, curve_halfwidth: float | None = None,
    peak_params: dict | None = None, center_param: str | None = None,
    intensity_param: str | None = None, width_param: str | None = None,
    target_peak_height: float | None = 45.0, background: float = 3.0,
    shoulder_fraction: float = 0.205, shoulder_shift: float | None = None,
    split_area_gain: float = 0.20, amplitude_gradient: float = 0.08,
    shoulder_direction: int = 1, noise_sigma: float = 0.9, seed: int = 42,
    require_unimodal: bool = True,
) -> SyntheticScan:
    """Генерирует U-полосы слабого плеча через spec.func; peak_catalog оставляет один видимый пик на пиксель."""
    theta=np.asarray(theta,float)
    if theta.ndim!=1 or len(theta)<3 or not np.all(np.isfinite(theta)) or not np.all(np.diff(theta)>0): raise ValueError("theta должен быть конечным строго возрастающим 1D-массивом")
    if int(Ny)!=Ny or int(Nx)!=Nx or int(n_curves)!=n_curves: raise ValueError("Ny, Nx и n_curves должны быть целыми числами")
    Ny,Nx,n_curves=int(Ny),int(Nx),int(n_curves)
    scalars=np.asarray([noise_sigma,background,shoulder_fraction,split_area_gain,amplitude_gradient],float)
    if Ny<2 or Nx<2 or n_curves<1 or not np.all(np.isfinite(scalars)) or noise_sigma<0 or background<0 or not (0<shoulder_fraction<0.5) or split_area_gain<0 or not (0<=amplitude_gradient<2) or shoulder_direction not in (-1,1): raise ValueError("Некорректные размеры или параметры синтетики")
    if (spec is None)==(model_name is None): raise ValueError("Задайте ровно один из аргументов spec или model_name")
    if model_name is not None:
        from roc_mcs.fitting.registry import MODEL_SPECS
        if model_name not in MODEL_SPECS: raise ValueError(f"Неизвестная модель {model_name!r}; доступны: {sorted(MODEL_SPECS)}")
        spec=MODEL_SPECS[model_name]
    names=tuple(spec.param_names); given=dict(peak_params or {})
    unknown=set(given)-set(names)
    if unknown: raise ValueError(f"Параметры {sorted(unknown)} отсутствуют в spec.param_names={names}")
    def role(value, aliases, label):
        found=value if value is not None else next((name for name in aliases if name in names),None)
        if found not in names: raise ValueError(f"Не удалось определить {label}; задайте его явно. Модель содержит: {names}")
        return found
    center_param=role(center_param, ("theta0","mu","center","x0"), "параметр центра")
    intensity_param=role(intensity_param, ("S","area","amplitude","A","height","I0"), "параметр масштаба/площади")
    if width_param is not None and width_param not in names: raise ValueError(f"width_param={width_param!r} отсутствует в spec.param_names={names}")
    width_param=width_param if width_param is not None else next((name for name in ("H","fwhm","width","sigma","gamma") if name in names),None)
    if width_param is None and shoulder_shift is None: raise ValueError(f"Не найден параметр ширины в {names}; задайте width_param и shoulder_shift явно")
    def fwhm_of(value, name):
        low=name.lower()
        return 2.354820045*value if low in ("sigma","std","stddev") else 2*value if low in ("gamma","hwhm","halfwidth") else value
    span=float(theta[-1]-theta[0]); step=float(np.median(np.diff(theta)))
    shape={}
    for name in names:
        if name in (center_param,intensity_param): continue
        if name in given: value=float(given[name])
        elif name==width_param:
            low=name.lower(); value=(0.08 if low in ("sigma","std","stddev") else 0.095 if low in ("gamma","hwhm","halfwidth") else 0.19)*span
        elif name.lower() in ("eta","mix","mixing","fraction"): value=0.35
        elif name.lower() in ("c","background","baseline","offset"): value=0.0
        else: raise ValueError(f"Для модели {model_name or getattr(spec,'name','<spec>')} задайте peak_params[{name!r}]")
        if not np.isfinite(value): raise ValueError(f"peak_params[{name!r}] должен быть конечным")
        shape[name]=value
    center=float(given.get(center_param,theta[len(theta)//2]))
    if not theta[0]<=center<=theta[-1]: raise ValueError("Центр базового пика лежит вне theta")
    shape[center_param]=center
    width_fwhm=fwhm_of(shape[width_param],width_param) if width_param else np.nan
    if width_param and shape[width_param]<=0: raise ValueError("Параметр ширины должен быть > 0")
    if shoulder_shift is None: shoulder_shift=1.05*width_fwhm
    shoulder_shift=float(shoulder_shift)
    if not np.isfinite(shoulder_shift) or shoulder_shift<=0: raise ValueError("shoulder_shift должен быть конечным числом > 0")
    if intensity_param in given:
        if target_peak_height is not None: raise ValueError("При явном intensity_param в peak_params задайте target_peak_height=None")
        base_scale=float(given[intensity_param])
        if not np.isfinite(base_scale) or base_scale<=0: raise ValueError("Масштаб/площадь пика должен быть > 0")
    else:
        if target_peak_height is None or not np.isfinite(target_peak_height) or target_peak_height<=background: raise ValueError("target_peak_height должен быть конечным и больше background")
        unit_params={**shape,intensity_param:1.0}; unit=np.asarray(spec.func(theta,**unit_params),float)
        if unit.shape!=theta.shape or not np.all(np.isfinite(unit)) or float(np.max(unit))<=0: raise ValueError("spec.func вернул некорректный профиль при единичном масштабе")
        base_scale=(float(target_peak_height)-background)/float(np.max(unit))
    if beam_x is None: beam_x=(50,min(210,Nx-1)) if Nx>=211 else (int(round(0.2*(Nx-1)),),int(round(0.82*(Nx-1))))
    if beam_y is None: beam_y=(0,Ny-1)
    x0,x1=map(int,beam_x); y0,y1=map(int,beam_y)
    if not (0<=x0<x1<Nx and 0<=y0<=y1<Ny): raise ValueError("beam_x/beam_y должны быть включительными границами внутри кадра; x должен иметь ширину >=2 px")
    bh=y1-y0+1; depth=float(curve_depth if curve_depth is not None else 0.36*bh); top=float(curve_y_end if curve_y_end is not None else y1-0.05*bh)
    if not np.all(np.isfinite([depth,top])) or depth<=0: raise ValueError("curve_depth/curve_y_end должны быть конечными; curve_depth > 0")
    bottom_min=y0+depth+0.12*bh
    spacing=float(curve_spacing if curve_spacing is not None else (top-bottom_min)/(n_curves-1) if n_curves>1 else 0.0)
    halfwidth=float(curve_halfwidth if curve_halfwidth is not None else max(1.0,0.065*bh))
    if not np.all(np.isfinite([spacing,halfwidth])) or halfwidth<=0 or (n_curves>1 and spacing<=0): raise ValueError("curve_halfwidth/curve_spacing должны быть конечными и > 0")
    if top>y1 or top-depth<y0: raise ValueError("curve_y_end/curve_depth выводят U за beam_y")
    if n_curves>1 and top-(n_curves-1)*spacing<bottom_min-1e-9: raise ValueError("U-кривые не помещаются в beam_y: уменьшите depth/n_curves или измените spacing/y_end")
    yy,xx=np.mgrid[0:Ny,0:Nx]; xn=(xx-x0)/(x1-x0); beam=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1); weight=np.zeros((Ny,Nx),float); band_id=np.full((Ny,Nx),-1,np.int16)
    for j in range(n_curves):
        t=xn; yc=top-j*spacing-depth*4*t*(1-t); slope=-4*depth*(1-2*t)/(x1-x0); distance=np.abs(yy-yc)/np.sqrt(1+slope*slope)
        q=np.where(distance<halfwidth,0.5*(1+np.cos(np.pi*np.minimum(distance/halfwidth,1.0))),0.0); q[~beam]=0.0; take=q>weight; weight[take]=q[take]; band_id[take]=j
    weight[~beam]=0.0
    if shoulder_shift is None: shoulder_shift=1.05*width_fwhm
    if np.any(weight>1+1e-12) or not np.all(np.isfinite(weight)): raise RuntimeError("Некорректное поле split_weight")
    rng=np.random.default_rng(seed); frame_yxt=np.empty((Ny,Nx,len(theta)),float); peak_rows=[]; component_rows=[]; truth={}; main_peak_id=0; component_peak_id=0
    for y in range(Ny):
        for x in range(Nx):
            w=float(weight[y,x]); bg=float(background); amp=1.0+amplitude_gradient*(xn[y,x]-0.5) if beam[y,x] else 1.0
            clean=np.full(len(theta),bg,float); components=[]; main_id=None
            if beam[y,x]:
                local_center=center; total_scale=base_scale*amp*(1.0+split_area_gain*w); frac=shoulder_fraction*w
                p0={**shape,center_param:local_center,intensity_param:total_scale*(1.0-frac)}; curve0=np.asarray(spec.func(theta,**p0),float)
                if curve0.shape!=theta.shape or not np.all(np.isfinite(curve0)): raise ValueError("spec.func вернул некорректную компоненту")
                clean+=curve0; main_id=component_peak_id
                comp0={"peak_id":component_peak_id,"component_id":0,"is_shoulder":False,"truth_branch_id":-1,"split_weight":w,"split_separation":shoulder_shift*w,"intensity_gain":1.0+split_area_gain*w,**p0}
                components.append(comp0); component_rows.append({"peak_id":component_peak_id,"x":x,"y":y,"background":bg,"component_id":0,"theta_index":int(np.argmin(np.abs(theta-local_center))),"theta":local_center,"truth_branch_id":-1,"is_shoulder":False,"split_weight":w,"split_separation":shoulder_shift*w,"intensity_gain":1.0+split_area_gain*w,"model_name":model_name or getattr(spec,"name",None),**p0})
                component_peak_id+=1
                if frac>0:
                    local_center2=local_center+shoulder_direction*shoulder_shift*w; p1={**shape,center_param:local_center2,intensity_param:total_scale*frac}; curve1=np.asarray(spec.func(theta,**p1),float)
                    if curve1.shape!=theta.shape or not np.all(np.isfinite(curve1)): raise ValueError("spec.func вернул некорректную компоненту плеча")
                    clean+=curve1; components.append({"peak_id":component_peak_id,"component_id":1,"is_shoulder":True,"truth_branch_id":int(band_id[y,x]),"split_weight":w,"split_separation":shoulder_shift*w,"intensity_gain":1.0+split_area_gain*w,**p1})
                    component_rows.append({"peak_id":component_peak_id,"x":x,"y":y,"background":bg,"component_id":1,"theta_index":int(np.argmin(np.abs(theta-local_center2))),"theta":local_center2,"truth_branch_id":int(band_id[y,x]),"is_shoulder":True,"split_weight":w,"split_separation":shoulder_shift*w,"intensity_gain":1.0+split_area_gain*w,"model_name":model_name or getattr(spec,"name",None),**p1})
                    component_peak_id+=1
                if require_unimodal:
                    slopes=np.sign(np.diff(clean)); slopes=slopes[slopes!=0]; maxima=int(np.count_nonzero((slopes[:-1]>0)&(slopes[1:]<0)))
                    if maxima>1: raise ValueError(f"Модель даёт {maxima} локальных максимумов при shoulder_shift={shoulder_shift:g}, shoulder_fraction={shoulder_fraction:g}; уменьшите одно из этих значений")
                main_peak_id=main_id
            observed=np.maximum(clean+rng.normal(0.0,noise_sigma,len(theta)),0.0) if noise_sigma else np.maximum(clean,0.0); frame_yxt[y,x]=observed
            if beam[y,x]:
                ti=int(np.argmax(observed)); prominence=float(max(observed[ti]-bg,0.0))
                peak_rows.append({"peak_id":main_peak_id,"x":x,"y":y,"theta_index":ti,"theta":float(theta[ti]),"prominence":prominence,"width_samples":float(width_fwhm/step) if np.isfinite(width_fwhm) else np.nan,"width_theta":float(width_fwhm) if np.isfinite(width_fwhm) else np.nan,"left_ip":np.nan,"right_ip":np.nan,"height":float(observed[ti]),"branch_id":np.nan,"component_id":0})
            truth[(x,y)]={"n_peaks":int(bool(beam[y,x])),"n_components":len(components),"background":bg,"components":components,"clean_curve":clean,"split_weight":w,"split_separation":shoulder_shift*w if beam[y,x] else 0.0,"intensity_gain":1.0+split_area_gain*w if beam[y,x] else 1.0,"truth_branch_id":int(band_id[y,x])}
    frame=np.moveaxis(frame_yxt,-1,0); catalog_columns=["peak_id","x","y","theta_index","theta","prominence","width_samples","width_theta","left_ip","right_ip","height","branch_id","component_id"]
    peak_catalog=pd.DataFrame(peak_rows,columns=catalog_columns); component_catalog=pd.DataFrame(component_rows)
    return SyntheticScan(frame=frame,theta=theta,peak_catalog=peak_catalog,truth=truth,component_catalog=component_catalog,split_weight=weight.astype(float),split_separation=(weight*shoulder_shift).astype(float))


def save_synthetic_scan_h5(scan: SyntheticScan, path, *, entry_id: int = 472, compression: str | None = "gzip", compression_opts: int = 4, include_truth: bool = False):
    """Сохраняет scan в формате H5ScanRepository; include_truth добавляет каталог параметров в отдельную группу."""
    import h5py
    from pathlib import Path
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); key=f"entry{int(entry_id)}_CALC"
    if scan.frame.ndim!=3 or scan.frame.shape[0]!=len(scan.theta): raise ValueError("scan.frame должен иметь форму (Ntheta,Ny,Nx), согласованную с scan.theta")
    opts={} if compression is None else {"compression":compression,"shuffle":True}
    if compression=="gzip": opts["compression_opts"]=compression_opts
    with h5py.File(path,"w") as f:
        g=f.create_group(key); g.attrs["axis"]=0; g.attrs["synthetic"]=True
        g.create_dataset("raw_data_2D",data=scan.frame,**opts); g.create_dataset("raw_data_th",data=scan.theta); g.create_dataset("raw_data_total_int",data=np.sum(scan.frame,axis=0),**opts)
        if include_truth:
            t=f.create_group("synthetic_truth"); t.attrs["entry_id"]=int(entry_id)
            for name,array in (("split_weight",scan.split_weight),("split_separation",scan.split_separation)):
                if array is not None: t.create_dataset(name,data=array,**opts)
            for table_name,table in (("component_catalog",scan.component_catalog),("peak_catalog",scan.peak_catalog)):
                if table is None: continue
                tg=t.create_group(table_name)
                for column in table.columns:
                    values=table[column].to_numpy()
                    if values.dtype.kind in "biuf": tg.create_dataset(column,data=values,**opts)
                    else:
                        text_values=np.asarray(["" if pd.isna(v) else str(v) for v in values],dtype=object)
                        tg.create_dataset(column,data=text_values,dtype=h5py.string_dtype("utf-8"),**opts)
    return path
