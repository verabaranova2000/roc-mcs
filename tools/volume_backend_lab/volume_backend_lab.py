"""Изолированный стенд сравнения Plotly, PyVista/VTK и napari для ROI Explorer.

Не импортирует и не меняет расчётные модули ROC-MCS. Данные принимаются по
протоколу ROIVolume: атрибуты volume, x, y, theta, coord_system.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, prod
from time import perf_counter
from typing import Any
import numpy as np

DEFAULT_STRIDE = 2
MAX_PLOTLY_NODES = 160_000
MAX_VTK_NODES = 450_000
MAX_PROJECTION_PIXELS = 420_000

@dataclass(frozen=True)
class RenderData:
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    projection_x: np.ndarray
    projection_y: np.ndarray
    volume: np.ndarray
    projection: np.ndarray
    stride: int
    source_shape: tuple[int, int, int]
    projection_stride: int
    prepare_seconds: float
    def stats(self) -> dict[str, Any]:
        return {"source_shape": self.source_shape, "render_shape": tuple(int(v) for v in self.volume.shape),
                "stride": self.stride, "render_nodes": int(self.volume.size),
                "projection_shape": tuple(int(v) for v in self.projection.shape),
                "projection_stride": self.projection_stride, "prepare_seconds": round(self.prepare_seconds, 4),
                "sampled_MiB_float32": round(self.volume.nbytes / (1024**2), 2)}

def _source_arrays(source: Any):
    for name in ("volume", "x", "y", "theta", "coord_system"):
        if not hasattr(source, name): raise TypeError(f"Объект данных должен содержать атрибут {name!r}.")
    volume = np.asanyarray(source.volume)
    x = np.asarray(source.x, dtype=np.float64).reshape(-1); y = np.asarray(source.y, dtype=np.float64).reshape(-1); theta = np.asarray(source.theta, dtype=np.float64).reshape(-1)
    if volume.ndim != 3 or volume.shape != (len(x), len(y), len(theta)):
        raise ValueError(f"Форма volume {volume.shape} не совпадает с (len(x), len(y), len(theta)) = {(len(x),len(y),len(theta))}.")
    if str(source.coord_system) not in ("xy", "uv"): raise ValueError("coord_system должен быть 'xy' или 'uv'.")
    if min(volume.shape) < 2: raise ValueError("Каждое измерение объёма должно содержать хотя бы 2 точки.")
    for label, axis in (("x",x),("y",y),("theta",theta)):
        if not np.all(np.isfinite(axis)): raise ValueError(f"Ось {label} содержит NaN/Inf.")
        delta = np.diff(axis)
        if not (np.all(delta > 0) or np.all(delta < 0)): raise ValueError(f"Ось {label} должна быть строго монотонной.")
    return volume, x, y, theta, str(source.coord_system)

def _pick_stride(shape: tuple[int, int, int], requested: int, max_nodes: int) -> int:
    stride = int(requested)
    if stride != requested or stride < 1: raise ValueError("Шаг прореживания должен быть положительным целым числом.")
    if max_nodes < 8: raise ValueError("max_nodes должен быть не меньше 8.")
    while prod((n + stride - 1) // stride for n in shape) > int(max_nodes): stride += 1
    return stride

def _sort_render_axes(x, y, theta, sampled, projection):
    if x[0] > x[-1]: x=x[::-1]; sampled=sampled[::-1,:,:]; projection=projection[::-1,:]
    if y[0] > y[-1]: y=y[::-1]; sampled=sampled[:,::-1,:]; projection=projection[:,::-1]
    if theta[0] > theta[-1]: theta=theta[::-1]; sampled=sampled[:,:,::-1]
    return x, y, theta, sampled, projection

def prepare_render_data(source: Any, *, stride: int = DEFAULT_STRIDE, smooth_sigma: float = 0.5,
                        max_nodes: int = MAX_PLOTLY_NODES) -> RenderData:
    """Подготавливает ограниченный объём; не создаёт полные X/Y/Theta 3D-сетки."""
    t0=perf_counter(); src,x,y,theta,coord_system=_source_arrays(source)
    sigma=float(smooth_sigma)
    if not np.isfinite(sigma) or sigma < 0: raise ValueError("smooth_sigma должен быть конечным неотрицательным числом.")
    step=_pick_stride(src.shape,stride,max_nodes)
    # 2D-проекция вычисляется по исходному объёму, чтобы не терять интенсивность вдоль θ.
    with np.errstate(invalid="ignore", over="ignore"):
        projection=np.nansum(src,axis=2,dtype=np.float64)
    projection[~np.isfinite(projection)]=np.nan
    if not np.isfinite(projection).any(): raise ValueError("В интегральной проекции нет конечных данных.")
    # Сглаживаем только двумерную проекцию и уже прореженный 3D-объём — без копии полного float64-тома.
    if sigma:
        from scipy.ndimage import gaussian_filter
        good=np.isfinite(projection); num=gaussian_filter(np.where(good,projection,0.0),sigma=sigma); den=gaussian_filter(good.astype(np.float32),sigma=sigma)
        projection=np.divide(num,den,out=np.full_like(projection,np.nan),where=den>1e-8)
    sampled=np.asarray(src[::step,::step,::step],dtype=np.float32)
    if sigma:
        from scipy.ndimage import gaussian_filter
        good=np.isfinite(sampled); num=gaussian_filter(np.where(good,sampled,0.0),sigma=(sigma,sigma,min(0.2,sigma))); den=gaussian_filter(good.astype(np.float32),sigma=(sigma,sigma,min(0.2,sigma)))
        sampled=np.divide(num,den,out=np.full_like(sampled,np.nan),where=den>1e-8)
    xs=x[::step]; ys=y[::step]; ts=theta[::step]; px=x.copy(); py=y.copy()
    # Сортируем рендер-оси; исходную проекцию сохраняем в полном разрешении.
    if xs[0] > xs[-1]: xs=xs[::-1]; sampled=sampled[::-1,:,:]
    if ys[0] > ys[-1]: ys=ys[::-1]; sampled=sampled[:,::-1,:]
    if ts[0] > ts[-1]: ts=ts[::-1]; sampled=sampled[:,:,::-1]
    if px[0] > px[-1]: px=px[::-1]; projection=projection[::-1,:]
    if py[0] > py[-1]: py=py[::-1]; projection=projection[:,::-1]
    good=np.isfinite(sampled)
    if not good.any(): raise ValueError("В прореженном объёме нет конечных значений.")
    vals=sampled[good]
    if float(np.nanmax(vals)) <= 0: raise ValueError("В объёме нет положительных интенсивностей.")
    pstep=max(1,ceil(np.sqrt((projection.shape[0]*projection.shape[1])/MAX_PROJECTION_PIXELS)))
    p=projection[::pstep,::pstep]
    finite=p[np.isfinite(p)]
    if not finite.size: raise ValueError("Проекция не содержит конечных интенсивностей.")
    # `projection` remains at source resolution here; backend applies its own visual decimation.
    return RenderData(xs,ys,ts,px,py,sampled,projection,step,tuple(int(v) for v in src.shape),pstep,perf_counter()-t0)

def _projection_stride(shape: tuple[int,int], max_pixels: int = MAX_PROJECTION_PIXELS) -> int:
    return max(1, int(ceil(np.sqrt((shape[0]*shape[1])/max_pixels))))

def build_plotly_reference(source: Any, *, theme: str = "light", stride: int = DEFAULT_STRIDE,
                           smooth_sigma: float = 0.5, max_nodes: int = MAX_PLOTLY_NODES):
    """Стилистический эталон Plotly: 2D-проекция + полупрозрачный Volume."""
    import plotly.graph_objects as go
    d=prepare_render_data(source,stride=stride,smooth_sigma=smooth_sigma,max_nodes=max_nodes)
    dark=str(theme).lower()=="dark"; bg="#111316" if dark else "#ffffff"; fg="#e1e4e8" if dark else "#111111"
    proj_step=_projection_stride(d.projection.shape); proj=d.projection[::proj_step,::proj_step]
    px=d.projection_x[::max(1,proj_step)]; py=d.projection_y[::max(1,proj_step)]; xx,yy=np.meshgrid(px,py,indexing="ij")
    finite_proj=proj[np.isfinite(proj)]; pmin,pmax=np.percentile(finite_proj,[2,98])
    if pmin>=pmax:pmax=pmin+max(abs(float(pmin))*1e-9,1e-12)
    values=d.volume[np.isfinite(d.volume)]; robust=float(np.percentile(values,99.9)); robust=max(robust,float(values.max())*1e-9,1e-12)
    zmin,zmax=float(d.theta[0]),float(d.theta[-1]); zfloor=zmin-max((zmax-zmin)*0.15,1e-9)
    mask=np.isfinite(d.volume); xv=np.broadcast_to(d.x[:,None,None],d.volume.shape)[mask]; yv=np.broadcast_to(d.y[None,:,None],d.volume.shape)[mask]; zv=np.broadcast_to(d.theta[None,None,:],d.volume.shape)[mask]; vv=d.volume[mask]
    maxxy=max(len(d.x),len(d.y),1); coord_system=str(getattr(source,"coord_system","xy")); xl,yl=("Detector X (pixels)","Detector Y (pixels)") if coord_system=="xy" else ("ROI u (pixels)","ROI v (pixels)")
    grid="#343941" if dark else "#e0e0e0"; axisline="#aeb5be" if dark else "#222222"
    axis=dict(showbackground=False,gridcolor=grid,zeroline=False,showspikes=False,linecolor=axisline,linewidth=1.2,ticks="inside",tickcolor=axisline,title_font=dict(color=fg,size=15,family="Times New Roman"),tickfont=dict(color=fg,size=12,family="Times New Roman"))
    fig=go.Figure()
    fig.add_trace(go.Surface(x=xx,y=yy,z=np.full(xx.shape,zfloor),surfacecolor=proj,colorscale="Greys_r" if dark else "Greys",cmin=float(pmin),cmax=float(pmax),showscale=False,hoverinfo="skip",opacity=0.75,connectgaps=False,contours=dict(x=dict(show=True,color="rgba(110,110,110,0.16)",width=1),y=dict(show=True,color="rgba(110,110,110,0.16)",width=1)),name="Интегральная проекция"))
    fig.add_trace(go.Volume(x=xv,y=yv,z=zv,value=vv,isomin=robust*0.16,isomax=robust*0.95,opacity=0.10,surface_count=25,colorscale="Magma" if dark else "Blues",showscale=False,caps=dict(x_show=False,y_show=False,z_show=False),lighting=dict(ambient=0.95,diffuse=0.25,specular=0.0),name="Интенсивность",hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>θ=%{z:.5g}°<br>I=%{value:.5g}<extra></extra>"))
    fig.update_layout(margin=dict(l=0,r=0,t=8,b=0),paper_bgcolor=bg,plot_bgcolor=bg,font=dict(family="Times New Roman",color=fg,size=12),scene=dict(bgcolor=bg,xaxis_title=xl,yaxis_title=yl,zaxis_title="Angle θ (°)",aspectmode="manual",aspectratio=dict(x=len(d.x)/maxxy*2.2,y=len(d.y)/maxxy*2.2,z=1.8),xaxis=axis,yaxis=axis,zaxis={**axis,"range":[zfloor,zmax]},camera=dict(eye=dict(x=1.6,y=-1.9,z=1.25),up=dict(x=0,y=0,z=1))),showlegend=False)
    meta=d.stats(); meta.update({"backend":"Plotly reference","projection_render_shape":tuple(int(v) for v in proj.shape),"volume_nodes_sent":int(vv.size),"robust_max":robust})
    return fig,meta

def build_pyvista_scene(plotter: Any, source: Any, *, theme: str = "dark", stride: int = DEFAULT_STRIDE,
                        smooth_sigma: float = 0.5, max_nodes: int = MAX_VTK_NODES):
    """Рисует сцену PyVista/VTK в переданном Plotter/QtInteractor."""
    try: import pyvista as pv
    except ImportError as exc: raise RuntimeError("PyVista не установлен. В отдельном тестовом окружении установи pyvista, pyvistaqt и vtk.") from exc
    d=prepare_render_data(source,stride=stride,smooth_sigma=smooth_sigma,max_nodes=max_nodes)
    dark=str(theme).lower()=="dark"; bg="#111316" if dark else "#ffffff"; fg="#e1e4e8" if dark else "#181818"
    # Приводим рендер-координаты к возрастающему порядку; prepare_render_data уже сортирует оси.
    vol=np.asarray(d.volume,dtype=np.float32).copy(); vol[~np.isfinite(vol)]=0.0
    # Визуально расширяем ось θ, как aspectratio в Plotly, но сохраняем реальные подписи углов.
    zrange=max(float(np.ptp(d.theta)),1e-9); xyspan=max(float(np.ptp(d.x)),float(np.ptp(d.y)),1.0); zscale=max(1.0,1.8*xyspan/zrange)
    theta_mid=(float(d.theta[0])+float(d.theta[-1]))/2; theta_render=theta_mid+(d.theta-theta_mid)*zscale
    grid=pv.RectilinearGrid(np.asarray(d.x),np.asarray(d.y),np.asarray(theta_render)); grid.point_data["Intensity"]=vol.ravel(order="F")
    vfinite=vol[np.isfinite(vol) & (vol>0)]; robust=max(float(np.percentile(vfinite,99.9)) if vfinite.size else 1.0,1e-12)
    plotter.clear(); plotter.set_background(bg)
    opacity=np.concatenate((np.zeros(24,dtype=float),np.linspace(0.002,0.12,232)))
    plotter.add_volume(grid,scalars="Intensity",cmap="magma" if dark else "Blues",clim=(0,robust),opacity=opacity,shade=True,ambient=0.95,diffuse=0.25,specular=0.0,show_scalar_bar=False,mapper="smart",name="intensity-volume")
    ps=_projection_stride(d.projection.shape); projection=d.projection[::ps,::ps]; px=d.projection_x[::ps]; py=d.projection_y[::ps]
    xx,yy=np.meshgrid(px,py,indexing="ij"); zmin,zmax=float(d.theta[0]),float(d.theta[-1]); zfloor=zmin-max((zmax-zmin)*0.15,1e-9); zfloor_render=theta_mid+(zfloor-theta_mid)*zscale; zz=np.full_like(xx,zfloor_render,dtype=float)
    plane=pv.StructuredGrid(xx,yy,zz); plane.point_data["Integrated intensity"]=np.nan_to_num(projection,nan=0.0).ravel(order="F")
    fin=projection[np.isfinite(projection)]; clim=tuple(np.percentile(fin,[2,98])) if fin.size else (0.0,1.0)
    if clim[0]>=clim[1]: clim=(float(clim[0]),float(clim[0])+max(abs(float(clim[0]))*1e-9,1e-12))
    plotter.add_mesh(plane,scalars="Integrated intensity",cmap="Greys_r" if dark else "Greys",clim=clim,opacity=0.75,lighting=False,show_scalar_bar=False,name="integrated-projection")
    maxrange=max(float(np.ptp(d.x)),float(np.ptp(d.y)),float(np.ptp(theta_render)),1.0); center=((d.x[0]+d.x[-1])/2,(d.y[0]+d.y[-1])/2,theta_mid)
    plotter.camera.focal_point=center; plotter.camera.position=(center[0]+1.6*maxrange,center[1]-1.9*maxrange,center[2]+1.25*maxrange); plotter.camera.up=(0,0,1)
    try: plotter.show_bounds(grid="back",location="outer",color=fg,font_size=10,axes_ranges=[float(d.x[0]),float(d.x[-1]),float(d.y[0]),float(d.y[-1]),zmin,zmax],xtitle="Detector X" if getattr(source,"coord_system","xy")=="xy" else "ROI u",ytitle="Detector Y" if getattr(source,"coord_system","xy")=="xy" else "ROI v",ztitle="Angle θ (°)",all_edges=False)
    except Exception: plotter.add_axes(line_width=1,color=fg)
    plotter.render(); meta=d.stats(); meta.update({"backend":"PyVista/VTK","projection_render_shape":tuple(int(v) for v in projection.shape),"volume_nodes_sent":int(vol.size),"robust_max":robust,"theta_visual_scale":round(zscale,3)}); return meta

def open_pyvista_window(source: Any, *, theme: str = "dark", stride: int = DEFAULT_STRIDE,
                        smooth_sigma: float = 0.5, max_nodes: int = MAX_VTK_NODES):
    """Открывает отдельное окно Qt для ноутбука; GUI ROI Explorer не затрагивается."""
    try: from pyvistaqt import BackgroundPlotter
    except ImportError as exc: raise RuntimeError("Установи pyvista, pyvistaqt и vtk в отдельном окружении для сравнения.") from exc
    plotter=BackgroundPlotter(title="ROI Explorer — PyVista/VTK backend test",show=True)
    metadata=build_pyvista_scene(plotter,source,theme=theme,stride=stride,smooth_sigma=smooth_sigma,max_nodes=max_nodes)
    plotter._roi_backend_lab_metadata=metadata
    return plotter

def open_napari_window(source: Any, *, theme: str = "dark", stride: int = DEFAULT_STRIDE, max_nodes: int = MAX_VTK_NODES):
    """Контрольный объёмный просмотр napari; не обещает Plotly-подобную projection plane."""
    try: import napari
    except ImportError as exc: raise RuntimeError("napari не установлен. Установи его в отдельном тестовом окружении.") from exc
    d=prepare_render_data(source,stride=stride,smooth_sigma=0.0,max_nodes=max_nodes)
    vol=np.transpose(np.asarray(d.volume,dtype=np.float32),(2,1,0)).copy(); good=vol[np.isfinite(vol)]
    if not good.size: raise ValueError("В объёме нет конечных значений.")
    clim=(float(np.percentile(good,1)),float(np.percentile(good,99.9)))
    if clim[0]>=clim[1]: clim=(float(good.min()),float(good.max()) if good.max()>good.min() else float(good.min()+1))
    def step(a): return float(np.median(np.diff(a))) if len(a)>1 else 1.0
    viewer=napari.Viewer(ndisplay=3); viewer.theme="dark" if str(theme).lower()=="dark" else "light"
    viewer.add_image(vol,name="Дифракционный объём",colormap="magma" if theme=="dark" else "blue",contrast_limits=clim,rendering="attenuated_mip",scale=(step(d.theta),step(d.y),step(d.x)))
    viewer.window._roi_backend_lab_metadata=d.stats(); return viewer

def make_synthetic_volume(shape=(72,64,44)):
    """Детерминированный тестовый объём без файлов лабораторных данных."""
    from types import SimpleNamespace
    nx,ny,nt=map(int,shape); x=np.linspace(0,nx-1,nx); y=np.linspace(0,ny-1,ny); theta=np.linspace(-1.6,1.6,nt)
    X,Y,T=np.meshgrid(x,y,theta,indexing="ij")
    vol=(80*np.exp(-((X-nx*.40)**2/(2*(nx*.08)**2)+(Y-ny*.52)**2/(2*(ny*.10)**2)+(T+0.25)**2/(2*.22**2)))
         +45*np.exp(-((X-nx*.67)**2/(2*(nx*.05)**2)+(Y-ny*.36)**2/(2*(ny*.065)**2)+(T-.45)**2/(2*.14**2)))
         +2.5*np.exp(-T*T/(2*.8**2))).astype(np.float32)
    # Симулируем редкие отсутствующие пиксели, не добавляя шума случайности.
    vol[::19,::17,::13]=np.nan
    return SimpleNamespace(volume=vol,x=x,y=y,theta=theta,coord_system="xy",roi=None)
