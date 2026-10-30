"""Подготовка Plotly 3D-сцены с ограничением памяти и числа узлов WebGL."""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from roc_mcs.roi.roi_types import ROIVolume

LIGHT_PROJECTION_SCALE = "Greys"
DARK_PROJECTION_SCALE = [[0.0, "#f7f7f7"], [0.55, "#dedede"], [1.0, "#ffffff"]]
VOLUME_SCALES = ("Blues", "Magma", "Viridis", "Cividis", "Inferno", "Plasma", "Hot", "Electric")
PROJECTION_SCALES = ("Slate", "Greys", "Cividis", "Viridis")
MAX_RENDER_POINTS = 18_000
MAX_PROJECTION_POINTS = 1_500_000
MAX_CACHE_VOXELS = 1_000_000
MAX_EXTRACTION_VOXELS = 6_000_000


def effective_volume_stride(shape, requested=2, max_points=MAX_RENDER_POINTS):
    """Минимальный шаг ≥2, ограничивающий сетку перед отправкой в WebGL."""
    dims=tuple(max(1,int(n)) for n in shape); stride=max(2,int(requested)); limit=max(1000,int(max_points))
    def points(step): return math.prod((n+step-1)//step for n in dims)
    active=max(1,sum(n>1 for n in dims)); total=math.prod(dims)
    if total>limit:stride=max(stride,int(math.ceil((total/limit)**(1/active))))
    while points(stride)>limit:stride+=1
    return stride


def estimate_roi_voxels(roi, x_coords, y_coords, n_theta):
    """Оценивает размер сетки ROI с учётом поведения экстрактора."""
    x=np.asarray(x_coords,dtype=np.float64).reshape(-1); y=np.asarray(y_coords,dtype=np.float64).reshape(-1)
    def step(values):
        delta=np.abs(np.diff(values)); delta=delta[np.isfinite(delta)&(delta>1e-12)]
        return float(np.median(delta)) if delta.size else 1.0
    width=max(0.0,abs(float(getattr(roi,"width",0.0)))); height=max(0.0,abs(float(getattr(roi,"height",0.0))))
    angle=math.radians(float(getattr(roi,"angle",0.0))); aligned=abs(math.sin(2*angle))<1e-10
    if aligned:
        c,s=abs(math.cos(angle)),abs(math.sin(angle)); nx=min(max(1,x.size),max(1,int(math.ceil((width*c+height*s)/step(x)))+2)); ny=min(max(1,y.size),max(1,int(math.ceil((width*s+height*c)/step(y)))+2))
    else:
        grid_step=min(step(x),step(y)); nx=max(2,int(math.ceil(width/grid_step))+1); ny=max(2,int(math.ceil(height/grid_step))+1)
    nt=max(1,int(n_theta)); return (nx,ny,nt),nx*ny*nt


def crop_raw_to_roi(raw, axis, x_coords, y_coords, roi):
    """Возвращает view raw в bbox ROI с одним узлом запаса; координаты остаются абсолютными."""
    data=np.asarray(raw); x=np.asarray(x_coords,dtype=np.float64).reshape(-1); y=np.asarray(y_coords,dtype=np.float64).reshape(-1)
    if data.ndim!=3 or x.size==0 or y.size==0: raise ValueError("Для извлечения ROI нужен 3D raw и непустые оси x/y")
    if axis==0 and data.shape[1:]!=(y.size,x.size): raise ValueError("Размеры raw не согласованы с осями y/x при axis=0")
    if axis==2 and data.shape[:2]!=(y.size,x.size): raise ValueError("Размеры raw не согласованы с осями y/x при axis=2")
    if axis not in (0,2): raise ValueError(f"Неподдерживаемая ось theta: {axis}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or (x.size>1 and np.any(np.diff(x)<=0)) or (y.size>1 and np.any(np.diff(y)<=0)): raise ValueError("Оси x/y должны быть конечными и возрастающими")
    r=roi.normalized() if callable(getattr(roi,"normalized",None)) else roi; x0,x1,y0,y1=map(float,r.bbox)
    if max(x0,x1)<x[0] or min(x0,x1)>x[-1] or max(y0,y1)<y[0] or min(y0,y1)>y[-1]: raise ValueError("ROI не пересекается с текущей областью детектора")
    def bounds(values,lo,hi):
        start=max(0,int(np.searchsorted(values,lo,side="left"))-1); stop=min(values.size,int(np.searchsorted(values,hi,side="right"))+1)
        if stop<=start:
            idx=max(0,min(values.size-1,int(np.searchsorted(values,(lo+hi)*0.5)))); start,stop=idx,idx+1
        return start,stop
    ix0,ix1=bounds(x,min(x0,x1),max(x0,x1)); iy0,iy1=bounds(y,min(y0,y1),max(y0,y1))
    view=data[:,iy0:iy1,ix0:ix1] if axis==0 else data[iy0:iy1,ix0:ix1,:]
    return view,x[ix0:ix1],y[iy0:iy1],(ix0,ix1,iy0,iy1)


def effective_extraction_stride(roi, x_coords, y_coords, n_theta, requested=2, max_voxels=MAX_EXTRACTION_VOXELS):
    """Выбирает шаг входных данных до экстракции, начиная минимум с 2."""
    shape,_=estimate_roi_voxels(roi,x_coords,y_coords,n_theta); step=max(2,int(requested)); limit=max(100_000,int(max_voxels))
    def points(s): return math.prod((n+s-1)//s for n in shape)
    while points(step)>limit: step+=1
    return step,shape,points(step)


def _finite_percentile(a, q, fallback=(0.0, 1.0)):
    values=np.asarray(a); values=values[np.isfinite(values)]
    if not values.size:return fallback
    lo,hi=np.percentile(values,q)
    if not np.isfinite(lo) or not np.isfinite(hi):return fallback
    if lo>=hi:hi=lo+max(abs(lo)*1e-9,1e-12)
    return float(lo),float(hi)


def _projection_scale(name: str, theme: str):
    if str(name).lower()=="slate":return DARK_PROJECTION_SCALE if theme=="dark" else LIGHT_PROJECTION_SCALE
    if theme=="dark" and str(name).lower()=="greys":return DARK_PROJECTION_SCALE
    return name


def prepare_volume(rv: ROIVolume, *, smooth_sigma=0.5, stride=2):
    """Сглаживает ограниченный ROI, считает полную 2D-проекцию и прореживает только 3D."""
    if not isinstance(rv,ROIVolume):raise TypeError("Ожидается ROIVolume")
    source=np.asarray(rv.volume); x,y,theta=(np.asarray(v,dtype=np.float64) for v in (rv.x,rv.y,rv.theta))
    if source.ndim!=3 or source.shape!=(len(x),len(y),len(theta)):raise ValueError("Форма ROIVolume не согласована с осями")
    if rv.coord_system not in {"xy","uv"}:raise ValueError(f"Неизвестная система координат: {rv.coord_system}")
    sigma=float(smooth_sigma)
    if not np.isfinite(sigma) or sigma<0:raise ValueError("smooth_sigma должен быть конечным и неотрицательным")
    # Извлекатель ограничивает объём ROI; здесь float32 снижает память, но сохраняется полная детализация проекции.
    full=np.asarray(source,dtype=np.float32); finite=np.isfinite(full)
    if not finite.any():raise ValueError("В объёме нет конечных значений")
    if sigma>0:
        sig=(sigma,sigma,0.2); num=gaussian_filter(np.where(finite,full,0).astype(np.float32,copy=False),sigma=sig,output=np.float32); den=gaussian_filter(finite.astype(np.float32),sigma=sig,output=np.float32)
        full=np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-8); finite=np.isfinite(full)
    projection=np.nansum(full,axis=2,dtype=np.float64).astype(np.float32)
    stride=effective_volume_stride(full.shape,stride)
    sampled=np.asarray(full[::stride,::stride,::stride],dtype=np.float32); xs,ys,ts=x[::stride],y[::stride],theta[::stride]
    finite=np.isfinite(sampled)&np.isfinite(xs)[:,None,None]&np.isfinite(ys)[None,:,None]&np.isfinite(ts)[None,None,:]
    vals=sampled[finite]
    if not vals.size or float(np.nanmax(vals))<=0:raise ValueError("В объёме нет положительных конечных интенсивностей")
    xv=np.broadcast_to(xs[:,None,None],sampled.shape)[finite]; yv=np.broadcast_to(ys[None,:,None],sampled.shape)[finite]; zv=np.broadcast_to(ts[None,None,:],sampled.shape)[finite]
    return sampled,projection,xv,yv,zv,vals

def build_volume_figure(rv: ROIVolume, settings: dict, theme="light", *, peak_catalog=None):
    """Строит объём с полным интегральным срезом и ограниченным числом точек WebGL."""
    import plotly.graph_objects as go
    theme="dark" if str(theme).lower()=="dark" else "light"
    requested=max(2,int(settings.get("volume_stride",2))); sigma=max(0.0,float(settings.get("smooth_sigma",0.5)))
    sampled,projection,xv,yv,zv,vals=prepare_volume(rv,smooth_sigma=sigma,stride=requested)
    stride=effective_volume_stride(np.asarray(rv.volume).shape,requested)
    x,y,theta=(np.asarray(v,dtype=np.float64) for v in (rv.x,rv.y,rv.theta))
    z_min,z_max=float(np.nanmin(theta)),float(np.nanmax(theta)); z_floor=z_min-max((z_max-z_min)*0.15,1e-9)
    p_min,p_max=_finite_percentile(projection,(2,98)); max_int=max(float(np.nanmax(vals)),1e-12)
    max_xy=max(len(x),len(y),1); aspect=dict(x=len(x)/max_xy*2.5,y=len(y)/max_xy*2.5,z=1.8)
    projection_stride=max(1,int(math.ceil(math.sqrt((len(x)*len(y))/MAX_PROJECTION_POINTS))))
    x2d,y2d=np.meshgrid(x[::projection_stride],y[::projection_stride],indexing="ij"); projection_plot=projection[::projection_stride,::projection_stride]; dark=theme=="dark"
    bg="#111316" if dark else "#ffffff"; text="#d9dde3" if dark else "#111111"; grid="#353a42" if dark else "#d8d8d8"; line="#8b929c" if dark else "#222222"; angle="#ffb300" if dark else "#9a6500"
    font=str(settings.get("font_family","Times New Roman")); axis=dict(showbackground=False,gridcolor=grid,zeroline=False,showspikes=False,linecolor=line,linewidth=1.2,ticks="inside",tickcolor=line,title_font=dict(color=text,size=int(settings.get("label_size",12)),family=font),tickfont=dict(color=text,size=int(settings.get("tick_size",10)),family=font))
    fig=go.Figure()
    if bool(settings.get("show_projection",True)):
        fig.add_trace(go.Surface(x=x2d,y=y2d,z=np.full(x2d.shape,z_floor,dtype=np.float32),surfacecolor=projection_plot,colorscale=_projection_scale(str(settings.get("projection_colorscale","Greys")),theme),cmin=p_min,cmax=p_max,showscale=False,hoverinfo="skip",opacity=float(settings.get("projection_opacity",0.75)),contours=dict(x=dict(show=True,color="rgba(130,130,130,0.12)",width=1),y=dict(show=True,color="rgba(130,130,130,0.12)",width=1)),name="Projection"))
    surface_count=max(2,min(25,int(settings.get("surface_count",25))))
    fig.add_trace(go.Volume(x=xv,y=yv,z=zv,value=vals,isomin=max_int*float(settings.get("isomin_fraction",0.16)),isomax=max_int*float(settings.get("isomax_fraction",0.95)),cmin=0,cmax=max_int,opacity=float(settings.get("volume_opacity",0.10)),surface=dict(count=surface_count),colorscale=str(settings.get("volume_colorscale_dark" if dark else "volume_colorscale_light","Magma" if dark else "Blues")),showscale=bool(settings.get("show_colorbar",False)),caps=dict(x_show=False,y_show=False,z_show=False),lighting=dict(ambient=0.95,diffuse=0.25,specular=0.0),name="Intensity volume",hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>θ=%{z:.5g}<br>I=%{value:.5g}<extra></extra>"))
    if peak_catalog is not None and getattr(peak_catalog,"empty",True) is False:
        cat=peak_catalog.copy(); required={"x","y","theta"}
        if required.issubset(cat.columns):
            px0,py0,ptheta=cat["x"].to_numpy(float),cat["y"].to_numpy(float),cat["theta"].to_numpy(float); px,py=rv.roi.to_local(px0,py0) if rv.coord_system=="uv" else (px0,py0)
            inside=rv.roi.contains(px0,py0)&np.isfinite(px)&np.isfinite(py)&np.isfinite(ptheta)
            fig.add_trace(go.Scatter3d(x=np.asarray(px)[inside],y=np.asarray(py)[inside],z=ptheta[inside],mode="markers",name="Пики",marker=dict(size=2,opacity=1,color="#ffb347")))
    x_label,y_label=("Detector X (pixels)","Detector Y (pixels)") if rv.coord_system=="xy" else ("ROI u (pixels)","ROI v (pixels)")
    camera=settings.get("camera") if isinstance(settings.get("camera"),dict) else None
    if not camera:camera=dict(eye=dict(x=1.6,y=-1.9,z=1.25),center=dict(x=0,y=0,z=0),up=dict(x=0,y=0,z=1))
    spatial_bg="#000000" if dark else "#f1f3f5"; angular_bg="#070707" if dark else "#fff7e8"
    z_axis={**axis,"showbackground":True,"backgroundcolor":angular_bg,"gridcolor":"#443311" if dark else "#e5d8b5","showgrid":bool(settings.get("show_grid",True)),"showspikes":True,"spikecolor":angle,"spikethickness":2,"linewidth":2,"linecolor":angle,"title_font":dict(color=angle,size=int(settings.get("label_size",12)),family=font),"tickfont":dict(color=angle,size=int(settings.get("tick_size",10)),family=font),"range":[z_floor,z_max]}
    xaxis={**axis,"showbackground":True,"backgroundcolor":spatial_bg,"gridcolor":"#333333" if dark else "#d5d9de","showgrid":bool(settings.get("show_grid",True)),"showspikes":True,"spikecolor":"#00bcd4" if dark else "#3a8490","spikethickness":1}; yaxis={**xaxis}
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),paper_bgcolor=bg,plot_bgcolor=bg,uirevision="trs-3d",scene=dict(bgcolor=bg,xaxis_title=x_label,yaxis_title=y_label,zaxis_title="Angle θ (°)",aspectmode="manual",aspectratio=aspect,xaxis=xaxis,yaxis=yaxis,zaxis=z_axis,camera=camera,dragmode=str(settings.get("dragmode","orbit")),camera_projection_type=str(settings.get("projection_type","perspective"))),legend=dict(font=dict(family=font,size=10,color=text),bgcolor="rgba(0,0,0,0)",borderwidth=0),showlegend=bool(settings.get("show_legend",False)))
    return fig
