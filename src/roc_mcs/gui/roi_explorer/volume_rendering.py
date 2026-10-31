"""Подготовка Plotly 3D-сцены с ограничением памяти и числа узлов WebGL."""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from roc_mcs.roi.roi_types import ROIVolume
from roc_mcs.gui.roi_explorer.colormap_catalog import plotly_colorscale

LIGHT_PROJECTION_SCALE = "Greys"
DARK_PROJECTION_SCALE = [[0.0, "#f7f7f7"], [0.55, "#dedede"], [1.0, "#ffffff"]]
VOLUME_SCALES = ("Blues", "Magma", "Viridis", "Cividis", "Inferno", "Plasma", "Hot", "Electric")
PROJECTION_SCALES = ("Slate", "Greys", "Cividis", "Viridis")
MAX_RENDER_POINTS = 95_000
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


def effective_extraction_stride(roi, x_coords, y_coords, n_theta, requested=1, max_voxels=MAX_EXTRACTION_VOXELS):
    """Сохраняет исходное разрешение ROI; прореживает вход только при превышении лимита памяти."""
    shape,_=estimate_roi_voxels(roi,x_coords,y_coords,n_theta); step=max(1,int(requested)); limit=max(100_000,int(max_voxels))
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
    if str(name).lower() in ("slate", "greys"):
        return "Greys_r" if theme == "dark" else "Greys"
    return name


def _smooth_masked(values, sigma):
    """Нормированное сглаживание: NaN не превращаются в чёрные провалы."""
    a=np.asarray(values,dtype=np.float32); good=np.isfinite(a)
    if not good.any(): return np.full(a.shape,np.nan,dtype=np.float32)
    if float(sigma)<=0: return a.copy()
    sig=(float(sigma),)*a.ndim
    if a.ndim==3: sig=(float(sigma),float(sigma),min(0.2,float(sigma)))
    num=gaussian_filter(np.where(good,a,0.0),sigma=sig,output=np.float32)
    den=gaussian_filter(good.astype(np.float32),sigma=sig,output=np.float32)
    return np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-8)


def prepare_volume(rv: ROIVolume, *, smooth_sigma=0.5, stride=2, max_points=MAX_RENDER_POINTS):
    """Полная XY-проекция + сглаженный прореженный 3D-объём; исходный rv не меняется."""
    if not isinstance(rv,ROIVolume): raise TypeError("Ожидается ROIVolume")
    source=np.asarray(rv.volume); x,y,theta=(np.asarray(v,dtype=np.float64).reshape(-1) for v in (rv.x,rv.y,rv.theta))
    if source.ndim!=3 or source.shape!=(len(x),len(y),len(theta)): raise ValueError("Форма ROIVolume не согласована с осями")
    if rv.coord_system not in {"xy","uv"}: raise ValueError(f"Неизвестная система координат: {rv.coord_system}")
    sigma=float(smooth_sigma)
    if not np.isfinite(sigma) or sigma<0: raise ValueError("smooth_sigma должен быть конечным и неотрицательным")
    finite=np.isfinite(source)
    if not finite.any(): raise ValueError("В объёме нет конечных значений")
    with np.errstate(invalid="ignore",over="ignore"):
        projection=np.nansum(source,axis=2,dtype=np.float64).astype(np.float32)
    projection=_smooth_masked(projection,sigma)
    step=effective_volume_stride(source.shape,stride,max_points)
    sampled=np.asarray(source[::step,::step,::step],dtype=np.float32)
    sampled=_smooth_masked(sampled,sigma)
    xs=x[::step]; ys=y[::step]; ts=theta[::step]
    if xs.size>1 and xs[0]>xs[-1]: xs=xs[::-1]; sampled=sampled[::-1,:,:]
    if ys.size>1 and ys[0]>ys[-1]: ys=ys[::-1]; sampled=sampled[:,::-1,:]
    if ts.size>1 and ts[0]>ts[-1]: ts=ts[::-1]; sampled=sampled[:,:,::-1]
    if x.size>1 and x[0]>x[-1]: x=x[::-1]; projection=projection[::-1,:]
    if y.size>1 and y[0]>y[-1]: y=y[::-1]; projection=projection[:,::-1]
    mask=np.isfinite(sampled)
    if not mask.any(): raise ValueError("В прореженном объёме нет конечных значений")
    vals=np.asarray(sampled[mask],dtype=np.float32)
    if float(np.nanmax(vals))<=0: raise ValueError("В объёме нет положительных конечных интенсивностей")
    xv=np.broadcast_to(xs.astype(np.float32)[:,None,None],sampled.shape)[mask]
    yv=np.broadcast_to(ys.astype(np.float32)[None,:,None],sampled.shape)[mask]
    zv=np.broadcast_to(ts.astype(np.float32)[None,None,:],sampled.shape)[mask]
    return sampled,projection,xv,yv,zv,vals


def _projection_extent(projection, x, y, max_pixels=MAX_PROJECTION_POINTS):
    projection=np.asarray(projection,dtype=np.float32); x=np.asarray(x,dtype=np.float64).reshape(-1); y=np.asarray(y,dtype=np.float64).reshape(-1)
    if projection.shape!=(x.size,y.size): raise ValueError(f"XY-проекция {projection.shape} не совпадает с осями {(x.size,y.size)}")
    step=max(1,int(math.ceil(math.sqrt((x.size*y.size)/max(1,int(max_pixels))))))
    if x.size>1 and x[0]>x[-1]: x=x[::-1]; projection=projection[::-1,:]
    if y.size>1 and y[0]>y[-1]: y=y[::-1]; projection=projection[:,::-1]
    px=x[::step].astype(np.float32); py=y[::step].astype(np.float32); z=projection[::step,::step]
    return px,py,z


def _regrid_rotated_volume_to_detector(rv: ROIVolume, roi):
    """Regrid a rotated local u/v volume onto a regular detector x/y grid for Plotly Volume.

    Plotly's volume trace is most reliable on a rectilinear Cartesian lattice.  A
    rotated ROI is extracted on a regular local (u, v, theta) lattice; converting
    every lattice point to detector x/y produces a skew lattice and can make the
    WebGL volume disappear.  This function changes only the display lattice: the
    extracted ROI values and ROI geometry stay untouched.
    """
    if rv.coord_system != "uv":
        return rv
    source=np.asarray(rv.volume,dtype=np.float32)
    u=np.asarray(rv.x,dtype=np.float64).reshape(-1); v=np.asarray(rv.y,dtype=np.float64).reshape(-1)
    if source.shape[:2] != (u.size,v.size) or u.size < 2 or v.size < 2:
        return rv
    corners=np.asarray(roi.corners(),dtype=np.float64).reshape(-1,2)
    if corners.shape[0] < 3:
        return rv
    du=float(np.median(np.abs(np.diff(u)))); dv=float(np.median(np.abs(np.diff(v))))
    step=max(1e-12,min(du,dv))
    xmin,xmax=float(np.min(corners[:,0])),float(np.max(corners[:,0])); ymin,ymax=float(np.min(corners[:,1])),float(np.max(corners[:,1]))
    nx=max(2,int(np.ceil((xmax-xmin)/step))+1); ny=max(2,int(np.ceil((ymax-ymin)/step))+1)
    # Keep the display regrid bounded even for pathological settings.
    max_xy=max(4, int(math.ceil(math.sqrt(MAX_EXTRACTION_VOXELS/max(1,source.shape[2])))))
    if max(nx,ny)>max_xy:
        scale=max(nx,ny)/max_xy; nx=max(2,int(math.ceil(nx/scale))); ny=max(2,int(math.ceil(ny/scale)))
    x=np.linspace(xmin,xmax,nx,dtype=np.float64); y=np.linspace(ymin,ymax,ny,dtype=np.float64)
    X,Y=np.meshgrid(x,y,indexing="ij"); U,V=roi.to_local(X,Y)
    iu=np.interp(U,u,np.arange(u.size,dtype=float),left=np.nan,right=np.nan); iv=np.interp(V,v,np.arange(v.size,dtype=float),left=np.nan,right=np.nan)
    inside=np.isfinite(iu)&np.isfinite(iv)&roi.contains(X,Y)
    coords=[np.where(inside,iu,-1.0),np.where(inside,iv,-1.0)]
    # IMPORTANT: Plotly Volume needs the complete Cartesian lattice.  Do not
    # remove the bbox corners with NaN: prepare_volume() flattens only finite
    # samples and that turns a rotated ROI into an irregular point cloud.
    # A finite zero baseline is safe because intensity isosurfaces use a
    # strictly positive isomin.  It preserves the rectangular x*y*theta grid
    # while keeping everything outside the physical ROI invisible.
    out=np.zeros((nx,ny,source.shape[2]),dtype=np.float32)
    for k in range(source.shape[2]):
        layer=source[:,:,k]; good=np.isfinite(layer)
        num=map_coordinates(np.where(good,layer,0.0),coords,order=1,mode="constant",cval=0.0)
        den=map_coordinates(good.astype(np.float32),coords,order=1,mode="constant",cval=0.0)
        valid=(den>1e-8)&inside
        out[:,:,k]=np.divide(num,den,out=np.zeros((nx,ny),dtype=np.float32),where=valid)
    return ROIVolume(x,y,np.asarray(rv.theta,dtype=np.float64).copy(),out,"xy",rv.roi)


def build_volume_figure(rv: ROIVolume, settings: dict, theme="light", *, peak_catalog=None,
                        full_projection=None, detector_x=None, detector_y=None, overlay_roi=None):
    """Стиль лабораторного Plotly-builder; опционально показывает ROI-объём на полном XY-детекторе."""
    import plotly.graph_objects as go
    theme="dark" if str(theme).lower()=="dark" else "light"
    requested=max(2,int(settings.get("volume_stride",2))); sigma=max(0.0,float(settings.get("smooth_sigma",0.5)))
    overlay=full_projection is not None and detector_x is not None and detector_y is not None and overlay_roi is not None
    display_rv=_regrid_rotated_volume_to_detector(rv,overlay_roi) if overlay and rv.coord_system=="uv" else rv
    sampled,projection,xv,yv,zv,vals=prepare_volume(display_rv,smooth_sigma=sigma,stride=requested,max_points=int(settings.get("max_render_points",MAX_RENDER_POINTS)))
    x=np.asarray(display_rv.x,dtype=np.float64); y=np.asarray(display_rv.y,dtype=np.float64); theta=np.asarray(display_rv.theta,dtype=np.float64)
    if overlay:
        plane=_smooth_masked(np.asarray(full_projection,dtype=np.float32),sigma)
        px=np.asarray(detector_x,dtype=np.float64); py=np.asarray(detector_y,dtype=np.float64)
        px_plot,py_plot,proj=_projection_extent(plane,px,py)
        x_label,y_label="Detector X (pixels)","Detector Y (pixels)"
        extent_x,extent_y=px,py
    else:
        px_plot,py_plot,proj=_projection_extent(projection,x,y)
        x_label,y_label=("Detector X (pixels)","Detector Y (pixels)") if rv.coord_system=="xy" else ("ROI u (pixels)","ROI v (pixels)")
        extent_x,extent_y=x,y
    proj_finite=proj[np.isfinite(proj)]
    if not proj_finite.size: raise ValueError("В XY-проекции нет конечных значений")
    pmin,pmax=np.percentile(proj_finite,[2,98]); pmin=float(pmin); pmax=float(pmax)
    if pmin>=pmax:pmax=pmin+max(abs(pmin)*1e-9,1e-12)
    threshold_values=vals[np.isfinite(vals)&(vals>0)]
    if not threshold_values.size: raise ValueError("В объёме нет положительных конечных интенсивностей")
    robust=float(np.percentile(threshold_values,99.9)); robust=max(robust,float(np.max(threshold_values))*1e-9,1e-12)
    z_min,z_max=float(np.nanmin(theta)),float(np.nanmax(theta)); z_floor=z_min-max((z_max-z_min)*0.15,1e-9)
    max_xy=max(len(extent_x),len(extent_y),1)
    aspect=dict(x=len(extent_x)/max_xy*2.2,y=len(extent_y)/max_xy*2.2,z=1.8)
    dark=theme=="dark"; bg="#111316" if dark else "#ffffff"; text="#e1e4e8" if dark else "#111111"
    grid="#343941" if dark else "#e0e0e0"; axisline="#aeb5be" if dark else "#222222"
    font=str(settings.get("font_family","Times New Roman"))
    axis=dict(showbackground=False,gridcolor=grid,zeroline=False,showspikes=False,linecolor=axisline,linewidth=1.2,ticks="inside",ticklen=float(settings.get("axis_tick_length",3.0)),tickwidth=1,tickcolor=axisline,
        title_font=dict(color=text,size=int(settings.get("label_size",15)),family=font),tickfont=dict(color=text,size=int(settings.get("tick_size",12)),family=font))
    fig=go.Figure()
    if bool(settings.get("show_projection",True)):
        # Surface: строки z соответствуют y, столбцы — x. Исходная проекция
        # хранится как (x, y), поэтому для 1D-осей передаём транспонированную матрицу.
        surface_shape=(len(py_plot),len(px_plot)); surface_z=np.full(surface_shape,z_floor,dtype=np.float32); surface_color=np.asarray(proj.T,dtype=np.float32)
        fig.add_trace(go.Surface(x=px_plot,y=py_plot,z=surface_z,surfacecolor=surface_color,
            colorscale=_projection_scale(str(settings.get("projection_colorscale","Greys")),theme),cmin=pmin,cmax=pmax,showscale=False,hoverinfo="skip",opacity=float(settings.get("projection_opacity",0.75)),connectgaps=False,
            contours=dict(x=dict(show=True,color="rgba(110,110,110,0.16)",width=1),y=dict(show=True,color="rgba(110,110,110,0.16)",width=1)),name="Интегральная проекция"))
    surface_count=max(2,min(25,int(settings.get("surface_count",25))))
    fig.add_trace(go.Volume(x=xv,y=yv,z=zv,value=vals,isomin=robust*float(settings.get("isomin_fraction",0.16)),isomax=robust*float(settings.get("isomax_fraction",0.95)),
        opacity=float(settings.get("volume_opacity",0.10)),surface=dict(count=surface_count),colorscale=plotly_colorscale(str(settings.get("volume_colormap", settings.get("volume_colorscale_dark" if dark else "volume_colorscale_light", "magma" if dark else "Blues")))),
        showscale=bool(settings.get("show_colorbar",False)),caps=dict(x_show=False,y_show=False,z_show=False),lighting=dict(ambient=0.95,diffuse=0.25,specular=0.0),name="Интенсивность",
        hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>θ=%{z:.5g}°<br>I=%{value:.5g}<extra></extra>"))
    if overlay and bool(settings.get("show_projection",True)):
        try:
            corners=np.asarray(overlay_roi.corners(),dtype=np.float64).reshape(-1,2)
            if corners.shape[0]>=3:
                corners=np.vstack((corners,corners[0])); raw_color=getattr(overlay_roi,"color","#ffb347"); color=raw_color.name() if callable(getattr(raw_color,"name",None)) else str(raw_color); color=color.strip() or "#ffb347"
                try:
                    from plotly.colors import validate_colors
                    validate_colors([color])
                except Exception:
                    color="#ffb347"
                fig.add_trace(go.Scatter3d(x=corners[:,0],y=corners[:,1],z=np.full(len(corners),z_floor,dtype=np.float32),mode="lines",name="Граница ROI",
                    line=dict(color=color,width=4),hoverinfo="skip",showlegend=False))
        except (AttributeError,TypeError,ValueError):
            pass
    if peak_catalog is not None and getattr(peak_catalog,"empty",True) is False:
        cat=peak_catalog.copy(); required={"x","y","theta"}
        if required.issubset(cat.columns):
            px0,py0,ptheta=cat["x"].to_numpy(float),cat["y"].to_numpy(float),cat["theta"].to_numpy(float)
            px,py=rv.roi.to_local(px0,py0) if rv.coord_system=="uv" and rv.roi is not None else (px0,py0)
            inside=rv.roi.contains(px0,py0) if rv.roi is not None else np.ones(px0.shape,dtype=bool)
            inside=np.asarray(inside)&np.isfinite(px)&np.isfinite(py)&np.isfinite(ptheta)
            fig.add_trace(go.Scatter3d(x=np.asarray(px)[inside],y=np.asarray(py)[inside],z=ptheta[inside],mode="markers",name="Пики",marker=dict(size=2,opacity=1,color="#ffb347")))
    camera=settings.get("camera") if isinstance(settings.get("camera"),dict) else None
    if not camera:camera=dict(eye=dict(x=1.6,y=-1.9,z=1.25),center=dict(x=0,y=0,z=0),up=dict(x=0,y=0,z=1))
    # Сохраняем визуальную схему лабораторного Plotly-эталона: прозрачные фоны осей,
    # нейтральная сетка и одинаковая типографика без декоративных цветных плоскостей.
    z_axis={**axis,"range":[z_floor,z_max]}
    fig.update_layout(margin=dict(l=0,r=0,t=8,b=0),paper_bgcolor=bg,plot_bgcolor=bg,font=dict(family=font,color=text,size=12),
        uirevision="trs-3d",scene=dict(bgcolor=bg,xaxis_title=x_label,yaxis_title=y_label,zaxis_title="Angle θ (°)",aspectmode="manual",aspectratio=aspect,
        xaxis=axis,yaxis=axis,zaxis=z_axis,camera=camera),showlegend=bool(settings.get("show_legend",False)))
    return fig
