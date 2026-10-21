from __future__ import annotations
import numpy as np
VERSION='dynamic_rocking_lifecycle_v5_exactjax_viz_v1_4'

def _mean_variance_table(r,n_bins=12):
    import pandas as pd
    y=np.asarray(r.data,float)[r.valid];mu=np.asarray(r.fitted_map,float)[r.valid];ok=np.isfinite(y)&np.isfinite(mu)&(mu>0);y=y[ok];mu=mu[ok]
    if len(mu)<8:return pd.DataFrame(columns=['mean_mu','empirical_mse','model_variance','n'])
    n_bins=max(3,min(int(n_bins),max(3,len(mu)//40)));q=np.unique(np.quantile(mu,np.linspace(0,1,n_bins+1)));
    if len(q)<3:q=np.linspace(float(mu.min()),float(mu.max())+1e-12,min(n_bins,6)+1)
    idx=np.clip(np.digitize(mu,q[1:-1],right=False),0,len(q)-2);alpha=float(r.noise_audit.get('alpha') or 0.);rows=[]
    for k in range(len(q)-1):
        m=idx==k
        if not np.any(m):continue
        mm=mu[m];yy=y[m];v=mm if r.config.noise=='poisson' else mm+alpha*mm*mm;rows.append({'mean_mu':float(np.mean(mm)),'empirical_mse':float(np.mean((yy-mm)**2)),'model_variance':float(np.mean(v)),'n':int(m.sum())})
    return pd.DataFrame(rows)


def _activation(g):
    a=g['activation'].to_numpy(float) if 'activation' in g else np.ones(len(g),float);return np.clip(a,0.,1.)

def _branch_colors(J):
    import matplotlib.pyplot as plt
    c=plt.rcParams['axes.prop_cycle'].by_key().get('color',['C0','C1','C2','C3']);return [c[j%len(c)] for j in range(int(J))]

def _display_alpha(a,floor=.14):
    a=np.clip(np.asarray(a,float),0.,1.);f=float(np.clip(floor,0.,1.));return f+(1.-f)*a

def _fade_line(ax,x,y,a,*,lw=1.35,label=None,marker=False,color=None,alpha_floor=.14):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    x=np.asarray(x,float);y=np.asarray(y,float);av=_display_alpha(a,alpha_floor);color=color or ax._get_lines.get_next_color();pts=np.column_stack([x,y]);seg=np.stack([pts[:-1],pts[1:]],axis=1);am=.5*(av[:-1]+av[1:]);rgba=np.tile(to_rgba(color),(len(seg),1));rgba[:,3]=am;lc=LineCollection(seg,colors=rgba,linewidths=lw,label=label);ax.add_collection(lc);ax.autoscale_view()
    if marker:
        c=np.tile(to_rgba(color),(len(x),1));c[:,3]=av;ax.scatter(x,y,s=13,c=c,zorder=3)
    return color

def plot_model_selection(r,figsize=(7.2,4.2)):
    import matplotlib.pyplot as plt
    d=r.model_table; fig,ax=plt.subplots(figsize=figsize,constrained_layout=True); ax.plot(d.J,d.delta_criterion,marker='o'); ax.axhline(0,ls='--',lw=1); ax.set(xlabel='Global branch count J',ylabel=f"Δ{r.config.criterion.upper()}",title='Global lifecycle model evidence'); ax.set_xticks(d.J); ax.grid(alpha=.2); return fig,ax


def plot_selected_slices(r,scan_ids,*,layout='strip',ncols=3,figsize=None,theta_lim=None,legend='figure',component_alpha_floor=.14):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ids=[int(i) for i in scan_ids];n=len(r.time_s);ids=[i if i>=0 else n+i for i in ids]
    if any(i<0 or i>=n for i in ids):raise IndexError('scan_id вне диапазона')
    a0=float(component_alpha_floor)
    if not 0<=a0<=1:raise ValueError('component_alpha_floor должен быть в [0,1]')
    mode=str(layout).lower();N=len(ids)
    if mode in {'strip','column','vertical'}:rows,cols=N,1;sharex=True;default=(10,2.45*N)
    elif mode in {'grid','tile','tiles'}:cols=max(1,min(int(ncols),N));rows=int(np.ceil(N/cols));sharex=False;default=(4.8*cols,3.6*rows)
    else:raise ValueError("layout должен быть 'strip' или 'grid'")
    fig,axs=plt.subplots(rows,cols,figsize=figsize or default,sharex=sharex,squeeze=False,constrained_layout=True);flat=axs.ravel();best=r.candidates[r.selected_J];colors=_branch_colors(r.selected_J)
    for ax,i in zip(flat,ids):
        ax.plot(r.theta,r.data[i],marker='.',ms=3,lw=.65,color='0.25');sm=np.asarray(r.spike_mask[i],bool)
        if np.any(sm):ax.scatter(r.theta[sm],r.data[i,sm],marker='x',s=24,color='0.1')
        ax.plot(r.theta,r.fitted_map[i],lw=1.8,color='k');ax.plot(r.theta,r.background_map[i],lw=1.05,ls=':',color='0.45')
        gg=r.branches[r.branches.scan_id==i].set_index('branch_id')
        for j,col in enumerate(colors):
            a=float(np.clip(gg.loc[j,'activation'],0,1)) if j in gg.index else 0.;alpha=a0+(1-a0)*a
            ax.plot(r.theta,r.background_map[i]+best['components'][i,j],ls='--',lw=1.25,color=col,alpha=alpha)
        ax.set(title=f'scan {i}; t={r.time_s[i]:.1f} s',xlabel=r'$\theta$ (arcsec)' if mode!='strip' else None,ylabel='Counts');ax.grid(alpha=.18)
        if theta_lim is not None:ax.set_xlim(theta_lim)
    for ax in flat[N:]:ax.set_visible(False)
    if mode=='strip':axs[-1,0].set_xlabel(r'$\theta$ (arcsec)')
    handles=[Line2D([0],[0],marker='.',lw=.65,color='0.25',label='raw counts'),Line2D([0],[0],lw=1.8,color='k',label='joint fit'),Line2D([0],[0],lw=1.05,ls=':',color='0.45',label='background')]+[Line2D([0],[0],lw=1.25,ls='--',color=colors[j],label=f'branch {j}') for j in range(r.selected_J)]
    if np.any(np.asarray(r.spike_mask)[ids]):handles.insert(1,Line2D([0],[0],marker='x',ls='None',color='0.1',label='masked spike'))
    if legend=='figure':fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.015),ncol=min(len(handles),4+r.selected_J),frameon=False,fontsize=8)
    elif legend=='each':
        for ax in axs.ravel()[:N]:ax.legend(handles=handles,frameon=False,fontsize=7)
    elif legend not in {None,False,'none'}:raise ValueError("legend: 'figure', 'each' или None")
    return fig,axs

def plot_noise_audit(r,figsize=(11.5,8.0),n_bins=12):
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    fig,axs=plt.subplots(2,2,figsize=figsize,constrained_layout=True);tab=_mean_variance_table(r,n_bins=n_bins)
    axs[0,0].plot(tab.mean_mu,tab.empirical_mse,marker='o',label='Empirical squared residual');axs[0,0].plot(tab.mean_mu,tab.model_variance,marker='o',label='Model variance');axs[0,0].plot(tab.mean_mu,tab.mean_mu,ls='--',lw=1,label='Poisson variance');axs[0,0].set(xlabel='Fitted mean μ',ylabel='Variance scale',title='A  Mean–variance calibration');axs[0,0].legend(frameon=False,fontsize=8);axs[0,0].grid(alpha=.2)
    mu=np.asarray(r.fitted_map)[r.valid];pr=np.asarray(r.residual_map)[r.valid];axs[0,1].scatter(mu,pr,s=5,alpha=.25);axs[0,1].axhline(0,ls='--',lw=1);axs[0,1].set(xlabel='Fitted mean μ',ylabel='Pearson residual',title=f"B  Residual vs mean; dispersion={r.noise_audit['pearson_dispersion']:.2f}");axs[0,1].grid(alpha=.2)
    q=np.asarray(r.rq_residual_map);q=q[np.isfinite(q)];p=(np.arange(1,len(q)+1)-.5)/len(q);theo=norm.ppf(p);obs=np.sort(q);axs[1,0].plot(theo,obs,lw=1);lo=min(theo.min(),obs.min());hi=max(theo.max(),obs.max());axs[1,0].plot([lo,hi],[lo,hi],ls='--',lw=1);axs[1,0].set(xlabel='Normal quantile',ylabel='Randomized quantile residual',title=f"C  RQ-residual Q–Q; mean={r.noise_audit['rq_mean']:.2f}, SD={r.noise_audit['rq_sd']:.2f}");axs[1,0].grid(alpha=.2)
    lim=float(np.nanpercentile(np.abs(r.rq_residual_map),99));im=axs[1,1].pcolormesh(r.time_s,r.theta,r.rq_residual_map.T,shading='auto',vmin=-lim,vmax=lim,rasterized=True);fig.colorbar(im,ax=axs[1,1],pad=.01,label='Randomized quantile residual');axs[1,1].set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title='D  Distribution-aware residual map');return fig,axs

def plot_background_diagnostics(r,scan_ids=(0,-1),figsize=(11.5,7.8),edge_fraction=.10):
    import matplotlib.pyplot as plt
    n=len(r.time_s);ids=[int(i) if int(i)>=0 else n+int(i) for i in scan_ids]
    if any(i<0 or i>=n for i in ids):raise IndexError('scan_id вне диапазона')
    fig,axs=plt.subplots(2,2,figsize=figsize,constrained_layout=True)
    for i in ids:
        axs[0,0].plot(r.theta,r.data[i],marker='.',ms=3,lw=.6,label=f'raw t={r.time_s[i]:.1f}s');axs[0,0].plot(r.theta,r.background_map[i],lw=1.5,label=f'background t={r.time_s[i]:.1f}s')
    axs[0,0].set(xlabel=r'$\theta$ (arcsec)',ylabel='Counts',title='A  Representative fitted backgrounds');axs[0,0].legend(frameon=False,fontsize=8);axs[0,0].grid(alpha=.2)
    bgfrac=np.sum(r.background_map,axis=1)/np.maximum(np.sum(r.fitted_map,axis=1),1e-12);axs[0,1].plot(r.time_s,bgfrac,marker='.');axs[0,1].set(xlabel='Time (s)',ylabel='Background / fitted total',title='B  Background fraction');axs[0,1].grid(alpha=.2)
    edge=max(2,int(round(len(r.theta)*float(edge_fraction))));em=np.r_[0:edge,len(r.theta)-edge:len(r.theta)];er=np.nanmean(np.asarray(r.residual_map)[:,em],axis=1);axs[1,0].plot(r.time_s,er,marker='.');axs[1,0].axhline(0,ls='--',lw=1);axs[1,0].set(xlabel='Time (s)',ylabel='Mean Pearson residual',title='C  Edge residual bias');axs[1,0].grid(alpha=.2)
    im=axs[1,1].pcolormesh(r.time_s,r.theta,r.background_map.T,shading='auto',rasterized=True);fig.colorbar(im,ax=axs[1,1],pad=.01,label='Background counts');axs[1,1].set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title=r'D  Background surface $B(t,\theta)$');return fig,axs

def plot_publication_atlas(r,*,theta_lim=None,trajectory_alpha_floor=.14,figsize=(12,10.5)):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    t,th=r.time_s,r.theta;colors=_branch_colors(r.selected_J);fig,axs=plt.subplots(4,1,figsize=figsize,sharex=True,gridspec_kw={'height_ratios':[3.2,1.25,1.25,2.]},constrained_layout=True);im=axs[0].pcolormesh(t,th,np.arcsinh(np.maximum(r.data.T,0)),shading='auto',rasterized=True);fig.colorbar(im,ax=axs[0],pad=.01,label='asinh(counts)')
    for bid,g in r.branches.groupby('branch_id'):
        g=g.sort_values('time_s');j=int(bid);a=_activation(g);col=colors[j];_fade_line(axs[0],g.time_s,g.theta0,a,lw=1.35,label=f'branch {j}',color=col,alpha_floor=trajectory_alpha_floor);rgba=np.tile(to_rgba(col),(len(g),1));rgba[:,3]=_display_alpha(a,trajectory_alpha_floor);axs[0].scatter(g.time_s,g.theta0,s=7+45*g.area_fraction,c=rgba);axs[1].plot(g.time_s,g.area_fraction,marker='.',ms=4,lw=1.2,color=col);_fade_line(axs[2],g.time_s,g.H,a,lw=1.2,marker=True,color=col,alpha_floor=trajectory_alpha_floor)
    axs[0].set(ylabel=r'$\theta$ (arcsec)',title=f'Lifecycle-aware joint NB decomposition, J={r.selected_J}');axs[0].legend(frameon=False,ncol=max(1,min(r.selected_J,4)),fontsize=8);axs[1].set(ylabel='Effective area fraction',ylim=(-.03,1.03),title='Redistribution of diffracting intensity');axs[2].set(ylabel='FWHM',title='Branch widths; opacity reflects fitted activation')
    lim=float(np.nanpercentile(np.abs(r.rq_residual_map),99));im2=axs[3].pcolormesh(t,th,r.rq_residual_map.T,shading='auto',vmin=-lim,vmax=lim,rasterized=True);fig.colorbar(im2,ax=axs[3],pad=.01,label='RQ residual');axs[3].set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title='Distribution-aware residual map')
    for ax in axs[1:3]:ax.grid(alpha=.2)
    if theta_lim is not None:axs[0].set_ylim(theta_lim);axs[3].set_ylim(theta_lim)
    return fig,axs

def plot_branch_mechanisms(r,*,trajectory_alpha_floor=.14,figsize=(11,9.5)):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=figsize,sharex=True,constrained_layout=True);colors=_branch_colors(r.selected_J)
    for bid,g in r.branches.groupby('branch_id'):
        g=g.sort_values('time_s');j=int(bid);a=_activation(g);col=colors[j];_fade_line(axs[0],g.time_s,g.theta0,a,lw=1.35,label=f'branch {j}',marker=True,color=col,alpha_floor=trajectory_alpha_floor);axs[1].plot(g.time_s,g.area_fraction,marker='.',color=col);_fade_line(axs[2],g.time_s,g.H,a,lw=1.35,marker=True,color=col,alpha_floor=trajectory_alpha_floor);axs[3].plot(g.time_s,g.conditional_delta_deviance,marker='.',color=col)
    axs[0].set(ylabel=r'$\theta_0$',title='Angular trajectories; opacity reflects fitted activation');axs[0].legend(frameon=False,ncol=max(1,min(r.selected_J,4)));axs[1].set(ylabel='Effective area fraction',title='Integrated-intensity redistribution');axs[2].set(ylabel='FWHM',title='Branch-specific broadening');axs[3].set(ylabel='Conditional Δdeviance',xlabel='Time (s)',title='Descriptive branch support (not a p-value)')
    for ax in axs:ax.grid(alpha=.2)
    return fig,axs


def _overlay_branch_trajectories(ax,r,mode,self_id=None,alpha_floor=.14,lw=1.35):
    """
    trajectories='self'   # только траектория соответствующей карты
    trajectories='all'    # все траектории на каждой карте
    trajectories=None     # вообще без траекторий
    """
    if mode in {None,False,'none'}:return
    if mode not in {'self','all'}:raise ValueError("trajectories: 'self', 'all' или None")
    colors=_branch_colors(r.selected_J); ids=[int(self_id)] if mode=='self' else range(r.selected_J)
    for j in ids:
        g=r.branches[r.branches.branch_id==j].sort_values('time_s');x=g.time_s.to_numpy(float);y=g.theta0.to_numpy(float);a=np.clip(g.activation.to_numpy(float),0,1);alpha=float(alpha_floor)+(1-float(alpha_floor))*a
        for k in range(len(x)-1):ax.plot(x[k:k+2],y[k:k+2],color=colors[j],lw=lw,alpha=float((alpha[k]+alpha[k+1])*.5),solid_capstyle='round',zorder=5)
        

def plot_component_surface_atlas(r,*,theta_lim=None,figsize=None,scale='shared',trajectories=None,trajectory_alpha_floor=.14,trajectory_lw=1.35):
    import matplotlib.pyplot as plt
    C=np.asarray(r.candidates[r.selected_J]['components'],float);J=r.selected_J
    if scale not in {'shared','individual'}:raise ValueError("scale: 'shared' или 'individual'")
    if not 0<=trajectory_alpha_floor<=1:raise ValueError('trajectory_alpha_floor должен быть в [0,1]')
    fig,axs=plt.subplots(1,J,figsize=figsize or (4.4*J,4.1),sharex=True,sharey=True,constrained_layout=True);axs=np.atleast_1d(axs);vmax=float(np.nanpercentile(C,99.5)) if scale=='shared' else None
    for j,ax in enumerate(axs):
        z=C[:,j,:].T;im=ax.pcolormesh(r.time_s,r.theta,z,shading='auto',vmin=0,vmax=vmax,rasterized=True)
        _overlay_branch_trajectories(ax,r,trajectories,self_id=j,alpha_floor=trajectory_alpha_floor,lw=trajectory_lw)
        ax.set(xlabel='Time (s)',title=f'Branch {j}: reconstructed $I_j(t,\\theta)$')
        if j==0:ax.set_ylabel(r'$\theta$ (arcsec)')
        if theta_lim is not None:ax.set_ylim(theta_lim)
    fig.colorbar(im,ax=axs.tolist(),pad=.01,label='Component counts');return fig,axs
    

def plot_component_ridge_atlas(r,*,step=None,figsize=None):
    import matplotlib.pyplot as plt
    C=np.asarray(r.candidates[r.selected_J]['components'],float); J=r.selected_J; fig,axs=plt.subplots(1,J,figsize=figsize or (4.5*J,5.0),sharex=True,constrained_layout=True); axs=np.atleast_1d(axs); mx=max(float(np.nanpercentile(C,99)),1.); dy=.8*mx if step is None else float(step)
    for j,ax in enumerate(axs):
        for i in range(len(r.time_s)): ax.plot(r.theta,C[i,j]+i*dy,lw=.8)
        ax.set(xlabel=r'$\theta$ (arcsec)',title=f'Branch {j}: time-stacked profiles')
        if j==0: ax.set_ylabel('Time stack + intensity')
    return fig,axs

def plot_component_mountains_3d(r,*,scan_step=2,opacity=.40,show_slices=True,show_centers=True,show_floor_contours=True,theta_lim=None,title=None):
    """Интерактивные branch-resolved хребты в общей физической геометрии."""
    import plotly.graph_objects as go
    C=np.asarray(r.candidates[r.selected_J]['components'],float);t=np.asarray(r.time_s,float);th=np.asarray(r.theta,float);T,TH=np.meshgrid(t,th,indexing='ij');colors=['#0072B2','#D55E00','#009E73','#CC79A7','#56B4E9','#E69F00'];fig=go.Figure();zmax=max(float(np.nanmax(C)),1.)
    for j in range(r.selected_J):
        col=colors[j%len(colors)];z=C[:,j,:];fig.add_trace(go.Surface(x=T,y=TH,z=z,colorscale=[[0,col],[1,col]],opacity=float(opacity),showscale=False,name=f'branch {j}',showlegend=True,hovertemplate=f'branch {j}<br>t=%{{x:.1f}} s<br>θ=%{{y:.2f}} arcsec<br>I=%{{z:.3g}}<extra></extra>',lighting=dict(ambient=.78,diffuse=.58,specular=.08,roughness=.9,fresnel=.02),contours_z=dict(show=bool(show_floor_contours),usecolormap=False,highlight=False,project_z=True,color=col,width=1)))
        if show_slices:
            for i in range(0,len(t),max(1,int(scan_step))):fig.add_trace(go.Scatter3d(x=np.full(len(th),t[i]),y=th,z=z[i],mode='lines',line=dict(color=col,width=2.2),opacity=.95,showlegend=False,hoverinfo='skip'))
        if show_centers:
            g=r.branches[r.branches.branch_id==j].sort_values('scan_id');a=_display_alpha(_activation(g),.12);yy=g.theta0.to_numpy(float);zz=np.max(z,axis=1)
            for i in range(len(t)-1):fig.add_trace(go.Scatter3d(x=t[i:i+2],y=yy[i:i+2],z=zz[i:i+2],mode='lines',line=dict(color=col,width=5),opacity=float(.5*(a[i]+a[i+1])),showlegend=False,hovertemplate=f'branch {j} ridge<br>t=%{{x:.1f}} s<br>θ₀=%{{y:.2f}}<extra></extra>'))
    fig.update_layout(template='plotly_white',title=title or f'Branch-resolved rocking-curve landscape, J={r.selected_J}',scene=dict(xaxis_title='Time (s)',yaxis_title='θ (arcsec)',zaxis_title='Component intensity (counts)',zaxis=dict(range=[0,1.04*zmax]),aspectmode='manual',aspectratio=dict(x=1.7,y=1.2,z=.78),camera=dict(eye=dict(x=1.55,y=-1.75,z=1.08))),legend=dict(x=.01,y=.99,bgcolor='rgba(255,255,255,.74)'),margin=dict(l=0,r=0,b=0,t=55))
    if theta_lim is not None:fig.update_scenes(yaxis_range=list(theta_lim))
    return fig

def plot_reconstruction_landscape_3d(r,*,scan_step=2,component_opacity=.38,show_total=False,show_background=False,theta_lim=None):
    """Общая 3D-сцена компонент; total/background добавляются только по запросу."""
    import plotly.graph_objects as go
    fig=plot_component_mountains_3d(r,scan_step=scan_step,opacity=component_opacity,show_slices=True,show_centers=True,theta_lim=theta_lim,title=f'Joint reconstructed diffraction landscape, J={r.selected_J}');t=np.asarray(r.time_s);th=np.asarray(r.theta);T,TH=np.meshgrid(t,th,indexing='ij')
    if show_total:fig.add_trace(go.Surface(x=T,y=TH,z=np.asarray(r.fitted_map),colorscale=[[0,'#555555'],[1,'#555555']],opacity=.10,showscale=False,name='total fit',showlegend=True,hoverinfo='skip'))
    if show_background:fig.add_trace(go.Surface(x=T,y=TH,z=np.asarray(r.background_map),colorscale=[[0,'#999999'],[1,'#999999']],opacity=.16,showscale=False,name='background',showlegend=True,hoverinfo='skip'))
    return fig

def save_plotly_html(fig,path,*,include_plotlyjs=True):fig.write_html(str(path),include_plotlyjs=include_plotlyjs,full_html=True);return str(path)

