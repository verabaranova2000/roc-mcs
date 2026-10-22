def plot_background_only_diagnostics(b,cii_result=None,scan_ids=(0,-1),figsize=(12,10)):
    import numpy as np
    import matplotlib.pyplot as plt
    n=len(b.time_s);ids=[int(i) if int(i)>=0 else n+int(i) for i in scan_ids]
    if any(i<0 or i>=n for i in ids):raise IndexError('scan_id вне диапазона')
    fig,axs=plt.subplots(3,2,figsize=figsize,constrained_layout=True)
    for i in ids:
        axs[0,0].plot(b.theta,b.data[i],marker='.',ms=3,lw=.6,label=f'raw t={b.time_s[i]:.1f}s');axs[0,0].plot(b.theta,b.background_map[i],lw=1.5,label=f'background t={b.time_s[i]:.1f}s');axs[0,1].plot(b.theta,b.corrected_map[i],lw=1,label=f't={b.time_s[i]:.1f}s')
    axs[0,0].set(xlabel=r'$\theta$ (arcsec)',ylabel='Counts',title='A  Raw data and background estimate');axs[0,0].legend(frameon=False,fontsize=8);axs[0,0].grid(alpha=.2)
    axs[0,1].axhline(0,ls='--',lw=1);axs[0,1].set(xlabel=r'$\theta$ (arcsec)',ylabel='Raw − background',title='B  Unclipped corrected slices');axs[0,1].grid(alpha=.2)
    im=axs[1,0].pcolormesh(b.time_s,b.theta,b.weight_map.T,shading='auto',vmin=0,vmax=1,rasterized=True);fig.colorbar(im,ax=axs[1,0],pad=.01,label='Background support');axs[1,0].set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title='C  Points used to fit background')
    im=axs[1,1].pcolormesh(b.time_s,b.theta,b.background_map.T,shading='auto',rasterized=True);fig.colorbar(im,ax=axs[1,1],pad=.01,label='Background counts');axs[1,1].set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title=r'D  Background surface $B(t,\theta)$')
    er=np.array([np.nanmean(b.standardized_residual_map[i][b.weight_map[i]>.5]) for i in range(n)]);axs[2,0].plot(b.time_s,er,marker='.');axs[2,0].axhline(0,ls='--',lw=1);axs[2,0].set(xlabel='Time (s)',ylabel='Mean standardized residual',title='E  Residual bias on fitted background support');axs[2,0].grid(alpha=.2)
    if cii_result is not None:
        for i in ids:axs[2,1].plot(b.theta,cii_result['cii_diagnostic'][i],lw=1.3,label=f't={b.time_s[i]:.1f}s')
        axs[2,1].axhline(0,ls=':',lw=.8);axs[2,1].axhline(1,ls=':',lw=.8);axs[2,1].set(xlabel=r'$\theta$ (arcsec)',ylabel='Cumulative signed residual / positive area',title='F  CII background diagnostic (unclipped)');axs[2,1].legend(frameon=False,fontsize=8);axs[2,1].grid(alpha=.2)
    else:axs[2,1].axis('off')
    m=b.metadata;fig.suptitle(f"Background-support count fit; success={b.converged}; iter={b.n_iter}; support={m.get('support_fraction',np.nan):.1%}; support residual mean/sd={m.get('support_residual_mean',np.nan):.2f}/{m.get('support_residual_sd',np.nan):.2f}",fontsize=10);return fig,axs



# ====================== НОВОЕ =============================

# def plot_background_support_map(b,*,theta_lim=None,figsize=(10.5,5.2)):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     fig,ax=plt.subplots(figsize=figsize,constrained_layout=True);z=np.arcsinh(np.maximum(b.data,0));im=ax.pcolormesh(b.time_s,b.theta,z.T,shading='auto',rasterized=True);m=np.asarray(b.weight_map>0.5,float);ax.contourf(b.time_s,b.theta,m.T,levels=[.5,1.5],alpha=.18);ax.contour(b.time_s,b.theta,m.T,levels=[.5],linewidths=.8);fig.colorbar(im,ax=ax,pad=.01,label='asinh(counts)');ax.set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title='Automatic background support on raw rocking map')
#     if theta_lim is not None:ax.set_ylim(theta_lim)
#     return fig,ax

# def plot_background_support_slices(b,scan_ids,*,ncols=2,theta_lim=None,figsize=None):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     ids=[int(i) if int(i)>=0 else len(b.time_s)+int(i) for i in scan_ids]
#     if any(i<0 or i>=len(b.time_s) for i in ids):raise IndexError('scan_id вне диапазона')
#     n=len(ids);cols=max(1,min(int(ncols),n));rows=int(np.ceil(n/cols));fig,axs=plt.subplots(rows,cols,figsize=figsize or (5.3*cols,3.3*rows),squeeze=False,constrained_layout=True)
#     for ax,i in zip(axs.ravel(),ids):
#         m=b.weight_map[i]>.5;ax.plot(b.theta,b.data[i],marker='.',ms=2.5,lw=.65,label='raw');ax.plot(b.theta,b.background_map[i],lw=1.45,label='background');ax.scatter(b.theta[m],b.data[i,m],s=9,alpha=.45,label='background support');ax.plot(b.theta,b.corrected_map[i],lw=.9,alpha=.8,label='raw − background');ax.axhline(0,lw=.7,ls=':');ax.set(title=f'scan {i}; t={b.time_s[i]:.1f} s',xlabel=r'$\theta$ (arcsec)',ylabel='Counts');ax.grid(alpha=.18)
#         if theta_lim is not None:ax.set_xlim(theta_lim)
#     for ax in axs.ravel()[n:]:ax.set_visible(False)
#     axs.ravel()[0].legend(frameon=False,fontsize=8);return fig,axs


def _support_masks(b,autos=None):
    import numpy as np
    support=np.asarray(autos.support_mask,bool) if autos is not None else np.asarray(b.weight_map>0.5,bool)
    ambiguous=np.asarray(getattr(autos,'ambiguous_mask',np.zeros_like(support)),bool) if autos is not None else np.zeros_like(support)
    return support,ambiguous


def plot_background_support_map(b,*,autos=None,theta_lim=None,figsize=(10.5,5.2)):
    import numpy as np
    import matplotlib.pyplot as plt
    support,ambiguous=_support_masks(b,autos);fig,ax=plt.subplots(figsize=figsize,constrained_layout=True);z=np.arcsinh(np.maximum(b.data,0));im=ax.pcolormesh(b.time_s,b.theta,z.T,shading='auto',rasterized=True);ax.contourf(b.time_s,b.theta,support.T.astype(float),levels=[.5,1.5],alpha=.18);ax.contour(b.time_s,b.theta,support.T.astype(float),levels=[.5],linewidths=.8)
    if np.any(ambiguous):ax.contour(b.time_s,b.theta,ambiguous.T.astype(float),levels=[.5],linewidths=.7,linestyles='--')
    fig.colorbar(im,ax=ax,pad=.01,label='asinh(counts)');ax.set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title='Automatic background support on raw rocking map')
    if theta_lim is not None:ax.set_ylim(theta_lim)
    return fig,ax


def plot_background_support_slices(b,scan_ids,*,autos=None,ncols=2,theta_lim=None,figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    support,ambiguous=_support_masks(b,autos);ids=[int(i) if int(i)>=0 else len(b.time_s)+int(i) for i in scan_ids]
    if any(i<0 or i>=len(b.time_s) for i in ids):raise IndexError('scan_id вне диапазона')
    n=len(ids);cols=max(1,min(int(ncols),n));rows=int(np.ceil(n/cols));fig,axs=plt.subplots(rows,cols,figsize=figsize or (5.3*cols,3.3*rows),squeeze=False,constrained_layout=True)
    for ax,i in zip(axs.ravel(),ids):
        m=support[i];a=ambiguous[i];ax.plot(b.theta,b.data[i],marker='.',ms=2.5,lw=.65,label='raw');ax.plot(b.theta,b.background_map[i],lw=1.45,label='background');ax.scatter(b.theta[m],b.data[i,m],s=9,alpha=.45,label='background support')
        if np.any(a):ax.scatter(b.theta[a],b.data[i,a],s=16,marker='x',alpha=.65,label='scale-ambiguous')
        ax.plot(b.theta,b.corrected_map[i],lw=.9,alpha=.8,label='raw − background');ax.axhline(0,lw=.7,ls=':');ax.set(title=f'scan {i}; t={b.time_s[i]:.1f} s',xlabel=r'$\theta$ (arcsec)',ylabel='Counts');ax.grid(alpha=.18)
        if theta_lim is not None:ax.set_xlim(theta_lim)
    for ax in axs.ravel()[n:]:ax.set_visible(False)
    axs.ravel()[0].legend(frameon=False,fontsize=8);return fig,axs