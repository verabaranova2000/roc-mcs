def _centers_to_edges(x):
    import numpy as np
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<2 or not np.all(np.diff(x)>0):raise ValueError('Координаты должны быть строго возрастающим 1D-массивом')
    e=np.empty(len(x)+1,float);e[1:-1]=.5*(x[:-1]+x[1:]);e[0]=x[0]-.5*(x[1]-x[0]);e[-1]=x[-1]+.5*(x[-1]-x[-2]);return e

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
    te=_centers_to_edges(b.time_s);he=_centers_to_edges(b.theta);im=axs[1,0].pcolormesh(te,he,b.weight_map.T,shading='flat',vmin=0,vmax=1,rasterized=True);fig.colorbar(im,ax=axs[1,0],pad=.01,label='Background support');axs[1,0].set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title='C  Points used to fit background')
    im=axs[1,1].pcolormesh(te,he,b.background_map.T,shading='flat',rasterized=True);fig.colorbar(im,ax=axs[1,1],pad=.01,label='Background counts');axs[1,1].set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title=r'D  Background surface $B(t,\theta)$')
    er=np.array([np.nanmean(b.standardized_residual_map[i][b.weight_map[i]>.5]) for i in range(n)]);axs[2,0].plot(b.time_s,er,marker='.');axs[2,0].axhline(0,ls='--',lw=1);axs[2,0].set(xlabel='Time (s)',ylabel='Mean standardized residual',title='E  Residual bias on fitted background support');axs[2,0].grid(alpha=.2)
    if cii_result is not None:
        for i in ids:axs[2,1].plot(b.theta,cii_result['cii_diagnostic'][i],lw=1.3,label=f't={b.time_s[i]:.1f}s')
        axs[2,1].axhline(0,ls=':',lw=.8);axs[2,1].axhline(1,ls=':',lw=.8);axs[2,1].set(xlabel=r'$\theta$ (arcsec)',ylabel='Cumulative signed residual / positive area',title='F  CII background diagnostic (unclipped)');axs[2,1].legend(frameon=False,fontsize=8);axs[2,1].grid(alpha=.2)
    else:axs[2,1].axis('off')
    m=b.metadata;fig.suptitle(f"Background-support count fit; success={b.converged}; support={m.get('support_fraction',np.nan):.1%}; residual mean/sd={m.get('support_residual_mean',np.nan):.2f}/{m.get('support_residual_sd',np.nan):.2f}",fontsize=10);return fig,axs

def _support_masks(b,autos=None):
    import numpy as np
    support=np.asarray(autos.support_mask,bool) if autos is not None else np.asarray(b.weight_map>0.5,bool);ambiguous=np.asarray(getattr(autos,'ambiguous_mask',np.zeros_like(support)),bool) if autos is not None else np.zeros_like(support);return support,ambiguous

# def plot_background_support_map(b,*,autos=None,theta_lim=None,figsize=(10.5,5.2)):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     support,ambiguous=_support_masks(b,autos);te=_centers_to_edges(b.time_s);he=_centers_to_edges(b.theta);fig,ax=plt.subplots(figsize=figsize,constrained_layout=True);z=np.arcsinh(np.maximum(b.data,0))
#     im=ax.pcolormesh(te,he,z.T,shading='flat',rasterized=True);overlay=np.ma.masked_where(~support.T,np.ones_like(support.T,float));ax.pcolormesh(te,he,overlay,shading='flat',cmap='Greys',vmin=0,vmax=4,alpha=.18,rasterized=True)
#     if np.any(ambiguous):
#         ax.contour(b.time_s,b.theta,ambiguous.T.astype(float),levels=[.5],colors='white',linewidths=2.2,linestyles='-');ax.contour(b.time_s,b.theta,ambiguous.T.astype(float),levels=[.5],colors='black',linewidths=.9,linestyles='--')
#     fig.colorbar(im,ax=ax,pad=.01,label='asinh(counts)');ax.set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title='Automatic background support on raw rocking map')
#     if theta_lim is not None:ax.set_ylim(theta_lim)
#     return fig,ax

def plot_background_support_map(b,*,autos=None,theta_lim=None,figsize=(10.5,5.2)):
    import numpy as np
    import matplotlib.pyplot as plt
    support,ambiguous=_support_masks(b,autos);te=_centers_to_edges(b.time_s);he=_centers_to_edges(b.theta);fig,ax=plt.subplots(figsize=figsize,constrained_layout=True);z=np.arcsinh(np.maximum(b.data,0))
    im=ax.pcolormesh(te,he,z.T,shading='flat',rasterized=True);overlay=np.ma.masked_where(~support.T,np.ones_like(support.T,float));ax.pcolormesh(te,he,overlay,shading='flat',cmap='Greys',vmin=0,vmax=4,alpha=.18,rasterized=True)
    if np.any(ambiguous):
        h=ax.contourf(b.time_s,b.theta,ambiguous.T.astype(float),levels=[.5,1.5],colors='none',hatches=['/'])
        if hasattr(h,'set_edgecolor'):h.set_edgecolor('#8f3f3f');h.set_linewidth(.35)
        elif hasattr(h,'collections'):
            for c in h.collections:c.set_edgecolor('#8f3f3f');c.set_linewidth(.35)
        ax.contour(b.time_s,b.theta,ambiguous.T.astype(float),levels=[.5],colors='#8f3f3f',linewidths=.65)
    fig.colorbar(im,ax=ax,pad=.01,label='asinh(counts)');ax.set(xlabel='Time (s)',ylabel=r'$\theta$ (arcsec)',title='Automatic background support on raw rocking map')
    if theta_lim is not None:ax.set_ylim(theta_lim)
    return fig,ax


def plot_background_support_slices(b,scan_ids,*,autos=None,ncols=2,theta_lim=None,figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    support,ambiguous=_support_masks(b,autos);ids=[int(i) if int(i)>=0 else len(b.time_s)+int(i) for i in scan_ids]
    if any(i<0 or i>=len(b.time_s) for i in ids):raise IndexError('scan_id вне диапазона')
    n=len(ids);cols=max(1,min(int(ncols),n));rows=int(np.ceil(n/cols));fig,axs=plt.subplots(rows,cols,figsize=figsize or (5.2*cols,3.2*rows),squeeze=False,constrained_layout=True)
    for ax,i in zip(axs.ravel(),ids):
        m=support[i];a=ambiguous[i];ax.plot(b.theta,b.data[i],marker='.',ms=2.5,lw=.65,label='raw');ax.plot(b.theta,b.background_map[i],lw=1.45,label='background');ax.scatter(b.theta[m],b.data[i,m],s=8,alpha=.42,label='background support')
        if np.any(a):ax.scatter(b.theta[a],b.data[i,a],s=15,marker='x',alpha=.6,label='scale-ambiguous')
        ax.plot(b.theta,b.corrected_map[i],lw=.9,alpha=.8,label='raw − background');ax.axhline(0,lw=.7,ls=':');ax.set(title=f'scan {i}; t={b.time_s[i]:.1f} s',xlabel=r'$\theta$ (arcsec)',ylabel='Counts');ax.grid(alpha=.18)
        if theta_lim is not None:ax.set_xlim(theta_lim)
    for ax in axs.ravel()[n:]:ax.set_visible(False)
    axs.ravel()[0].legend(frameon=False,fontsize=8);return fig,axs

def plot_background_auto_convergence(autos,*,figsize=(7.6,5.6)):
    import numpy as np
    import matplotlib.pyplot as plt
    h=autos.history
    if not h:raise ValueError('autos.history пуст')
    it=np.array([x['iteration'] for x in h]);sf=np.array([x['support_fraction'] for x in h],float);ch=np.array([x['mask_change_fraction'] for x in h],float);sd=np.array([x['support_residual_sd'] for x in h],float)
    fig,axs=plt.subplots(2,1,figsize=figsize,sharex=True,constrained_layout=True);axs[0].plot(it,sf,marker='o',label='support fraction');axs[0].plot(it,sd,marker='s',label='support residual SD');axs[0].axhline(1,ls=':',lw=.8);axs[0].set(ylabel='Fraction / residual SD',title=f"Automatic support refinement; converged={autos.metadata['converged']}");axs[0].legend(frameon=False);axs[0].grid(alpha=.2)
    m=np.isfinite(ch);axs[1].plot(it[m],ch[m],marker='o');axs[1].axhline(0,ls=':',lw=.8);axs[1].set(xlabel='Refinement iteration',ylabel='Changed-mask fraction',title='Newly excluded support per iteration');axs[1].grid(alpha=.2);return fig,axs

def plot_cii_sensitivity(audit,*,metrics=('q50','w80_cii','w90_cii'),figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    vals=audit['values'];mets=tuple(metrics);fig,axs=plt.subplots(len(mets),1,figsize=figsize or (9,2.8*len(mets)),sharex=True,squeeze=False,constrained_layout=True);axs=axs[:,0]
    for ax,metric in zip(axs,mets):
        d=vals[vals.metric==metric]
        if d.empty:raise KeyError(f'Нет метрики {metric!r}')
        for fdr,g in d.groupby('fdr'):ax.plot(g.time_s,g.value,lw=1,alpha=.55,label=f'FDR={fdr:g}')
        piv=d.pivot(index='time_s',columns='fdr',values='value').sort_index();ax.plot(piv.index,piv.median(axis=1),lw=1.6,label='median across FDR');ax.fill_between(piv.index,piv.min(axis=1),piv.max(axis=1),alpha=.18);ax.set(ylabel=metric);ax.grid(alpha=.2)
    axs[0].legend(frameon=False,ncol=2,fontsize=8);axs[-1].set_xlabel('Time (s)');fig.suptitle('CII/background sensitivity to screening level');return fig,axs

def plot_cii_sensitivity_ranges(audit,*,metrics=('q10','q50','q90','q95','w80_cii','w90_cii','centroid'),figsize=(9,4.8)):
    import numpy as np
    import matplotlib.pyplot as plt
    r=audit['ranges'];fig,ax=plt.subplots(figsize=figsize,constrained_layout=True)
    for metric in metrics:
        d=r[r.metric==metric]
        if len(d):ax.plot(d.time_s,d.range_bins,marker='.',lw=1,label=metric)
    ax.axhline(1,ls=':',lw=.8);ax.set(xlabel='Time (s)',ylabel='Sensitivity range / angular-bin width',title='Angular sensitivity across FDR settings');ax.grid(alpha=.2);ax.legend(frameon=False,ncol=2,fontsize=8);return fig,ax

def plot_cii_robustness_publication(audit,*,quantiles=('q10','q50','q90'),widths=('w80_cii','w90_cii'),figsize=(10,6.8)):
    import matplotlib.pyplot as plt
    vals=audit['values'];fig,axs=plt.subplots(2,1,figsize=figsize,sharex=True,constrained_layout=True)
    for metric in quantiles:
        d=vals[vals.metric==metric];p=d.pivot(index='time_s',columns='fdr',values='value').sort_index();axs[0].plot(p.index,p.median(axis=1),lw=1.4,label=metric);axs[0].fill_between(p.index,p.min(axis=1),p.max(axis=1),alpha=.15)
    for metric in widths:
        d=vals[vals.metric==metric];p=d.pivot(index='time_s',columns='fdr',values='value').sort_index();axs[1].plot(p.index,p.median(axis=1),lw=1.4,label=metric);axs[1].fill_between(p.index,p.min(axis=1),p.max(axis=1),alpha=.15)
    axs[0].set(ylabel=r'$\theta$ (arcsec)',title='CII quantiles: median estimate and FDR-sensitivity envelope');axs[1].set(xlabel='Time (s)',ylabel='Width (arcsec)',title='CII widths: median estimate and FDR-sensitivity envelope')
    for ax in axs:ax.grid(alpha=.2);ax.legend(frameon=False,ncol=3,fontsize=8)
    return fig,axs

def plot_background_support_quality(b,*,figsize=(9.2,5.0)):
    import numpy as np
    import matplotlib.pyplot as plt
    status=np.asarray(b.metadata.get('support_status',['unknown']*len(b.time_s)),object);codes={'bilateral_observed':0,'unilateral_temporal':1,'sparse_temporal':2,'temporal_interpolation':3,'temporal_extrapolation':4,'temporal_only':4,'unknown':5};y=np.array([codes.get(str(s),5) for s in status],int);rse=np.asarray(b.metadata.get('prediction_relative_se_map',np.full_like(b.background_map,np.nan)),float);q=np.nanmedian(rse,axis=1)
    fig,axs=plt.subplots(2,1,figsize=figsize,sharex=True,constrained_layout=True);axs[0].step(b.time_s,y,where='mid',lw=1.3);axs[0].scatter(b.time_s,y,s=18);axs[0].set(ylabel='Support status',yticks=range(6),yticklabels=['bilateral observed','unilateral + temporal','sparse + temporal','temporal interpolation','temporal extrapolation','unknown'],title='Background observability / fallback status');axs[0].grid(axis='x',alpha=.2)
    axs[1].plot(b.time_s,q,marker='.',lw=1);axs[1].set(xlabel='Time (s)',ylabel='Median relative SE',title='Approximate uncertainty of fitted background');axs[1].grid(alpha=.2);return fig,axs


