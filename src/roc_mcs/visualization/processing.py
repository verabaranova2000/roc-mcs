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
