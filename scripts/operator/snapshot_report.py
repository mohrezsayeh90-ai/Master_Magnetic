"""Two pages per validated case: input + full fixed-time field, then panels 2–5.

Snapshots are selected from the original waveform, never manufactured. Prefer
strong axial interference notches; otherwise choose peak field. Constant inputs
are shown at the midpoint, because their spatial shape does not vary in time.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from magnetic_operator.physics import R,D,BS
from field_snapshot import field

def select_snapshot(c):
    times=np.linspace(0,c['duration'],1601)
    a=np.asarray([c['field'](t) for t in times]);strength=np.abs(a).sum(1)
    z=np.sort(np.r_[np.linspace(-.08,.068,401),0.])
    w=1j*z[None,:]/D
    axial=BS*np.abs(a[:,0:1]/(1+w)**2-a[:,1:2]/(1-w)**2)
    peak=axial.max(1);centre=BS*np.abs(a[:,0]-a[:,1])
    notch=peak-centre
    eligible=strength>=.4*strength.max()
    score=np.where(eligible,notch,-np.inf)
    if np.max(np.ptp(a,axis=0))<1e-12:
        i=len(times)//2;rule='Constant input: midpoint; spatial pattern is time independent.'
    elif score.max()>.01*peak.max():
        i=int(np.argmax(score));rule='Maximum axial peak-minus-centre notch, among times with amplitude L1 >= 40% of its maximum.'
    else:
        i=int(np.argmax(peak));rule='No substantial axial notch: strongest actual centreline field.'
    t=float(times[i]);amplitudes=tuple(float(v) for v in c['field'](t))
    return {'case':c['slug'],'time_s':t,'amplitudes':amplitudes,'selection_rule':rule,
            'axial_notch_mT':float(notch[i]*1000),'peak_centreline_mT':float(peak[i]*1000),
            'candidate_times':1601,'waveform_unchanged':True}

def export(items,panel,out):
    z=np.linspace(-.08,.068,601);x=np.linspace(-R,R,201);Z,X=np.meshgrid(z,x)
    selections=[];target=out/'trained_cases_complete_with_snapshots.pdf'
    with PdfPages(target) as pdf:
        for item in items:
            c=item[0];p=c['parameters'];s=select_snapshot(c);selections.append(s)
            t=s['time_s'];a=s['amplitudes'];B=field(X,Z,a)
            fig,axes=plt.subplots(2,2,figsize=(16,10))
            fig.subplots_adjust(left=.075,right=.955,top=.81,bottom=.15,hspace=.60,wspace=.35)
            panel(axes[0,0],0,item,False)
            axes[0,0].axvline(t,color='#173f59',ls='--',lw=1.2,label=f'Snapshot: {t:.4f} s')
            axes[0,0].scatter([t,t],a,c=['#e08a00','#e52caa'],s=35,zorder=6)
            axes[0,0].legend(fontsize=8,loc='best')
            ax=axes[0,1];im=ax.pcolormesh(z*1000,x*1e6,B*1000,cmap='viridis',vmin=0,vmax=float(B.max()*1000),shading='auto',rasterized=True)
            cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.045);cb.set_label('|B| (mT)',fontsize=10)
            ax.set(title=f'01b  Full spatial field at t* = {t:.4f} s',xlabel='z (mm)',ylabel='x (µm)',xlim=(-80,68),ylim=(-75,75))
            sections=[(-.04,'#dc8500'),(0.,'#dc299b'),(.023,'#008da8')]
            for zz,color in sections:ax.axvline(zz*1000,color=color,ls='--',lw=1.2)
            ax=axes[1,0];ax.plot(z*1000,field(0,z,a)*1000,color='#173f59',lw=2)
            for zz,color in sections:ax.scatter(zz*1000,field(0,zz,a)*1000,color=color,s=30)
            ax.set(title='01c  Axial field along the channel centre',xlabel='z (mm)',ylabel='|B(0,z)| (mT)',xlim=(-80,68));ax.grid(alpha=.16)
            ax=axes[1,1]
            for zz,color in sections:
                ax.plot(x*1e6,(field(x,zz,a)-field(0,zz,a))*1e6,color=color,lw=2,label=f'z = {zz*1000:g} mm')
            ax.axhline(0,color='#64748b',ls=':',lw=.8);ax.set(title='01d  Transverse variation relative to x = 0',xlabel='x (µm)',ylabel='|B(x,z)| − |B(0,z)| (µT)',xlim=(-75,75));ax.legend(fontsize=9);ax.grid(alpha=.16)
            fig.suptitle('TRAINED MASTER10  |  '+c['title'],fontsize=18,color='#064565',weight='bold',y=.975)
            fig.text(.5,.91,f"Field detail after input 01  ·  t*={t:.4f} s  ·  a_left={a[0]:.4f}, a_right={a[1]:.4f}",ha='center',fontsize=12,color='#475569')
            short='Strong axial interference notch' if s['selection_rule'].startswith('Maximum') else ('Constant input: snapshot at midpoint' if s['selection_rule'].startswith('Constant') else 'Peak actual field amplitude')
            fig.text(.5,.865,'Snapshot selection: '+short,ha='center',fontsize=10,color='#475569')
            fig.text(.075,.065,'01b/c: actual prescribed field magnitude. 01d: difference from each section centre value, not field magnitude or prediction error.',fontsize=10,color='#475569')
            fig.text(.075,.025,'Same two-source geometry and original input functions; no new field learned or invented. Spatial axes use mm and µm.',fontsize=10,color='#475569')
            pdf.savefig(fig)
            if c['slug'] in ('07_offset_alternating','12_phase_interference'):
                fig.savefig(out/(c['slug']+'_selected_snapshot.png'),dpi=160)
            plt.close(fig)
            # Continue the original no-error sequence immediately afterwards.
            fig,axes=plt.subplots(2,2,figsize=(16,10))
            fig.subplots_adjust(left=.075,right=.955,top=.81,bottom=.15,hspace=.70,wspace=.35)
            for ax,j in zip(axes.flat,range(1,5)):panel(ax,j,item,False)
            fig.suptitle('TRAINED MASTER10  |  '+c['title'],fontsize=18,color='#064565',weight='bold',y=.975)
            fig.text(.5,.91,f"Continuation: panels 02–05  ·  x₀={c['initial'][0]*1e6:.2f} µm, z₀={c['initial'][1]*1000:.1f} mm  ·  Umean={p.mean*1000:g} mm/s  ·  η={p.eta*1000:.3f} mPa·s  ·  T={c['duration']:g} s",ha='center',fontsize=11,color='#475569')
            fig.text(.075,.065,'Magenta: trained network with B. Dark open circles: direct physical ODE. Orange dashed: same release without B.',fontsize=10,color='#064565')
            fig.text(.075,.025,'Validated trajectories and weights unchanged. Fluid view: full channel; particle paths: zoomed, no fluid-speed colourbar.',fontsize=10,color='#475569')
            pdf.savefig(fig);plt.close(fig)
    (out/'snapshot_selection.json').write_text(json.dumps({'pdf':target.name,'pages':24,'selection_uses_prediction_errors':False,'selections':selections},indent=2))
    print('SAVED',target,flush=True)
