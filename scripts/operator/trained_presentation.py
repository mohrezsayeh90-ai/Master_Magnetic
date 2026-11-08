"""Publication plots ONLY for the existing trained Master10 equation/domain.

No altered flow, no new spatial magnetic basis, no retraining, no ODE fallback
inside inference. Twelve cases: six originals plus six shifted/complex inputs.
"""
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch
import sys,json,hashlib,zipfile,textwrap
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation
from pptx.util import Inches
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from magnetic_operator.physics import R,D,RP,BS,AMAX,Parameters,observables
from magnetic_operator.train import load
from magnetic_operator.inference import predict,reference
from trained_cases import cases

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/trained_presentation'
NETWORK='#e52caa';NUMERICAL='#173f59';BASE='#e08a00';CYAN='#008da8'

def checksum():return hashlib.sha256((ROOT/'results/operator/checkpoint.pt').read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);model=load(ROOT/'results/operator')
    sha=checksum();items=[]
    reuse='--reuse-data' in sys.argv
    saved=json.loads((OUT/'validation.json').read_text()) if reuse else None
    if reuse and (not saved['all_passed'] or saved['checkpoint_sha256']!=sha):
        raise RuntimeError('Cached results do not match the validated checkpoint')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
      'axes.titleweight':'bold','axes.titlecolor':'#006581','pdf.fonttype':42,'svg.fonttype':'none',
      'figure.facecolor':'white','axes.spines.top':True,'axes.spines.right':True})
    for c in cases():
        p=c['parameters'].validate();initial=np.array(c['initial']);t=np.linspace(0,c['duration'],801)
        amps=np.array([c['field'](v) for v in t]);assert abs(amps).sum(1).max()<=AMAX+1e-10
        if reuse:
            data=np.loadtxt(OUT/(c['slug']+'.csv'),delimiter=',',skiprows=1)
            row=next(r for r in saved['cases'] if r['case']==c['slug'])
            assert row['parameters']==asdict(p) and np.allclose(row['initial_m'],initial)
            assert np.allclose(data[:,0],t) and np.allclose(data[:,1:3],amps)
            baseline,pred,truth,error=data[:,7:9],data[:,9:11],data[:,11:13],data[:,13:15]
            assert np.allclose((pred-truth)*1e6,error,rtol=1e-10,atol=1e-12)
            items.append((c,t,amps,pred,baseline,truth,error,row))
            continue
        # Any attempted solve_ivp call during network inference fails immediately.
        with patch('magnetic_operator.inference.solve_ivp',side_effect=AssertionError('ODE called during inference')):
            pred,meta=predict(model,c['field'],t,p,initial,max_window_tau=.0125,sampling_tolerance=.0005)
            baseline,_=predict(model,lambda t:(0.,0.),t,p,initial,max_window_tau=.0125)
        truth=reference(c['field'],t,p,initial,max_step_s=.002)
        refined=reference(c['field'],t,p,initial,max_step_s=.001)
        error=(pred-truth)*1e6;reference_change=np.max(abs(truth-refined),axis=0)*1e6
        analytic=np.column_stack((np.full_like(t,initial[0]),initial[1]+2*p.mean*(1-(initial[0]/R)**2)*t))
        baseline_error=float(abs(baseline-analytic).max())
        ex,ez=np.max(abs(error),axis=0)
        if not (ex<.75 and ez<15 and np.max(reference_change)<.001 and baseline_error<1e-11):
            raise RuntimeError(f'Acceptance failed for {c["slug"]}: {ex,ez,reference_change,baseline_error}')
        obs=observables(c['field'],t,pred)
        row={'case':c['slug'],'group':c['group'],'title':c['title'],'field_function':c['field'].__name__,
          'parameters':asdict(p),'initial_m':initial.tolist(),'duration_s':c['duration'],
          'initial_fluid_speed_mm_s':float(2*p.mean*(1-(initial[0]/R)**2)*1000),
          'max_amplitude_L1':float(abs(amps).sum(1).max()),'max_x_error_um':float(ex),'max_z_error_um':float(ez),
          'rms_x_error_um':float(np.sqrt(np.mean(error[:,0]**2))),'rms_z_error_um':float(np.sqrt(np.mean(error[:,1]**2))),
          'reference_step_refinement_change_um':reference_change.tolist(),'no_field_analytic_error_m':baseline_error,
          'inference_seconds':meta['seconds'],'inference_ODE_calls':0,'retraining_steps':0,'passed':True}
        header='time_s,a_left,a_right,Bx_T,Bz_T,Fmx_N,Fmz_N,x_no_B_m,z_no_B_m,x_network_m,z_network_m,x_numerical_m,z_numerical_m,error_x_um,error_z_um'
        np.savetxt(OUT/(c['slug']+'.csv'),np.column_stack((t,amps,obs[:,1:],baseline,pred,truth,error)),delimiter=',',comments='',header=header)
        items.append((c,t,amps,pred,baseline,truth,error,row))
        print(c['slug'],ex,ez,flush=True)
    assert sha==checksum()
    bmax=BS*.7071067812*1000
    def panel(ax,j,item,wide):
        c,t,a,pred,baseline,truth,error,row=item;p=c['parameters'];initial=np.array(c['initial'])
        zlo=max(-.08,min(initial[1],pred[:,1].min(),truth[:,1].min())-.002)
        zhi=min(.068,max(pred[:,1].max(),baseline[:,1].max(),truth[:,1].max())+.002)
        z=np.linspace(zlo,zhi,180);x=np.linspace(-R,R,121);Z,X=np.meshgrid(z,x)
        if j==0:
            ax.plot(t,a[:,0],color=BASE,lw=1.6,label='Left source');ax.plot(t,a[:,1],color=NETWORK,lw=1.6,label='Right source')
            ax.set(xlabel='Time (s)',ylabel='Relative field amplitude',xlim=(0,c['duration']),ylim=(-.72,.72));ax.legend(fontsize=7);ax.grid(alpha=.16)
        elif j==1:
            # Space-time evolution of the actual applied B field at release x.
            w=initial[0]/D+1j*z[None,:]/D
            b=a[:,0:1]/(1+w)**2-a[:,1:2]/(1-w)**2
            im=ax.pcolormesh(z*1000,t,abs(b)*BS*1000,cmap='viridis',vmin=0,vmax=bmax,shading='auto',rasterized=True)
            cb=plt.colorbar(im,ax=ax,orientation='horizontal',pad=.21 if wide else .22,fraction=.07,aspect=28)
            cb.set_label(f'|B| (mT) at x = {initial[0]*1e6:.1f} µm',fontsize=8);cb.ax.tick_params(labelsize=7)
            ax.set(xlabel='z (mm)',ylabel='Time (s)')
        elif j==2:
            # Fluid context: full computational channel and physical speed scale.
            zlo,zhi=-.08,.068
            z=np.linspace(zlo,zhi,241);x=np.linspace(-R,R,151);Z,X=np.meshgrid(z,x)
            v=2*p.mean*(1-(X/R)**2)*1000
            vmax=2*p.mean*1000
            im=ax.pcolormesh(Z*1000,X*1e6,v,cmap='viridis',vmin=0,vmax=vmax,shading='auto',rasterized=True)
            for xx in np.linspace(-60,60,7):
                ax.axhline(xx,color='white',alpha=.4,lw=.6)
                for zz in np.linspace(zlo,zhi,5)[1:-1]:
                    ax.annotate('',xy=((zz+.03*(zhi-zlo))*1000,xx),xytext=(zz*1000,xx),arrowprops={'arrowstyle':'->','color':'white','lw':.7})
            cb=plt.colorbar(im,ax=ax,orientation='horizontal',pad=.21 if wide else .22,fraction=.07,aspect=28)
            cb.set_ticks(np.linspace(0,vmax,5))
            cb.set_label('Fluid speed (mm/s)',fontsize=8);cb.ax.tick_params(labelsize=7)
            ax.set(xlabel='z (mm)',ylabel='x (µm)',xlim=(-80,68),ylim=(-75,75),xticks=[-80,-40,0,40,68],yticks=[-75,-50,-25,0,25,50,75])
            ax.axhline(-75,color=NUMERICAL,lw=1.2);ax.axhline(75,color=NUMERICAL,lw=1.2)
        elif j==3:
            # Trajectory detail: zoom to the paths, no fluid image or colorbar.
            ax.set_facecolor('white')
            ax.plot(baseline[:,1]*1000,baseline[:,0]*1e6,'--',color=BASE,lw=1.5,label='No B · network')
            ax.plot(pred[:,1]*1000,pred[:,0]*1e6,color=NETWORK,lw=2,label='With B · network')
            ax.plot(truth[::32,1]*1000,truth[::32,0]*1e6,'o',ms=3,mfc='white',mec=NUMERICAL,mew=.8,label='ODE · no network')
            ylo=min(pred[:,0].min(),truth[:,0].min(),initial[0])*1e6
            yhi=max(pred[:,0].max(),truth[:,0].max(),initial[0])*1e6
            pad=max(.3,(yhi-ylo)*.3)
            ax.set(xlabel='z (mm)',ylabel='x (µm)',xlim=(zlo*1000,zhi*1000),ylim=(ylo-pad,yhi+pad))
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.grid(alpha=.15);ax.legend(fontsize=6.5,loc='best')
        elif j==4:
            ax.plot(t,baseline[:,0]*1e6,'--',color=BASE,lw=1.5,label='No B')
            ax.plot(t,pred[:,0]*1e6,color=NETWORK,lw=2,label='Network')
            ax.plot(t[::32],truth[::32,0]*1e6,'o',ms=3,mfc='white',mec=NUMERICAL,mew=.8,label='Numerical')
            ax.set(xlabel='Time (s)',ylabel='x(t) (µm)',xlim=(0,c['duration']));ax.grid(alpha=.16);ax.legend(fontsize=7)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        else:
            for k,color,label in ((0,NETWORK,'|x network − x ODE|'),(1,CYAN,'|z network − z ODE|')):
                values=abs(error[:,k]);ax.semilogy(t,np.where(values>1e-10,values,np.nan),color=color,lw=1.2,label=label)
            ax.set(xlabel='Time (s)',ylabel='Absolute error (µm) · log scale',xlim=(0,c['duration']));ax.grid(alpha=.18);ax.legend(fontsize=6.5)
            ax.text(.04,.04,f'Max x: {row["max_x_error_um"]:.4f} µm\nMax z: {row["max_z_error_um"]:.3f} µm',transform=ax.transAxes,fontsize=8,bbox={'facecolor':'white','alpha':.9,'edgecolor':'#ddd'})
        ax.set_title(['01  Magnetic input','02  Space–time magnetic field','03  Trained-domain fluid flow','04  Network / numerical paths','05  Transverse response','06  Actual prediction error'][j],pad=10)
    with PdfPages(OUT/'trained_cases_slides.pdf') as pdf,PdfPages(OUT/'trained_cases_chains.pdf') as widepdf:
      for item in items:
        c,*_=item;p=c['parameters'];initial=c['initial']
        for wide in (True,False):
            fig,axes=plt.subplots(1,6,figsize=(27,5.2)) if wide else plt.subplots(2,3,figsize=(16,9))
            fig.subplots_adjust(left=.045,right=.985,top=.80 if wide else .83,bottom=.24 if wide else .15,wspace=.54 if wide else .4,hspace=.92)
            for j,ax in enumerate(axes.flat):panel(ax,j,item,wide)
            fig.suptitle('TRAINED MASTER10  |  '+c['title'],fontsize=15 if wide else 19,color='#064565',weight='bold',y=.98)
            subtitle=f"{c['group']} case  ·  x₀={initial[0]*1e6:.2f} µm, z₀={initial[1]*1000:.1f} mm  ·  Umean={p.mean*1000:g} mm/s, η={p.eta*1000:.3f} mPa·s  ·  T={c['duration']:g} s"
            fig.text(.5,.90 if wide else .91,subtitle,ha='center',fontsize=10,color='#566576')
            fig.text(.045,.05,'Magenta: trained network with B. Dark open circles: direct physical ODE (no neural model). Orange dashed: same release without B.',fontsize=9,color='#064565')
            fig.text(.045,.012,'Original straight-channel geometry and overdamped equation. No independent launch-velocity input. No-B paths remain straight. Spatial axes use different units.',fontsize=8,color='#566576')
            if wide:
                for xp in (.186,.347,.507,.669,.831):fig.text(xp,.94,'→',color='#c08012',fontsize=19)
                fig.savefig(OUT/(c['slug']+'_chain.png'),dpi=220);fig.savefig(OUT/(c['slug']+'_chain.svg'));widepdf.savefig(fig)
            else:fig.savefig(OUT/(c['slug']+'_slide.png'),dpi=220);pdf.savefig(fig)
            plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,item in zip(axes.flat,items[6:]):
        c,t,a,pred,baseline,truth,error,row=item
        ax.plot(t,(pred[:,0]-c['initial'][0])*1e6,color=NETWORK,lw=1.8,label='Network')
        ax.plot(t[::32],(truth[::32,0]-c['initial'][0])*1e6,'o',ms=3,mfc='none',mec=NUMERICAL,label='Numerical')
        ax.axhline(0,color=BASE,ls='--',label='No B');ax.set(title=textwrap.fill(c['title'],width=37),xlabel='Time (s)',ylabel='x(t) − x₀ (µm)');ax.legend(fontsize=8);ax.grid(alpha=.16)
    fig.suptitle('Six additional cases · same trained weights · direct numerical overlay',fontsize=18,color='#064565');fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'additional_cases_overview.png',dpi=220);fig.savefig(OUT/'additional_cases_overview.pdf');plt.close(fig)
    deck=Presentation();deck.slide_width=Inches(16);deck.slide_height=Inches(9)
    for item in items:
        slide=deck.slides.add_slide(deck.slide_layouts[6]);slide.shapes.add_picture(str(OUT/(item[0]['slug']+'_slide.png')),0,0,width=deck.slide_width,height=deck.slide_height)
    deck.save(OUT/'trained_cases_presentation.pptx')
    sample=next(i for i in items if i[0]['slug']=='07_offset_alternating')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5));fig.subplots_adjust(left=.08,right=.97,bottom=.25,top=.86,wspace=.32)
    panel(axes[0],2,sample,False);panel(axes[1],3,sample,False)
    fig.suptitle('Full-channel fluid flow and zoomed particle trajectories',fontsize=14,color='#064565',weight='bold')
    fig.savefig(OUT/'07_offset_alternating_flow_and_paths.png',dpi=220);fig.savefig(OUT/'07_offset_alternating_flow_and_paths.pdf');plt.close(fig)
    for slug in ('07_offset_alternating','11_downstream_switch'):
        sample=next(i for i in items if i[0]['slug']==slug)
        fig,ax=plt.subplots(figsize=(6,4.5));fig.subplots_adjust(left=.16,right=.96,bottom=.17,top=.87)
        panel(ax,3,sample,False)
        assert len(fig.axes)==1 and len(ax.collections)==0
        fig.savefig(OUT/(slug+'_paths_no_colorbar.png'),dpi=220)
        fig.savefig(OUT/(slug+'_paths_no_colorbar.pdf'));plt.close(fig)
    report={'all_passed':True,'checkpoint_sha256':sha,'checkpoint_unchanged':checksum()==sha,'trained_model':'Existing Master10; no retraining',
      'geometry':'Original fixed straight cylindrical-channel center-plane model and two mirror magnetic sources',
      'equation':'Overdamped v_particle = v_fluid + F_m/(6*pi*eta*rp); F_m=alpha/2 grad(|B|^2)',
      'inference_ODE_calls':0,'retraining_steps':0,'independent_reference':'SciPy DOP853 of analytic physics.numpy_rhs; no network or learned flow/field. Step .002s refined to .001s.',
      'acceptance_thresholds':{'max_x_um':.75,'max_z_um':15,'reference_refinement_um':.001,'baseline_analytic_m':1e-11},
      'inference_settings':{'max_window_tau':.0125,'sampling_tolerance':.0005},
      'plots':'Shared B scale 0..111.07 mT. Fluid panel: full z=-80..68 mm and x=-75..75 um, flow scale 0..2U per case. Trajectory panel: zoomed to paths on white background, no speed field or colorbar. Temporal response y axes auto-scaled; error plot log scale, exact/roundoff zeros omitted.',
      'plot_revision':'Full-channel fluid view retained; zoomed plain trajectory view without any fluid speed plot or colorbar.',
      'rendered_from_cached_validated_data':reuse,
      'initial_velocity':'Not an independent parameter of this trained overdamped model; changing launch location and U changes local fluid/terminal particle speed.',
      'cases':[i[-1] for i in items]}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    print('ALL_TWELVE_TRAINED_CASES_PASSED',flush=True)

if __name__=='__main__':main()
