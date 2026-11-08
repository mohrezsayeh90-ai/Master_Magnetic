"""Reproducible 16:9 magnetic-input -> fluid -> particle comparison slides.

All plotted trajectories are predicted by the GPU-trained operator, and checked
against the original continuous input using an independent DOP853 solve.
Optional export dependency: python-pptx (only for the PowerPoint deck).
"""
from pathlib import Path
import sys,json,hashlib,zipfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from magnetic_operator.physics import Parameters,R,D,BS,AMAX,observables
from magnetic_operator.inference import predict,reference
from magnetic_operator.train import load

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/presentation'
CASES=[
 ('01_constant_right','Constant right source',lambda t:(0.,.45),r'$a_L=0,\quad a_R=0.45$'),
 ('02_constant_left','Constant left source',lambda t:(.45,0.),r'$a_L=0.45,\quad a_R=0$'),
 ('03_gaussian_pulse','Gaussian pulse on the right',lambda t:(0.,.62*np.exp(-((t-1.9)/.35)**2)),r'$a_L=0,\quad a_R=0.62e^{-((t-1.9)/0.35)^2}$'),
 ('04_sinusoidal','Sinusoidal right source',lambda t:(0.,.55*np.sin(2*np.pi*.8*t)),r'$a_L=0,\quad a_R=0.55\sin(2\pi\,0.8t)$'),
 ('05_source_switch','Smooth switch: left to right',lambda t:(.25*(1-np.tanh((t-1.5)/.15)),.25*(1+np.tanh((t-1.5)/.15))),r'$a_{L,R}=0.25[1\mp\tanh((t-1.5)/0.15)]$'),
 ('06_two_source_chirp','Two-source chirp and cosine',lambda t:(.3*np.sin(2*np.pi*(.3*t+.25*t*t)),.22*np.cos(2*np.pi*.35*t)),r'$a_L=0.30\sin(2\pi(0.3t+0.25t^2)),\quad a_R=0.22\cos(2\pi\,0.35t)$'),
]

def field_map(a,X,Z):
    w=X/D+1j*Z/D
    b=a[0]/(1+w)**2-a[1]/(1-w)**2
    return np.abs(b)*BS*1000

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
      'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
      'figure.facecolor':'#f6f8fc','axes.facecolor':'white','savefig.facecolor':'#f6f8fc',
      'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    model=load(ROOT/'results/operator');par=Parameters();times=np.linspace(0,3,601)
    initial=(0.,-.08)
    nofield=lambda t:(0.,0.)
    baseline,_=predict(model,nofield,times,par,initial,max_window_tau=.025)
    analytic=np.column_stack((np.zeros_like(times),initial[1]+2*par.mean*times))
    assert abs(baseline-analytic).max()<1e-12
    x=np.linspace(-R,R,151);z=np.linspace(-.08,.02,241);Z,X=np.meshgrid(z,x)
    velocity=2*par.mean*(1-(X/R)**2)*1000
    records=[];paths=[]
    for slug,title,field,formula in CASES:
        a=np.asarray([field(t) for t in times]);assert np.abs(a).sum(1).max()<=AMAX
        path,meta=predict(model,field,times,par,initial,max_window_tau=.025,sampling_tolerance=.001)
        truth=reference(field,times,par,initial)
        error=np.max(abs(path-truth)*1e6,axis=0)
        if error[0]>.75 or error[1]>15:raise RuntimeError(f'{slug}: independent acceptance failed: {error}')
        obs=observables(field,times,path)
        snap=int(np.argmax((a*a).sum(1)));tstar=times[snap]
        records.append({'case':slug,'title':title,'formula':formula,'snapshot_time_s':float(tstar),
                        'max_x_error_um':float(error[0]),'max_z_error_um':float(error[1]),
                        'final_x_um':float(path[-1,0]*1e6),'max_abs_x_um':float(abs(path[:,0]).max()*1e6),
                        'inference_seconds':meta['seconds'],'amplitude_max_L1':float(abs(a).sum(1).max()),'passed':True})
        paths.append((slug,title,path,truth))
        table=np.column_stack((times,a,obs[:,1:],baseline,path,truth))
        np.savetxt(OUT/(slug+'.csv'),table,delimiter=',',comments='',header='time_s,a_left,a_right,Bx_T,Bz_T,Fmx_N,Fmz_N,x_no_field_m,z_no_field_m,x_operator_m,z_operator_m,x_reference_m,z_reference_m')
    all_Bmax=max(field_map(np.array([field(r['snapshot_time_s'])])[0],X,Z).max() for (_,_,field,_),r in zip(CASES,records))
    normB=Normalize(0,all_Bmax);normU=Normalize(0,30)
    with PdfPages(OUT/'magnetic_particle_presentation.pdf') as pdf:
      for (slug,title,field,formula),row,(_,_,path,truth) in zip(CASES,records,paths):
        a=np.asarray([field(t) for t in times]);tstar=row['snapshot_time_s']
        span=max(2.,1.5*float(np.max(np.abs(path[:,0]))*1e6))
        fig=plt.figure(figsize=(17.78,10));gs=fig.add_gridspec(2,3,left=.06,right=.97,bottom=.11,top=.82,hspace=.48,wspace=.35)
        fig.text(.06,.94,'MAGNETIC INPUT  →  FLUID FLOW  →  PARTICLE RESPONSE',color='#53647c',fontsize=13,weight='bold')
        fig.text(.06,.885,title,color='#102c4c',fontsize=25,weight='bold')
        fig.text(.97,.89,formula,ha='right',fontsize=13,color='#253852')
        ax=fig.add_subplot(gs[0,0]);ax.plot(times,a[:,0],color='#8e44ad',lw=2,label='Left source');ax.plot(times,a[:,1],color='#e67e22',lw=2,label='Right source')
        ax.axvline(tstar,color='#687b93',ls=':',lw=1);ax.set(title='1a  Prescribed magnetic amplitudes',xlabel='Time t (s)',ylabel='Relative source amplitude a(t)',ylim=(-.72,.72),xlim=(0,3));ax.legend(loc='best',fontsize=9);ax.grid(alpha=.14)
        ax=fig.add_subplot(gs[0,1]);B=field_map(field(tstar),X,Z)
        ax.pcolormesh(Z*1000,X*1e6,B,cmap='magma',norm=normB,shading='auto',rasterized=True)
        fig.colorbar(ScalarMappable(norm=normB,cmap='magma'),ax=ax,pad=.02,label='|B| (mT)')
        ax.set(title=f'1b  Magnetic field at t = {tstar:.2f} s',xlabel='Axial position z (mm)',ylabel='Transverse position x (µm)');ax.set_ylim(-75,75)
        ax=fig.add_subplot(gs[0,2]);ax.pcolormesh(Z*1000,X*1e6,velocity,cmap='Blues',norm=normU,shading='auto',rasterized=True)
        zz,xx=np.meshgrid(np.linspace(-73,12,9),np.linspace(-60,60,5));ax.quiver(zz,xx,2*par.mean*(1-(xx*1e-6/R)**2)*1000,np.zeros_like(xx),angles='xy',scale_units='xy',scale=5,color='#234d72',width=.003)
        fig.colorbar(ScalarMappable(norm=normU,cmap='Blues'),ax=ax,pad=.02,label='Fluid speed (mm/s)')
        ax.set(title='2  Prescribed Poiseuille fluid flow',xlabel='Axial position z (mm)',ylabel='Transverse position x (µm)',ylim=(-75,75))
        ax=fig.add_subplot(gs[1,:2]);ax.pcolormesh(Z*1000,X*1e6,velocity,cmap='Blues',norm=normU,shading='auto',alpha=.25,rasterized=True)
        ax.plot(baseline[:,1]*1000,baseline[:,0]*1e6,'--',color='#20354b',lw=2,label='No magnetic field')
        ax.plot(path[:,1]*1000,path[:,0]*1e6,color='#d34e32',lw=2.8,label='With magnetic field · DeepONet')
        ax.plot(truth[::24,1]*1000,truth[::24,0]*1e6,'o',ms=3,mfc='none',mec='#243f75',label='Independent ODE check')
        for k in (0,200,400,600):
            ax.scatter(path[k,1]*1000,path[k,0]*1e6,s=28,color='#d34e32',zorder=5)
            ax.annotate(f'{times[k]:.0f} s',(path[k,1]*1000,path[k,0]*1e6),xytext=(4,9),textcoords='offset points',fontsize=9,color='#273b55')
        ax.set(title='3a  Same release and flow: trajectories with / without magnetic forcing',xlabel='Axial position z (mm)',ylabel='Transverse position x (µm)',xlim=(-82,21),ylim=(-span,span));ax.legend(loc='lower left',fontsize=9);ax.grid(alpha=.1)
        ax=fig.add_subplot(gs[1,2]);ax.plot(times,np.zeros_like(times),'--',color='#20354b',lw=2,label='No field');ax.plot(times,path[:,0]*1e6,color='#d34e32',lw=2.5,label='With field');ax.set(title='3b  Transverse particle displacement',xlabel='Time t (s)',ylabel='x(t) (µm)',xlim=(0,3),ylim=(-span,span));ax.grid(alpha=.15);ax.legend(fontsize=9)
        fig.text(.06,.052,'Same settings in every case: Umean = 15 mm/s  |  η = 3.277 mPa·s  |  channel radius = 75 µm  |  particle radius = 0.5 µm',fontsize=10,color='#41566f')
        fig.text(.06,.024,'Time-varying amplitudes; fixed spatial source geometry. Overdamped induced-dipole model. Axial and transverse axes use different units.',fontsize=9,color='#596e87')
        fig.text(.97,.024,f'Independent max error: x {row["max_x_error_um"]:.3f} µm · z {row["max_z_error_um"]:.2f} µm',ha='right',fontsize=9,color='#41566f')
        fig.savefig(OUT/(slug+'.png'),dpi=220);fig.savefig(OUT/(slug+'.svg'));fig.savefig(OUT/(slug+'.pdf'));pdf.savefig(fig);plt.close(fig)
    # A compact comparison sheet keeps all axes and scales identical.
    fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
    for ax,(_,title,path,truth) in zip(axes.flat,paths):
        ax.axhline(0,color='#20354b',ls='--',lw=1.5,label='No field');ax.plot(path[:,1]*1000,path[:,0]*1e6,color='#d34e32',lw=2,label='With field')
        ax.set(title=title,xlim=(-80,20),ylim=(-15,15),xlabel='z (mm)',ylabel='x (µm)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Six magnetic inputs · same particle, release and fluid flow',fontsize=20,color='#102c4c');fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'overview.png',dpi=220);fig.savefig(OUT/'overview.pdf');plt.close(fig)
    from pptx import Presentation
    from pptx.util import Inches
    deck=Presentation();deck.slide_width=Inches(16);deck.slide_height=Inches(9)
    for slug,_,_,_ in CASES:
        slide=deck.slides.add_slide(deck.slide_layouts[6]);slide.shapes.add_picture(str(OUT/(slug+'.png')),0,0,width=deck.slide_width,height=deck.slide_height)
    deck.save(OUT/'magnetic_particle_presentation.pptx')
    metadata={'all_passed':True,'cases':records,'duration_s':3,'query_points':601,
      'parameters':{'eta_Pas':par.eta,'mean_m_s':par.mean,'rho_kg_m3':par.rho,'initial_x_m':0,'initial_z_m':-.08},
      'checkpoint_sha256':hashlib.sha256((ROOT/'results/operator/checkpoint.pt').read_bytes()).hexdigest(),
      'magnetic_map':'Snapshot at maximum sampled source-amplitude energy, not necessarily maximum field along the particle path.',
      'color_scales':'One global B scale and one fluid scale for all cases.',
      'trajectory_axes':'Individual slides auto-scale transverse axes for visibility; overview uses common +/-15 um axes.',
      'spatial_geometry':'Fixed two-source geometry; sources lie outside the narrow plotted channel.',
      'model':'Overdamped induced-dipole with Poiseuille flow, no Brownian/gravity. Flow is prescribed and unaffected by particle/field.',
      'export':'PNG 220 dpi; vector SVG/PDF (heatmaps rasterized); six-page PDF and 16:9 PowerPoint with full-slide images.'}
    (OUT/'validation.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
