"""Fixed-time spatial view of the prescribed field in trained case 12."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from magnetic_operator.physics import R,D,BS,numpy_field_force
from trained_cases import phase_chirp

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/field_snapshot'

def field(x,z,amplitudes):
    w=np.asarray(x)/D+1j*np.asarray(z)/D
    return BS*np.abs(amplitudes[0]/(1+w)**2-amplitudes[1]/(1-w)**2)

def main():
    OUT.mkdir(parents=True,exist_ok=True);t=.8;a=phase_chirp(t)
    z=np.linspace(-.08,.068,601);x=np.linspace(-R,R,201);Z,X=np.meshgrid(z,x)
    B=field(X,Z,a)
    # Verify against the project's independent complex-field implementation.
    for xx,zz in ((-R,-.04),(0,0),(R,.023)):
        b,_=numpy_field_force(np.array([xx/R,zz/D]),a)
        assert np.isclose(field(xx,zz,a),np.linalg.norm(b)*BS,rtol=1e-12)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
                        'axes.titlecolor':'#006581','axes.titleweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,8));gs=fig.add_gridspec(2,2,left=.07,right=.95,bottom=.15,top=.81,wspace=.35,hspace=.5,width_ratios=[1.15,1])
    ax=fig.add_subplot(gs[:,0]);im=ax.pcolormesh(z*1000,x*1e6,B*1000,cmap='viridis',shading='auto',rasterized=True)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.14,fraction=.05);cb.set_label('Actual field magnitude |B| (mT)')
    ax.set(title='A  Full channel: magnetic field at one instant',xlabel='Axial coordinate z (mm)',ylabel='Transverse coordinate x (µm)',xlim=(-80,68),ylim=(-75,75),yticks=[-75,-50,-25,0,25,50,75])
    sections=[(-.04,'#dc8500'),(0.,'#dc299b'),(.023,'#008da8')]
    for zz,color in sections:ax.axvline(zz*1000,color=color,ls='--',lw=1.5)
    ax=fig.add_subplot(gs[0,1]);ax.plot(z*1000,field(0,z,a)*1000,color='#173f59',lw=2)
    for zz,color in sections:ax.scatter(zz*1000,field(0,zz,a)*1000,color=color,s=32,zorder=4)
    ax.set(title='B  Axial profile along the channel centre',xlabel='z (mm)',ylabel='|B(0,z)| (mT)',xlim=(-80,68));ax.grid(alpha=.18)
    ax=fig.add_subplot(gs[1,1])
    rows=[]
    for zz,color in sections:
        transverse=field(x,zz,a);baseline=field(0,zz,a)
        ax.plot(x*1e6,(transverse-baseline)*1e6,color=color,lw=2,label=f'z = {zz*1000:g} mm')
        rows.append({'z_mm':zz*1000,'B_at_x_minus75_um_mT':float(transverse[0]*1000),'B_at_x0_mT':float(baseline*1000),'B_at_x_plus75_um_mT':float(transverse[-1]*1000)})
    ax.axhline(0,color='#64748b',lw=.8,ls=':');ax.set(title='C  Small transverse variation, revealed separately',xlabel='x (µm)',ylabel='|B(x,z)| − |B(0,z)| (µT)',xlim=(-75,75));ax.grid(alpha=.18);ax.legend(fontsize=9)
    fig.suptitle('FIXED-TIME MAGNETIC FIELD  |  Case 12: phase-controlled two-source chirp',fontsize=18,color='#064565',weight='bold',y=.97)
    fig.text(.5,.90,f't = {t:.2f} s  ·  a_left = {a[0]:.4f}  ·  a_right = {a[1]:.4f}  ·  full channel width = 150 µm',ha='center',fontsize=12,color='#475569')
    fig.text(.07,.063,'A: actual magnitude. C: difference from the centre value at each chosen z, not the actual magnitude and not prediction error.',fontsize=10,color='#475569')
    fig.text(.07,.025,'Prescribed analytical source field used by Master10. Channel axes use mm and µm; the displayed aspect ratio is not the physical aspect ratio.',fontsize=10,color='#475569')
    fig.savefig(OUT/'case12_t0p8_field.png',dpi=200);fig.savefig(OUT/'case12_t0p8_field.pdf');plt.close(fig)
    report={'case':'12_phase_interference','time_s':t,'a_left':float(a[0]),'a_right':float(a[1]),'field_kind':'Prescribed analytical field used by Master10, no new model or geometry',
            'transverse_sections':rows,'plot_C':'Signed deviation of field magnitude from its x=0 value at each z; microtesla, not error.'}
    (OUT/'snapshot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
