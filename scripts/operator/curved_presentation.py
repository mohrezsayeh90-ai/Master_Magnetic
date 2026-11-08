"""Curved-channel prescribed-flow ODE illustrations; NOT Master10 inference.

Streamfunction: psi=2U*(y-y^3/(3R^2)), y=x-c(z).
Then u_z=psi_x, u_x=-psi_z=c'(z)*u_z: divergence-free, no slip
on x=c(z)+/-R. This is a prescribed kinematic flow, not a solved NS field.
"""
from pathlib import Path
import sys,json,zipfile
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from pptx import Presentation
from pptx.util import Inches
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from magnetic_operator.physics import R,D,RP,BS,ALPHA,AMAX,Parameters,numpy_field_force
from presentation import CASES

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/curved_presentation'
Z0=-.075;T=3.2;U=.015;PAR=Parameters();GAMMA=6*np.pi*PAR.eta*RP
COLORS=['#e58b00','#e233ab','#00a6ad'];OFFSETS=np.array([.32,0,-.32])*R

def center(z):
    s=2*np.pi*(np.asarray(z)-Z0)/.075
    return 18e-6*np.sin(s)+8e-6*np.sin(2*s)

def slope(z):
    s=2*np.pi*(np.asarray(z)-Z0)/.075
    return 2*np.pi/.075*(18e-6*np.cos(s)+16e-6*np.cos(2*s))

def fluid(x,z):
    y=x-center(z);vz=2*U*(1-(y/R)**2)
    return slope(z)*vz,vz

def solve(field,offset,times,method='DOP853'):
    def rhs(t,q):
        x,z=q*np.array([R,D]);vx,vz=fluid(x,z)
        _,f=numpy_field_force(q,field(t));fm=ALPHA*BS**2/D*f
        return (np.array([vx,vz])+fm/GAMMA)/[R,D]
    def exit_domain(t,q):return .7*R-abs(q[0]*R-center(q[1]*D))
    exit_domain.terminal=True;exit_domain.direction=-1
    sol=solve_ivp(rhs,[0,T],[offset/R,Z0/D],method=method,rtol=2e-10,atol=2e-12,
                  max_step=.005 if method=='DOP853' else .01,dense_output=True,events=exit_domain)
    if not sol.success or sol.t[-1]<T-1e-10:raise RuntimeError('Trajectory exited local |x-c(z)| <= 0.7R domain')
    return sol.sol(times).T*[R,D]

def main():
    OUT.mkdir(parents=True,exist_ok=True);times=np.linspace(0,T,641)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                        'axes.titleweight':'bold','axes.titlecolor':'#00627d','axes.spines.top':True,
                        'axes.spines.right':True,'figure.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
    z=np.linspace(Z0,.025,280);x=np.linspace(-105e-6,105e-6,181);Z,X=np.meshgrid(z,x)
    vx,vz=fluid(X,Z);mask=np.abs(X-center(Z))<=R
    speed=np.ma.masked_where(~mask,np.sqrt(vx*vx+vz*vz)*1000)
    base=[];base_err=[]
    for off in OFFSETS:
        p=solve(lambda t:(0.,0.),off,times)
        za=Z0+2*U*(1-(off/R)**2)*times;analytic=np.column_stack((center(za)+off,za))
        base_err.append(float(np.max(abs(p-analytic))));base.append(p)
    if max(base_err)>1e-10:raise RuntimeError('No-field streamline analytic check failed')
    # Finite-difference incompressibility check of the prescribed velocity field.
    rng=np.random.default_rng(441);zz=rng.uniform(Z0,.025,200);xx=center(zz)+rng.uniform(-.6,.6,200)*R
    ex=1e-9;ez=1e-6
    div=(fluid(xx+ex,zz)[0]-fluid(xx-ex,zz)[0])/(2*ex)+(fluid(xx,zz+ez)[1]-fluid(xx,zz-ez)[1])/(2*ez)
    if abs(div).max()>1e-6:raise RuntimeError('Velocity divergence check failed')
    computed=[]
    for slug,title,field,formula in CASES:
        av=np.array([field(t) for t in times]);assert np.abs(av).sum(1).max()<=AMAX
        ps=[];errs=[]
        for i,off in enumerate(OFFSETS):
            p=solve(field,off,times);r=solve(field,off,times,'Radau');error=abs(p-r).max(0)*1e6
            if error[0]>.01 or error[1]>.1:raise RuntimeError(f'Cross-solver check failed: {slug} {error}')
            ps.append(p);errs.append(error.tolist())
            np.savetxt(OUT/f'{slug}_launch_{i+1}.csv',np.column_stack((times,av,base[i],p,r)),delimiter=',',comments='',
              header='time_s,a_left,a_right,x_no_field_m,z_no_field_m,x_with_field_m,z_with_field_m,x_Radau_m,z_Radau_m')
        computed.append((slug,title,field,formula,ps,errs))
    def field_map(a):
        w=X/D+1j*Z/D;b=a[0]/(1+w)**2-a[1]/(1-w)**2
        return np.ma.masked_where(~mask,abs(b)*BS*1000)
    bmax=max(field_map(field(times[np.argmax(np.sum(np.array([field(t) for t in times])**2,axis=1))])).max() for _,_,field,_,_,_ in computed)
    def background(ax,kind,field=None,tstar=0):
        values=speed if kind=='flow' else field_map(field(tstar))
        image=ax.pcolormesh(Z*1000,X*1e6,values,cmap='viridis',vmin=0,vmax=30 if kind=='flow' else bmax,shading='auto',rasterized=True)
        ax.plot(z*1000,(center(z)+R)*1e6,color='#394552',lw=.8);ax.plot(z*1000,(center(z)-R)*1e6,color='#394552',lw=.8)
        if kind=='flow':
            for eta in np.linspace(-.8,.8,9):
                ax.plot(z*1000,(center(z)+eta*R)*1e6,color='white',alpha=.45,lw=.6)
                for zp in (-.06,-.03,0.):
                    dx=.002;ax.annotate('',xy=((zp+dx)*1000,(center(zp+dx)+eta*R)*1e6),xytext=(zp*1000,(center(zp)+eta*R)*1e6),arrowprops={'arrowstyle':'->','lw':.7,'color':'white','alpha':.6})
        return image
    def panel(ax,j,item,wide):
        slug,title,field,formula,ps,errs=item
        av=np.array([field(t) for t in times]);tstar=times[np.argmax((av*av).sum(1))]
        if j==0:
            ax.plot(times,av[:,0],color='#e58b00',lw=1.7,label='Left source');ax.plot(times,av[:,1],color='#e233ab',lw=1.7,label='Right source')
            ax.set(xlabel='Time (s)',ylabel='Source amplitude',xlim=(0,T),ylim=(-.72,.72));ax.legend(fontsize=7);ax.grid(alpha=.18)
        elif j in (1,2,3):
            im=background(ax,'magnetic' if j==1 else 'flow',field,tstar)
            cb=plt.colorbar(im,ax=ax,orientation='horizontal',pad=.21 if wide else .22,fraction=.07,aspect=28)
            cb.set_label(f'|B| (mT), snapshot t={tstar:.2f}s' if j==1 else 'Fluid speed (mm/s)',fontsize=8);cb.ax.tick_params(labelsize=7)
            if j==3:
                for i,(p,b) in enumerate(zip(ps,base)):
                    ax.plot(b[:,1]*1000,b[:,0]*1e6,'--',color=COLORS[i],lw=1.6)
                    ax.plot(p[:,1]*1000,p[:,0]*1e6,color=COLORS[i],lw=1.8)
                    ax.plot(p[::128,1]*1000,p[::128,0]*1e6,'o',mfc='white',mec=COLORS[i],ms=3)
            ax.set(xlabel='z (mm)',ylabel='x (µm)',xlim=(-75,25),ylim=(-105,105))
        elif j==4:
            for i,(p,b) in enumerate(zip(ps,base)):
                ax.plot(b[:,1]*1000,b[:,0]*1e6,'--',color=COLORS[i],lw=1.4,label='No B' if i==0 else None)
                ax.plot(p[:,1]*1000,p[:,0]*1e6,color=COLORS[i],lw=1.8,label='With B' if i==0 else None)
            ax.set(xlabel='z (mm)',ylabel='x (µm)',xlim=(-75,25),ylim=(-65,65));ax.grid(alpha=.18);ax.legend(fontsize=7)
        else:
            for i,(off,p) in enumerate(zip(OFFSETS,ps)):
                ax.plot(times,(p[:,0]-center(p[:,1])-off)*1e6,color=COLORS[i],label=f'Launch {i+1}',lw=1.7)
            ax.axhline(0,color='#444',lw=1,ls='--',label='All no-B cases');ax.set(xlabel='Time (s)',ylabel='Migration across streamlines (µm)',xlim=(0,T));ax.grid(alpha=.18);ax.legend(fontsize=7)
        ax.set_title(['01  Magnetic input','02  Magnetic field','03  Curved fluid flow','04  Particles in that flow','05  With / without B','06  Magnetic migration'][j],pad=10)
    with PdfPages(OUT/'curved_channel_slides.pdf') as pdf, PdfPages(OUT/'curved_channel_chains.pdf') as widepdf:
      for item in computed:
        slug,title,*_=item
        for wide in (True,False):
            fig,axes=plt.subplots(1,6,figsize=(27,5.2)) if wide else plt.subplots(2,3,figsize=(16,9))
            fig.subplots_adjust(left=.045,right=.985,top=.80 if wide else .85,bottom=.24 if wide else .15,wspace=.54 if wide else .37,hspace=.92)
            for j,ax in enumerate(axes.flat):panel(ax,j,item,wide)
            fig.suptitle('CURVED-CHANNEL LAUNCHES  |  '+title,fontsize=15 if wide else 19,color='#064565',weight='bold',y=.98)
            fig.text(.5,.90 if wide else .91,'NUMERICAL ODE DEMONSTRATION  ·  prescribed divergence-free flow  ·  fixed magnetic-source geometry',ha='center',fontsize=10,color='#566576')
            fig.text(.045,.05,'Solid: with B; dashed: without B. Orange / magenta / cyan: three identical paired releases. Open circles: equal time intervals.',fontsize=9,color='#064565')
            fig.text(.045,.012,'No-field paths curve with the flow. Not Master10 inference; this new flow has not been used to train that network. Spatial axes use different units.',fontsize=8,color='#566576')
            if wide:
                for xpos in (.186,.347,.507,.669,.831):fig.text(xpos,.94,'→',color='#c08012',fontsize=19)
                fig.savefig(OUT/(slug+'_chain.png'),dpi=220);fig.savefig(OUT/(slug+'_chain.svg'));widepdf.savefig(fig)
            else:
                fig.savefig(OUT/(slug+'_slide.png'),dpi=220);pdf.savefig(fig)
            plt.close(fig)
    deck=Presentation();deck.slide_width=Inches(16);deck.slide_height=Inches(9)
    for slug,*_ in computed:
        slide=deck.slides.add_slide(deck.slide_layouts[6]);slide.shapes.add_picture(str(OUT/(slug+'_slide.png')),0,0,width=deck.slide_width,height=deck.slide_height)
    deck.save(OUT/'curved_channel_presentation.pptx')
    report={'method':'Numerical DOP853 with Radau cross-check; NOT a PINN/DeepONet prediction',
       'flow':'Prescribed planar divergence-free streamfunction; not a Navier-Stokes/Maxwell solve',
       'centerline_m':'18e-6 sin(2pi(z+.075)/.075) + 8e-6 sin(4pi(z+.075)/.075)',
       'velocity':'u_z=2U(1-((x-c(z))/R)^2), u_x=c_prime(z)*u_z',
       'U_profile_scale_m_s':U,'note_U':'U is a profile scale retained from Master10; mean across this planar width is 4U/3, not U.',
       'duration_s':T,'release_local_offsets_m':OFFSETS.tolist(),'no_field_analytic_max_error_m':max(base_err),
       'finite_difference_max_divergence_per_s':float(abs(div).max()),
       'assumptions':'Overdamped induced dipole and Stokes drag; no Brownian/gravity/inertial launch transient.',
       'cases':[{'case':s,'title':title,'max_cross_solver_errors_um':errs,'all_passed':True} for s,title,_,_,_,errs in computed],
       'all_passed':True}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
