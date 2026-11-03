"""Generate inspectable physical plots from GPU-trained weights; no retraining."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from master9.train import load
from master9.physics import *
from master9.validate import transition
out=Path('results/master9');model,_=load(out);model=model.double()
fig,axs=plt.subplots(1,2,figsize=(11,4))
x=np.linspace(-1,1,201)
for mean in (.005,.015,.03):
 p=Parameters(mean=mean)
 with torch.no_grad():w=model.flow(torch.tensor((x*x)[:,None],dtype=torch.float64),p.tensor(dtype=torch.float64).expand(len(x),-1)).numpy()[:,0]
 axs[0].plot(x*RV*1e6,2*mean*w*1000,label=f'U={mean*1000:g} mm/s')
axs[0].set(xlabel='x (micrometres)',ylabel='Fluid axial speed (mm/s)',title='Residual-trained fluid profiles');axs[0].legend()
for eta in (.001,.006):
 p=Parameters(eta=eta,x0=0,z0=-.04,vx0=.001,vz0=.06);q,v=p.initial();times=np.geomspace(.001,20,90)*p.relaxation_s
 velocities=[]
 for t in times:
  _,vp=transition(model,q,v,0,p,h=t/TS,velocity=True);velocities.append(vp*np.array([RV,D])/TS)
 velocities=np.array(velocities)
 axs[1].semilogx(times*1e6,velocities[:,0]*1000,label=f'eta={eta:g} Pa.s')
axs[1].set(xlabel='Time (microseconds)',ylabel='Particle vx (mm/s)',title='Initial transverse velocity relaxation');axs[1].legend()
fig.tight_layout();fig.savefig(out/'fluid_and_initial_velocity.png',dpi=160)
# Derive magnetic quantities for each existing control schedule; positions are from independent Radau replay.
for source in sorted(out.glob('control_*.csv')):
 a=np.loadtxt(source,delimiter=',',skiprows=1);name=source.stem.removeprefix('control_')
 report=json.loads((out/'control_tests.json').read_text())[name]
 p=Parameters(**report['parameters']);u=a[:,4];xx=a[:,5]/RV;zz=a[:,6]/D;side=np.where(u<0,1.,-1.)
 y=1+side*xx*RV/D;r2=y*y+zz*zz
 bmag=BS/r2*np.sqrt(abs(u));force=-2*np.column_stack((side*y,zz))/r2[:,None]**3*abs(u[:,None])*ALPHA*BS**2/D
 table=np.column_stack((a[:,0],u,side,np.sqrt(abs(u)),bmag,force))
 np.savetxt(out/f'magnetic_{name}.csv',table,delimiter=',',header='time_s,u,source_side_minus1_is_right,source_amplitude_scale,B_magnitude_T,Fmag_x_N,Fmag_z_N',comments='')
print('Physical plots and magnetic force schedules exported')
