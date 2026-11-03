"""Train/validate Master9 or follow a user CSV with explicit SI parameters."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json
import numpy as np
from master9.train import train,load
from master9.validate import run
from master9.control import demos,follow
from master9.physics import Parameters
ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/master9');ap.add_argument('--train',action='store_true');ap.add_argument('--require-gpu',action='store_true');ap.add_argument('--path');ap.add_argument('--Re',type=float,default=.8);ap.add_argument('--mean',type=float,default=.015);ap.add_argument('--rho',type=float,default=1000.);ap.add_argument('--x0',type=float,default=0.);ap.add_argument('--z0',type=float,default=-.08);ap.add_argument('--vx0',type=float,default=0.);ap.add_argument('--vz0',type=float,default=.03)
a=ap.parse_args();out=Path(a.output)
if a.train:train(out,flow_steps=6000,particle_steps=15000,require_gpu=a.require_gpu)
if a.path:
 p=Parameters.from_re(a.Re,mean=a.mean,rho=a.rho,x0=a.x0,z0=a.z0,vx0=a.vx0,vz0=a.vz0)
 m,_=load(out);data,report=follow(m.double(),p,target_csv=a.path)
 np.savetxt(out/'custom_path_control.csv',data,delimiter=',',header='time_s,x_PINN_m,z_PINN_m,target_x_m,u,x_oracle_m,z_oracle_m,target_at_oracle_m',comments='')
 (out/'custom_path_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 if not report['passed']:raise SystemExit('Target rejected: tracking/domain acceptance failed')
else:
 run(out,require_gpu=a.require_gpu);m,_=load(out);demos(m.double(),out)
