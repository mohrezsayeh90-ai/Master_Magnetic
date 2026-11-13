"""Recover two Master10 commands from explicitly dimensioned observed positions."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib,dataclasses
import numpy as np
import torch
from magnetic_operator.train import load
from magnetic_operator.inverse import Config,read_csv,fit,interpolate
from magnetic_operator.inference import reference
from magnetic_operator.physics import R,D,TS

def main():
 p=argparse.ArgumentParser();p.add_argument('--observations',type=Path,required=True);p.add_argument('--units',choices=['SI','nondimensional'],required=True)
 p.add_argument('--parameters',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--prior',choices=['signed','left-positive'],default='signed');p.add_argument('--knots',type=int,default=7)
 p.add_argument('--iterations',type=int,default=180);p.add_argument('--starts',type=int,default=3);args=p.parse_args()
 if args.output.exists() and any(args.output.iterdir()):p.error('Output must be empty; preserve previous fits')
 torch.set_num_threads(1);meta=json.loads(args.parameters.read_text());t,q,q0,par=read_csv(args.observations,args.units,meta)
 scales=tuple(np.maximum([5.5e-5,4e-5],meta.get('noise_sigma_XZ',[0,0])))
 cp=ROOT/'results/operator/checkpoint.pt';digest=hashlib.sha256(cp.read_bytes()).hexdigest()
 if digest!='080b7e5dd76c98a75fca7f92cc4ec314907a7f7340c809b739b1e2eabc0e3133':raise ValueError('Unexpected model checkpoint')
 result=fit(load(cp.parent),t,q,q0,par,Config(prior=args.prior,knots=args.knots,iterations=args.iterations,starts=args.starts,scales=scales))
 field=lambda seconds:interpolate(result['knots'],result['coefficients'],[seconds/TS])[0]
 replay=reference(field,t*TS,par,q0*[R,D])/[R,D]
 refined=reference(field,t*TS,par,q0*[R,D],max_step_s=.001)/[R,D]
 args.output.mkdir(parents=True,exist_ok=True)
 for name,header,values in [('observations','tau,X_obs,Z_obs',np.column_stack([t,q])),
  ('replay','tau,X_fit,Z_fit,X_replay,Z_replay',np.column_stack([t,result['prediction'],replay])),
  ('knots','tau,a_L_raw,a_R_raw',np.column_stack([result['knots'],result['coefficients']]))]:
  np.savetxt(args.output/(name+'.csv'),values,delimiter=',',header=header,comments='')
 report=dict(result['report'],checkpoint_sha256=digest,parameters=dataclasses.asdict(par),
  replay_maximum_observed=np.max(np.abs(replay-q),axis=0).tolist(),
  reference_refinement=np.max(np.abs(replay-refined),axis=0).tolist(),independent_replay_complete=True,
  reference_waveform_available=False,sign_alignment='Unavailable for measured data; preserve raw coefficients')
 (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 assert hashlib.sha256(cp.read_bytes()).hexdigest()==digest
if __name__=='__main__':main()
