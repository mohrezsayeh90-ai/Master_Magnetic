"""Predetermined synthetic/archived inverse suite; immutable forward artifacts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import argparse,json,time,hashlib,csv,dataclasses,traceback
import numpy as np
import torch
from magnetic_operator.train import load
from magnetic_operator.physics import Parameters,R,D,TS,numpy_field_force
from magnetic_operator.inference import reference
from magnetic_operator.inverse import Config,fit,Rollout,interpolate,feasible

def write_json(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def save(p,names,a):np.savetxt(p,a,delimiter=',',header=names,comments='')
def errors(a,b):
 d=np.asarray(a)-np.asarray(b)
 return dict(rmse=np.sqrt(np.mean(d*d,axis=0)).tolist(),maximum=np.max(np.abs(d),axis=0).tolist())
def replay(field,t,q0,par,step=.002):return reference(lambda s:field(s/TS),t*TS,par,np.array(q0)*[R,D],max_step_s=step)/[R,D]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run_case(model,out,name,field,q0,par,config,noise_um=0,seed=0,sparse=False,archive=None):
 folder=out/name;folder.mkdir(exist_ok=True);start=time.perf_counter()
 horizon=2.25 if archive else 1.5;t=np.linspace(0,horizon,121)
 if sparse:t=t[::4]
 if archive is None:clean=replay(field,t,q0,par)
 else:
  a=np.genfromtxt(ROOT/archive,delimiter=',',names=True);indices=np.linspace(0,len(a)-1,121).round().astype(int)
  t=.75*a['time_s'][indices];clean=np.column_stack([a['x_numerical_m'][indices]/R,a['z_numerical_m'][indices]/D])
 rng=np.random.default_rng(seed);sigma=noise_um*1e-6/np.array([R,D]);obs=clean+rng.normal(size=clean.shape)*sigma;obs[0]=q0
 save(folder/'observations.csv','tau,X_obs,Z_obs',np.column_stack([t,obs]))
 config=dataclasses.replace(config,scales=tuple(np.maximum(config.scales,sigma)))
 result=fit(model,t,obs,q0,par,config)
 write_json(folder/'fit_report.json',result['report'])
 save(folder/'knots.csv','tau,a_L_raw,a_R_raw',np.column_stack([result['knots'],result['coefficients']]))
 dense=np.linspace(0,horizon,601);truth=replay(field,dense,q0,par);truth_half=replay(field,dense,q0,par,.001)
 knots=result['knots'];raw=result['coefficients'];command=lambda t:interpolate(knots,raw,np.atleast_1d(t))[0]
 independent=replay(command,dense,q0,par);half=replay(command,dense,q0,par,.001)
 roll=Rollout(model,dense,q0,par,knots,config.h)
 with torch.no_grad():pred,states=roll(torch.tensor(raw))
 pred=pred.numpy();a_ref=np.array([field(v) for v in dense]);a_raw=interpolate(knots,raw,dense)
 sign=1 if np.linalg.norm(a_raw-a_ref)<=np.linalg.norm(-a_raw-a_ref) else -1
 if config.prior=='left-positive':sign=1
 aligned=sign*a_raw;den=np.linalg.norm(a_ref)
 raw_error=np.linalg.norm(a_raw-a_ref)/den;aligned_error=np.linalg.norm(aligned-a_ref)/den
 zero=replay(lambda t:np.zeros(2),dense,q0,par)
 field_ref=np.array([np.r_[*numpy_field_force(q,a)] for q,a in zip(truth,a_ref)])
 field_fit=np.array([np.r_[*numpy_field_force(q,a)] for q,a in zip(truth,aligned)])
 save(folder/'field_common_positions.csv','tau,X_ref,Z_ref,Bx_ref,Bz_ref,Fx_ref,Fz_ref,Bx_aligned,Bz_aligned,Fx_fit,Fz_fit',np.column_stack([dense,truth,field_ref,field_fit]))
 near=[]
 for candidate in result['report']['starts']:
  coeff=np.array(candidate['coefficients'])
  try:
   q=replay(lambda t:interpolate(knots,coeff,[t])[0],dense,q0,par)
   candidate_a=interpolate(knots,coeff,dense)
   aligned_error=min(np.linalg.norm(candidate_a-a_ref),np.linalg.norm(-candidate_a-a_ref))/np.linalg.norm(a_ref)
   near.append(dict(start=candidate['start'],replay=errors(q,truth),completed=True,
    waveform_global_aligned_relative_l2=float(aligned_error),objective=candidate['objective']))
  except ValueError as exc:near.append(dict(start=candidate['start'],completed=False,error=str(exc)))
 report=dict(result['report'],case=name,parameters=dataclasses.asdict(par),q0=q0,observation_count=len(t),noise_um=noise_um,noise_seed=seed,
  source=archive or 'DOP853 synthetic; truth amplitudes never supplied to fit',evaluation_count=len(dense),sign=sign,
  waveform_raw_relative_l2=float(raw_error),waveform_aligned_relative_l2=float(aligned_error),waveform_raw=errors(a_raw,a_ref),waveform_aligned=errors(aligned,a_ref),
  neural_vs_clean=errors(pred,truth),replay_vs_clean=errors(independent,truth),neural_vs_observed=errors(result['prediction'],obs),
  zero_vs_clean=errors(zero,truth),reference_refinement=errors(truth,truth_half),replay_refinement=errors(independent,half),
  field_aligned_at_reference=errors(field_fit,field_ref),field_scales='B/BS, F/(ALPHA*BS^2/D); evaluated at q_ref(tau)',
  independent_replay_complete=True,domain_feasible=feasible(states.numpy()) and feasible(independent),
  amplitude_margin=float(np.sqrt(.5)-np.abs(raw).sum(1).max()),state_X_margin=float(.7-np.abs(independent[:,0]).max()),
  near_best_replays=near,total_seconds=time.perf_counter()-start,
  error_definition='Component RMSE and max on stated grid; waveform relative norm is joint across sources/time; no per-coordinate relative position errors',
  status='completed',assessment='No numeric inverse pass gate; optimizer convergence, replay error, and waveform error reported separately')
 save(folder/'replay.csv','tau,X_ref,Z_ref,X_fit,Z_fit,X_replay,Z_replay',np.column_stack([dense,truth,pred,independent]))
 save(folder/'controls.csv','tau,a_L_ref,a_R_ref,a_L_fit_raw,a_R_fit_raw,a_L_fit_aligned,a_R_fit_aligned',np.column_stack([dense,a_ref,a_raw,aligned]))
 save(folder/'knots.csv','tau,a_L_raw,a_R_raw',np.column_stack([knots,raw]))
 write_json(folder/'report.json',report);return report

def ambiguity(out):
 p=out/'E_gap_sign';p.mkdir(exist_ok=True)
 knots=np.array([0,.25,.5,.75,1,1.25,1.5]);a=np.array([[0,0],[.3,.12],[0,0],[0,0],[0,0],[.18,.3],[0,0]])
 b=a.copy();b[4:]*=-1;t=np.linspace(0,1.5,601);par=Parameters();q0=[.12,-1]
 fa=lambda t:interpolate(knots,a,[t])[0];fb=lambda t:interpolate(knots,b,[t])[0]
 q=replay(fa,t,q0,par);r=replay(fb,t,q0,par);half=replay(fa,t,q0,par,.001)
 av=interpolate(knots,a,t);bv=interpolate(knots,b,t)
 mismatch=min(np.linalg.norm(av-bv),np.linalg.norm(av+bv))/np.linalg.norm(av)
 save(p/'controls.csv','tau,a_L_1,a_R_1,a_L_2,a_R_2',np.column_stack([t,av,bv]))
 save(p/'replay.csv','tau,X_1,Z_1,X_2,Z_2',np.column_stack([t,q,r]))
 report=dict(status='completed',case='E_gap_sign',replay_difference=errors(q,r),reference_refinement=errors(q,half),global_aligned_waveform_relative_l2=float(mismatch),
  meaning='Physical piecewise sign ambiguity across a common zero interval; not a surrogate-based identifiability claim')
 write_json(p/'report.json',report);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/operator_inverse');p.add_argument('--only');p.add_argument('--iterations',type=int,default=90);args=p.parse_args()
 torch.set_num_threads(1);out=args.output;out.mkdir(parents=True,exist_ok=True)
 cp=ROOT/'results/operator/checkpoint.pt';digest=sha(cp);assert digest=='080b7e5dd76c98a75fca7f92cc4ec314907a7f7340c809b739b1e2eabc0e3133'
 model=load(ROOT/'results/operator');knots=np.linspace(0,1.5,7)
 single=np.column_stack([[.12,.18,.28,.32,.25,.18,.12],np.zeros(7)])
 both=np.column_stack([[.12,.18,.27,.32,.25,.18,.12],[.30,.26,.20,.14,.18,.24,.31]])
 f=lambda a:lambda t:interpolate(knots,a,[t])[0]
 cfg=Config(iterations=args.iterations)
 cases=[('A_single',f(single),dataclasses.replace(cfg,prior='left-positive'),{}),('B_signed',f(both),cfg,{}),
  ('B_knots13',f(both),dataclasses.replace(cfg,knots=13,starts=2),{}),
  ('B_window_half',f(both),dataclasses.replace(cfg,h=.00625,starts=2),{}),
  ('B_weights',f(both),dataclasses.replace(cfg,scales=(5.5e-5,8e-5),starts=2),{}),
  ('B_lambda',f(both),dataclasses.replace(cfg,regularization=1.,starts=2),{}),
  ('C_switch',lambda t:(.25*(1-np.tanh((t*TS-1.5)/.15)),.25*(1+np.tanh((t*TS-1.5)/.15))),cfg,
   dict(archive='results/trained_presentation/05_source_switch.csv')),
  ('D_sparse',f(both),dataclasses.replace(cfg,starts=2),dict(sparse=True))]
 for noise in [.05,.2]:
  for seed in [1103,2907]:cases.append((f'D_noise_{noise}_{seed}',f(both),dataclasses.replace(cfg,starts=2),dict(noise_um=noise,seed=seed)))
 write_json(out/'predeclared_config.json',dict(checkpoint_sha256=digest,iterations=args.iterations,cases=[n for n,*_ in cases]+['E_gap_sign'],
  floor_scales=[5.5e-5,4e-5],floor_basis='Rounded-up maxima of the archived 12-case forward discrepancies (5.349e-5 X,3.257e-5 Z); fixed before target fitting',
  design='A development sanity check; B/C evaluation; D noise seeds fixed; no inverse numerical pass gate',
  objective='mean(sum(((q-q_obs)/max(noise_sigma,forward_floor))^2))+lambda*sum(||a[j+1]-a[j]||^2/dt)',
  physical_model='Original frozen straight-tube Master10',default_config=dataclasses.asdict(cfg)))
 results=[]
 for name,field,config,extra in cases:
  if args.only and name!=args.only:continue
  existing=out/name/'report.json'
  if existing.exists():
   saved=json.loads(existing.read_text())
   if saved.get('status')=='completed':
    expected=dataclasses.asdict(config);expected['scales']=np.maximum(config.scales,extra.get('noise_um',0)*1e-6/np.array([R,D])).tolist()
    if saved['config']!=json.loads(json.dumps(expected)):raise ValueError('Existing run configuration differs; select a fresh output directory')
   results.append(saved);continue
  try:
   r=run_case(model,out,name,field,[0.,-2] if name=='C_switch' else [.12,-1],Parameters(),config,**extra)
   print(name,'completed','waveform',r['waveform_aligned_relative_l2'],'seconds',r['total_seconds'],flush=True)
  except Exception as exc:
   r=dict(case=name,status='failed',error=str(exc),traceback=traceback.format_exc());(out/name).mkdir(exist_ok=True);write_json(out/name/'report.json',r);print(name,'FAILED',exc,flush=True)
  r['checkpoint_sha256']=digest;write_json(out/name/'report.json',r);results.append(r)
  assert sha(cp)==digest
  if not args.only:write_json(out/'summary.json',results)
 if not args.only or args.only=='E_gap_sign':results.append(ambiguity(out))
 if args.only:return
 write_json(out/'summary.json',results)
 with (out/'summary.csv').open('w') as file:
  keys=['case','status','waveform_aligned_relative_l2','replay_X_max','replay_Z_max','optimizer_converged']
  w=csv.DictWriter(file,fieldnames=keys);w.writeheader()
  for r in results:
   row={k:r.get(k,'') for k in keys};row['replay_X_max'],row['replay_Z_max']=r.get('replay_vs_clean',{}).get('maximum',['',''])
   row['optimizer_converged']=r.get('starts',[{}])[r.get('selected_start',0)].get('optimizer_converged','');w.writerow(row)
 assert sha(cp)==digest
if __name__=='__main__':main()
