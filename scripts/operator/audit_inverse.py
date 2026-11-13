"""Audit saved outputs against their reports without training or integration."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=ROOT/'results/operator_inverse';results=json.loads((p/'summary.json').read_text());cfg=json.loads((p/'predeclared_config.json').read_text())
 assert {r['case'] for r in results}==set(cfg['cases'])
 assert hashlib.sha256((ROOT/'results/operator/checkpoint.pt').read_bytes()).hexdigest()==cfg['checkpoint_sha256']
 count=0
 for r in results:
  if r['status']=='failed':continue
  folder=p/r['case'];q=np.genfromtxt(folder/'replay.csv',delimiter=',',names=True)
  a=np.genfromtxt(folder/'controls.csv',delimiter=',',names=True)
  if r['case']=='E_gap_sign':
   difference=np.column_stack([q['X_1']-q['X_2'],q['Z_1']-q['Z_2']])
   assert np.max(np.abs(difference))<=max(r['reference_refinement']['maximum'])+1e-10
   assert r['global_aligned_waveform_relative_l2']>.1
   continue
  assert r['independent_replay_complete'] and r['domain_feasible'] and r['amplitude_margin']>=-1e-12
  for key,prefix in [('neural_vs_clean','fit'),('replay_vs_clean','replay')]:
   d=np.column_stack([q[f'{c}_{prefix}']-q[f'{c}_ref'] for c in ['X','Z']])
   np.testing.assert_allclose(np.sqrt(np.mean(d*d,axis=0)),r[key]['rmse'],rtol=1e-10,atol=1e-14)
   np.testing.assert_allclose(np.max(np.abs(d),axis=0),r[key]['maximum'],rtol=1e-10,atol=1e-14)
  ref=np.column_stack([a['a_L_ref'],a['a_R_ref']]);raw=np.column_stack([a['a_L_fit_raw'],a['a_R_fit_raw']]);aligned=r['sign']*raw
  assert np.max(np.abs(raw).sum(1))<=np.sqrt(.5)+1e-12
  np.testing.assert_allclose(np.linalg.norm(aligned-ref)/np.linalg.norm(ref),r['waveform_aligned_relative_l2'],rtol=1e-12)
  assert len(r['near_best_replays'])==len(r['starts'])
  if r['case']=='A_single':assert np.linalg.norm(r['replay_vs_clean']['rmse'])<np.linalg.norm(r['zero_vs_clean']['rmse'])
  count+=1
 report=dict(cases=len(results),inverse_cases_audited=count,failed=[r['case'] for r in results if r['status']=='failed'],
             checkpoint_unchanged=True,all_predetermined_cases_present=True)
 (p/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
