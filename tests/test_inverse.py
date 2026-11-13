"""Meaningful parity, propagation gradient, input, and physical symmetry checks."""
import unittest,tempfile,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from magnetic_operator.train import load
from magnetic_operator.physics import Parameters,R,D,TS,AMAX
from magnetic_operator.inference import predict,reference
from magnetic_operator.inverse import Rollout,observations,interpolate,read_csv,feasible

torch.set_num_threads(1)
class InverseChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.model=load('results/operator');cls.par=Parameters()
 def test_parity_and_partial(self):
  for t in [np.linspace(0,.05,13),np.array([0,.001,.0125,.019,.027,.043])]:
   knots=np.array([0,.025,t[-1]]);a=np.array([[.1,.2],[.2,-.1],[.12,.18]])
   r=Rollout(self.model,t,[.12,-1],self.par,knots)
   q,_=r(torch.tensor(a));p,_=predict(self.model,lambda s:interpolate(knots,a,[s/TS])[0],t*TS,self.par,[.12*R,-D],max_window_tau=.0125,sampling_tolerance=.0005)
   np.testing.assert_allclose(q.detach(),p/[R,D],atol=2e-14,rtol=0)
 def test_propagated_gradient(self):
  t=np.linspace(0,.075,9);a=np.array([[.11,.2],[.2,.13],[.15,.19]])
  r=Rollout(self.model,t,[.12,-1],self.par,np.linspace(0,t[-1],3))
  x=torch.tensor(a,requires_grad=True);v=r(x)[0][-1].sum();g=torch.autograd.grad(v,x)[0].numpy();fd=np.empty_like(a)
  for k in np.ndindex(a.shape):
   p=a.copy();m=a.copy();p[k]+=1e-6;m[k]-=1e-6
   fd[k]=(r(torch.tensor(p))[0][-1].sum()-r(torch.tensor(m))[0][-1].sum()).item()/2e-6
  np.testing.assert_allclose(g,fd,rtol=2e-5,atol=2e-9)
 def test_inputs_and_units(self):
  for t,q in [([0,0],[[0,-1],[0,-1]]),([0,1],[[0,-1],[np.nan,0]]),([1,2],[[0,-1],[0,-1]])]:
   with self.assertRaises(ValueError):observations(t,q,[0,-1])
  with self.assertRaises(ValueError):observations([0,1],[[.8,0],[0,0]],[.8,0])
  with self.assertRaises(ValueError):Parameters(mean=.1).validate()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.csv';np.savetxt(p,[[0,.12*R,-D],[TS*.1,.13*R,-.9*D]],delimiter=',',header='time_s,x_m,z_m',comments='')
   meta=dict(eta=self.par.eta,mean=self.par.mean,rho=self.par.rho,initial_convention='exact_first_observation',time_origin=0)
   t,q,_,_=read_csv(p,'SI',meta);np.testing.assert_allclose(t,[0,.1]);np.testing.assert_allclose(q,[[.12,-1],[.13,-.9]])
   with self.assertRaises(ValueError):read_csv(p,'nondimensional',meta)
 def test_zero_exit_and_bounds(self):
  t=np.linspace(0,.05,5);r=Rollout(self.model,t,[.12,-1],self.par,[0,.05]);q,_=r(torch.zeros((2,2),dtype=torch.float64))
  np.testing.assert_allclose(q[:,0],.12);np.testing.assert_allclose(q[:,1],-1+t*(1-.12**2))
  self.assertFalse(feasible([[.1,1.8]]))
  exit_roll=Rollout(self.model,[0,.025],[0,1.69],Parameters(mean=.03),[0,.025])
  _,states=exit_roll(torch.zeros((2,2),dtype=torch.float64));self.assertFalse(feasible(states.detach().numpy()))
  a=np.array([[AMAX,0],[0,-AMAX]]);self.assertLessEqual(np.max(np.abs(interpolate([0,1],a,np.linspace(0,1,101))).sum(1)),AMAX+1e-14)
 def test_physical_sign(self):
  t=np.linspace(0,.1,21);field=lambda s:np.array([.2,.1]);q0=[.12*R,-D]
  a=reference(field,t*TS,self.par,q0);b=reference(lambda s:-field(s),t*TS,self.par,q0)
  np.testing.assert_allclose(a,b,atol=1e-15,rtol=0)

if __name__=='__main__':unittest.main()
