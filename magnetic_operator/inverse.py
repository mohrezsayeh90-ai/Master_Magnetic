"""Position-only recovery with frozen Master10 and a fixed differentiable clock."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json, time
import numpy as np
import torch
from scipy.optimize import minimize
from .physics import Parameters, R, D, TS, AMAX, HMIN, NSENSORS
from .inference import check_state

@dataclass(frozen=True)
class Config:
    h: float = .0125
    knots: int = 7
    scales: tuple = (5.5e-5, 4e-5)
    regularization: float = .1
    starts: int = 3
    iterations: int = 180
    seed: int = 731
    prior: str = 'signed'


def observations(t, q, q0, origin=0.):
    t=np.asarray(t,float);q=np.asarray(q,float);q0=check_state(q0)
    if t.ndim!=1 or len(t)<2 or q.shape!=(len(t),2) or not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError('Expected finite aligned time and two position columns')
    if origin!=0 or t[0]!=0 or np.any(np.diff(t)<=0):
        raise ValueError('Declare origin zero; times must start at zero and increase')
    if not np.allclose(q[0],q0,atol=1e-12,rtol=0):raise ValueError('Initial observation must equal the declared exact initial state')
    return t,q,q0


def read_csv(path, units, parameters):
    a=np.genfromtxt(path,delimiter=',',names=True)
    if units=='nondimensional' and a.dtype.names==('tau','X','Z'):
        t=a['tau'];q=np.column_stack([a['X'],a['Z']])
    elif units=='SI' and a.dtype.names==('time_s','x_m','z_m'):
        t=a['time_s']/TS;q=np.column_stack([a['x_m']/R,a['z_m']/D])
    else:raise ValueError('Explicit units and exact schema tau,X,Z or time_s,x_m,z_m required')
    if parameters.get('initial_convention')!='exact_first_observation' or parameters.get('time_origin')!=0:
        raise ValueError('Declare initial_convention=exact_first_observation and time_origin=0')
    par=Parameters(**{k:parameters[k] for k in ['eta','mean','rho']}).validate()
    t,q,q0=observations(t,q,q[0]);return t,q,q0,par


def interpolate(knots, values, times):
    return np.column_stack([np.interp(times,knots,values[:,i]) for i in range(2)])


class Rollout:
    def __init__(self, model, times, q0, parameters, knots, h=.0125):
        self.times=np.asarray(times,float);self.q0=check_state(q0);self.par=parameters.validate()
        if self.times.ndim!=1 or len(self.times)<2 or self.times[0]!=0 or not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0):raise ValueError('Invalid clock')
        if h not in (.0125,.00625):raise ValueError('Use supported fixed window .0125 or .00625')
        self.knots=np.asarray(knots,float)
        if not np.isfinite(self.knots).all() or len(self.knots)<2 or self.knots[0]!=0 or self.knots[-1]!=self.times[-1] or np.any(np.diff(self.knots)<=0):raise ValueError('Invalid knot clock')
        self.model=model.eval();self.h=h;self.schedule=[]
        for p in model.parameters():p.requires_grad_(False)
        self.dtype=next(model.parameters()).dtype
        if self.dtype!=torch.float64 or next(model.parameters()).device.type!='cpu':raise ValueError('Initial implementation requires float64 CPU')
        self.p=parameters.tensor(dtype=torch.float64)
        stop=self.times[-1]
        for j in range(int(np.ceil((stop-1e-12)/h))):
            start=j*h;end=min(stop,(j+1)*h);frac=(end-start)/h
            idx=np.flatnonzero((self.times>start+1e-12)&(self.times<=end+1e-12))
            query=np.r_[(self.times[idx]-start)/h,np.linspace(0,frac,9),frac]
            sensor=np.minimum(stop,start+np.linspace(0,1,NSENSORS)*h)
            k=np.clip(np.searchsorted(self.knots,sensor,side='right')-1,0,len(self.knots)-2)
            w=(sensor-self.knots[k])/(self.knots[k+1]-self.knots[k])
            self.schedule.append((idx,torch.tensor(query[:,None]),torch.tensor(k),torch.tensor(w[:,None])))

    def __call__(self, coefficients):
        q=torch.tensor(self.q0[None]);outputs=[q];states=[]
        for idx,s,k,w in self.schedule:
            a=coefficients[k]*(1-w)+coefficients[k+1]*w;n=len(s)
            out=self.model(s,a[None].expand(n,-1,-1),q.expand(n,-1),self.p.expand(n,-1),torch.full((n,1),self.h,dtype=torch.float64))
            outputs.append(out[:len(idx)]);states.append(out[len(idx):]);q=out[-1:]
        return torch.cat(outputs),torch.cat(states)

    def sampling_error(self, a):
        maximum=0.
        stop=self.times[-1]
        for j in range(len(self.schedule)):
            start=j*self.h;grid=np.minimum(stop,start+np.linspace(0,1,NSENSORS)*self.h)
            probes=np.unique(np.r_[grid,(grid[:-1]+grid[1:])/2,self.knots[(self.knots>=start)&(self.knots<=min(stop,start+self.h))]])
            true=interpolate(self.knots,a,probes);sample=interpolate(self.knots,a,grid)
            approx=np.column_stack([np.interp(probes,grid,sample[:,i]) for i in range(2)])
            maximum=max(maximum,float(np.max(np.abs(true-approx))))
        return maximum


def decode(x, prior):
    if prior=='left-positive':return torch.stack((x,torch.zeros_like(x)),1)
    if prior!='signed':raise ValueError('Unknown prior')
    v=x.reshape(-1,2);return torch.stack(((v[:,0]+v[:,1])/2,(v[:,0]-v[:,1])/2),1)


def feasible(q):
    q=np.asarray(q,float)
    return bool(np.isfinite(q).all() and np.all(np.abs(q[:,0])<=.7+1e-10) and np.all(q[:,1]>=-2-1e-10) and np.all(q[:,1]<=1.7+1e-10))


def fit(model,t,q,q0,parameters,config=Config()):
    t,q,q0=observations(t,q,q0);parameters.validate()
    if not np.isfinite(config.scales).all() or not np.isfinite(config.regularization) or config.knots<2 or config.starts<1 or config.iterations<1 or config.regularization<0 or len(config.scales)!=2 or min(config.scales)<=0:raise ValueError('Invalid inverse configuration')
    knots=np.linspace(0,t[-1],config.knots);roll=Rollout(model,t,q0,parameters,knots,config.h)
    target=torch.tensor(q);scales=torch.tensor(config.scales);dt=torch.tensor(np.diff(knots))
    def objective(x, details=False):
        x=torch.tensor(x,dtype=torch.float64,requires_grad=True);a=decode(x,config.prior)
        pred,states=roll(a)
        data=((pred-target)/scales).square().sum(1).mean()
        smooth=((a[1:]-a[:-1]).square().sum(1)/dt).sum()*config.regularization
        violation=torch.relu(states[:,0].abs()-.7).square().sum()+torch.relu(-2-states[:,1]).square().sum()+torch.relu(states[:,1]-1.7).square().sum()
        penalty=1e8*violation;loss=data+smooth+penalty
        grad=torch.autograd.grad(loss,x)[0]
        if details:return dict(objective=float(loss.detach()),data_term=float(data.detach()),regularization_term=float(smooth.detach()),domain_penalty=float(penalty.detach()))
        return float(loss.detach()),grad.detach().numpy()
    n=config.knots*(1 if config.prior=='left-positive' else 2)
    bounds=[(0,AMAX) if config.prior=='left-positive' else (-AMAX,AMAX)]*n
    rng=np.random.default_rng(config.seed);starts=[];solutions=[];begin=time.perf_counter()
    for i in range(config.starts):
        # Fixed nonzero starts, independent of the test command.
        initial=rng.uniform(.06,.4,n) if config.prior=='left-positive' else rng.uniform(-.4,.4,n)
        start=time.perf_counter();result=minimize(objective,initial,method='L-BFGS-B',jac=True,bounds=bounds,
            options=dict(maxiter=config.iterations,ftol=1e-11,gtol=1e-6,maxls=30,maxcor=15))
        coeff=decode(torch.tensor(result.x),config.prior).numpy()
        with torch.no_grad():pred,states=roll(torch.tensor(coeff))
        ok=feasible(states.numpy()) and np.abs(coeff).sum(1).max()<=AMAX+1e-10
        entry=dict(start=i,**objective(result.x,True),iterations=int(result.nit),evaluations=int(result.nfev),
            optimizer_converged=bool(result.success),reason=str(result.message),feasible=bool(ok),
            sampling_error=roll.sampling_error(coeff),seconds=time.perf_counter()-start,coefficients=coeff.tolist())
        starts.append(entry);solutions.append((coeff,pred.numpy()))
    candidates=[i for i,v in enumerate(starts) if v['feasible'] and v['sampling_error']<=.0005]
    if not candidates:raise ValueError('No feasible, resolved rollout; all candidates rejected')
    best=min(candidates,key=lambda i:starts[i]['objective'])
    return dict(knots=knots,coefficients=solutions[best][0],prediction=solutions[best][1],
        report=dict(config=asdict(config),selected_start=best,starts=starts,seconds=time.perf_counter()-begin,
                    initial_convention='exact_first_observation',time_origin=0,units='tau,X,Z',
                    rejection_rule='Require domain-feasible sampled rollout, amplitude bound, sampling error <=0.0005; independent replay required separately'))
