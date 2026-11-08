"""Six original and six additional inputs, all inside Master10's trained domain."""
import numpy as np
from presentation import CASES
from magnetic_operator.physics import R,Parameters

def alternating(t):
    s=np.sin(2*np.pi*.9*t)
    return .275*(1+s),.275*(1-s)

def pulse_train(t):
    g=lambda c:np.exp(-((t-c)/.14)**2)
    return .54*(g(.45)+g(1.85)),.54*(g(1.15)+g(2.5))

def dual_chirp(t):
    return .28*np.sin(2*np.pi*(.6*t+.5*t*t)),.28*np.cos(2*np.pi*(.3*t+.2*t*t)+.8)

def multisine(t):
    return (.18*(np.sin(2*np.pi*.6*t)+.4*np.sin(2*np.pi*2.2*t)+.2*np.sin(2*np.pi*4.1*t)),
            .22*(np.cos(2*np.pi*.9*t)+.3*np.cos(2*np.pi*3.1*t)))

def soft_switch(t):
    s=.5*(1+np.tanh(3*np.sin(2*np.pi*1.2*t)))
    envelope=.6*(.8+.2*np.cos(2*np.pi*.4*t))
    return envelope*(1-s),envelope*s

def phase_chirp(t):
    p=2*np.pi*(.4*t+.18*t*t)
    return .28*np.sin(p),-.28*np.sin(p+np.pi/3)

def cases():
    out=[dict(slug=s,title=n,field=f,parameters=Parameters(),initial=(0.,-.08),duration=3.,group='Original') for s,n,f,_ in CASES]
    extra=[
      ('07_offset_alternating','Off-axis release · alternating attraction',alternating,Parameters(eta=.002,mean=.008),(-.3*R,-.04),3.2),
      ('08_alternating_pulses','Shifted release · alternating pulse train',pulse_train,Parameters(eta=.0015,mean=.01),(.35*R,-.045),2.8),
      ('09_slow_dual_chirp','Slow flow · two-source chirps',dual_chirp,Parameters(eta=.001,mean=.005),(.15*R,-.025),3.),
      ('10_fast_multisine','Fast flow · multi-frequency excitation',multisine,Parameters(eta=.0035,mean=.025),(-.4*R,-.075),1.8),
      ('11_downstream_switch','Downstream release · repeated source switching',soft_switch,Parameters(eta=.0022,mean=.008),(.25*R,.02),2.),
      ('12_phase_interference','Shifted release · phase-controlled interference',phase_chirp,Parameters(eta=.0045,mean=.012),(-.25*R,-.06),3.2),
    ]
    out.extend(dict(slug=s,title=n,field=f,parameters=p,initial=i,duration=t,group='Additional') for s,n,f,p,i,t in extra)
    return out
