#!/usr/bin/env python3
"""Reproduce the synthetic experiments, figures and tables of the revised article.

Run from any directory: python code/reproduce.py
No observational data or fitted regional parameters are used.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import csv, json, math, platform, time
import numpy as np
import scipy
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
for sub in ['data','figures','tables']:(ROOT/sub).mkdir(exist_ok=True)
SEED=27102026
TGRID=np.linspace(0,30,3001)
COLORS=['#315A7D','#23836F','#C06E36','#825B9A','#AF414B']
NAMES=['S0 reference','S1 accelerated adoption','S2 delayed transition','S3 demand control','S4 supply disruption']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'axes.titlesize':9,
 'axes.labelsize':8.4,'legend.fontsize':7.1,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'ps.fonttype':42,
 'savefig.bbox':'tight','savefig.pad_inches':.035})

@dataclass(frozen=True)
class Parameters:
 a:float=1.2
 h:float=.4
 c0:float=.8
 d0:float=.5374
 beta:float=.834
 ell:float=.08
 eta:float=.6
 k:float=.3
 chi:float=.025
 eps:float=.05
 omega:float=.8
 mu:float=.15
 sigma:float=.4
 z0:float=.15

def response(x,p):return x/((p.ell+x)*(1+p.beta*x))

def derivatives_r(x,p,n=3):
 """Exact Taylor division for r and its first n derivatives; no finite differencing."""
 den=[(p.ell+x)*(1+p.beta*x),1+p.beta*p.ell+2*p.beta*x,p.beta]
 co=[]
 for j in range(n+1):
  num=x if j==0 else (1. if j==1 else 0.)
  co.append((num-sum(den[k]*co[j-k] for k in (1,2) if j>=k))/den[0])
 return np.array([co[j]*math.factorial(j) for j in range(n+1)])

def policy(t,scenario,p):
 P,S,mu=.2,.15,p.mu
 if scenario==1:P,S=1.,.7
 if scenario==2:P,S,mu=0.,.03,p.mu+.10
 if scenario==4 and 8<=t<14:S,mu=0.,p.mu+.90
 return P,S,mu

def H(t,z,p,scenario=0):
 P,S,mu=policy(t,scenario,p)
 return z*(1-z)*(1+p.omega*P)-mu*z+p.sigma*S*(1-z)

def fast(q,z,p):
 x,y=q;c=p.c0*(1-p.eta*z);D=p.h*c+p.d0+p.k*z
 return np.array([(p.a+p.chi*z)*x-p.a*p.beta*x*x-p.h*y*response(x,p),c*p.a*x-D*y])

def jacobian(q,z,p):
 x,y=q;r0,r1=derivatives_r(x,p,1);c=p.c0*(1-p.eta*z);D=p.h*c+p.d0+p.k*z
 return np.array([[p.a+p.chi*z-2*p.a*p.beta*x-p.h*y*r1,-p.h*r0],[c*p.a,-D]])

def equilibrium(z,p):
 c=p.c0*(1-p.eta*z);D=p.h*c+p.d0+p.k*z;K=p.h*c*p.a/D
 upper=(p.a+p.chi*z)/(p.a*p.beta)
 x=brentq(lambda x:p.a+p.chi*z-p.a*p.beta*x-K*response(x,p),0,upper,xtol=2e-14)
 return np.array([x,c*p.a*x/D])

def qz_and_lag(z,p):
 q=equilibrium(z,p);A=jacobian(q,z,p);x,y=q
 fz=np.array([p.chi*x,-p.c0*p.eta*(p.a*x-p.h*y)-p.k*y])
 qz=-np.linalg.solve(A,fz)
 return q,qz,np.linalg.solve(A,qz)

class Branch:
 def __init__(self,p,n=251):
  self.p=p;self.z=np.linspace(0,1,n)
  vals=[qz_and_lag(z,p) for z in self.z]
  self.q=CubicSpline(self.z,np.array([v[0] for v in vals]),axis=0)
  self.lag=CubicSpline(self.z,np.array([v[2] for v in vals]),axis=0)
  self.alpha=min(-np.max(np.linalg.eigvals(jacobian(v[0],z,p)).real) for z,v in zip(self.z,vals))

def scenario_parameters(p,s):return replace(p,beta=p.beta*1.12) if s==3 else p

def integrate(fun,initial,p,scenario,tgrid,dimension,rtol=2e-10,atol=2e-12):
 """Restart at dated discontinuities; do not let an adaptive step straddle a jump."""
 breaks=[tgrid[0]]+([8.,14.] if scenario==4 and tgrid[-1]>14 else [])+[tgrid[-1]]
 result=np.empty((len(tgrid),dimension));u=np.array(initial,dtype=float)
 for lo,hi in zip(breaks[:-1],breaks[1:]):
  # Freeze the side of the discontinuity consistently at both endpoints.
  midpoint=(lo+hi)/2
  def f(t,u):
   tt=t
   if scenario==4 and (t==lo or t==hi):tt=np.nextafter(t,midpoint)
   return fun(tt,u)
  sol=solve_ivp(f,(lo,hi),u,method='DOP853',rtol=rtol,atol=atol,dense_output=True)
  if not sol.success:raise RuntimeError(sol.message)
  ix=np.flatnonzero((tgrid>=lo)&(tgrid<=hi));result[ix]=sol.sol(tgrid[ix]).T
  u=sol.y[:,-1]
 return result

def full(p,scenario=0,initial=None,tgrid=TGRID,rtol=2e-10,atol=2e-12):
 initial=np.r_[equilibrium(p.z0,p),p.z0] if initial is None else np.array(initial)
 return integrate(lambda t,u:np.r_[fast(u[:2],u[2],p),p.eps*H(t,u[2],p,scenario)],
   initial,p,scenario,tgrid,3,rtol,atol)

def reduced(p,scenario=0,initial=None,tgrid=TGRID,order=1,branch=None):
 initial=np.r_[equilibrium(p.z0,p),p.z0] if initial is None else np.array(initial)
 b=branch or Branch(p)
 z=integrate(lambda t,u:[p.eps*H(t,u[0],p,scenario)],[initial[2]],p,scenario,tgrid,1)[:,0]
 qs=b.q(z);lag=p.eps*b.lag(z)*np.array([H(t,zz,p,scenario) for t,zz in zip(tgrid,z)])[:,None]
 q=qs+(lag if order else 0)
 # Composite leading initial layer, with frozen A(0), evaluated vectorially.
 u0=initial[:2]-q[0]
 A=jacobian(equilibrium(initial[2],p),initial[2],p)
 eig,V=np.linalg.eig(A);v0=np.linalg.solve(V.astype(complex),u0.astype(complex))
 transient=np.real((np.exp(np.outer(tgrid-tgrid[0],eig))*v0)@V.T)
 q=q+transient
 return np.column_stack([q,z])

def errors(u,v):
 diff=u[:,:2]-v[:,:2]
 return {'E':float(np.max(np.linalg.norm(diff,axis=1))/np.max(np.linalg.norm(u[:,:2],axis=1))),
  'max_X':float(np.max(abs(diff[:,0]))),'max_Y':float(np.max(abs(diff[:,1]))),
  'rmse_X':float(np.sqrt(np.mean(diff[:,0]**2))),'rmse_Y':float(np.sqrt(np.mean(diff[:,1]**2)))}

def metrics(u,t=TGRID):
 return {'X30':float(u[-1,0]),'Y30':float(u[-1,1]),'Z30':float(u[-1,2]),
  'I30':float(u[-1,1]/u[-1,0]),'CX':float(trapezoid(u[:,0],t)),
  'CY':float(trapezoid(u[:,1],t)),'min_state':float(u.min()),
  'max_Z':float(u[:,2].max())}

def write_csv(name,rows,fields=None):
 rows=list(rows)
 with (ROOT/'data'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def savefig(name):
 plt.savefig(ROOT/'figures'/f'{name}.pdf')
 plt.savefig(ROOT/'figures'/f'{name}.png',dpi=220)
 plt.close()

def hopf_data():
 p=Parameters(a=1.,h=.5,c0=.8,d0=.01,beta=1.,ell=.01,eta=0,k=0,chi=0)
 q=equilibrium(0,p);x=q[0];D=p.h*p.c0+p.d0
 L=jacobian(q,0,p)[0,0]
 aH=D/L;p=replace(p,a=aH);q=equilibrium(0,p);A=jacobian(q,0,p)
 vals,V=np.linalg.eig(A);idx=np.argmax(vals.imag);omega=vals[idx].imag
 v=V[:,idx]/V[0,idx];adj=np.linalg.inv(np.column_stack([v,v.conj()]))[0]
 _,r1,r2,r3=derivatives_r(q[0],p)
 fxx=-2*p.a*p.beta-p.h*q[1]*r2;fxy=-p.h*r1
 fxxx=-p.h*q[1]*r3;fxxy=-p.h*r2
 def B(u,v):return np.array([fxx*u[0]*v[0]+fxy*(u[0]*v[1]+u[1]*v[0]),0j])
 def C(u,v,w):return np.array([fxxx*u[0]*v[0]*w[0]+fxxy*(u[0]*v[0]*w[1]+u[0]*v[1]*w[0]+u[1]*v[0]*w[0]),0j])
 h20=np.linalg.solve(2j*omega*np.eye(2)-A,B(v,v));h11=-np.linalg.solve(A,B(v,v.conj()))
 c3=.5*(adj@(C(v,v,v.conj())+B(v.conj(),h20)+2*B(v,h11)))
 M=np.linalg.det(A)/aH
 Delta=.2;a1=-2*Delta/L;omega1=a1*M/(2*omega)
 return dict(p=p,q=q,A=A,L=L,M=M,aH=aH,omega=omega,v=v,adj=adj,h20=h20,
             h11=np.real_if_close(h11).real,c3=c3,Delta=Delta,a1=a1,omega1=omega1)

def nf_map(w,hd):
 w=np.asarray(w)
 return 2*np.real(w[...,None]*hd['v'])+np.real(w[...,None]**2*hd['h20'])+(abs(w)**2)[...,None]*hd['h11']

def nf_inverse(u,hd):
 w=u@hd['adj'];v=hd['v'];h20=hd['h20'];h11=hd['h11']
 for _ in range(6):
  err=nf_map(w,hd)-u;wr=w.real;wi=w.imag
  J0=2*v.real+2*wr[:,None]*h20.real-2*wi[:,None]*h20.imag+2*wr[:,None]*h11
  J1=-2*v.imag-2*wi[:,None]*h20.real-2*wr[:,None]*h20.imag+2*wi[:,None]*h11
  det=J0[:,0]*J1[:,1]-J0[:,1]*J1[:,0]
  dr=(err[:,0]*J1[:,1]-err[:,1]*J1[:,0])/det
  di=(J0[:,0]*err[:,1]-J0[:,1]*err[:,0])/det
  w=w-dr-1j*di
 return w

def rg_experiment(hd,eps=.02,R0=.2,slow_horizon=1.5,n=4001):
 p=replace(hd['p'],a=hd['aH']+eps*hd['a1'])
 qe=equilibrium(0,p);t=np.linspace(0,slow_horizon/eps,n);T=eps*t
 w0=np.sqrt(eps)*R0;initial=qe+nf_map(np.array(w0),hd)
 sol=solve_ivp(lambda t,u:fast(u,0,p),(0,t[-1]),initial,t_eval=t,
                method='DOP853',rtol=2e-11,atol=2e-13)
 if not sol.success:raise RuntimeError(sol.message)
 def fun(T,u):
  a=u[0]+1j*u[1]
  da=(-hd['Delta']+1j*hd['omega1'])*a+hd['c3']*abs(a)**2*a
  return [da.real,da.imag]
 def leave(T,u):return .60-np.linalg.norm(u)
 leave.terminal=True;leave.direction=-1
 r=solve_ivp(fun,(0,T[-1]),[R0,0],t_eval=T,events=leave,rtol=2e-11,atol=2e-13)
 if len(r.t)!=len(T):return dict(rejected=True,eps=eps,R0=R0,exit_slow_time=float(r.t[-1]))
 amp=r.y[0]+1j*r.y[1];w=np.sqrt(eps)*amp*np.exp(1j*hd['omega']*t)
 qr=qe+nf_map(w,hd)
 lin=np.sqrt(eps)*R0*np.exp((-hd['Delta']+1j*hd['omega1'])*T+1j*hd['omega']*t)
 qlin=qe+nf_map(lin,hd)
 truew=nf_inverse(sol.y.T-qe,hd)
 E=lambda q:float(np.max(np.linalg.norm(sol.y.T-q,axis=1))/np.max(np.linalg.norm(sol.y.T,axis=1)))
 return dict(rejected=False,eps=eps,R0=R0,t=t,T=T,full=sol.y.T,rg=qr,linear=qlin,
  amplitude=amp,full_amplitude=truew/np.sqrt(eps),E=E(qr),E_linear=E(qlin),p=p)

def main():
 p=Parameters()
 # Set the reference initial equilibrium to (1,1,z0) exactly.
 beta=brentq(lambda b:p.a*(1-b)+p.chi*p.z0-p.h/((p.ell+1)*(1+b)),.1,.99)
 p=replace(p,beta=beta)
 assert np.max(abs(equilibrium(p.z0,p)-1))<1e-12
 start=time.perf_counter();initial=np.array([1.,1.,p.z0]);fulls=[];reds=[];branches=[];rows=[]
 for s in range(5):
  pp=scenario_parameters(p,s);b=Branch(pp);u=full(pp,s,initial);v=reduced(pp,s,initial,branch=b)
  fulls.append(u);reds.append(v);branches.append(b)
  mm=metrics(u);er=errors(u,v);rows.append(dict(scenario=s,**mm,**er,alpha=b.alpha))
  write_csv(f'scenario_S{s}.csv',[
   dict(t=float(t),X=float(a[0]),Y=float(a[1]),Z=float(a[2]),X_reduced=float(bv[0]),Y_reduced=float(bv[1]),Z_reduced=float(bv[2]))
   for t,a,bv in zip(TGRID,u,v)])
 for row in rows:
  row['avoided_fraction']=1-row['CY']/rows[0]['CY'];row['service_ratio']=row['CX']/rows[0]['CX']
 write_csv('scenario_summary.csv',rows)
 print('Scenario summaries',json.dumps(rows,indent=2),flush=True)

 # First-order versus leading-order convergence, over a fixed slow horizon.
 convergence=[]
 for ep in [.005,.01,.02,.04,.08,.16]:
  pp=replace(p,eps=ep);t=np.linspace(0,1.5/ep,4001);u=full(pp,0,initial,t)
  e0=errors(u,reduced(pp,0,initial,t,0,branches[0]))['E']
  e1=errors(u,reduced(pp,0,initial,t,1,branches[0]))['E']
  convergence.append(dict(eps=ep,E0=e0,E1=e1))
 write_csv('slow_manifold_convergence.csv',convergence)

 # Measured error grid: vary physical initial displacement and scale ratio.
 epsgrid=np.array([.01,.02,.04,.08,.16,.25]);rgrid=np.array([0,.05,.1,.2,.35,.5])
 errgrid=np.zeros((len(rgrid),len(epsgrid)));gridrows=[]
 for j,R0 in enumerate(rgrid):
  for i,ep in enumerate(epsgrid):
   pp=replace(p,eps=ep);t=np.linspace(0,1.5/ep,2001);ini=initial.copy();ini[:2]+=R0*np.array([.8,-.6])
   u=full(pp,0,ini,t);v=reduced(pp,0,ini,t,1,branches[0]);E=errors(u,v)['E'];errgrid[j,i]=E
   gridrows.append(dict(eps=ep,R0=R0,E=E))
 write_csv('slow_manifold_error_grid.csv',gridrows)

 # Paired scenario ensemble: specified parameter perturbations, not posterior samples.
 rng=np.random.default_rng(SEED);ensemble=[];rejection_count=0;rank_changes=0;feas_changes=0
 for rep in range(128):
  factors=np.exp(rng.uniform(np.log(.85),np.log(1.15),6))
  pp=replace(p,a=p.a*factors[0],h=p.h*factors[1],c0=p.c0*factors[2],
   d0=p.d0*factors[3],eta=p.eta*factors[4],k=p.k*factors[5])
  draw=[];draw_reduced=[]
  for s in range(5):
   ps=scenario_parameters(pp,s);br=Branch(ps,n=101)
   u=full(ps,s,initial,tgrid=np.linspace(0,30,1001));v=reduced(ps,s,initial,tgrid=np.linspace(0,30,1001),branch=br)
   em=errors(u,v)['E'];mf=metrics(u,np.linspace(0,30,1001));mr=metrics(v,np.linspace(0,30,1001))
   draw.append(mf);draw_reduced.append(mr)
   if em>.01:rejection_count+=1
   ensemble.append(dict(draw=rep,scenario=s,E=em,CY=mf['CY'],CX=mf['CX'],
    CY_reduced=mr['CY'],CX_reduced=mr['CX'],alpha=br.alpha))
  for j in range(5):
   ensemble[-5+j]['avoided_fraction']=1-draw[j]['CY']/draw[0]['CY']
   ensemble[-5+j]['service_ratio']=draw[j]['CX']/draw[0]['CX']
  rank_changes+=int(not np.array_equal(np.argsort([r['CY'] for r in draw]),np.argsort([r['CY'] for r in draw_reduced])))
  feas_changes+=int(any((draw[j]['CX']/draw[0]['CX']>=.95)!=(draw_reduced[j]['CX']/draw_reduced[0]['CX']>=.95) for j in range(5)))
 write_csv('paired_parameter_ensemble.csv',ensemble)
 ensstats=[]
 for s in range(5):
  rr=[r for r in ensemble if r['scenario']==s]
  ensstats.append(dict(scenario=s,Emax=max(r['E'] for r in rr),
   avoided_q05=float(np.quantile([r['avoided_fraction'] for r in rr],.05)),
   avoided_q50=float(np.median([r['avoided_fraction'] for r in rr])),
   avoided_q95=float(np.quantile([r['avoided_fraction'] for r in rr],.95)),
   service_q05=float(np.quantile([r['service_ratio'] for r in rr],.05)),
   service_q50=float(np.median([r['service_ratio'] for r in rr])),
   service_q95=float(np.quantile([r['service_ratio'] for r in rr],.95))))
 write_csv('ensemble_summary.csv',ensstats)
 print('Ensemble complete',flush=True)

 # RG normal form: a verified frozen-technology near-Hopf family.
 hd=hopf_data();rg=rg_experiment(hd);rgsweep=[]
 for ep in [.005,.01,.02,.04,.08]:
  for R0 in [.05,.1,.2,.28,.35]:
   rr=rg_experiment(hd,ep,R0,n=2001)
   rgsweep.append(dict(eps=ep,R0=R0,rejected=int(rr['rejected']),
      E=rr.get('E',np.nan),E_linear=rr.get('E_linear',np.nan),exit_slow_time=rr.get('exit_slow_time',np.nan)))
 write_csv('rg_error_grid.csv',rgsweep)
 write_csv('rg_worked_trajectory.csv',[
  dict(t=float(t),T=float(T),X=float(q[0]),Y=float(q[1]),X_RG=float(qr[0]),Y_RG=float(qr[1]),
   X_linear=float(ql[0]),Y_linear=float(ql[1]),R_full=float(abs(af)),R_RG=float(abs(ar)))
  for t,T,q,qr,ql,af,ar in zip(rg['t'],rg['T'],rg['full'],rg['rg'],rg['linear'],rg['full_amplitude'],rg['amplitude'])])

 # Numerical reference-solution convergence and interpolation accuracy.
 convergence_solver=[]
 for s in range(5):
  uu=full(scenario_parameters(p,s),s,initial,rtol=2e-12,atol=2e-14)
  convergence_solver.append(errors(uu,fulls[s])['E'])
 probe=np.linspace(.001,.999,50)
 interpolation_error=max(np.linalg.norm(branches[0].q(z)-equilibrium(z,p)) for z in probe)

 # Runtime: online evaluation and setup kept separate, same output grid and tolerances.
 timings=[]
 for _ in range(11):
  t0=time.perf_counter();full(p,0,initial);tf=time.perf_counter()-t0
  t0=time.perf_counter();reduced(p,0,initial,branch=branches[0]);tr=time.perf_counter()-t0
  timings.append((tf,tr))
 t0=time.perf_counter();Branch(p);setup=time.perf_counter()-t0
 ft=float(np.median(np.array(timings)[1:,0]));rt=float(np.median(np.array(timings)[1:,1]))

 # Figure 1 uses only dependencies in the printed equations.
 from render_diagram import render
 render()

 # Figure 2: full trajectories and first-order reconstructions from the same arrays.
 fig,axs=plt.subplots(2,2,figsize=(7.2,5.1),sharex=True)
 labels=['Passenger activity $X$','Annual emissions $Y$','Adoption $Z$','Emissions intensity $Y/X$']
 for s,(u,v) in enumerate(zip(fulls,reds)):
  for j in range(4):
   a=axs.flat[j];fullval=u[:,j] if j<3 else u[:,1]/u[:,0]
   redval=v[:,j] if j<3 else v[:,1]/v[:,0]
   a.plot(TGRID,fullval,color=COLORS[s],lw=1.45,label=NAMES[s])
   if j!=2:a.plot(TGRID,redval,color=COLORS[s],lw=.8,ls='--')
 for j,a in enumerate(axs.flat):
  a.set_ylabel(labels[j]);a.set_title(f'({chr(97+j)})',loc='left')
  if j>=2:a.set_xlabel('Planning time (years; $T_f=1$ year)')
 axs[0,0].legend(loc='best',frameon=False,fontsize=6.6)
 fig.tight_layout();savefig('figure2')

 # Figure 3: actual convergence and actual error sweep.
 fig,axs=plt.subplots(1,2,figsize=(7.1,2.9))
 ee=np.array([r['eps'] for r in convergence]);e0=np.array([r['E0'] for r in convergence]);e1=np.array([r['E1'] for r in convergence])
 axs[0].loglog(ee,e0,'o-',color=COLORS[0],label='leading branch + initial layer')
 axs[0].loglog(ee,e1,'s-',color=COLORS[1],label='first-order + initial layer')
 axs[0].loglog(ee,e1[0]*(ee/ee[0])**2,':',color='#777',label='$O(\\varepsilon^2)$ reference')
 axs[0].set(xlabel='Scale ratio $\\varepsilon$',ylabel='Relative trajectory error $E$',title='(a) Fixed slow horizon $T=1.5$')
 axs[0].legend(frameon=False,fontsize=6.6)
 im=axs[1].pcolormesh(epsgrid,rgrid,errgrid,shading='nearest',cmap='viridis')
 if errgrid.min()<.01<errgrid.max():axs[1].contour(epsgrid,rgrid,errgrid,levels=[.01],colors='white',linewidths=1)
 axs[1].set(xlabel='Scale ratio $\\varepsilon$',ylabel='Initial displacement $R_0$',title='(b) Measured first-order error')
 fig.colorbar(im,ax=axs[1],label='$E$');fig.tight_layout();savefig('figure3')

 # Figure 4: model-derived RG coefficients and a near-critical experiment.
 fig,axs=plt.subplots(2,2,figsize=(7.2,4.8))
 for j in range(2):
  a=axs[0,j];a.plot(rg['t'],rg['full'][:,j],color=COLORS[0],lw=1.4,label='full fast system')
  a.plot(rg['t'],rg['rg'][:,j],color=COLORS[1],ls='--',lw=1.1,label='cubic RG')
  a.plot(rg['t'],rg['linear'][:,j],color=COLORS[2],ls=':',lw=1,label='linear envelope')
  a.set(xlabel='Fast time $\\tau$',ylabel=['Passenger activity $X$','Annual emissions $Y$'][j],title=f'({chr(97+j)}) Near-critical reconstruction')
 axs[0,0].legend(frameon=False,fontsize=6.7)
 axs[1,0].plot(rg['T'],abs(rg['full_amplitude']),color=COLORS[0],label='inverse reconstructed full mode')
 axs[1,0].plot(rg['T'],abs(rg['amplitude']),color=COLORS[1],ls='--',label='RG amplitude')
 axs[1,0].plot(rg['T'],.2*np.exp(-hd['Delta']*rg['T']),color=COLORS[2],ls=':',label='linear amplitude')
 axs[1,0].set(xlabel='Slow time $T$',ylabel='Scaled amplitude $|A|$',title='(c) Nonlinear envelope')
 axs[1,0].legend(frameon=False,fontsize=6.1)
 for R0 in [.05,.1,.2,.28]:
  rr=[r for r in rgsweep if r['R0']==R0]
  axs[1,1].loglog([r['eps'] for r in rr],[r['E'] for r in rr],'o-',ms=3,label=f'$R_0={R0}$')
 axs[1,1].set(xlabel='$\\varepsilon_H$',ylabel='Relative reconstruction error',title='(d) Measured local RG error')
 axs[1,1].legend(frameon=False,fontsize=6.3)
 fig.tight_layout();savefig('figure4')

 # Figure 5: paired synthetic uncertainty; no inferential confidence claim.
 fig,axs=plt.subplots(1,2,figsize=(7.1,3.1))
 for s,stat in enumerate(ensstats):
  for j,prefix in [(0,'avoided'),(1,'service')]:
   center=stat[prefix+'_q50'];lo=stat[prefix+'_q05'];hi=stat[prefix+'_q95']
   axs[j].errorbar(s,center,yerr=[[center-lo],[hi-center]],fmt='o',capsize=3,color=COLORS[s],ms=5)
   nom=rows[s]['avoided_fraction' if j==0 else 'service_ratio'];axs[j].plot(s,nom,'x',color='black',ms=5)
 for a in axs:
  a.set_xticks(range(5),[f'S{i}' for i in range(5)]);a.set_xlabel('Scenario')
 axs[0].axhline(0,color='#777',lw=.7);axs[0].set(ylabel='Avoided-emissions fraction relative to S0',title='(a) Cumulative emissions')
 axs[1].axhline(.95,color='#777',lw=.9,ls='--');axs[1].set(ylabel='Cumulative activity / reference activity',title='(b) Service retention')
 fig.tight_layout();savefig('figure5')

 # Figure 6: fixed accounting boundary and model-dependent trajectories.
 fig,axs=plt.subplots(1,2,figsize=(7.1,3.0))
 allx=[];ally=[]
 for s,u in enumerate(fulls):
  pp=scenario_parameters(p,s);f=np.array([fast(q,z,pp) for q,z in zip(u[:,:2],u[:,2])])
  gx=f[:,0]/u[:,0];gi=gx-f[:,1]/u[:,1];allx.extend(gx);ally.extend(gi)
  axs[0].plot(gx,gi,color=COLORS[s],lw=1.1,label=f'S{s}')
 lo=min(min(allx),min(ally))-.003;hi=max(max(allx),max(ally))+.003
 axs[0].plot([lo,hi],[lo,hi],color='#666',lw=.8,ls='--');axs[0].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Activity growth $g_X$',ylabel='Intensity improvement $g_I$',title='(a) Fixed accounting boundary')
 axs[0].legend(frameon=False,fontsize=6.5)
 u=fulls[4];c=p.c0*(1-p.eta*u[:,2]);D=p.h*c+p.d0+p.k*u[:,2]
 margin=u[:,1]/u[:,0]-c*p.a/D
 axs[1].plot(TGRID,margin,color=COLORS[4]);axs[1].axhline(0,color='#777',ls='--',lw=.8)
 axs[1].axvspan(8,14,alpha=.1,color=COLORS[4]);axs[1].set(xlabel='Planning time (years)',ylabel='$Y/X-ca/D$',title='(b) S4 emissions-decline margin')
 fig.tight_layout();savefig('figure6')

 # Tables and machine-readable numerical facts used in the manuscript.
 def tex_table(name,header,rows):
  (ROOT/'tables'/name).write_text(header+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
 tex_table('scenario_results.tex','% Generated by code/reproduce.py',[
  f"S{r['scenario']} & {r['X30']:.4f} & {r['Y30']:.4f} & {r['Z30']:.4f} & {r['I30']:.4f} & {r['CY']:.3f} & {100*r['avoided_fraction']:.2f} & {100*r['service_ratio']:.2f} \\\\" for r in rows])
 tex_table('reduction_results.tex','% Generated by code/reproduce.py',[
  f"S{r['scenario']} & {r['E']:.3e} & {r['max_X']:.3e} & {r['max_Y']:.3e} & {r['alpha']:.3f} \\\\" for r in rows])
 tex_table('convergence.tex','% Generated by code/reproduce.py',[
  f"{r['eps']:.3f} & {r['E0']:.3e} & {r['E1']:.3e} \\\\" for r in convergence])
 summary=dict(parameters=asdict(p),scenarios=rows,convergence=convergence,ensemble=ensstats,
   ensemble_draws=128,ensemble_rejected=rejection_count,ensemble_rank_changes=rank_changes,
   ensemble_feasibility_changes=feas_changes,seed=SEED,reference_convergence=max(convergence_solver),
   branch_interpolation_error=float(interpolation_error),
   runtime=dict(full_online_s=ft,reduced_online_s=rt,setup_s=setup,online_speedup=ft/rt,repetitions=10),
   hopf=dict(aH=hd['aH'],q=hd['q'].tolist(),omega=hd['omega'],L=hd['L'],M=hd['M'],
    a1=hd['a1'],Delta=hd['Delta'],omega1=hd['omega1'],c3=[hd['c3'].real,hd['c3'].imag],
    v=[[x.real,x.imag] for x in hd['v']],h20=[[x.real,x.imag] for x in hd['h20']],h11=hd['h11'].tolist(),
    example_error=rg['E'],linear_error=rg['E_linear'],Rcrit=float(np.sqrt(hd['Delta']/hd['c3'].real))),
   software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
   elapsed_seconds=time.perf_counter()-start)
 (ROOT/'data'/'results.json').write_text(json.dumps(summary,indent=2))
 # Stable numeric macros: the manuscript never needs hand-entered computed results.
 facts={'BetaNominal':p.beta,'RGError':rg['E'],'RGLinearError':rg['E_linear'],'HopfA':hd['aH'],
  'HopfOmega':hd['omega'],'CubicReal':hd['c3'].real,'CubicImag':hd['c3'].imag,
  'RGDetuning':hd['omega1'],'CriticalAmplitude':np.sqrt(hd['Delta']/hd['c3'].real),
  'SolverCheck':max(convergence_solver),'BranchCheck':interpolation_error,
  'OnlineSpeedup':ft/rt,'FullMilliseconds':ft*1000,'ReducedMilliseconds':rt*1000,'SetupMilliseconds':setup*1000,
  'EnsembleMaxError':max(r['Emax'] for r in ensstats),'EnsembleRejects':rejection_count,
  'RankChanges':rank_changes,'FeasChanges':feas_changes}
 def fmt(v):return str(int(v)) if isinstance(v,(int,np.integer)) else f'{v:.6g}'
 (ROOT/'tables'/'computed_values.tex').write_text('% Generated numerical values\n'+'\n'.join('\\newcommand{\\'+k+'}{'+fmt(v)+'}' for k,v in facts.items())+'\n')
 print('FINAL SUMMARY',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
