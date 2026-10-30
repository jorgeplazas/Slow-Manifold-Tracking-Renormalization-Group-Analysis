#!/usr/bin/env python3
"""Independent analytical/numerical checks of the revised manuscript's results."""
from pathlib import Path
import json, math
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq
from reproduce import (ROOT, Parameters, derivatives_r, response, fast, jacobian,
 equilibrium, qz_and_lag, H, hopf_data, nf_map, nf_inverse, rg_experiment)

def run():
 record=json.loads((ROOT/'data/results.json').read_text());p=Parameters(**record['parameters'])
 checks={};rng=np.random.default_rng(31)
 # Boundary fluxes, including extreme but admissible forcing values.
 for _ in range(100):
  x,y,z=rng.uniform(0,5),rng.uniform(0,5),rng.uniform(0,1)
  assert fast([0,y],z,p)[0]==0
  assert fast([x,0],z,p)[1]>=0
  for sc in range(5):
   assert H(10,0,p,sc)>=0
   assert H(10,1,p,sc)<=0
 checks['boundary_fluxes']='pass'

 # Analytical r derivatives from independent partial fractions.
 maxder=0.
 for x in [.1,.4,1,2]:
  exact=np.array([(-1)**n*math.factorial(n)/(1-p.beta*p.ell)*
    (-p.ell/(x+p.ell)**(n+1)+p.beta**n/(1+p.beta*x)**(n+1)) for n in range(4)])
  maxder=max(maxder,float(max(abs(exact-derivatives_r(x,p)))))
 assert maxder<2e-10
 checks['r_derivative_max_absolute_error']=maxder

 maxres=maxjac=maxqz=0.
 for z in np.linspace(.01,.99,30):
  q,qz,lag=qz_and_lag(z,p);A=jacobian(q,z,p)
  maxres=max(maxres,float(max(abs(fast(q,z,p)))))
  assert np.linalg.det(A)>0
  step=1e-5
  An=np.column_stack([(fast(q+step*np.eye(2)[j],z,p)-fast(q-step*np.eye(2)[j],z,p))/(2*step) for j in range(2)])
  maxjac=max(maxjac,float(max(abs(A-An).flat)))
  qzn=(equilibrium(z+step,p)-equilibrium(z-step,p))/(2*step)
  maxqz=max(maxqz,float(max(abs(qz-qzn))))
 assert maxres<1e-11 and maxjac<2e-8 and maxqz<2e-8
 checks.update(equilibrium_residual=maxres,finite_difference_jacobian_error=maxjac,
               branch_derivative_error=maxqz)

 # Constant-forcing adoption solution, independent of the ODE integrator.
 arr=np.genfromtxt(ROOT/'data/scenario_S0.csv',delimiter=',',names=True)
 t=arr['t'];A=1+p.omega*.2;b=p.sigma*.15
 disc=np.sqrt((A-p.mu-b)**2+4*A*b)
 zp=(A-p.mu-b+disc)/(2*A);zm=(A-p.mu-b-disc)/(2*A)
 ratio=(p.z0-zp)/(p.z0-zm)*np.exp(-p.eps*A*(zp-zm)*t)
 z_exact=(zp-ratio*zm)/(1-ratio)
 dz=float(max(abs(arr['Z']-z_exact)));assert dz<2e-9
 checks['riccati_solution_max_error']=dz

 # Hopf identities and RG closed-form solution.
 hd=hopf_data()
 assert abs(np.trace(hd['A']))<2e-13
 assert np.linalg.det(hd['A'])>0
 assert max(abs(hd['A']@hd['v']-1j*hd['omega']*hd['v']))<1e-13
 assert abs(hd['adj']@hd['v']-1)<1e-13
 assert abs(hd['adj']@hd['v'].conj())<1e-13
 assert hd['c3'].real>0
 rg=rg_experiment(hd);T=rg['T'];R0=.2;Delta=hd['Delta'];g=hd['c3'].real
 den=1-g*R0**2/Delta*(1-np.exp(-2*Delta*T))
 R=R0*np.exp(-Delta*T)/np.sqrt(den)
 phase=hd['omega1']*T-hd['c3'].imag/(2*g)*np.log(den)
 err=float(max(abs(R*np.exp(1j*phase)-rg['amplitude'])))
 assert err<2e-10
 checks['RG_closed_form_max_error']=err

 # A stroboscopic nonlinear integration estimates c3 without the tensor formula.
 w0=.003;end=5*2*np.pi/hd['omega'];grid=np.linspace(0,end,6001)
 initial=hd['q']+nf_map(np.array(w0),hd)
 sol=solve_ivp(lambda t,u:fast(u,0,hd['p']),(0,end),initial,t_eval=grid,
               method='DOP853',rtol=2e-12,atol=2e-14)
 w=nf_inverse(sol.y.T-hd['q'],hd)
 integral=trapezoid(abs(w)**2,grid)
 greal=np.log(abs(w[-1])/abs(w[0]))/integral
 gimag=(np.unwrap(np.angle(w))[-1]-np.angle(w[0])-hd['omega']*end)/integral
 checks['cubic_from_nonlinear_integration']=[float(greal),float(gimag)]
 assert abs(greal/hd['c3'].real-1)<.03
 assert abs(gimag/hd['c3'].imag-1)<.03

 # Recalculate displayed metrics directly from the stored trajectory arrays.
 for sc,row in enumerate(record['scenarios']):
  data=np.genfromtxt(ROOT/f'data/scenario_S{sc}.csv',delimiter=',',names=True)
  assert min(data['X'])>=0 and min(data['Y'])>=0
  assert min(data['Z'])>=0 and max(data['Z'])<=1
  assert abs(trapezoid(data['Y'],data['t'])-row['CY'])<1e-11
  assert abs(trapezoid(data['X'],data['t'])-row['CX'])<1e-11
  assert abs(data['Y'][-1]/data['X'][-1]-row['I30'])<1e-12
 checks['all_nominal_state_bounds_and_table_metrics']='pass'
 conv=record['convergence'][:4]
 e=np.array([r['eps'] for r in conv]);e0=np.array([r['E0'] for r in conv]);e1=np.array([r['E1'] for r in conv])
 slope0=np.polyfit(np.log(e),np.log(e0),1)[0];slope1=np.polyfit(np.log(e),np.log(e1),1)[0]
 assert .9<slope0<1.1 and 1.85<slope1<2.2
 checks['observed_convergence_orders']=[float(slope0),float(slope1)]
 ens=np.genfromtxt(ROOT/'data/paired_parameter_ensemble.csv',delimiter=',',names=True)
 assert len(ens)==128*5
 ranks=feas=0
 for i in range(128):
  draw=ens[ens['draw']==i]
  draw=draw[np.argsort(draw['scenario'])]
  assert np.array_equal(draw['scenario'],np.arange(5))
  ranks+=int(not np.array_equal(np.argsort(draw['CY']),np.argsort(draw['CY_reduced'])))
  feas+=int(any((draw['CX']/draw['CX'][0]>=.95)!=(draw['CX_reduced']/draw['CX_reduced'][0]>=.95)))
 rejects=int(sum(ens['E']>.01))
 assert min(ens['alpha'])>0
 assert rejects==record['ensemble_rejected']==0
 assert ranks==record['ensemble_rank_changes']==0
 assert feas==record['ensemble_feasibility_changes']==0
 checks['paired_decision_comparisons']='pass'
 (ROOT/'data/verification.json').write_text(json.dumps(checks,indent=2))
 print(json.dumps(checks,indent=2));print('All verification checks passed.')

if __name__=='__main__':run()
