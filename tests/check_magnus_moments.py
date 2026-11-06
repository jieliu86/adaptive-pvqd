"""Numerical checks for the Magnus constraint experiment."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from hamiltonian import xyz_floquet_hamiltonian
from moment_constraints import magnus5, MomentController
from moment_statevector import initial,pool,evolution,LocalLoss
from run_moment_apvqd import Objective,moments
H=lambda t:xyz_floquet_hamiltonian(8,[1,.8,.6,0],t)
K=magnus5(H,.4,.05);M=K.to_matrix()
assert np.linalg.norm(M-M.conj().T)<1e-11
assert np.linalg.norm(M-magnus5(H,.4,.05,12).to_matrix())<1e-10
static=H(.4);assert np.linalg.norm(magnus5(lambda t:static,.4,.05).to_matrix()-static.to_matrix())<1e-10
errors=[]
for dt in [.2,.1,.05]:
 k=magnus5(H,.4,dt).to_matrix()
 ref=solve_ivp(lambda t,v:-1j*H(t).to_matrix()@v,[.4,.4+dt],initial,method='DOP853',rtol=2e-13,atol=2e-15)
 errors.append(float(np.linalg.norm(expm(-1j*dt*k)@initial-ref.y[:,-1])))
assert errors[0]>30*errors[1] and errors[1]>20*errors[2],errors
rng=np.random.default_rng(42);ops=pool;theta=rng.normal(0,.15,len(ops));ep,et,_=evolution(.425)
local=LocalLoss([],np.zeros(0),ep,et)
ctl=MomentController(.002,.002,.01,.01,.01,-.01)
obj=Objective(local,ops,K.to_spmatrix().tocsr(),ctl,moments(M,initial))
f,g,d,_,_=obj.evaluate(theta)
assert d['penalty']>0
checks=[]
for j in [0,9,16,24,35,44]:
 delta=np.zeros(len(theta));delta[j]=1e-6
 finite=(obj.evaluate(theta+delta,False)[0]-obj.evaluate(theta-delta,False)[0])/2e-6
 assert abs(g[j]-finite)<1e-5,(j,g[j],finite)
 checks.append(dict(index=j,error=float(abs(g[j]-finite))))
values,jac=obj.constraint_values_jacobian(theta)
for j in [9,24,44]:
 delta=np.zeros(len(theta));delta[j]=1e-6
 finite=(obj.constraint_values_jacobian(theta+delta)[0]-obj.constraint_values_jacobian(theta-delta)[0])/2e-6
 assert np.max(abs(finite-jac[:,j]))<1e-7
# Independent Qiskit echo observable and circuit parameter-shift check.
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
c=QuantumCircuit(8)
for q in range(1,8,2):c.x(q)
for (op,qs),angle in zip(ops,theta):getattr(c,'r'+op)(float(angle),*qs)
from moment_statevector import weights, sequence
qc=QuantumCircuit(8)
qc.append(evolution(.425)[2].inverse(),range(8))
for q in range(1,8,2):qc.x(q)
qecho=Statevector(c.compose(qc)).data
assert abs(np.sum(weights*abs(qecho)**2)-d['local_cost'])<1e-11
for j in [9,24,44]:
 delta=np.zeros(len(theta));delta[j]=np.pi/2
 vp=obj.evaluate(theta+delta,False)[2];vm=obj.evaluate(theta-delta,False)[2]
 shift_gradient=(moments(M,vp)[0]-moments(M,vm)[0])/2
 assert abs(shift_gradient-jac[2,j])<1e-10
zero=Objective(local,ops,K.to_spmatrix().tocsr(),MomentController(energy_weight=0,variance_weight=0),moments(M,initial))
zvalue,zgrad,*_=zero.evaluate(theta)
lvalue,lgrad=local.evaluate(ops,theta)
assert abs(zvalue-lvalue)<1e-12 and np.max(abs(zgrad-lgrad))<1e-12
old=(ctl.sum_energy,ctl.sum_variance)
try:ctl.commit(dict(feasible=False))
except ValueError:pass
else:raise AssertionError('infeasible commit accepted')
assert old==(ctl.sum_energy,ctl.sum_variance)
print(json.dumps(dict(hermiticity='passed',static_limit='passed',quadrature='passed',Magnus_state_errors=errors,gradient=checks,rejected_commit='passed',hard_constraint_jacobian='passed',qiskit_echo='passed',moment_parameter_shift='passed',zero_penalty_regression='passed'),indent=2))
