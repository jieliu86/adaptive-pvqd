"""Statevector implementation of repository local echo loss and Pauli pool."""
import numpy as np
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from hamiltonian import xyz_floquet_hamiltonian
N=8
ids=np.arange(2**N)
cache={}
xmask=sum(1<<q for q in range(1,N,2))
xor=ids^xmask
initial=np.zeros(2**N,complex)
initial[xmask]=1
weights=np.array([int(i).bit_count()/N for i in ids])
def rot(v,theta,P):return np.cos(theta/2)*v-1j*np.sin(theta/2)*P(v)
def pauli(op,qs):
 key=(op,tuple(qs))
 if key not in cache:
  dest=ids.copy();phase=np.ones(len(ids),complex)
  for p,q in zip(op,qs):
   bit=(ids>>q)&1
   if p in 'xy':dest ^= 1<<q
   if p=='y':phase *= 1j*(1-2*bit)
   if p=='z':phase *= 1-2*bit
  cache[key]=(dest,phase)
 dest,phase=cache[key]
 def apply(v):
  out=np.empty_like(v);out[dest]=phase*v;return out
 return apply

def rot(v,theta,P):return np.cos(theta/2)*v-1j*np.sin(theta/2)*P(v)
def sequence(v,Ps,ts):
 for P,t in zip(Ps,ts):v=rot(v,t,P)
 return v

def evolution(t):
 gate=PauliEvolutionGate(xyz_floquet_hamiltonian(N,[1,.8,.6,0],t),time=.05,synthesis=SuzukiTrotter(order=2,reps=2))
 c=gate.definition.decompose();p=[];ts=[]
 for x in c.data:
  assert x.operation.name in ['rxx','ryy','rzz','rz']
  qs=[c.find_bit(q).index for q in x.qubits];p.append(pauli(x.operation.name[1:],qs));ts.append(float(x.operation.params[0]))
 return p,np.array(ts),gate

class LocalLoss:
 def __init__(self,oldops,oldtheta,evoP,evoT):
  self.oldP=[pauli(*op) for op in oldops];self.oldT=oldtheta.copy();self.evoP=evoP;self.evoT=evoT
 def echo(self,v):
  v=sequence(v,self.evoP[::-1],-self.evoT[::-1]);v=sequence(v,self.oldP[::-1],-self.oldT[::-1]);return v[xor]
 def echo_inverse(self,v):
  v=v[xor];v=sequence(v,self.oldP,self.oldT);return sequence(v,self.evoP,self.evoT)
 def evaluate(self,ops,theta,with_grad=True):
  Ps=[pauli(*op) for op in ops];states=[initial]
  for P,t in zip(Ps,theta):states.append(rot(states[-1],t,P))
  echo=self.echo(states[-1]);loss=float(np.sum(weights*abs(echo)**2))
  if not with_grad:return loss
  back=self.echo_inverse(weights*echo);grad=np.empty(len(theta))
  for j in range(len(theta)-1,-1,-1):
   grad[j]=2*np.real(np.vdot(back,-.5j*Ps[j](states[j+1])))
   back=rot(back,-theta[j],Ps[j])
  return loss,grad
 def pool_gradients(self,ops,theta,pool):
  psi=sequence(initial,[pauli(*op) for op in ops],theta);echo=self.echo(psi);back=self.echo_inverse(weights*echo)
  return np.array([abs(2*np.real(np.vdot(back,-.5j*pauli(*op)(psi)))) for op in pool])

pool=[(a,[q]) for a in ['x','y','z'] for q in range(N)] + [(a,[q,q+1]) for a in ['xx','yy','zz'] for q in range(N-1)]
