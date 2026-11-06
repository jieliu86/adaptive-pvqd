"""Matched midpoint APVQD experiment, t<=2; exact-state simulation only.

Uses the repository local echo loss, Pauli pool and Tetris growth. Adjoint
statevector derivatives replace hardware parameter shifts for speed. Both
branches use identical L-BFGS-B refinement; constraint branches use full-loss
pool gradients. No reference states enter optimization or acceptance.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from hamiltonian import xyz_floquet_hamiltonian
from moment_constraints import magnus5, MomentController
from moment_statevector import N, initial, pool, pauli, rot, sequence, evolution, LocalLoss


def moments(K, v):
    kv=K@v
    m=float(np.vdot(v,kv).real)
    return m/N, float((np.vdot(kv,kv).real-m*m)/N)


class Objective:
    def __init__(self, local, ops, K, controller, input_moments):
        self.local=local; self.ops=ops; self.P=[pauli(*op) for op in ops]
        self.K=K; self.controller=controller; self.input=input_moments
    def evaluate(self, theta, gradient=True):
        states=[initial]
        for P,t in zip(self.P,theta):states.append(rot(states[-1],t,P))
        v=states[-1]; echo=self.local.echo(v)
        from moment_statevector import weights
        cost=float(np.sum(weights*abs(echo)**2))
        E,V=moments(self.K,v)
        d=self.controller.evaluate(E,V,*self.input)
        d.update(local_cost=cost,energy=E,variance=V)
        value=cost+d['penalty']
        if not gradient:return value,d,v
        kv=self.K@v; mean=E*N
        back=self.local.echo_inverse(weights*echo)
        back += d['derivative_energy']*kv/N
        back += d['derivative_variance']*(self.K@kv-2*mean*kv)/N
        endback=back.copy();g=np.empty(len(theta))
        for j in range(len(theta)-1,-1,-1):
            g[j]=2*np.real(np.vdot(back,-.5j*self.P[j](states[j+1])))
            back=rot(back,-theta[j],self.P[j])
        return value,g,d,v,endback

    def constraint_values_jacobian(self, theta):
        states=[initial]
        for P,t in zip(self.P,theta):states.append(rot(states[-1],t,P))
        v=states[-1];kv=self.K@v;E,V=moments(self.K,v)
        d=self.controller.evaluate(E,V,*self.input)
        from moment_statevector import weights
        echo=self.local.echo(v);cost=float(np.sum(weights*abs(echo)**2))
        backs=np.array([self.local.echo_inverse(weights*echo),kv/N,
                        (self.K@kv-2*E*N*kv)/N])
        gradients=np.empty((3,len(theta)))
        for j in range(len(theta)-1,-1,-1):
            direction=-.5j*self.P[j](states[j+1])
            gradients[:,j]=2*np.real(backs.conj()@direction)
            for k in range(3):backs[k]=rot(backs[k],-theta[j],self.P[j])
        ge,gv=gradients[1:];re=d['residual_energy'];rv=d['residual_variance']
        de=self.controller.energy_budget;dv=self.controller.variance_budget
        # Tiny inward margin prevents roundoff-only budget violations.
        margin=1e-8
        values=np.array([1e-4-cost,de-margin-re,de-margin+re,dv-margin-rv,dv-margin+rv])
        jac=np.array([-gradients[0],-ge,ge,-gv,gv])
        return values,jac


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['baseline','penalty'],required=True)
    parser.add_argument('--pool',choices=['local','nonlocal'],default='local')
    parser.add_argument('--rebuild-every-step',action='store_true')
    parser.add_argument('--max-growth-rounds',type=int,default=30)
    parser.add_argument('--gradient-tolerance',type=float,default=5e-5)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--budget',type=float,default=.02)
    parser.add_argument('--weight',type=float,default=.01)
    args=parser.parse_args()
    if args.max_growth_rounds < 1:
        parser.error('--max-growth-rounds must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    operator_pool=list(pool)
    if args.pool=='nonlocal':
        # Preserve original pool ordering, then append longer-range pairs.
        operator_pool += [(a,[i,j]) for a in ['xx','yy','zz']
                          for i in range(N) for j in range(i+2,N)]
    H=lambda t:xyz_floquet_hamiltonian(N,[1,.8,.6,0],t)
    ctl=MomentController(args.budget,args.budget,args.weight if args.mode=='penalty' else 0.,args.weight if args.mode=='penalty' else 0.)
    times=np.linspace(0,2,41)
    ref=solve_ivp(lambda t,v:-1j*(H(t).to_matrix()@v),[0,2],initial,t_eval=times,method='DOP853',rtol=1e-11,atol=1e-13)
    assert ref.success
    ops=[];theta=np.zeros(0);v=initial.copy();rows=[];states=[v.copy()]
    start=time.time();failure=None
    for n in range(40):
        t=times[n];Kop=magnus5(H,t,.05);K=Kop.to_spmatrix().tocsr()
        eP,eT,_=evolution(t+.025)
        local=LocalLoss(list(ops),theta,eP,eT)
        inp=moments(K,v)
        trialops=[] if args.rebuild_every_step else list(ops)
        trialtheta=np.zeros(0) if args.rebuild_every_step else theta.copy()
        history=[]
        accepted=False
        for rnd in range(args.max_growth_rounds):
            obj=Objective(local,trialops,K,ctl,inp)
            if len(trialtheta):
                result=minimize(lambda x:obj.evaluate(x)[:2],trialtheta,jac=True,method='L-BFGS-B',options=dict(maxiter=200,gtol=args.gradient_tolerance,ftol=1e-13,maxls=30))
                trialtheta=result.x
                history.append(dict(iterations=int(result.nit),success=bool(result.success),message=str(result.message)))
            value,g,d,newv,back=obj.evaluate(trialtheta)
            # A finite soft penalty alone need not satisfy hard budgets.
            # Polish on the SAME topology before spending additional gates.
            if args.mode=='penalty' and len(trialtheta) and not d['feasible']:
                cache={}
                def constraints(x):
                    key=x.tobytes()
                    if cache.get('key')!=key:
                        cache['key']=key;cache['result']=obj.constraint_values_jacobian(x)
                    return cache['result']
                # On the feasible set the hinge penalty is zero, so minimizing
                # pure local cost here has exactly the same constrained objective
                # and avoids stiff artificial curvature outside the budgets.
                polished=minimize(lambda x:local.evaluate(trialops,x),trialtheta,jac=True,
                    method='SLSQP',constraints=[dict(type='ineq',fun=lambda x:constraints(x)[0],jac=lambda x:constraints(x)[1])],
                    options=dict(maxiter=200,ftol=1e-12))
                history.append(dict(phase='hard_budget_polish',iterations=int(polished.nit),success=bool(polished.success),message=str(polished.message)))
                trialtheta=polished.x
                value,g,d,newv,back=obj.evaluate(trialtheta)
            accepted=d['local_cost']<=1e-4 and (args.mode=='baseline' or d['feasible'])
            if accepted:break
            # Candidate gradients of the complete nonlinear objective.
            scores=[abs(2*np.real(np.vdot(back,-.5j*pauli(*op)(newv)))) for op in operator_pool]
            remaining=list(range(len(operator_pool)));selected=[]
            while remaining:
                k=max(remaining,key=lambda j:scores[j]);selected.append(k)
                remaining=[j for j in remaining if set(operator_pool[k][1]).isdisjoint(operator_pool[j][1])]
            selected.sort(key=lambda k:operator_pool[k][1][0])
            trialops.extend(operator_pool[k] for k in selected);trialtheta=np.r_[trialtheta,np.zeros(len(selected))]
        if not accepted:
            failure=dict(time=float(t+.05),diagnostics=d,reason=f'{args.max_growth_rounds} growth rounds exhausted; candidate rejected')
            break
        # Commit only after all acceptance gates; baseline records signed sums without enforcing them.
        if args.mode=='penalty':ctl.commit(d)
        else:ctl.sum_energy=d['residual_energy'];ctl.sum_variance=d['residual_variance']
        ops=trialops;theta=trialtheta;v=newv
        states.append(v.copy())
        c=QuantumCircuit(N)
        for q in range(1,N,2):c.x(q)
        for (op,qs),angle in zip(ops,theta):getattr(c,'r'+op)(float(angle),*qs)
        assert 1-abs(np.vdot(Statevector(c).data,v))**2<1e-10
        row=dict(time=float(t+.05),**d,global_infidelity=float(1-abs(np.vdot(ref.y[:,n+1],v))**2),parameters=len(theta),raw_cx=2*sum(len(q)==2 for _,q in ops),rounds=rnd+1,history=history,K_terms=len(Kop.primitive),K2_terms=len((Kop@Kop).reduce(atol=1e-12).primitive))
        rows.append(row)
        np.savez(args.out/'trajectory.npz',times=times[:len(states)],states=np.array(states),parameters=theta)
        (args.out/'results.json').write_text(json.dumps(dict(settings=vars(args)|{'out':str(args.out),'dt':.05,'end':2,'propagator':'midpoint Suzuki2 reps2','optimizer':'L-BFGS-B max200 per growth round','magnus_order':5,'qubits':N,'model_parameters':[1,.8,.6,0],'initial_state':'Neel, X on odd sites','pool':args.pool,'pool_size':len(operator_pool),'local_cost_threshold':1e-4,'max_growth_rounds':args.max_growth_rounds,'constraint_polish':'SLSQP max200, ftol1e-12, margin1e-8'},rows=rows,failure=failure,elapsed=time.time()-start),indent=2))
        print(args.mode,round(t+.05,2),row['global_infidelity'],len(theta),d['residual_energy'],d['residual_variance'],flush=True)
    (args.out/'status.json').write_text(json.dumps(dict(completed=len(rows)==40,failure=failure),indent=2))
    # Save executable final circuit and parameterized topology independently of experiment paths.
    if rows:
        (args.out/'final.qasm').write_text(c.qasm())
        (args.out/'topology.json').write_text(json.dumps(dict(ops=ops,parameters=theta.tolist())))

if __name__=='__main__':main()
