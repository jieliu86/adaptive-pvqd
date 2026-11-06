import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analyze_moment_apvqd import analyze
from qiskit import QuantumCircuit,transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
root=Path(__file__).resolve().parent
previous={}
summary={}
for name in ['local','nonlocal']:
 assert json.loads((root/(name+'_inherited')/'status.json').read_text())['completed']
 previous[name],_=analyze(root/(name+'_inherited'))
 summary[name+'_inherited']=previous[name]
reference=np.load((Path(__file__).resolve().parent.parent/'exact_reference_t2.npz'))['states'][:41]
validations={};raw={}
for name in ['local','nonlocal']:
 folder=root/name
 status=json.loads((folder/'status.json').read_text());assert status['completed'],(name,status)
 summary[name+'_rebuilt'],raw[name]=analyze(folder)
 result=raw[name];states=np.load(folder/'trajectory.npz')['states'];rows=result['rows']
 assert len(rows)==40 and result['settings']['rebuild_every_step'] and result['settings']['mode']=='baseline'
 assert result['settings']['gradient_tolerance']==1e-5
 assert all(r['local_cost']<=1e-4 and r['penalty']==0 for r in rows)
 err=max(abs(1-abs(np.vdot(a,b))**2-row['global_infidelity']) for a,b,row in zip(reference[1:],states[1:],rows));assert err<1e-8
 oldfolder=root/(name+'_inherited')
 oldstates=np.load(oldfolder/'trajectory.npz')['states']
 first_error=float(1-abs(np.vdot(states[1],oldstates[1]))**2);assert first_error<1e-10
 c=QuantumCircuit.from_qasm_file(str(folder/'final.qasm'))
 qerr=float(1-abs(np.vdot(Statevector(c).data,states[-1]))**2);assert qerr<1e-10
 compiled=transpile(c,basis_gates=['rz','sx','x','cx'],coupling_map=CouplingMap.from_line(8,bidirectional=True),initial_layout=list(range(8)),optimization_level=3,seed_transpiler=42,approximation_degree=1.)
 layout=compiled.layout.final_layout;mapping=list(range(8)) if layout is None else [layout[q] for q in compiled.qubits]
 ids=np.array([sum(((i>>q)&1)<<mapping[q] for q in range(8)) for i in range(256)])
 cerr=float(1-abs(np.vdot(Statevector(compiled).data[ids],states[-1]))**2);assert cerr<1e-8
 validations[name+'_rebuilt']=dict(reference_infidelity_max_discrepancy=err,first_step_vs_inherited_infidelity=first_error,QASM_infidelity=qerr,compiled_infidelity=cerr,compiled_cx=int(compiled.count_ops().get('cx',0)),compiled_depth=int(compiled.depth()),largest_growth_rounds=max(r['rounds'] for r in rows),parameter_decreases=sum(rows[j]['parameters']<rows[j-1]['parameters'] for j in range(1,40)))
for name in ['local','nonlocal']:
 folder=root/(name+'_inherited');states=np.load(folder/'trajectory.npz')['states'];result=json.loads((folder/'results.json').read_text());rows=result['rows']
 assert result['settings']['gradient_tolerance']==1e-5 and not result['settings']['rebuild_every_step']
 assert all(r['local_cost']<=1e-4 and r['penalty']==0 for r in rows)
 c=QuantumCircuit.from_qasm_file(str(folder/'final.qasm'))
 compiled=transpile(c,basis_gates=['rz','sx','x','cx'],coupling_map=CouplingMap.from_line(8,bidirectional=True),initial_layout=list(range(8)),optimization_level=3,seed_transpiler=42,approximation_degree=1.)
 layout=compiled.layout.final_layout;mapping=list(range(8)) if layout is None else [layout[q] for q in compiled.qubits]
 ids=np.array([sum(((i>>q)&1)<<mapping[q] for q in range(8)) for i in range(256)])
 cerr=float(1-abs(np.vdot(Statevector(compiled).data[ids],states[-1]))**2);assert cerr<1e-8
 err=max(abs(1-abs(np.vdot(a,b))**2-row['global_infidelity']) for a,b,row in zip(reference[1:],states[1:],rows));assert err<1e-8
 validations[name+'_inherited']=dict(reference_infidelity_max_discrepancy=err,compiled_infidelity=cerr,compiled_cx=int(compiled.count_ops().get('cx',0)),compiled_depth=int(compiled.depth()),largest_growth_rounds=max(r['rounds'] for r in rows),parameter_decreases=sum(rows[j]['parameters']<rows[j-1]['parameters'] for j in range(1,40)))
(root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'validation.json').write_text(json.dumps(validations,indent=2))
lagrows=[]
for lag in range(40):
 row=dict(lag=lag,time_lag=.05*lag,pairs=40-lag)
 for name,values in summary.items():
  row[name+'_rho']=values['correlation'][lag]['rho']
  row[name+'_C']=values['correlation'][lag]['C']
 lagrows.append(row)
with (root/'lag_correlations.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(lagrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(lagrows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(12,8))
for name in ['local','nonlocal']:
 for mode in ['inherited','rebuilt']:
  label=name+'_'+mode;s=summary[label];corr=s['correlation']
  axs[0,1].plot([r['time_lag'] for r in corr],[r['rho'] for r in corr],label=label)
  if mode=='rebuilt':rows=raw[name]['rows']
  else:
   folder=root/(name+'_inherited')
   rows=json.loads((folder/'results.json').read_text())['rows']
  t=[r['time'] for r in rows]
  axs[0,0].plot(t,[r['global_infidelity'] for r in rows],label=label)
  axs[1,0].plot(t,[r['parameters'] for r in rows],label=label)
  axs[1,1].plot(t,[r['local_cost'] for r in rows],label=label)
for ax,title in zip(axs.flat,['Global infidelity','Mean normalized transported residual correlation','Parameters in each state-preparation circuit','Local echo loss']):
 ax.set_title(title);ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('Time / lag')
axs[0,1].axhline(0,color='gray',lw=.7)
fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160)
lines=['# Rebuild ansatz every step: t=0..2','','## Protocol','','Every step uses the previous accepted variational state as propagation input. Its old circuit is frozen in the echo target. The candidate state-preparation ansatz starts from the Neel product state, empty operator sequence and zero parameters; no previous parameters or generators are copied. The trajectory itself is NOT reset to the exact state. Inherited runs also reoptimize all existing candidate angles: they inherit structure and initialization, not frozen old angles. Only the input/reference circuit is frozen within a step in both methods. No penalties. Same midpoint Suzuki2 reps2, dt=.05, local-cost threshold 1e-4, L-BFGS-B max200 per growth round, gtol1e-5. Pool definitions and ordering match inherited runs. Rebuilt runs allow at most 100 growth rounds versus 30 in inherited runs; actual maxima are recorded below.','','The local echo objective depends on the reference circuit representation as well as the target state. Rebuilding therefore changes both the optimization path and subsequent echo geometry. This experiment tests the complete rebuild strategy, not a uniquely isolated causal effect of circuit inheritance.','','|Method|Global infidelity (%)|Residual diagonal sum|Cross terms|Amplification|Final parameters|Optimizer iterations|','|---|---:|---:|---:|---:|---:|---:|']
for name,s in summary.items():lines.append(f"|{name}|{100*s['global_infidelity']:.5f}|{s['sum_residual_norm_squared']:.7g}|{s['cross_terms']:.7g}|{s['amplification']:.4f}|{s['parameters']}|{s['optimizer_iterations']}|")
lines += ['','## Lag correlations','','|Method|rho(.05)|rho(.25)|rho(.5)|rho(1.0)|','|---|---:|---:|---:|---:|']
for name,s in summary.items():lines.append('|'+name+'|'+'|'.join(f"{s['correlation'][j]['rho']:.5f}" for j in [1,5,10,20])+'|')
lines += ['','## Final circuit costs','','|Pool|Compiled CX|Compiled depth|Max growth rounds|Number of parameter-count decreases|','|---|---:|---:|---:|---:|']
for name,x in validations.items():lines.append(f"|{name}|{x['compiled_cx']}|{x['compiled_depth']}|{x['largest_growth_rounds']}|{x['parameter_decreases']}|")
lines += ['','The first gtol=5e-5 local rebuild attempt failed its 100-round cap at t=1.0 with local cost 1.063229e-4, and its nonlocal counterpart was interrupted for matched reruns. These partial attempts are preserved but excluded from this table. All four reported trajectories use gtol=1e-5.','','![Comparison](comparison.png)','','Residuals use recursive phase alignment and backward transport by the shared discrete propagator. The residual sum reconstruction is checked to 1e-9. Vector-error squared and infidelity are different quantities. Single deterministic trajectories, no statistical independence/significance claim. Optimization and stopping errors are included. Final circuit compilation uses rz/sx/x/cx, line coupling, level3, seed42; routing permutation is undone for the independent state check. Full exact reference is used only for evaluation. All accepted steps pass local-cost gates; final QASM and reference errors are checked separately.','','Run: `python run_moment_apvqd.py --mode baseline --pool local --rebuild-every-step --max-growth-rounds 100 --gradient-tolerance 0.00001 --out OUTPUT`; replace local with nonlocal for the other pool.']
meta=json.loads((root/'provenance.json').read_text())
lines += ['',f"Direct midpoint Suzuki propagation without variational projection has t=2 infidelity {meta['direct_midpoint_Suzuki_infidelity_to_exact_t2']:.9g} versus the continuous exact reference."]
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:{a:b for a,b in s.items() if a!='correlation'} for k,s in summary.items()},indent=2));print(json.dumps(validations,indent=2))
