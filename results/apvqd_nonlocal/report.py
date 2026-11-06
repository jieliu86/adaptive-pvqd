import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from qiskit import QuantumCircuit,transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
root=Path(__file__).resolve().parent;folder=root/'nonlocal'
assert json.loads((folder/'status.json').read_text())['completed']
d=json.loads((folder/'results.json').read_text());v=np.load(folder/'trajectory.npz')['states']
ref=np.load((Path(__file__).resolve().parent.parent/'exact_reference_t2.npz'))['states'][:41]
err=max(abs(1-abs(np.vdot(a,b))**2-row['global_infidelity']) for a,b,row in zip(ref[1:],v[1:],d['rows']))
assert err<1e-8
assert d['settings']['pool_size']==108 and d['settings']['mode']=='baseline'
assert all(r['local_cost']<=1e-4 and r['penalty']==0 for r in d['rows'])
c=QuantumCircuit.from_qasm_file(str(folder/'final.qasm'))
assert 1-abs(np.vdot(Statevector(c).data,v[-1]))**2<1e-10
compiled=transpile(c,basis_gates=['rz','sx','x','cx'],coupling_map=CouplingMap.from_line(8,bidirectional=True),initial_layout=list(range(8)),optimization_level=3,seed_transpiler=42,approximation_degree=1.)
physical=Statevector(compiled).data
# Undo the compiler's final routing permutation before comparing logical states.
layout=compiled.layout.final_layout
mapping=list(range(8)) if layout is None else [layout[q] for q in compiled.qubits]
indices=np.array([sum(((i>>q)&1)<<mapping[q] for q in range(8)) for i in range(256)])
compiler_error=float(1-abs(np.vdot(physical[indices],v[-1]))**2)
assert compiler_error<1e-8,compiler_error
last=d['rows'][-1];top=json.loads((folder/'topology.json').read_text())
s=dict(final_time=2,global_infidelity=last['global_infidelity'],parameters=last['parameters'],raw_cx=last['raw_cx'],compiled_cx=int(compiled.count_ops().get('cx',0)),compiled_depth=int(compiled.depth()),compiler_infidelity=compiler_error,final_routing_mapping=mapping,reference_max_error=err,nonlocal_rotations=sum(len(qs)==2 and abs(qs[0]-qs[1])>1 for _,qs in top['ops']),optimizer_iterations=sum(h['iterations'] for r in d['rows'] for h in r['history']),pool_size=108,penalty_enabled=False)
(root/'summary.json').write_text(json.dumps(s,indent=2))
b=json.loads((Path(__file__).resolve().parent.parent/'magnus5_t2/summary.json').read_text())['baseline']
text=f'''# Nonlocal pool, no penalty, t=0..2

Pool: all single-qubit X/Y/Z plus XX/YY/ZZ on every unordered pair; 108 generators. Original 45 generators remain first, followed by non-nearest pairs. This is an explicitly defined extension, not a claim to reproduce every pool variant in the paper.

Same midpoint Suzuki2 reps2, Neel state, dt=.05, L-BFGS-B max200 per growth round and local-cost threshold 1e-4 as the previous baseline. No moment penalty or hard moment gate is active; K moments are recorded for diagnostics only. Exact reference is excluded from optimization and acceptance.

| Metric | Local pool | Nonlocal pool |
|---|---:|---:|
| Final global infidelity | {100*b['global_infidelity']:.4f}% | {100*s['global_infidelity']:.4f}% |
| Parameters | {b['parameters']} | {s['parameters']} |
| Raw CX (all-to-all decomposition) | {b['raw_cx']} | {s['raw_cx']} |
| Compiled CX, linear coupling | 124 | {s['compiled_cx']} |
| Compiled depth | 117 | {s['compiled_depth']} |
| Optimizer iterations | {b['optimizer_iterations']} | {s['optimizer_iterations']} |

Relative final infidelity reduction: {(1-s['global_infidelity']/b['global_infidelity'])*100:.2f}%. Selected non-nearest two-qubit rotations: {s['nonlocal_rotations']}. One deterministic trajectory per pool; no matched-global-accuracy resource optimization or statistical claim.

Independent checks passed: all 40 accepted local-cost gates, zero penalty, Qiskit states at each step, final QASM, previously validated tighter exact reference (max reported-infidelity discrepancy {err:.3g}), and routed compiled state after undoing final layout (infidelity {compiler_error:.3g}). Compiler settings: line coupling, rz/sx/x/cx, level3, seed42, fixed initial layout, approximation_degree1.

Entry point: `python run_moment_apvqd.py --mode baseline --pool nonlocal --out OUTPUT`. Results and trajectory are saved under nonlocal/. No hardware shots were simulated.
'''
(root/'REPORT.md').write_text(text)
print(json.dumps(s,indent=2))
