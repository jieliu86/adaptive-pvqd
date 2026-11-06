# Nonlocal pool, no penalty, t=0..2

Pool: all single-qubit X/Y/Z plus XX/YY/ZZ on every unordered pair; 108 generators. Original 45 generators remain first, followed by non-nearest pairs. This is an explicitly defined extension, not a claim to reproduce every pool variant in the paper.

Same midpoint Suzuki2 reps2, Neel state, dt=.05, L-BFGS-B max200 per growth round and local-cost threshold 1e-4 as the previous baseline. No moment penalty or hard moment gate is active; K moments are recorded for diagnostics only. Exact reference is excluded from optimization and acceptance.

| Metric | Local pool | Nonlocal pool |
|---|---:|---:|
| Final global infidelity | 4.4341% | 1.7421% |
| Parameters | 192 | 232 |
| Raw CX (all-to-all decomposition) | 224 | 416 |
| Compiled CX, linear coupling | 124 | 1064 |
| Compiled depth | 117 | 961 |
| Optimizer iterations | 2623 | 5202 |

Relative final infidelity reduction: 60.71%. Selected non-nearest two-qubit rotations: 141. One deterministic trajectory per pool; no matched-global-accuracy resource optimization or statistical claim.

Independent checks passed: all 40 accepted local-cost gates, zero penalty, Qiskit states at each step, final QASM, previously validated tighter exact reference (max reported-infidelity discrepancy 1.07e-11), and routed compiled state after undoing final layout (infidelity 4.88e-15). Compiler settings: line coupling, rz/sx/x/cx, level3, seed42, fixed initial layout, approximation_degree1.

Entry point: `python run_moment_apvqd.py --mode baseline --pool nonlocal --out OUTPUT`. Results and trajectory are saved under nonlocal/. No hardware shots were simulated.
