# Magnus-5 cumulative-moment APVQD experiment

The new entry point `run_moment_apvqd.py` implements the original local echo
cost and original 45-generator pool (8 sites, x/y/z and adjacent xx/yy/zz),
with Tetris growth and midpoint Suzuki order 2, reps 2. The original
`adaptive_pvqd.py` ADAM entry point is unchanged. Both experiment branches use
L-BFGS-B, max 200 iterations per growth round, gradient tolerance 5e-5.
Consequently compare these two matched branches to isolate the penalty;
comparison to previous ADAM/right-endpoint runs changes multiple factors.

`moment_constraints.magnus5` implements Eq. (1) of PRL 133, 010603 (2024),
with 8-point Gauss-Legendre integration of A1,A2,A3 and Pauli commutators.
K is constructed from the continuous H(t), not the Suzuki logarithm.
At each interval both endpoint moments use the same K. Mean and variance
are extensive moments divided by N. Signed sums are retained across intervals.

Penalty = lambda_E max(0,abs(R_E)/d_E-1)^2
        + lambda_V max(0,abs(R_V)/d_V-1)^2.
Default d_E=d_V=0.02, lambda_E=lambda_V=0.01. These are exploratory
parameters, not optimized hyperparameters or fidelity guarantees.
A step is committed only if local cost <=1e-4 and both budgets hold.
If a candidate violates a moment budget, SLSQP (max 200 iterations) polishes
the local cost on the same topology with explicit local-cost and moment
inequalities before more gates are added. The hinge penalty is identically zero
on the feasible set, so this is the same constrained objective without stiff
penalty curvature at infeasible trial points. An inward margin of 1e-8 protects
against roundoff. These additional iterations are recorded, not free resources.
At most 30 growth rounds are allowed. Failure is explicit in status.json;
no failed candidate is committed. The baseline records but does not enforce
moment budgets. No dt adaptation or penalty-weight escalation is used.

Run from this directory with the repository's Qiskit 0.43 environment:

```sh
OPENBLAS_NUM_THREADS=1 python tests/check_magnus_moments.py
OPENBLAS_NUM_THREADS=1 python run_moment_apvqd.py --mode baseline --out results/magnus5-baseline
OPENBLAS_NUM_THREADS=1 python run_moment_apvqd.py --mode penalty --out results/magnus5-penalty
```

Both runs stop at t=2, dt=0.05, start from the Neel state with zero parameters.
The exact DOP853 reference is used exclusively for reporting, never for
losses, pool selection or acceptance. A Qiskit statevector independently
checks the circuit at each accepted step. Final QASM and topology are saved.

This entry point is an exact statevector experimental implementation: adjoint
gradients and sparse matrix-vector moments accelerate simulation. It does not
implement shot allocation, measurement grouping, hardware execution or
statistical budget acceptance. K is already represented as a measurable
Pauli sum, but these hardware features remain future work. Finite-order
Magnus conservation defects and two-moment constraints do not certify global
state fidelity. The statevector backend reuses the original algorithm's local
cost in `moment_statevector.py`; tests should be run when changing conventions.

The tighter exploratory budget 0.002 was stopped for computational cost
before t=2; it is not the default or a demonstrated infeasibility result.
