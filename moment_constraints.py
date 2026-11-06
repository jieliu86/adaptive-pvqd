"""Piecewise Magnus-5 observables and signed cumulative moment penalties.

Equations (1), (5), PRL 133, 010603 (2024). Operators are PauliSumOp;
no matrix logarithm or exact-state oracle is used to build constraints.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss


def magnus5(hamiltonian, t, dt, quadrature=8):
    if dt <= 0:
        raise ValueError('dt must be positive')
    nodes, weights = leggauss(quadrature)
    x = (nodes + 1) / 2
    samples = [hamiltonian(t + dt * z) for z in x]
    polys = [np.ones_like(x), 2*x-1, 6*x*x-6*x+1]
    A = []
    for j, poly in enumerate(polys, 1):
        value = samples[0] * 0
        for H, w, p in zip(samples, weights/2, poly):
            value += H * complex(-1j*(2*j-1)*dt*w*p)
        A.append(value.reduce(atol=1e-14))
    def comm(a, b):
        return (a @ b - b @ a).reduce(atol=1e-14)
    a, b, c = A
    ab = comm(a, b)
    omega3 = -ab / 6
    omega5 = (comm(a, comm(a, c))/60 - comm(b, ab)/60
              + comm(a, comm(a, ab))/360 - comm(b, c)/30)
    return ((a + omega3 + omega5) * (1j/dt)).reduce(atol=1e-12)


def hinge(residual, budget):
    if budget <= 0:
        raise ValueError('moment budgets must be positive')
    excess = max(0., abs(residual)/budget - 1)
    return excess**2, 2*excess*np.sign(residual)/budget


@dataclass
class MomentController:
    energy_budget: float = .02
    variance_budget: float = .02
    energy_weight: float = .01
    variance_weight: float = .01
    sum_energy: float = 0.
    sum_variance: float = 0.

    def evaluate(self, energy, variance, input_energy, input_variance):
        re = self.sum_energy + energy - input_energy
        rv = self.sum_variance + variance - input_variance
        pe, de = hinge(re, self.energy_budget)
        pv, dv = hinge(rv, self.variance_budget)
        return dict(residual_energy=float(re), residual_variance=float(rv),
                    penalty=float(self.energy_weight*pe+self.variance_weight*pv),
                    derivative_energy=self.energy_weight*de,
                    derivative_variance=self.variance_weight*dv,
                    feasible=bool(abs(re)<=self.energy_budget and abs(rv)<=self.variance_budget))

    def commit(self, diagnostics):
        if not diagnostics['feasible']:
            raise ValueError('Cannot commit an infeasible candidate')
        self.sum_energy = diagnostics['residual_energy']
        self.sum_variance = diagnostics['residual_variance']
