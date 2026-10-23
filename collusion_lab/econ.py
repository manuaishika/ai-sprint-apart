"""Logit-demand Bertrand oligopoly (Calvano et al. 2020, AER) and price grid."""
from dataclasses import dataclass, field
import itertools

import numpy as np
from scipy.optimize import brentq, minimize_scalar


@dataclass
class Market:
    n: int = 2
    a: float = 2.0      # product quality (symmetric)
    a0: float = 0.0     # outside good
    c: float = 1.0      # marginal cost
    mu: float = 0.25    # horizontal differentiation
    m: int = 15         # grid size
    xi: float = 0.1     # grid extension beyond [p_N, p_M]
    p_nash: float = field(init=False)
    p_mon: float = field(init=False)
    grid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.p_nash = self._nash()
        self.p_mon = self._monopoly()
        lo = self.p_nash - self.xi * (self.p_mon - self.p_nash)
        hi = self.p_mon + self.xi * (self.p_mon - self.p_nash)
        self.grid = np.linspace(lo, hi, self.m)

    # demand / profit -------------------------------------------------------
    def demand(self, p):
        p = np.asarray(p, dtype=float)
        e = np.exp((self.a - p) / self.mu)
        return e / (e.sum(-1, keepdims=True) + np.exp(self.a0 / self.mu))

    def profit(self, p):
        p = np.asarray(p, dtype=float)
        return (p - self.c) * self.demand(p)

    def consumer_surplus(self, p):
        p = np.asarray(p, dtype=float)
        e = np.exp((self.a - p) / self.mu)
        return self.mu * np.log(e.sum(-1) + np.exp(self.a0 / self.mu))

    # benchmarks ------------------------------------------------------------
    def _nash(self):
        # symmetric FOC: p - c = mu / (1 - q_i)
        def foc(p):
            q = self.demand(np.full(self.n, p))[0]
            return p - self.c - self.mu / (1 - q)
        return brentq(foc, self.c + 1e-6, self.c + 10)

    def _monopoly(self):
        r = minimize_scalar(lambda p: -self.profit(np.full(self.n, p)).sum(),
                            bounds=(self.c, self.c + 10), method="bounded")
        return r.x

    @property
    def pi_nash(self):
        return float(self.profit(np.full(self.n, self.p_nash))[0])

    @property
    def pi_mon(self):
        return float(self.profit(np.full(self.n, self.p_mon))[0])

    def delta(self, pi):
        """Calvano's collusion index: 0 = static Nash, 1 = joint monopoly."""
        return (np.asarray(pi) - self.pi_nash) / (self.pi_mon - self.pi_nash)

    # tables for the numba kernels -------------------------------------------
    def profit_table(self):
        """PI[k, i]: profit of firm i at joint action index k (base-m, firm 0 most significant)."""
        idx = np.array(list(itertools.product(range(self.m), repeat=self.n)))
        return self.profit(self.grid[idx]).astype(np.float64)

    def cs_table(self):
        idx = np.array(list(itertools.product(range(self.m), repeat=self.n)))
        return self.consumer_surplus(self.grid[idx]).astype(np.float64)
