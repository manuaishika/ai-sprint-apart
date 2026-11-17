"""Follow-up experiments answering reviewer questions.

    python -m experiments.mechanisms relative      # which one-bit signals sustain collusion?
    python -m experiments.mechanisms convergence   # does Δ mean anything when no session settles?
    python -m experiments.mechanisms delay         # why does a one-period delay break collusion?
    python -m experiments.mechanisms all

Writes results/mech_*.csv.
"""
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from collusion_lab.econ import Market
from collusion_lab.qlearning import play
from collusion_lab.session import Config, run_session, tables

from .mind_vs_hands import read_policy

R = os.path.join(os.path.dirname(__file__), "..", "results")


# --- 1. relative vs. absolute one-bit signals -----------------------------------------
RELATIVE = {"absolute 2 bands (rival in upper/lower half of grid)": dict(coarsen=2),
            "relative 2 values (rival >= my price?)": dict(coarsen=-2),
            "relative 3 values (rival lower / equal / higher than mine)": dict(coarsen=-3),
            "absolute 1 band (no information)": dict(coarsen=1)}


def _rel(a):
    name, seed = a
    r = run_session(replace(Config(), **RELATIVE[name]), seed)
    return dict(signal=name, seed=seed, delta=r["delta"], converged=r["converged"],
                punish=r["probe_resp1"], dev_gain=r["probe_gain"])


# --- 2. convergence --------------------------------------------------------------------
CONV = {"jitter p=0.05": dict(jitter=0.05), "jitter p=0.1": dict(jitter=0.10), "noise eps=0.2": dict(eps=0.2)}
TMAX = [1_000_000, 3_000_000, 6_000_000, 12_000_000]


def _conv(a):
    name, tmax, seed = a
    r = run_session(replace(Config(), T_max=tmax, **CONV[name]), seed, audits=False)
    return dict(condition=name, T_max=tmax, seed=seed, delta=r["delta"], converged=r["converged"],
                t_used=r["t_conv"])


# --- 3. delay mechanism ----------------------------------------------------------------
LAGS = 9


def _delay(a):
    d, seed = a
    cfg = replace(Config(), delay=d)
    mk, PI, _ = tables(cfg.n, cfg.m)
    res, Q, hist, t = run_session(cfg, seed, audits=False, return_agents=True)
    arrs = cfg.arrays()
    n, m, at = cfg.n, cfg.m, 10
    args = (PI, Q, hist, t, n, m, arrs["blind"], arrs["eps"], cfg.coarsen, cfg.delay, 0.0, 0.0, at + LAGS + 2)
    base, dev = play(*args, -1, 0, seed), play(*args, at, 0, seed)
    unit, punit = mk.p_mon - mk.p_nash, mk.pi_mon - mk.pi_nash
    joint = lambda p: p[:, 0] * m + p[:, 1]
    resp = (mk.grid[dev[:, 1]] - mk.grid[base[:, 1]]) / unit
    gain = (PI[joint(dev), 0] - PI[joint(base), 0]) / punit
    intended = read_policy(cfg, Q, hist, t, mk, PI)["w_br"]
    return [dict(delay=d, seed=seed, delta=res["delta"], intended=intended, lag=l, resp=float(resp[at + l]),
                 gain=float(gain[at + l - 1])) for l in range(1, LAGS)]


def grim_trigger_bound(mk, delta=0.95):
    """Highest symmetric price (and its Δ) a grim-trigger scheme can sustain when a deviation
    goes unpunished for k periods (k = 1 + observation delay), with discount factor `delta`."""
    out = []
    pi = lambda p: float(mk.profit(np.full(2, p))[0])
    xs = np.linspace(mk.c, mk.c + 2, 4001)
    dev = lambda p: max(float(mk.profit(np.array([x, p]))[0]) for x in xs[::8])
    for k in range(1, 9):
        f = lambda p: pi(p) - (dev(p) * (1 - delta ** k) + delta ** k * mk.pi_nash)
        ps = np.linspace(mk.p_nash, mk.p_mon, 60)
        ok = [p for p in ps if f(p) >= 0]
        pmax = max(ok) if ok else mk.p_nash
        out.append(dict(k=k, delay=k - 1, p_max=pmax, delta_max=float(mk.delta(pi(pmax)))))
    return pd.DataFrame(out)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    procs = os.cpu_count()
    with Pool(procs) as p:
        if which in ("relative", "all"):
            jobs = [(n, s) for n in RELATIVE for s in range(100)]
            pd.DataFrame(p.map(_rel, jobs, chunksize=2)).to_csv(f"{R}/mech_relative.csv", index=False)
            print("relative done", flush=True)
        if which in ("delay", "all"):
            jobs = [(d, s) for d in (0, 1, 2, 4) for s in range(100)]
            rows = [r for rs in p.map(_delay, jobs, chunksize=2) for r in rs]
            pd.DataFrame(rows).to_csv(f"{R}/mech_delay.csv", index=False)
            grim_trigger_bound(Market()).to_csv(f"{R}/mech_grim.csv", index=False)
            print("delay done", flush=True)
        if which in ("convergence", "all"):
            jobs = [(n, t, s) for n in CONV for t in TMAX for s in range(60)]
            pd.DataFrame(p.map(_conv, jobs, chunksize=1)).to_csv(f"{R}/mech_convergence.csv", index=False)
            print("convergence done", flush=True)


if __name__ == "__main__":
    main()
