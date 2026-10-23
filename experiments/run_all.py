"""Run every experiment in the report and write CSVs to results/.

    python -m experiments.run_all            # full run (~10 min on 4 cores)
    python -m experiments.run_all --quick    # smoke test
"""
import argparse
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from collusion_lab.qlearning import init_q, play, train
from collusion_lab.session import Config, outcomes, probe, run_session, tables

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _job(args):
    cfg, seed = args
    return run_session(cfg, seed)


def sweep(cfgs, seeds, procs):
    jobs = [(c, s) for c in cfgs for s in range(seeds)]
    with Pool(procs) as p:
        return pd.DataFrame(p.map(_job, jobs, chunksize=4))


# --- E1/E2: baselines and interventions --------------------------------------------
def intervention_configs():
    b = Config()
    cfgs = [b, replace(b, n=3, label="entry_n3"), replace(b, blind=True, label="blind")]
    cfgs += [replace(b, eps=e, label="noise") for e in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)]
    cfgs += [replace(b, coarsen=k, label="coarsen") for k in (8, 4, 2, 1)]
    cfgs += [replace(b, delay=d, label="delay") for d in (1, 2, 4, 8)]
    cfgs += [replace(b, jitter=j, label="jitter") for j in (0.02, 0.05, 0.1, 0.2)]
    return cfgs


# --- E3: swapping agents ------------------------------------------------------------
def _train_pair(seed):
    res, Q, hist, t = run_session(Config(), seed, audits=False, return_agents=True)
    return seed, Q, hist, t, res["delta"]


def _swap_job(args):
    kind, seed, Qa, Qb, hist, t_conv = args
    cfg = Config()
    mk, PI, CS = tables(cfg.n, cfg.m)
    n, m = cfg.n, cfg.m
    arrs = cfg.arrays()
    Q = np.stack([Qa[0], Qb[1]]).copy()
    hist = hist.copy()
    t0 = np.array([t_conv, t_conv], dtype=float)
    learn = np.ones(n, dtype=np.bool_)
    gamma = np.full(n, cfg.gamma)
    if kind in ("matched", "crossplay"):
        t = t_conv
    else:
        fresh_gamma = 0.0 if kind == "myopic_entrant" else cfg.gamma
        Qf = init_q(PI, n, m, np.array([fresh_gamma, fresh_gamma]), m ** n)
        Q[1] = Qf[1]
        gamma[1] = fresh_gamma
        t0[1] = 0.0
        if kind == "entrant_vs_frozen":
            learn[0] = False
        t, _ = train(PI, Q, hist, n, m, cfg.alpha, gamma, cfg.beta, t0, learn,
                     arrs["blind"], arrs["eps"], cfg.coarsen, cfg.delay, cfg.jitter,
                     cfg.T_max, cfg.conv_window, seed + 99)
    path = play(PI, Q, hist, t, n, m, arrs["blind"], arrs["eps"], cfg.coarsen,
                cfg.delay, cfg.jitter, 0.0, 500, -1, 0, seed)
    out = dict(kind=kind, seed=seed, **outcomes(cfg, PI, CS, mk, path[100:]))
    out.update(probe(replace(cfg, gamma=0.95), PI, mk, Q, hist, t, arrs, 1, seed))
    return out


def swap_experiment(n_sessions, procs):
    with Pool(procs) as p:
        pairs = p.map(_train_pair, range(n_sessions))
    jobs = []
    for i, (s, Q, h, t, d) in enumerate(pairs):
        jobs.append(("matched", s, Q, Q, h, t))
        # firm 0 of session i meets firm 1 of session i+1 (fixed derangement)
        _, Q2, _, _, _ = pairs[(i + 1) % n_sessions]
        jobs.append(("crossplay", s, Q, Q2, h, t))
        for kind in ("fresh_entrant", "entrant_vs_frozen", "myopic_entrant"):
            jobs.append((kind, s, Q, Q, h, t))
    with Pool(procs) as p:
        return pd.DataFrame(p.map(_swap_job, jobs))


# --- E4: detection ------------------------------------------------------------------
def detection_configs():
    b = Config()
    return [replace(b, label="collusive"),
            replace(b, gamma=0.0, label="myopic"),
            replace(b, gamma=0.5, label="short_horizon"),
            replace(b, blind=True, label="blind"),
            replace(b, eps=0.2, label="collusive_noisy_obs")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    seeds = 6 if a.quick else 100
    os.makedirs(OUT, exist_ok=True)

    print("E1/E2 interventions ...", flush=True)
    sweep(intervention_configs(), seeds, a.procs).to_csv(f"{OUT}/interventions.csv", index=False)
    print("E3 swaps ...", flush=True)
    swap_experiment(seeds, a.procs).to_csv(f"{OUT}/swaps.csv", index=False)
    print("E4 detection ...", flush=True)
    sweep(detection_configs(), 2 * seeds, a.procs).to_csv(f"{OUT}/detection.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
