"""Robustness: do the main lever results hold under other learning parameters?

    python -m experiments.robustness      # ~10 min on 4 cores -> results/robustness.csv
"""
import os
from dataclasses import replace
from multiprocessing import Pool

import pandas as pd

from collusion_lab.session import Config, run_session

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "robustness.csv")
SETTINGS = {"slower exploration (β=4e-6, Calvano)": dict(beta=4e-6, T_max=6_000_000),
            "lower learning rate (α=0.1)": dict(alpha=0.10),
            "higher learning rate (α=0.25)": dict(alpha=0.25)}
LEVERS = {"none": {}, "1 absolute bit (upper/lower half)": dict(coarsen=2), "20% noise": dict(eps=0.2),
          "1-period delay": dict(delay=1), "blind": dict(blind=True)}


def job(a):
    setting, lever, seed = a
    cfg = replace(Config(), **SETTINGS[setting], **LEVERS[lever])
    r = run_session(cfg, seed, audits=False)
    return dict(setting=setting, lever=lever, seed=seed, delta=r["delta"], converged=r["converged"])


if __name__ == "__main__":
    jobs = [(s, l, k) for s in SETTINGS for l in LEVERS for k in range(50)]
    with Pool(os.cpu_count()) as p:
        pd.DataFrame(p.map(job, jobs, chunksize=2)).to_csv(OUT, index=False)
    print("done")
