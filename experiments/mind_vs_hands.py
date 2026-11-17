"""Reading the agent's strategy ("mind") vs. watching its prices ("hands").

A white-box auditor reads each agent's learned policy directly: in the market's current
state, what would the agent do if its rival undercut? No market interaction is needed,
and observation noise does not blur the answer. We compare this with the black-box
detectors from the detection study on the same sessions (same configs, same seeds).

    python -m experiments.mind_vs_hands     # re-runs every detection session -> results/mind_vs_hands.csv
"""
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from collusion_lab.qlearning import play
from collusion_lab.session import Config, run_session, tables

from .run_all import detection_configs

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "mind_vs_hands.csv")


def read_policy(cfg, Q, hist, t, mk, PI):
    """White-box scores, averaged over both firms acting as the responder.

    w_all: mean change in the responder's planned price when its rival's last price is any
           lower grid price instead of the on-path one (units of p_M - p_N; negative = cuts)
    w_br:  the same for the single undercut the probe uses (rival's static best response)
    """
    m, n = cfg.m, cfg.n
    if cfg.blind:
        return dict(w_all=0.0, w_br=0.0)
    zeros = np.zeros(n)
    path = play(PI, Q, hist, t, n, m, np.full(n, False), zeros, cfg.coarsen, cfg.delay, 0.0, 0.0, 200, -1, 0, 0)
    a = path[-1]
    unit = mk.p_mon - mk.p_nash
    w_all, w_br = [], []
    for j in range(n):              # responder
        i = 1 - j                   # deviator
        def state(own, rival):
            v = [0, 0]
            v[j] = own
            v[i] = (rival * cfg.coarsen) // m if cfg.coarsen < m else rival
            return v[0] * m + v[1]
        on = int(np.argmax(Q[j, state(a[j], a[i])]))
        devs = range(0, a[i])
        if len(devs):
            w_all.append(np.mean([mk.grid[int(np.argmax(Q[j, state(a[j], u)]))] - mk.grid[on] for u in devs]) / unit)
        # rival's static best response to the responder's price
        tmp = [0, 0]
        best, bv = a[i], -1e9
        for c in range(m):
            tmp[i], tmp[j] = c, a[j]
            v = PI[tmp[0] * m + tmp[1], i]
            if v > bv:
                bv, best = v, c
        w_br.append((mk.grid[int(np.argmax(Q[j, state(a[j], best)]))] - mk.grid[on]) / unit)
    return dict(w_all=float(np.mean(w_all)) if w_all else 0.0, w_br=float(np.mean(w_br)))


def job(args):
    cfg, seed = args
    mk, PI, _ = tables(cfg.n, cfg.m)
    res, Q, hist, t = run_session(cfg, seed, audits=False, return_agents=True)
    return dict(label=cfg.label, seed=seed, delta_check=res["delta"], **read_policy(cfg, Q, hist, t, mk, PI))


def figure_and_table():
    """Figure 3 + a markdown table, joining with the black-box detectors (same sessions)."""
    from .analyze import auc_ci, matched_sample
    from .style import BLUE, INK2, MUTED, ORANGE, plt

    R = os.path.dirname(OUT)
    df = pd.read_csv(os.path.join(R, "detection.csv")).merge(pd.read_csv(OUT), on=["label", "seed"])
    assert np.allclose(df.delta, df.delta_check), "white-box sessions must reproduce the detection sessions"
    names = {"collusive": "colluders,\nexact view", "collusive_noisy_obs": "colluders,\nnoisy view",
             "short_horizon": "short-horizon\n(γ = 0.5)", "myopic": "myopic\n(γ = 0)", "blind": "blind"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    for k, (col, lab, c) in enumerate([("w_br", "intended (read from strategy)", ORANGE),
                                       ("probe_resp1", "enacted (forced price cut in market)", BLUE)]):
        v = [df[df.label == l][col].mean() for l in names]
        se = [df[df.label == l][col].std() / np.sqrt((df.label == l).sum()) for l in names]
        ax.bar(x + (k - 0.5) * 0.36, v, 0.34, color=c, label=lab, yerr=np.multiply(se, 1.96),
               error_kw=dict(lw=0.8, ecolor=INK2))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, list(names.values()), fontsize=7.5)
    ax.set_ylabel("rival's price response to an undercut\n(units of p_M − p_N; < 0 = retaliates)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    ax.set_title("A   Intended vs. enacted retaliation")

    ax = axes[1]
    rows = []
    scen = {"all agents": df, "equal prices (Δ-matched)": matched_sample(df)}
    dets = [("prices high", "delta", 1, MUTED), ("forced cut: punished", "probe_resp1", -1, BLUE),
            ("forced cut: doesn't pay", "probe_gain", -1, BLUE), ("lead–lag reaction", "passive_leadlag", 1, BLUE),
            ("read strategy", "w_all", -1, ORANGE)]
    width = 0.38
    for k, (sname, g) in enumerate(scen.items()):
        y = g.label.isin(["collusive", "collusive_noisy_obs"]).values
        for i, (dn, col, sg, c) in enumerate(dets):
            v, lo, hi = auc_ci(sg * g[col].values, y)
            rows.append(dict(scenario=sname, detector=dn, auc=v, lo=lo, hi=hi))
            ax.barh(i + (k - 0.5) * width, v, width * 0.92, color=c, alpha=1.0 if k == 0 else 0.55)
            ax.plot([lo, hi], [i + (k - 0.5) * width] * 2, color=INK2, lw=0.8)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(dets)), [d[0] for d in dets])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("ROC AUC (solid: all agents; faded: equal prices)")
    ax.grid(axis="y", visible=False)
    ax.set_title("B   Watching hands vs. reading the mind")
    fig.tight_layout(w_pad=2)
    fig.savefig(os.path.join(R, "figures", "mind_vs_hands.png"))
    plt.close(fig)
    t = pd.DataFrame(rows)
    t["AUC (95% CI)"] = t.apply(lambda r: f"{r.auc:.2f} ({r.lo:.2f}–{r.hi:.2f})", axis=1)
    print(t.pivot(index="detector", columns="scenario", values="AUC (95% CI)").to_markdown())
    print(df.groupby("label")[["w_br", "w_all", "probe_resp1"]].mean().round(2).to_markdown())


if __name__ == "__main__":
    n_seeds = int(pd.read_csv(os.path.join(os.path.dirname(OUT), "detection.csv")).groupby("label").size().min())
    cfgs = detection_configs()
    jobs = [(c, s) for c in cfgs for s in range(n_seeds)]
    with Pool(os.cpu_count()) as p:
        pd.DataFrame(p.map(job, jobs, chunksize=4)).to_csv(OUT, index=False)
    print("done", n_seeds, "sessions per condition")
    figure_and_table()
