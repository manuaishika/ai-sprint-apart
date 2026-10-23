"""Turn results/*.csv into tables (results/summary.md) and figures (results/figures/)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(R, "figures")
M = 15
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160,
    "lines.linewidth": 2,
})


def ci(x, reps=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (reps, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def bits_noise(eps, m=M):
    p_hit = 1 - eps + eps / m
    p_miss = eps / m
    h = -p_hit * np.log2(p_hit) - ((m - 1) * p_miss * np.log2(p_miss) if p_miss > 0 else 0)
    return np.log2(m) - h


def bits_coarsen(k, m=M):
    counts = np.bincount((np.arange(m) * k) // m)
    p = counts[counts > 0] / m
    return float(-(p * np.log2(p)).sum())


def auc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def auc_ci(score, label, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(score, float), np.asarray(label, bool)
    vals = []
    for _ in range(reps):
        i = rng.integers(0, len(s), len(s))
        if y[i].all() or (~y[i]).all():
            continue
        vals.append(auc(s[i], y[i]))
    return auc(s, y), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def fmt(m, lo, hi, d=2):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ------------------------------------------------------------------------------------
def interventions(out):
    df = pd.read_csv(f"{R}/interventions.csv")
    base = df[df.label == "baseline"]
    b_cs, b_w = base.cs.mean(), base.welfare.mean()

    def dose(r):
        return {"baseline": "-", "entry_n3": "3 firms", "blind": "no rival info",
                "noise": f"eps={r.eps}", "coarsen": f"{r.coarsen} buckets",
                "delay": f"{r.delay} periods", "jitter": f"p={r.jitter}"}[r.label]

    def bits(r):
        if r.label == "blind":
            return 0.0
        if r.label == "noise":
            return bits_noise(r.eps)
        if r.label == "coarsen":
            return bits_coarsen(r.coarsen)
        return np.log2(M)

    df["dose"] = df.apply(dose, axis=1)
    df["bits"] = df.apply(bits, axis=1)
    rows = []
    for (lab, d), g in df.groupby(["label", "dose"], sort=False):
        dm = ci(g.delta)
        rows.append(dict(intervention=lab, dose=d, bits=g.bits.iloc[0], delta=fmt(*dm),
                         delta_mean=dm[0], converged=f"{g.converged.mean():.0%}",
                         dCS=g.cs.mean() - b_cs, dW=g.welfare.mean() - b_w,
                         punish=g.probe_resp1.mean(), dev_gain=g.probe_gain.mean()))
    t = pd.DataFrame(rows)
    out.append("## E1/E2: interventions (n=2 unless noted, 100 seeds each)\n")
    out.append("`delta`: collusion index, mean [95% bootstrap CI]; 0 = static Nash, 1 = monopoly. "
               "`bits`: information about each rival price the agent can observe (uniform prior). "
               "`dCS`/`dW`: change in consumer surplus / total welfare vs. baseline. "
               "`punish`: rival's price response one period after a forced deviation "
               "(units of p_M - p_N; negative = punishment). `dev_gain`: discounted gain from "
               "the deviation (units of pi_M - pi_N; negative = deviation unprofitable).\n")
    out.append(t.drop(columns="delta_mean").to_markdown(index=False, floatfmt=".3f") + "\n")
    t.to_csv(f"{R}/interventions_summary.csv", index=False)

    # figure: dose-response small multiples
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    b = ci(base.delta)
    panels = [("noise", "eps", "observation noise eps"),
              ("coarsen", "coarsen", "rival-price buckets"),
              ("delay", "delay", "observation delay (periods)"),
              ("jitter", "jitter", "mandated price jitter p")]
    for ax, (lab, col, xl) in zip(axes, panels):
        g = df[df.label == lab]
        xs = sorted(g[col].unique())
        base_x = {"noise": 0.0, "coarsen": 15, "delay": 0, "jitter": 0.0}[lab]
        pts = sorted([(base_x, b)] + [(x, ci(g[g[col] == x].delta)) for x in xs], key=lambda z: z[0])
        xs_all = [x for x, _ in pts]
        stats = [st for _, st in pts]
        mu = [s[0] for s in stats]
        ax.fill_between(xs_all, [s[1] for s in stats], [s[2] for s in stats],
                        color=SERIES[0], alpha=0.15, lw=0)
        ax.plot(xs_all, mu, color=SERIES[0], marker="o", ms=5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(xl)
        if lab == "coarsen":
            ax.set_xscale("log", base=2)
            ax.invert_xaxis()
            ax.set_xticks([15, 8, 4, 2, 1], ["15", "8", "4", "2", "1"])
    axes[0].set_ylabel("collusion index Δ")
    fig.suptitle("Which restriction removes collusion? Mean Δ with 95% CI, 100 sessions per point",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/dose_response.png")
    plt.close(fig)

    # figure: delta vs bits retained (information-only interventions)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for i, lab in enumerate(["noise", "coarsen"]):
        g = t[t.intervention.isin([lab, "baseline", "blind"])].sort_values("bits")
        ax.plot(g.bits, g.delta_mean, marker="o", ms=5, color=SERIES[i], label=lab)
    ax.set_xlabel("bits observable about each rival price")
    ax.set_ylabel("mean collusion index Δ")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Collusion vs. information channel capacity", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/delta_vs_bits.png")
    plt.close(fig)
    return t


def swaps(out):
    df = pd.read_csv(f"{R}/swaps.csv")
    names = {"matched": "original pair (no swap)",
             "crossplay": "cross-play: partner from another session, no relearning",
             "fresh_entrant": "fresh Q-learner replaces firm 1, both keep learning",
             "entrant_vs_frozen": "fresh Q-learner vs. frozen incumbent",
             "myopic_entrant": "myopic (gamma=0) learner replaces firm 1"}
    rows = []
    for k in names:
        g = df[df.kind == k]
        rows.append(dict(condition=names[k], delta=fmt(*ci(g.delta)),
                         punish=g.probe_resp1.mean(), dev_gain=g.probe_gain.mean(),
                         share_delta_gt_half=f"{(g.delta > 0.5).mean():.0%}"))
    out.append("## E3: does collusion survive swapping an agent? (100 sessions)\n")
    out.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f") + "\n")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, k in enumerate(names):
        m, lo, hi = ci(df[df.kind == k].delta)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=SERIES[0], ms=6, capsize=0)
    ax.set_yticks(range(len(names)), [names[k].split(":")[0].split(",")[0] for k in names])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("collusion index Δ (mean, 95% CI)")
    ax.set_title("Swapping one agent", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/swaps.png")
    plt.close(fig)


def detection(out):
    df = pd.read_csv(f"{R}/detection.csv")
    df["y"] = df.label.isin(["collusive", "collusive_noisy_obs"])
    detectors = {
        "D1 price level (Δ)": "delta",
        "D2 parallel prices (level corr.)": "passive_level_corr",
        "D3 lead-lag reaction (passive)": "passive_leadlag",
        "D4 active probe: punishment": "probe_resp1",
        "D5 active probe: deviation unprofitable": "probe_gain",
    }
    sign = {"probe_resp1": -1, "probe_gain": -1, "passive_leadlag": 1}

    def score(g, col):
        return sign.get(col, 1) * g[col].values

    scenarios = {
        "all controls": df[df.label.isin(["collusive", "myopic", "short_horizon", "blind"])],
        "vs. myopic only": df[df.label.isin(["collusive", "myopic"])],
        "high prices only (Δ>0.2)": df[df.label.isin(["collusive", "myopic", "short_horizon", "blind"])
                                        & (df.delta > 0.2)],
        "noisy-obs collusion vs. all controls": df[df.label.isin(["collusive_noisy_obs", "myopic",
                                                                   "short_horizon", "blind"])],
    }
    # Δ-matched sample: every myopic session with supracompetitive prices is paired with
    # the γ=0.95 session (clean or noisy observation) of nearest Δ, without replacement.
    # Pairs further apart than a caliper of 0.02 in Δ are dropped.
    cands = df[(df.label == "myopic") & (df.delta > 0.1)].sort_values("delta", ascending=False)
    pool = df[df.label.isin(["collusive", "collusive_noisy_obs"])].copy()
    neg_idx, pos_idx = [], []
    for i, d in cands.delta.items():
        dist = (pool.delta - d).abs()
        if len(pool) and dist.min() <= 0.02:
            j = dist.idxmin()
            neg_idx.append(i)
            pos_idx.append(j)
            pool = pool.drop(j)
    matched = df.loc[neg_idx + pos_idx]
    scenarios["Δ-matched (caliper 0.02): γ=0.95 vs. myopic"] = matched
    gap = np.abs(df.loc[neg_idx].delta.values - df.loc[pos_idx].delta.values).mean()

    out.append("## E4: telling collusion from coincidence (200 sessions per condition)\n")
    out.append(f"Δ-matched scenario: mean |Δ| gap between matched pairs = {gap:.3f}.\n")
    desc = df.groupby("label").agg(delta=("delta", "mean"), share_high=("delta", lambda d: (d > 0.2).mean()),
                                   punish=("probe_resp1", "mean"), dev_gain=("probe_gain", "mean"),
                                   level_corr=("passive_level_corr", "mean"),
                                   leadlag=("passive_leadlag", "mean"))
    out.append(desc.to_markdown(floatfmt=".3f") + "\n")
    out.append("ROC AUC (95% bootstrap CI). Positive class = agents that can condition on rivals "
               "and value the future (gamma=0.95); negatives = myopic (gamma=0), short-horizon "
               "(gamma=0.5) and rival-blind agents.\n")
    rows = []
    for sname, g in scenarios.items():
        r = {"scenario": f"{sname} (n={len(g)}, pos={int(g.y.sum())})"}
        for dname, col in detectors.items():
            r[dname] = fmt(*auc_ci(score(g, col), g.y))
        rows.append(r)
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    # ROC figure for the hardest scenario
    g = scenarios["Δ-matched (caliper 0.02): γ=0.95 vs. myopic"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for i, (dname, col) in enumerate(detectors.items()):
        s = score(g, col)
        thr = np.unique(s)[::-1]
        tpr = [((s >= t) & g.y).sum() / g.y.sum() for t in thr]
        fpr = [((s >= t) & ~g.y).sum() / (~g.y).sum() for t in thr]
        ax.plot([0] + fpr + [1], [0] + tpr + [1], color=SERIES[i], label=f"{dname}  AUC={auc(s, g.y):.2f}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("Auditing at equal price levels (Δ-matched)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/detection_roc.png")
    plt.close(fig)

    # distribution of delta by condition
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labs = ["collusive", "short_horizon", "myopic", "blind", "collusive_noisy_obs"]
    for i, lab in enumerate(labs):
        d = df[df.label == lab].delta.values
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(d))
        ax.scatter(d, i + jit, s=9, color=SERIES[0] if lab.startswith("collusive") else SERIES[1],
                   alpha=0.6, lw=0)
    ax.set_yticks(range(len(labs)), labs)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("collusion index Δ (one dot per session)")
    ax.set_title("High prices also arise without collusive capacity", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/delta_by_condition.png")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    out = ["# Results summary (auto-generated by experiments/analyze.py)\n"]
    interventions(out)
    swaps(out)
    detection(out)
    open(f"{R}/summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
