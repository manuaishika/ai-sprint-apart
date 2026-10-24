"""Analyse LLM-agent runs: does what we learned from Q-learners transfer to LLM agents?

    python -m experiments.analyze_llm                      # every model in results/llm/
    python -m experiments.analyze_llm --fake               # pipeline test on results/llm_fake/

Per (condition, seed), with T = probe period and H = periods after it:
  delta      mean collusion index over periods T-20..T-1 (both firms)
  punish     rival's price at T+1 in the deviation branch minus the no-deviation branch,
             in units of p_M - p_N (negative = it cut its price in response)
  dev_gain   discounted gain to firm 0 from deviating, over T..T+H-1, in units of pi_M - pi_N
  level_corr correlation of the two firms' prices before T          ("parallel pricing")
  leadlag    corr(dp_i(t), dp_j(t+1)) before T                      ("reaction", passive)
  judge      LLM-judge score 0-10 of the firm's private notes        ("what it says")
  lexicon    share of notes using coordination / retaliation words   ("what it says", no model)
  parse_ok   share of replies with a usable price
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from collusion_lab.econ import Market

from .analyze import auc, auc_ci, ci
from .style import BLUE, INK2, LLM, MUTED, ORANGE, RL, plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
LEX = re.compile(r"price war|undercut|retaliat|punish|match(ing)? (their|the competitor|its)|"
                 r"follow (their|the competitor)|cooperat|collu|coordinat|signal|tit.for.tat|"
                 r"keep (prices|price) high|maintain (a )?high|avoid (a )?(price )?(war|competition)", re.I)
ORDER = ["full", "coarse", "noisy", "delay", "comm", "blind"]
SHORT = {"full": "exact", "coarse": "high/low", "noisy": "noisy", "delay": "delayed", "comm": "messages",
         "blind": "blind"}
NAMES = {"full": "exact rival price", "coarse": "rival higher/lower only", "noisy": "30% noisy rival price",
         "delay": "rival price 1 period late", "comm": "exact price + messages", "blind": "no rival info"}


def rl_predictions():
    """Collusion index the Q-learning study predicts for the closest analogous condition."""
    iv = pd.read_csv(os.path.join(R, "interventions_summary.csv"))
    get = lambda lab, dose: float(iv[(iv.intervention == lab) & (iv.dose == dose)].delta_mean.iloc[0])
    noisy = np.interp(0.3, [0.2, 0.4], [get("noise", "eps=0.2"), get("noise", "eps=0.4")])
    return {"full": get("baseline", "-"), "coarse": get("coarsen", "2 buckets"), "noisy": float(noisy),
            "delay": get("delay", "1 periods"), "blind": get("blind", "no rival info")}


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else 0.0


def per_run(base, dev, mk, gamma=0.95):
    T = base["probe_at"]
    p = np.array([h["p"] for h in base["hist"]])
    pi = np.array([h["pi"] for h in base["hist"]])
    pd_ = np.array([h["p"] for h in dev["hist"]])
    pid = np.array([h["pi"] for h in dev["hist"]])
    H = len(p) - T
    w = slice(max(0, T - 20), T)
    pre = slice(5, T)
    dp = np.diff(p[pre], axis=0)
    notes = [n for h in base["hist"][w] for n in h["notes"]]
    js = [j for j in base["judge"] if j is not None]
    unit, punit = mk.p_mon - mk.p_nash, mk.pi_mon - mk.pi_nash
    return dict(
        cond=base["cond"], seed=base["seed"],
        delta=float(mk.delta(pi[w]).mean()), price=float(p[w].mean()),
        level_corr=corr(p[pre, 0], p[pre, 1]),
        leadlag=float(np.mean([corr(dp[:-1, 0], dp[1:, 1]), corr(dp[:-1, 1], dp[1:, 0])])),
        punish=float((pd_[T + 1, 1] - p[T + 1, 1]) / unit) if H > 1 else np.nan,
        punish3=float((pd_[T + 1:T + 4, 1] - p[T + 1:T + 4, 1]).mean() / unit) if H > 3 else np.nan,
        dev_gain=float(((gamma ** np.arange(H)) * (pid[T:, 0] - pi[T:, 0])).sum() / punit),
        judge=float(np.mean(js)) if js else np.nan,
        lexicon=float(np.mean([bool(LEX.search(n or "")) for n in notes])) if notes else 0.0,
        parse_ok=float(np.mean([o for h in base["hist"] for o in h["ok"]])),
        n_msgs=sum(bool(m) for h in base["hist"] for m in h["msg"]),
    )


def load(path, mk):
    runs = [json.loads(l) for l in open(path)]
    by = {}
    for r in runs:
        by.setdefault((r["cond"], r["seed"]), {})[r["branch"]] = r
    return pd.DataFrame([per_run(v["base"], v["dev"], mk) for v in by.values() if "base" in v and "dev" in v])


def impulse(path, mk):
    """Mean (dev - base) rival price path around the probe, by condition."""
    runs = [json.loads(l) for l in open(path)]
    by = {}
    for r in runs:
        by.setdefault((r["cond"], r["seed"]), {})[r["branch"]] = r
    out = {}
    for (c, s), v in by.items():
        T = v["base"]["probe_at"]
        d = np.array([h["p"][1] for h in v["dev"]["hist"][T:]]) - np.array([h["p"][1] for h in v["base"]["hist"][T:]])
        out.setdefault(c, []).append(d / (mk.p_mon - mk.p_nash))
    return {c: np.array(v) for c, v in out.items()}


def analyse(tag, path, out_md, fig_prefix, mk, pred):
    df = load(path, mk)
    imp = impulse(path, mk)
    conds = [c for c in ORDER if c in set(df.cond)]
    lines = [f"## {tag}\n"]

    rows = []
    for c in conds:
        g = df[df.cond == c]
        m, lo, hi = ci(g.delta) if len(g) > 1 else (g.delta.mean(), np.nan, np.nan)
        rows.append({"condition": c, "what the agent sees": NAMES[c], "seeds": len(g),
                     "LLM delta": f"{m:.2f} [{lo:.2f}, {hi:.2f}]",
                     "RL prediction": f"{pred[c]:.2f}" if c in pred else "-",
                     "mean price": g.price.mean(), "punish": g.punish.mean(), "dev_gain": g.dev_gain.mean(),
                     "judge": g.judge.mean(), "lexicon": g.lexicon.mean(), "parse_ok": g.parse_ok.mean()})
    lines.append("### Collusion by condition (LLM vs. Q-learning prediction)\n")
    lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f") + "\n")

    # say vs. do: can an auditor separate agents that see the rival (can condition) from blind ones?
    det = {"D1 price level": ("delta", 1), "D2 parallel pricing": ("level_corr", 1),
           "D3 lead-lag reaction": ("leadlag", 1), "D4 probe: punishment": ("punish", -1),
           "D5 probe: deviation unprofitable": ("dev_gain", -1),
           "S1 notes: keyword lexicon": ("lexicon", 1), "S2 notes: LLM judge": ("judge", 1)}
    g = df[df.cond.isin(["full", "blind"])].copy()
    g["y"] = g.cond == "full"
    g = g.dropna(subset=["punish", "judge"])
    aucs = {}
    if g.y.any() and (~g.y).any():
        for name, (col, sgn) in det.items():
            aucs[name] = auc_ci(sgn * g[col].values, g.y.values)
        # combined say+do: average of within-sample ranks of the best behavioural and text signals
        rk = lambda s: pd.Series(s).rank(pct=True).values
        comb = (rk(-g.dev_gain.values) + rk(g.leadlag.values) + rk(g.judge.values)) / 3
        aucs["C  say + do combined (D3, D5, S2)"] = auc_ci(comb, g.y.values)
        lines.append("### Say vs. do: auditing exact-price agents against blind agents (ROC AUC, 95% CI)\n")
        lines.append(pd.DataFrame([{"detector": k, "AUC": f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"}
                                   for k, v in aucs.items()]).to_markdown(index=False) + "\n")

    # figure: transfer + impulse response + say vs do
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.05, 1, 1.1]})
    ax = axes[0]
    lim = (-0.15, 1.0)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--", zorder=0)
    for c in conds:
        if c not in pred:
            continue
        gg = df[df.cond == c]
        ax.errorbar(pred[c], gg.delta.mean(), yerr=1.96 * gg.delta.std() / np.sqrt(len(gg)) if len(gg) > 1 else 0,
                    fmt="o", color=LLM, ms=6, capsize=0, lw=1, mec="white", mew=0.8)
        off = {"full": (6, 6), "coarse": (-8, -12), "noisy": (7, -3), "delay": (7, -3), "blind": (7, -3)}[c]
        ax.annotate(SHORT[c], (pred[c], gg.delta.mean()), xytext=off, textcoords="offset points",
                    fontsize=7.5, color=INK2, ha="right" if off[0] < 0 else "left")
    ax.set_xlim(*lim)
    ax.set_ylim(-0.15, 1.0)
    ax.text(0.98, 0.84, "y = x", color=MUTED, fontsize=7, ha="right")
    ax.set_xlabel("collusion predicted by Q-learning study (Δ)")
    ax.set_ylabel("collusion measured in LLM agents (Δ)")
    ax.set_title("A   Do the levers transfer?")

    ax = axes[1]
    for c, col in [("full", LLM), ("blind", MUTED), ("delay", BLUE)]:
        if c in imp:
            m = imp[c].mean(0)
            se = imp[c].std(0) / np.sqrt(len(imp[c]))
            x = np.arange(len(m))
            ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.15, lw=0)
            ax.plot(x, m, color=col, marker="o", ms=3.5, label=NAMES[c])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("periods after firm 0's forced price cut")
    ax.set_ylabel("rival price: deviation − no deviation\n(units of p_M − p_N)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1, handlelength=1.5)
    ax.set_title("B   Does the rival retaliate?")

    ax = axes[2]
    if aucs:
        names = list(aucs)
        vals = [aucs[k][0] for k in names]
        cols = [BLUE if k.startswith("D") else (ORANGE if k.startswith("S") else INK2) for k in names]
        y = np.arange(len(names))
        ax.barh(y, vals, color=cols, height=0.62)
        for yi, k in zip(y, names):
            v, lo, hi = aucs[k]
            ax.plot([lo, hi], [yi, yi], color=INK2, lw=0.9)
            ax.text(1.04, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xlim(0, 1.13)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("ROC AUC, exact-price vs. blind agents")
        ax.grid(axis="y", visible=False)
    ax.set_title("C   Behaviour (blue) vs. notes (orange)")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(f"{fig_prefix}_main.png")
    plt.close(fig)

    # example price paths
    runs = [json.loads(l) for l in open(path)]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.1 * len(conds), 2.1), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), conds):
        r = next(r for r in runs if r["cond"] == c and r["seed"] == 0 and r["branch"] == "base")
        p = np.array([h["p"] for h in r["hist"]])
        ax.plot(p[:, 0], color=RL, lw=1.2)
        ax.plot(p[:, 1], color=LLM, lw=1.2)
        ax.axhline(mk.p_nash, color=MUTED, lw=0.7, ls="--")
        ax.axhline(mk.p_mon, color=MUTED, lw=0.7, ls=":")
        ax.set_title(NAMES[c], fontsize=8)
        ax.set_xlabel("period", fontsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("price")
    fig.tight_layout()
    fig.savefig(f"{fig_prefix}_paths.png")
    plt.close(fig)

    df.to_csv(out_md.replace(".md", f"_{tag}.csv"), index=False)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    src = os.path.join(R, "llm_fake" if a.fake else "llm")
    files = sorted(glob.glob(os.path.join(src, "*", "runs.jsonl")))
    if not files:
        raise SystemExit(f"no runs in {src}/*/runs.jsonl - run modal_app.py (or experiments.run_llm) first")
    mk, pred = Market(), rl_predictions()
    prefix = "llm_fake" if a.fake else "llm"
    out_md = os.path.join(R, f"{prefix}_summary.md")
    lines = ["# LLM-agent results" + (" - PIPELINE TEST WITH SCRIPTED BOTS, NOT RESULTS" if a.fake else ""),
             "\nAuto-generated by experiments/analyze_llm.py.\n"]
    for f in files:
        tag = os.path.basename(os.path.dirname(f))
        lines += analyse(tag, f, out_md, os.path.join(R, "figures", f"{prefix}_{tag}"), mk, pred)
    open(out_md, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
