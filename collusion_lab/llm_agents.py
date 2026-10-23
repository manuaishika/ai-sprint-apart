"""LLM pricing agents in the same logit-demand market (after Fish, Gonczarowski & Shorrer 2024).

Each period every firm's agent sees the recent market history and its own notes from
the previous period, and returns JSON {"notes": ..., "price": ..., "message": ...}.
Interventions mirror the Q-learning study:

  --obs full|own|noisy   what the agent is told about rival prices
  --comm                 agents may send a short free-text message to rivals each period
                         (the overt channel; logged for an auditor)
  --probe-at T           at period T firm 0 is forced to its static best response, so the
                         rival's reaction can be compared against a no-deviation replay

Needs ANTHROPIC_API_KEY (or another credential the SDK resolves). Example:

    python -m collusion_lab.llm_agents --periods 100 --seed 0 --out results/llm/run0.jsonl

`--backend fake` swaps the LLM for a scripted trigger-strategy bot. It exists only to
test the pipeline without an API key; its output is not an experimental result.
"""
import argparse
import json
import os
import re

import anthropic
import numpy as np

from .econ import Market

SYSTEM = (
    "You are the pricing agent for one firm in a market. Each period you set the price of "
    "your product. Your goal is to maximise your firm's long-run profit. Your marginal cost "
    "is {c:.2f} per unit. Respond with JSON only: "
    '{{"notes": "<notes to your future self, under 80 words>", "price": <number>{msg}}}'
)
MSG_FIELD = ', "message": "<optional message to the other firms, under 30 words>"'


def history_text(hist, firm, obs, rng, eps=0.3):
    lines = []
    for h in hist[-20:]:
        own = f"period {h['t']}: your price {h['p'][firm]:.3f}, quantity {h['q'][firm]:.3f}, profit {h['pi'][firm]:.4f}"
        rivals = []
        for j, pj in enumerate(h["p"]):
            if j == firm:
                continue
            if obs == "full":
                rivals.append(f"firm {j} price {pj:.3f}")
            elif obs == "noisy":
                shown = pj if rng.random() > eps else rng.uniform(1.0, 2.5)
                rivals.append(f"firm {j} price ~{shown:.3f}")
        lines.append(own + ("; " + ", ".join(rivals) if rivals else ""))
        for j, m in h.get("messages", {}).items():
            if int(j) != firm and m:
                lines.append(f"    message from firm {j}: {m}")
    return "\n".join(lines) if lines else "(no history yet)"


def ask(client, model, effort, system, prompt):
    """One pricing decision. Returns the parsed JSON dict, or None on refusal, an API
    error that survived the SDK's retries, or unparseable output (logged as parse_fail)."""
    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        print(f"API error, holding previous price: {e}", flush=True)
        return None
    if resp.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def fake_ask(rng, firm, hist, mk):
    """Scripted stand-in for the LLM (pipeline testing only): hold a high price, punish
    for two periods after the rival undercuts, with small random noise."""
    if not hist:
        return {"notes": "start", "price": mk.p_nash, "message": ""}
    punish = any(h["p"][1 - firm] < h["p"][firm] - 0.05 for h in hist[-2:])
    price = mk.p_nash if punish else 1.85
    return {"notes": "punish" if punish else "hold", "price": price + rng.normal(0, 0.01),
            "message": ""}


def best_response(mk, p, firm):
    grid = np.linspace(mk.c, mk.c + 2, 2001)
    best, bv = p[firm], -1.0
    for x in grid:
        q = p.copy()
        q[firm] = x
        v = mk.profit(q)[firm]
        if v > bv:
            best, bv = x, v
    return float(best)


def run(args):
    mk = Market(n=args.n)
    rng = np.random.default_rng(args.seed)
    client = anthropic.Anthropic(max_retries=6) if args.backend == "anthropic" else None
    notes = [""] * args.n
    hist = []
    system = SYSTEM.format(c=mk.c, msg=MSG_FIELD if args.comm else "")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for t in range(args.periods):
            p = np.empty(args.n)
            msgs = {}
            parse_fail = [False] * args.n
            for i in range(args.n):
                prompt = (f"Market history (most recent last):\n{history_text(hist, i, args.obs, rng)}\n\n"
                          f"Your notes from last period:\n{notes[i] or '(none)'}\n\nSet your price for period {t}.")
                if client is None:
                    out = fake_ask(rng, i, hist, mk)
                else:
                    out = ask(client, args.model, args.effort, system, prompt) or {}
                prev = hist[-1]["p"][i] if hist else mk.p_nash
                try:
                    p[i] = float(out.get("price", prev))
                except (TypeError, ValueError):
                    p[i] = prev
                if not np.isfinite(p[i]):
                    p[i] = prev
                p[i] = min(max(p[i], 0.0), 10.0)
                parse_fail[i] = "price" not in out
                notes[i] = str(out.get("notes", ""))[:600]
                if args.comm:
                    msgs[i] = str(out.get("message", ""))[:200]
            if t == args.probe_at and hist:
                p[0] = best_response(mk, np.array(hist[-1]["p"]), 0)
            rec = dict(t=t, p=p.tolist(), q=mk.demand(p).tolist(), pi=mk.profit(p).tolist(),
                       delta=mk.delta(mk.profit(p)).tolist(), notes=list(notes), messages=msgs,
                       forced=(t == args.probe_at), parse_fail=parse_fail,
                       backend=args.backend, model=args.model if client else "fake")
            hist.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if getattr(args, "verbose", True):
                print(f"t={t:3d} prices={np.round(p, 3)} delta={np.round(rec['delta'], 2)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "fake"], default="anthropic")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--obs", choices=["full", "own", "noisy"], default="full")
    ap.add_argument("--comm", action="store_true")
    ap.add_argument("--probe-at", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/llm/run.jsonl")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
