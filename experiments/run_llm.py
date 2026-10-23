"""Run the LLM-agent experiment grid. Resumable: finished runs are skipped.

    export ANTHROPIC_API_KEY=...
    python -m experiments.run_llm                       # 4 conditions x 5 seeds x 100 periods
    python -m experiments.run_llm --seeds 3 --periods 60
    python -m experiments.run_llm --backend fake        # pipeline test, no API key; writes results/llm_fake/
"""
import argparse
import json
import os
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

from collusion_lab.llm_agents import run

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")

# condition name -> llm_agents options
CONDITIONS = {
    "full": dict(obs="full", comm=False),      # baseline (Fish et al. style)
    "noisy": dict(obs="noisy", comm=False),    # observation-noise intervention
    "own": dict(obs="own", comm=False),        # rival-blind intervention
    "comm": dict(obs="full", comm=True),       # overt message channel (Track 3 carrier)
}


def done(path, periods):
    try:
        with open(path) as f:
            return sum(1 for _ in f) >= periods
    except FileNotFoundError:
        return False


def job(spec):
    name, seed, a = spec
    out = os.path.join(a.out_dir, f"{name}_s{seed}.jsonl")
    if done(out, a.periods):
        return f"skip {out}"
    args = Namespace(backend=a.backend, model=a.model, effort=a.effort, n=2, periods=a.periods,
                     probe_at=a.probe_at, seed=seed, out=out, verbose=False, **CONDITIONS[name])
    run(args)
    return f"done {out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "fake"], default="anthropic")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--probe-at", type=int, default=None, help="default: 80%% of periods")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.probe_at is None:
        a.probe_at = int(0.8 * a.periods)
    a.out_dir = os.path.join(ROOT, "llm" if a.backend == "anthropic" else "llm_fake")
    os.makedirs(a.out_dir, exist_ok=True)
    specs = [(c, s, a) for c in a.conditions for s in range(a.seeds)]
    print(f"{len(specs)} runs x {a.periods} periods x 2 firms = {len(specs) * a.periods * 2} model calls "
          f"(backend={a.backend}, model={a.model}) -> {a.out_dir}", flush=True)
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(a).items()}, f, indent=2)
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(job, specs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
