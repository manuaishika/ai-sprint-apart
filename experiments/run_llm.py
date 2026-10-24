"""Run the LLM-agent grid locally (same output format as modal_app.py).

    python -m experiments.run_llm --engine fake                 # pipeline test, no API key
    python -m experiments.run_llm --engine anthropic --seeds 4  # Claude, needs ANTHROPIC_API_KEY

For open-weight models on a GPU use modal_app.py instead.
"""
import argparse
import json
import os

from collusion_lab.llm_market import AnthropicEngine, FakeEngine, simulate

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fake", "anthropic"], default="fake")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--periods", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=10)
    a = ap.parse_args()
    engine = FakeEngine() if a.engine == "fake" else AnthropicEngine(model=a.model)
    tag = "fake" if a.engine == "fake" else a.model
    out = os.path.join(ROOT, "llm_fake" if a.engine == "fake" else "llm", tag)
    os.makedirs(out, exist_ok=True)
    runs = simulate(engine, seeds=a.seeds, periods=a.periods, horizon=a.horizon)
    with open(os.path.join(out, "runs.jsonl"), "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(model=tag, seeds=a.seeds, periods=a.periods, horizon=a.horizon,
                       engine=a.engine), f, indent=2)
    print(f"wrote {len(runs)} runs to {out}")


if __name__ == "__main__":
    main()
