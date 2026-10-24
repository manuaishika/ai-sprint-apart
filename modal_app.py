"""Run the LLM-agent collusion experiments on Modal GPUs with an open-weight model (vLLM).

One-time setup on your own computer:
    pip install modal
    modal setup                      # opens a browser to log in; sprint credits apply

Run (from the repository root):
    modal run modal_app.py                                   # Qwen2.5-7B-Instruct on an L4, ~40 min
    modal run modal_app.py --seeds 2 --periods 20            # 3-minute smoke test first (recommended)
    modal run modal_app.py --model Qwen/Qwen2.5-14B-Instruct # bigger model, runs on an L40S

Results are written to results/llm/<model>/runs.jsonl on your computer. Then:
    python -m experiments.analyze_llm
"""
import json
import os
import time

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    # pinned together: vLLM 0.6.6 predates transformers 5 and numpy 2
    .pip_install("vllm==0.6.6.post1", "transformers==4.47.1", "scipy==1.14.1", "hf_transfer==0.1.8")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("collusion_lab")
)
app = modal.App("collusion-llm-agents", image=image)
hf_cache = modal.Volume.from_name("collusion-hf-cache", create_if_missing=True)
SMALL, LARGE = "L4", "L40S"


def _grid(model, seeds, periods, horizon, temperature, max_tokens):
    from vllm import LLM, SamplingParams

    from collusion_lab.llm_market import simulate

    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.90, seed=0)
    tok = llm.get_tokenizer()

    def engine(chats, seeds_):
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        params = [SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=int(s)) for s in seeds_]
        return [o.outputs[0].text for o in llm.generate(prompts, params, use_tqdm=False)]

    t0 = time.time()
    runs = simulate(engine, seeds=seeds, periods=periods, horizon=horizon,
                    log=lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True))
    for r in runs:  # keep the payload small
        for h in r["hist"]:
            h["raw"] = [x[:1500] for x in h["raw"]]
    return runs


@app.function(gpu=SMALL, timeout=4 * 3600, volumes={"/root/.cache/huggingface": hf_cache})
def run_small(**kw):
    return _grid(**kw)


@app.function(gpu=LARGE, timeout=4 * 3600, volumes={"/root/.cache/huggingface": hf_cache})
def run_large(**kw):
    return _grid(**kw)


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", seeds: int = 10, periods: int = 60,
         horizon: int = 10, temperature: float = 0.7, max_tokens: int = 400):
    big = any(s in model for s in ("14B", "32B", "13B"))
    fn = run_large if big else run_small
    print(f"Running {model} on {LARGE if big else SMALL}: 6 conditions x {seeds} seeds, "
          f"{periods} periods + {horizon} after the probe")
    runs = fn.remote(model=model, seeds=seeds, periods=periods, horizon=horizon,
                     temperature=temperature, max_tokens=max_tokens)
    out = os.path.join("results", "llm", model.split("/")[-1])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "runs.jsonl"), "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(model=model, seeds=seeds, periods=periods, horizon=horizon,
                       temperature=temperature, max_tokens=max_tokens, engine="vllm==0.6.6.post1"), f, indent=2)
    print(f"Wrote {len(runs)} runs to {out}. Next: python -m experiments.analyze_llm")
