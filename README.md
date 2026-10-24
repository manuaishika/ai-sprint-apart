# Same Levers, Different Agents

**Do anti-collusion interventions and audits transfer from reinforcement-learning to LLM pricing agents?**
Apart Research AI Collusion Research Sprint, 23–25 October 2026.

![setup](results/figures/setup.png)

Two pricing agents compete in a repeated market. We control what each agent can see of its rival's price (exact, rounded to higher/lower, noisy, delayed, hidden, or exact plus messages), and we ask three questions:

1. **Which restriction breaks collusion?** (sprint questions 1.5 and 3.2)
2. **Does collusion survive swapping an agent?** (question 1.2)
3. **What evidence tells collusion from coincidence?** Here we compare behaviour (prices, reactions, a forced-deviation probe) with what LLM agents *write* in their private notes (questions 2.2 and 2.3).

The **Q-learning study** (Calvano et al., 2020 setting) is complete: about 3,600 sessions on a CPU. Its results become predictions that the **LLM study** tests. The LLM study runs open-weight models on Modal GPUs.

![Q-learning results](results/figures/rl_main.png)

## Run the LLM study on Modal (you, about 45 minutes, about $1–2 of sprint credit)

```bash
git clone https://github.com/manuaishika/ai-sprint-apart && cd ai-sprint-apart
pip install modal numpy scipy pandas matplotlib tabulate
modal setup                                         # log in once in the browser

modal run modal_app.py --seeds 2 --periods 20       # 3-min smoke test: check it finishes
modal run modal_app.py                              # full run: Qwen2.5-7B, 6 conditions x 10 seeds
python -m experiments.analyze_llm                   # tables + figures

git add results/ && git commit -m "Add LLM results" && git push
```

- **Why the smoke test first:** the first run also builds the GPU image (about 5 minutes) and downloads the model (about 15 GB). Both are cached for later runs.
- **Optional second model:** `modal run modal_app.py --model Qwen/Qwen2.5-14B-Instruct` runs on an L40S.
- **No Modal?** `python -m experiments.run_llm --engine anthropic` runs the same grid with Claude, if you have an `ANTHROPIC_API_KEY`.

## Layout

```
collusion_lab/
  econ.py          logit-demand Bertrand market (Calvano et al. 2020), Nash/monopoly benchmarks, collusion index Δ
  qlearning.py     numba Q-learning kernels with observation/action interventions and forced deviations
  session.py       one Q-learning session: train → outcomes → active probe → passive audit
  llm_market.py    LLM agents in the same market, batched across runs; counterfactual fork at the probe;
                   LLM-judge of agents' notes; engines: vLLM (via modal_app.py), Anthropic, Fake (tests)
modal_app.py       runs the LLM grid on a Modal GPU with vLLM
experiments/
  run_all.py       all Q-learning experiments → results/*.csv           (about 15 min, 4 CPU cores)
  analyze.py       Q-learning tables (results/summary.md) and figures
  run_llm.py       LLM grid locally (fake bots for testing, or Claude)
  analyze_llm.py   LLM tables (results/llm_summary.md) and figures, compared with Q-learning predictions
  schematic.py     Figure 1
  style.py         shared figure style
report/            the submission (Apart template): report.md, report.pdf, report.docx
tests/             market benchmarks, determinism, prompt and parse checks, counterfactual fork
```

## Reproduce the Q-learning study

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m experiments.run_all && python -m experiments.analyze
```
