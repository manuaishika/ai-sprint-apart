# Cheap Levers, Real Tests: Breaking and Auditing Algorithmic Collusion

Submission for the **Apart Research AI Collusion Research Sprint** (23–25 Oct 2026; Apart Research × Poseidon Research × AE Studio).

**Write-up:** [`report/report.md`](report/report.md) · **Numbers:** [`results/summary.md`](results/summary.md) · **Figures:** [`results/figures/`](results/figures)

## What this project asks

In a standard repeated pricing game (Calvano et al., 2020) where Q-learning agents learn to price above competitive levels:

| Sprint question | What we measure |
|---|---|
| **1.5 / 3.2**: which single restriction removes the most collusion at the lowest cost? | Dose–response curves for five interventions: rival-price observation noise, coarsening, delay, mandated price jitter, and full rival blindness. We also add a third firm (entry). Costs are reported in bits of information withheld and in consumer and total welfare. |
| **1.2**: does collusion survive swapping a participant? | Cross-play between agents from different sessions, a fresh learner entering against an incumbent, and a myopic entrant. |
| **2.3 / 1.3**: what counts as convincing evidence of collusion, as opposed to coincidence? | Five detectors are compared by ROC AUC against control agents that also reach supracompetitive prices but *cannot* sustain collusion (myopic, short-horizon, rival-blind). Three detectors are passive: price level, parallel pricing, and lead–lag reaction. The other two are an active probe: a forced deviation, then measuring the rival's punishment and whether the deviation paid off. |

An LLM-agent version of the same market and probe is in `collusion_lab/llm_agents.py`. It follows Fish et al. (2024) and adds optional free-text messaging between agents and observation interventions (see *LLM extension* below).

## Layout

```
collusion_lab/
  econ.py         logit-demand Bertrand market, Nash/monopoly benchmarks, collusion index Δ
  qlearning.py    numba kernels: Q-learning with observation/action interventions; greedy play + forced deviations
  session.py      one session = train → outcomes → active probe → passive audit
  llm_agents.py   Claude-based pricing agents in the same market (needs an API key)
experiments/
  run_all.py      all Q-learning experiments → results/*.csv
  analyze.py      tables (results/summary.md) and figures (results/figures/*.png)
report/report.md  the write-up
tests/            sanity tests (benchmarks, table layout, determinism, blind agents can't punish)
```

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m experiments.run_all      # about 15 min on 4 CPU cores; no GPU or API key needed
python -m experiments.analyze
```

`python -m experiments.run_all --quick` is a smoke test that takes about 30 seconds.

## LLM extension (not run here)

```bash
export ANTHROPIC_API_KEY=...
python -m collusion_lab.llm_agents --periods 100 --probe-at 80 --out results/llm/full_s0.jsonl
python -m collusion_lab.llm_agents --periods 100 --probe-at 80 --obs noisy --out results/llm/noisy_s0.jsonl
python -m collusion_lab.llm_agents --periods 100 --comm --out results/llm/comm_s0.jsonl
```

The sandbox used to build this project had no API key, so **no LLM results are reported**. The harness is provided so the same interventions and probe can be run on Claude agents, for example on Modal or locally. At 2 firms × 100 periods, one run makes 200 calls.

## Sprint resources this builds on

- Calvano, Calzolari, Denicolò & Pastorello (2020), *Artificial Intelligence, Algorithmic Pricing, and Collusion*, AER. This project uses its environment, parameters and impulse-response test.
- Fish, Gonczarowski & Shorrer (2024), *Algorithmic Collusion by Large Language Models*, arXiv:2404.00806. This is the basis for the LLM harness.
- Motwani et al. (2024), *Secret Collusion among AI Agents: Multi-Agent Deception via Steganography*, arXiv:2402.07510.
- *Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs*, arXiv:2510.04303.
- *On the Fragility of AI Agent Collusion*, arXiv:2603.20281.
- *Evaluating LLM Agent Collusion in Double Auctions*, arXiv:2507.01413.
- Hammond et al. (2025), *Multi-Agent Risks from Advanced AI*, arXiv:2502.14143.
- *Algorithmic collusion under asynchronous price updating*, arXiv:2608.01406.
