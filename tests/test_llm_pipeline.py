import json
from argparse import Namespace

from collusion_lab.llm_agents import run


def test_fake_backend_runs_and_punishes_probe(tmp_path):
    out = tmp_path / "run.jsonl"
    run(Namespace(backend="fake", model="x", effort="low", n=2, periods=30, probe_at=20,
                  seed=0, out=str(out), obs="full", comm=False, verbose=False))
    recs = [json.loads(l) for l in open(out)]
    assert len(recs) == 30 and recs[20]["forced"]
    # the scripted trigger bot cuts its price right after the forced deviation
    assert recs[21]["p"][1] < recs[19]["p"][1] - 0.2
