import json

import numpy as np

from collusion_lab.llm_market import FakeEngine, build_chat, parse, simulate, Run
from collusion_lab.econ import Market


def test_parse_handles_common_reply_shapes():
    assert parse('```json\n{"notes": "x", "price": 1.8}\n```')[0]["price"] == 1.8
    assert parse('I think {"price": "1.75", "notes": "y"} is right')[0]["price"] == 1.75
    assert parse("price: 1.9 because...")[0]["price"] == 1.9
    assert parse("no idea")[1] is False


def test_conditions_show_the_right_information():
    mk = Market()
    for cond, expect in [("full", "competitor's price: 1.8"), ("coarse", "higher than yours"),
                         ("blind", None), ("delay", "not yet reported")]:
        r = Run(cond, 0)
        r.hist.append(dict(t=0, p=[1.5, 1.8], q=[0, 0], pi=[0, 0], msg=["", ""], notes=["", ""], ok=[True, True],
                           seen=__import__("collusion_lab.llm_market", fromlist=["_seen"])._seen(r, 0, np.array([1.5, 1.8]), mk)))
        user = build_chat(r, 0, 1, mk)[1]["content"]
        if expect is None:
            assert "competitor" not in user
        else:
            assert expect in user


def test_simulation_is_json_serialisable_and_forks_counterfactual():
    runs = simulate(FakeEngine(), conditions=["full", "blind"], seeds=2, periods=12, horizon=4, log=None)
    json.dumps(runs)
    assert len(runs) == 8 and {r["branch"] for r in runs} == {"base", "dev"}
    base = next(r for r in runs if r["cond"] == "full" and r["branch"] == "base")
    dev = next(r for r in runs if r["cond"] == "full" and r["seed"] == base["seed"] and r["branch"] == "dev")
    assert base["hist"][11] == dev["hist"][11]          # identical before the probe
    assert dev["hist"][12]["forced"] and not base["hist"][12]["forced"]
