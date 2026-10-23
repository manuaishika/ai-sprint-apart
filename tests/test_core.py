import numpy as np

from collusion_lab.econ import Market
from collusion_lab.session import Config, run_session


def test_calvano_benchmarks():
    mk = Market()
    assert abs(mk.p_nash - 1.4729) < 1e-3
    assert abs(mk.p_mon - 1.9250) < 1e-3
    assert abs(mk.delta(mk.pi_nash)) < 1e-12 and abs(mk.delta(mk.pi_mon) - 1) < 1e-12


def test_nash_is_a_best_response():
    mk = Market()
    xs = np.linspace(1.0, 3.0, 4001)
    br = xs[np.argmax([mk.profit(np.array([x, mk.p_nash]))[0] for x in xs])]
    assert abs(br - mk.p_nash) < 1e-3


def test_profit_table_order():
    mk = Market()
    PI = mk.profit_table()
    i, j = 3, 11
    assert np.allclose(PI[i * mk.m + j], mk.profit(mk.grid[[i, j]]))


def test_session_is_deterministic_and_blind_agents_do_not_punish():
    r1 = run_session(Config(T_max=300_000), 1)
    r2 = run_session(Config(T_max=300_000), 1)
    assert r1["delta"] == r2["delta"]
    rb = run_session(Config(blind=True), 3)
    assert rb["probe_resp1"] == 0.0
