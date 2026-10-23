"""One learning session: train agents, then measure outcomes and run audits."""
from dataclasses import dataclass, asdict, field
from functools import lru_cache

import numpy as np

from .econ import Market
from .qlearning import init_q, play, train


@dataclass
class Config:
    n: int = 2
    m: int = 15
    alpha: float = 0.15
    beta: float = 1e-5
    gamma: float = 0.95
    blind: bool = False          # agents cannot see rival prices
    eps: float = 0.0             # observation noise on rival prices
    coarsen: int = 15            # buckets for observed rival prices (m = none)
    delay: int = 0               # periods of delay on observed rival prices
    jitter: float = 0.0          # mandated price randomisation probability
    T_max: int = 3_000_000
    conv_window: int = 100_000
    label: str = "baseline"

    def arrays(self):
        n = self.n
        return dict(blind=np.full(n, self.blind), eps=np.full(n, float(self.eps)))


@lru_cache(maxsize=8)
def tables(n, m):
    mk = Market(n=n, m=m)
    return mk, mk.profit_table(), mk.cs_table()


def joint_index(path, m):
    k = np.zeros(path.shape[0], dtype=np.int64)
    for j in range(path.shape[1]):
        k = k * m + path[:, j]
    return k


def outcomes(cfg, PI, CS, mk, path):
    k = joint_index(path, cfg.m)
    pi = PI[k].mean(0)
    return dict(
        delta=float(mk.delta(pi).mean()),
        price=float(mk.grid[path].mean()),
        cs=float(CS[k].mean()),
        welfare=float(CS[k].mean() + PI[k].sum(1).mean()),
    )


def probe(cfg, PI, mk, Q, hist, t_end, arrs, reps, seed, horizon=30, at=10):
    """Active audit (Calvano-style impulse response): force firm 0 to play its static
    best response for one period, compare to the identical-seed counterfactual."""
    resp1, resp5, gain = [], [], []
    m, n = cfg.m, cfg.n
    disc = cfg.gamma ** np.arange(horizon - at)
    unit = mk.p_mon - mk.p_nash
    for r in range(reps):
        sd = seed * 1000 + r
        args = (PI, Q, hist, t_end, n, m, arrs["blind"], arrs["eps"], cfg.coarsen,
                cfg.delay, cfg.jitter, 0.0, horizon)
        base = play(*args, -1, 0, sd)
        dev = play(*args, at, 0, sd)
        pb, pd = mk.grid[base], mk.grid[dev]
        resp1.append((pd[at + 1, 1:] - pb[at + 1, 1:]).mean() / unit)
        resp5.append((pd[at + 1:at + 6, 1:] - pb[at + 1:at + 6, 1:]).mean() / unit)
        g = PI[joint_index(dev[at:], m), 0] - PI[joint_index(base[at:], m), 0]
        gain.append((disc * g).sum() / (mk.pi_mon - mk.pi_nash))
    # "deviation size": how far the forced move was from what firm 0 would have done
    return dict(probe_resp1=float(np.mean(resp1)), probe_resp5=float(np.mean(resp5)),
                probe_gain=float(np.mean(gain)))


def passive(cfg, PI, mk, Q, hist, t_end, arrs, seed, explore=0.05, T=20_000):
    """Observational audit on a market with residual exploration noise."""
    path = play(PI, Q, hist, t_end, cfg.n, cfg.m, arrs["blind"], arrs["eps"],
                cfg.coarsen, cfg.delay, cfg.jitter, explore, T, -1, 0, seed + 7)
    p = mk.grid[path]
    dp = np.diff(p, axis=0)
    level_corr, lead_lag = [], []
    for i in range(cfg.n):
        for j in range(cfg.n):
            if i == j:
                continue
            if p[:, i].std() > 0 and p[:, j].std() > 0:
                level_corr.append(np.corrcoef(p[:, i], p[:, j])[0, 1])
            x, y = dp[:-1, i], dp[1:, j]
            if x.std() > 0 and y.std() > 0:
                lead_lag.append(np.corrcoef(x, y)[0, 1])
    # conditional response: mean change of j after i cut its price
    cut = dp[:-1, 0] < 0
    react = float((dp[1:, 1:].mean(1)[cut]).mean() / (mk.p_mon - mk.p_nash)) if cut.any() else 0.0
    return dict(passive_level_corr=float(np.mean(level_corr)) if level_corr else 0.0,
                passive_leadlag=float(np.mean(lead_lag)) if lead_lag else 0.0,
                passive_react=react)


def run_session(cfg: Config, seed: int, audits=True, return_agents=False):
    mk, PI, CS = tables(cfg.n, cfg.m)
    n, m = cfg.n, cfg.m
    S = m ** n
    gamma = np.full(n, cfg.gamma)
    Q = init_q(PI, n, m, gamma, S)
    H = cfg.delay + 2
    rng = np.random.default_rng(seed)
    hist = rng.integers(0, m, size=(H, n)).astype(np.int64)
    arrs = cfg.arrays()
    t, conv = train(PI, Q, hist, n, m, cfg.alpha, gamma, cfg.beta, np.zeros(n),
                    np.ones(n, dtype=np.bool_), arrs["blind"], arrs["eps"], cfg.coarsen,
                    cfg.delay, cfg.jitter, cfg.T_max, cfg.conv_window, seed)
    stochastic = cfg.eps > 0 or cfg.jitter > 0
    T_eval = 5000 if stochastic else 500
    path = play(PI, Q, hist, t, n, m, arrs["blind"], arrs["eps"], cfg.coarsen, cfg.delay,
                cfg.jitter, 0.0, T_eval, -1, 0, seed + 1)
    burn = 100
    res = dict(asdict(cfg), seed=seed, t_conv=int(t), converged=bool(conv))
    res.update(outcomes(cfg, PI, CS, mk, path[burn:]))
    if audits:
        res.update(probe(cfg, PI, mk, Q, hist, t, arrs, 30 if stochastic else 1, seed))
        res.update(passive(cfg, PI, mk, Q, hist, t, arrs, seed))
    if return_agents:
        return res, Q, hist, t
    return res
