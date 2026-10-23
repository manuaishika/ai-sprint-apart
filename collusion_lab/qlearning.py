"""Numba kernels: Q-learning pricing agents with configurable market interventions.

Agents follow Calvano, Calzolari, Denicolo & Pastorello (2020): tabular Q-learning,
one-period memory (state = last period's price profile), epsilon_t = exp(-beta t).

Interventions act on what agents *observe* or *do*:
  blind[i]   agent i sees only its own last price (no rival information)
  eps[i]     each rival price agent i observes is replaced by a uniform random
             grid price with probability eps (observation noise)
  coarsen    rival prices are observed only up to `coarsen` buckets
  delay      rival prices are observed `delay` periods late
  jitter     with probability `jitter` each executed price is replaced by a
             uniform random grid price (mandated price randomisation)

Every step draws the same number of random numbers whatever the branch taken, so
two runs with the same seed share common random numbers (used by the probe).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _observe(true_idx, eps, coarsen, m, u, r):
    o = true_idx
    if u < eps:
        o = r
    if coarsen < m:
        o = (o * coarsen) // m
    return o


@njit(cache=True)
def _states(hist, t, H, n, m, blind, eps, coarsen, delay, out):
    """State of every agent at time t (uses prices at t-1, rivals at t-1-delay)."""
    own_slot = (t - 1 + 4 * H) % H
    riv_slot = (t - 1 - delay + 4 * H) % H
    for i in range(n):
        s = 0
        for j in range(n):
            u = np.random.random()
            r = np.random.randint(m)
            if j == i:
                v = hist[own_slot, j]
            elif blind[i]:
                v = 0
            else:
                v = _observe(hist[riv_slot, j], eps[i], coarsen, m, u, r)
            s = s * m + v
        out[i] = s


@njit(cache=True)
def _argmax(row):
    best = 0
    bv = row[0]
    for a in range(1, row.shape[0]):
        if row[a] > bv:
            bv = row[a]
            best = a
    return best


@njit(cache=True)
def _joint(a, n, m):
    k = 0
    for j in range(n):
        k = k * m + a[j]
    return k


@njit(cache=True)
def train(PI, Q, hist, n, m, alpha, gamma, beta, t0, learn, blind, eps, coarsen,
          delay, jitter, T_max, conv_window, seed):
    """Run Q-learning in place on Q (n, S, m) and hist (H, n). Returns (t, converged)."""
    np.random.seed(seed)
    H = hist.shape[0]
    S = Q.shape[1]
    greedy = np.empty((n, S), dtype=np.int64)
    for i in range(n):
        for s in range(S):
            greedy[i, s] = _argmax(Q[i, s])
    s_cur = np.empty(n, dtype=np.int64)
    s_nxt = np.empty(n, dtype=np.int64)
    a = np.empty(n, dtype=np.int64)
    _states(hist, 0, H, n, m, blind, eps, coarsen, delay, s_cur)
    stable = 0
    t = 0
    while t < T_max:
        for i in range(n):
            u1 = np.random.random()
            r1 = np.random.randint(m)
            u2 = np.random.random()
            r2 = np.random.randint(m)
            if u1 < np.exp(-beta * (t + t0[i])):
                a[i] = r1
            else:
                a[i] = greedy[i, s_cur[i]]
            if u2 < jitter:
                a[i] = r2
        for i in range(n):
            hist[t % H, i] = a[i]
        k = _joint(a, n, m)
        _states(hist, t + 1, H, n, m, blind, eps, coarsen, delay, s_nxt)
        changed = False
        for i in range(n):
            if not learn[i]:
                continue
            s = s_cur[i]
            target = PI[k, i] + gamma[i] * Q[i, s_nxt[i], greedy[i, s_nxt[i]]]
            Q[i, s, a[i]] += alpha * (target - Q[i, s, a[i]])
            g = _argmax(Q[i, s])
            if g != greedy[i, s]:
                greedy[i, s] = g
                changed = True
        if changed:
            stable = 0
        else:
            stable += 1
        for i in range(n):
            s_cur[i] = s_nxt[i]
        t += 1
        if stable >= conv_window:
            return t, True
    return t, False


@njit(cache=True)
def play(PI, Q, hist0, t_start, n, m, blind, eps, coarsen, delay, jitter, explore,
         T, force_t, force_agent, seed):
    """Greedy play with no learning. Optionally at step force_t, `force_agent` plays
    its static best response to rivals' previous prices. Returns price index path (T, n)."""
    np.random.seed(seed)
    H = hist0.shape[0]
    hist = hist0.copy()
    path = np.empty((T, n), dtype=np.int64)
    s = np.empty(n, dtype=np.int64)
    a = np.empty(n, dtype=np.int64)
    tmp = np.empty(n, dtype=np.int64)
    for step in range(T):
        t = t_start + step
        _states(hist, t, H, n, m, blind, eps, coarsen, delay, s)
        for i in range(n):
            u1 = np.random.random()
            r1 = np.random.randint(m)
            u2 = np.random.random()
            r2 = np.random.randint(m)
            if u1 < explore:
                a[i] = r1
            else:
                a[i] = _argmax(Q[i, s[i]])
            if u2 < jitter:
                a[i] = r2
        if step == force_t:
            prev = (t - 1 + 4 * H) % H
            for j in range(n):
                tmp[j] = hist[prev, j]
            best_v = -1e9
            best_a = 0
            for c in range(m):
                tmp[force_agent] = c
                v = PI[_joint(tmp, n, m), force_agent]
                if v > best_v:
                    best_v = v
                    best_a = c
            a[force_agent] = best_a
        for i in range(n):
            hist[t % H, i] = a[i]
            path[step, i] = a[i]
    return path


def init_q(PI, n, m, gamma, S):
    """Calvano initialisation: discounted payoff against uniformly random rivals."""
    Q = np.empty((n, S, m))
    pis = PI.reshape((m,) * n + (n,))
    for i in range(n):
        mean_pi = np.moveaxis(pis[..., i], i, 0).reshape(m, -1).mean(1)
        Q[i, :, :] = mean_pi[None, :] / (1.0 - gamma[i])
    return Q
