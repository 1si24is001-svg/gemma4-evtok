"""Context selectors under a hard token budget B.

A packet is a dict {node_index: tier} with tier in {0: outline, 1: signature+docstring, 2: full source}.
Its cost is sum(cost[v, tier]); every selector guarantees cost <= B.

Selectors
- flat_full:        nodes in score order, full source only (top-k retrieval at matched budget).
- greedy_tiers:     greedy on marginal utility per token over (node, tier-upgrade) actions.
                    expand=False -> flat multi-resolution packing (no graph).
                    expand=True  -> seed-rooted connected selection: a node becomes eligible only
                                    after one of its graph neighbours is selected (or it is a seed).
- graph_expand:     h-hop neighbourhood of seeds re-ranked with a proximity bonus, full source only.
- random_connected: random growth from the same seeds (control).
- oracle:           gold nodes first (full), then filled from another packet (diagnostic upper bound).
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from .data import RepoGraph

TIER_W = (0.1, 0.45, 1.0)


def packet_cost(g: RepoGraph, pk: dict[int, int]) -> int:
    return int(sum(g.cost[v, t] for v, t in pk.items()))


def flat_full(g: RepoGraph, order, B: int, max_scan: int = 3000) -> dict[int, int]:
    pk: dict[int, int] = {}
    used = 0
    for n, v in enumerate(order):
        if n >= max_scan or B - used < 8:
            break
        v = int(v)
        c = int(g.cost[v, 2])
        if used + c <= B:
            pk[v] = 2
            used += c
    return pk


def bfs_dist(g: RepoGraph, seeds: list[int], h: int) -> dict[int, int]:
    dist = {s: 0 for s in seeds}
    dq = deque(seeds)
    while dq:
        v = dq.popleft()
        if dist[v] >= h:
            continue
        for n in g.nbrs[v]:
            n = int(n)
            if n not in dist and g.pool[n]:
                dist[n] = dist[v] + 1
                dq.append(n)
    return dist


def utility(s: np.ndarray, dist: dict[int, int] | None, beta: float, tau: float, temp: float) -> np.ndarray:
    sp = s.astype(np.float64).copy()
    if dist and beta:
        for v, d in dist.items():
            sp[v] += beta * np.exp(-d / tau)
    smax = float(sp[sp > -1e8].max()) if np.any(sp > -1e8) else 0.0
    u = np.exp(np.clip((sp - smax) / temp, -60, 0))
    u[s <= -1e8] = 0.0
    return u


def greedy_tiers(
    g: RepoGraph,
    util: np.ndarray,
    B: int,
    candidates,
    tiers=(0, 1, 2),
    rho: float = 1.0,
    expand: bool = False,
    region: dict[int, int] | None = None,
    wt=TIER_W,
) -> dict[int, int]:
    cur: dict[int, int] = {}
    heap: list = []
    cnt = 0
    pushed: set[int] = set()

    def push(v: int):
        nonlocal cnt
        c0 = cur.get(v, -1)
        w0 = wt[c0] if c0 >= 0 else 0.0
        k0 = int(g.cost[v, c0]) if c0 >= 0 else 0
        for t in tiers:
            if t <= c0:
                continue
            gain = util[v] * (wt[t] - w0)
            dc = int(g.cost[v, t]) - k0
            if gain <= 0 or dc <= 0:
                continue
            heapq.heappush(heap, (-gain / dc**rho, cnt, v, c0, t, dc))
            cnt += 1

    for v in candidates:
        v = int(v)
        if g.pool[v] and v not in pushed:
            pushed.add(v)
            push(v)
    used = 0
    while heap and B - used >= 4:
        _, _, v, c0, t, dc = heapq.heappop(heap)
        if cur.get(v, -1) != c0 or used + dc > B:
            continue
        cur[v] = t
        used += dc
        if c0 == -1 and expand:
            for n in g.nbrs[v]:
                n = int(n)
                if n in pushed or not g.pool[n] or (region is not None and n not in region):
                    continue
                pushed.add(n)
                push(n)
        push(v)
    return cur


def graph_expand(g: RepoGraph, s: np.ndarray, seeds: list[int], h: int, beta: float, tau: float, B: int) -> dict[int, int]:
    dist = bfs_dist(g, seeds, h)
    region = np.fromiter(dist.keys(), dtype=np.int64)
    sp = s[region] + beta * np.exp(-np.fromiter(dist.values(), dtype=np.float64) / tau)
    order = region[np.argsort(-sp, kind="stable")]
    return flat_full(g, order, B)


def random_connected(g: RepoGraph, seeds: list[int], B: int, rng: np.random.Generator) -> dict[int, int]:
    pk: dict[int, int] = {}
    used = 0
    frontier = list(dict.fromkeys(seeds))
    seen = set(frontier)
    fails = 0
    while frontier and B - used >= 8 and fails < 200:
        v = frontier.pop(int(rng.integers(len(frontier))))
        c = int(g.cost[v, 2])
        if used + c > B:
            fails += 1
            continue
        pk[v] = 2
        used += c
        for n in g.nbrs[v]:
            n = int(n)
            if n not in seen and g.pool[n]:
                seen.add(n)
                frontier.append(n)
    return pk


def oracle(g: RepoGraph, gold: set[int], fill: dict[int, int], B: int) -> dict[int, int]:
    pk: dict[int, int] = {}
    used = 0
    for v in sorted(gold, key=lambda x: int(g.cost[x, 2])):
        c = int(g.cost[v, 2])
        if used + c <= B:
            pk[v] = 2
            used += c
    for v, t in fill.items():
        if v in pk:
            continue
        for tt in range(t, -1, -1):
            c = int(g.cost[v, tt])
            if used + c <= B:
                pk[v] = tt
                used += c
                break
    return pk
