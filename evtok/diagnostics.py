"""Structural headroom: could graph neighbourhoods of retrieved seeds reach the gold functions?

Usage: python -m evtok.diagnostics --data <DATA_DIR> --out results/tier_a
For each task with gold nodes: degree of gold nodes, hop distance from BM25 top-k seeds to the
nearest gold node, and whether BM25 top-k at 8k already hits gold.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict, deque

import numpy as np

from . import selectors as S
from .data import graph_key, load_graph, load_tasks
from .gold import gold_nodes
from .retrieval import hybrid_score, task_signals


def hop(g, seeds, targets, hmax=4):
    dist = {s: 0 for s in seeds}
    dq = deque(seeds)
    while dq:
        v = dq.popleft()
        if v in targets:
            return dist[v]
        if dist[v] >= hmax:
            continue
        for n in g.nbrs[v]:
            n = int(n)
            if n not in dist:
                dist[n] = dist[v] + 1
                dq.append(n)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/tier_a")
    a = ap.parse_args()
    tasks = load_tasks(a.data)
    groups = defaultdict(list)
    for t in tasks:
        groups[graph_key(t)].append(t)
    rows = []
    for k in sorted(groups):
        g = load_graph(a.data, k)
        deg = np.array([len(n) for n in g.nbrs])
        for t in groups[k]:
            gd = gold_nodes(g, t["patch"])
            G = gd["nodes"]
            if not G:
                continue
            s = hybrid_score(g, task_signals(g, t["problem_statement"]), 0.0, 0.0)
            order = np.argsort(-s, kind="stable")
            ff = S.flat_full(g, order, 8000)
            hit8 = bool(set(ff) & G)
            rank = min(int(np.where(order == v)[0][0]) for v in G)
            rows.append({
                "instance_id": t["instance_id"], "repo": t["repo"], "n_gold": len(G),
                "gold_min_degree": int(min(deg[v] for v in G)), "gold_isolated_frac": float(np.mean([deg[v] == 0 for v in G])),
                "graph_isolated_frac": float(np.mean(deg[g.pool] == 0)),
                "best_gold_rank_bm25": rank, "bm25_hit8k": hit8,
                "hop_top1": hop(g, [int(order[0])], G), "hop_top10": hop(g, [int(v) for v in order[:10]], G),
                "hop_selected8k": hop(g, list(ff), G),
            })
    with open(os.path.join(a.out, "diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    miss = [r for r in rows if not r["bm25_hit8k"]]
    fmt = lambda c: dict(sorted(Counter("none(>4)" if x is None else x for x in c).items(), key=lambda kv: str(kv[0])))
    print(f"tasks with gold: {n}; BM25 misses at 8k: {len(miss)}")
    print("share of pool nodes with no edges (mean over tasks):", round(np.mean([r['graph_isolated_frac'] for r in rows]), 3))
    print("tasks where every gold node is isolated:", sum(r["gold_isolated_frac"] == 1 for r in rows))
    print("hop from BM25 top-1 to nearest gold (all):", fmt(r["hop_top1"] for r in rows))
    print("hop from BM25 top-10 to nearest gold (all):", fmt(r["hop_top10"] for r in rows))
    print("BM25 8k misses — hop from the selected 8k packet to nearest gold:", fmt(r["hop_selected8k"] for r in miss))
    print("BM25 8k misses — every gold node isolated:", sum(r["gold_isolated_frac"] == 1 for r in miss))


if __name__ == "__main__":
    main()
