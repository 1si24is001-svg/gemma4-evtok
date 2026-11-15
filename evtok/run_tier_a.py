"""Tier A: evaluate every selector configuration on every task at every budget (CPU only).

Usage:
    python -m evtok.run_tier_a --data <competition data dir> --out results/tier_a [--workers 4]

Writes <out>/grid.pkl: {instance_id: {"meta": {...}, "res": {config_key: {budget: metrics}}}}.
Hyper-parameter selection (5-fold CV) and statistics are done afterwards by evtok.analyze;
nothing here looks at gold labels except to compute metrics after a packet is built.
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import selectors as S
from .data import graph_key, load_graph, load_tasks
from .gold import gold_nodes
from .metrics import packet_metrics
from .retrieval import hybrid_score, task_signals, zscore

BUDGETS = (2000, 4000, 8000, 16000)
KEEP = ("recall", "recall_any", "hit", "hit_any", "density", "file_hit", "file_hit_any", "cost", "n_nodes", "n_full", "components")

HYBRIDS = [(0.0, 0.0, "anchor")] + [
    (wd, wa, src) for wd in (0.0, 0.5, 1.0) for wa in (0.0, 1.0, 2.0) for src in ("anchor", "bm") if not (wd == 0.0 and src == "bm") and (wd, wa) != (0.0, 0.0)
]
TIERW = {"a": (0.02, 0.2, 1.0), "b": (0.1, 0.45, 1.0)}
CONN_GRID = list(itertools.product((0.0, 1.0), (1, 2), (3, 5, 10), (0.5, 1.0), ("a", "b")))  # beta,h,K,rho,tierw
GX_GRID = list(itertools.product((0.0, 1.0), (1, 2), (3, 5, 10)))  # beta,h,K
FLAT_TIER_GRID = list(itertools.product((0.5, 1.0), ("a", "b")))
TAU, TEMP = 1.0, 1.5


def hkey(h):
    return f"wd{h[0]}_wa{h[1]}_{h[2]}"


def _m(g, pk, gold):
    m = packet_metrics(g, pk, gold)
    return {k: m[k] for k in KEEP}


def run_group(data_dir: str, key: str, tasks: list[dict], seed: int = 0) -> dict:
    g = load_graph(data_dir, key)
    out = {}
    for t in tasks:
        gold = gold_nodes(g, t["patch"])
        sig = task_signals(g, t["problem_statement"])
        res: dict[str, dict[int, dict]] = defaultdict(dict)
        # pure dense baselines
        for name, d in (("dense_anchor", sig["dense_anchor"]), ("dense_bm", sig["dense_bm"])):
            s = zscore(d.astype(np.float64), g.pool)
            order = np.argsort(-s, kind="stable")
            for B in BUDGETS:
                res[f"flat|{name}"][B] = _m(g, S.flat_full(g, order, B), gold)
        rng = np.random.default_rng(seed)
        for h in HYBRIDS:
            hk = hkey(h)
            s = hybrid_score(g, sig, *h)
            order = np.argsort(-s, kind="stable")
            top = [int(v) for v in order[:400] if s[v] > -1e8]
            seeds_by_k = {k: top[:k] for k in (3, 5, 10)}
            u_flat = S.utility(s, None, 0.0, TAU, TEMP)
            dists = {(k, hh): S.bfs_dist(g, seeds_by_k[k], hh) for k in (3, 5, 10) for hh in (1, 2)}
            utils = {(k, hh, beta): S.utility(s, dists[(k, hh)], beta, TAU, TEMP) for k in (3, 5, 10) for hh in (1, 2) for beta in (0.0, 1.0)}
            for B in BUDGETS:
                ff = S.flat_full(g, order, B)
                res[f"flat|{hk}"][B] = _m(g, ff, gold)
                for rho, tw in FLAT_TIER_GRID:
                    pk = S.greedy_tiers(g, u_flat, B, top, rho=rho, expand=False, wt=TIERW[tw])
                    res[f"flatT|{hk}|rho{rho}_{tw}"][B] = _m(g, pk, gold)
                for beta, hh, k in GX_GRID:
                    pk = S.graph_expand(g, s, seeds_by_k[k], hh, beta, TAU, B)
                    res[f"gx|{hk}|b{beta}_h{hh}_k{k}"][B] = _m(g, pk, gold)
                    pk = S.greedy_tiers(g, utils[(k, hh, beta)], B, seeds_by_k[k], tiers=(2,), rho=1.0, expand=True, region=dists[(k, hh)])
                    res[f"conn|{hk}|b{beta}_h{hh}_k{k}"][B] = _m(g, pk, gold)
                for beta, hh, k, rho, tw in CONN_GRID:
                    pk = S.greedy_tiers(g, utils[(k, hh, beta)], B, seeds_by_k[k], rho=rho, expand=True, region=dists[(k, hh)], wt=TIERW[tw])
                    res[f"connT|{hk}|b{beta}_h{hh}_k{k}_rho{rho}_{tw}"][B] = _m(g, pk, gold)
                # random connected control (3 draws, averaged later), seeds = top-5
                for r in range(3):
                    pk = S.random_connected(g, seeds_by_k[5], B, rng)
                    res[f"rand|{hk}|r{r}"][B] = _m(g, pk, gold)
                res[f"oracle|{hk}"][B] = _m(g, S.oracle(g, gold["nodes"], ff, B), gold)
        meta = {
            "repo": t["repo"],
            "graph": key,
            "n_gold": len(gold["nodes"]),
            "gold_ids": sorted(g.ids[v] for v in gold["nodes"]),
            "gold_in_pool": all(g.pool[v] for v in gold["nodes"]),
            "addition_only": gold["addition_only"],
            "n_files": len(gold["files"]),
            "n_anchor": sig["n_anchor"],
            "n_nodes_graph": len(g.ids),
            "n_pool": int(g.pool.sum()),
            "gold_cost_full": int(sum(g.cost[v, 2] for v in gold["nodes"])),
        }
        out[t["instance_id"]] = {"meta": meta, "res": dict(res)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/tier_a")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N graph groups")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = load_tasks(a.data)
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        groups[graph_key(t)].append(t)
    keys = sorted(groups)
    if a.limit:
        keys = keys[: a.limit]
    t0 = time.time()
    allres: dict = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_group, a.data, k, groups[k]): k for k in keys}
        for n, f in enumerate(as_completed(futs), 1):
            allres.update(f.result())
            print(f"[{n}/{len(keys)}] {futs[f]}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "grid.pkl"), "wb") as f:
        pickle.dump({"budgets": BUDGETS, "results": allres}, f)
    print("done", len(allres), "tasks", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
