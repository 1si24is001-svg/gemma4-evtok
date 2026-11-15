"""Tier A extensions: learned candidate ranker, budget-split graph rescue, Steiner-connected top-k.

Usage:
    python -m evtok.extensions --data <DATA_DIR> --out results/tier_a_ext --workers 8

Pass 1 (parallel over graphs): per task, build candidates and features (problem statement, graph and
embeddings only), compute packets for the non-learned extension baselines, and store gold labels for
evaluation/training. Pass 2: 5-fold task-level CV (same folds as evtok.analyze); the ranker is trained only
on training-fold tasks and scored on held-out tasks; every reported number is out-of-fold.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import selectors as S
from .analyze import boot_ci, folds_for, mcnemar_exact
from .data import graph_key, load_graph, load_tasks
from .gold import gold_nodes
from .metrics import packet_metrics
from .retrieval import hybrid_score, task_signals
from .run_tier_a import BUDGETS, KEEP, TAU, TEMP, TIERW
from .textproc import clean_problem, ident_tokens

FEATS = [
    "bm25_z", "bm25_logrank", "anchor", "dense_anchor_z", "dense_bm_z",
    "hop_top1", "hop_top3", "hop_top10", "n_adj_top10", "log_degree",
    "is_class", "depth", "log_len", "same_parent_top1", "share_prefix_top10",
    "name_in_query", "id_token_overlap", "in_bm25_top200",
]
RESCUE_GRID = list(itertools.product((0.25, 0.5), (3, 10), (1, 2)))  # frac, m seeds, rescue tier
STEINER_GRID = list(itertools.product((3, 5, 10), (0, 1)))  # K, connector tier


def multi_bfs(g, seeds, hmax=5):
    dist = {s: 0 for s in seeds}
    dq = deque(seeds)
    while dq:
        v = dq.popleft()
        if dist[v] >= hmax:
            continue
        for n in g.nbrs[v]:
            n = int(n)
            if n not in dist and g.pool[n]:
                dist[n] = dist[v] + 1
                dq.append(n)
    return dist


def zs(x, idx):
    v = x[idx]
    sd = v.std()
    return (x - v.mean()) / (sd if sd > 1e-9 else 1.0)


def task_features(g, sig):
    bm = np.log1p(sig["bm25"].astype(np.float64))
    pool_idx = np.where(g.pool)[0]
    order = pool_idx[np.argsort(-bm[pool_idx], kind="stable")]
    rank = {int(v): r for r, v in enumerate(order)}
    top = [int(v) for v in order[:10]]
    d1, d3, d10 = multi_bfs(g, top[:1]), multi_bfs(g, top[:3]), multi_bfs(g, top[:10])
    cand = list(dict.fromkeys([int(v) for v in order[:200]] + [v for v, d in multi_bfs(g, top, 2).items()]))
    bmz = zs(bm, pool_idx)
    daz = zs(sig["dense_anchor"].astype(np.float64), pool_idx)
    dbz = zs(sig["dense_bm"].astype(np.float64), pool_idx)
    qtok = set(ident_tokens(clean_problem(sig["_problem"])))
    parent1 = g.ids[top[0]].rsplit(".", 1)[0] if top else ""
    pref = lambda s: ".".join(s.split(".")[:2])
    top_pref = [pref(g.ids[v]) for v in top]
    topset = set(top)
    X = np.zeros((len(cand), len(FEATS)), dtype=np.float32)
    for r, v in enumerate(cand):
        nid = g.ids[v]
        last = nid.rsplit(".", 1)[-1]
        idt = set(ident_tokens(nid.replace(".", " ")))
        X[r] = [
            bmz[v], math.log1p(rank.get(v, 10**5)), sig["anchor"][v], daz[v], dbz[v],
            d1.get(v, 6), d3.get(v, 6), d10.get(v, 6), sum(int(n) in topset for n in g.nbrs[v]), math.log1p(len(g.nbrs[v])),
            float(g.text[v].lstrip().startswith("class ")), nid.count("."), math.log1p(len(g.text[v])),
            float(bool(parent1) and (nid.startswith(parent1 + ".") or nid.rsplit(".", 1)[0] == parent1)),
            sum(pref(nid) == p for p in top_pref) / max(1, len(top_pref)),
            float(last.lower() in qtok), (len(idt & qtok) / len(idt)) if idt else 0.0, float(rank.get(v, 10**5) < 200),
        ]
    return np.array(cand, dtype=np.int64), X, order, top


def rescue_packet(g, order, s, B, frac, m, tier):
    base = S.flat_full(g, order, int(B * (1 - frac)))
    used = S.packet_cost(g, base)
    seeds = [int(v) for v in order[:m]]
    nb = [int(n) for v in seeds for n in g.nbrs[v] if g.pool[int(n)] and int(n) not in base]
    nb = sorted(set(nb), key=lambda v: -s[v])
    pk = dict(base)
    for v in nb:
        for t in ((tier, 0) if tier == 1 else (2, 1, 0)):
            c = int(g.cost[v, t])
            if used + c <= B:
                pk[v] = t
                used += c
                break
    for v in order:  # fill any leftover with flat full source
        v = int(v)
        if B - used < 8:
            break
        if v not in pk and used + int(g.cost[v, 2]) <= B:
            pk[v] = 2
            used += int(g.cost[v, 2])
    return pk


def steiner_packet(g, order, B, K, conn_tier):
    """Connect the top-K seeds to the top-1 seed by shortest paths (<=3 hops); seeds full, connectors at conn_tier."""
    seeds = [int(v) for v in order[:K]]
    if not seeds:
        return {}
    root = seeds[0]
    parent = {root: None}
    dist = {root: 0}
    dq = deque([root])
    while dq:
        v = dq.popleft()
        if dist[v] >= 3:
            continue
        for n in g.nbrs[v]:
            n = int(n)
            if n not in dist and g.pool[n]:
                dist[n] = dist[v] + 1
                parent[n] = v
                dq.append(n)
    tree = set()
    for sd in seeds:
        x = sd if sd in parent else None
        while x is not None:
            tree.add(x)
            x = parent[x]
    tree.add(root)
    pk, used = {}, 0
    for v in seeds:  # seeds first, full source, in rank order
        c = int(g.cost[v, 2])
        if used + c <= B:
            pk[v] = 2
            used += c
    for v in tree - set(seeds):
        c = int(g.cost[v, conn_tier])
        if used + c <= B:
            pk[v] = conn_tier
            used += c
    for v in order:
        v = int(v)
        if B - used < 8:
            break
        if v not in pk and used + int(g.cost[v, 2]) <= B:
            pk[v] = 2
            used += int(g.cost[v, 2])
    return pk


def _m(g, pk, gold):
    m = packet_metrics(g, pk, gold)
    return {k: m[k] for k in KEEP}


def run_group(data_dir, key, tasks):
    g = load_graph(data_dir, key)
    out = {}
    for t in tasks:
        gold = gold_nodes(g, t["patch"])
        sig = task_signals(g, t["problem_statement"])
        sig["_problem"] = t["problem_statement"]
        cand, X, order, top = task_features(g, sig)
        s = hybrid_score(g, sig, 0.0, 0.0, "anchor")
        res = defaultdict(dict)
        for B in BUDGETS:
            res["bm25"][B] = _m(g, S.flat_full(g, order, B), gold)
            for frac, m, tier in RESCUE_GRID:
                res[f"rescue|f{frac}_m{m}_t{tier}"][B] = _m(g, rescue_packet(g, order, s, B, frac, m, tier), gold)
            for K, ct in STEINER_GRID:
                res[f"steiner|k{K}_t{ct}"][B] = _m(g, steiner_packet(g, order, B, K, ct), gold)
        y = np.array([int(v) in gold["nodes"] for v in cand], dtype=np.int8)
        out[t["instance_id"]] = {
            "repo": t["repo"], "graph": key, "n_gold": len(gold["nodes"]), "cand": cand, "X": X, "y": y,
            "gold_in_cand": bool(set(cand.tolist()) & gold["nodes"]) if gold["nodes"] else False,
            "res": dict(res),
        }
    return out


def score_learned(data_dir, groups, preds):
    """Build packets from learned scores for held-out tasks. preds: iid -> (cand, prob)."""
    out = {}
    for key, tasks in groups.items():
        tasks = [t for t in tasks if t["instance_id"] in preds]
        if not tasks:
            continue
        g = load_graph(data_dir, key)
        for t in tasks:
            gold = gold_nodes(g, t["patch"])
            cand, p = preds[t["instance_id"]]
            sc = np.full(len(g.ids), -1e9)
            sc[cand] = p
            order = cand[np.argsort(-p, kind="stable")]
            res = {"learned": {}, "learnedT": {}}
            u = np.zeros(len(g.ids))
            u[cand] = p / max(p.max(), 1e-9)
            for B in BUDGETS:
                res["learned"][B] = _m(g, S.flat_full(g, order, B), gold)
                res["learnedT"][B] = _m(g, S.greedy_tiers(g, u, B, cand, rho=0.5, expand=False, wt=TIERW["a"]), gold)
            out[t["instance_id"]] = res
    return out


def auc(r):
    v = [r[B]["recall"] for B in BUDGETS]
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/tier_a_ext")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    os.makedirs(a.out, exist_ok=True)
    tasks = load_tasks(a.data)
    groups = defaultdict(list)
    for t in tasks:
        groups[graph_key(t)].append(t)
    t0 = time.time()
    R = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_group, a.data, k, v) for k, v in sorted(groups.items())]
        for n, f in enumerate(as_completed(futs), 1):
            R.update(f.result())
            if n % 20 == 0:
                print(f"features {n}/{len(futs)} {time.time() - t0:.0f}s", flush=True)
    ids = sorted(R)
    fold = folds_for({i: {"meta": {"repo": R[i]["repo"]}} for i in ids})
    eval_ids = [i for i in ids if R[i]["n_gold"] > 0]
    oof = defaultdict(dict)
    choices = defaultdict(list)
    preds_lr, preds_gb = {}, {}
    for f in range(5):
        train = [i for i in eval_ids if fold[i] != f]
        test = [i for i in ids if fold[i] == f]
        # non-learned extensions: select config on training folds
        for fam in ("rescue", "steiner"):
            keys = [k for k in R[ids[0]]["res"] if k.startswith(fam + "|")]
            best = max(keys, key=lambda k: np.nanmean([auc(R[i]["res"][k]) for i in train]))
            choices[fam].append(best)
            for i in test:
                oof[fam][i] = R[i]["res"][best]
        for i in test:
            oof["bm25"][i] = R[i]["res"]["bm25"]
        # learned ranker
        Xtr = np.concatenate([R[i]["X"] for i in train])
        ytr = np.concatenate([R[i]["y"] for i in train])
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                            class_weight="balanced", random_state=0).fit(Xtr, ytr)
        choices["learned_lr_coef"].append(dict(zip(FEATS, np.round(lr.coef_[0], 3).tolist())))
        for i in test:
            preds_lr[i] = (R[i]["cand"], lr.predict_proba(sc.transform(R[i]["X"]))[:, 1])
            preds_gb[i] = (R[i]["cand"], gb.predict_proba(R[i]["X"])[:, 1])
    print(f"training done {time.time() - t0:.0f}s", flush=True)
    for name, preds in (("lr", preds_lr), ("gb", preds_gb)):
        L = score_learned(a.data, groups, preds)
        for i, r in L.items():
            oof[f"Learned ranker ({name})"][i] = r["learned"]
            oof[f"Learned ranker + tiers ({name})"][i] = r["learnedT"]
    names = {"bm25": "BM25 top-k", "rescue": "Budget-split graph rescue", "steiner": "Steiner-connected top-k (shortest paths)"}
    for k, v in names.items():
        oof[v] = oof.pop(k)
    order = ["BM25 top-k", "Budget-split graph rescue", "Steiner-connected top-k (shortest paths)",
             "Learned ranker (lr)", "Learned ranker + tiers (lr)", "Learned ranker (gb)", "Learned ranker + tiers (gb)"]
    lines = ["# Tier A extensions (out-of-fold, same 5 folds as Tier A)\n",
             f"Tasks with gold: {len(eval_ids)}. Gold inside learned candidate set: {sum(R[i]['gold_in_cand'] for i in eval_ids)}.\n",
             "| Method | " + " | ".join(f"R@{B // 1000}k" for B in BUDGETS) + " | Recall-AUC | Hit@8k | ΔAUC vs BM25 [95% CI] | Hit@8k wins/losses, p |",
             "|" + "---|" * (len(BUDGETS) + 5)]
    stats = {}
    for name in order:
        per = oof[name]
        rec = [np.nanmean([per[i][B]["recall"] for i in eval_ids]) for B in BUDGETS]
        A = np.nanmean([auc(per[i]) for i in eval_ids])
        h8 = np.nanmean([per[i][8000]["hit"] for i in eval_ids])
        if name == "BM25 top-k":
            delta, mc = "—", "—"
        else:
            d = np.array([auc(per[i]) - auc(oof["BM25 top-k"][i]) for i in eval_ids])
            m_, lo, hi = boot_ci(d)
            b = sum(per[i][8000]["hit"] == 1 and oof["BM25 top-k"][i][8000]["hit"] == 0 for i in eval_ids)
            c = sum(per[i][8000]["hit"] == 0 and oof["BM25 top-k"][i][8000]["hit"] == 1 for i in eval_ids)
            delta, mc = f"{m_:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{b}/{c}, p={mcnemar_exact(b, c):.3f}"
            stats[name] = {"auc_delta": [m_, lo, hi], "hit8": [b, c]}
        lines.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in rec) + f" | {A:.3f} | {h8:.3f} | {delta} | {mc} |")
    # subgroups for the best learned variant
    ng = {i: R[i]["n_gold"] for i in eval_ids}
    lines.append("\n## By number of gold functions (post hoc): ΔAUC vs BM25\n")
    lines.append("| Method | 1 | 2–3 | ≥4 |")
    lines.append("|---|---|---|---|")
    for name in order[1:]:
        cells = []
        for lo_, hi_ in ((1, 1), (2, 3), (4, 999)):
            s_ = [i for i in eval_ids if lo_ <= ng[i] <= hi_]
            m_, lo, hi = boot_ci(np.array([auc(oof[name][i]) - auc(oof["BM25 top-k"][i]) for i in s_]))
            cells.append(f"{m_:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(s_)})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("\n## CV choices\n")
    for k in ("rescue", "steiner"):
        lines.append(f"- {k}: " + "; ".join(choices[k]))
    lines.append("- logistic coefficients per fold (standardized features):")
    for c in choices["learned_lr_coef"]:
        lines.append("  - " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(c.items(), key=lambda kv: -abs(kv[1]))[:8]))
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "oof_per_task.csv"), "w", encoding="utf-8") as fh:
        fh.write("instance_id,repo,fold,n_gold,method,budget,recall,recall_any,hit\n")
        for name in order:
            for i in ids:
                if i not in oof[name]:
                    continue
                for B in BUDGETS:
                    m = oof[name][i][B]
                    fh.write(f"{i},{R[i]['repo']},{fold[i]},{R[i]['n_gold']},{name},{B},{m['recall']},{m['recall_any']},{m['hit']}\n")
    json.dump({"choices": {k: v for k, v in choices.items()}, "stats": stats}, open(os.path.join(a.out, "cv_choices.json"), "w"), indent=1, default=str)
    print("\n".join(lines))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
