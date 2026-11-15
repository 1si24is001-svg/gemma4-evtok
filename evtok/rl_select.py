"""RL with token penalties for context selection (REINFORCE, CPU).

MDP per task: the state is the set of functions selected so far. An action adds one candidate function
(full source) or STOPs. The reward per step is (new gold functions captured / |gold|) - lambda * tokens/1000,
so the policy trades evidence against token "fuel" and chooses its own budget per task (hard cap 16k).
Candidates and node features come from evtok.extensions (problem statement, graph and embeddings only).
The policy is a linear softmax over [node features, adjacency-to-selected, log cost] with a STOP logit
over [1, tokens used/1000, steps]. Training uses gold labels from training-fold tasks only; evaluation is
a greedy rollout on held-out tasks, compared with BM25 top-k given the SAME number of tokens per task.

Usage: python -m evtok.rl_select --data <DATA_DIR> --out results/rl --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import selectors as S
from .analyze import boot_ci, folds_for, mcnemar_exact
from .data import graph_key, load_graph, load_tasks
from .extensions import FEATS, task_features
from .gold import gold_nodes
from .retrieval import task_signals

CAP = 16000
LAMBDAS = (0.02, 0.05, 0.1)  # penalty per 1,000 tokens
MAX_STEPS = 60


def prep_group(data_dir, key, tasks):
    g = load_graph(data_dir, key)
    out = {}
    for t in tasks:
        gold = gold_nodes(g, t["patch"])
        sig = task_signals(g, t["problem_statement"])
        sig["_problem"] = t["problem_statement"]
        cand, X, order, _ = task_features(g, sig)
        pos = {int(v): i for i, v in enumerate(cand)}
        adj = [[pos[int(n)] for n in g.nbrs[int(v)] if int(n) in pos] for v in cand]
        bm_order = [int(v) for v in order[:3000]]
        out[t["instance_id"]] = {
            "repo": t["repo"], "n_gold": len(gold["nodes"]), "cand": cand, "X": X,
            "y": np.array([int(v) in gold["nodes"] for v in cand], dtype=np.int8),
            "cost": g.cost[cand, 2].astype(np.float64), "adj": adj,
            "bm_order": bm_order, "bm_cost": {v: int(g.cost[v, 2]) for v in bm_order},
            "gold": sorted(gold["nodes"]),
        }
    return out


class Policy:
    def __init__(self, d, rng):
        self.w = rng.normal(0, 0.01, d)
        self.ws = np.zeros(3)
        self.ws[0] = -2.0  # start biased against stopping

    def logits(self, F, stop_feat):
        return np.append(F @ self.w, stop_feat @ self.ws)


def node_feats(T, mu, sd, selected_mask, adj_count):
    Xs = (T["X"] - mu) / sd
    return np.column_stack([Xs, np.minimum(adj_count, 3) / 3.0, np.log1p(T["cost"]) / 8.0])


def rollout(T, pol, mu, sd, lam, rng, greedy=False):
    n = len(T["cand"])
    sel = np.zeros(n, bool)
    adj_count = np.zeros(n)
    used, steps = 0.0, 0
    traj = []  # (F, stop_feat, valid, action, reward)
    ng = max(1, int(T["y"].sum())) if T["y"].sum() else max(1, T["n_gold"])
    base = node_feats(T, mu, sd, sel, adj_count)
    while steps < MAX_STEPS:
        valid = (~sel) & (used + T["cost"] <= CAP)
        stop_feat = np.array([1.0, used / 1000.0, steps / 10.0])
        F = base.copy()
        F[:, -2] = np.minimum(adj_count, 3) / 3.0
        lg = pol.logits(F, stop_feat)
        lg[:-1][~valid] = -1e9
        lg = lg - lg.max()
        p = np.exp(lg)
        p /= p.sum()
        a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        if a == n or not valid.any():
            traj.append((F, stop_feat, valid, n, 0.0, p))
            break
        r = T["y"][a] / ng - lam * T["cost"][a] / 1000.0
        traj.append((F, stop_feat, valid, a, r, p))
        sel[a] = True
        used += T["cost"][a]
        for j in T["adj"][a]:
            adj_count[j] += 1
        steps += 1
    return traj, sel, used


def reinforce(train, R, mu, sd, lam, seed, epochs=40, samples=4, lr=0.05):
    rng = np.random.default_rng(seed)
    d = len(FEATS) + 2
    pol = Policy(d, rng)
    baseline = defaultdict(float)
    for ep in range(epochs):
        order = rng.permutation(train)
        for iid in order:
            T = R[iid]
            gw, gs = np.zeros_like(pol.w), np.zeros_like(pol.ws)
            for _ in range(samples):
                traj, _, _ = rollout(T, pol, mu, sd, lam, rng)
                rets = np.cumsum([x[4] for x in traj][::-1])[::-1]
                G0 = rets[0] if len(rets) else 0.0
                for (F, sf, valid, a, r, p), Gt in zip(traj, rets):
                    adv = Gt - baseline[iid]
                    n = F.shape[0]
                    pn = p[:n] * valid
                    if a == n:
                        gs += adv * sf * (1 - p[n])
                        gw -= adv * (pn @ F)
                    else:
                        gw += adv * (F[a] - pn @ F)
                        gs -= adv * p[n] * sf
                baseline[iid] = 0.9 * baseline[iid] + 0.1 * G0
            pol.w += lr * gw / samples
            pol.ws += lr * gs / samples
    return pol


def bm25_at(T, tokens):
    pk, used = [], 0
    for v in T["bm_order"]:
        c = T["bm_cost"][v]
        if used + c <= tokens:
            pk.append(v)
            used += c
    gold = set(T["gold"])
    hit = len(gold & set(pk))
    return hit / len(gold), float(hit > 0), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/rl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3, help="independent training seeds; seed 0 is the headline run")
    a = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    groups = defaultdict(list)
    for t in load_tasks(a.data):
        groups[graph_key(t)].append(t)
    R = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(prep_group, a.data, k, v) for k, v in sorted(groups.items())]):
            R.update(f.result())
    print(f"prepared {len(R)} tasks {time.time() - t0:.0f}s", flush=True)
    ids = sorted(R)
    fold = folds_for({i: {"meta": {"repo": R[i]["repo"]}} for i in ids})
    eval_ids = [i for i in ids if R[i]["n_gold"] > 0]
    jobs = [(lam, f, s) for s in range(a.seeds) for lam in LAMBDAS for f in range(5)]
    by_seed = defaultdict(dict)  # (lam, seed) -> iid -> dict
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(train_eval, R, fold, eval_ids, lam, f, a.epochs, s): (lam, f, s) for lam, f, s in jobs}
        for n, fu in enumerate(as_completed(futs), 1):
            lam, f, s = futs[fu]
            by_seed[(lam, s)].update(fu.result())
            print(f"[{n}/{len(jobs)}] seed {s} lambda {lam} fold {f} done {time.time() - t0:.0f}s", flush=True)
    results = {lam: by_seed[(lam, 0)] for lam in LAMBDAS}  # headline table: seed 0
    lines = ["# RL with token penalties for context selection (REINFORCE, out-of-fold)\n",
             f"Tasks with gold: {len(eval_ids)}. Reward per step = gold captured / |gold| − λ · tokens/1000; cap {CAP} tokens. "
             "Baseline: BM25 top-k given exactly the tokens the policy spent on each task.\n",
             "| λ per 1k tokens | mean tokens used | RL recall | BM25 recall at same tokens | Δ recall [95% CI] | RL Hit | BM25 Hit | Hit wins/losses, p | mean stop step |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for lam in LAMBDAS:
        rr = results[lam]
        rec = np.array([rr[i]["rl_recall"] for i in eval_ids])
        brec = np.array([rr[i]["bm_recall"] for i in eval_ids])
        m_, lo, hi = boot_ci(rec - brec)
        b = sum(rr[i]["rl_hit"] == 1 and rr[i]["bm_hit"] == 0 for i in eval_ids)
        c = sum(rr[i]["rl_hit"] == 0 and rr[i]["bm_hit"] == 1 for i in eval_ids)
        tok = np.mean([rr[i]["tokens"] for i in eval_ids])
        lines.append(f"| {lam} | {tok:.0f} | {rec.mean():.3f} | {brec.mean():.3f} | {m_:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{np.mean([rr[i]['rl_hit'] for i in eval_ids]):.3f} | {np.mean([rr[i]['bm_hit'] for i in eval_ids]):.3f} | "
                     f"{b}/{c}, p={mcnemar_exact(b, c):.3f} | {np.mean([rr[i]['steps'] for i in eval_ids]):.1f} |")
        toks = np.array([rr[i]["tokens"] for i in eval_ids])
        summary[lam] = {"tokens_mean": tok, "tokens_p10_p90": [float(np.percentile(toks, 10)), float(np.percentile(toks, 90))],
                        "rl_recall": float(rec.mean()), "bm_recall": float(brec.mean()), "delta": [m_, lo, hi], "hit_wins_losses": [b, c]}
    lines.append("\nToken spend varies per task (10th–90th percentile): " +
                 "; ".join(f"λ={l}: {summary[l]['tokens_p10_p90'][0]:.0f}–{summary[l]['tokens_p10_p90'][1]:.0f}" for l in LAMBDAS))
    if a.seeds > 1:
        lines.append(f"\n## Stability across {a.seeds} training seeds\n")
        lines.append("The table above is seed 0. Each seed retrains every fold from a different initialization and sampling stream.\n")
        lines.append("| λ per 1k tokens | mean tokens, per seed | Δ recall vs BM25, per seed | mean Δ | seeds with RL below BM25 |")
        lines.append("|---|---|---|---|---|")
        below = total = 0
        for lam in LAMBDAS:
            toks, deltas = [], []
            for s in range(a.seeds):
                rr = by_seed[(lam, s)]
                toks.append(np.mean([rr[i]["tokens"] for i in eval_ids]))
                deltas.append(np.mean([rr[i]["rl_recall"] - rr[i]["bm_recall"] for i in eval_ids]))
            below += sum(d < 0 for d in deltas)
            total += len(deltas)
            lines.append(f"| {lam} | " + ", ".join(f"{x:.0f}" for x in toks) + " | " + ", ".join(f"{d:+.3f}" for d in deltas) +
                         f" | {np.mean(deltas):+.3f} | {sum(d < 0 for d in deltas)}/{len(deltas)} |")
            summary[lam]["per_seed"] = {"tokens": [float(x) for x in toks], "delta": [float(d) for d in deltas]}
        lines.append(f"\nRL is below BM25 at matched tokens in {below} of {total} (λ, seed) runs.")
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(lines))
    print(f"total {time.time() - t0:.0f}s")


def train_eval(R, fold, eval_ids, lam, f, epochs, s=0):
    train = [i for i in eval_ids if fold[i] != f]
    test = [i for i in eval_ids if fold[i] == f]
    Xtr = np.concatenate([R[i]["X"] for i in train])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    pol = reinforce(train, R, mu, sd, lam, seed=f + 1000 * s, epochs=epochs)  # s=0 reproduces the single-seed run
    out = {}
    rng = np.random.default_rng(0)
    for i in test:
        T = R[i]
        traj, sel, used = rollout(T, pol, mu, sd, lam, rng, greedy=True)
        gold = set(T["gold"])
        chosen = set(int(v) for v in T["cand"][sel])
        rl_rec = len(gold & chosen) / len(gold)
        bm_rec, bm_hit, _ = bm25_at(T, used)
        out[i] = {"rl_recall": rl_rec, "rl_hit": float(rl_rec > 0), "bm_recall": bm_rec, "bm_hit": bm_hit,
                  "tokens": used, "steps": int(sel.sum())}
    return out


if __name__ == "__main__":
    main()
