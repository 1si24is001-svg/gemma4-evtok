"""Tier A analysis: 5-fold task-level CV config selection, out-of-fold metrics, paired statistics.

Usage:
    python -m evtok.analyze --grid results/tier_a/grid.pkl --out results/tier_a

Selection objective (training folds only): mean over tasks with gold nodes of
mean-over-budgets function Recall@B (full-source tier).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict

import numpy as np

METHODS = [
    # name, config-key prefix, tuned?
    ("BM25 top-k", "flat|wd0.0_wa0.0_anchor", False),
    ("Dense top-k (anchor query)", "flat|dense_anchor", False),
    ("Dense top-k (BM25-centroid query)", "flat|dense_bm", False),
    ("Hybrid top-k", "flat|", True),
    ("Hybrid + tiers (no graph)", "flatT|", True),
    ("Graph expansion (full only)", "gx|", True),
    ("Connected (full only)", "conn|", True),
    ("Connected + tiers (EpT)", "connT|", True),
]
DEFAULTS = {
    "Hybrid top-k [default]": "flat|wd0.5_wa1.0_anchor",
    "Connected + tiers (EpT) [default]": "connT|wd0.5_wa1.0_anchor|b1.0_h2_k5_rho1.0_b",
}
BASELINE = "Hybrid top-k"


def folds_for(tasks: dict, k: int = 5, seed: int = 0) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    by_repo = defaultdict(list)
    for iid, r in sorted(tasks.items()):
        by_repo[r["meta"]["repo"]].append(iid)
    f = {}
    off = 0
    for repo in sorted(by_repo):
        ids = by_repo[repo]
        rng.shuffle(ids)
        for i, iid in enumerate(ids):
            f[iid] = (i + off) % k
        off += len(ids)
    return f


def auc(res_cfg: dict, budgets, key="recall") -> float:
    v = [res_cfg[B][key] for B in budgets]
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def candidate_keys(sample_res: dict, prefix: str, tuned: bool) -> list[str]:
    if not tuned:
        return [prefix] if prefix in sample_res else []
    ks = [k for k in sample_res if k.startswith(prefix)]
    if prefix == "flat|":
        ks = [k for k in ks if not k.startswith("flat|dense")]
    return sorted(ks)


def select(train_ids, R, keys, budgets) -> str:
    best, bv = keys[0], -1.0
    for k in keys:
        v = [auc(R[i]["res"][k], budgets) for i in train_ids]
        v = [x for x in v if not math.isnan(x)]
        m = float(np.mean(v)) if v else -1.0
        if m > bv + 1e-12:
            best, bv = k, m
    return best


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def boot_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return (float("nan"),) * 3
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/tier_a/grid.pkl")
    ap.add_argument("--out", default="results/tier_a")
    a = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    G = pickle.load(open(a.grid, "rb"))
    budgets = list(G["budgets"])
    R = G["results"]
    ids = sorted(R)
    fold = folds_for(R)
    sample = R[ids[0]]["res"]
    eval_ids = [i for i in ids if R[i]["meta"]["n_gold"] > 0]

    oof: dict[str, dict[str, dict]] = defaultdict(dict)  # method -> iid -> {B: metrics}
    choices: dict[str, list] = defaultdict(list)
    for name, prefix, tuned in METHODS:
        keys = candidate_keys(sample, prefix, tuned)
        for f in range(5):
            train = [i for i in eval_ids if fold[i] != f]
            test = [i for i in ids if fold[i] == f]
            k = select(train, R, keys, budgets) if tuned else keys[0]
            choices[name].append(k)
            for i in test:
                oof[name][i] = R[i]["res"][k]
            if name == "Connected + tiers (EpT)":
                hk = k.split("|")[1]
                for i in test:
                    draws = [R[i]["res"][f"rand|{hk}|r{r}"] for r in range(3)]
                    oof["Random connected (control)"][i] = {
                        B: {m: float(np.nanmean([d[B][m] for d in draws])) for m in draws[0][B]} for B in budgets
                    }
            if name == "Hybrid top-k":
                hk = k.split("|")[1]
                for i in test:
                    oof["Oracle (gold first)"][i] = R[i]["res"][f"oracle|{hk}"]
    for name, k in DEFAULTS.items():
        for i in ids:
            oof[name][i] = R[i]["res"][k]

    order = [m[0] for m in METHODS] + ["Random connected (control)", "Oracle (gold first)"] + list(DEFAULTS)
    lines = []
    lines.append(f"# Tier A results (out-of-fold, 5-fold task-level CV)\n")
    lines.append(f"Tasks: {len(ids)}; with gold function nodes: {len(eval_ids)}; addition-only: {len(ids) - len(eval_ids)}.\n")
    # main table
    hdr = "| Method | " + " | ".join(f"Recall@{B//1000}k" for B in budgets) + " | Recall-AUC | Hit@4k | Hit@8k | AnyTier@4k | File@4k |"
    lines.append(hdr)
    lines.append("|" + "---|" * (hdr.count("|") - 1))
    rows_csv = []
    for name in order:
        per = oof[name]
        rec = [np.nanmean([per[i][B]["recall"] for i in eval_ids]) for B in budgets]
        aucv = np.nanmean([auc(per[i], budgets) for i in eval_ids])
        h4 = np.nanmean([per[i][4000]["hit"] for i in eval_ids])
        h8 = np.nanmean([per[i][8000]["hit"] for i in eval_ids])
        a4 = np.nanmean([per[i][4000]["recall_any"] for i in eval_ids])
        f4 = np.nanmean([per[i][4000]["file_hit"] for i in ids if not math.isnan(per[i][4000]["file_hit"])])
        lines.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in rec) + f" | {aucv:.3f} | {h4:.3f} | {h8:.3f} | {a4:.3f} | {f4:.3f} |")
        rows_csv.append([name] + [f"{x:.4f}" for x in rec] + [f"{aucv:.4f}", f"{h4:.4f}", f"{h8:.4f}", f"{a4:.4f}", f"{f4:.4f}"])
    # paired stats vs baseline
    lines.append(f"\n## Paired differences vs {BASELINE} (bootstrap 95% CI over {len(eval_ids)} tasks; McNemar exact on Hit)\n")
    lines.append("| Method | ΔRecall-AUC [95% CI] | " + " | ".join(f"ΔRecall@{B//1000}k [CI]" for B in budgets) + " | Hit@4k b/c, p | Hit@8k b/c, p |")
    lines.append("|" + "---|" * (len(budgets) + 4))
    base = oof[BASELINE]
    stats = {}
    for name in order:
        if name == BASELINE:
            continue
        per = oof[name]
        d_auc = np.array([auc(per[i], budgets) - auc(base[i], budgets) for i in eval_ids])
        cells = []
        m, lo, hi = boot_ci(d_auc)
        cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        st = {"auc": (m, lo, hi)}
        for B in budgets:
            d = np.array([per[i][B]["recall"] - base[i][B]["recall"] for i in eval_ids])
            m, lo, hi = boot_ci(d)
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            st[B] = (m, lo, hi)
        for B in (4000, 8000):
            b = sum(per[i][B]["hit"] == 1 and base[i][B]["hit"] == 0 for i in eval_ids)
            c = sum(per[i][B]["hit"] == 0 and base[i][B]["hit"] == 1 for i in eval_ids)
            cells.append(f"{b}/{c}, p={mcnemar_exact(b, c):.3f}")
            st[f"mcnemar{B}"] = (b, c, mcnemar_exact(b, c))
        stats[name] = st
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    # per-repo AUC
    repos = sorted({R[i]["meta"]["repo"] for i in eval_ids})
    lines.append("\n## Recall-AUC by repository\n")
    lines.append("| Method | " + " | ".join(f"{r} (n={sum(R[i]['meta']['repo']==r for i in eval_ids)})" for r in repos) + " |")
    lines.append("|" + "---|" * (len(repos) + 1))
    for name in order:
        per = oof[name]
        lines.append(f"| {name} | " + " | ".join(f"{np.nanmean([auc(per[i], budgets) for i in eval_ids if R[i]['meta']['repo']==r]):.3f}" for r in repos) + " |")
    # packet shape
    lines.append("\n## Packet composition at 8k (mean)\n")
    lines.append("| Method | nodes | full-source nodes | components | cost | evidence density |")
    lines.append("|---|---|---|---|---|---|")
    for name in order:
        per = oof[name]
        g = lambda k: np.nanmean([per[i][8000][k] for i in eval_ids])
        lines.append(f"| {name} | {g('n_nodes'):.1f} | {g('n_full'):.1f} | {g('components'):.1f} | {g('cost'):.0f} | {g('density'):.3f} |")
    lines.append("\n## CV choices per fold\n")
    for name in order:
        if name in choices:
            lines.append(f"- {name}: " + "; ".join(choices[name]))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"recall@{B}" for B in budgets] + ["recall_auc", "hit@4k", "hit@8k", "anytier@4k", "file@4k"])
        w.writerows(rows_csv)
    with open(os.path.join(a.out, "oof_per_task.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "repo", "fold", "n_gold", "method", "budget", "recall", "recall_any", "hit", "file_hit", "cost", "n_nodes", "components"])
        for name in order:
            for i in ids:
                for B in budgets:
                    m = oof[name][i][B]
                    w.writerow([i, R[i]["meta"]["repo"], fold[i], R[i]["meta"]["n_gold"], name, B, m["recall"], m["recall_any"], m["hit"], m["file_hit"], m["cost"], m["n_nodes"], m["components"]])
    json.dump({"choices": choices, "folds": fold}, open(os.path.join(a.out, "cv_choices.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
